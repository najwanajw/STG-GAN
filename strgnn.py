"""StrGNN-PyG-window-adapted, Twitter Tennis only, five seeds, 100 epochs.
Paper-based reimplementation (not the official code): enclosing subgraphs,
DRNL labeling, three GCNs, SortPooling, GRU, context-dependent negatives.
Adaptations documented in results: four snapshots; directed candidates with
undirected subgraphs; removal of queried edge; fixed structural label vocabulary;
64 training target edges/window; validation-selected window aggregation.
This is a validation-calibrated adaptation, not a verified official StrGNN.
No improvement in test performance is guaranteed.
No STG-GAN or AddGraph training is executed.
Dependencies: torch, torch-geometric, torch-geometric-temporal, numpy, sklearn.
Paper: https://arxiv.org/abs/2005.07427
Reference repository: https://github.com/KnowledgeDiscovery/StrGNN
Spyder: %runfile twitter_strgnn_validation_adapted.py --wdir
"""
import copy
import random
import time
import json
import hashlib
import argparse
from pathlib import Path
from collections import deque
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv
from torch_geometric_temporal.dataset import TwitterTennisDatasetLoader
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, confusion_matrix

SEEDS = list(range(42,47))
EPOCHS = 100
SEQ_LEN = 4
TRAIN_FRACTION = 0.5
TEST_RATIO = VAL_ANOMALY_RATIO = 0.30
NODE_RATIO = 0.03
TEST_INJECTION_SEED = 2026
VAL_CALIB_SEED_OFFSET = 5000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
METHOD = 'StrGNN-PyG-validation-adapted'
GCN_HIDDEN = 32
GRU_HIDDEN = 256
LABEL_LIMIT = 128
LR = 1e-4
BATCH_SIZE = 32
# Computational budget: all edges are used for validation/test, without sampling.
# Set to None to train on every available edge (substantially slower).
TRAIN_EDGES_PER_WINDOW = 64
OUTPUT_FILE = Path(__file__).resolve().with_name('twitter_strgnn_validation_adapted_results.json')

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def windows(snaps):
    return [snaps[i:i + SEQ_LEN] for i in range(len(snaps) - SEQ_LEN + 1)]

def inject(seq, ratio=NODE_RATIO):
    modified, node_sets = [], []
    for snapshot in seq:
        s = copy.copy(snapshot)
        s.x, s.edge_index = snapshot.x.clone(), snapshot.edge_index.clone()
        n = s.x.size(0)
        nodes = torch.randperm(n)[:max(1, int(ratio * n))]
        s.x[nodes] = 1.5 * s.x[nodes] + 0.5 * torch.randn_like(s.x[nodes])
        if len(nodes) > 1:
            mask = (torch.rand(len(nodes), len(nodes)) < 0.20) & ~torch.eye(len(nodes), dtype=torch.bool)
            source, target = mask.nonzero(as_tuple=True)
            new_edges = torch.stack((nodes[source], nodes[target]))
            s.edge_index = torch.cat((s.edge_index, new_edges), dim=1)
        modified.append(s); node_sets.append(nodes)
    return modified, node_sets

def make_test(test_seqs, seed, ratio=TEST_RATIO, node_ratio=NODE_RATIO):
    rng = np.random.default_rng(seed + 1000)
    chosen = set(rng.choice(len(test_seqs), size=round(ratio * len(test_seqs)), replace=False))
    seed_all(seed + 2000)
    evaluation, labels, true_nodes = [], [], []
    for i, seq in enumerate(test_seqs):
        altered, nodes = inject(seq, ratio=node_ratio) if i in chosen else (seq, None)
        evaluation.append(altered); labels.append(int(i in chosen)); true_nodes.append(nodes)
    return evaluation, np.array(labels), true_nodes

class SnapshotGraph:
    """Sparse undirected neighborhoods, plus original directed candidates."""
    def __init__(self, snapshot):
        self.n = snapshot.x.size(0)
        self.edges = sorted({(int(a),int(b)) for a,b in snapshot.edge_index.cpu().t().tolist() if a != b})
        self.edge_set = set(self.edges)
        self.neighbors = [set() for _ in range(self.n)]
        for a,b in self.edges:
            self.neighbors[a].add(b)
            self.neighbors[b].add(a)

    def subgraph(self, a, b):
        # One-hop enclosing subgraph, with the queried link removed to avoid
        # a trivial observed/non-observed edge classification shortcut.
        nodes = sorted({a,b} | self.neighbors[a] | self.neighbors[b])
        index = {v:i for i,v in enumerate(nodes)}
        local = [set() for _ in nodes]
        for v in nodes:
            for w in self.neighbors[v]:
                if w in index and {v,w} != {a,b}:
                    local[index[v]].add(index[w])
        def distances(root, blocked):
            distance = np.full(len(nodes), np.inf)
            distance[root] = 0
            queue = deque([root])
            while queue:
                v = queue.popleft()
                for w in local[v]:
                    if w != blocked and not np.isfinite(distance[w]):
                        distance[w] = distance[v] + 1
                        queue.append(w)
            return distance
        ia, ib = index[a], index[b]
        da, db = distances(ia,ib), distances(ib,ia)
        labels = np.zeros(len(nodes),dtype=np.int64)
        valid = np.isfinite(da) & np.isfinite(db)
        total = (da[valid]+db[valid]).astype(np.int64)
        half = total//2
        labels[valid] = 1 + np.minimum(da[valid],db[valid]).astype(np.int64) + half*(half+total%2-1)
        labels[[ia,ib]] = 1
        # Reserve final category for labels exceeding this predeclared vocabulary.
        labels = np.minimum(labels,LABEL_LIMIT)
        pairs = [(i,j) for i,adj in enumerate(local) for j in sorted(adj)]
        ei = torch.tensor(pairs,dtype=torch.long).t().contiguous() if pairs else torch.empty(2,0,dtype=torch.long)
        return Data(labels=torch.from_numpy(labels),edge_index=ei,num_nodes=len(nodes))


def graph_windows(sequences):
    # Shared clean snapshots are converted once, using object identity.
    memo = {}
    result = []
    for seq in sequences:
        row=[]
        for s in seq:
            if id(s) not in memo:
                memo[id(s)] = SnapshotGraph(s)
            row.append(memo[id(s)])
        result.append(row)
    return result


def training_samples(graph_seqs, seed):
    rng = np.random.default_rng(seed)
    samples=[]
    for graphs in graph_seqs:
        available=[(t,a,b) for t,g in enumerate(graphs) for a,b in g.edges]
        if not available:
            continue
        limit=len(available) if TRAIN_EDGES_PER_WINDOW is None else min(len(available),TRAIN_EDGES_PER_WINDOW)
        ids=rng.choice(len(available),size=limit,replace=False)
        for i in ids:
            t,a,b=available[i]
            samples.append((graphs,t,a,b,0.0))
            # Context-dependent negatives: retain one endpoint from observed edge.
            negative=None
            for _ in range(100):
                node=int(rng.integers(graphs[t].n))
                pair=(node,b) if rng.random()<0.5 else (a,node)
                if pair[0]!=pair[1] and pair not in graphs[t].edge_set and pair[::-1] not in graphs[t].edge_set:
                    negative=pair
                    break
            if negative is not None:
                samples.append((graphs,t,*negative,1.0))
    if not samples:
        raise ValueError('No training edge samples available.')
    return samples


class SubgraphCache:
    def __init__(self):
        self.data={}
    def sequence(self, graphs, t, a, b):
        # Only history at or before target snapshot t; never future snapshots.
        context=graphs[max(0,t-SEQ_LEN+1):t+1]
        result=[None]*(SEQ_LEN-len(context))
        for graph in context:
            key=(id(graph),a,b)
            if key not in self.data:
                self.data[key]=graph.subgraph(a,b)
            result.append(self.data[key])
        return result


class StrGNNWindow(nn.Module):
    def __init__(self, sort_k):
        super().__init__()
        self.sort_k=sort_k
        self.layers=nn.ModuleList([GCNConv(LABEL_LIMIT+1,GCN_HIDDEN),
                                  GCNConv(GCN_HIDDEN,GCN_HIDDEN),
                                  GCNConv(GCN_HIDDEN,GCN_HIDDEN)])
        self.rank=GCNConv(3*GCN_HIDDEN,1)
        self.gru=nn.GRU(sort_k*(3*GCN_HIDDEN+1),GRU_HIDDEN,batch_first=True)
        self.classifier=nn.Linear(GRU_HIDDEN,1)
    def structural(self, graphs):
        batch=Batch.from_data_list(graphs).to(DEVICE)
        x=F.one_hot(batch.labels,num_classes=LABEL_LIMIT+1).float()
        hidden=[]
        for layer in self.layers:
            x=F.relu(layer(x,batch.edge_index))
            hidden.append(x)
        features=torch.cat(hidden,dim=1)
        rank=torch.tanh(self.rank(features,batch.edge_index))
        features=torch.cat([features,rank],dim=1)
        pooled=[]
        for start,end in zip(batch.ptr[:-1].tolist(),batch.ptr[1:].tolist()):
            local=features[start:end]
            order=local[:,-1].argsort(descending=True,stable=True)[:self.sort_k]
            chosen=local[order]
            pooled.append(F.pad(chosen,(0,0,0,self.sort_k-len(chosen))).flatten())
        return torch.stack(pooled)
    def forward(self, contexts):
        # Zero-padded missing historical snapshots do not duplicate current data.
        valid=[graph for context in contexts for graph in context if graph is not None]
        encoded=self.structural(valid)
        sequences=[]
        i=0
        for context in contexts:
            row=[]
            for graph in context:
                if graph is None:
                    row.append(encoded.new_zeros(encoded.size(1)))
                else:
                    row.append(encoded[i]); i+=1
            sequences.append(torch.stack(row))
        _,state=self.gru(torch.stack(sequences))
        return self.classifier(state[-1]).flatten()


def fit_model(samples,seed):
    cache=SubgraphCache()
    contexts=[cache.sequence(graphs,t,a,b) for graphs,t,a,b,_ in samples]
    labels=torch.tensor([s[-1] for s in samples],dtype=torch.float32,device=DEVICE)
    sizes=[g.num_nodes for c in contexts for g in c if g is not None]
    # Training-only 60th percentile; no test topology used for selecting K.
    sort_k=max(10,int(np.ceil(np.quantile(sizes,0.60))))
    model=StrGNNWindow(sort_k).to(DEVICE)
    optimizer=torch.optim.Adam(model.parameters(),lr=LR)
    print(f'Training examples={len(samples)}; SortPooling K={sort_k}; cached subgraphs={len(cache.data)}')
    rng=np.random.default_rng(seed+100)
    model.train()
    for epoch in range(EPOCHS):
        total=0.0
        order=rng.permutation(len(samples))
        for start in range(0,len(samples),BATCH_SIZE):
            ids=order[start:start+BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            logits=model([contexts[int(i)] for i in ids])
            loss=F.binary_cross_entropy_with_logits(logits,labels[torch.as_tensor(ids,device=DEVICE)])
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss.')
            loss.backward(); optimizer.step()
            total+=loss.item()*len(ids)
        if epoch==0 or (epoch+1)%10==0:
            print(f'  Epoch {epoch+1}/{EPOCHS}: StrGNN BCE={total/len(samples):.4f}')
    model.eval()
    return model


@torch.no_grad()
def score_candidates(model,graph_seqs):
    """Fixed candidate set. Selection uses validation labels only.
    Negative candidates are an explicitly supervised window-level adaptation,
    not a reinterpretation of the edge classifier's anomaly probability.
    """
    result={key:[] for key in ('mean_high','mean_low','tail_high','tail_low')}
    for graphs in graph_seqs:
        per_snapshot={key:[] for key in result}
        cache=SubgraphCache()
        for t,graph in enumerate(graphs):
            values=[]
            for start in range(0,len(graph.edges),BATCH_SIZE):
                pairs=graph.edges[start:start+BATCH_SIZE]
                contexts=[cache.sequence(graphs,t,a,b) for a,b in pairs]
                values.extend(torch.sigmoid(model(contexts)).cpu().tolist())
                cache.data.clear()
            if not values:
                for key in result: per_snapshot[key].append(0.0)
                continue
            values=np.sort(np.asarray(values,dtype=float))
            if not np.isfinite(values).all(): raise FloatingPointError('Invalid edge scores')
            k=max(1,int(np.ceil(0.10*len(values))))
            per_snapshot['mean_high'].append(float(values.mean()))
            per_snapshot['mean_low'].append(float(-values.mean()))
            per_snapshot['tail_high'].append(float(values[-k:].mean()))
            per_snapshot['tail_low'].append(float(-values[:k].mean()))
        for key in result: result[key].append(float(np.mean(per_snapshot[key])))
    return {key:np.asarray(values) for key,values in result.items()}


def score_windows(model,graph_seqs):
    return score_candidates(model,graph_seqs)['mean_high']


def select_validation_score(candidates,labels):
    # Ranking selected by validation AP, ties by validation AUC, then fixed order.
    # Threshold optimized separately for F1 on that same held-out validation set.
    diagnostics={key:dict(pr_auc=float(average_precision_score(labels,values)),
                        roc_auc=float(roc_auc_score(labels,values)),
                        normal_median=float(np.median(values[labels==0])),
                        injected_median=float(np.median(values[labels==1])))
                 for key,values in candidates.items()}
    selected=max(candidates,key=lambda key:(diagnostics[key]['pr_auc'],diagnostics[key]['roc_auc']))
    return selected,calibrate(candidates[selected],labels),diagnostics


def calibrate(scores,labels):
    if not np.isfinite(scores).all():
        raise FloatingPointError('Invalid validation score.')
    unique=np.unique(scores)
    candidates=np.r_[np.nextafter(unique[0],-np.inf),
                     (unique[:-1]+unique[1:])/2,unique[-1]]
    best_t,best_f=float(candidates[0]),-1.0
    for t in candidates:
        _,_,f,_=precision_recall_fscore_support(labels,scores>t,average='binary',zero_division=0)
        if f>best_f:
            best_t,best_f=float(t),float(f)
    return best_t


def fingerprint(sequences):
    digest=hashlib.sha256()
    for seq in sequences:
        for s in seq:
            for tensor in (s.x,s.edge_index):
                a=tensor.cpu().contiguous().numpy()
                digest.update(str((a.shape,a.dtype)).encode()); digest.update(a.tobytes())
    return digest.hexdigest()


def main():
    snaps=list(TwitterTennisDatasetLoader().get_dataset())
    end=int(0.7*len(snaps)); dev,test=snaps[:end],snaps[end:]
    val_size=max(SEQ_LEN,round(0.2*len(dev))); train,val=dev[:-val_size],dev[-val_size:]
    all_train=windows(train)
    training=all_train[-max(1,round(TRAIN_FRACTION*len(all_train))):]
    validation,test_windows=windows(val),windows(test)
    if not training or not validation or len(test_windows)<4:
        raise ValueError('Too few windows.')
    evaluation,labels,_=make_test(test_windows,seed=TEST_INJECTION_SEED)
    test_hash=fingerprint(evaluation)
    # If AddGraph results are present, verify the shared evaluation inputs exactly.
    prior=OUTPUT_FILE.with_name('twitter_baseline_results.json')
    if prior.exists():
        other=json.loads(prior.read_text(encoding='utf-8'))
        if other['test_fingerprint']!=test_hash or other['test_labels']!=labels.tolist():
            raise ValueError('Test injection differs from saved AddGraph run; do not compare these runs.')
    training_graphs=graph_windows(training)
    test_graphs=graph_windows(evaluation)
    report=dict(method=METHOD,dataset='Twitter Tennis',seeds=SEEDS,epochs=EPOCHS,
                test_injection_seed=TEST_INJECTION_SEED,test_fingerprint=test_hash,
                test_labels=labels.tolist(),snapshots_split=[len(train),len(val),len(test)],
                train_windows=len(training),config=dict(gcn_hidden=GCN_HIDDEN,gru_hidden=GRU_HIDDEN,
                lr=LR,batch_size=BATCH_SIZE,context=SEQ_LEN,hops=1,label_limit=LABEL_LIMIT,
                train_edges_per_window=TRAIN_EDGES_PER_WINDOW),
                implementation='paper-based reimplementation; not official validated reproduction',
                deviations=['context four snapshots, early histories zero padded',
                'structural one-hot labels; observed node attributes are not used',
                'undirected enclosing subgraphs; directed edge candidates',
                'queried edge removed from enclosing subgraphs',
                'fixed structural label vocabulary with overflow category',
                'training target edge budget; endpoint-corrupted non-edges',
                'flattened SortPooling vectors feed GRU directly',
                '60th percentile of training sizes for SortPooling K',
                'four fixed window scoring candidates; validation AP selects aggregation and direction',
                '10 percent upper/lower edge tails or mean; snapshot scores averaged',
                '100 epochs, no automatic hyperparameter search'],runs=[])
    print(f'{METHOD} ONLY; device={DEVICE}; split={report["snapshots_split"]}; '
          f'train windows={len(training)}/{len(all_train)}; test windows={len(labels)}')
    for seed in SEEDS:
        val_eval,val_labels,_=make_test(validation,seed=seed+VAL_CALIB_SEED_OFFSET,ratio=VAL_ANOMALY_RATIO)
        val_hash=fingerprint(val_eval)
        seed_all(seed)
        start=time.perf_counter()
        model=fit_model(training_samples(training_graphs,seed),seed)
        val_candidates=score_candidates(model,graph_windows(val_eval))
        selected,threshold,val_diagnostics=select_validation_score(val_candidates,val_labels)
        val_scores=val_candidates[selected]
        raw_threshold=calibrate(val_candidates['mean_high'],val_labels)
        print(f'  Validation selection: {selected}; diagnostics={val_diagnostics}')
        # Freeze calibration before computing any test candidate or test metric.
        test_candidates=score_candidates(model,test_graphs)
        scores=test_candidates[selected]
        torch.save(dict(state_dict=model.state_dict(),sort_k=model.sort_k,
                        selected_score=selected,threshold=threshold,seed=seed),
                   OUTPUT_FILE.with_name(f'strgnn_validation_adapted_seed_{seed}.pt'))
        if not np.isfinite(scores).all():
            raise FloatingPointError('Invalid test scores.')
        pred=scores>threshold
        p,r,f,_=precision_recall_fscore_support(labels,pred,average='binary',zero_division=0)
        metrics=dict(roc_auc=float(roc_auc_score(labels,scores)),pr_auc=float(average_precision_score(labels,scores)),
                     precision=float(p),recall=float(r),f1=float(f),accuracy=float(np.mean(pred==labels)),
                     cm=confusion_matrix(labels,pred,labels=[0,1]).tolist(),threshold=threshold,
                     train_preprocess_calibrate_test_seconds=round(time.perf_counter()-start,2))
        raw_scores=test_candidates['mean_high']
        raw_pred=raw_scores>raw_threshold
        raw_p,raw_r,raw_f,_=precision_recall_fscore_support(labels,raw_pred,average='binary',zero_division=0)
        raw_metrics=dict(roc_auc=float(roc_auc_score(labels,raw_scores)),
                        pr_auc=float(average_precision_score(labels,raw_scores)),
                        precision=float(raw_p),recall=float(raw_r),f1=float(raw_f),
                        accuracy=float(np.mean(raw_pred==labels)),threshold=float(raw_threshold))
        report['runs'].append(dict(seed=seed,metrics=metrics,sort_k=model.sort_k,
                             selected_score=selected,validation_diagnostics=val_diagnostics,
                             raw_reference_metrics=raw_metrics,
                             validation_candidate_scores={k:v.tolist() for k,v in val_candidates.items()},
                             test_candidate_scores={k:v.tolist() for k,v in test_candidates.items()},
                             validation_fingerprint=val_hash,validation_labels=val_labels.tolist(),
                             validation_scores=val_scores.tolist(),test_scores=scores.tolist()))
        assert fingerprint(evaluation)==test_hash
        assert fingerprint(val_eval)==val_hash
        temporary=OUTPUT_FILE.with_suffix('.tmp')
        temporary.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8'); temporary.replace(OUTPUT_FILE)
        print(f'Seed {seed}: {metrics}')
    print('\nStrGNN: mean +/- sample standard deviation over five seeds')
    for key in ('roc_auc','pr_auc','precision','recall','f1','accuracy'):
        a=np.array([run['metrics'][key] for run in report['runs']])
        print(f'  {key}: {a.mean():.4f} +/- {a.std(ddof=1):.4f}')
    print(f'Saved: {OUTPUT_FILE}')


def self_test():
    seed_all(42)
    data=Data(x=torch.randn(6,3),edge_index=torch.tensor([[0,1,2,3],[1,2,3,4]],dtype=torch.long))
    graph=SnapshotGraph(data)
    sg=graph.subgraph(1,2)
    assert (sg.labels==1).sum()==2
    assert sg.labels.min()>=0 and sg.labels.max()<=LABEL_LIMIT
    samples=training_samples([[graph]*4],42)
    for graphs,t,a,b,label in samples:
        if label==1:
            assert (a,b) not in graphs[t].edge_set
    cache=SubgraphCache()
    contexts=[cache.sequence([graph]*4,0,1,2),cache.sequence([graph]*4,3,0,1)]
    assert contexts[0][:3]==[None]*3
    model=StrGNNWindow(10).to(DEVICE)
    logits=model(contexts)
    assert logits.shape==(2,) and torch.isfinite(logits).all()
    F.binary_cross_entropy_with_logits(logits,torch.tensor([0.,1.],device=DEVICE)).backward()
    assert model.layers[0].lin.weight.grad is not None
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    assert np.isfinite(score_windows(model,[[graph]*4])).all()
    assert 0.2<calibrate(np.array([0.1,0.2,0.8,0.9]),np.array([0,0,1,1]))<0.8
    print('Self-test passed: subgraphs, structural labels, negatives, causal histories, gradients, aggregation.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    self_test() if args.self_test else main()
