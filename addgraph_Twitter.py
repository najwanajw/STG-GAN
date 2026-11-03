# COMPARISON: STG-GAN v2 vs AddGraph-PyG-window-adapted.
# Paper-based adaptation, not the official AddGraph implementation.
# Source: Zheng et al., IJCAI 2019, https://www.ijcai.org/proceedings/2019/0614.pdf
# Differences: H0 from the first observed X; independent L=4 windows;
# static trained weights during evaluation (no online test refinement);
# mean edge scores -> snapshot score -> mean window score.
# A comparison with original AddGraph requires validating this port separately.
# Dependencies: your existing STG-GAN dependencies, no additional package.
# Spyder: %runfile twitter_stggan_addgraph_comparison.py --wdir
# CLI: python twitter_stggan_addgraph_comparison.py --self-test

"""Twitter Tennis STG-GAN: 50% des fenêtres d'entraînement, cinq graines.
Dependencies: torch, torch-geometric, torch-geometric-temporal, scikit-learn, numpy.

v2 :
  - correction de protocole : l'injection de test est tirée une seule fois
    (seed fixe), hors boucle des seeds d'entraînement ;
  - seuil calibré par F1 sur une validation augmentée d'anomalies injectées
    (Solution 2), au lieu de mean + 2*std ;
  - régularisation légère : weight_decay + dropout (Solution 3).
"""
import copy
import random
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric_temporal.dataset import TwitterTennisDatasetLoader
from torch_geometric_temporal.nn.recurrent import GConvGRU
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, confusion_matrix

FAST = True
SEEDS = list(range(42, 47))
EPOCHS = 100
TRAIN_FRACTION = 0.5 if FAST else 1.0
RUN_ABLATION = not FAST
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN, HIDDEN, LAMBDA_RECON = 4, 32, 5.0
TEST_RATIO, NODE_RATIO = 0.30, 0.03

# --- Solution 3 : régularisation légère ---------------------------------
WEIGHT_DECAY = 1e-4
DROPOUT = 0.15

# --- Solution 2 : seuil F1-calibré sur validation augmentée -------------
VAL_ANOMALY_RATIO = 0.30
VAL_CALIB_SEED_OFFSET = 5000

# --- Correction de protocole : injection de test fixe --------------------
TEST_INJECTION_SEED = 2026


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def windows(snaps):
    return [snaps[i:i + SEQ_LEN] for i in range(len(snaps) - SEQ_LEN + 1)]

def prep(s):
    x, edges = s.x.to(DEVICE), s.edge_index.to(DEVICE)
    return x, edges, torch.zeros(x.size(0), device=DEVICE, dtype=torch.long)

class Generator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.rnn = GConvGRU(dim, HIDDEN, K=2)
        self.decoder = nn.Linear(HIDDEN, dim)
    def forward(self, x, edges, h):
        h = self.rnn(x, edges, None, h)
        return self.decoder(h), h

class Discriminator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gat1 = GATConv(dim, 16, heads=2)
        self.gat2 = GATConv(32, 32)
        # --- Solution 3 : dropout léger après chaque couche GAT ---
        self.drop1 = nn.Dropout(DROPOUT)
        self.drop2 = nn.Dropout(DROPOUT)
        layer = nn.TransformerEncoderLayer(32, 2, batch_first=True, dropout=DROPOUT)
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.classifier = nn.Linear(32, 1)
    def forward(self, sequence, attention=False):
        pooled, weights = [], []
        for x, edges, batch in sequence:
            if attention:
                h, (ei1, a1) = self.gat1(x, edges, return_attention_weights=True)
                h = self.drop1(F.elu(h))
                h, (ei2, a2) = self.gat2(h, edges, return_attention_weights=True)
                h = self.drop2(F.elu(h))
                weights.append(((ei1, a1), (ei2, a2), x.size(0)))
            else:
                h = self.drop1(F.elu(self.gat1(x, edges)))
                h = self.drop2(F.elu(self.gat2(h, edges)))
            pooled.append(global_mean_pool(h, batch))
        logits = self.classifier(self.transformer(torch.stack(pooled, 1)).mean(1)).flatten()
        return (logits, weights) if attention else logits

def generated(generator, prepared):
    h, fake, losses = None, [], []
    for x, edges, batch in prepared:
        x_hat, h = generator(x, edges, h)
        fake.append((x_hat, edges, batch))
        losses.append(F.mse_loss(x_hat, x))
    return fake, torch.stack(losses).mean()

def train(train_seqs, dim, adversarial):
    g = Generator(dim).to(DEVICE)
    d = Discriminator(dim).to(DEVICE) if adversarial else None
    # --- Solution 3 : weight_decay sur les deux optimiseurs ---
    opt_g = Adam(g.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
    opt_d = Adam(d.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY) if d else None
    for epoch in range(EPOCHS):
        g.train()
        if d: d.train()
        recon_total = g_total = d_total = 0.0
        for seq in train_seqs:
            prepared = [prep(s) for s in seq]
            if d:
                opt_d.zero_grad(set_to_none=True)
                real = d(prepared)
                with torch.no_grad(): fake, _ = generated(g, prepared)
                fake_logits = d(fake)
                d_loss = F.binary_cross_entropy_with_logits(real, torch.ones_like(real)) + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
                d_loss.backward(); opt_d.step()
                d_total += d_loss.item()
                # Freeze discriminator parameters while preserving gradients through its input.
                for p in d.parameters(): p.requires_grad_(False)
            opt_g.zero_grad(set_to_none=True)
            fake, recon = generated(g, prepared)
            loss = LAMBDA_RECON * recon
            if d:
                logits = d(fake)
                loss = loss + F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
            loss.backward(); opt_g.step()
            g_total += loss.item()
            if d:
                for p in d.parameters(): p.requires_grad_(True)
            recon_total += recon.item()
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == EPOCHS:
            n = len(train_seqs)
            print(f'  Epoch {epoch + 1}/{EPOCHS}: 3G_loss={g_total / n:.4f} '
                  f'AGD_loss={d_total / n:.4f} reconstruction={recon_total / n:.4f}')
    return g, d

def score(g, seq):
    with torch.no_grad():
        return generated(g, [prep(s) for s in seq])[1].item()

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

def explanation(d, seq, true_nodes):
    with torch.no_grad(): _, attentions = d([prep(s) for s in seq], attention=True)
    hits, hit_rates = [], []
    for ((ei1, a1), (ei2, a2), n), ground_truth in zip(attentions, true_nodes):
        importance = torch.zeros(n, device=DEVICE)
        for edges, alpha in ((ei1, a1), (ei2, a2)):
            edge_weight = alpha.mean(dim=1)
            layer = torch.zeros_like(importance)
            layer.index_add_(0, edges[0], edge_weight)
            layer.index_add_(0, edges[1], edge_weight)
            importance += layer / layer.sum().clamp_min(1e-8)
        k = len(ground_truth)
        selected = set(importance.topk(k).indices.cpu().tolist())
        hits.append(len(selected.intersection(ground_truth.tolist())) / k)
        hit_rates.append(float(bool(selected.intersection(ground_truth.tolist()))))
    return hits, hit_rates

def calibrate_threshold(g, val_seqs, seed):
    """Solution 2 : seuil qui maximise le F1 sur une validation augmentée
    d'anomalies injectées (seed différente de celle du test)."""
    val_eval, val_labels, _ = make_test(val_seqs, seed=seed + VAL_CALIB_SEED_OFFSET, ratio=VAL_ANOMALY_RATIO)
    val_scores = np.array([score(g, seq) for seq in val_eval])
    candidates = np.unique(val_scores)
    midpoints = (candidates[:-1] + candidates[1:]) / 2
    if len(midpoints) == 0:
        midpoints = candidates
    best_t, best_f1 = float(midpoints[0]), -1.0
    for t in midpoints:
        pred = (val_scores > t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(val_labels, pred, average='binary', zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t

def evaluate(g, d, val_seqs, evaluation, labels, true_nodes, seed):
    g.eval()
    if d: d.eval()
    threshold = calibrate_threshold(g, val_seqs, seed)
    scores = np.array([score(g, seq) for seq in evaluation])
    preds = (scores > threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    result = dict(roc_auc=float(roc_auc_score(labels, scores)),
                  pr_auc=float(average_precision_score(labels, scores)),
                  precision=float(precision), recall=float(recall), f1=float(f1),
                  accuracy=float((preds == labels).mean()),
                  cm=[[int(tn), int(fp)], [int(fn), int(tp)]], threshold=float(threshold))
    if d:
        all_hits, all_hit_rates = [], []
        for seq, label, nodes in zip(evaluation, labels, true_nodes):
            if label:
                hit, hit_rate = explanation(d, seq, nodes)
                all_hits.extend(hit); all_hit_rates.extend(hit_rate)
        # k equals the number of injected nodes: Precision@k = Recall@k.
        result.update(precision_at_k=float(np.mean(all_hits)), recall_at_k=float(np.mean(all_hits)), hit_rate_at_k=float(np.mean(all_hit_rates)))
    return result

def main():
    snapshots = list(TwitterTennisDatasetLoader().get_dataset())
    dev_end = int(0.7 * len(snapshots))
    dev, test = snapshots[:dev_end], snapshots[dev_end:]
    val_size = max(SEQ_LEN, round(0.2 * len(dev)))
    train_snapshots, val = dev[:-val_size], dev[-val_size:]
    all_train = windows(train_snapshots)
    # A contiguous half retains intact temporal windows; validation and test stay fixed.
    train_seqs = all_train[-max(1, round(TRAIN_FRACTION * len(all_train))):]
    val_seqs, test_seqs = windows(val), windows(test)
    if not train_seqs or not val_seqs or len(test_seqs) < 4:
        raise ValueError('Too few temporal windows for training, validation, or test.')
    print(f'Device={DEVICE}; snapshots train/val/test={len(train_snapshots)}/{len(val)}/{len(test)}; training windows={len(train_seqs)}/{len(all_train)}')
    # Correction de protocole : injection de test fixée UNE SEULE FOIS, hors
    # boucle des seeds.
    evaluation, labels, nodes = make_test(test_seqs, seed=TEST_INJECTION_SEED)
    main_results, ablations = [], []
    for seed in SEEDS:
        seed_all(seed)
        start = time.perf_counter()
        g, d = train(train_seqs, train_snapshots[0].x.size(1), adversarial=True)
        result = evaluate(g, d, val_seqs, evaluation, labels, nodes, seed)
        result['seconds'] = round(time.perf_counter() - start, 1)
        main_results.append(result)
        print(f'Seed {seed}: {result}')
        if RUN_ABLATION:
            seed_all(seed)
            baseline, _ = train(train_seqs, train_snapshots[0].x.size(1), adversarial=False)
            ablation = evaluate(baseline, None, val_seqs, evaluation, labels, nodes, seed)
            ablations.append(ablation)
            print(f'Ablation {seed}: {ablation}')
    if len(SEEDS) > 1:
        summaries = [('STG-GAN', main_results)]
        if ablations:
            summaries.append(('Reconstruction seule', ablations))
        for name, results in summaries:
            print(name)
            for metric in ('roc_auc', 'pr_auc', 'f1', 'accuracy'):
                values = np.array([r[metric] for r in results])
                print(f'  {metric}: {values.mean():.4f} ± {values.std(ddof=1):.4f}')

# The comparison entry point below replaces the original main().

import argparse
import hashlib
import json
from pathlib import Path
from torch_geometric.nn import GCNConv
from torch_geometric.utils import coalesce, to_undirected, remove_self_loops

BASELINE_NAME = 'AddGraph-PyG-window-adapted'
ADDGRAPH_CONFIG = dict(hidden=32, context=4, lr=0.002, weight_decay=5e-7,
                       dropout=0.2, margin=0.5, beta=1.0, mu=0.3)
OUTPUT_FILE = Path(__file__).resolve().with_name('twitter_baseline_results.json')
# True also reruns STG-GAN on EXACTLY the same data. False runs AddGraph alone.
RUN_STGGAN = False


class AddGraphWindow(nn.Module):
    """Paper-based GCN + contextual attention + GRU + edge margin scorer.

    GCN propagates the previous hidden states; the contextual attention uses
    preceding hidden states. Explicit gates implement Eq. 12--15 instead of
    substituting a different GRU convention. All deviations are in the header.
    """
    def __init__(self, dim, cfg):
        super().__init__()
        self.cfg = cfg
        h = cfg['hidden']
        self.initial = nn.Linear(dim, h)
        self.gcn1, self.gcn2 = GCNConv(h, h), GCNConv(h, h)
        self.att_q = nn.Linear(h, h, bias=False)
        self.att_r = nn.Linear(h, 1, bias=False)
        self.up, self.wp = nn.Linear(h, h), nn.Linear(h, h, bias=False)
        self.ur, self.wr = nn.Linear(h, h), nn.Linear(h, h, bias=False)
        self.uc, self.wc = nn.Linear(h, h, bias=False), nn.Linear(h, h, bias=False)
        self.edge_a = nn.Parameter(torch.empty(h))
        self.edge_b = nn.Parameter(torch.empty(h))
        nn.init.normal_(self.edge_a, std=0.1)
        nn.init.normal_(self.edge_b, std=0.1)

    def encode(self, seq):
        initial_x = seq[0].x.to(DEVICE)
        h = torch.tanh(self.initial(initial_x))
        n = len(initial_x)
        history, states = [h], []
        for snapshot in seq:
            if snapshot.x.size(0) != n:
                raise ValueError('This Twitter adapter requires aligned fixed node IDs.')
            edges = snapshot.edge_index.to(DEVICE)
            # Undirected structural propagation; edge scoring stays directed.
            graph_edges = to_undirected(coalesce(edges, num_nodes=n), num_nodes=n)
            current = F.relu(self.gcn1(h, graph_edges))
            current = F.dropout(current, self.cfg['dropout'], self.training)
            current = F.relu(self.gcn2(current, graph_edges))
            current = F.dropout(current, self.cfg['dropout'], self.training)
            past = torch.stack(history[-self.cfg['context']:], dim=1)
            weights = self.att_r(torch.tanh(self.att_q(past))).softmax(dim=1)
            short = (weights * past).sum(dim=1)
            gate = torch.sigmoid(self.up(current) + self.wp(short))
            reset = torch.sigmoid(self.ur(current) + self.wr(short))
            proposal = torch.tanh(self.uc(current) + self.wc(reset * short))
            h = (1.0 - gate) * short + gate * proposal
            states.append(h)
            history.append(h)
        return states

    def edge_scores(self, h, edges):
        # Eq. 16 defines sigmoid as 1/(1+exp(x)), i.e. torch.sigmoid(-x).
        a, b = edges
        distance = ((h[a] * self.edge_a + h[b] * self.edge_b) ** 2).sum(dim=1)
        return torch.sigmoid(-self.cfg['beta'] * (distance - self.cfg['mu']))


def endpoint_negative_pairs(edges, n, max_attempts=50):
    """Replace one endpoint, using incident-degree Bernoulli probabilities.

    Negatives are not observed directed edges and are not self-loops.
    Returns only successfully sampled pairs, with their matching positives.
    This never allocates a dense N x N adjacency matrix.
    """
    edges = coalesce(remove_self_loops(edges)[0], num_nodes=n)
    if edges.numel() == 0:
        return edges, edges
    existing = edges[0] * n + edges[1]
    degree = torch.bincount(edges.flatten(), minlength=n).float()
    a, b = edges
    probability = degree[a] / (degree[a] + degree[b]).clamp_min(1)
    replace_source = torch.rand(a.size(0), device=edges.device) < probability
    negative = edges.clone()
    remaining = torch.ones(a.size(0), device=edges.device, dtype=torch.bool)
    for _ in range(max_attempts):
        ids = remaining.nonzero(as_tuple=True)[0]
        if not ids.numel():
            break
        sampled = torch.randint(n, (ids.numel(),), device=edges.device)
        source = replace_source[ids]
        negative[0, ids] = torch.where(source, sampled, a[ids])
        negative[1, ids] = torch.where(source, b[ids], sampled)
        bad = (negative[0, ids] == negative[1, ids]) | torch.isin(
            negative[0, ids] * n + negative[1, ids], existing)
        remaining[ids] = bad
    success = ~remaining
    return edges[:, success], negative[:, success]


def train_addgraph(train_seqs, dim):
    model = AddGraphWindow(dim, ADDGRAPH_CONFIG).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=ADDGRAPH_CONFIG['lr'],
                     weight_decay=ADDGRAPH_CONFIG['weight_decay'])
    model.train()
    for epoch in range(EPOCHS):
        total = count = 0
        for seq in train_seqs:
            optimizer.zero_grad(set_to_none=True)
            states = model.encode(seq)
            terms = []
            for snapshot, h in zip(seq, states):
                pos, neg = endpoint_negative_pairs(snapshot.edge_index.to(DEVICE), h.size(0))
                if pos.numel():
                    terms.append(F.relu(ADDGRAPH_CONFIG['margin']
                                       + model.edge_scores(h, pos)
                                       - model.edge_scores(h, neg)).mean())
            if not terms:
                continue
            loss = torch.stack(terms).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite AddGraph loss.')
            loss.backward()
            optimizer.step()
            total += loss.item()
            count += 1
        if not count:
            raise ValueError('No non-edge negative pairs are available for training.')
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f'  Epoch {epoch+1}/{EPOCHS}: AddGraph margin_loss={total/count:.4f}')
    model.eval()
    return model


@torch.no_grad()
def addgraph_score(model, seq):
    """Fixed aggregation, chosen before examining test scores or labels."""
    scores = []
    for snapshot, h in zip(seq, model.encode(seq)):
        edges = coalesce(remove_self_loops(snapshot.edge_index.to(DEVICE))[0],
                         num_nodes=h.size(0))
        if edges.numel():
            scores.append(model.edge_scores(h, edges).mean())
        else:
            scores.append(h.new_zeros(()))
    return torch.stack(scores).mean().item()


def threshold_from_validation(scores, labels):
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        raise FloatingPointError('Non-finite validation scores.')
    unique = np.unique(scores)
    candidates = (unique[:-1] + unique[1:]) / 2 if len(unique) > 1 else unique
    best_t, best_f1 = float(candidates[0]), -1.0
    for t in candidates:
        _, _, f1, _ = precision_recall_fscore_support(
            labels, scores > t, average='binary', zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t


def detection_metrics(scores, labels, threshold):
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        raise FloatingPointError('Non-finite test scores.')
    pred = scores > threshold
    p, r, f, _ = precision_recall_fscore_support(labels, pred, average='binary', zero_division=0)
    return dict(roc_auc=float(roc_auc_score(labels, scores)),
                pr_auc=float(average_precision_score(labels, scores)),
                precision=float(p), recall=float(r), f1=float(f),
                accuracy=float((pred == labels).mean()),
                cm=confusion_matrix(labels, pred, labels=[0, 1]).tolist(),
                threshold=float(threshold))


def data_fingerprint(windows_to_hash):
    digest = hashlib.sha256()
    for seq in windows_to_hash:
        for s in seq:
            for tensor in (s.x, s.edge_index):
                arr = tensor.detach().cpu().contiguous().numpy()
                digest.update(str((arr.shape, arr.dtype)).encode())
                digest.update(arr.tobytes())
    return digest.hexdigest()


def write_report(report):
    # Save after every completed seed/method, so a long run can be reviewed.
    temporary = OUTPUT_FILE.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(OUTPUT_FILE)


def comparison_main():
    snapshots = list(TwitterTennisDatasetLoader().get_dataset())
    dev_end = int(0.7 * len(snapshots))
    dev, test = snapshots[:dev_end], snapshots[dev_end:]
    val_size = max(SEQ_LEN, round(0.2 * len(dev)))
    train_snapshots, val = dev[:-val_size], dev[-val_size:]
    all_train = windows(train_snapshots)
    train_seqs = all_train[-max(1, round(TRAIN_FRACTION * len(all_train))):]
    val_seqs, test_seqs = windows(val), windows(test)
    if not train_seqs or not val_seqs or len(test_seqs) < 4:
        raise ValueError('Too few windows.')
    # Exactly the same function/seeds as the provided STG-GAN v2 script.
    evaluation, labels, nodes = make_test(test_seqs, seed=TEST_INJECTION_SEED)
    fixed_hash = data_fingerprint(evaluation)
    report = dict(dataset='Twitter Tennis', seeds=SEEDS, epochs=EPOCHS,
                  snapshots_split=[len(train_snapshots), len(val), len(test)],
                  train_windows=len(train_seqs), all_train_windows=len(all_train),
                  test_injection_seed=TEST_INJECTION_SEED, test_labels=labels.tolist(),
                  test_fingerprint=fixed_hash, test_ratio=TEST_RATIO, node_ratio=NODE_RATIO,
                  aggregation='AddGraph: mean directed edge scores, then mean snapshots',
                  baseline_name=BASELINE_NAME, baseline_config=ADDGRAPH_CONFIG,
                  baseline_implementation='paper-based adaptation, not official reproduction',
                  deviations=['project first-snapshot features to H0',
                              'reset hidden state for each overlapping window',
                              'symmetrized GCN propagation; directed edge scoring',
                              'frozen weights at validation and test; no online refinement',
                              'hidden width 32; parameters configured, not tuned',
                              'no node-localization metric for this edge-score baseline'],
                  runs=[])
    print(f'Device={DEVICE}; snapshots={report["snapshots_split"]}; '
          f'train windows={len(train_seqs)}/{len(all_train)}; test={len(labels)}, '
          f'injected={int(labels.sum())}; seeds={SEEDS}')
    print(f'Baseline: {BASELINE_NAME}; no automatic hyperparameter search.')
    methods = (['STG-GAN-v2'] if RUN_STGGAN else []) + [BASELINE_NAME]
    for seed in SEEDS:
        # Calibration data are constructed once per seed, then shared by methods.
        val_eval, val_labels, _ = make_test(
            val_seqs, seed=seed + VAL_CALIB_SEED_OFFSET, ratio=VAL_ANOMALY_RATIO)
        val_hash = data_fingerprint(val_eval)
        for name in methods:
            seed_all(seed)
            start = time.perf_counter()
            print(f'\n{name}, seed {seed}')
            if name == 'STG-GAN-v2':
                g, d = train(train_seqs, snapshots[0].x.size(1), adversarial=True)
                g.eval(); d.eval()
                scorer = lambda seq: score(g, seq)
            else:
                model = train_addgraph(train_seqs, snapshots[0].x.size(1))
                scorer = lambda seq: addgraph_score(model, seq)
            val_scores = np.array([scorer(seq) for seq in val_eval])
            threshold = threshold_from_validation(val_scores, val_labels)
            test_scores = np.array([scorer(seq) for seq in evaluation])
            metric = detection_metrics(test_scores, labels, threshold)
            metric['train_calibrate_test_seconds'] = round(time.perf_counter()-start, 2)
            # Attention ranking retained for STG-GAN; no invented baseline explanation.
            if name == 'STG-GAN-v2':
                hits, rates = [], []
                for seq, label, truth in zip(evaluation, labels, nodes):
                    if label:
                        h, r = explanation(d, seq, truth)
                        hits.extend(h); rates.extend(r)
                metric.update(precision_at_k=float(np.mean(hits)),
                              recall_at_k=float(np.mean(hits)),
                              hit_rate_at_k=float(np.mean(rates)))
            assert data_fingerprint(evaluation) == fixed_hash, 'Test input changed!'
            assert data_fingerprint(val_eval) == val_hash, 'Validation input changed!'
            print(metric)
            report['runs'].append(dict(method=name, seed=seed, metrics=metric,
                                       validation_fingerprint=val_hash,
                                       validation_labels=val_labels.tolist(),
                                       validation_scores=val_scores.tolist(),
                                       test_scores=test_scores.tolist()))
            write_report(report)
    print('\nMEAN +/- SAMPLE STANDARD DEVIATION, FIVE SEEDS')
    for name in methods:
        runs = [r['metrics'] for r in report['runs'] if r['method'] == name]
        print(name)
        for key in ('roc_auc','pr_auc','precision','recall','f1','accuracy'):
            a = np.array([r[key] for r in runs])
            print(f'  {key}: {a.mean():.4f} +/- {a.std(ddof=1):.4f}')
    print(f'\nSaved: {OUTPUT_FILE}')


def self_test():
    """Meaningful checks without downloading Twitter or running 100 epochs."""
    from torch_geometric.data import Data
    seed_all(42)
    seq = [Data(x=torch.randn(8, 3), edge_index=torch.tensor(
        [[0,1,2,3,4,5,6],[1,2,3,4,5,6,7]], dtype=torch.long)) for _ in range(4)]
    model = AddGraphWindow(3, ADDGRAPH_CONFIG).to(DEVICE)
    states = model.encode(seq)
    assert len(states) == 4 and states[0].shape == (8, 32)
    pos, neg = endpoint_negative_pairs(seq[0].edge_index.to(DEVICE), 8)
    assert pos.shape == neg.shape and pos.shape[1] > 0
    assert not torch.isin(neg[0]*8+neg[1], pos[0]*8+pos[1]).any()
    assert (neg[0] != neg[1]).all()
    loss = F.relu(0.5 + model.edge_scores(states[-1], pos)
                  - model.edge_scores(states[-1], neg)).mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.initial.weight.grad is not None
    model.eval()
    assert np.isfinite(addgraph_score(model, seq))
    threshold = threshold_from_validation([0.1,0.2,0.8,0.9], np.array([0,0,1,1]))
    assert detection_metrics([0.1,0.2,0.8,0.9], np.array([0,0,1,1]), threshold)['f1'] == 1
    before = data_fingerprint([seq])
    altered, _ = inject(seq)
    assert data_fingerprint([seq]) == before
    assert data_fingerprint([altered]) != before
    print('Self-test passed: negative pairs, gradients, score aggregation, calibration, injection immutability.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    self_test() if args.self_test else comparison_main()
