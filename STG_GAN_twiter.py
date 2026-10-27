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

if __name__ == '__main__':
    main()