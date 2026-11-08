"""STG-GAN pour Reddit Hyperlinks (liens datés entre subreddits).
Exécution Spyder : adapter DATA_PATH, puis %runfile reddit_stggan_v2.py --wdir
Dépendances : pandas, numpy, torch, torch-geometric, torch-geometric-temporal, scikit-learn.
Les labels d'évaluation sont synthétiques (injection hybride), pas LINK_SENTIMENT.

v2 :
  - correction de protocole : l'injection de test est désormais tirée une
    seule fois (seed fixe), hors boucle des seeds d'entraînement, pour ne
    pas mélanger variance du modèle et variance de l'injection ;
  - seuil calibré par F1 sur une validation augmentée d'anomalies
    injectées (Solution 2), au lieu de mean + 2*std ;
  - régularisation légère : weight_decay + dropout (Solution 3), pour
    stabiliser la convergence (cf. seed 45 qui explosait en temps de calcul).
"""
import copy
import random
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric_temporal.nn.recurrent import GConvGRU
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, precision_recall_fscore_support

DATA_PATH =  r'path\to\dataset\soc-redditHyperlinks-body.tsv' # Update this path according to the dataset location on your computer.
NUM_SNAPSHOTS = 120             # bins temporels de durée égale ; ajuster selon le fichier
SEEDS = range(42, 47)
EPOCHS = 100
TRAIN_FRACTION = 0.5
SEQ_LEN = 4
HIDDEN = 32
LAMBDA_RECON = 5.0
TEST_ANOMALY_RATIO = 0.30
NODE_ANOMALY_RATIO = 0.03
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


def load_graphs(path):
    # Sep=None reconnaît le TSV Reddit et un éventuel CSV ; PROPERTIES reste une chaîne.
    df = pd.read_csv(path, sep=None, engine='python', usecols=lambda c: c.upper() in
                     {'SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', 'TIMESTAMP', 'LINK_SENTIMENT', 'PROPERTIES'})
    df.columns = df.columns.str.upper()
    required = {'SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', 'TIMESTAMP'}
    if not required.issubset(df.columns):
        raise ValueError(f'Colonnes absentes : {required - set(df.columns)}')
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], errors='coerce', utc=True)
    df = df.dropna(subset=['TIMESTAMP', 'SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT']).sort_values('TIMESTAMP').reset_index(drop=True)
    if df.empty: raise ValueError('Aucun lien daté valide.')
    # Le contenu textuel des posts et leurs IDs ne servent pas de labels.
    nodes = pd.Index(pd.unique(pd.concat([df.SOURCE_SUBREDDIT, df.TARGET_SUBREDDIT], ignore_index=True)))
    node_ids = pd.Series(np.arange(len(nodes), dtype=np.int64), index=nodes)
    df['src'] = node_ids.loc[df.SOURCE_SUBREDDIT].to_numpy()
    df['dst'] = node_ids.loc[df.TARGET_SUBREDDIT].to_numpy()
    duration = max((df.TIMESTAMP.max() - df.TIMESTAMP.min()).total_seconds(), 1.0)
    offsets = (df.TIMESTAMP - df.TIMESTAMP.min()).dt.total_seconds().to_numpy()
    df['bin'] = np.minimum((offsets / duration * NUM_SNAPSHOTS).astype(int), NUM_SNAPSHOTS - 1)
    # Définition des features sur les liens ; agrégation par nœud actif et par date.
    sentiment = pd.to_numeric(df.get('LINK_SENTIMENT', pd.Series(0, index=df.index)), errors='coerce').fillna(0).to_numpy(np.float32)
    # PROPERTIES : seulement les quatre premières valeurs numériques ; pas de dépendance aux labels.
    prop = np.zeros((len(df), 4), dtype=np.float32)
    if 'PROPERTIES' in df:
        for i, value in enumerate(df.PROPERTIES.fillna('').astype(str)):
            vals = np.fromstring(value, sep=',', dtype=np.float32)
            prop[i, :min(4, len(vals))] = vals[:4]
    prop[~np.isfinite(prop)] = 0
    # Compression des très grands attributs, sans ajustement statistique sur le test.
    edge_features = np.column_stack((sentiment, np.sign(prop) * np.log1p(np.abs(prop))))
    graphs = []
    for _, part in df.groupby('bin', sort=True):
        ids = np.unique(np.concatenate((part.src.to_numpy(), part.dst.to_numpy())))
        src = np.searchsorted(ids, part.src.to_numpy())
        dst = np.searchsorted(ids, part.dst.to_numpy())
        edge_index = torch.tensor(np.stack((src, dst)), dtype=torch.long)  # orienté source -> cible
        x = np.zeros((len(ids), 2 + 2 * edge_features.shape[1]), dtype=np.float32)
        ef = edge_features[part.index.to_numpy()]
        np.add.at(x[:, 0], src, 1); np.add.at(x[:, 1], dst, 1)
        np.add.at(x[:, 2:2 + ef.shape[1]], src, ef)
        np.add.at(x[:, 2 + ef.shape[1]:], dst, ef)
        x[:, 2:2 + ef.shape[1]] /= np.maximum(x[:, 0:1], 1)
        x[:, 2 + ef.shape[1]:] /= np.maximum(x[:, 1:2], 1)
        x[:, :2] = np.log1p(x[:, :2])
        graphs.append(Data(x=torch.from_numpy(x), edge_index=edge_index,
                           node_ids=torch.from_numpy(ids.astype(np.int64))))
    print(f'Liens={len(df)}, subreddits={len(nodes)}, snapshots actifs={len(graphs)}')
    if len(graphs) < 20:
        raise ValueError('Trop peu de snapshots actifs : réduire NUM_SNAPSHOTS ou vérifier TIMESTAMP.')
    return graphs


def windows(graphs):
    return [graphs[i:i + SEQ_LEN] for i in range(len(graphs) - SEQ_LEN + 1)]


def prep(snapshot):
    x = snapshot.x.to(DEVICE)
    return x, snapshot.edge_index.to(DEVICE), torch.zeros(len(x), dtype=torch.long, device=DEVICE), snapshot.node_ids.to(DEVICE)


class Generator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.rnn = GConvGRU(dim, HIDDEN, K=2)
        self.decoder = nn.Linear(HIDDEN, dim)

    def forward(self, sequence):
        # L'état caché est remappé par identifiant global pour chaque snapshot.
        prev_ids = prev_h = None
        fake, losses = [], []
        for x, edges, batch, ids in sequence:
            h = x.new_zeros((len(x), HIDDEN))
            if prev_ids is not None:
                old_ids, old_order = prev_ids.sort()
                pos = torch.searchsorted(old_ids, ids)
                valid = pos < len(old_ids)
                valid[valid.clone()] = old_ids[pos[valid]] == ids[valid]
                h[valid] = prev_h[old_order[pos[valid]]]
            h = self.rnn(x, edges, None, h)
            x_hat = self.decoder(h)
            fake.append((x_hat, edges, batch, ids))
            losses.append(F.mse_loss(x_hat, x))
            prev_ids, prev_h = ids, h
        return fake, torch.stack(losses).mean()


class AGD(nn.Module):
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
        for x, edges, batch, _ in sequence:
            if attention:
                h, (e1, a1) = self.gat1(x, edges, return_attention_weights=True)
                h = self.drop1(F.elu(h))
                h, (e2, a2) = self.gat2(h, edges, return_attention_weights=True)
                h = self.drop2(F.elu(h))
                weights.append(((e1, a1), (e2, a2), len(x)))
            else:
                h = self.drop1(F.elu(self.gat1(x, edges)))
                h = self.drop2(F.elu(self.gat2(h, edges)))
            pooled.append(global_mean_pool(h, batch))
        logit = self.classifier(self.transformer(torch.stack(pooled, dim=1)).mean(1)).flatten()
        return (logit, weights) if attention else logit


def train(seqs, dim):
    g, d = Generator(dim).to(DEVICE), AGD(dim).to(DEVICE)
    # --- Solution 3 : weight_decay sur les deux optimiseurs ---
    go = Adam(g.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
    do = Adam(d.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
    for epoch in range(EPOCHS):
        g.train(); d.train()
        sums = np.zeros(3)
        for sequence in seqs:
            real = [prep(s) for s in sequence]
            do.zero_grad(set_to_none=True)
            with torch.no_grad(): fake, _ = g(real)
            real_logits, fake_logits = d(real), d(fake)
            dl = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits)) + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
            dl.backward(); do.step()
            for p in d.parameters(): p.requires_grad_(False)
            go.zero_grad(set_to_none=True)
            fake, recon = g(real)
            logits = d(fake)
            gl = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits)) + LAMBDA_RECON * recon
            gl.backward(); go.step()
            for p in d.parameters(): p.requires_grad_(True)
            sums += [gl.item(), dl.item(), recon.item()]
        if epoch == 0 or (epoch + 1) % 10 == 0:
            a, b, c = sums / len(seqs)
            print(f'  Epoch {epoch + 1}/{EPOCHS}: 3G_loss={a:.4f} AGD_loss={b:.4f} reconstruction={c:.4f}')
    return g, d


def inject(sequence, ratio=NODE_ANOMALY_RATIO):
    changed, true_nodes = [], []
    for snapshot in sequence:
        s = copy.copy(snapshot)
        s.x, s.edge_index = snapshot.x.clone(), snapshot.edge_index.clone()
        n = len(s.x)
        chosen = torch.randperm(n)[:max(1, int(n * ratio))]
        s.x[chosen] = s.x[chosen] * 1.5 + torch.randn_like(s.x[chosen]) * 0.5
        if len(chosen) > 1:
            mask = (torch.rand(len(chosen), len(chosen)) < 0.2) & ~torch.eye(len(chosen), dtype=torch.bool)
            source, target = mask.nonzero(as_tuple=True)
            s.edge_index = torch.cat((s.edge_index, torch.stack((chosen[source], chosen[target]))), 1)
        changed.append(s); true_nodes.append(chosen)
    return changed, true_nodes


def make_test(seqs, seed, ratio=TEST_ANOMALY_RATIO, node_ratio=NODE_ANOMALY_RATIO):
    rng = np.random.default_rng(seed + 1000)
    selected = set(rng.choice(len(seqs), round(len(seqs) * ratio), replace=False))
    seed_all(seed + 2000)
    evaluation, labels, nodes = [], [], []
    for i, seq in enumerate(seqs):
        altered, marked = inject(seq, ratio=node_ratio) if i in selected else (seq, None)
        evaluation.append(altered); labels.append(int(i in selected)); nodes.append(marked)
    return evaluation, np.array(labels), nodes


def explain(d, sequence, marked):
    with torch.no_grad(): _, weights = d([prep(s) for s in sequence], attention=True)
    hits, at_least_one = [], []
    for ((e1, a1), (e2, a2), n), truth in zip(weights, marked):
        importance = torch.zeros(n, device=DEVICE)
        for edges, alpha in ((e1, a1), (e2, a2)):
            layer = torch.zeros_like(importance)
            vals = alpha.mean(1)
            layer.index_add_(0, edges[0], vals); layer.index_add_(0, edges[1], vals)
            importance += layer / layer.sum().clamp_min(1e-8)
        selected = set(importance.topk(len(truth)).indices.cpu().tolist())
        match = len(selected.intersection(truth.tolist()))
        hits.append(match / len(truth)); at_least_one.append(float(match > 0))
    return hits, at_least_one


def score(g, seq):
    with torch.no_grad(): return g([prep(s) for s in seq])[1].item()


def calibrate_threshold(g, val_seqs, seed):
    """Solution 2 : seuil qui maximise le F1 sur une validation augmentée
    d'anomalies injectées (même protocole que le test, seed différente pour
    ne jamais réutiliser les mêmes fenêtres/anomalies que le test)."""
    val_eval, val_labels, _ = make_test(
        val_seqs, seed=seed + VAL_CALIB_SEED_OFFSET, ratio=VAL_ANOMALY_RATIO)
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


def evaluate(g, d, val_seqs, test_seqs, labels, true_nodes, seed):
    g.eval(); d.eval()
    threshold = calibrate_threshold(g, val_seqs, seed)
    scores = np.array([score(g, s) for s in test_seqs])
    preds = (scores > threshold).astype(int)
    p, r, f, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    result = dict(roc_auc=float(roc_auc_score(labels, scores)), pr_auc=float(average_precision_score(labels, scores)), precision=float(p), recall=float(r), f1=float(f), accuracy=float(np.mean(labels == preds)), cm=[[int(tn), int(fp)], [int(fn), int(tp)]], threshold=float(threshold))
    matches, hits = [], []
    for seq, label, nodes in zip(test_seqs, labels, true_nodes):
        if label:
            matched, hit = explain(d, seq, nodes)
            matches.extend(matched); hits.extend(hit)
    result.update(precision_at_k=float(np.mean(matches)), recall_at_k=float(np.mean(matches)), hit_rate_at_k=float(np.mean(hits)))
    return result


def main():
    graphs = load_graphs(DATA_PATH)
    dev_end = int(0.7 * len(graphs))
    dev, test = graphs[:dev_end], graphs[dev_end:]
    val_size = max(SEQ_LEN, round(len(dev) * 0.2))
    train_graphs, val = dev[:-val_size], dev[-val_size:]
    all_train = windows(train_graphs)
    train_seqs = all_train[-max(1, round(TRAIN_FRACTION * len(all_train))):]
    val_seqs, test_seqs = windows(val), windows(test)
    if not train_seqs or not val_seqs or len(test_seqs) < 4:
        raise ValueError('Pas assez de fenêtres : ajuster NUM_SNAPSHOTS et SEQ_LEN.')
    # Normalisation entraînement seulement, appliquée aux trois partitions.
    train_values = torch.cat([s.x for s in train_graphs], dim=0)
    mean, std = train_values.mean(0), train_values.std(0).clamp_min(1e-6)
    for snapshot in graphs: snapshot.x = ((snapshot.x - mean) / std).clamp(-10, 10)
    print(f'Device={DEVICE}; train/val/test snapshots={len(train_graphs)}/{len(val)}/{len(test)}; training windows={len(train_seqs)}/{len(all_train)}')
    # Correction de protocole : injection de test fixée UNE SEULE FOIS, hors
    # boucle des seeds, pour ne pas mélanger variance du modèle et variance
    # de l'injection entre les seeds.
    evaluation, labels, marked = make_test(test_seqs, seed=TEST_INJECTION_SEED)
    results = []
    for seed in SEEDS:
        seed_all(seed)
        start = time.perf_counter()
        generator, discriminator = train(train_seqs, graphs[0].x.size(1))
        metrics = evaluate(generator, discriminator, val_seqs, evaluation, labels, marked, seed)
        metrics['seconds'] = round(time.perf_counter() - start, 1)
        results.append(metrics)
        print(f'Seed {seed}: {metrics}')
    print('STG-GAN Reddit (moyenne ± écart-type sur 5 graines)')
    for key in ('roc_auc', 'pr_auc', 'precision', 'recall', 'f1', 'accuracy', 'precision_at_k'):
        values = np.array([m[key] for m in results])
        print(f'  {key}: {values.mean():.4f} ± {values.std(ddof=1):.4f}')


if __name__ == '__main__':
    main()
