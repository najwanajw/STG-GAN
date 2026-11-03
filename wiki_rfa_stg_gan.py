"""STG-GAN sur Wiki-RfA avec injection de votes hybrides visibles.

Spyder : adapter DATA_PATH puis %runfile wiki_rfa_stggan_hybrid.py --wdir
Dépendances : numpy, pandas, torch, torch-geometric,
torch-geometric-temporal, scikit-learn.

Les étiquettes sont synthétiques. Une fenêtre est positive si elle reçoit
des votes injectés. Ce protocole ne mesure pas des anomalies réelles de RfA.
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
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric_temporal.nn.recurrent import GConvGRU
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             confusion_matrix, precision_recall_fscore_support)

DATA_PATH = r'C:\Users\HP\mon-projet\dataset\wiki-RfA.txt'
NUM_SNAPSHOTS = 120
SEQ_LEN = 4
SEEDS = range(42, 47)
EPOCHS = 100
TRAIN_FRACTION = 0.5
HIDDEN = 32
LAMBDA_RECON = 5.0
TEST_ANOMALY_RATIO = 0.30
NODE_ANOMALY_RATIO = 0.03
VOTES_PER_SOURCE = 16            # injection volontairement visible
THRESHOLD_STD = 2.0               # seuil sur validation normale uniquement
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_votes(path):
    """Chaque bloc commence par SRC; TXT et les autres champs sont ignorés."""
    rows, current = [], {}

    def flush():
        if {'SRC', 'TGT', 'DAT', 'VOT'} <= current.keys():
            rows.append((current['SRC'], current['TGT'], current['DAT'],
                         current['VOT']))

    with open(path, encoding='utf-8', errors='replace') as handle:
        for raw in handle:
            line = raw.rstrip('\r\n')
            if line.startswith('SRC:'):
                if current:
                    flush()
                current = {'SRC': line[4:].strip()}
            elif line[:4] in ('TGT:', 'DAT:', 'VOT:'):
                current[line[:3]] = line[4:].strip()
        if current:
            flush()

    df = pd.DataFrame(rows, columns=['SRC', 'TGT', 'DAT', 'VOT'])
    if df.empty:
        raise ValueError('Aucun bloc SRC/TGT/DAT/VOT trouvé.')
    # Wiki-RfA emploie principalement "HH:MM, DD Month YYYY".
    dt = pd.to_datetime(df.DAT, format='%H:%M, %d %B %Y',
                        errors='coerce', utc=True)
    missing = dt.isna() & df.DAT.ne('')
    if missing.any():
        dt.loc[missing] = pd.to_datetime(df.loc[missing, 'DAT'],
                                         format='mixed', dayfirst=True,
                                         errors='coerce', utc=True)
    df['time'] = dt
    vote = pd.to_numeric(df.VOT, errors='coerce')
    good = (df.time.notna() & df.SRC.ne('') & df.TGT.ne('') &
            vote.isin([-1, 1]))
    print(f'Votes conservés={good.sum()}, exclus={(~good).sum()} '
          f'(dates invalides={df.time.isna().sum()}, votes non ±1='
          f'{(~vote.isin([-1, 1])).sum()})')
    df = df.loc[good].copy().sort_values('time').reset_index(drop=True)
    if df.empty:
        raise ValueError('Aucun vote valide après filtrage.')
    df['vote'] = pd.to_numeric(df.VOT).astype(np.int8)
    return df


def features(edge_index, edge_vote, n):
    """Six caractéristiques, toutes cohérentes avec les votes du graphe."""
    src, dst = edge_index.numpy()
    votes = edge_vote.numpy()
    x = np.zeros((n, 6), dtype=np.float32)
    np.add.at(x[:, 0], src, 1)
    np.add.at(x[:, 1], dst, 1)
    np.add.at(x[:, 2], src, votes == 1)
    np.add.at(x[:, 3], src, votes == -1)
    np.add.at(x[:, 4], dst, votes == 1)
    np.add.at(x[:, 5], dst, votes == -1)
    return torch.from_numpy(np.log1p(x))


def load_graphs(path):
    df = read_votes(path)
    names = pd.Index(pd.unique(pd.concat([df.SRC, df.TGT], ignore_index=True)))
    mapping = pd.Series(np.arange(len(names), dtype=np.int64), index=names)
    df['src'] = mapping.loc[df.SRC].to_numpy()
    df['dst'] = mapping.loc[df.TGT].to_numpy()
    span = max((df.time.iloc[-1] - df.time.iloc[0]).total_seconds(), 1.0)
    offsets = (df.time - df.time.iloc[0]).dt.total_seconds().to_numpy()
    df['bin'] = np.minimum((offsets / span * NUM_SNAPSHOTS).astype(int),
                           NUM_SNAPSHOTS - 1)
    graphs = []
    for _, part in df.groupby('bin', sort=True):
        ids = np.unique(np.r_[part.src.to_numpy(), part.dst.to_numpy()])
        src = np.searchsorted(ids, part.src.to_numpy())
        dst = np.searchsorted(ids, part.dst.to_numpy())
        edges = torch.from_numpy(np.stack([src, dst]).astype(np.int64))
        votes = torch.from_numpy(part.vote.to_numpy(np.int8))
        graphs.append(Data(x=features(edges, votes, len(ids)),
                           edge_index=edges, edge_vote=votes,
                           node_ids=torch.from_numpy(ids.astype(np.int64))))
    print(f'Utilisateurs={len(names)}, snapshots actifs={len(graphs)}')
    if len(graphs) < 20:
        raise ValueError('Trop peu de snapshots actifs; revoir NUM_SNAPSHOTS.')
    return graphs


def windows(graphs):
    return [graphs[i:i + SEQ_LEN] for i in range(len(graphs) - SEQ_LEN + 1)]


def prep(snapshot):
    x = snapshot.x.to(DEVICE)
    return (x, snapshot.edge_index.to(DEVICE),
            torch.zeros(len(x), dtype=torch.long, device=DEVICE),
            snapshot.node_ids.to(DEVICE))


class Generator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.rnn = GConvGRU(dim, HIDDEN, K=2)
        self.decoder = nn.Linear(HIDDEN, dim)

    def forward(self, sequence):
        prev_ids = prev_h = None
        fake, losses = [], []
        for x, edges, batch, ids in sequence:
            h = x.new_zeros((len(x), HIDDEN))
            if prev_ids is not None:
                old_ids, order = prev_ids.sort()
                pos = torch.searchsorted(old_ids, ids)
                in_range = pos < len(old_ids)
                matching = torch.zeros_like(in_range)
                matching[in_range] = old_ids[pos[in_range]] == ids[in_range]
                h[matching] = prev_h[order[pos[matching]]]
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
        layer = nn.TransformerEncoderLayer(32, 2, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.classifier = nn.Linear(32, 1)

    def forward(self, sequence, attention=False):
        pooled, weights = [], []
        for x, edges, batch, _ in sequence:
            if attention:
                h, (e1, a1) = self.gat1(x, edges, return_attention_weights=True)
                h, (e2, a2) = self.gat2(F.elu(h), edges,
                                       return_attention_weights=True)
                weights.append(((e1, a1), (e2, a2), len(x)))
            else:
                h = self.gat2(F.elu(self.gat1(x, edges)), edges)
            pooled.append(global_mean_pool(F.elu(h), batch))
        logit = self.classifier(self.transformer(torch.stack(pooled, 1)).mean(1))
        return (logit.flatten(), weights) if attention else logit.flatten()


def train(seqs, dim):
    g, d = Generator(dim).to(DEVICE), AGD(dim).to(DEVICE)
    go, do = Adam(g.parameters(), lr=1e-3), Adam(d.parameters(), lr=1e-3)
    for epoch in range(EPOCHS):
        g.train(); d.train()
        sums = np.zeros(3)
        for seq in seqs:
            real = [prep(s) for s in seq]
            do.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake, _ = g(real)
            real_logit, fake_logit = d(real), d(fake)
            dl = (F.binary_cross_entropy_with_logits(real_logit,
                  torch.ones_like(real_logit)) +
                  F.binary_cross_entropy_with_logits(fake_logit,
                  torch.zeros_like(fake_logit)))
            dl.backward(); do.step()
            for p in d.parameters():
                p.requires_grad_(False)
            go.zero_grad(set_to_none=True)
            fake, recon = g(real)
            logit = d(fake)
            gl = (F.binary_cross_entropy_with_logits(logit,
                  torch.ones_like(logit)) + LAMBDA_RECON * recon)
            gl.backward(); go.step()
            for p in d.parameters():
                p.requires_grad_(True)
            sums += [gl.item(), dl.item(), recon.item()]
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == EPOCHS:
            a, b, c = sums / len(seqs)
            print(f'  Epoch {epoch + 1}/{EPOCHS}: 3G={a:.4f} '
                  f'AGD={b:.4f} reconstruction={c:.4f}')
    return g, d


def inject(sequence, normalizer):
    """Même groupe source si présent; campagne de votes négatifs par snapshot."""
    common = set(sequence[0].node_ids.tolist())
    for snapshot in sequence[1:]:
        common.intersection_update(snapshot.node_ids.tolist())
    pool = sorted(common)
    if len(pool) < 3:
        raise ValueError('Pas assez d’utilisateurs communs dans cette fenêtre.')
    # Écarter une cible du groupe de votants; les deux côtés sont marqués.
    n_source = min(len(pool) - 1,
                   max(1, round(NODE_ANOMALY_RATIO * min(len(s.x) for s in sequence))))
    picked = random.sample(pool, n_source + 1)
    source_ids, target_id = picked[:-1], picked[-1]
    mean, std = normalizer
    altered, marked = [], []
    for original in sequence:
        s = copy.copy(original)
        ids = original.node_ids.tolist()
        local = {node: i for i, node in enumerate(ids)}
        source = torch.tensor([local[v] for v in source_ids], dtype=torch.long)
        target = torch.full((n_source,), local[target_id], dtype=torch.long)
        extra = torch.stack((source, target), dim=0).repeat_interleave(
            VOTES_PER_SOURCE, dim=1)
        s.edge_index = torch.cat((original.edge_index, extra), dim=1)
        s.edge_vote = torch.cat((original.edge_vote,
                                 -torch.ones(extra.size(1), dtype=torch.int8)))
        raw = features(s.edge_index, s.edge_vote, len(ids))
        s.x = ((raw - mean) / std).clamp(-10, 10)
        altered.append(s)
        marked.append(torch.tensor([*source.tolist(), local[target_id]],
                                   dtype=torch.long))
    return altered, marked


def make_test(seqs, normalizer, seed):
    # Les fenêtres se chevauchent, mais chaque séquence est évaluée séparément.
    eligible = []
    for i, seq in enumerate(seqs):
        common = set(seq[0].node_ids.tolist())
        for s in seq[1:]:
            common.intersection_update(s.node_ids.tolist())
        if len(common) >= 3:
            eligible.append(i)
    rng = np.random.default_rng(seed + 1000)
    count = max(1, round(TEST_ANOMALY_RATIO * len(seqs)))
    if len(eligible) < count or count >= len(seqs):
        raise ValueError('Pas assez de fenêtres test injectables pour ce taux.')
    selected = set(rng.choice(eligible, size=count, replace=False).tolist())
    seed_all(seed + 2000)
    evaluation, labels, marked = [], [], []
    for i, seq in enumerate(seqs):
        altered, truth = inject(seq, normalizer) if i in selected else (seq, None)
        evaluation.append(altered)
        labels.append(int(i in selected))
        marked.append(truth)
    return evaluation, np.array(labels), marked


def score(g, seq):
    with torch.no_grad():
        return g([prep(s) for s in seq])[1].item()


def explain(d, seq, truth):
    """Attention incidente; mesure de localisation, pas preuve causale."""
    with torch.no_grad():
        _, weights = d([prep(s) for s in seq], attention=True)
    precision, hit = [], []
    for ((e1, a1), (e2, a2), n), targets in zip(weights, truth):
        importance = torch.zeros(n, device=DEVICE)
        for edges, alpha in ((e1, a1), (e2, a2)):
            layer = torch.zeros(n, device=DEVICE)
            vals = alpha.mean(dim=1)
            layer.index_add_(0, edges[0], vals)
            layer.index_add_(0, edges[1], vals)
            importance += layer / layer.sum().clamp_min(1e-8)
        k = len(targets)
        selected = set(importance.topk(k).indices.cpu().tolist())
        matches = len(selected.intersection(targets.tolist()))
        precision.append(matches / k)
        hit.append(float(matches > 0))
    return precision, hit


def evaluate(g, d, val_seqs, test_seqs, labels, marked):
    g.eval(); d.eval()
    val_scores = np.array([score(g, seq) for seq in val_seqs])
    threshold = float(val_scores.mean() + THRESHOLD_STD * val_scores.std())
    scores = np.array([score(g, seq) for seq in test_seqs])
    pred = (scores > threshold).astype(int)
    p, r, f, _ = precision_recall_fscore_support(
        labels, pred, average='binary', zero_division=0)
    cm = confusion_matrix(labels, pred, labels=[0, 1])
    result = dict(roc_auc=float(roc_auc_score(labels, scores)),
                  pr_auc=float(average_precision_score(labels, scores)),
                  precision=float(p), recall=float(r), f1=float(f),
                  accuracy=float(np.mean(labels == pred)),
                  cm=cm.tolist(), threshold=threshold,
                  clean_median=float(np.median(scores[labels == 0])),
                  injected_median=float(np.median(scores[labels == 1])))
    hits, precision_at_k = [], []
    for seq, label, truth in zip(test_seqs, labels, marked):
        if label:
            a, b = explain(d, seq, truth)
            precision_at_k.extend(a); hits.extend(b)
    result['precision_at_k'] = float(np.mean(precision_at_k))
    result['hit_rate_at_k'] = float(np.mean(hits))
    return result


def main():
    graphs = load_graphs(DATA_PATH)
    dev_end = int(0.7 * len(graphs))
    dev, test = graphs[:dev_end], graphs[dev_end:]
    val_size = max(SEQ_LEN, round(0.2 * len(dev)))
    train_graphs, val = dev[:-val_size], dev[-val_size:]
    all_train = windows(train_graphs)
    train_seqs = all_train[-max(1, round(TRAIN_FRACTION * len(all_train))):]
    val_seqs, test_seqs = windows(val), windows(test)
    if not train_seqs or not val_seqs or len(test_seqs) < 4:
        raise ValueError('Pas assez de fenêtres train/validation/test.')
    used = train_graphs[-(len(train_seqs) + SEQ_LEN - 1):]
    values = torch.cat([s.x for s in used], dim=0)
    mean, std = values.mean(0), values.std(0).clamp_min(1e-6)
    normalizer = (mean, std)
    for s in graphs:
        s.x = ((s.x - mean) / std).clamp(-10, 10)
    print(f'Device={DEVICE}; snapshots train/val/test='
          f'{len(train_graphs)}/{len(val)}/{len(test)}; '
          f'fenêtres entraînées={len(train_seqs)}/{len(all_train)}')
    # Une injection fixe pour comparer des graines d'entraînement distinctes.
    evaluation, labels, truth = make_test(test_seqs, normalizer, seed=2026)
    results = []
    for seed in SEEDS:
        seed_all(seed)
        start = time.perf_counter()
        g, d = train(train_seqs, graphs[0].x.size(1))
        metrics = evaluate(g, d, val_seqs, evaluation, labels, truth)
        metrics['seconds'] = round(time.perf_counter() - start, 1)
        results.append(metrics)
        print(f'Seed {seed}: {metrics}')
    if len(results) > 1:
        print('Wiki-RfA, moyennes ± écarts-types (anomalies injectées)')
        for key in ('roc_auc', 'pr_auc', 'precision', 'recall', 'f1',
                    'accuracy', 'precision_at_k', 'hit_rate_at_k'):
            v = np.array([m[key] for m in results])
            print(f'  {key}: {v.mean():.4f} ± {v.std(ddof=1):.4f}')


if __name__ == '__main__':
    main()
