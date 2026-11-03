# -*- coding: utf-8 -*-

"""
====================================================================
STG-GAN - REDDIT HYPERLINKS BODY
Dataset-aware version
====================================================================

Graph:
    Node = subreddit
    Directed edge = hyperlink SOURCE -> TARGET
    Time = TIMESTAMP

Node features:
    Aggregation of REAL interaction attributes:
        LINK_SENTIMENT
        + first 20 PROPERTIES

For each snapshot:
    node feature = mean attributes of incident interactions

Protocol:
    70% development / 30% test
    80% of development train / 20% validation

Architecture:
    3G  = GConvGRU + Linear decoder
    AGD = GAT + GAT + Transformer

Anomaly injection:
    30% anomalous sequences
    3% ACTIVE nodes
    attribute + structural perturbation

Threshold:
    selected ONLY on contaminated validation set
    using validation F1

Test:
    independent corruption seed
====================================================================
"""

import os
import copy
import random
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam

from torch_geometric.data import Data
from torch_geometric.nn import (
    GATConv,
    global_mean_pool
)

from torch_geometric_temporal.nn.recurrent import (
    GConvGRU
)

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix
)


# ==============================================================
# 1. CONFIGURATION
# ==============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", DEVICE)


DATA_PATH = (
    r"C:\Users\HP\mon-projet\dataset"
    r"\soc-redditHyperlinks-body.tsv"
)

OUTPUT_DIR = (
    "reddit_stggan_properties_final"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# --------------------------------------------------------------
# Dataset
# --------------------------------------------------------------

MAX_EDGES = 10000

NUM_SNAPSHOTS = 100

# Keep computation reasonable on CPU
NUM_PROPERTIES = 20


# --------------------------------------------------------------
# STG-GAN
# --------------------------------------------------------------

MODEL_SEED = 42

SEQ_LEN = 4

EPOCHS = 30

HIDDEN = 32

LR_G = 1e-3
LR_D = 1e-3

LAMBDA_RECON = 5.0

EPS = 1e-8


# --------------------------------------------------------------
# Anomaly generation
# --------------------------------------------------------------

VALIDATION_ANOMALY_RATIO = 0.30

TEST_ANOMALY_RATIO = 0.30

NODE_ANOMALY_RATIO = 0.03

FEATURE_SCALE = 1.50

NOISE_STD = 0.50

CONNECT_PROB = 0.20


# Independent seeds
VALIDATION_CORRUPTION_SEED = 1026

TEST_CORRUPTION_SEED = 2026


# ==============================================================
# 2. SEEDS
# ==============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# ==============================================================
# 3. LOAD THE REAL REDDIT ATTRIBUTES
# ==============================================================

print(
    "\nLoading Reddit Hyperlinks..."
)


df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    usecols=[
        "SOURCE_SUBREDDIT",
        "TARGET_SUBREDDIT",
        "TIMESTAMP",
        "LINK_SENTIMENT",
        "PROPERTIES"
    ]
)


df["TIMESTAMP"] = pd.to_datetime(
    df["TIMESTAMP"],
    errors="coerce"
)


df = df.dropna(
    subset=[
        "SOURCE_SUBREDDIT",
        "TARGET_SUBREDDIT",
        "TIMESTAMP",
        "LINK_SENTIMENT",
        "PROPERTIES"
    ]
)


df = df.sort_values(
    "TIMESTAMP"
).reset_index(
    drop=True
)


print(
    "Full interactions:",
    len(df)
)

print(
    "Temporal range:",
    df["TIMESTAMP"].min(),
    "->",
    df["TIMESTAMP"].max()
)


# ==============================================================
# 4. PARSE PROPERTIES
# ==============================================================

def parse_properties(value):

    try:

        values = np.asarray(
            [
                float(x)
                for x
                in str(value).split(",")
            ],
            dtype=np.float32
        )

    except Exception:

        values = np.zeros(
            NUM_PROPERTIES,
            dtype=np.float32
        )


    # ----------------------------------------------------------
    # Ensure fixed dimensionality
    # ----------------------------------------------------------

    if len(values) >= NUM_PROPERTIES:

        values = values[
            :NUM_PROPERTIES
        ]

    else:

        padded = np.zeros(
            NUM_PROPERTIES,
            dtype=np.float32
        )

        padded[
            :len(values)
        ] = values

        values = padded


    return values


print(
    "\nParsing PROPERTIES..."
)


property_matrix = np.stack(
    df["PROPERTIES"]
    .apply(
        parse_properties
    )
    .to_numpy()
)


print(
    "Property matrix:",
    property_matrix.shape
)


# ==============================================================
# 5. TEMPORALLY DISTRIBUTED SAMPLING
# ==============================================================

if len(df) > MAX_EDGES:

    selected_indices = np.linspace(
        0,
        len(df) - 1,
        MAX_EDGES,
        dtype=int
    )


    df = (
        df.iloc[
            selected_indices
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    property_matrix = (
        property_matrix[
            selected_indices
        ]
    )


print(
    "\nRetained interactions:",
    len(df)
)

print(
    "Retained property matrix:",
    property_matrix.shape
)


# ==============================================================
# 6. CREATE EDGE ATTRIBUTE MATRIX
# ==============================================================

sentiment = (
    df["LINK_SENTIMENT"]
    .astype(
        np.float32
    )
    .to_numpy()
    .reshape(
        -1,
        1
    )
)


# sentiment + 20 properties
edge_attributes = np.concatenate(
    [
        sentiment,
        property_matrix
    ],
    axis=1
).astype(
    np.float32
)


FEATURE_DIM = (
    edge_attributes.shape[1]
)


print(
    "Edge feature dimension:",
    FEATURE_DIM
)


# ==============================================================
# 7. GLOBAL NODE MAPPING
# ==============================================================

all_nodes = pd.concat(
    [
        df[
            "SOURCE_SUBREDDIT"
        ],

        df[
            "TARGET_SUBREDDIT"
        ]
    ],
    ignore_index=True
).unique()


node_to_id = {
    node: i
    for i, node
    in enumerate(
        all_nodes
    )
}


df["src"] = (
    df[
        "SOURCE_SUBREDDIT"
    ]
    .map(
        node_to_id
    )
    .astype(int)
)


df["dst"] = (
    df[
        "TARGET_SUBREDDIT"
    ]
    .map(
        node_to_id
    )
    .astype(int)
)


NUMBER_NODES = len(
    node_to_id
)


print(
    "Global nodes:",
    NUMBER_NODES
)


# ==============================================================
# 8. BUILD TEMPORAL SNAPSHOTS
# ==============================================================

start_time_data = (
    df["TIMESTAMP"].min()
)

end_time_data = (
    df["TIMESTAMP"].max()
)


time_boundaries = pd.date_range(
    start=start_time_data,
    end=end_time_data,
    periods=NUM_SNAPSHOTS + 1
)


snapshots = []


print(
    "\nBuilding snapshots..."
)


for t in range(
    NUM_SNAPSHOTS
):

    if t < NUM_SNAPSHOTS - 1:

        mask = (
            (
                df["TIMESTAMP"]
                >=
                time_boundaries[t]
            )
            &
            (
                df["TIMESTAMP"]
                <
                time_boundaries[t + 1]
            )
        )

    else:

        mask = (
            (
                df["TIMESTAMP"]
                >=
                time_boundaries[t]
            )
            &
            (
                df["TIMESTAMP"]
                <=
                time_boundaries[t + 1]
            )
        )


    row_indices = np.where(
        mask.to_numpy()
    )[0]


    # ----------------------------------------------------------
    # Empty snapshot
    # ----------------------------------------------------------

    if len(row_indices) == 0:

        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long
        )


        x = np.zeros(
            (
                NUMBER_NODES,
                FEATURE_DIM
            ),
            dtype=np.float32
        )


        snapshots.append(
            Data(
                x=torch.tensor(
                    x,
                    dtype=torch.float32
                ),
                edge_index=edge_index
            )
        )


        continue


    current_df = df.iloc[
        row_indices
    ]


    src = current_df[
        "src"
    ].to_numpy(
        dtype=np.int64
    )


    dst = current_df[
        "dst"
    ].to_numpy(
        dtype=np.int64
    )


    current_attributes = (
        edge_attributes[
            row_indices
        ]
    )


    edge_index = torch.tensor(
        np.vstack(
            [
                src,
                dst
            ]
        ),
        dtype=torch.long
    )


    # ==========================================================
    # EDGE ATTRIBUTES -> NODE ATTRIBUTES
    # ==========================================================

    node_feature_sum = np.zeros(
        (
            NUMBER_NODES,
            FEATURE_DIM
        ),
        dtype=np.float32
    )


    node_interaction_count = np.zeros(
        NUMBER_NODES,
        dtype=np.float32
    )


    # Source node receives interaction attributes
    np.add.at(
        node_feature_sum,
        src,
        current_attributes
    )


    np.add.at(
        node_interaction_count,
        src,
        1
    )


    # Target node also receives interaction attributes
    np.add.at(
        node_feature_sum,
        dst,
        current_attributes
    )


    np.add.at(
        node_interaction_count,
        dst,
        1
    )


    # Avoid division by zero
    denominator = np.maximum(
        node_interaction_count,
        1.0
    ).reshape(
        -1,
        1
    )


    x = (
        node_feature_sum
        /
        denominator
    )


    x[
        node_interaction_count == 0
    ] = 0.0


    snapshots.append(
        Data(
            x=torch.tensor(
                x,
                dtype=torch.float32
            ),
            edge_index=edge_index
        )
    )


    active_nodes = np.unique(
        np.concatenate(
            [
                src,
                dst
            ]
        )
    )


    print(
        f"Snapshot {t:03d} | "
        f"edges={len(src):4d} | "
        f"active={len(active_nodes):4d}"
    )


print(
    "\nSnapshots:",
    len(snapshots)
)


# ==============================================================
# 9. CHRONOLOGICAL SPLIT
# ==============================================================

development_end = int(
    0.70
    *
    len(snapshots)
)


development = snapshots[
    :development_end
]


test_snaps = snapshots[
    development_end:
]


validation_size = max(
    SEQ_LEN,
    int(
        round(
            0.20
            *
            len(development)
        )
    )
)


train_snaps = development[
    :-validation_size
]


val_snaps = development[
    -validation_size:
]


print(
    "\nTrain snapshots:",
    len(train_snaps)
)

print(
    "Validation snapshots:",
    len(val_snaps)
)

print(
    "Test snapshots:",
    len(test_snaps)
)


# ==============================================================
# 10. NORMALIZATION
# ==============================================================

# IMPORTANT:
# estimate normalization ONLY from active training nodes.

active_training_features = []


for snapshot in train_snaps:

    if snapshot.edge_index.numel() == 0:

        continue


    active = torch.unique(
        snapshot.edge_index.flatten()
    )


    active_training_features.append(
        snapshot.x[
            active
        ]
    )


training_matrix = torch.cat(
    active_training_features,
    dim=0
)


FEATURE_MEAN = (
    training_matrix.mean(
        dim=0
    )
)


FEATURE_STD = (
    training_matrix.std(
        dim=0
    )
)


FEATURE_STD = torch.where(
    FEATURE_STD < EPS,
    torch.ones_like(
        FEATURE_STD
    ),
    FEATURE_STD
)


print(
    "\nFeature dimension:",
    FEATURE_DIM
)

print(
    "Train feature mean shape:",
    FEATURE_MEAN.shape
)


def normalize_snapshots(
    snapshot_list
):

    output = []


    for snapshot in snapshot_list:

        new_snapshot = copy.deepcopy(
            snapshot
        )


        # ------------------------------------------------------
        # Keep inactive nodes at zero.
        # Normalize ACTIVE nodes only.
        # ------------------------------------------------------

        if (
            new_snapshot.edge_index.numel()
            >
            0
        ):

            active = torch.unique(
                new_snapshot
                .edge_index
                .flatten()
            )


            new_snapshot.x[
                active
            ] = (
                new_snapshot.x[
                    active
                ]
                -
                FEATURE_MEAN
            ) / FEATURE_STD


        output.append(
            new_snapshot
        )


    return output


train_snaps = normalize_snapshots(
    train_snaps
)

val_snaps = normalize_snapshots(
    val_snaps
)

test_snaps = normalize_snapshots(
    test_snaps
)


# ==============================================================
# 11. SEQUENCES
# ==============================================================

def make_sequences(
    snapshot_list
):

    return [
        snapshot_list[
            i:i + SEQ_LEN
        ]

        for i in range(
            len(snapshot_list)
            -
            SEQ_LEN
            +
            1
        )
    ]


train_seqs = make_sequences(
    train_snaps
)

val_seqs = make_sequences(
    val_snaps
)

test_seqs = make_sequences(
    test_snaps
)


print(
    "\nTrain sequences:",
    len(train_seqs)
)

print(
    "Validation sequences:",
    len(val_seqs)
)

print(
    "Test sequences:",
    len(test_seqs)
)


# ==============================================================
# 12. GENERATOR 3G
# ==============================================================

class Generator3G(
    nn.Module
):

    def __init__(
        self,
        input_dim
    ):

        super().__init__()


        self.rnn = GConvGRU(
            input_dim,
            HIDDEN,
            K=2
        )


        self.decoder = nn.Linear(
            HIDDEN,
            input_dim
        )


    def forward(
        self,
        x,
        edge_index,
        hidden=None
    ):

        hidden = self.rnn(
            x,
            edge_index,
            None,
            hidden
        )


        reconstructed = (
            self.decoder(
                hidden
            )
        )


        return (
            reconstructed,
            hidden
        )


# ==============================================================
# 13. AGD
# ==============================================================

class AGD(
    nn.Module
):

    def __init__(
        self,
        input_dim
    ):

        super().__init__()


        self.gat1 = GATConv(
            input_dim,
            16,
            heads=2,
            concat=True
        )


        self.gat2 = GATConv(
            32,
            32,
            heads=1,
            concat=True
        )


        layer = (
            nn.TransformerEncoderLayer(
                d_model=32,
                nhead=2,
                batch_first=True
            )
        )


        self.transformer = (
            nn.TransformerEncoder(
                layer,
                num_layers=1
            )
        )


        self.classifier = nn.Linear(
            32,
            1
        )


    def forward(
        self,
        sequence
    ):

        graph_embeddings = []


        for (
            x,
            edge_index,
            batch
        ) in sequence:

            h = F.elu(
                self.gat1(
                    x,
                    edge_index
                )
            )


            h = F.elu(
                self.gat2(
                    h,
                    edge_index
                )
            )


            graph_embedding = (
                global_mean_pool(
                    h,
                    batch
                )
            )


            graph_embeddings.append(
                graph_embedding
            )


        temporal = torch.stack(
            graph_embeddings,
            dim=1
        )


        temporal = (
            self.transformer(
                temporal
            )
        )


        representation = (
            temporal.mean(
                dim=1
            )
        )


        return (
            self.classifier(
                representation
            ).view(-1)
        )


# ==============================================================
# 14. PREP
# ==============================================================

def prep(snapshot):

    x = snapshot.x.to(
        DEVICE
    )


    edge_index = (
        snapshot
        .edge_index
        .to(
            DEVICE
        )
    )


    batch = torch.zeros(
        x.shape[0],
        dtype=torch.long,
        device=DEVICE
    )


    return (
        x,
        edge_index,
        batch
    )


# ==============================================================
# 15. ANOMALY INJECTION
# ==============================================================

def inject_anomalies(
    sequence
):

    contaminated = []


    for snapshot in sequence:

        new_snapshot = copy.deepcopy(
            snapshot
        )


        x = new_snapshot.x.clone()


        edge_index = (
            new_snapshot
            .edge_index
            .clone()
        )


        if edge_index.numel() == 0:

            contaminated.append(
                new_snapshot
            )

            continue


        # ------------------------------------------------------
        # ACTIVE nodes
        # ------------------------------------------------------

        active_nodes = torch.unique(
            edge_index.flatten()
        )


        n_active = (
            active_nodes.numel()
        )


        n_anomalous = max(
            1,
            int(
                round(
                    NODE_ANOMALY_RATIO
                    *
                    n_active
                )
            )
        )


        n_anomalous = min(
            n_anomalous,
            n_active
        )


        positions = torch.randperm(
            n_active
        )[
            :n_anomalous
        ]


        anomaly_nodes = (
            active_nodes[
                positions
            ]
        )


        # ------------------------------------------------------
        # ATTRIBUTE PERTURBATION
        # ------------------------------------------------------

        noise = (
            torch.randn(
                n_anomalous,
                FEATURE_DIM
            )
            *
            NOISE_STD
        )


        x[
            anomaly_nodes
        ] = (
            FEATURE_SCALE
            *
            x[
                anomaly_nodes
            ]
            +
            noise
        )


        # ------------------------------------------------------
        # STRUCTURAL PERTURBATION
        # ------------------------------------------------------

        if n_anomalous > 1:

            src = (
                anomaly_nodes
                .repeat_interleave(
                    n_anomalous
                )
            )


            dst = (
                anomaly_nodes
                .repeat(
                    n_anomalous
                )
            )


            valid = (
                src != dst
            )


            src = src[
                valid
            ]

            dst = dst[
                valid
            ]


            keep = (
                torch.rand(
                    src.numel()
                )
                <
                CONNECT_PROB
            )


            src = src[
                keep
            ]

            dst = dst[
                keep
            ]


            if src.numel() > 0:

                extra_edges = (
                    torch.stack(
                        [
                            src,
                            dst
                        ]
                    )
                )


                edge_index = (
                    torch.cat(
                        [
                            edge_index,
                            extra_edges
                        ],
                        dim=1
                    )
                )


        new_snapshot.x = x

        new_snapshot.edge_index = (
            edge_index
        )


        contaminated.append(
            new_snapshot
        )


    return contaminated


# ==============================================================
# 16. CREATE LABELED SET
# ==============================================================

def create_evaluation_set(
    sequences,
    anomaly_ratio,
    seed
):

    n = len(
        sequences
    )


    number_anomalies = max(
        1,
        int(
            round(
                anomaly_ratio
                *
                n
            )
        )
    )


    rng = (
        np.random.default_rng(
            seed
        )
    )


    anomaly_indices = set(
        rng.choice(
            n,
            number_anomalies,
            replace=False
        ).tolist()
    )


    set_seed(
        seed + 1000
    )


    output = []

    labels = []


    for i, sequence in enumerate(
        sequences
    ):

        if i in anomaly_indices:

            output.append(
                inject_anomalies(
                    sequence
                )
            )

            labels.append(
                1
            )

        else:

            output.append(
                copy.deepcopy(
                    sequence
                )
            )

            labels.append(
                0
            )


    return (
        output,
        np.asarray(
            labels,
            dtype=int
        ),
        anomaly_indices
    )


(
    validation_eval,
    validation_labels,
    validation_anomaly_indices
) = create_evaluation_set(
    val_seqs,
    VALIDATION_ANOMALY_RATIO,
    VALIDATION_CORRUPTION_SEED
)


(
    test_eval,
    test_labels,
    test_anomaly_indices
) = create_evaluation_set(
    test_seqs,
    TEST_ANOMALY_RATIO,
    TEST_CORRUPTION_SEED
)


print(
    "\nValidation anomaly indices:",
    sorted(
        validation_anomaly_indices
    )
)

print(
    "Test anomaly indices:",
    sorted(
        test_anomaly_indices
    )
)


# ==============================================================
# 17. TRAIN
# ==============================================================

def train_stggan():

    set_seed(
        MODEL_SEED
    )


    G = Generator3G(
        FEATURE_DIM
    ).to(
        DEVICE
    )


    D = AGD(
        FEATURE_DIM
    ).to(
        DEVICE
    )


    optimizer_G = Adam(
        G.parameters(),
        lr=LR_G
    )


    optimizer_D = Adam(
        D.parameters(),
        lr=LR_D
    )


    BCE = (
        nn.BCEWithLogitsLoss()
    )


    MSE = nn.MSELoss()


    history = []


    for epoch in range(
        1,
        EPOCHS + 1
    ):

        G.train()

        D.train()


        total_D = 0.0
        total_G = 0.0
        total_R = 0.0


        for sequence in train_seqs:

            prepared = [
                prep(snapshot)
                for snapshot
                in sequence
            ]


            # ==================================================
            # TRAIN D
            # ==================================================

            optimizer_D.zero_grad()


            real_logits = D(
                prepared
            )


            loss_real = BCE(
                real_logits,
                torch.ones_like(
                    real_logits
                )
            )


            fake_sequence = []

            hidden = None


            with torch.no_grad():

                for (
                    x,
                    edge_index,
                    batch
                ) in prepared:

                    (
                        x_hat,
                        hidden
                    ) = G(
                        x,
                        edge_index,
                        hidden
                    )


                    fake_sequence.append(
                        (
                            x_hat,
                            edge_index,
                            batch
                        )
                    )


            fake_logits = D(
                fake_sequence
            )


            loss_fake = BCE(
                fake_logits,
                torch.zeros_like(
                    fake_logits
                )
            )


            D_loss = (
                loss_real
                +
                loss_fake
            )


            D_loss.backward()


            optimizer_D.step()


            # ==================================================
            # TRAIN G
            # ==================================================

            optimizer_G.zero_grad()


            generated = []

            reconstruction_terms = []

            hidden = None


            for (
                x,
                edge_index,
                batch
            ) in prepared:

                (
                    x_hat,
                    hidden
                ) = G(
                    x,
                    edge_index,
                    hidden
                )


                reconstruction_terms.append(
                    MSE(
                        x_hat,
                        x
                    )
                )


                generated.append(
                    (
                        x_hat,
                        edge_index,
                        batch
                    )
                )


            reconstruction_loss = (
                torch.stack(
                    reconstruction_terms
                ).mean()
            )


            adversarial_loss = BCE(
                D(
                    generated
                ),
                torch.ones(
                    1,
                    device=DEVICE
                )
            )


            G_loss = (
                adversarial_loss
                +
                LAMBDA_RECON
                *
                reconstruction_loss
            )


            G_loss.backward()


            optimizer_G.step()


            total_D += (
                D_loss.item()
            )

            total_G += (
                G_loss.item()
            )

            total_R += (
                reconstruction_loss.item()
            )


        n = len(
            train_seqs
        )


        history.append(
            [
                epoch,
                total_D / n,
                total_G / n,
                total_R / n
            ]
        )


        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"D={total_D/n:.4f} | "
            f"G={total_G/n:.4f} | "
            f"Recon={total_R/n:.4f}"
        )


    return (
        G,
        D,
        pd.DataFrame(
            history,
            columns=[
                "Epoch",
                "D_Loss",
                "G_Loss",
                "Recon_Loss"
            ]
        )
    )


# ==============================================================
# 18. RECONSTRUCTION SCORE
# ==============================================================

def reconstruction_score(
    sequence,
    G
):

    hidden = None

    errors = []


    for snapshot in sequence:

        (
            x,
            edge_index,
            _
        ) = prep(
            snapshot
        )


        (
            x_hat,
            hidden
        ) = G(
            x,
            edge_index,
            hidden
        )


        # ------------------------------------------------------
        # IMPORTANT:
        # score ACTIVE nodes only.
        #
        # Otherwise thousands of inactive zero nodes dominate
        # the Reddit reconstruction score.
        # ------------------------------------------------------

        active_nodes = torch.unique(
            edge_index.flatten()
        )


        if active_nodes.numel() == 0:

            continue


        error = (
            (
                x_hat[
                    active_nodes
                ]
                -
                x[
                    active_nodes
                ]
            )
            ** 2
        ).mean()


        errors.append(
            error
        )


    if len(errors) == 0:

        return 0.0


    return (
        torch.stack(
            errors
        )
        .mean()
        .item()
    )


# ==============================================================
# 19. SCORE DATA
# ==============================================================

def get_scores(
    sequences,
    G
):

    G.eval()


    scores = []


    with torch.no_grad():

        for sequence in sequences:

            scores.append(
                reconstruction_score(
                    sequence,
                    G
                )
            )


    return np.asarray(
        scores,
        dtype=float
    )


# ==============================================================
# 20. THRESHOLD FROM VALIDATION ONLY
# ==============================================================

def select_threshold(
    labels,
    scores
):

    unique_scores = np.sort(
        np.unique(
            scores
        )
    )


    if len(unique_scores) > 1:

        candidates = (
            unique_scores[:-1]
            +
            unique_scores[1:]
        ) / 2.0

    else:

        candidates = (
            unique_scores
        )


    best_threshold = None

    best_f1 = -1.0

    best_precision = -1.0


    for threshold in candidates:

        predictions = (
            scores
            >
            threshold
        ).astype(int)


        current_f1 = f1_score(
            labels,
            predictions,
            zero_division=0
        )


        current_precision = (
            precision_score(
                labels,
                predictions,
                zero_division=0
            )
        )


        if (
            current_f1 > best_f1
            or
            (
                np.isclose(
                    current_f1,
                    best_f1
                )
                and
                current_precision
                >
                best_precision
            )
        ):

            best_f1 = (
                current_f1
            )

            best_precision = (
                current_precision
            )

            best_threshold = (
                threshold
            )


    return (
        float(
            best_threshold
        ),
        float(
            best_f1
        )
    )


# ==============================================================
# 21. RUN
# ==============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "TRAINING STG-GAN"
)

print(
    "=" * 70
)


start = time.time()


G, D, history = (
    train_stggan()
)


runtime = (
    time.time()
    -
    start
)


# ==============================================================
# 22. VALIDATION
# ==============================================================

validation_scores = get_scores(
    validation_eval,
    G
)


validation_roc = roc_auc_score(
    validation_labels,
    validation_scores
)


validation_pr = (
    average_precision_score(
        validation_labels,
        validation_scores
    )
)


(
    threshold,
    validation_f1
) = select_threshold(
    validation_labels,
    validation_scores
)


print(
    "\n"
    + "=" * 70
)

print(
    "VALIDATION"
)

print(
    "=" * 70
)


print(
    f"ROC-AUC   : "
    f"{validation_roc:.4f}"
)

print(
    f"PR-AUC    : "
    f"{validation_pr:.4f}"
)

print(
    f"Best F1   : "
    f"{validation_f1:.4f}"
)

print(
    f"Threshold : "
    f"{threshold:.6f}"
)


# ==============================================================
# 23. TEST
# ==============================================================

test_scores = get_scores(
    test_eval,
    G
)


predictions = (
    test_scores
    >
    threshold
).astype(int)


roc = roc_auc_score(
    test_labels,
    test_scores
)


pr = average_precision_score(
    test_labels,
    test_scores
)


precision = precision_score(
    test_labels,
    predictions,
    zero_division=0
)


recall = recall_score(
    test_labels,
    predictions,
    zero_division=0
)


f1 = f1_score(
    test_labels,
    predictions,
    zero_division=0
)


accuracy = accuracy_score(
    test_labels,
    predictions
)


cm = confusion_matrix(
    test_labels,
    predictions,
    labels=[
        0,
        1
    ]
)


# ==============================================================
# 24. FINAL RESULTS
# ==============================================================

print(
    "\n"
    + "#" * 75
)

print(
    "FINAL REDDIT RESULTS"
)

print(
    "#" * 75
)


print(
    f"ROC-AUC   : {roc:.4f}"
)

print(
    f"PR-AUC    : {pr:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1        : {f1:.4f}"
)

print(
    f"Accuracy  : {accuracy:.4f}"
)


print(
    "\nConfusion Matrix:"
)

print(
    cm
)


print(
    f"\nValidation-selected "
    f"threshold: {threshold:.6f}"
)


print(
    f"\nRuntime: {runtime:.2f} sec"
)


# ==============================================================
# 25. DETAILED SCORES
# ==============================================================

results = pd.DataFrame(
    {
        "Sequence":
            np.arange(
                len(
                    test_labels
                )
            ),

        "Label":
            test_labels,

        "Score":
            test_scores,

        "Prediction":
            predictions
    }
)


print(
    "\nTEST SCORES"
)


print(
    results
    .round(5)
    .to_string(
        index=False
    )
)


# ==============================================================
# 26. SAVE
# ==============================================================

results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "test_results.csv"
    ),
    index=False
)


history.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "training_history.csv"
    ),
    index=False
)


summary = pd.DataFrame(
    [
        {
            "Seed":
                MODEL_SEED,

            "Input_Features":
                FEATURE_DIM,

            "Validation_ROC":
                validation_roc,

            "Validation_PR":
                validation_pr,

            "Validation_F1":
                validation_f1,

            "Threshold":
                threshold,

            "Test_ROC":
                roc,

            "Test_PR":
                pr,

            "Precision":
                precision,

            "Recall":
                recall,

            "F1":
                f1,

            "Accuracy":
                accuracy,

            "Runtime_sec":
                runtime
        }
    ]
)


summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "summary.csv"
    ),
    index=False
)


print(
    "\nResults saved in:",
    OUTPUT_DIR
)