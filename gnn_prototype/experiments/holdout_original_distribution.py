import json
import tempfile
from pathlib import Path
from collections import Counter

import numpy as np
import torch

from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)

from torch_geometric.loader import DataLoader

from generate_data import generate_random_graph, assign_attributes
from heuristics import calculate_labels
from dataset import NetworkDataset
from model import NetworkGNN
from train import _stratified_split


# ---------------------------------------------------------
# 1. Generate independent synthetic dataset
# ---------------------------------------------------------

SEED = 2027
NUM_SAMPLES = 2000

rng = np.random.default_rng(SEED)

temp_root = Path(tempfile.mkdtemp(prefix="linktelligence-holdout-"))
data_dir = temp_root / "data"
data_dir.mkdir(parents=True)

print(f"Generating independent dataset: seed={SEED}")

for i in range(NUM_SAMPLES):
    num_nodes = int(rng.integers(5, 20))

    graph = generate_random_graph(num_nodes, rng)
    nodes, edges = assign_attributes(graph, rng)
    labels = calculate_labels(nodes, edges)

    sample = {
        "sample_id": i + 1,
        "nodes": nodes,
        "edges": edges,
        "metadata": labels,
    }

    path = data_dir / f"sample_{i + 1:04d}.json"
    path.write_text(
        json.dumps(sample, indent=2),
        encoding="utf-8",
    )

print(f"Generated {NUM_SAMPLES} independent graphs.")


# ---------------------------------------------------------
# 2. Load original training split for baseline calculation
# ---------------------------------------------------------

original_dataset = NetworkDataset(root=".")

train_set, _, _ = _stratified_split(
    original_dataset,
    0.70,
    0.15,
    0.15,
    seed=42,
)

train_scores = [
    float(train_set[i].y[0, 0])
    for i in range(len(train_set))
]

train_bottleneck = [
    int(train_set[i].y[0, 1])
    for i in range(len(train_set))
]

train_security = [
    int(train_set[i].y[0, 2])
    for i in range(len(train_set))
]

score_median = float(np.median(train_scores))
score_mean = float(np.mean(train_scores))

bn_majority = Counter(train_bottleneck).most_common(1)[0][0]
sec_majority = Counter(train_security).most_common(1)[0][0]


# ---------------------------------------------------------
# 3. Load independent dataset and model
# ---------------------------------------------------------

holdout_dataset = NetworkDataset(root=str(temp_root))

model_path = Path("/tmp/linktelligence-best-model-v2.pt")

if not model_path.is_file():
    raise FileNotFoundError(model_path)

model = NetworkGNN.from_pretrained(str(model_path))

loader = DataLoader(
    holdout_dataset,
    batch_size=32,
    shuffle=False,
)


# ---------------------------------------------------------
# 4. Run inference
# ---------------------------------------------------------

score_true, score_pred = [], []
bn_true, bn_pred = [], []
sec_true, sec_pred = [], []

model.eval()

with torch.no_grad():
    for batch in loader:
        score, bottleneck, security = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
        )

        score_pred.extend(
            torch.clamp(score.squeeze(-1), 0.0, 1.0).cpu().tolist()
        )
        score_true.extend(batch.y[:, 0].cpu().tolist())

        bn_pred.extend(bottleneck.argmax(1).cpu().tolist())
        bn_true.extend(batch.y[:, 1].long().cpu().tolist())

        sec_pred.extend(security.argmax(1).cpu().tolist())
        sec_true.extend(batch.y[:, 2].long().cpu().tolist())


# ---------------------------------------------------------
# 5. Evaluate regression
# ---------------------------------------------------------

print("\n=== SCORE REGRESSION ===")

print(
    "Model MAE:",
    round(mean_absolute_error(score_true, score_pred), 4),
)

print(
    "Model RMSE:",
    round(mean_squared_error(score_true, score_pred) ** 0.5, 4),
)

print(
    "Median baseline MAE:",
    round(
        mean_absolute_error(
            score_true,
            [score_median] * len(score_true),
        ),
        4,
    ),
)

print(
    "Mean baseline RMSE:",
    round(
        mean_squared_error(
            score_true,
            [score_mean] * len(score_true),
        ) ** 0.5,
        4,
    ),
)


# ---------------------------------------------------------
# 6. Evaluate classification
# ---------------------------------------------------------

print("\n=== BOTTLENECK ===")

print("Balanced accuracy:", balanced_accuracy_score(bn_true, bn_pred))
print("Macro F1:", f1_score(bn_true, bn_pred, average="macro"))
print(classification_report(bn_true, bn_pred, labels=[0, 1], zero_division=0))

print("Majority baseline F1:", f1_score(
    bn_true,
    [bn_majority] * len(bn_true),
    labels=[0, 1],
    average="macro",
    zero_division=0,
))


print("\n=== SECURITY ===")

print("Balanced accuracy:", balanced_accuracy_score(sec_true, sec_pred))
print("Macro F1:", f1_score(sec_true, sec_pred, average="macro"))
print(classification_report(
    sec_true,
    sec_pred,
    labels=[0, 1, 2],
    zero_division=0,
))

print("Majority baseline F1:", f1_score(
    sec_true,
    [sec_majority] * len(sec_true),
    labels=[0, 1, 2],
    average="macro",
    zero_division=0,
))


# ---------------------------------------------------------
# 7. Dataset distribution
# ---------------------------------------------------------

print("\n=== HOLDOUT DISTRIBUTION ===")
print("Bottleneck:", dict(sorted(Counter(bn_true).items())))
print("Security:", dict(sorted(Counter(sec_true).items())))
print("Joint:", dict(sorted(Counter(zip(bn_true, sec_true)).items())))

print("\nIndependent holdout evaluation complete.")
print("Dataset location:", temp_root)