import json
import tempfile
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)

from torch_geometric.loader import DataLoader

from generate_data import generate_random_graph, assign_attributes
from experiments.generate_balanced_data import (
    SIZE_BUCKETS,
    SCENARIOS,
    SAMPLES_PER_SCENARIO,
    apply_security_scenario,
)
from heuristics import calculate_labels
from dataset import NetworkDataset
from model import NetworkGNN


SEED = 4040
MODEL_PATH = "/tmp/linktelligence-best-model-v3.pt"

rng = np.random.default_rng(SEED)

temp_root = Path(
    tempfile.mkdtemp(prefix="linktelligence-v3-holdout-")
)
data_dir = temp_root / "data"
data_dir.mkdir(parents=True)

# ---------------------------------------------------------
# 1. Generate independent holdout dataset
# ---------------------------------------------------------

plan = []

for bucket, (min_nodes, max_nodes) in SIZE_BUCKETS.items():
    for issue_a, issue_b in SCENARIOS:
        for _ in range(SAMPLES_PER_SCENARIO):
            num_nodes = int(
                rng.integers(min_nodes, max_nodes + 1)
            )

            plan.append(
                (bucket, num_nodes, issue_a, issue_b)
            )

rng.shuffle(plan)

for i, (bucket, num_nodes, issue_a, issue_b) in enumerate(plan):
    graph = generate_random_graph(num_nodes, rng)
    nodes, edges = assign_attributes(graph, rng)

    apply_security_scenario(
        nodes,
        edges,
        rng,
        issue_a,
        issue_b,
    )

    labels = calculate_labels(nodes, edges)

    expected_security = int(issue_a) + int(issue_b)

    assert labels["security_flaw"] == expected_security

    sample = {
        "sample_id": i + 1,
        "nodes": nodes,
        "edges": edges,
        "metadata": labels,
    }

    (data_dir / f"sample_{i + 1:04d}.json").write_text(
        json.dumps(sample),
        encoding="utf-8",
    )

print(f"Generated {len(plan)} independent graphs.")
print(f"Dataset: {data_dir}")

# ---------------------------------------------------------
# 2. Load checkpoint and dataset
# ---------------------------------------------------------

dataset = NetworkDataset(root=str(temp_root))
model = NetworkGNN.from_pretrained(MODEL_PATH)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
)

buckets = {
    name: {
        "score_true": [],
        "score_pred": [],
        "bn_true": [],
        "bn_pred": [],
        "sec_true": [],
        "sec_pred": [],
    }
    for name in SIZE_BUCKETS
}

model.eval()

# ---------------------------------------------------------
# 3. Inference
# ---------------------------------------------------------

with torch.no_grad():
    for batch in loader:
        score, bottleneck, security = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
        )

        score_pred = torch.clamp(
            score.squeeze(-1), 0.0, 1.0
        ).cpu().tolist()

        bn_pred = bottleneck.argmax(1).cpu().tolist()
        sec_pred = security.argmax(1).cpu().tolist()

        score_true = batch.y[:, 0].cpu().tolist()
        bn_true = batch.y[:, 1].long().cpu().tolist()
        sec_true = batch.y[:, 2].long().cpu().tolist()

        node_counts = (
            batch.ptr[1:] - batch.ptr[:-1]
        ).cpu().tolist()

        for i, num_nodes in enumerate(node_counts):
            bucket = next(
                name
                for name, (lo, hi) in SIZE_BUCKETS.items()
                if lo <= num_nodes <= hi
            )

            result = buckets[bucket]

            result["score_true"].append(score_true[i])
            result["score_pred"].append(score_pred[i])
            result["bn_true"].append(bn_true[i])
            result["bn_pred"].append(bn_pred[i])
            result["sec_true"].append(sec_true[i])
            result["sec_pred"].append(sec_pred[i])

# ---------------------------------------------------------
# 4. Evaluate each size bucket
# ---------------------------------------------------------

print("\n=== INDEPENDENT V3 HOLDOUT ===")

for name, result in buckets.items():
    n = len(result["score_true"])

    if not n:
        print(f"\n{name}: no samples")
        continue

    print(f"\n--- {name} nodes | {n} graphs ---")

    print(
        "Score MAE:",
        round(mean_absolute_error(
            result["score_true"],
            result["score_pred"],
        ), 4),
    )

    print(
        "Score RMSE:",
        round(mean_squared_error(
            result["score_true"],
            result["score_pred"],
        ) ** 0.5, 4),
    )

    print(
        "Bottleneck balanced accuracy:",
        round(balanced_accuracy_score(
            result["bn_true"],
            result["bn_pred"],
        ), 4),
    )

    print(
        "Bottleneck macro F1:",
        round(f1_score(
            result["bn_true"],
            result["bn_pred"],
            labels=[0, 1],
            average="macro",
            zero_division=0,
        ), 4),
    )

    print(
        "Security balanced accuracy:",
        round(balanced_accuracy_score(
            result["sec_true"],
            result["sec_pred"],
        ), 4),
    )

    print(
        "Security macro F1:",
        round(f1_score(
            result["sec_true"],
            result["sec_pred"],
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        ), 4),
    )

print("\nIndependent v3 holdout complete.")
print("Dataset:", temp_root)