import json
import tempfile
from pathlib import Path

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
from heuristics import calculate_labels
from dataset import NetworkDataset
from model import NetworkGNN


SEED = 2028
NUM_SAMPLES = 2000
MODEL_PATH = "/tmp/linktelligence-best-model-v2.pt"

rng = np.random.default_rng(SEED)

temp_root = Path(tempfile.mkdtemp(prefix="linktelligence-stress-"))
data_dir = temp_root / "data"
data_dir.mkdir(parents=True)

print(f"Generating {NUM_SAMPLES} graphs, 20-100 nodes")

for i in range(NUM_SAMPLES):
    num_nodes = int(rng.integers(20, 101))

    graph = generate_random_graph(num_nodes, rng)
    nodes, edges = assign_attributes(graph, rng)
    labels = calculate_labels(nodes, edges)

    if len(edges) > 500:
        raise ValueError(f"Too many edges: {len(edges)}")

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

dataset = NetworkDataset(root=str(temp_root))
model = NetworkGNN.from_pretrained(MODEL_PATH)

loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Buckets by graph size.
ranges = {
    "20-39": (20, 40),
    "40-59": (40, 60),
    "60-79": (60, 80),
    "80-100": (80, 101),
}

results = {
    name: {
        "score_true": [],
        "score_pred": [],
        "bn_true": [],
        "bn_pred": [],
        "sec_true": [],
        "sec_pred": [],
    }
    for name in ranges
}

model.eval()

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
                for name, (lo, hi) in ranges.items()
                if lo <= num_nodes < hi
            )

            result = results[bucket]

            result["score_true"].append(score_true[i])
            result["score_pred"].append(score_pred[i])

            result["bn_true"].append(bn_true[i])
            result["bn_pred"].append(bn_pred[i])

            result["sec_true"].append(sec_true[i])
            result["sec_pred"].append(sec_pred[i])


print("\n=== STRESS TEST RESULTS ===")

for name, result in results.items():
    count = len(result["score_true"])

    if count == 0:
        print(f"\n{name}: no samples")
        continue

    print(f"\n--- {name} nodes | {count} graphs ---")

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

print("\nStress test complete.")
print("Temporary dataset:", temp_root)