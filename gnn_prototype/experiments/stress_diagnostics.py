from collections import Counter

from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

import torch
from torch_geometric.loader import DataLoader

from dataset import NetworkDataset
from model import NetworkGNN


DATA_ROOT = "/tmp/linktelligence-stress-dk9sko8u"
MODEL_PATH = "/tmp/linktelligence-best-model-v2.pt"

BUCKETS = {
    "20-39": (20, 40),
    "40-59": (40, 60),
    "60-79": (60, 80),
    "80-100": (80, 101),
}

dataset = NetworkDataset(root=DATA_ROOT)
model = NetworkGNN.from_pretrained(MODEL_PATH)

results = {
    name: {"true": [], "pred": []}
    for name in BUCKETS
}

model.eval()

with torch.no_grad():
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    for batch in loader:
        _, _, security = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
        )

        predictions = security.argmax(dim=1).cpu().tolist()
        targets = batch.y[:, 2].long().cpu().tolist()

        node_counts = (
            batch.ptr[1:] - batch.ptr[:-1]
        ).cpu().tolist()

        for n, target, prediction in zip(
            node_counts, targets, predictions
        ):
            for name, (lo, hi) in BUCKETS.items():
                if lo <= n < hi:
                    results[name]["true"].append(target)
                    results[name]["pred"].append(prediction)
                    break


for name, result in results.items():
    y_true = result["true"]
    y_pred = result["pred"]

    if not y_true:
        continue

    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)

    observed_labels = sorted(set(y_true) | set(y_pred))

    print(f"\n{'=' * 55}")
    print(f"BUCKET {name} | {len(y_true)} graphs")
    print("=" * 55)

    print("True distribution:", dict(sorted(true_counts.items())))
    print("Predicted distribution:", dict(sorted(pred_counts.items())))

    print("\nConfusion matrix (labels 0, 1, 2):")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1, 2]))

    print("\nClassification report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        zero_division=0,
    ))

    print(
        "Balanced accuracy:",
        round(balanced_accuracy_score(y_true, y_pred), 4),
    )

    print(
        "Macro F1, all 3 labels:",
        round(f1_score(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        ), 4),
    )

    print(
        "Macro F1, observed labels:",
        round(f1_score(
            y_true,
            y_pred,
            labels=observed_labels,
            average="macro",
            zero_division=0,
        ), 4),
    )