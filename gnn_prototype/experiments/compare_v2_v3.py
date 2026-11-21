from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report
import torch

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)

from torch_geometric.loader import DataLoader

from dataset import NetworkDataset
from model import NetworkGNN


DATA_ROOT = "/tmp/linktelligence-stress-dk9sko8u"

MODELS = {
    "v2": "/tmp/linktelligence-best-model-v2.pt",
    "v3": "/tmp/linktelligence-best-model-v3.pt",
}

BUCKETS = {
    "20-39": (20, 40),
    "40-59": (40, 60),
    "60-79": (60, 80),
    "80-100": (80, 101),
}

dataset = NetworkDataset(root=DATA_ROOT)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
)

results = {
    model_name: {
        bucket: {
            "score_true": [],
            "score_pred": [],
            "bn_true": [],
            "bn_pred": [],
            "sec_true": [],
            "sec_pred": [],
        }
        for bucket in BUCKETS
    }
    for model_name in MODELS
}


for model_name, model_path in MODELS.items():
    print(f"\nEvaluating {model_name}: {model_path}")

    model = NetworkGNN.from_pretrained(model_path)
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

            for i, n in enumerate(node_counts):
                bucket = next(
                    name
                    for name, (lo, hi) in BUCKETS.items()
                    if lo <= n < hi
                )

                result = results[model_name][bucket]

                result["score_true"].append(score_true[i])
                result["score_pred"].append(score_pred[i])

                result["bn_true"].append(bn_true[i])
                result["bn_pred"].append(bn_pred[i])

                result["sec_true"].append(sec_true[i])
                result["sec_pred"].append(sec_pred[i])


for bucket in BUCKETS:
    print(f"\n{'=' * 60}")
    print(f"GRAPH SIZE: {bucket}")
    print("=" * 60)

    for model_name in MODELS:
        r = results[model_name][bucket]

        score_mae = mean_absolute_error(
            r["score_true"], r["score_pred"]
        )

        bn_f1 = f1_score(
            r["bn_true"],
            r["bn_pred"],
            labels=[0, 1],
            average="macro",
            zero_division=0,
        )

        sec_true_labels = sorted(set(r["sec_true"]))

        sec_f1 = f1_score(
            r["sec_true"],
            r["sec_pred"],
            labels=sec_true_labels,
            average="macro",
            zero_division=0,
        )

        true_dist = Counter(r["sec_true"])
        pred_dist = Counter(r["sec_pred"])

        print(f"\n{model_name.upper()}")
        print(f"Score MAE: {score_mae:.4f}")
        print(f"Bottleneck macro-F1: {bn_f1:.4f}")
        print(f"Security macro-F1 (observed classes): {sec_f1:.4f}")
        print("Security true distribution:", dict(sorted(true_dist.items())))
        print("Security predicted distribution:", dict(sorted(pred_dist.items())))
        print("Security balanced accuracy:", round(
        balanced_accuracy_score(
            r["sec_true"],
            r["sec_pred"],
            ),
            4,
        ))

        print("Security confusion matrix [0, 1, 2]:")
        print(confusion_matrix(
            r["sec_true"],
            r["sec_pred"],
            labels=[0, 1, 2],
        ))

        print("Security classification report:")
        print(classification_report(
            r["sec_true"],
            r["sec_pred"],
            labels=[0, 1, 2],
            zero_division=0,
        ))