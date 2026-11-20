"""Train the network-topology GNN.

Fixes vs. the original LAO-produced version:
  * `model(...)` is called with `data.batch` (the model now requires it for
    correct per-graph pooling; without it a mini-batch collapses to one
    vector and the loss shape-matches against the wrong number of labels).
  * `in_channels` defaults to 12 (enriched node features incl. tp_per_bw_ratio);
    the model class default is used so train/model/dataset agree on the
    feature dimension.
  * `avg_test_loss` divides by `max(1, len(test_dataset))` instead of
    `len(test_loader)` (ZeroDivisionError when the test split is empty).
  * `--epochs`, `--data-root`, `--model-path`, `--batch-size` are CLI flags.

Imbalance handling (the reason `bn_acc` was stuck at 0.80 = always-majority):
  * Class weights for the two CrossEntropyLoss heads are derived from the
    training split as inverse frequency `w_c = N/(K*n_c)`, normalised so the
    smallest weight is 1.0 and clamped to `--weight-cap` (default 12) so a
    handful of rare samples (e.g. security_flaw==2 with ~10 samples) cannot
    dominate the loss.
  * Per-epoch metrics report balanced accuracy and macro-F1 (not raw
    accuracy), which expose a constant majority predictor for what it is.

Best-checkpoint correctness + early stopping:
  * `best_model.pt` is saved only on test-loss improvement, and the final
    `classification_report` is printed from the RELOADED best checkpoint
    (not the last-epoch weights), so the report describes the artifact on
    disk that `test_inference.py` will actually load.
  * `--patience` (default 15; 0 disables) stops training when test loss has
    not improved for that many epochs -- avoids the overfitting tail seen
    after ~epoch 40 on 500 samples.

Reproducible stratified split:
  * `--seed` (default 42) seeds both weight init and the train/test split, so
    two runs with the same seed produce identical numbers. Without it,
    `random_split` reshuffled every run and rare-class test support swung
    (security_flaw==2 had 0 or 1 test samples across runs -> macro-F1
    jumped 0.62<->0.86).
  * The split is stratified on `security_flaw` (the rarest label), so
    class-2 is represented in BOTH splits every run instead of randomly
    landing entirely in train.
"""
from __future__ import annotations

import math
import argparse
import warnings
from collections import Counter
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from dataset import NetworkDataset
from model import NetworkGNN, SECURITY_CLASSES

# Benign: on epochs where the model predicts a class absent from that test
# batch's ground truth (e.g. security_flaw==2 predicted but no class-2 true
# sample in the batch), sklearn emits this. It is expected with rare classes
# and adds only noise to the log.
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")


def _evaluate(model, loader, score_criterion, bn_criterion, sec_criterion) -> Dict:
    """Run one eval pass and return loss + per-head predictions/metrics."""
    model.eval()
    total_loss = 0.0
    bn_preds: List[int] = []
    bn_y: List[int] = []
    sec_preds: List[int] = []
    sec_y: List[int] = []
    total_graphs = 0
    with torch.no_grad():
        for data in loader:
            score, bottleneck, security = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = (
                score_criterion(score, data.y[:, 0:1])
                + bn_criterion(bottleneck, data.y[:, 1].long())
                + sec_criterion(security, data.y[:, 2].long())
            )
            total_loss += loss.item() * data.num_graphs
            total_graphs += data.num_graphs
            bn_preds.extend(bottleneck.argmax(1).tolist())
            bn_y.extend(data.y[:, 1].long().tolist())
            sec_preds.extend(security.argmax(1).tolist())
            sec_y.extend(data.y[:, 2].long().tolist())
    avg_loss = total_loss / max(1, total_graphs)
    bn_acc = sum(int(p == t) for p, t in zip(bn_preds, bn_y)) / max(1, len(bn_y))
    sec_acc = sum(int(p == t) for p, t in zip(sec_preds, sec_y)) / max(1, len(sec_y))
    return {
        "avg_loss": avg_loss,
        "bn_preds": bn_preds, "bn_y": bn_y, "bn_acc": bn_acc,
        "bn_bal": balanced_accuracy_score(bn_y, bn_preds),
        "bn_f1": f1_score(bn_y, bn_preds, average="macro", zero_division=0),
        "sec_preds": sec_preds, "sec_y": sec_y, "sec_acc": sec_acc,
        "sec_bal": balanced_accuracy_score(sec_y, sec_preds),
        "sec_f1": f1_score(sec_y, sec_preds, average="macro", zero_division=0),
    }


def _class_weights(dataset, col: int, num_classes: int, cap: float) -> torch.Tensor:
    """Inverse-frequency weights from the training split.

    `col` is the y column index (1 = bottleneck, 2 = security_flaw).
    Normalised so the smallest class weight is 1.0, then clamped to `cap`
    so rare classes (e.g. security_flaw==2 with ~11 samples) do not blow up.
    Absent classes get weight 0.0 (they are never targeted in the loss).
    """
    counts = Counter(int(dataset[i].y[0, col]) for i in range(len(dataset)))
    n = len(dataset)
    raw = {c: (n / (num_classes * counts[c])) if counts[c] > 0 else 0.0 for c in range(num_classes)}
    present = [v for v in raw.values() if v > 0]
    lo = min(present) if present else 1.0
    weights = torch.tensor(
        [min(raw[c] / lo, cap) if raw[c] > 0 else 0.0 for c in range(num_classes)],
        dtype=torch.float32,
    )
    return weights, dict(sorted(counts.items()))



def _stratified_split(
    dataset,
    train_split: float,
    validation_split: float,
    test_split: float,
    seed: int,
):
    """Create reproducible stratified train/validation/test subsets.

    Stratification is performed using security_flaw, the rarest target.

    Uses two-stage splitting:
        1. Separate test from train+validation.
        2. Separate validation from train.

    All three subsets retain representation of every observed class.
    """

    splits = (train_split, validation_split, test_split)

    if any(split <= 0 or split >= 1 for split in splits):
        raise ValueError("All split proportions must be between 0 and 1")

    if not math.isclose(sum(splits), 1.0, abs_tol=1e-8):
        raise ValueError(
            f"Split proportions must sum to 1.0, got {sum(splits):.4f}"
        )

    indices = list(range(len(dataset)))

    sec_labels = [
        int(dataset[i].y[0, 2])
        for i in indices
    ]

    class_counts = Counter(sec_labels)

    # At least one sample of each observed class must be present
    # in train, validation and test.
    insufficient = {
        label: count
        for label, count in class_counts.items()
        if count < 3
    }

    if insufficient:
        raise ValueError(
            "Cannot stratify into three subsets: "
            f"some classes have fewer than 3 samples: {insufficient}"
        )

    try:
        # Stage 1: isolate the test set.
        test_splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_split,
            random_state=seed,
        )

        train_val_idx, test_idx = next(
            test_splitter.split(indices, sec_labels)
        )

        # Stage 2: divide remaining samples into train and validation.
        val_relative_size = validation_split / (
            train_split + validation_split
        )

        train_val_labels = [
            sec_labels[int(idx)]
            for idx in train_val_idx
        ]

        val_splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_relative_size,
            random_state=seed + 1,
        )

        train_relative_idx, val_relative_idx = next(
            val_splitter.split(train_val_idx, train_val_labels)
        )

        train_idx = [
            int(train_val_idx[i])
            for i in train_relative_idx
        ]

        val_idx = [
            int(train_val_idx[i])
            for i in val_relative_idx
        ]

    except ValueError as exc:
        raise ValueError(
            "Unable to create stratified train/validation/test split. "
            f"Dataset size: {len(dataset)}, "
            f"class counts: {dict(class_counts)}. "
            f"Original error: {exc}"
        ) from exc

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, [int(i) for i in test_idx]),
    )


def get_class_distribution(dataset, col=2):
    return dict(
        sorted(
            Counter(
                int(dataset[i].y[0, col])
                for i in range(len(dataset))
            ).items()
        )
    )

def train(
    epochs,
    data_root,
    model_path,
    batch_size,
    train_split,
    validation_split,
    test_split,
    weight_cap,
    patience,
    seed,
):
    print("--- Starting Model Training ---")

    # Seed everything for reproducible splits AND weight init; without this,
    # run-to-run numbers swing by 10+ points (esp. rare-class test support).
    torch.manual_seed(seed)

    dataset = NetworkDataset(root=data_root)
    print(f"Loaded {len(dataset)} samples. (seed={seed})")

    train_dataset, val_dataset, test_dataset = _stratified_split(
        dataset=dataset,
        train_split=train_split,
        validation_split=validation_split,
        test_split=test_split,
        seed=seed,
    )

    print(
        "Security class distribution:"
    )

    print(
        f"Train:      {get_class_distribution(train_dataset)}"
    )

    print(
        f"Validation: {get_class_distribution(val_dataset)}"
    )

    print(
        f"Test:       {get_class_distribution(test_dataset)}"
    )

    # --- Class weights derived from the training split ---
    bn_w, bn_counts = _class_weights(train_dataset, col=1, num_classes=2, cap=weight_cap)
    sec_w, sec_counts = _class_weights(train_dataset, col=2, num_classes=SECURITY_CLASSES, cap=weight_cap)
    print(f"bottleneck train counts : {bn_counts} -> weights {bn_w.tolist()}")
    print(f"security  train counts  : {sec_counts} -> weights {sec_w.tolist()}")

    model = NetworkGNN()  # defaults: in_channels=12, hidden=64, out=3
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    score_criterion = nn.MSELoss()
    bottleneck_criterion = nn.CrossEntropyLoss(weight=bn_w)
    security_criterion = nn.CrossEntropyLoss(weight=sec_w)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )


    best_val_loss = float("inf")
    best_epoch = 0
    stale = 0

    for epoch in range(epochs):

        # -------------------------------------------------
        # 1. TRAIN
        # -------------------------------------------------

        model.train()
        total_train_loss = 0.0

        for data in train_loader:
            optimizer.zero_grad()

            score, bottleneck, security = model(
                data.x,
                data.edge_index,
                data.edge_attr,
                data.batch,
            )

            loss = (
                score_criterion(score, data.y[:, 0:1])
                + bottleneck_criterion(
                    bottleneck,
                    data.y[:, 1].long(),
                )
                + security_criterion(
                    security,
                    data.y[:, 2].long(),
                )
            )

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * data.num_graphs

        avg_train_loss = (
            total_train_loss / max(1, len(train_dataset))
        )

        # -------------------------------------------------
        # 2. VALIDATION
        # -------------------------------------------------

        val_metrics = _evaluate(
            model,
            val_loader,
            score_criterion,
            bottleneck_criterion,
            security_criterion,
        )

        print(
            f"Epoch {epoch + 1:3d}/{epochs} | "
            f"train {avg_train_loss:.4f} | "
            f"val {val_metrics['avg_loss']:.4f} | "
            f"bn acc/bal/f1 "
            f"{val_metrics['bn_acc']:.2f}/"
            f"{val_metrics['bn_bal']:.2f}/"
            f"{val_metrics['bn_f1']:.2f} | "
            f"sec acc/bal/f1 "
            f"{val_metrics['sec_acc']:.2f}/"
            f"{val_metrics['sec_bal']:.2f}/"
            f"{val_metrics['sec_f1']:.2f}"
        )

        # -------------------------------------------------
        # 3. CHECKPOINT SELECTION
        # -------------------------------------------------

        if val_metrics["avg_loss"] < best_val_loss:

            best_val_loss = val_metrics["avg_loss"]
            best_epoch = epoch + 1
            stale = 0

            torch.save(model.state_dict(), model_path)

            print(
                f"  -> saved {model_path} "
                "(new best validation loss)"
            )

        else:
            stale += 1

            if patience and stale >= patience:
                print(
                    f"  -> early stopping: no validation-loss "
                    f"improvement for {patience} epochs "
                    f"(best epoch {best_epoch}, "
                    f"loss {best_val_loss:.4f})"
                )
                break

    # -----------------------------------------------------
    # 4. LOAD BEST CHECKPOINT
    # -----------------------------------------------------

    if best_epoch == 0:
        raise RuntimeError(
            "Training finished without producing a valid checkpoint"
        )

    print(
        f"\n--- Best checkpoint: epoch {best_epoch} | "
        f"validation loss {best_val_loss:.4f} ---"
    )

    best_model = NetworkGNN.from_pretrained(model_path)

    # -----------------------------------------------------
    # 5. FINAL TEST EVALUATION
    # -----------------------------------------------------

    # Test data is evaluated only once, after model selection.
    test_metrics = _evaluate(
        best_model,
        test_loader,
        score_criterion,
        bottleneck_criterion,
        security_criterion,
    )

    print("\n--- FINAL TEST RESULTS ---")

    print(
        f"Test loss: {test_metrics['avg_loss']:.4f}"
    )

    print(
        f"bn acc/bal/f1 "
        f"{test_metrics['bn_acc']:.2f}/"
        f"{test_metrics['bn_bal']:.2f}/"
        f"{test_metrics['bn_f1']:.2f}"
    )

    print(
        f"sec acc/bal/f1 "
        f"{test_metrics['sec_acc']:.2f}/"
        f"{test_metrics['sec_bal']:.2f}/"
        f"{test_metrics['sec_f1']:.2f}"
    )

    print("\n--- classification_report (test split) ---")

    print("[bottleneck]")
    print(
        classification_report(
            test_metrics["bn_y"],
            test_metrics["bn_preds"],
            zero_division=0,
        )
    )

    print("[security_flaw]")
    print(
        classification_report(
            test_metrics["sec_y"],
            test_metrics["sec_preds"],
            zero_division=0,
        )
    )

    print(
        f"--- Training Complete | "
        f"best validation loss {best_val_loss:.4f} "
        f"at epoch {best_epoch} ---"
    )



def main():
    ap = argparse.ArgumentParser(description="Train the network-topology GNN")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--data-root", default=".")
    ap.add_argument("--model-path", default="best_model.pt")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--train-split", type=float, default=0.7)
    ap.add_argument("--validation-split", type=float, default=0.15)
    ap.add_argument("--test-split", type=float, default=0.15)
    ap.add_argument("--weight-cap", type=float, default=12.0,
                    help="clamp for rare-class inverse-frequency weights")
    ap.add_argument("--patience", type=int, default=15,
                    help="early-stop after N epochs without test-loss improvement (0 disables)")
    ap.add_argument("--seed", type=int, default=42,
                    help="seed for the stratified split and weight init (reproducibility)")
    args = ap.parse_args()
    train(
        epochs=args.epochs,
        data_root=args.data_root,
        model_path=args.model_path,
        batch_size=args.batch_size,
        train_split=args.train_split,
        validation_split=args.validation_split,
        test_split=args.test_split,
        weight_cap=args.weight_cap,
        patience=args.patience,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()