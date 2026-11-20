"""Run inference on one sample with the trained GNN.

Fixes vs. the original LAO-produced version (this is the task that LAO
escalated to `max_attempts_reached` / failed):
  * Uses the real `NetworkGNN`, not a `ModelPlaceholder(nn.Linear(10,3))`
    whose `load_state_dict` always failed on key mismatch.
  * Calls `model(x, edge_index, edge_attr, batch)` -- the real signature --
    instead of `model(input_tensor)` with a single tensor argument.
  * Reads the three head outputs as tensors and formats them correctly
    (`score.item()`, `argmax` for the two classification heads) instead of
    treating the return value as a Python list and calling `.tolist()`/
    `:0.2f` formatting on tensors.
  * Also prints the ground-truth labels for comparison.
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from torch_geometric.data import Data

from dataset import NetworkDataset
from model import NetworkGNN


def run_inference(model_path: str, data_root: str, sample_idx: int = 0):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}; run train.py first.")

    model = NetworkGNN.from_pretrained(model_path)
    print(f"Loaded model from {model_path}")

    dataset = NetworkDataset(root=data_root)
    if sample_idx >= len(dataset):
        raise IndexError(f"sample_idx {sample_idx} >= dataset size {len(dataset)}")

    sample: Data = dataset[sample_idx]
    # Single-graph batch vector: all nodes belong to graph 0.
    batch = torch.zeros(sample.num_nodes, dtype=torch.long)

    with torch.no_grad():
        score, bottleneck, security = model(sample.x, sample.edge_index, sample.edge_attr, batch)

    # Recover the ground-truth labels from the source JSON for comparison.
    meta = _ground_truth(data_root, sample_idx)
    bn_pred = int(bottleneck.argmax(1).item())
    sec_pred = int(security.argmax(1).item())

    print("\n--- Inference Results ---")
    print(f"sample index    : {sample_idx}")
    print(f"nodes / edges   : {sample.num_nodes} / {sample.num_edges}")
    print(f"score           : pred {score.item():.3f}   | true {meta['score']}")
    print(f"bottleneck      : pred {bn_pred}            | true {meta['bottleneck']}  "
          f"(logits {[round(v, 3) for v in bottleneck.view(-1).tolist()]})")
    print(f"security_flaw   : pred {sec_pred}            | true {meta['security_flaw']}  "
          f"(logits {[round(v, 3) for v in security.view(-1).tolist()]})")


def _ground_truth(data_root: str, sample_idx: int) -> dict:
    data_dir = os.path.join(data_root, "data")
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".json"))
    with open(os.path.join(data_dir, files[sample_idx])) as f:
        return json.load(f)["metadata"]


def main():
    ap = argparse.ArgumentParser(description="Run inference on one sample")
    ap.add_argument("--model-path", default="best_model.pt")
    ap.add_argument("--data-root", default=".")
    ap.add_argument("--sample", type=int, default=0)
    args = ap.parse_args()
    run_inference(args.model_path, args.data_root, args.sample)


if __name__ == "__main__":
    main()