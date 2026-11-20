"""PyTorch Geometric dataset for the network-topology GNN prototype.

Fixes vs. the original LAO-produced version:
  * Labels are read from `data['metadata']` (the original read
    `data['score']` / `data['bottleneck']` / `data['security_flaw']` at the
    top level, which does not exist -> every sample raised KeyError and was
    silently dropped by a bare try/except, leaving an empty dataset).
  * Node features are enriched from one-hot type (7) to 11 channels:
    7 one-hot type + degree + incident_bandwidth_sum (normalised)
    + incident_throughput_sum (normalised) + is_critical(firewall|router).
    The model can now actually see the signals the bottleneck heuristic
    depends on.
  * Edges are added in both directions (undirected graph) and edge_attr
    is duplicated to match, so message passing is symmetric.
  * Processing errors are counted and surfaced (no silent empty dataset).

Subclasses `torch.utils.data.Dataset` (not PyG's `Dataset`) on purpose: the
eager in-memory list works with `torch_geometric.loader.DataLoader` and with
`torch.utils.data.random_split`, and avoids PyG's raw/processed-dir contract
that the original bypassed anyway.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import torch
import torch_geometric.data
from torch.utils.data import Dataset

NODE_TYPES = ["router", "switch", "printer", "host", "firewall", "ids_ips", "l3_switch"]
NUM_NODE_TYPES = len(NODE_TYPES)
TYPE_TO_INDEX = {t: i for i, t in enumerate(NODE_TYPES)}
CRITICAL_TYPES = {"firewall", "router"}
# 7 one-hot type + degree + inc_bandwidth + inc_throughput + is_critical
# + tp_per_bw_ratio (the explicit bottleneck decision variable).
NODE_FEATURE_DIM = NUM_NODE_TYPES + 5


def _minmax(values: List[float]) -> List[float]:
    """Min-max normalise a list to [0, 1]; all-equal -> zeros."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo <= 0:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def build_graph_data(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert a raw topology into PyG tensors (no labels).

    Shared by `NetworkDataset` (training) and the FastAPI inference path, so
    the feature pipeline lives in one place. Returns:
      x          [N, 12] node features
      edge_index [2, 2E] both directions (undirected)
      edge_attr  [2E, 2] normalised (bandwidth, throughput)
      node_map   {original id -> 0..N-1} for callers that map back to ids
      node_ids   sorted original ids
    """
    node_ids = [n["id"] for n in nodes]
    node_map = {nid: i for i, nid in enumerate(sorted(node_ids))}

    # Per-node aggregates needed for enriched features.
    degree = [0] * len(node_ids)
    inc_bw = [0.0] * len(node_ids)
    inc_tp = [0.0] * len(node_ids)
    for e in edges:
        s = node_map[e["source"]]
        t = node_map[e["target"]]
        bw = float(e["bandwidth"])
        tp = float(e["throughput"])
        degree[s] += 1
        degree[t] += 1
        inc_bw[s] += bw
        inc_bw[t] += bw
        inc_tp[s] += tp
        inc_tp[t] += tp

    deg_n = _minmax([float(d) for d in degree])
    bw_n = _minmax(inc_bw)
    tp_n = _minmax(inc_tp)

    # Raw throughput/bandwidth ratio per node -- the bottleneck decision
    # variable (heuristic flags bottleneck==1 when any critical node has
    # ratio>1). Clamped to [0, 10] for numerical stability.
    ratio = [
        min(10.0, inc_tp[i] / inc_bw[i]) if inc_bw[i] > 0 else 0.0
        for i in range(len(node_ids))
    ]

    # Build node feature matrix [N, 12].
    x_rows: List[List[float]] = []
    for n in nodes:
        idx = node_map[n["id"]]
        onehot = [0.0] * NUM_NODE_TYPES
        tidx = TYPE_TO_INDEX.get(n["type"])
        if tidx is not None:
            onehot[tidx] = 1.0
        is_critical = 1.0 if n["type"] in CRITICAL_TYPES else 0.0
        x_rows.append(onehot + [deg_n[idx], bw_n[idx], tp_n[idx], is_critical, ratio[idx]])
    x = torch.tensor(x_rows, dtype=torch.float32)

    # Build edge_index (both directions) + edge_attr (normalised [bw, tp]).
    raw_bw = [float(e["bandwidth"]) for e in edges] * 2  # duplicated for reverse
    raw_tp = [float(e["throughput"]) for e in edges] * 2
    bw_norm = _minmax(raw_bw)
    tp_norm = _minmax(raw_tp)

    src_dst: List[List[int]] = []
    attrs: List[List[float]] = []
    for k, e in enumerate(edges):
        s = node_map[e["source"]]
        t = node_map[e["target"]]
        src_dst.append([s, t])
        src_dst.append([t, s])
        attrs.append([bw_norm[k], tp_norm[k]])          # forward
        attrs.append([bw_norm[len(edges) + k], tp_norm[len(edges) + k]])  # reverse

    edge_index = torch.tensor(src_dst, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(attrs, dtype=torch.float32)

    return {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "node_map": node_map,
        "node_ids": node_ids,
    }


class NetworkDataset(Dataset):
    """Loads graph data from JSON files in `<root>/data/` as PyG `Data`."""

    def __init__(self, root: str):
        self.root = root
        self.data_files = self._load_json_files(root)
        self.processed_data: List[torch_geometric.data.Data] = []
        self._failures: List[str] = []
        self._process_all_data()
        if not self.processed_data:
            raise RuntimeError(
                f"No usable samples in {os.path.join(root, 'data')}. "
                f"{len(self._failures)} file(s) failed: {self._failures[:3]}..."
            )

    @staticmethod
    def _load_json_files(root: str) -> List[str]:
        data_dir = os.path.join(root, "data")
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Data directory not found at {data_dir}")
        return sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".json")
        )

    def _process_single_graph(self, file_path: str) -> torch_geometric.data.Data:
        with open(file_path, "r") as f:
            data: Dict[str, Any] = json.load(f)

        # Feature pipeline is shared with the inference path (build_graph_data)
        # so training and serving cannot drift apart.
        g = build_graph_data(data["nodes"], data["edges"])

        # Labels (read from metadata, not the top level).
        meta = data["metadata"]
        score = float(meta["score"])
        bottleneck = int(meta["bottleneck"])
        security_flaw = int(meta["security_flaw"])
        # Graph-level label, stored as [1, 3] so that PyG batching yields
        # [batch_size, 3] (a 1-D [3] per graph would concatenate into [batch*3]).
        y = torch.tensor([[score, bottleneck, security_flaw]], dtype=torch.float32)

        return torch_geometric.data.Data(
            x=g["x"], edge_index=g["edge_index"], edge_attr=g["edge_attr"], y=y
        )

    def _process_all_data(self) -> None:
        print(f"Processing {len(self.data_files)} JSON file(s)...")
        for file_path in self.data_files:
            try:
                self.processed_data.append(self._process_single_graph(file_path))
            except Exception as exc:  # noqa: BLE001 - surface, don't drop silently
                self._failures.append(f"{os.path.basename(file_path)}: {exc}")
        print(
            f"Processed {len(self.processed_data)}/{len(self.data_files)} samples"
            + (f" ({len(self._failures)} failed)" if self._failures else "")
        )

    def __len__(self) -> int:
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]


if __name__ == "__main__":
    # Quick standalone sanity check.
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "."
    ds = NetworkDataset(root=root)
    sample = ds[0]
    print(f"len={len(ds)} | sample.x={tuple(sample.x.shape)} "
          f"edge_index={tuple(sample.edge_index.shape)} "
          f"edge_attr={tuple(sample.edge_attr.shape)} y={sample.y.tolist()}")