"""GNN model for the network-topology prototype (GAT, edge-aware).

Fixes vs. the original LAO-produced version:
  * `GATConv` now takes `edge_dim=2` so the normalised
    (bandwidth, throughput) edge features are actually consumed by the
    attention mechanism. The original passed `edge_attr=` to GATConv but
    never set `edge_dim`, so the features were silently discarded -- and
    those features are exactly what the bottleneck heuristic depends on.
  * conv2 uses `concat=False` so its output dimension is `hidden_channels`
    (not `hidden_channels * heads`), matching what the readout heads expect.
  * `forward` now takes a `batch` vector and calls
    `global_mean_pool(x, batch)`. The original called
    `global_mean_pool(x, num_nodes=x.size(0))`, which is a TypeError
    (`num_nodes` is not a parameter of `global_mean_pool`) and would also
    have collapsed a whole mini-batch into a single graph vector.
  * The security head has 3 outputs (security_flaw is a count in {0,1,2}),
    not 2.
  * `from_pretrained` instantiates with `in_channels=NODE_FEATURE_DIM=11`.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_max_pool, global_mean_pool

# Must match dataset.NODE_FEATURE_DIM
# (7 one-hot + degree + inc_bw + inc_tp + is_critical + tp_per_bw_ratio).
NODE_FEATURE_DIM = 12
SECURITY_CLASSES = 3  # security_flaw count in {0, 1, 2}


class NetworkGNN(nn.Module):
    """Edge-aware GAT that predicts score / bottleneck / security_flaw."""

    def __init__(self, in_channels: int = NODE_FEATURE_DIM, hidden_channels: int = 64, out_channels: int = 3):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, edge_dim=2)
        # conv1 output: hidden_channels * 4 (concat=True, the default).
        self.conv2 = GATConv(hidden_channels * 4, hidden_channels, heads=4, edge_dim=2, concat=False)
        # conv2 output: hidden_channels (concat=False).

        # Readout heads operate on the pooled graph vector (hidden_channels).
        self.head_score = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
        )
        self.head_bottleneck = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 2),
        )
        self.head_security = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, SECURITY_CLASSES),
        )

    @classmethod
    def from_pretrained(cls, model_path: str):
        model = cls(in_channels=NODE_FEATURE_DIM, hidden_channels=64, out_channels=3)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        return model

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ):
        # 1. Message passing (edge features now actually feed the attention).
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = torch.relu(x)

        # 2. Graph-level readout. `batch` maps each node to its graph index;
        #    without it a mini-batch of N graphs would pool into one vector.
        #    Bottleneck is a single-node event ("any critical node with
        #    ratio>1"), so its head reads from global_max_pool -- the worst
        #    node -- instead of the mean, which would dilute the offender
        #    across ~5-7 nodes. score/security keep the mean readout.
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)

        # 3. Task heads.
        score = self.head_score(x_mean)
        bottleneck = self.head_bottleneck(x_max)
        security_flaw = self.head_security(x_mean)
        return score, bottleneck, security_flaw


if __name__ == "__main__":
    # Tiny smoke test on a single random graph (needs torch_geometric installed).
    m = NetworkGNN()
    n = 8
    x = torch.randn(n, NODE_FEATURE_DIM)
    ei = torch.tensor([[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7]], dtype=torch.long)
    ea = torch.randn(7, 2)
    out = m(x, ei, ea, torch.zeros(n, dtype=torch.long))
    print("forward OK ->",
          "score", tuple(out[0].shape),
          "bottleneck", tuple(out[1].shape),
          "security", tuple(out[2].shape))