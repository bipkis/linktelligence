"""Generate synthetic network-topology graphs for the GNN prototype.

Fixes vs. the original LAO-produced version:
  * line 72 SyntaxError (stray `'` after `neighbor`) -> removed;
    edge lookup is now O(1) via a frozenset-indexed dict instead of an
    O(E) scan per neighbour.
  * global RNG seed added (the old "fixed seed" comment was misleading:
    `seed=np.random.randint(...)` was randomised per call).
  * `security_flaw` is now a clean integer count of distinct issue types
    in {0, 1, 2} (was a float that accumulated 0.2 per printer-router
    edge and then got force-cast to int downstream, losing severity).
  * `score` is clamped to [0, 1].

Labels live under `metadata` (unchanged schema), so `dataset.py` must read
them from `data['metadata']`, not the top level.
"""
from __future__ import annotations


import json
import os
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
from heuristics import calculate_labels
# --- Configuration ---
NUM_SAMPLES = 2000
NODE_TYPES = ["router", "switch", "printer", "host", "firewall", "ids_ips", "l3_switch"]
# Weighted node-type distribution. Under uniform assignment security_flaw==2
# (which needs BOTH issue A "host present with no firewall AND no ids_ips"
# AND issue B "a printer<->router edge") occurred only ~2% of the time,
# because issue A rarely fires when firewall/ids_ips are as common as other
# types. Making firewall/ids_ips rarer (~8% each) and host/switch/l3_switch
# commoner lifts issue A and brings class-2 to ~6-8% of samples, so the
# stratified split gives it enough train/test support to learn. The
# heuristic rules in calculate_heuristics are unchanged -- only the input
# distribution shifts.
NODE_TYPE_WEIGHTS = [0.12, 0.20, 0.12, 0.20, 0.08, 0.08, 0.20]
MIN_BANDWIDTH = 100
MAX_BANDWIDTH = 10000
MIN_THROUGHPUT = 10
MAX_THROUGHPUT = 5000
SEED = 42


def generate_random_graph(num_nodes: int, rng: np.random.Generator) -> nx.Graph:
    """Generate a random connected graph with the specified number of nodes."""
    G = nx.gnm_random_graph(num_nodes, m=max(num_nodes - 1, num_nodes * 2), seed=int(rng.integers(0, 2**31 - 1)))
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            nodes1 = list(components[i])
            nodes2 = list(components[i + 1])
            u = int(rng.choice(nodes1))
            v = int(rng.choice(nodes2))
            G.add_edge(u, v)
    return G


def assign_attributes(G: nx.Graph, rng: np.random.Generator) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Assign random node types and edge bandwidth/throughput."""
    p = np.array(NODE_TYPE_WEIGHTS, dtype=float)
    p = p / p.sum()  # guard against float drift in rng.choice's sum check
    node_data = [
        {"id": int(n), "type": str(rng.choice(NODE_TYPES, p=p))}
        for n in G.nodes()
    ]

    edge_data = []
    for u, v in G.edges():
        bandwidth = int(rng.integers(MIN_BANDWIDTH, MAX_BANDWIDTH + 1))
        throughput = int(rng.integers(MIN_THROUGHPUT, MAX_THROUGHPUT + 1))
        edge_data.append({"source": int(u), "target": int(v), "bandwidth": bandwidth, "throughput": throughput})

    return node_data, edge_data



def main() -> None:
    rng = np.random.default_rng(SEED)
    print(f"Generating {NUM_SAMPLES} random network graphs (seed={SEED})...")
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    for i in range(NUM_SAMPLES):
        num_nodes = int(rng.integers(5, 20))
        G = generate_random_graph(num_nodes, rng)
        node_data, edge_data = assign_attributes(G, rng)
        heuristics = calculate_labels(node_data, edge_data)

        sample_record = {
            "sample_id": i + 1,
            "nodes": node_data,
            "edges": edge_data,
            "metadata": {
                "score": round(heuristics["score"], 3),
                "bottleneck": heuristics["bottleneck"],
                "security_flaw": heuristics["security_flaw"],
            },
        }

        file_path = os.path.join(data_dir, f"sample_{i + 1:04d}.json")
        with open(file_path, "w") as f:
            json.dump(sample_record, f, indent=4)

    print(f"Data generation complete. {NUM_SAMPLES} files saved to '{data_dir}/'.")


if __name__ == "__main__":
    main()