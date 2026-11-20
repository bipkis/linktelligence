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


def calculate_heuristics(
    G: nx.Graph, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]
) -> Dict[str, float]:
    """Compute the three labels from deterministic heuristic rules.

    score          : float in [0, 1] (regression target).
    bottleneck     : int {0, 1}      (any firewall/router whose incident
                                      throughput sum exceeds its incident
                                      bandwidth sum).
    security_flaw  : int {0, 1, 2}   (count of distinct issue types:
                                      +1 if a host exists with neither
                                      firewall nor ids_ips; +1 if any
                                      printer<->router direct edge).
    """
    score = 1.0
    bottleneck = 0
    security_issues = 0

    node_types = {d["id"]: d["type"] for d in nodes}

    # O(1) edge lookup keyed by an unordered endpoint pair.
    edge_lookup = {frozenset((e["source"], e["target"])): e for e in edges}

    # --- Bottleneck check: aggregate per critical node (firewall/router) ---
    critical_nodes = [n for n, t in node_types.items() if t in ("firewall", "router")]
    for node in critical_nodes:
        total_throughput = 0
        incident_bandwidth = 0
        for neighbor in G.neighbors(node):
            e = edge_lookup.get(frozenset((node, neighbor)))
            if e is not None:
                total_throughput += e["throughput"]
                incident_bandwidth += e["bandwidth"]
        if total_throughput > incident_bandwidth:
            bottleneck = 1
            score -= 0.3

    # --- Security issue A: host without firewall and without ids_ips ---
    has_host = any(t == "host" for t in node_types.values())
    has_firewall = any(t == "firewall" for t in node_types.values())
    has_ids_ips = any(t == "ids_ips" for t in node_types.values())
    if has_host and not has_firewall and not has_ids_ips:
        security_issues += 1
        score -= 0.4

    # --- Security issue B: any printer<->router direct connection ---
    for u, v in G.edges():
        tu, tv = node_types.get(u), node_types.get(v)
        if {tu, tv} == {"printer", "router"}:
            security_issues += 1
            score -= 0.2
            break  # one occurrence is enough to flag the issue type

    return {
        "score": float(max(0.0, min(1.0, score))),
        "bottleneck": int(bottleneck),
        "security_flaw": int(security_issues),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    print(f"Generating {NUM_SAMPLES} random network graphs (seed={SEED})...")
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    for i in range(NUM_SAMPLES):
        num_nodes = int(rng.integers(5, 20))
        G = generate_random_graph(num_nodes, rng)
        node_data, edge_data = assign_attributes(G, rng)
        heuristics = calculate_heuristics(G, node_data, edge_data)

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