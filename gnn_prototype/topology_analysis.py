"""Heuristic expertise engine for LAN topologies.

Produces the actionable, human-readable part of the analysis: WHICH node is
the bottleneck, WHICH hosts are unprotected, WHICH printer<->router edges
violate segmentation. This mirrors the label rules in `generate_data.py`'s
`calculate_heuristics` (so the model's predictions and these explanations
describe the same risk model), but returns per-node / per-edge detail instead
of just integer counts.

Standalone on purpose -- no torch, no networkx -- so it is trivial to unit
test and to reuse outside the GNN stack. Input schema (same as the generated
samples minus metadata):

    nodes: [{"id": int, "type": "router"|"switch"|"printer"|"host"|
                                 "firewall"|"ids_ips"|"l3_switch"}]
    edges: [{"source": int, "target": int, "bandwidth": int, "throughput": int}]
"""
from __future__ import annotations

from typing import Any, Dict, List
from heuristics import analyze_topology


def summarize(report: Dict[str, Any]) -> str:
    """One-paragraph human summary of the heuristic report."""
    parts: List[str] = []
    bn = report["bottleneck"]
    if bn["present"]:
        ids = ", ".join(str(n["id"]) for n in bn["offending_nodes"])
        parts.append(
            f"Bottleneck detected: {len(bn['offending_nodes'])} critical node(s) "
            f"({ids}) are overloaded (throughput > bandwidth)."
        )
    else:
        parts.append("No bottleneck: every critical node has spare capacity.")

    sec = report["security"]
    if sec["class"] == 0:
        parts.append("No security topology issues detected.")
    else:
        flags = []
        if sec["issue_a"]["present"]:
            flags.append("unprotected hosts")
        if sec["issue_b"]["present"]:
            flags.append("printer-router segmentation violation")
        parts.append(f"Security issues: {', '.join(flags)} (class {sec['class']}).")

    parts.append(f"Heuristic score {report['score']:.2f}/1.00.")
    return " ".join(parts)


if __name__ == "__main__":
    # Self-test on a tiny risky topology: host w/o firewall + printer-router edge.
    demo_nodes = [
        {"id": 1, "type": "router"},
        {"id": 2, "type": "printer"},
        {"id": 3, "type": "host"},
        {"id": 4, "type": "switch"},
    ]
    demo_edges = [
        {"source": 1, "target": 2, "bandwidth": 1000, "throughput": 2000},  # overload + printer-router
        {"source": 1, "target": 4, "bandwidth": 10000, "throughput": 100},
        {"source": 4, "target": 3, "bandwidth": 1000, "throughput": 100},
    ]
    import json as _json
    r = analyze_topology(demo_nodes, demo_edges)
    print(_json.dumps(r, indent=2))
    print("--- summary ---")
    print(summarize(r))