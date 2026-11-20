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

CRITICAL_TYPES = {"firewall", "router"}
NODE_TYPES = ["router", "switch", "printer", "host", "firewall", "ids_ips", "l3_switch"]


def _per_node_aggregates(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]):
    """degree / incident bandwidth / incident throughput per node id."""
    deg: Dict[int, int] = {n["id"]: 0 for n in nodes}
    inc_bw: Dict[int, float] = {n["id"]: 0.0 for n in nodes}
    inc_tp: Dict[int, float] = {n["id"]: 0.0 for n in nodes}
    for e in edges:
        s, t = e["source"], e["target"]
        bw, tp = float(e["bandwidth"]), float(e["throughput"])
        deg[s] += 1
        deg[t] += 1
        inc_bw[s] += bw
        inc_bw[t] += bw
        inc_tp[s] += tp
        inc_tp[t] += tp
    return deg, inc_bw, inc_tp


def analyze_topology(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a structured expertise report for a single topology.

    The dict is JSON-serialisable and is what the FastAPI endpoint returns
    (alongside the model's predictions) under ``heuristic``.
    """
    if not nodes:
        raise ValueError("topology has no nodes")

    node_type = {n["id"]: n["type"] for n in nodes}
    deg, inc_bw, inc_tp = _per_node_aggregates(nodes, edges)

    # --- Bottleneck: any critical node whose incident throughput exceeds its
    # incident bandwidth (capacity overload). Report each offending node with
    # the raw numbers so the operator knows where to add capacity. ---
    offending_nodes: List[Dict[str, Any]] = []
    for n in nodes:
        nid, ntype = n["id"], n["type"]
        if ntype not in CRITICAL_TYPES:
            continue
        bw, tp = inc_bw[nid], inc_tp[nid]
        if bw <= 0:
            continue
        ratio = tp / bw
        if tp > bw:
            offending_nodes.append({
                "id": nid,
                "type": ntype,
                "throughput": int(tp),
                "bandwidth": int(bw),
                "ratio": round(ratio, 3),
                "overshoot": int(tp - bw),
                "degree": deg[nid],
            })
    bottleneck_present = len(offending_nodes) > 0

    # --- Security issue A: a host exists but the graph has NO firewall and NO
    # ids_ips anywhere -- every host is therefore unmonitored/unfiltered. ---
    types_present = set(node_type.values())
    has_host = "host" in types_present
    has_firewall = "firewall" in types_present
    has_ids_ips = "ids_ips" in types_present
    issue_a_present = has_host and not has_firewall and not has_ids_ips
    host_ids = [n["id"] for n in nodes if n["type"] == "host"]

    # --- Security issue B: any direct printer<->router edge -- a printer on a
    # router-facing segment bypasses the usual switch/firewall control point. ---
    printer_router_edges: List[Dict[str, Any]] = []
    for e in edges:
        tu, tv = node_type.get(e["source"]), node_type.get(e["target"])
        if {tu, tv} == {"printer", "router"}:
            printer_router_edges.append({
                "source": e["source"],
                "target": e["target"],
                "source_type": tu,
                "target_type": tv,
                "bandwidth": e["bandwidth"],
                "throughput": e["throughput"],
            })
    issue_b_present = len(printer_router_edges) > 0

    security_issues = (1 if issue_a_present else 0) + (1 if issue_b_present else 0)

    # --- Heuristic score: start at 1.0 (ideal) and subtract per risk, clamped
    # to [0, 1]. Matches generate_data.calculate_heuristics exactly. ---
    score = 1.0
    if bottleneck_present:
        score -= 0.3
    if issue_a_present:
        score -= 0.4
    if issue_b_present:
        score -= 0.2
    score = max(0.0, min(1.0, score))

    return {
        "score": round(score, 3),
        "bottleneck": {
            "present": bottleneck_present,
            "offending_nodes": offending_nodes,
        },
        "security": {
            "class": int(security_issues),
            "issue_a": {
                "present": issue_a_present,
                "label": "Unprotected hosts (no firewall and no IDS/IPS in topology)",
                "affected_host_ids": host_ids,
                "explanation": (
                    f"{len(host_ids)} host node(s) are present but the topology "
                    f"contains no firewall and no ids_ips device -- hosts have "
                    f"no filtering or intrusion monitoring."
                ) if issue_a_present else (
                    "Hosts are present but a firewall and/or ids_ips device "
                    "exists in the topology."
                ),
            },
            "issue_b": {
                "present": issue_b_present,
                "label": "Printer directly connected to a router",
                "edges": printer_router_edges,
                "explanation": (
                    f"{len(printer_router_edges)} printer<->router edge(s) bypass "
                    f"the switch/firewall control point -- printers should sit "
                    f"behind a switch, not on a router-facing segment."
                ) if issue_b_present else (
                    "No printer is directly connected to a router."
                ),
            },
        },
        "counts": {
            "nodes": len(nodes),
            "edges": len(edges),
            "by_type": {t: sum(1 for n in nodes if n["type"] == t) for t in NODE_TYPES if any(n["type"] == t for n in nodes)},
        },
    }


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