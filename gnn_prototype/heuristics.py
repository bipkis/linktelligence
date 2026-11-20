
"""Shared heuristic engine for LAN topology analysis.

This module is the single source of truth for:
- bottleneck detection;
- topology security issue detection;
- heuristic score calculation;
- labels used to generate the training dataset.

Used by:
- generate_data.py — generating ML targets;
- topology_analysis.py — producing detailed analysis for API.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence


Node = Mapping[str, Any]
Edge = Mapping[str, Any]


NODE_TYPES = (
    "router",
    "switch",
    "printer",
    "host",
    "firewall",
    "ids_ips",
    "l3_switch",
)

CRITICAL_TYPES = frozenset({"firewall", "router"})

# Penalty constants are shared between dataset generation and inference.
BOTTLENECK_PENALTY = 0.3
UNPROTECTED_HOSTS_PENALTY = 0.4
PRINTER_ROUTER_PENALTY = 0.2


def analyze_topology(
    nodes: Sequence[Node],
    edges: Sequence[Edge],
) -> Dict[str, Any]:
    """Analyze a LAN topology and return a detailed heuristic report.

    Bottleneck:
        A critical node (router/firewall) is overloaded when its
        total incident throughput exceeds its total incident bandwidth.

    Security issue A:
        At least one host exists, but the topology contains neither
        a firewall nor an IDS/IPS device.

    Security issue B:
        At least one direct printer-router connection exists.

    Score:
        Starts at 1.0. Each detected issue category applies its
        penalty only once, regardless of the number of affected nodes
        or edges. The final score is clamped to [0.0, 1.0].

    Security class:
        Number of distinct security issue categories, in {0, 1, 2}.

    Preconditions:
        - node IDs are unique;
        - all edge endpoints reference existing nodes;
        - bandwidth and throughput are non-negative;
        - input node types belong to NODE_TYPES.
    """

    if not nodes:
        raise ValueError("Topology must contain at least one node")

    node_types = {
        node["id"]: node["type"]
        for node in nodes
    }

    # ---------------------------------------------------------
    # 1. Calculate per-node aggregates
    # ---------------------------------------------------------

    degree: Dict[int, int] = {
        node_id: 0 for node_id in node_types
    }

    incident_bandwidth: Dict[int, float] = {
        node_id: 0.0 for node_id in node_types
    }

    incident_throughput: Dict[int, float] = {
        node_id: 0.0 for node_id in node_types
    }

    for edge in edges:
        source = edge["source"]
        target = edge["target"]

        bandwidth = float(edge["bandwidth"])
        throughput = float(edge["throughput"])

        degree[source] += 1
        degree[target] += 1

        incident_bandwidth[source] += bandwidth
        incident_bandwidth[target] += bandwidth

        incident_throughput[source] += throughput
        incident_throughput[target] += throughput

    # ---------------------------------------------------------
    # 2. Bottleneck detection
    # ---------------------------------------------------------

    offending_nodes: List[Dict[str, Any]] = []

    for node_id in sorted(node_types):
        node_type = node_types[node_id]

        if node_type not in CRITICAL_TYPES:
            continue

        bandwidth = incident_bandwidth[node_id]
        throughput = incident_throughput[node_id]

        if throughput > bandwidth:
            ratio = (
                round(throughput / bandwidth, 3)
                if bandwidth > 0
                else None
            )

            offending_nodes.append({
                "id": node_id,
                "type": node_type,
                "throughput": int(throughput),
                "bandwidth": int(bandwidth),
                "ratio": ratio,
                "overshoot": int(throughput - bandwidth),
                "degree": degree[node_id],
            })

    bottleneck_present = bool(offending_nodes)

    # ---------------------------------------------------------
    # 3. Security issue A:
    #    Hosts without firewall and IDS/IPS
    # ---------------------------------------------------------

    types_present = set(node_types.values())

    has_host = "host" in types_present
    has_firewall = "firewall" in types_present
    has_ids_ips = "ids_ips" in types_present

    issue_a_present = (
        has_host
        and not has_firewall
        and not has_ids_ips
    )

    host_ids = sorted(
        node_id
        for node_id, node_type in node_types.items()
        if node_type == "host"
    )

    # ---------------------------------------------------------
    # 4. Security issue B:
    #    Direct printer-router connections
    # ---------------------------------------------------------

    printer_router_edges: List[Dict[str, Any]] = []

    for edge in edges:
        source = edge["source"]
        target = edge["target"]

        source_type = node_types[source]
        target_type = node_types[target]

        if {source_type, target_type} == {"printer", "router"}:
            printer_router_edges.append({
                "source": source,
                "target": target,
                "source_type": source_type,
                "target_type": target_type,
                "bandwidth": edge["bandwidth"],
                "throughput": edge["throughput"],
            })

    issue_b_present = bool(printer_router_edges)

    # Security class represents distinct issue categories,
    # NOT the number of affected nodes or edges.
    security_issues = (
        int(issue_a_present)
        + int(issue_b_present)
    )

    # ---------------------------------------------------------
    # 5. Unified score calculation
    # ---------------------------------------------------------

    score = 1.0

    if bottleneck_present:
        score -= BOTTLENECK_PENALTY

    if issue_a_present:
        score -= UNPROTECTED_HOSTS_PENALTY

    if issue_b_present:
        score -= PRINTER_ROUTER_PENALTY

    score = max(0.0, min(1.0, score))
    score = round(score, 3)

    # ---------------------------------------------------------
    # 6. Topology statistics
    # ---------------------------------------------------------

    by_type = {
        node_type: sum(
            1
            for current_type in node_types.values()
            if current_type == node_type
        )
        for node_type in NODE_TYPES
        if node_type in types_present
    }

    # ---------------------------------------------------------
    # 7. Detailed report
    # ---------------------------------------------------------

    return {
        "score": score,

        "bottleneck": {
            "present": bottleneck_present,
            "offending_nodes": offending_nodes,
        },

        "security": {
            "class": security_issues,

            "issue_a": {
                "present": issue_a_present,
                "label": (
                    "Unprotected hosts "
                    "(no firewall and no IDS/IPS in topology)"
                ),
                "affected_host_ids": host_ids,
                "explanation": (
                    f"{len(host_ids)} host node(s) are present but "
                    "the topology contains no firewall and no "
                    "ids_ips device -- hosts have no filtering "
                    "or intrusion monitoring."
                ) if issue_a_present else (
                    "Hosts are present but a firewall and/or "
                    "ids_ips device exists in the topology."
                ),
            },

            "issue_b": {
                "present": issue_b_present,
                "label": "Printer directly connected to a router",
                "edges": printer_router_edges,
                "explanation": (
                    f"{len(printer_router_edges)} printer<->router "
                    "edge(s) bypass the switch/firewall control "
                    "point -- printers should sit behind a switch, "
                    "not on a router-facing segment."
                ) if issue_b_present else (
                    "No printer is directly connected to a router."
                ),
            },
        },

        "counts": {
            "nodes": len(nodes),
            "edges": len(edges),
            "by_type": by_type,
        },
    }


def calculate_labels(
    nodes: Sequence[Node],
    edges: Sequence[Edge],
) -> Dict[str, float | int]:
    """Return ML training labels using the shared heuristic engine.

    This function is intended for generate_data.py.

    Returns:
        {
            "score": float,
            "bottleneck": int,
            "security_flaw": int,
        }
    """

    report = analyze_topology(nodes, edges)

    return {
        "score": report["score"],
        "bottleneck": int(report["bottleneck"]["present"]),
        "security_flaw": report["security"]["class"],
    }
