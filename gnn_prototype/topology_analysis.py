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


def build_diagnostics(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build actionable rule-based diagnostics without changing GNN outputs.

    The rules are the same deterministic rules used to generate training
    labels. They explain observable topology patterns; they do not prove
    that a device is correctly configured or that a network is secure.
    """
    report = analyze_topology(nodes, edges)
    node_by_id = {node["id"]: node for node in nodes}

    firewall_ids = sorted(
        node["id"] for node in nodes if node["type"] == "firewall"
    )
    ids_ips_ids = sorted(
        node["id"] for node in nodes if node["type"] == "ids_ips"
    )
    host_ids = sorted(
        node["id"] for node in nodes if node["type"] == "host"
    )
    critical_nodes = [
        node for node in nodes
        if node["type"] in {"router", "firewall"}
    ]

    issues: List[Dict[str, Any]] = []
    positive_results: List[Dict[str, Any]] = []

    # Bottleneck diagnostics.
    bottleneck = report["bottleneck"]
    if bottleneck["present"]:
        offending = bottleneck["offending_nodes"]
        details = [
            (
                f"Устройство {node['id']} ({node['type']}): "
                f"throughput {node['throughput']} превышает "
                f"bandwidth {node['bandwidth']} "
                f"на {node['overshoot']}."
            )
            for node in offending
        ]
        issues.append({
            "category": "bottleneck",
            "code": "critical_node_overload",
            "title": "Перегрузка критических устройств",
            "description": "\n".join(details),
            "recommendation": (
                "Проверьте фактическую загрузку интерфейсов, доступную "
                "пропускную способность каналов и распределение трафика. "
                "При необходимости рассмотрите увеличение пропускной "
                "способности или перераспределение нагрузки."
            ),
            "affected_nodes": [
                {"id": node["id"], "type": node["type"]}
                for node in offending
            ],
            "affected_edges": [],
        })
    elif critical_nodes:
        positive_results.append({
            "category": "bottleneck",
            "code": "no_critical_node_overload",
            "title": "Признаки перегрузки не обнаружены",
            "description": (
                "По текущему правилу суммарный throughput не превышает "
                "суммарную bandwidth ни у одного Router или Firewall."
            ),
        })
    else:
        positive_results.append({
            "category": "bottleneck",
            "code": "bottleneck_check_not_applicable",
            "title": "Проверка bottleneck неприменима",
            "description": (
                "В топологии нет устройств Router или Firewall, "
                "для которых предусмотрена эта проверка."
            ),
        })

    # Security issue A: hosts without firewall and IDS/IPS.
    issue_a = report["security"]["issue_a"]
    if issue_a["present"]:
        issues.append({
            "category": "security",
            "code": "unprotected_hosts",
            "title": "Хосты без Firewall и IDS/IPS",
            "description": (
                f"Узлы Host с ID {', '.join(map(str, host_ids))} "
                "присутствуют, но в топологии нет ни Firewall, ни IDS/IPS."
            ),
            "recommendation": (
                "Проверьте архитектуру защиты пользовательских узлов. "
                "Рассмотрите размещение межсетевого экрана и/или IDS/IPS "
                "в соответствующих сегментах и проверьте маршруты трафика."
            ),
            "affected_nodes": [
                {"id": node_id, "type": "host"}
                for node_id in host_ids
            ],
            "affected_edges": [],
        })
    else:
        if not host_ids:
            protection_devices = []
            if firewall_ids:
                protection_devices.append(
                    "Firewall (ID " + ", ".join(map(str, firewall_ids)) + ")"
                )
            if ids_ips_ids:
                protection_devices.append(
                    "IDS/IPS (ID " + ", ".join(map(str, ids_ips_ids)) + ")"
                )
            device_note = (
                " В топологии присутствуют " + " и ".join(protection_devices) + "."
                if protection_devices
                else " Устройства Firewall и IDS/IPS также отсутствуют."
            )
            description = (
                "Узлы типа Host в топологии отсутствуют, поэтому правило "
                "поиска незащищённых хостов не срабатывает."
                + device_note
            )
        else:
            present_devices = []
            if firewall_ids:
                present_devices.append(
                    "Firewall (ID " + ", ".join(map(str, firewall_ids)) + ")"
                )
            if ids_ips_ids:
                present_devices.append(
                    "IDS/IPS (ID " + ", ".join(map(str, ids_ips_ids)) + ")"
                )
            description = (
                "В топологии присутствуют узлы Host и "
                + " и/или ".join(present_devices)
                + ". По текущему правилу категория незащищённых хостов "
                "не фиксируется. Проверка не оценивает конфигурацию этих "
                "устройств, их доступность и фактическое покрытие трафика."
            )

        positive_results.append({
            "category": "security",
            "code": "no_unprotected_hosts_by_current_rule",
            "title": (
                "Проверка хостов неприменима: Host отсутствуют"
                if not host_ids
                else "Незащищённые хосты не выявлены по текущему правилу"
            ),
            "description": description,
        })

    # Security issue B: direct printer-router links.
    issue_b_edges = report["security"]["issue_b"]["edges"]
    if issue_b_edges:
        edge_descriptions = []
        affected_nodes_by_id = {}
        for edge in issue_b_edges:
            source_id = edge["source"]
            target_id = edge["target"]
            source_type = node_by_id[source_id]["type"]
            target_type = node_by_id[target_id]["type"]
            edge_descriptions.append(
                f"Устройство {source_id} ({source_type}) напрямую "
                f"подключено к устройству {target_id} ({target_type})."
            )
            affected_nodes_by_id[source_id] = {
                "id": source_id,
                "type": source_type,
            }
            affected_nodes_by_id[target_id] = {
                "id": target_id,
                "type": target_type,
            }

        issues.append({
            "category": "security",
            "code": "printer_router_connection",
            "title": "Прямая связь принтера с маршрутизатором",
            "description": "\n".join(edge_descriptions),
            "recommendation": (
                "Проверьте необходимость такого подключения и рассмотрите "
                "сегментацию принтера через коммутатор, а также настройку "
                "соответствующих правил доступа."
            ),
            "affected_nodes": list(affected_nodes_by_id.values()),
            "affected_edges": issue_b_edges,
        })
    else:
        positive_results.append({
            "category": "security",
            "code": "no_direct_printer_router_connection",
            "title": "Прямых соединений Printer ↔ Router не обнаружено",
            "description": (
                "В топологии нет прямых рёбер между принтерами и маршрутизаторами."
            ),
        })

    return {
        "source": "heuristics.py",
        "disclaimer": (
            "Это отдельная диагностика по детерминированным правилам проекта, "
            "а не объяснение внутренних решений GNN. Наличие Firewall или "
            "IDS/IPS в графе само по себе не подтверждает корректность их "
            "конфигурации и защиты сети."
        ),
        "issues": issues,
        "positive_results": positive_results,
        "rule_results": {
            "bottleneck": bool(bottleneck["present"]),
            "security_class": int(report["security"]["class"]),
        },
        "counts": {
            "issues": len(issues),
            "positive_results": len(positive_results),
        },
    }


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