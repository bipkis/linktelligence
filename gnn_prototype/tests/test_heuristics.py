
from heuristics import analyze_topology, calculate_labels
from topology_analysis import build_diagnostics


def test_multiple_bottlenecks_apply_penalty_only_once():
    nodes = [
        {"id": 1, "type": "router"},
        {"id": 2, "type": "router"},
    ]

    edges = [
        {
            "source": 1,
            "target": 2,
            "bandwidth": 100,
            "throughput": 200,
        }
    ]

    report = analyze_topology(nodes, edges)

    assert report["bottleneck"]["present"] is True
    assert len(report["bottleneck"]["offending_nodes"]) == 2

    # Two overloaded routers, but one bottleneck category.
    assert report["score"] == 0.7


def test_security_flaw_counts_distinct_categories():
    nodes = [
        {"id": 1, "type": "router"},
        {"id": 2, "type": "printer"},
        {"id": 3, "type": "printer"},
        {"id": 4, "type": "host"},
    ]

    edges = [
        {
            "source": 1,
            "target": 2,
            "bandwidth": 1000,
            "throughput": 100,
        },
        {
            "source": 1,
            "target": 3,
            "bandwidth": 1000,
            "throughput": 100,
        },
    ]

    report = analyze_topology(nodes, edges)

    assert report["security"]["class"] == 2

    # Two printer-router edges still represent one issue category.
    assert len(report["security"]["issue_b"]["edges"]) == 2

    assert report["score"] == 0.4


def test_training_labels_match_analysis_report():
    nodes = [
        {"id": 1, "type": "router"},
        {"id": 2, "type": "host"},
    ]

    edges = [
        {
            "source": 1,
            "target": 2,
            "bandwidth": 100,
            "throughput": 200,
        }
    ]

    report = analyze_topology(nodes, edges)
    labels = calculate_labels(nodes, edges)

    assert labels["score"] == report["score"]
    assert labels["bottleneck"] == int(
        report["bottleneck"]["present"]
    )
    assert labels["security_flaw"] == report["security"]["class"]



def test_diagnostics_localize_printer_router_connection():
    nodes = [
        {"id": 0, "type": "printer"},
        {"id": 1, "type": "router"},
        {"id": 2, "type": "firewall"},
        {"id": 3, "type": "ids_ips"},
    ]
    edges = [
        {"source": 0, "target": 1, "bandwidth": 1000, "throughput": 300},
        {"source": 0, "target": 2, "bandwidth": 1000, "throughput": 300},
    ]

    result = build_diagnostics(nodes, edges)
    issue = next(
        item for item in result["issues"]
        if item["code"] == "printer_router_connection"
    )

    assert "Устройство 0 (printer)" in issue["description"]
    assert "Устройство 1 (router)" in issue["description"]
    assert issue["recommendation"]
    assert result["rule_results"]["security_class"] == 1


def test_diagnostics_marks_host_check_not_applicable_when_no_hosts():
    nodes = [{"id": 1, "type": "router"}]
    result = build_diagnostics(nodes, [])

    host_check = next(
        item for item in result["positive_results"]
        if item["code"] == "no_unprotected_hosts_by_current_rule"
    )
    assert "Host" in host_check["description"]
    assert "отсутствуют" in host_check["description"]
