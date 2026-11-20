
from heuristics import analyze_topology, calculate_labels


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
