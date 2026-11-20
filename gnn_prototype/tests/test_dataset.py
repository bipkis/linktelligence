import torch

from dataset import build_graph_data, TYPE_TO_INDEX


import json

import pytest
import torch

from dataset import (
    NetworkDataset,
    TYPE_TO_INDEX,
    build_graph_data,
)


def sample_nodes():
    return [
        {"id": 20, "type": "router"},
        {"id": 10, "type": "host"},
    ]


def sample_edges():
    return [
        {
            "source": 20,
            "target": 10,
            "bandwidth": 1000,
            "throughput": 500,
        }
    ]


def test_node_order_does_not_affect_feature_mapping():
    """Features must follow sorted node IDs."""

    graph = build_graph_data(
        sample_nodes(),
        sample_edges(),
    )

    assert graph["node_ids"] == [10, 20]

    assert graph["node_map"] == {
        10: 0,
        20: 1,
    }

    # Index 0 must describe host (ID 10).
    assert graph["x"][0, TYPE_TO_INDEX["host"]].item() == 1.0

    # Index 1 must describe router (ID 20).
    assert graph["x"][1, TYPE_TO_INDEX["router"]].item() == 1.0


def test_node_permutation_produces_same_graph():
    """Changing JSON node order must not change graph tensors."""

    nodes = sample_nodes()
    edges = sample_edges()

    graph1 = build_graph_data(nodes, edges)

    graph2 = build_graph_data(
        list(reversed(nodes)),
        edges,
    )

    assert torch.equal(graph1["x"], graph2["x"])
    assert torch.equal(graph1["edge_index"], graph2["edge_index"])
    assert torch.equal(graph1["edge_attr"], graph2["edge_attr"])


def test_graph_without_edges():
    """Isolated nodes must produce correctly shaped edge tensors."""

    graph = build_graph_data(
        [{"id": 1, "type": "router"}],
        [],
    )

    assert graph["x"].shape == (1, 12)
    assert graph["edge_index"].shape == (2, 0)
    assert graph["edge_attr"].shape == (0, 2)


def test_duplicate_node_ids_raise():
    """Duplicate node IDs must be rejected."""

    nodes = [
        {"id": 1, "type": "router"},
        {"id": 1, "type": "host"},
    ]

    with pytest.raises(ValueError, match="Duplicate node IDs"):
        build_graph_data(nodes, [])


def test_unknown_edge_endpoint_raises():
    """Edges must reference existing nodes."""

    nodes = [
        {"id": 1, "type": "router"},
        {"id": 2, "type": "host"},
    ]

    edges = [
        {
            "source": 1,
            "target": 999,
            "bandwidth": 1000,
            "throughput": 500,
        }
    ]

    with pytest.raises(ValueError, match="unknown node"):
        build_graph_data(nodes, edges)


def test_network_dataset_reads_labels_from_metadata(tmp_path):
    """Dataset must load graph-level labels from metadata."""

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    sample = {
        "sample_id": 1,
        "nodes": [
            {"id": 2, "type": "router"},
            {"id": 1, "type": "printer"},
            {"id": 3, "type": "host"},
        ],
        "edges": [
            {
                "source": 1,
                "target": 2,
                "bandwidth": 100,
                "throughput": 50,
            }
        ],
        "metadata": {
            "score": 0.4,
            "bottleneck": 0,
            "security_flaw": 2,
        },
    }

    with open(data_dir / "sample_0001.json", "w") as f:
        json.dump(sample, f)

    dataset = NetworkDataset(root=str(tmp_path))

    assert len(dataset) == 1

    graph = dataset[0]

    assert graph.x.shape == (3, 12)
    assert graph.y.shape == (1, 3)

    expected_labels = torch.tensor(
        [[0.4, 0.0, 2.0]],
        dtype=torch.float32,
    )

    assert torch.allclose(graph.y, expected_labels)


def test_node_order_does_not_affect_feature_mapping():
    nodes = [
        {"id": 20, "type": "router"},
        {"id": 10, "type": "host"},
    ]

    edges = [
        {
            "source": 20,
            "target": 10,
            "bandwidth": 1000,
            "throughput": 500,
        }
    ]

    graph = build_graph_data(nodes, edges)

    assert graph["node_ids"] == [10, 20]

    assert graph["node_map"] == {
        10: 0,
        20: 1,
    }

    # First row must describe host (ID 10).
    assert graph["x"][0, TYPE_TO_INDEX["host"]].item() == 1.0

    # Second row must describe router (ID 20).
    assert graph["x"][1, TYPE_TO_INDEX["router"]].item() == 1.0


def test_node_permutation_produces_same_graph():
    nodes = [
        {"id": 20, "type": "router"},
        {"id": 10, "type": "host"},
    ]

    edges = [
        {
            "source": 20,
            "target": 10,
            "bandwidth": 1000,
            "throughput": 500,
        }
    ]

    graph1 = build_graph_data(nodes, edges)
    graph2 = build_graph_data(list(reversed(nodes)), edges)

    assert torch.equal(graph1["x"], graph2["x"])
    assert torch.equal(graph1["edge_index"], graph2["edge_index"])
    assert torch.equal(graph1["edge_attr"], graph2["edge_attr"])

def test_graph_without_edges():
    nodes = [
        {"id": 1, "type": "router"},
    ]

    graph = build_graph_data(nodes, [])

    assert graph["x"].shape == (1, 12)
    assert graph["edge_index"].shape == (2, 0)
    assert graph["edge_attr"].shape == (0, 2)