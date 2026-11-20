
import pytest
from pydantic import ValidationError

from server import TopologyRequest


def valid_payload():
    return {
        "nodes": [
            {"id": 1, "type": "router"},
            {"id": 2, "type": "host"},
        ],
        "edges": [
            {
                "source": 1,
                "target": 2,
                "bandwidth": 1000,
                "throughput": 500,
            }
        ],
    }


def test_valid_topology():
    request = TopologyRequest(**valid_payload())

    assert len(request.nodes) == 2
    assert len(request.edges) == 1


def test_topology_without_edges():
    payload = valid_payload()
    payload["edges"] = []

    request = TopologyRequest(**payload)

    assert len(request.edges) == 0


def test_duplicate_node_ids():
    payload = valid_payload()
    payload["nodes"].append(
        {"id": 1, "type": "host"}
    )

    with pytest.raises(ValidationError):
        TopologyRequest(**payload)


def test_unknown_edge_endpoint():
    payload = valid_payload()
    payload["edges"][0]["target"] = 999

    with pytest.raises(ValidationError):
        TopologyRequest(**payload)


def test_self_loop():
    payload = valid_payload()
    payload["edges"][0]["target"] = 1

    with pytest.raises(ValidationError):
        TopologyRequest(**payload)


def test_duplicate_undirected_edge():
    payload = valid_payload()
    payload["edges"].append({
        "source": 2,
        "target": 1,
        "bandwidth": 2000,
        "throughput": 100,
    })

    with pytest.raises(ValidationError):
        TopologyRequest(**payload)


@pytest.mark.parametrize("bandwidth", [0, -1, -100])
def test_invalid_bandwidth(bandwidth):
    payload = valid_payload()
    payload["edges"][0]["bandwidth"] = bandwidth

    with pytest.raises(ValidationError):
        TopologyRequest(**payload)


@pytest.mark.parametrize("throughput", [-1, -100])
def test_invalid_throughput(throughput):
    payload = valid_payload()
    payload["edges"][0]["throughput"] = throughput

    with pytest.raises(ValidationError):
        TopologyRequest(**payload)


def test_empty_nodes():
    payload = valid_payload()
    payload["nodes"] = []

    with pytest.raises(ValidationError):
        TopologyRequest(**payload)


def test_too_many_nodes():
    payload = valid_payload()
    payload["nodes"] = [
        {"id": i, "type": "host"}
        for i in range(101)
    ]

    with pytest.raises(ValidationError):
        TopologyRequest(**payload)


def test_too_many_edges():
    payload = valid_payload()
    payload["nodes"] = [
        {"id": i, "type": "host"}
        for i in range(33)
    ]

    payload["edges"] = [
        {
            "source": i,
            "target": j,
            "bandwidth": 1000,
            "throughput": 100,
        }
        for i in range(33)
        for j in range(i + 1, 33)
    ]

    with pytest.raises(ValidationError):
        TopologyRequest(**payload)
