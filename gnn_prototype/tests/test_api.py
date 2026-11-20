import json

import pytest
from fastapi.testclient import TestClient

import server


import json

import pytest
from fastapi.testclient import TestClient

import server


@pytest.fixture(scope="module")
def client():
    with TestClient(server.app) as test_client:
        yield test_client


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


def test_health_endpoint(client):
    response = client.get("/api/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_analyze_uses_only_gnn_predictions(client):
    response = client.post(
        "/api/analyze",
        json=valid_payload(),
    )

    assert response.status_code == 200

    data = response.json()

    #assert "model" in data
    assert "model" in data
    assert "heuristic" not in data
    assert "summary" in data
    assert "topology" in data

    assert data["model"]["bottleneck"]["predicted"] in (0, 1)
    assert data["model"]["security"]["predicted_class"] in (0, 1, 2)
    assert "summary" in data
    assert "topology" in data

    # No heuristic analysis in runtime response.
    #assert "heuristic" not in data

    model = data["model"]

    assert 0.0 <= model["score"] <= 1.0

    assert model["bottleneck"]["predicted"] in (0, 1)
    assert 0.0 <= model["bottleneck"]["probability"] <= 1.0

    assert model["security"]["predicted_class"] in (0, 1, 2)
    assert len(model["security"]["probabilities"]) == 3

    assert abs(
        sum(model["security"]["probabilities"]) - 1.0
    ) < 1e-5

    assert isinstance(data["summary"], str)


def test_analyze_topology_without_edges(client):
    payload = {
        "nodes": [
            {"id": 1, "type": "router"},
        ],
        "edges": [],
    }

    response = client.post(
        "/api/analyze",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert "model" in data
    assert "heuristic" not in data


def test_sample_does_not_expose_training_metadata(
    client,
    tmp_path,
    monkeypatch,
):
    sample_path = tmp_path / "sample_topology.json"

    sample = {
        "sample_id": 1,
        "nodes": [
            {"id": 1, "type": "router"},
            {"id": 2, "type": "host"},
        ],
        "edges": [],
        "metadata": {
            "score": 1.0,
            "bottleneck": 0,
            "security_flaw": 0,
        },
    }


    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(sample, f)

    monkeypatch.setattr(
        server,
        "DEMO_SAMPLE_PATH",
        str(sample_path),
    )

    response = client.get("/api/sample")

    assert response.status_code == 200

    data = response.json()

    assert "nodes" in data
    assert "edges" in data
    assert "source_file" in data

    assert "metadata" not in data


@pytest.fixture(scope="module")
def client():
    with TestClient(server.app) as test_client:
        yield test_client


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


def test_health_endpoint(client):
    response = client.get("/api/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["security_classes"] == 3


def test_analyze_valid_topology(client):
    response = client.post(
        "/api/analyze",
        json=valid_payload(),
    )

    assert response.status_code == 200

    data = response.json()

    assert "model" in data
    assert "heuristic" not in data
    assert "summary" in data
    assert "topology" in data

    assert 0.0 <= data["model"]["score"] <= 1.0
    assert data["model"]["bottleneck"]["predicted"] in (0, 1)
    assert data["model"]["security"]["predicted_class"] in (0, 1, 2)

    assert len(data["topology"]["nodes"]) == 2
    assert len(data["topology"]["edges"]) == 1

def test_analyze_topology_without_edges(client):
    payload = {
        "nodes": [
            {"id": 1, "type": "router"},
        ],
        "edges": [],
    }

    response = client.post(
        "/api/analyze",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert "model" in data
    assert "heuristic" not in data

    assert 0.0 <= data["model"]["score"] <= 1.0
    assert data["topology"]["edges"] == []

def test_unknown_node_type_returns_422(client):
    payload = valid_payload()
    payload["nodes"][1]["type"] = "workstation"

    response = client.post(
        "/api/analyze",
        json=payload,
    )

    assert response.status_code == 422
    assert "Unknown node type" in response.json()["detail"]

def test_sample_endpoint(client, tmp_path, monkeypatch):
    sample_path = tmp_path / "sample_topology.json"

    sample = {
        "sample_id": 1,
        "nodes": [
            {"id": 1, "type": "router"},
            {"id": 2, "type": "host"},
        ],
        "edges": [],
        "metadata": {
            "score": 1.0,
            "bottleneck": 0,
            "security_flaw": 0,
        },
    }

    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(sample, f)

    monkeypatch.setattr(
        server,
        "DEMO_SAMPLE_PATH",
        str(sample_path),
    )

    response = client.get("/api/sample")

    assert response.status_code == 200

    data = response.json()

    assert len(data["nodes"]) == 2
    assert data["source_file"] == "sample_topology.json"
    assert "metadata" not in data
    assert "nodes" in data
    assert "edges" in data