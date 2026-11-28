"""FastAPI backend wrapping the trained network-topology GNN.

Exposes the GNN as an expertise service: submit a LAN topology as JSON
(`{nodes, edges}`), get back the model's predictions (score / bottleneck /
security_flaw).

Run:
    python server.py                 # http://localhost:8000
    uvicorn server:app --reload      # dev

The feature pipeline is shared with training via `dataset.build_graph_data`,
so what the model sees at serve time is exactly what it was trained on.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from dataset import NODE_TYPES, build_graph_data
from model import NetworkGNN, SECURITY_CLASSES
from topology_analysis import build_diagnostics


MAX_NODES = 100
MAX_EDGES = 500

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "best_model.pt")
DEMO_SAMPLE_PATH = os.path.join(
    HERE,
    "demo",
    "sample_topology.json",
)
FRONTEND = os.path.join(HERE, "static", "index.html")

SECURITY_LABELS = {0: "no topology security issues", 1: "one security issue", 2: "two security issues"}

# --- Load the trained model once at import time (ready when uvicorn boots). ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"best_model.pt not found at {MODEL_PATH}. Run `python train.py` first."
    )
MODEL = NetworkGNN.from_pretrained(MODEL_PATH)
print(f"[server] loaded model from {MODEL_PATH}")


# --- Request / response schemas ------------------------------------------------

class Node(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int = Field(strict=True)
    type: str


class Edge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: int = Field(strict=True)
    target: int = Field(strict=True)

    bandwidth: int = Field(gt=0, strict=True)
    throughput: int = Field(ge=0, strict=True)


class TopologyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nodes: List[Node] = Field(
        min_length=1,
        max_length=MAX_NODES,
    )

    edges: List[Edge] = Field(
        default_factory=list,
        max_length=MAX_EDGES,
    )


    @model_validator(mode="after")
    def validate_topology(self) -> "TopologyRequest":
        """Validate graph-level constraints."""

        # 1. Check node ID uniqueness.
        node_ids = [node.id for node in self.nodes]

        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Duplicate node IDs are not allowed")

        known_ids = set(node_ids)

        # 2. Validate edge endpoints and topology structure.
        seen_edges = set()

        for edge in self.edges:
            source = edge.source
            target = edge.target

            # Both endpoints must exist.
            if source not in known_ids or target not in known_ids:
                raise ValueError(
                    f"Edge references unknown node: "
                    f"{source} -> {target}"
                )

            # No self-loops.
            if source == target:
                raise ValueError(
                    f"Self-loop is not allowed: node {source}"
                )

            # Undirected edge canonicalization.
            edge_key = (min(source, target), max(source, target))

            if edge_key in seen_edges:
                raise ValueError(
                    f"Duplicate edge is not allowed: "
                    f"{source} <-> {target}"
                )

            seen_edges.add(edge_key)

        return self



app = FastAPI(
    title="Linktelligence GNN Expertise API",
    description="Submit a LAN topology and get GNN model predictions.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # prototype: the frontend is a static file, may be opened from file://
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "model_loaded": True, "security_classes": SECURITY_CLASSES}


def _validate_types(nodes: List[Node]) -> List[str]:
    bad = sorted({n.type for n in nodes if n.type not in NODE_TYPES})
    return bad


def _run_model(nodes_raw: List[Node], edges_raw: List[Edge]) -> Dict[str, Any]:
    """Build features, run the GNN, return predictions + probabilities."""
    nodes = [n.model_dump() for n in nodes_raw]
    edges = [e.model_dump() for e in edges_raw]
    g = build_graph_data(nodes, edges)

    with torch.no_grad():
        score, bottleneck, security = MODEL(
            g["x"], g["edge_index"], g["edge_attr"], torch.zeros(g["x"].size(0), dtype=torch.long)
        )

    score_v = float(torch.clamp(score[0, 0], 0.0, 1.0).item())
    bn_probs = torch.softmax(bottleneck[0], dim=0)
    sec_probs = torch.softmax(security[0], dim=0)
    return {
        "score": round(score_v, 3),
        "bottleneck": {
            "predicted": int(bn_probs.argmax().item()),
            "probability": round(float(bn_probs[1].item()), 6),  # P(bottleneck present)
        },
        "security": {
            "predicted_class": int(sec_probs.argmax().item()),
            "label": SECURITY_LABELS.get(int(sec_probs.argmax().item()), "?"),
            "probabilities": [round(float(p), 6) for p in sec_probs.tolist()],
        },
    }


def summarize_predictions(model_out: Dict[str, Any]) -> str:
    """Generate a textual summary exclusively from GNN predictions."""

    score = model_out["score"]

    bottleneck = model_out["bottleneck"]
    security = model_out["security"]

    bn_probability = bottleneck["probability"]
    bn_predicted = bottleneck["predicted"]

    security_class = security["predicted_class"]
    security_label = security["label"]

    bottleneck_text = (
        "GNN обнаружила признаки bottleneck."
        if bn_predicted == 1
        else "GNN не обнаружила признаки bottleneck."
    )

    return (
        f"Оценка качества по GNN: {score:.2f}/1.00. "
        f"{bottleneck_text} "
        f"P(bottleneck)={bn_probability:.1%}. "
        f"Класс безопасности: {security_class} — {security_label}."
    )

@app.post("/api/analyze")
def analyze(req: TopologyRequest) -> JSONResponse:
    """Analyze topology exclusively using the trained GNN."""

    bad = _validate_types(req.nodes)

    if bad:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown node type(s): {bad}. Allowed: {NODE_TYPES}",
        )

    nodes = [
        node.model_dump()
        for node in req.nodes
    ]

    edges = [
        edge.model_dump()
        for edge in req.edges
    ]

    # The only analysis engine.
    model_out = _run_model(req.nodes, req.edges)

    # Text is generated from model predictions only.
    summary = summarize_predictions(model_out)

    diagnostics = build_diagnostics(nodes, edges)
    rule_results = diagnostics["rule_results"]
    diagnostics["comparison"] = {
        "bottleneck": {
            "gnn": model_out["bottleneck"]["predicted"] == 1,
            "rules": rule_results["bottleneck"],
            "matches": (
                (model_out["bottleneck"]["predicted"] == 1)
                == rule_results["bottleneck"]
            ),
        },
        "security_class": {
            "gnn": model_out["security"]["predicted_class"],
            "rules": rule_results["security_class"],
            "matches": (
                model_out["security"]["predicted_class"]
                == rule_results["security_class"]
            ),
        },
    }

    return JSONResponse({
        "model": model_out,
        "summary": summary,
        "diagnostics": diagnostics,
        "topology": {
            "nodes": nodes,
            "edges": edges,
        },
    })

def _pick_sample() -> Optional[Dict[str, Any]]:
    """Load the dedicated demo topology without training metadata."""

    if not os.path.isfile(DEMO_SAMPLE_PATH):
        return None

    try:
        with open(DEMO_SAMPLE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(data, dict):
        return None

    nodes = data.get("nodes")
    edges = data.get("edges")

    if not isinstance(nodes, list) or not isinstance(edges, list):
        return None

    return {
        "nodes": nodes,
        "edges": edges,
        "_source_file": os.path.basename(DEMO_SAMPLE_PATH),
    }


@app.get("/api/sample")
def sample() -> JSONResponse:
    """Return a demo topology without exposing training labels."""

    data = _pick_sample()

    if data is None:
        raise HTTPException(
            status_code=404,
            detail="No sample data available",
        )

    return JSONResponse({
        "nodes": data["nodes"],
        "edges": data["edges"],
        "source_file": data["_source_file"],
    })


@app.get("/")
def index():
    """Serve the frontend from static/index.html.

    If the file is missing (e.g. only server.py was copied without the
    static/ dir), return a clear HTML message instead of crashing with 500,
    so the API still works and the operator knows exactly what to copy.
    """
    if os.path.exists(FRONTEND):
        return FileResponse(FRONTEND, media_type="text/html")
    return HTMLResponse(
        "<!DOCTYPE html><html lang='ru'><head><meta charset='utf-8'>"
        "<title>Linktelligence — фронтенд не найден</title></head>"
        "<body style='font-family:sans-serif;max-width:640px;margin:2rem auto;padding:1rem'>"
        "<h2>Фронтенд не найден</h2>"
        f"<p>Файл <code>{FRONTEND}</code> отсутствует. API (<code>/api/analyze</code>, "
        "<code>/api/sample</code>) работает, но UI не отдаётся.</p>"
        "<p>Скопируйте каталог <code>static/</code> (с <code>index.html</code>) рядом с "
        "<code>server.py</code> и перезагрузите страницу.</p>"
        "<p><a href='/api/health'>/api/health</a> &middot; <a href='/api/sample'>/api/sample</a></p>"
        "</body></html>",
        status_code=200,
    )


@app.get("/favicon.ico")
def favicon():
    # Tiny inline SVG favicon so browsers don't spam 404s in the log.
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
        '<circle cx="16" cy="16" r="13" fill="#3b82f6"/>'
        '<circle cx="16" cy="16" r="5" fill="#0f172a"/></svg>'
    )
    return HTMLResponse(svg, headers={"Content-Type": "image/svg+xml"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)