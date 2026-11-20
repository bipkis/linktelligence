"""FastAPI backend wrapping the trained network-topology GNN.

Exposes the GNN as an expertise service: submit a LAN topology as JSON
(`{nodes, edges}`), get back the model's predictions (score / bottleneck /
security_flaw) together with an actionable heuristic explanation (which node
is overloaded, which hosts are unprotected, which printer<->router edges
violate segmentation).

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
from pydantic import BaseModel, Field

from dataset import NODE_TYPES, build_graph_data
from model import NetworkGNN, SECURITY_CLASSES
from topology_analysis import analyze_topology, summarize

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "best_model.pt")
DATA_DIR = os.path.join(HERE, "data")
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
    id: int
    type: str


class Edge(BaseModel):
    source: int
    target: int
    bandwidth: int
    throughput: int


class TopologyRequest(BaseModel):
    nodes: List[Node] = Field(..., min_length=1)
    edges: List[Edge] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None  # ignored -- labels are predicted, not read


app = FastAPI(
    title="Linktelligence GNN Expertise API",
    description="Submit a LAN topology, get model predictions + heuristic analysis.",
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
            "probability": round(float(bn_probs[1].item()), 3),  # P(bottleneck present)
        },
        "security": {
            "predicted_class": int(sec_probs.argmax().item()),
            "label": SECURITY_LABELS.get(int(sec_probs.argmax().item()), "?"),
            "probabilities": [round(float(p), 3) for p in sec_probs.tolist()],
        },
    }


@app.post("/api/analyze")
def analyze(req: TopologyRequest) -> JSONResponse:
    bad = _validate_types(req.nodes)
    if bad:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown node type(s): {bad}. Allowed: {NODE_TYPES}",
        )

    # Edge endpoints must reference existing node ids.
    ids = {n.id for n in req.nodes}
    dangling = [
        {"edge": e.model_dump(), "missing": {e.source, e.target} - ids}
        for e in req.edges
        if e.source not in ids or e.target not in ids
    ]
    if dangling:
        raise HTTPException(
            status_code=422,
            detail=f"Edge endpoint(s) reference unknown node id(s): {dangling}",
        )

    nodes = [n.model_dump() for n in req.nodes]
    edges = [e.model_dump() for e in req.edges]

    model_out = _run_model(req.nodes, req.edges)
    heuristic = analyze_topology(nodes, edges)
    summary = summarize(heuristic)

    return JSONResponse({
        "model": model_out,
        "heuristic": heuristic,
        "summary": summary,
        "topology": {"nodes": nodes, "edges": edges},
    })


def _pick_sample() -> Optional[Dict[str, Any]]:
    """Return an interesting sample topology (security_flaw==2 preferred).

    Scans data/ once for a sample that exercises both security issues; falls
    back to the first sample. Returns the raw JSON (nodes, edges, metadata).
    """
    if not os.path.isdir(DATA_DIR):
        return None
    fallback = None
    for name in sorted(os.listdir(DATA_DIR)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(DATA_DIR, name)) as f:
            d = json.load(f)
        if fallback is None:
            fallback = d
        meta = d.get("metadata", {})
        # Prefer a sample showing both security issues (class 2) + a bottleneck.
        if meta.get("security_flaw") == 2 and meta.get("bottleneck") == 1:
            d["_source_file"] = name
            return d
    if fallback is not None:
        fallback["_source_file"] = "sample_0001.json"
    return fallback


@app.get("/api/sample")
def sample() -> JSONResponse:
    """A demo topology the frontend can load into the editor."""
    d = _pick_sample()
    if d is None:
        raise HTTPException(status_code=404, detail="no sample data available")
    meta = d.get("metadata")
    return JSONResponse({
        "nodes": d["nodes"],
        "edges": d["edges"],
        "metadata": meta,  # known labels, for reference only
        "source_file": d.get("_source_file"),
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