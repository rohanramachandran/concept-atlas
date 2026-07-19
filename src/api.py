"""FastAPI app: JSON endpoints for concept sets and graphs, plus the explorer.

Run with::

    uvicorn src.api:app --reload

API routes live under ``/api/*``; everything else is served from ``static/``,
so http://127.0.0.1:8000 opens the D3 explorer directly.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
CONCEPTS_DIR = ROOT / "concepts"
GRAPH_PATH = ROOT / "static" / "graph.json"

app = FastAPI(
    title="concept-atlas",
    description="The knowledge graph of the model, not for it.",
    version="0.1.0",
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/concepts")
def list_concepts() -> list[dict]:
    """Summaries of every seed concept set in ``concepts/``."""
    sets = []
    for path in sorted(CONCEPTS_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        sets.append(
            {
                "name": data["name"],
                "n_items": len(data["items"]),
                "n_templates": len(data["templates"]),
                "items": data["items"],
            }
        )
    return sets


@app.get("/api/concepts/{name}")
def get_concept(name: str) -> dict:
    path = CONCEPTS_DIR / f"{name}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"no concept set named {name!r}")
    return json.loads(path.read_text())


@app.get("/api/graph")
def get_graph() -> dict:
    """The current extracted graph (demo data until you run an extraction)."""
    if not GRAPH_PATH.exists():
        raise HTTPException(status_code=404, detail="no graph has been extracted yet")
    return json.loads(GRAPH_PATH.read_text())


# Mounted last so /api/* wins; html=True serves index.html at "/".
app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")
