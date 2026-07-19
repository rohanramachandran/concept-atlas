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
