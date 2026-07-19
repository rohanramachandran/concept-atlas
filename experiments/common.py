"""Shared pieces for the reproducible experiment runs."""
from __future__ import annotations

from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"

MODELS = {
    "gpt2": ("torch", "gpt2"),
    "llama": ("mlx", "mlx-community/Llama-3.1-8B-Instruct-4bit"),
}

CONCEPT_SETS = ["colors", "professions", "countries"]
PROBE_SEEDS = [0, 1, 2]


def make_backend(alias: str):
    kind, model = MODELS[alias]
    from src.extract import make_backend as build
    return build(kind, model)


def patching_template(concepts: dict) -> str:
    """The template whose ``{}`` sits closest to the end, so the prefix is a
    natural base prompt and the item is what the model is about to say."""

    def terminal_score(template: str) -> tuple:
        prefix, _, suffix = template.partition("{}")
        return (len(suffix.strip(" .,!")) == 0, len(prefix))

    best = max(concepts["templates"], key=terminal_score)
    if not best.partition("{}")[0].strip():
        raise ValueError(f"no template with a usable prefix in {concepts['name']}")
    return best


def split_template(template: str) -> str:
    """Base prompt: everything before ``{}``, trailing whitespace stripped."""
    prefix = template.partition("{}")[0]
    return prefix.rstrip()


def top_edges(matrix: np.ndarray, items: list[str], k: int = 10) -> list[dict]:
    """Top-k off-diagonal entries by absolute effect."""
    edges = []
    for i, source in enumerate(items):
        for j, target in enumerate(items):
            if i != j:
                edges.append({"source": source, "target": target,
                              "effect": float(matrix[i, j])})
    edges.sort(key=lambda e: abs(e["effect"]), reverse=True)
    return edges[:k]
