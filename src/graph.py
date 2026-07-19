"""Typed, weighted concept graph and its D3-compatible JSON serialization.

Nodes are concepts located by probes; edges are measured causal effects.
The JSON layout (``{"nodes": [...], "links": [...], "meta": {...}}``) is
exactly what the D3 explorer in ``static/`` consumes.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ConceptNode:
    """A concept located in the model."""

    id: str
    name: str
    group: str          # concept set, e.g. "colors"
    layer: int          # home layer (peak probe accuracy)
    accuracy: float     # probe validation accuracy at the home layer


@dataclass(frozen=True)
class CausalEdge:
    """A measured causal link between two concepts."""

    source: str
    target: str
    weight: float       # signed effect size from activation patching
    kind: str           # "excitatory" (weight > 0) or "inhibitory" (weight < 0)
    layer: int          # layer at which the patch was applied
