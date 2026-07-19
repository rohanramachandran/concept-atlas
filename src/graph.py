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


@dataclass
class ConceptGraph:
    """Container with pruning and (de)serialization."""

    model_name: str = "unknown"
    nodes: list[ConceptNode] = field(default_factory=list)
    edges: list[CausalEdge] = field(default_factory=list)

    def add_node(self, node: ConceptNode) -> None:
        if any(n.id == node.id for n in self.nodes):
            raise ValueError(f"duplicate node id: {node.id!r}")
        self.nodes.append(node)

    def add_edge(self, source: str, target: str, weight: float, layer: int) -> None:
        ids = {n.id for n in self.nodes}
        for endpoint in (source, target):
            if endpoint not in ids:
                raise ValueError(f"edge endpoint {endpoint!r} is not a known node")
        kind = "excitatory" if weight > 0 else "inhibitory"
        self.edges.append(CausalEdge(source, target, weight, kind, layer))

    def prune(self, min_weight: float) -> None:
        """Drop edges with |weight| below ``min_weight``."""
        self.edges = [e for e in self.edges if abs(e.weight) >= min_weight]

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "meta": {"model": self.model_name, "format": "concept-atlas/v1"},
            "nodes": [asdict(n) for n in self.nodes],
            "links": [asdict(e) for e in self.edges],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConceptGraph":
        graph = cls(model_name=data.get("meta", {}).get("model", "unknown"))
        graph.nodes = [ConceptNode(**n) for n in data["nodes"]]
        graph.edges = [CausalEdge(**e) for e in data["links"]]
        return graph

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ConceptGraph":
        return cls.from_dict(json.loads(Path(path).read_text()))
