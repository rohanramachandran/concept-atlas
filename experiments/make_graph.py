"""Build the explorer graph from measured results.

    python -m experiments.make_graph --model llama --min-weight 1.0

Replaces ``static/graph.json`` with a graph whose nodes come from the probe
runs (home layer and accuracy per concept set) and whose edges are measured
patching effects. Every number in the explorer is then something an
experiment produced, reproducible from the JSONs in ``experiments/results/``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.common import CONCEPT_SETS, RESULTS_DIR
from src.graph import ConceptGraph, ConceptNode

STATIC_GRAPH = Path(__file__).resolve().parent.parent / "static" / "graph.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gpt2", "llama"], default="llama")
    parser.add_argument("--min-weight", type=float, default=1.0)
    parser.add_argument("--out", default=str(STATIC_GRAPH))
    args = parser.parse_args()

    probes = json.loads((RESULTS_DIR / f"probes-{args.model}.json").read_text())
    graph = ConceptGraph(model_name=f"{probes['model']} (measured)")

    for set_name in CONCEPT_SETS:
        set_data = probes["sets"][set_name]
        items = json.loads(
            (Path(__file__).resolve().parent.parent / "concepts" / f"{set_name}.json").read_text()
        )["items"]
        for item in items:
            graph.add_node(ConceptNode(
                id=item, name=item, group=set_name,
                layer=set_data["home_layer"],
                accuracy=round(set_data["home_accuracy"], 3),
            ))

    for set_name in CONCEPT_SETS:
        path = RESULTS_DIR / f"patching-{args.model}-{set_name}.json"
        if not path.exists():
            print(f"skipping {set_name}: no patching results at {path}")
            continue
        patching = json.loads(path.read_text())
        items = patching["items"]
        for i, source in enumerate(items):
            for j, target in enumerate(items):
                if i != j:
                    graph.add_edge(source, target,
                                   weight=round(patching["matrix"][i][j], 3),
                                   layer=patching["layer"])

    before = len(graph.edges)
    graph.prune(min_weight=args.min_weight)
    graph.save(args.out)
    print(f"{len(graph.nodes)} nodes, {len(graph.edges)} edges "
          f"(pruned {before - len(graph.edges)} below |{args.min_weight}|)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
