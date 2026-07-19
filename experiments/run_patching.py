"""Activation-patching effect sizes between concept items, for one model.

    python -m experiments.run_patching --model gpt2 --set colors

Design: the base prompt is a template prefix that stops right before the item
("She bought a"). For each source item, the full prompt's activations are
captured at the probe home layer and patched into the base run (tail-aligned).
One patched forward serves every target, because the readout per target is a
logit metric: mean first-token logit of the target item minus the mean over
the other items. Effect(source, target) = patched metric - base metric.

Sanity property: the diagonal (source == target) should be strongly positive,
since patching an item's own activations should push the model toward saying
that item. The off-diagonal structure is the interesting part.

Output: experiments/results/patching-<model>-<set>.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from experiments.common import RESULTS_DIR, make_backend, patching_template, split_template, top_edges
from src.extract import load_concepts


def item_first_token(backend, item: str) -> int:
    ids = backend.token_ids(" " + item)
    if not ids:
        ids = backend.token_ids(item)
    return ids[0]


def metric_scores(logits: np.ndarray, token_ids: list[int]) -> np.ndarray:
    """Per-item score at the final position: item logit minus mean of the rest."""
    last = logits[0, -1]
    raw = np.array([last[t] for t in token_ids], dtype=np.float64)
    total = raw.sum()
    others_mean = (total - raw) / (len(raw) - 1)
    return raw - others_mean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gpt2", "llama"], required=True)
    parser.add_argument("--set", dest="set_name", default="colors")
    args = parser.parse_args()

    probes_path = RESULTS_DIR / f"probes-{args.model}.json"
    if not probes_path.exists():
        raise SystemExit(f"run experiments.run_probes --model {args.model} first")
    probes = json.loads(probes_path.read_text())
    layer = probes["sets"][args.set_name]["home_layer"]

    concepts = load_concepts(args.set_name)
    items = concepts["items"]
    template = patching_template(concepts)
    base_prompt = split_template(template)

    backend = make_backend(args.model)
    token_ids = [item_first_token(backend, item) for item in items]
    if len(set(token_ids)) != len(token_ids):
        raise SystemExit(f"first tokens collide for {args.set_name}; refine the item list")

    print(f"{args.model}/{args.set_name}: layer {layer}, base prompt {base_prompt!r}")
    base_scores = metric_scores(backend.logits(base_prompt), token_ids)

    matrix = np.zeros((len(items), len(items)))
    for i, source in enumerate(items):
        source_acts = backend.capture_sequence(template.format(source), layer)
        patched = backend.logits(base_prompt, patch=(layer, source_acts, None))
        matrix[i] = metric_scores(patched, token_ids) - base_scores
        print(f"  {source:>10}: self-effect {matrix[i, i]:+.3f}")

    diagonal = np.diag(matrix)
    off = matrix[~np.eye(len(items), dtype=bool)]
    summary = {
        "model": backend.model_name,
        "alias": args.model,
        "set": args.set_name,
        "layer": layer,
        "template": template,
        "base_prompt": base_prompt,
        "items": items,
        "matrix": [[round(float(v), 4) for v in row] for row in matrix],
        "diagonal_median": round(float(np.median(diagonal)), 4),
        "off_diagonal_median_abs": round(float(np.median(np.abs(off))), 4),
        "top_edges": top_edges(matrix, items, k=10),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"patching-{args.model}-{args.set_name}.json"
    path.write_text(json.dumps(summary, indent=1))
    print(f"diagonal median {summary['diagonal_median']:+.3f}, "
          f"off-diagonal median |effect| {summary['off_diagonal_median_abs']:.3f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
