"""Probe accuracy per layer, per concept set, for one model.

    python -m experiments.run_probes --model gpt2
    python -m experiments.run_probes --model llama

Caches activations through src.extract (chunked, on disk), then trains one
probe per layer per seed and reports mean and spread across seeds. Output:
experiments/results/probes-<model>.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from experiments.common import CONCEPT_SETS, PROBE_SEEDS, RESULTS_DIR, make_backend
from src.activation_store import ActivationRun
from src.extract import extract, load_concepts
from src.probes import sweep_layers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gpt2", "llama"], required=True)
    parser.add_argument("--chunk-size", type=int, default=8)
    args = parser.parse_args()

    backend = make_backend(args.model)
    print(f"{args.model}: {backend.n_layers} layers, d_model {backend.d_model}")

    out = {"model": backend.model_name, "alias": args.model,
           "n_layers": backend.n_layers, "seeds": PROBE_SEEDS, "sets": {}}

    for set_name in CONCEPT_SETS:
        concepts = load_concepts(set_name)
        print(f"[{set_name}] caching activations ...")
        run_dir = extract(backend, concepts, "activations", chunk_size=args.chunk_size)
        run = ActivationRun(run_dir)
        activations = {
            layer: torch.from_numpy(np.asarray(run.layer(layer)).copy())
            for layer in run.layers
        }
        labels = torch.from_numpy(run.labels())

        per_seed = []
        for seed in PROBE_SEEDS:
            reports = sweep_layers(activations, labels, seed=seed)
            per_seed.append([r.accuracy for r in reports])
        acc = np.array(per_seed)  # (n_seeds, n_layers)
        mean, spread = acc.mean(axis=0), acc.std(axis=0)
        home = int(mean.argmax())
        n_items = len(concepts["items"])
        out["sets"][set_name] = {
            "n_prompts": int(labels.shape[0]),
            "n_val": int(labels.shape[0] * 0.2),
            "chance": 1.0 / n_items,
            "layers": list(range(backend.n_layers)),
            "acc_mean": [round(float(v), 4) for v in mean],
            "acc_std": [round(float(v), 4) for v in spread],
            "home_layer": home,
            "home_accuracy": round(float(mean[home]), 4),
        }
        print(f"[{set_name}] home layer {home}, accuracy {mean[home]:.3f} "
              f"(chance {1.0 / n_items:.3f})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"probes-{args.model}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
