"""Extraction pipeline: concept set -> chunked cached activations -> probe sweep.

Usage:
    python -m src.extract --backend torch --model gpt2 --concepts colors
    python -m src.extract --backend mlx --model mlx-community/Llama-3.1-8B-Instruct-4bit \
        --concepts colors --chunk-size 4

Activations land under ``activations/<run name>/`` (gitignored); per-layer
probe accuracies print as a table and are written to ``probes.json`` inside
the run directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from src.activation_store import ActivationRun, ActivationWriter
from src.backends import Backend

CONCEPTS_DIR = Path(__file__).resolve().parent.parent / "concepts"


def load_concepts(name_or_path: str) -> dict:
    path = Path(name_or_path)
    if not path.exists():
        path = CONCEPTS_DIR / f"{name_or_path}.json"
    return json.loads(path.read_text())


def build_prompts(concepts: dict) -> tuple[list[str], np.ndarray, list[str]]:
    """Every (template, item) pair; labels index into the item list."""
    prompts, labels = [], []
    items = list(concepts["items"])
    for template in concepts["templates"]:
        for index, item in enumerate(items):
            prompts.append(template.format(item))
            labels.append(index)
    return prompts, np.array(labels, dtype=np.int64), items


def run_name(backend: Backend, concepts: dict) -> str:
    digest = hashlib.sha256(json.dumps(concepts, sort_keys=True).encode()).hexdigest()[:8]
    model_slug = backend.model_name.replace("/", "-") or "model"
    return f"{model_slug}-{concepts['name']}-{digest}"


def extract(backend: Backend, concepts: dict, out_root: str | Path, chunk_size: int = 8) -> Path:
    """Capture last-token activations for every prompt, chunk by chunk, to disk."""
    prompts, labels, items = build_prompts(concepts)
    layers = list(range(backend.n_layers))
    run_dir = Path(out_root) / run_name(backend, concepts)

    writer = ActivationWriter(run_dir, layers=layers, d_model=backend.d_model,
                              capacity=len(prompts))
    started = time.perf_counter()
    done = 0
    for chunk in backend.iter_last_token(prompts, layers, chunk_size=chunk_size):
        writer.append(chunk)
        done += chunk[layers[0]].shape[0]
        rate = done / (time.perf_counter() - started)
        print(f"  cached {done}/{len(prompts)} prompts ({rate:.1f} prompts/s)", flush=True)
    writer.finalize(labels, meta={
        "model": backend.model_name,
        "concept_set": concepts["name"],
        "items": items,
        "n_templates": len(concepts["templates"]),
        "prompts_per_second": round(done / (time.perf_counter() - started), 2),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    return run_dir


def probe_run(run_dir: str | Path) -> list[dict]:
    """Train per-layer probes over a cached run; write and return reports."""
    import torch

    from src.probes import sweep_layers

    run = ActivationRun(run_dir)
    activations = {
        layer: torch.from_numpy(np.asarray(run.layer(layer)).copy()) for layer in run.layers
    }
    labels = torch.from_numpy(run.labels())
    reports = sweep_layers(activations, labels)
    payload = [report.__dict__ for report in reports]
    (Path(run_dir) / "probes.json").write_text(json.dumps(payload, indent=1))
    return payload


def make_backend(kind: str, model: str) -> Backend:
    if kind == "torch":
        from src.backends import TorchBackend
        return TorchBackend.from_pretrained(model)
    if kind == "mlx":
        from src.backends import MlxBackend
        return MlxBackend(model)
    raise ValueError(f"unknown backend {kind!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["torch", "mlx"], default="torch")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--concepts", default="colors")
    parser.add_argument("--out", default="activations")
    parser.add_argument("--chunk-size", type=int, default=8)
    args = parser.parse_args()

    concepts = load_concepts(args.concepts)
    print(f"loading {args.model} ({args.backend}) ...")
    backend = make_backend(args.backend, args.model)
    print(f"{backend.n_layers} layers, d_model {backend.d_model}")

    run_dir = extract(backend, concepts, args.out, chunk_size=args.chunk_size)
    reports = probe_run(run_dir)

    print(f"\nprobe accuracy by layer ({concepts['name']}, {backend.model_name}):")
    best = max(reports, key=lambda r: r["accuracy"])
    for report in reports:
        marker = "  <- home layer" if report is best else ""
        print(f"  layer {report['layer']:2d}: {report['accuracy']:.3f}{marker}")
    print(f"\nrun: {run_dir}")


if __name__ == "__main__":
    main()
