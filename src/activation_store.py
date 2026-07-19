"""Chunked on-disk activation storage.

Large-model activation collection cannot hold a whole corpus in memory, so
runs are written chunk by chunk into per-layer memory-mapped ``.npy`` files.
Peak RAM is one chunk of one forward pass regardless of corpus size, and a
finished run is a plain directory of arrays plus a ``meta.json`` that records
where the activations came from.

Layout of a run directory::

    <root>/<run_name>/
        meta.json          # model, layers, d_model, n, extra provenance
        labels.npy         # (n,) int labels, written at finalize
        layer_00.npy       # (n, d_model) float32
        layer_01.npy
        ...
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class ActivationWriter:
    """Preallocates per-layer memmaps and fills them chunk by chunk."""

    def __init__(self, run_dir: str | Path, *, layers: list[int], d_model: int, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.layers = sorted(layers)
        self.d_model = d_model
        self.capacity = capacity
        self.cursor = 0
        self._maps = {
            layer: np.lib.format.open_memmap(
                self.run_dir / f"layer_{layer:02d}.npy",
                mode="w+", dtype=np.float32, shape=(capacity, d_model))
            for layer in self.layers
        }

    def append(self, chunk: dict[int, np.ndarray]) -> None:
        """Write one chunk: ``{layer: (b, d_model)}`` for every open layer."""
        sizes = {a.shape[0] for a in chunk.values()}
        if set(chunk) != set(self.layers) or len(sizes) != 1:
            raise ValueError("chunk must cover exactly the writer's layers with equal batch sizes")
        b = sizes.pop()
        if self.cursor + b > self.capacity:
            raise ValueError(f"chunk overflows capacity {self.capacity}")
        for layer, acts in chunk.items():
            self._maps[layer][self.cursor:self.cursor + b] = acts.astype(np.float32)
        self.cursor += b

    def finalize(self, labels: np.ndarray, meta: dict) -> None:
        """Flush maps, write labels and provenance. Truncates to written rows."""
        if labels.shape[0] != self.cursor:
            raise ValueError(f"{labels.shape[0]} labels for {self.cursor} written rows")
        n = self.cursor
        for layer in self.layers:
            self._maps[layer].flush()
            if n < self.capacity:
                arr = np.asarray(self._maps[layer][:n]).copy()
                np.save(self.run_dir / f"layer_{layer:02d}.npy", arr)
        np.save(self.run_dir / "labels.npy", labels.astype(np.int64))
        payload = dict(meta)
        payload.update({"layers": self.layers, "d_model": self.d_model, "n": n})
        (self.run_dir / "meta.json").write_text(json.dumps(payload, indent=1))


class ActivationRun:
    """Read side: lazy per-layer memmaps over a finalized run."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        meta_path = self.run_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"{self.run_dir} is not a finalized activation run")
        self.meta = json.loads(meta_path.read_text())

    @property
    def layers(self) -> list[int]:
        return list(self.meta["layers"])

    def layer(self, index: int) -> np.ndarray:
        arr = np.load(self.run_dir / f"layer_{index:02d}.npy", mmap_mode="r")
        if arr.shape != (self.meta["n"], self.meta["d_model"]):
            raise ValueError(f"layer {index} shape {arr.shape} disagrees with meta")
        return arr

    def labels(self) -> np.ndarray:
        return np.load(self.run_dir / "labels.npy")

    def all_layers(self) -> dict[int, np.ndarray]:
        return {i: self.layer(i) for i in self.layers}
