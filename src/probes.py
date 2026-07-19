"""Linear probes over residual-stream activations.

A probe is a single linear layer (with input standardization folded in as
buffers) trained with cross-entropy. High validation accuracy at a layer is
evidence the concept is linearly represented there; the layer with peak
accuracy is treated as the concept's *home layer*.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class LinearProbe(nn.Module):
    """Logistic-regression probe with stored input standardization."""

    def __init__(self, d_model: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(d_model, n_classes)
        self.register_buffer("mu", torch.zeros(d_model))
        self.register_buffer("sigma", torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear((x - self.mu) / self.sigma)

    @torch.no_grad()
    def accuracy(self, x: torch.Tensor, y: torch.Tensor) -> float:
        preds = self(x).argmax(dim=-1)
        return (preds == y).float().mean().item()


@dataclass(frozen=True)
class ProbeReport:
    """Per-layer probe result from a sweep."""

    layer: int
    accuracy: float
    n_train: int
    n_val: int


def train_probe(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    epochs: int = 200,
    lr: float = 1e-2,
    weight_decay: float = 1e-3,
    val_frac: float = 0.2,
    seed: int = 0,
) -> tuple[LinearProbe, float]:
    """Train a probe on ``(n, d_model)`` activations; return (probe, val accuracy).

    Deterministic for a fixed seed. Full-batch Adam is plenty for the small
    activation sets produced by concept templates.
    """
    if x.ndim != 2:
        raise ValueError(f"expected (n, d_model) activations, got shape {tuple(x.shape)}")
    x = x.float()
    y = y.long()
    n = x.shape[0]
    n_classes = int(y.max().item()) + 1

    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    n_val = int(n * val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    probe = LinearProbe(x.shape[1], n_classes)
    probe.mu.copy_(x_train.mean(dim=0))
    probe.sigma.copy_(x_train.std(dim=0).clamp_min(1e-6))

    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    probe.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(probe(x_train), y_train)
        loss.backward()
        opt.step()
    probe.eval()

    acc = probe.accuracy(x_val, y_val) if n_val > 0 else probe.accuracy(x_train, y_train)
    return probe, acc


def sweep_layers(
    activations: dict[int, torch.Tensor],
    labels: torch.Tensor,
    **train_kwargs: object,
) -> list[ProbeReport]:
    """Train one probe per layer; return reports sorted by layer index.

    ``activations`` maps layer index to ``(n, d_model)`` tensors (e.g. the
    last-token slice of a ``ResidualCache``).
    """
    reports = []
    for layer in sorted(activations):
        x = activations[layer]
        _, acc = train_probe(x, labels, **train_kwargs)  # type: ignore[arg-type]
        n = x.shape[0]
        n_val = int(n * float(train_kwargs.get("val_frac", 0.2)))  # type: ignore[arg-type]
        reports.append(ProbeReport(layer=layer, accuracy=acc, n_train=n - n_val, n_val=n_val))
    return reports


def best_layer(reports: list[ProbeReport]) -> ProbeReport:
    """The report with peak accuracy (ties break toward earlier layers)."""
    if not reports:
        raise ValueError("no probe reports")
    return max(reports, key=lambda r: (r.accuracy, -r.layer))
