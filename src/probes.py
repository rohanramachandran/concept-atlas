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
