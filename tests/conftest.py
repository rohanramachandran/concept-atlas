"""Shared fixtures: a tiny deterministic transformer-shaped model.

The toy model exposes its blocks as ``.blocks`` (one of the attribute paths
``src.hooks.resolve_blocks`` searches), takes pre-embedded float inputs of
shape ``(batch, seq, d_model)``, and returns logits of shape
``(batch, seq, vocab)`` — enough surface to exercise hooks, patching, and
metrics exactly as with a Hugging Face model, with no downloads.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

D_MODEL = 16
N_LAYERS = 3
VOCAB = 11


class ToyBlock(nn.Module):
    """Residual MLP block: x + f(x), mirroring a transformer block's shape."""

    def __init__(self, d_model: int):
        super().__init__()
        self.ff = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(x)


class ToyModel(nn.Module):
    def __init__(self, d_model: int = D_MODEL, n_layers: int = N_LAYERS, vocab: int = VOCAB):
        super().__init__()
        self.blocks = nn.ModuleList(ToyBlock(d_model) for _ in range(n_layers))
        self.head = nn.Linear(d_model, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.head(x)


@pytest.fixture()
def toy_model() -> ToyModel:
    torch.manual_seed(0)
    model = ToyModel()
    model.eval()
    return model


@pytest.fixture()
def toy_inputs() -> torch.Tensor:
    """A deterministic batch: (batch=2, seq=5, d_model)."""
    gen = torch.Generator().manual_seed(1)
    return torch.randn(2, 5, D_MODEL, generator=gen)
