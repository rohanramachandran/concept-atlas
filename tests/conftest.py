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


class ToyTokenizer:
    """Character tokenizer with HF-style padded batch output (right padding)."""

    pad_id = 0

    def _ids(self, text: str) -> list[int]:
        return [3 + (ord(c) % 7) for c in text]

    def __call__(self, texts, return_tensors="pt", padding=True):
        ids = [self._ids(t) for t in texts]
        width = max(len(i) for i in ids)
        input_ids = torch.tensor([i + [self.pad_id] * (width - len(i)) for i in ids])
        mask = torch.tensor([[1] * len(i) + [0] * (width - len(i)) for i in ids])
        return {"input_ids": input_ids, "attention_mask": mask}


class ToyLM(nn.Module):
    """Token-in, logits-out toy model for backend tests. Ignores the mask."""

    def __init__(self, d_model: int = D_MODEL, n_layers: int = N_LAYERS, vocab: int = VOCAB):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList(ToyBlock(d_model) for _ in range(n_layers))
        self.head = nn.Linear(d_model, vocab)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


@pytest.fixture()
def toy_backend():
    from src.backends import TorchBackend

    torch.manual_seed(0)
    model = ToyLM().eval()
    return TorchBackend(model, ToyTokenizer(), d_model=D_MODEL, device="cpu", model_name="toy")


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
