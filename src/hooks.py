"""Residual-stream capture and patching via forward hooks.

Works with any model that exposes its transformer blocks as an indexable
module list under a conventional attribute path (GPT-2, LLaMA-family,
GPT-NeoX, BERT encoders) or a bare ``blocks`` attribute (used by the toy
model in the test suite).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Sequence

import torch
from torch import nn

# Attribute paths searched, in order, to find the stack of transformer blocks.
_BLOCK_PATHS = (
    "transformer.h",      # GPT-2 family
    "model.layers",       # LLaMA / Mistral family
    "gpt_neox.layers",    # GPT-NeoX / Pythia
    "encoder.layer",      # BERT encoders
    "blocks",             # generic / toy models
)


def resolve_blocks(model: nn.Module) -> list[nn.Module]:
    """Return the model's transformer blocks as a flat list.

    Raises ``ValueError`` with the searched paths if none match, so
    unsupported architectures fail loudly rather than probing nothing.
    """
    for path in _BLOCK_PATHS:
        obj: object = model
        for attr in path.split("."):
            if not hasattr(obj, attr):
                obj = None
                break
            obj = getattr(obj, attr)
        if isinstance(obj, (nn.ModuleList, nn.Sequential, list)):
            return list(obj)
    raise ValueError(
        f"Could not locate transformer blocks on {type(model).__name__}; "
        f"searched attribute paths {_BLOCK_PATHS}."
    )


def _hidden(output: object) -> torch.Tensor:
    """Extract the hidden-state tensor from a block output (tuple or tensor)."""
    return output[0] if isinstance(output, tuple) else output  # type: ignore[index]


class ResidualCache:
    """Context manager that records each block's output hidden states.

    Usage::

        with ResidualCache(model) as cache:
            model(input_ids)
        acts = cache.activations   # {layer_index: (batch, seq, d_model)}
    """

    def __init__(self, model: nn.Module, layers: Iterable[int] | None = None):
        self.blocks = resolve_blocks(model)
        self.layers = list(range(len(self.blocks))) if layers is None else list(layers)
        self.activations: dict[int, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, layer: int):
        def hook(_module: nn.Module, _inputs: tuple, output: object) -> None:
            self.activations[layer] = _hidden(output).detach()

        return hook

    def __enter__(self) -> "ResidualCache":
        for layer in self.layers:
            self._handles.append(
                self.blocks[layer].register_forward_hook(self._make_hook(layer))
            )
        return self

    def __exit__(self, *exc: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
