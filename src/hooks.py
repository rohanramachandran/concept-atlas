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
