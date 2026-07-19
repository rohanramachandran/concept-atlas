"""Causal-effect measurement via activation patching.

The core operation: run the model on a *base* input, but substitute the
residual-stream activations recorded from a *source* input at one layer,
and measure how much a scalar metric of the output moves. A nonzero effect
means the information written at that layer causally influences the metric.
"""
from __future__ import annotations

from typing import Callable, Sequence

import torch
from torch import nn

from src.hooks import ResidualCache, patch_residual

Metric = Callable[[torch.Tensor], float]


def _logits(output: object) -> torch.Tensor:
    """Support both HF outputs (with ``.logits``) and plain tensors."""
    return output.logits if hasattr(output, "logits") else output  # type: ignore[union-attr,return-value]


def logit_diff_metric(
    target_ids: Sequence[int],
    against_ids: Sequence[int] | None = None,
) -> Metric:
    """Mean last-position logit of ``target_ids``, minus ``against_ids`` if given.

    This is the standard patching metric: how strongly does the model favor
    the target concept's tokens at the next-token position?
    """

    def metric(logits: torch.Tensor) -> float:
        last = logits[:, -1, :]
        score = last[:, list(target_ids)].mean()
        if against_ids is not None:
            score = score - last[:, list(against_ids)].mean()
        return score.item()

    return metric
