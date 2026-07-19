import pytest
import torch
from torch import nn

from src.hooks import ResidualCache, patch_residual, resolve_blocks
from tests.conftest import D_MODEL, N_LAYERS


def test_resolve_blocks_finds_toy_stack(toy_model):
    blocks = resolve_blocks(toy_model)
    assert len(blocks) == N_LAYERS


def test_resolve_blocks_rejects_unknown_architecture():
    with pytest.raises(ValueError, match="Could not locate transformer blocks"):
        resolve_blocks(nn.Linear(4, 4))


def test_cache_captures_every_layer_with_correct_shapes(toy_model, toy_inputs):
    with ResidualCache(toy_model) as cache:
        toy_model(toy_inputs)
    assert sorted(cache.activations) == list(range(N_LAYERS))
    for acts in cache.activations.values():
        assert acts.shape == (2, 5, D_MODEL)


def test_cache_respects_layer_subset(toy_model, toy_inputs):
    with ResidualCache(toy_model, layers=[1]) as cache:
        toy_model(toy_inputs)
    assert list(cache.activations) == [1]


def test_cache_removes_hooks_on_exit(toy_model, toy_inputs):
    with ResidualCache(toy_model) as cache:
        toy_model(toy_inputs)
    first = {k: v.clone() for k, v in cache.activations.items()}
    toy_model(toy_inputs * 2.0)  # outside the context: must not re-capture
    for layer, acts in cache.activations.items():
        assert torch.equal(acts, first[layer])


def test_patching_last_layer_reproduces_source_logits(toy_model, toy_inputs):
    """Patching source activations at the final block must yield the source's
    logits exactly, since logits = head(final block output)."""
    source = toy_inputs
    base = toy_inputs.flip(0) + 0.5

    with torch.no_grad():
        source_logits = toy_model(source)
        with ResidualCache(toy_model, layers=[N_LAYERS - 1]) as cache:
            toy_model(source)
        with patch_residual(toy_model, N_LAYERS - 1, cache.activations[N_LAYERS - 1]):
            patched_logits = toy_model(base)

    assert torch.allclose(patched_logits, source_logits, atol=1e-6)


def test_patching_positions_only_alters_those_positions(toy_model, toy_inputs):
    replacement = torch.zeros(2, 5, D_MODEL)
    with torch.no_grad():
        base_logits = toy_model(toy_inputs)
        with patch_residual(toy_model, N_LAYERS - 1, replacement, positions=[0]):
            patched_logits = toy_model(toy_inputs)

    assert not torch.allclose(patched_logits[:, 0, :], base_logits[:, 0, :])
    assert torch.allclose(patched_logits[:, 1:, :], base_logits[:, 1:, :], atol=1e-6)
