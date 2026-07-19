"""TorchBackend capture, chunking, and backend-agnostic patching on the toy LM."""
import numpy as np
import pytest

from src.backends import causal_effect, logit_diff_metric

TEXTS = ["ab", "abcdef", "xyz", "hello", "hi"]


def test_capture_last_token_shapes(toy_backend):
    acts = toy_backend.capture_last_token(TEXTS[:3], layers=[0, 2])
    assert set(acts) == {0, 2}
    for layer_acts in acts.values():
        assert layer_acts.shape == (3, toy_backend.d_model)
        assert layer_acts.dtype == np.float32


def test_padding_does_not_corrupt_last_token(toy_backend):
    """The short text's last-token vector must match its unpadded solo run."""
    batched = toy_backend.capture_last_token(["ab", "abcdef"], layers=[1])[1]
    solo = toy_backend.capture_last_token(["ab"], layers=[1])[1]
    np.testing.assert_allclose(batched[0], solo[0], atol=1e-5)


def test_iter_last_token_chunks_match_oneshot(toy_backend):
    chunks = list(toy_backend.iter_last_token(TEXTS, layers=[0], chunk_size=2))
    assert [c[0].shape[0] for c in chunks] == [2, 2, 1]
    stacked = np.concatenate([c[0] for c in chunks])
    oneshot = toy_backend.capture_last_token(TEXTS, layers=[0])[0]
    np.testing.assert_allclose(stacked, oneshot, atol=1e-5)


def test_self_patch_has_no_effect(toy_backend):
    metric = logit_diff_metric(target_ids=[1, 2])
    effect = causal_effect(toy_backend, "hello", "hello", layer=1, metric=metric)
    assert abs(effect) < 1e-4


def test_cross_patch_moves_logits(toy_backend):
    metric = logit_diff_metric(target_ids=[1, 2])
    effect = causal_effect(toy_backend, "hello", "xyz", layer=1, metric=metric)
    assert abs(effect) > 1e-6


def test_tail_alignment_handles_length_mismatch(toy_backend):
    """Source shorter and longer than base must both patch without error."""
    metric = logit_diff_metric(target_ids=[1])
    for source in ("ab", "abcdefghij"):
        effect = causal_effect(toy_backend, "hello", source, layer=0, metric=metric)
        assert np.isfinite(effect)


def test_positions_patch_changes_specified_position(toy_backend):
    src = toy_backend.capture_sequence("hello", layer=0)
    base = toy_backend.logits("world")
    patched = toy_backend.logits("world", patch=(0, src, [0]))
    assert not np.allclose(base[0, 0], patched[0, 0])


def test_capture_sequence_shape(toy_backend):
    acts = toy_backend.capture_sequence("hello", layer=2)
    assert acts.shape == (1, 5, toy_backend.d_model)


def test_unknown_layers_raise(toy_backend):
    with pytest.raises((IndexError, KeyError)):
        toy_backend.capture_last_token(["ab"], layers=[99])
