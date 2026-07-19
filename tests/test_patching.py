import torch

from src.patching import causal_effect, logit_diff_metric, pairwise_effects
from tests.conftest import D_MODEL, N_LAYERS


def test_causal_effect_at_final_layer_matches_direct_computation(toy_model, toy_inputs):
    """At the last block, the patched run's logits are exactly the source
    run's logits, so the effect must equal metric(source) - metric(base)."""
    base = toy_inputs
    source = toy_inputs.flip(0) + 0.5
    metric = logit_diff_metric(target_ids=[2], against_ids=[7])

    with torch.no_grad():
        expected = metric(toy_model(source)) - metric(toy_model(base))
    effect = causal_effect(
        toy_model, base_ids=base, source_ids=source, layer=N_LAYERS - 1, metric=metric
    )
    assert abs(effect - expected) < 1e-5


def test_identical_source_and_base_has_zero_effect(toy_model, toy_inputs):
    metric = logit_diff_metric(target_ids=[0])
    effect = causal_effect(
        toy_model, base_ids=toy_inputs, source_ids=toy_inputs, layer=0, metric=metric
    )
    assert abs(effect) < 1e-6


def test_logit_diff_metric_reads_last_position():
    logits = torch.zeros(1, 4, 10)
    logits[0, -1, 3] = 2.0
    logits[0, 0, 3] = 99.0  # earlier positions must be ignored
    assert logit_diff_metric(target_ids=[3])(logits) == 2.0
    assert logit_diff_metric(target_ids=[3], against_ids=[5])(logits) == 2.0


def test_pairwise_effects_covers_all_ordered_pairs(toy_model):
    gen = torch.Generator().manual_seed(5)
    inputs = {
        "red": torch.randn(1, 4, D_MODEL, generator=gen),
        "blue": torch.randn(1, 4, D_MODEL, generator=gen),
        "doctor": torch.randn(1, 4, D_MODEL, generator=gen),
    }
    metrics = {name: logit_diff_metric(target_ids=[i]) for i, name in enumerate(inputs)}

    effects = pairwise_effects(toy_model, inputs, metrics, layer=1)

    assert len(effects) == 6  # 3 concepts, ordered pairs, no self-loops
    assert all(a != b for a, b in effects)
    assert all(isinstance(v, float) for v in effects.values())
