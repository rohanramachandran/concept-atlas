import torch

from src.probes import best_layer, sweep_layers, train_probe


def _separable_data(n_per_class: int = 100, d: int = 16, seed: int = 0):
    """Three well-separated Gaussian clusters."""
    gen = torch.Generator().manual_seed(seed)
    centers = torch.tensor([[4.0], [-4.0], [0.0]]) * torch.ones(3, d)
    centers[2, : d // 2] = 6.0
    xs, ys = [], []
    for label, center in enumerate(centers):
        xs.append(center + torch.randn(n_per_class, d, generator=gen))
        ys.append(torch.full((n_per_class,), label))
    return torch.cat(xs), torch.cat(ys)


def test_probe_learns_separable_concepts():
    x, y = _separable_data()
    probe, acc = train_probe(x, y)
    assert acc > 0.9
    assert probe(x).shape == (x.shape[0], 3)


def test_probe_is_deterministic_for_fixed_seed():
    x, y = _separable_data()
    _, acc1 = train_probe(x, y, seed=0)
    _, acc2 = train_probe(x, y, seed=0)
    assert acc1 == acc2


def test_probe_near_chance_on_noise():
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(200, 16, generator=gen)
    y = torch.randint(0, 2, (200,), generator=gen)
    _, acc = train_probe(x, y)
    assert acc < 0.8  # noise must not look like structure


def test_sweep_identifies_the_informative_layer():
    x, y = _separable_data()
    gen = torch.Generator().manual_seed(4)
    noise = torch.randn(x.shape, generator=gen)
    reports = sweep_layers({0: noise, 1: x}, y)

    assert [r.layer for r in reports] == [0, 1]
    winner = best_layer(reports)
    assert winner.layer == 1
    assert winner.accuracy > 0.9
