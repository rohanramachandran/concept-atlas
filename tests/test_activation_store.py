"""Chunked activation store: roundtrip, truncation, and misuse errors."""
import numpy as np
import pytest

from src.activation_store import ActivationRun, ActivationWriter


def chunk(b, d, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((b, d)).astype(np.float32)


def test_roundtrip_across_chunks(tmp_path):
    writer = ActivationWriter(tmp_path / "run", layers=[0, 2], d_model=8, capacity=5)
    parts = {0: [], 2: []}
    for i, b in enumerate((2, 2, 1)):
        c = {0: chunk(b, 8, seed=i), 2: chunk(b, 8, seed=100 + i)}
        writer.append(c)
        for layer in (0, 2):
            parts[layer].append(c[layer])
    labels = np.array([0, 0, 1, 1, 2])
    writer.finalize(labels, meta={"model": "toy"})

    run = ActivationRun(tmp_path / "run")
    assert run.meta["n"] == 5
    assert run.meta["model"] == "toy"
    assert run.layers == [0, 2]
    for layer in (0, 2):
        np.testing.assert_array_equal(np.asarray(run.layer(layer)), np.concatenate(parts[layer]))
    np.testing.assert_array_equal(run.labels(), labels)


def test_truncates_to_written_rows(tmp_path):
    writer = ActivationWriter(tmp_path / "run", layers=[1], d_model=4, capacity=10)
    writer.append({1: chunk(3, 4, seed=0)})
    writer.finalize(np.zeros(3, dtype=int), meta={})
    run = ActivationRun(tmp_path / "run")
    assert np.asarray(run.layer(1)).shape == (3, 4)


def test_overflow_raises(tmp_path):
    writer = ActivationWriter(tmp_path / "run", layers=[0], d_model=4, capacity=2)
    with pytest.raises(ValueError):
        writer.append({0: chunk(3, 4, seed=0)})


def test_wrong_layers_raise(tmp_path):
    writer = ActivationWriter(tmp_path / "run", layers=[0, 1], d_model=4, capacity=4)
    with pytest.raises(ValueError):
        writer.append({0: chunk(2, 4, seed=0)})


def test_label_count_mismatch_raises(tmp_path):
    writer = ActivationWriter(tmp_path / "run", layers=[0], d_model=4, capacity=4)
    writer.append({0: chunk(2, 4, seed=0)})
    with pytest.raises(ValueError):
        writer.finalize(np.zeros(3, dtype=int), meta={})


def test_reading_unfinalized_run_raises(tmp_path):
    ActivationWriter(tmp_path / "run", layers=[0], d_model=4, capacity=4)
    with pytest.raises(FileNotFoundError):
        ActivationRun(tmp_path / "run")
