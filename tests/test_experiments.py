"""Unit coverage for the experiment utilities (no models involved)."""
import numpy as np
import pytest

from experiments.common import patching_template, split_template, top_edges


def test_split_template_strips_trailing_space():
    assert split_template("She bought a {} scarf.") == "She bought a"
    assert split_template("The color was {}.") == "The color was"


def test_patching_template_prefers_terminal_placeholder():
    concepts = {"name": "x", "templates": [
        "A {} balloon drifted.",
        "His favorite color is {}.",
        "The {} car parked.",
    ]}
    assert patching_template(concepts) == "His favorite color is {}."


def test_patching_template_rejects_leading_placeholder_only():
    with pytest.raises(ValueError):
        patching_template({"name": "x", "templates": ["{} is nice."]})


def test_top_edges_excludes_diagonal_and_sorts_by_magnitude():
    matrix = np.array([
        [9.0, -0.5, 0.1],
        [0.3, 9.0, -0.9],
        [0.2, 0.4, 9.0],
    ])
    edges = top_edges(matrix, ["a", "b", "c"], k=3)
    assert [(e["source"], e["target"]) for e in edges] == [("b", "c"), ("a", "b"), ("c", "b")]
    assert all(e["source"] != e["target"] for e in edges)
