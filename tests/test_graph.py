import pytest

from src.graph import ConceptGraph, ConceptNode


def make_graph() -> ConceptGraph:
    g = ConceptGraph(model_name="toy")
    g.add_node(ConceptNode(id="red", name="red", group="colors", layer=1, accuracy=0.9))
    g.add_node(ConceptNode(id="blue", name="blue", group="colors", layer=1, accuracy=0.8))
    g.add_node(ConceptNode(id="doctor", name="doctor", group="professions", layer=2, accuracy=0.85))
    g.add_edge("red", "blue", weight=-0.3, layer=1)
    g.add_edge("blue", "doctor", weight=0.02, layer=2)
    g.add_edge("doctor", "red", weight=0.4, layer=2)
    return g


def test_edge_kinds_follow_sign():
    g = make_graph()
    kinds = {(e.source, e.target): e.kind for e in g.edges}
    assert kinds[("red", "blue")] == "inhibitory"
    assert kinds[("doctor", "red")] == "excitatory"


def test_duplicate_node_rejected():
    g = make_graph()
    with pytest.raises(ValueError, match="duplicate node id"):
        g.add_node(ConceptNode(id="red", name="red", group="colors", layer=3, accuracy=0.5))


def test_edge_requires_known_endpoints():
    g = make_graph()
    with pytest.raises(ValueError, match="not a known node"):
        g.add_edge("red", "ghost", weight=0.1, layer=1)


def test_prune_drops_weak_edges():
    g = make_graph()
    g.prune(min_weight=0.1)
    assert len(g.edges) == 2
    assert all(abs(e.weight) >= 0.1 for e in g.edges)


def test_json_round_trip(tmp_path):
    g = make_graph()
    path = tmp_path / "graph.json"
    g.save(path)
    loaded = ConceptGraph.load(path)

    assert loaded.model_name == "toy"
    assert loaded.nodes == g.nodes
    assert loaded.edges == g.edges


def test_d3_format_keys():
    d = make_graph().to_dict()
    assert set(d) == {"meta", "nodes", "links"}
    assert {"id", "name", "group", "layer", "accuracy"} <= set(d["nodes"][0])
    assert {"source", "target", "weight", "kind", "layer"} <= set(d["links"][0])
