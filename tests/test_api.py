from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_concepts_lists_all_seed_sets():
    r = client.get("/api/concepts")
    assert r.status_code == 200
    names = {c["name"] for c in r.json()}
    assert {"colors", "professions", "countries"} <= names
    for c in r.json():
        assert c["n_items"] > 0
        assert c["n_templates"] > 0


def test_single_concept_set():
    r = client.get("/api/concepts/colors")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "colors"
    assert "{}" in body["templates"][0]


def test_unknown_concept_set_404s():
    assert client.get("/api/concepts/nonexistent").status_code == 404


def test_graph_is_d3_shaped():
    r = client.get("/api/graph")
    assert r.status_code == 200
    body = r.json()
    assert {"meta", "nodes", "links"} <= set(body)
    node_ids = {n["id"] for n in body["nodes"]}
    for link in body["links"]:
        assert link["source"] in node_ids
        assert link["target"] in node_ids


def test_explorer_served_at_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "concept-atlas" in r.text
