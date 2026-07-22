"""The optional FastAPI service over the pipeline."""
from fastapi.testclient import TestClient

from ragevallab.api import create_app
from ragevallab.data import EVAL_SET


def _client() -> TestClient:
    return TestClient(create_app())


def test_healthz():
    r = _client().get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_query_returns_answer_with_citations_and_ranking():
    q = EVAL_SET[0]["q"]                       # a query the demo corpus covers
    r = _client().post("/query", json={"query": q, "k": 4})
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == q
    assert body["text"]                        # a non-empty extractive answer
    assert body["retrieved"]                   # ranked chunk ids
    assert isinstance(body["citations"], list)


def test_query_rejects_empty_input():
    r = _client().post("/query", json={"query": "", "k": 4})
    assert r.status_code == 422                # min_length=1


def test_query_rejects_out_of_range_k():
    r = _client().post("/query", json={"query": "anything", "k": 999})
    assert r.status_code == 422                # k <= 20


def test_eval_runs_the_suite_and_flags_the_planted_case():
    r = _client().post("/eval", json={"k": 4})
    assert r.status_code == 200
    body = r.json()
    assert body["metrics"]["n_cases"] >= 1
    assert "faithfulness" in body["metrics"]
    # The planted hallucination must be caught, or the harness isn't working.
    assert body["metrics"]["flagged_cases"] >= 1
    assert any(c["flagged"] for c in body["cases"])
