"""The benchmark loader, on a tiny fixture.

Loading is where a benchmark lies to you. Every bug here produces a *plausible*
number rather than an error: score the unjudged queries and the retriever looks
three times worse; compare chunk ids to doc ids and everything reads 0.0; drop
the title and the score sags for a reason you'd never guess from the metric.
None of that throws. All of it gets published.
"""
import json

import pytest

from ragevallab.benchmark import docs_from_chunks, load_beir, run_benchmark


@pytest.fixture
def beir(tmp_path):
    """A miniature BEIR dataset: 3 docs, 3 queries, only 2 of them judged."""
    (tmp_path / "qrels").mkdir()
    docs = [
        {"_id": "d1", "title": "Venus is hot", "text": "Surface temperature is 460C."},
        {"_id": "d2", "title": "Jupiter is large", "text": "It is the largest planet."},
        {"_id": "d3", "title": "Unrelated", "text": "Nothing to do with planets."},
    ]
    (tmp_path / "corpus.jsonl").write_text("\n".join(json.dumps(d) for d in docs))
    queries = [
        {"_id": "q1", "text": "How hot is Venus surface temperature"},
        {"_id": "q2", "text": "Which planet is the largest"},
        {"_id": "q3", "text": "An unjudged query nobody scored"},
    ]
    (tmp_path / "queries.jsonl").write_text("\n".join(json.dumps(q) for q in queries))
    (tmp_path / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td2\t1\n"
    )
    return tmp_path


def test_loads_docs_queries_and_qrels(beir):
    d = load_beir(beir)
    assert len(d.docs) == 3
    assert d.qrels == {"q1": ["d1"], "q2": ["d2"]}


def test_only_judged_queries_are_scored(beir):
    """q3 has no relevance judgement. Including it would average in a zero.

    BEIR ships 1,109 SciFact queries and judges 300 of them. Scoring all 1,109
    silently reports a retriever as roughly a third as good as it is — a wrong
    number that looks exactly like a right one.
    """
    d = load_beir(beir)
    assert set(d.queries) == {"q1", "q2"}
    assert "q3" not in d.queries


def test_title_is_part_of_the_document(beir):
    """SciFact titles carry the signal; dropping them quietly costs recall."""
    d = load_beir(beir)
    venus = next(x for x in d.docs if x["id"] == "d1")
    assert "Venus is hot" in venus["text"] and "460C" in venus["text"]


def test_zero_scored_qrels_are_not_relevant(beir):
    (beir / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td3\t0\n"
    )
    assert load_beir(beir).qrels == {"q1": ["d1"]}


def test_missing_dataset_says_so(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_beir(tmp_path)


def test_chunks_collapse_to_documents_keeping_best_rank():
    assert docs_from_chunks(["d2#1", "d1#0", "d2#0", "d1#3"]) == ["d2", "d1"]


def test_a_document_with_no_chunk_suffix_still_works():
    assert docs_from_chunks(["d1", "d2#0"]) == ["d1", "d2"]


def test_benchmark_scores_the_real_pipeline(beir):
    """End to end on the fixture: the obvious answers should rank first."""
    r = run_benchmark(load_beir(beir), k=3)
    assert r["n_queries"] == 2 and r["n_docs"] == 3
    assert r["ndcg@3"] == 1.0, "the matching abstract should rank first for both"
