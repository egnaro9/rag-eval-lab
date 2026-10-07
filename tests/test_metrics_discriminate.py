"""Fixtures chosen so the wrong formula gives a different number.

A mutation audit replaced ten expressions in this package with constants or with their
sibling metric, and all 45 tests stayed green. The cause was uniform: the demo fixtures are
degenerate. Every case in EVAL_SET has precision 0.25 and recall 1.0, so a mean that is always
1.0 is only visibly wrong if something asserts the real value; 'alpha beta gamma' beats
'alpha only' under either coverage denominator; and a reranked list already in retrieval order
cannot show whether the retrieval prior did anything.

Each test below is built on an input where the real implementation and the plausible wrong one
DISAGREE, and asserts the exact number.
"""
from __future__ import annotations

import pytest

from ragevallab.data import EVAL_SET
from ragevallab.evals import (citation_present, evaluate, precision_at_k,
                              recall_at_k)
from ragevallab.retrieval import reciprocal_rank_fusion


# ─────────────────────────────── precision vs recall ───────────────────────────────

def test_precision_and_recall_disagree_on_this_input():
    """In the demo corpus every case has precision 0.25 and recall 1.0, so swapping the two
    fields changes each case's value. Nothing asserted either, so `"precision@k": round(p, 3)`
    could return `round(r, 3)` with the suite green."""
    retrieved, gold = ["a", "x", "y", "z"], ["a"]
    assert precision_at_k(retrieved, gold, 4) == 0.25
    assert recall_at_k(retrieved, gold, 4) == 1.0


def test_precision_divides_by_what_was_retrieved_not_by_k():
    """`hits / len(top)` could become `hits / k`. They differ only when fewer than k documents
    came back, which the demo corpus (6 chunks) reaches at k > 6."""
    assert precision_at_k(["a"], ["a"], 3) == 1.0, "one retrieved doc, and it is right: 1.0"
    assert precision_at_k(["a", "b"], ["a"], 5) == 0.5
    assert precision_at_k([], ["a"], 3) == 0.0


def test_citation_present_can_report_absence():
    """`return 1.0 if citations else 0.0` could become `return 1.0`. Nothing passed it an
    empty list, and `cli eval --k 0` reaches exactly that."""
    assert citation_present(["c1"]) == 1.0
    assert citation_present([]) == 0.0


# ─────────────────────────────── the aggregate mean ───────────────────────────────

def test_the_aggregate_mean_is_a_mean_and_not_a_constant():
    """`sum(vals) / len(vals)` could become `1.0`. Over EVAL_SET the real precision@k is 0.25,
    so a constant 1.0 is a fourfold overstatement of the headline number this repo publishes."""
    run = evaluate(EVAL_SET, _stub_answer, k=4)
    assert run.metrics["precision@k"] == pytest.approx(0.25, abs=1e-3)
    assert run.metrics["precision@k"] != 1.0


def test_a_failing_case_is_flagged_and_the_flag_reaches_the_run():
    """`flagged=flagged` could become `flagged=False`. The CI gate counts flags on the PLANTED
    case, which cli builds outside evaluate(), so the gate could not see this either."""
    run = evaluate(EVAL_SET, _ungrounded_answer, k=4)
    assert any(c.flagged for c in run.cases), (
        "an answer with no support in its contexts must be flagged")


def test_a_well_grounded_run_flags_nothing():
    """The other side, so the flag cannot be satisfied by always setting it."""
    run = evaluate(EVAL_SET, _stub_answer, k=4)
    assert not all(c.flagged for c in run.cases)


# ─────────────────────────────── rank fusion ───────────────────────────────

def test_fusion_uses_rank_and_is_not_a_vote_count():
    """`1.0 / (c + rank)` could become `1.0`, turning RRF into counting appearances. These two
    rankings give every document the same number of votes, so only rank can separate them."""
    a = ["TOP", "x", "y", "z", "LOW"]
    b = ["TOP", "p", "q", "r", "LOW"]
    fused = dict(reciprocal_rank_fusion([a, b], k=10))
    assert fused["TOP"] > fused["LOW"], (
        "a document ranked first by both retrievers must beat one ranked last by both")


def test_fusion_ranks_a_doc_both_retrievers_liked_above_one_only_a_single_liked():
    """The property the docstring claims: agreement wins."""
    both = ["AGREED", "solo-a"]
    other = ["AGREED", "solo-b"]
    order = [doc for doc, _ in reciprocal_rank_fusion([both, other], k=10)]
    assert order[0] == "AGREED"


# ─────────────────────────────── helpers ───────────────────────────────

class _Ans:
    def __init__(self, text, retrieved, citations, contexts=None):
        self.text = text
        self.retrieved = retrieved
        self.citations = citations
        self.contexts = contexts if contexts is not None else [text]


def _stub_answer(q: str):
    """A plausible grounded answer: the gold doc first, and the answer echoes its context."""
    gold = _gold_for(q)
    return _Ans(text="supporting sentence for " + q, retrieved=[gold, "x", "y", "z"],
                citations=[gold], contexts=["supporting sentence for " + q])


def _ungrounded_answer(q: str):
    """An answer with nothing behind it, which is what the flag exists to catch."""
    gold = _gold_for(q)
    return _Ans(text="an unrelated assertion about quasars and tax law",
                retrieved=[gold, "x", "y", "z"], citations=[gold],
                contexts=["supporting sentence for " + q])


def _gold_for(q: str) -> str:
    for item in EVAL_SET:
        if item["q"] == q:
            ids = item.get("gold_ids") or []
            return ids[0] if ids else "g"
    return "g"


# ─────────────────────────────── the reranker ───────────────────────────────

from ragevallab.retrieval import LexicalReranker


def test_coverage_is_query_recall_not_passage_precision():
    """`len(q_terms & terms) / len(q_terms)` could become `/ len(terms)`. The existing fixture
    cannot separate them: 'alpha beta gamma' beats 'alpha only' under either denominator.

    Here a SHORT passage covering half the query must lose to a LONG one covering all of it.
    Dividing by the passage's own term count inverts that, because the short passage looks
    denser.
    """
    rr = LexicalReranker(coverage=1.0, phrase=0.0, prior=0.0)
    ranked = dict(rr.rerank("alpha beta", [
        ("short-half", "alpha"),                                  # 1 of 2 query terms, 1 term
        ("long-full", "alpha beta delta epsilon zeta eta theta"),  # 2 of 2 query terms, 7 terms
    ], k=10))
    assert ranked["long-full"] > ranked["short-half"], (
        "covering the whole query must beat covering half of it, however long the passage")


def test_the_retrieval_prior_actually_moves_the_order():
    """`prior = 1.0 - rank / n` could become `prior = 0.0` and the test named for it still
    passed, because its fixture decided the outcome on coverage alone.

    With coverage and phrase switched off, the prior is the ONLY term left, so retrieval order
    has to survive into the reranked order.
    """
    rr = LexicalReranker(coverage=0.0, phrase=0.0, prior=1.0)
    scored = dict(rr.rerank("anything", [
        ("first", "unrelated text"), ("second", "unrelated text"), ("third", "unrelated text"),
    ], k=10))
    # Assert the SCORES, not the order. Asserting order is not enough: with the prior deleted
    # every score is 0.0 and Python's stable sort preserves the input order, so an
    # order-only assertion passes against the very mutation it is written to catch.
    assert scored["first"] == pytest.approx(1.0)
    assert scored["second"] == pytest.approx(2 / 3)
    assert scored["third"] == pytest.approx(1 / 3)


def test_coverage_still_outranks_the_prior_at_default_weights():
    """The other direction, so the prior cannot be widened until it swamps the signal the
    reranker exists for."""
    rr = LexicalReranker()
    ranked = dict(rr.rerank("alpha beta", [
        ("retrieved-first-but-empty", "nothing relevant here"),
        ("retrieved-last-but-exact", "alpha beta"),
    ], k=10))
    assert ranked["retrieved-last-but-exact"] > ranked["retrieved-first-but-empty"]


# ─────────────────────────────── the API honours k ───────────────────────────────

def test_the_query_endpoint_honours_the_callers_k():
    """`k=body.k` could become `k=1`: /query then returns one chunk for k=1, 4 and 6 alike."""
    from fastapi.testclient import TestClient

    from ragevallab.api import create_app
    client = TestClient(create_app())
    seen = {}
    for k in (1, 3, 6):
        r = client.post("/query", json={"query": "Which planet is the hottest?", "k": k})
        assert r.status_code == 200, r.text
        seen[k] = len(r.json()["retrieved"])
    assert len(set(seen.values())) > 1, f"/query returned the same count for every k: {seen}"
    assert seen[1] < seen[6], seen


def test_the_eval_endpoint_honours_the_callers_k():
    """`compute_eval_run((body or EvalIn()).k)` could become `compute_eval_run(4)`, so /eval
    would report the k=4 numbers whatever the caller asked for."""
    from fastapi.testclient import TestClient

    from ragevallab.api import create_app
    client = TestClient(create_app())
    a = client.post("/eval", json={"k": 1}).json()
    b = client.post("/eval", json={"k": 6}).json()
    assert a["metrics"]["precision@k"] != b["metrics"]["precision@k"], (
        f"/eval reported identical metrics for k=1 and k=6: {a['metrics']}")


# ─────────────────────────── the emitted run's own numbers ───────────────────────────

def test_flagged_cases_counts_the_flags_it_reports():
    """`float(sum(1 for c in run.cases if c.flagged))` could be replaced by the literal 1.0 and
    nothing noticed, because the CI gate counts case-level flags rather than reading this
    metric. It is the number eval-history ingests and eval-dashboard renders."""
    from ragevallab.cli import compute_eval_run
    run = compute_eval_run(4)
    counted = sum(1 for c in run.cases if c.flagged)
    assert run.metrics["flagged_cases"] == float(counted), (
        f"metric says {run.metrics['flagged_cases']}, the cases say {counted}")
    # Honest limit: on the shipped corpus this count is 1.0 for every k in 1..20, so replacing
    # the sum with the literal 1.0 is an EQUIVALENT mutant against every input reachable here.
    # The assertion is still the right one (the metric must be derived from the cases), but it
    # cannot discriminate until the corpus contains a second flaggable case.
    assert run.metrics["n_cases"] == float(len(run.cases))


def test_the_planted_hallucination_is_the_flagged_one():
    """The gate's claim, asserted on the emitted run rather than on a count: the planted case
    is flagged and it is the reason the count is not zero."""
    from ragevallab.cli import compute_eval_run
    run = compute_eval_run(4)
    flagged = [c for c in run.cases if c.flagged]
    assert flagged, "the planted hallucination is not being caught"
    assert run.metrics["flagged_cases"] >= 1.0
