"""Retrieval strategies, checked on cases small enough to reason about by hand.

The SciFact number (BM25 nDCG@10 0.664, matching the published baseline) is only
trustworthy if BM25 itself is right — so these pin the behaviour that makes it
right: length normalisation, term saturation, and idf that de-weights common
terms. If BM25 silently broke, the benchmark would still print *a* number, just
the wrong one.
"""
from ragevallab.retrieval import BM25Index, LexicalReranker, reciprocal_rank_fusion


def test_bm25_ranks_the_on_topic_document_first():
    idx = BM25Index().fit({
        "venus": "Venus is the hottest planet in the solar system.",
        "mars": "Mars is a cold red planet with the tallest volcano.",
        "off": "Shakespeare wrote many plays including Hamlet.",
    })
    top = idx.search("which planet is hottest", k=3)
    assert top[0][0] == "venus"
    assert "off" not in [i for i, _ in top]  # no query terms → not returned


def test_bm25_length_normalisation_prefers_the_focused_document():
    """Same term, two docs — the shorter, more focused one should win.

    Plain TF-IDF cosine doesn't reliably do this; BM25's `b` is the whole point.
    """
    idx = BM25Index().fit({
        "short": "quantum computing.",
        "long": "quantum computing " + "and many other unrelated topics " * 30,
    })
    top = idx.search("quantum computing", k=2)
    assert top[0][0] == "short"


def test_bm25_saturation_caps_repeated_terms():
    """A term repeated 100x should not score 100x a single occurrence."""
    idx = BM25Index().fit({"a": "spam", "b": "spam " * 100})
    scores = dict(idx.search("spam", k=2))
    assert scores["b"] < 5 * scores["a"]  # saturated, not linear


def test_bm25_common_term_carries_little_weight():
    idx = BM25Index().fit({f"d{i}": "the common word" for i in range(10)} | {"rare": "the unicorn word"})
    # "the" is in every doc (idf~0); "unicorn" only in one — it should decide.
    top = idx.search("the unicorn", k=1)
    assert top[0][0] == "rare"


def test_rrf_rewards_agreement_between_rankers():
    # B is #2 in BOTH lists; A is #1 in one but absent from the other.
    # Being liked by both beats being loved by one and unseen by the other.
    fused = reciprocal_rank_fusion([["A", "B", "C"], ["D", "B", "E"]], k=3)
    ids = [i for i, _ in fused]
    assert ids[0] == "B"


def test_rrf_empty_lists_are_safe():
    assert reciprocal_rank_fusion([[], []], k=5) == []


def test_reranker_prior_keeps_a_strong_retrieval_hit_on_top():
    """With no lexical reason to move it, the retriever's #1 stays #1."""
    rr = LexicalReranker(coverage=1.0, phrase=0.5, prior=1.0)
    cands = [("first", "exactly the query words here"), ("second", "unrelated text about nothing")]
    out = rr.rerank("the query words", cands, k=2)
    assert out[0][0] == "first"


def test_reranker_promotes_fuller_query_coverage():
    """A passage covering all query terms beats one covering half, at equal rank."""
    rr = LexicalReranker(coverage=2.0, phrase=0.0, prior=0.0)
    cands = [("half", "alpha only"), ("full", "alpha beta gamma")]
    out = rr.rerank("alpha beta gamma", cands, k=2)
    assert out[0][0] == "full"
