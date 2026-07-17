"""nDCG, checked against numbers worked out by hand.

The point of nDCG here is that it's what published benchmarks report — so it has
to mean the same thing they mean. A metric named `ndcg@10` that computes
something subtly different is worse than not having one, because the comparison
it invites is then false.
"""
import math

from ragevallab.evals import ndcg_at_k


def test_gold_at_rank_one_is_perfect():
    assert ndcg_at_k(["a", "b", "c"], ["a"], 10) == 1.0


def test_rank_matters_which_is_the_whole_point():
    """precision@k can't see this difference; that's why nDCG exists."""
    first = ndcg_at_k(["a", "b", "c"], ["a"], 10)
    second = ndcg_at_k(["b", "a", "c"], ["a"], 10)
    third = ndcg_at_k(["b", "c", "a"], ["a"], 10)
    assert first > second > third
    assert math.isclose(second, 1 / math.log2(3))   # DCG = 1/log2(2+1), IDCG = 1
    assert math.isclose(third, 1 / math.log2(4))


def test_a_miss_scores_zero():
    assert ndcg_at_k(["b", "c"], ["a"], 10) == 0.0


def test_idcg_uses_available_gold_not_k():
    """One gold document ranked first is a perfect result, not 1/k of one.

    If IDCG assumed k relevant docs existed, a query with a single answer could
    never score 1.0 — every SciFact query would be punished for the nine slots
    it had nothing to put in.
    """
    assert ndcg_at_k(["a"] + [f"x{i}" for i in range(9)], ["a"], 10) == 1.0


def test_two_gold_in_the_best_possible_order():
    assert ndcg_at_k(["a", "b", "c"], ["a", "b"], 10) == 1.0


def test_two_gold_in_the_worst_order_still_beats_missing_them():
    both_late = ndcg_at_k(["x", "y", "a", "b"], ["a", "b"], 10)
    assert 0 < both_late < 1.0


def test_k_truncates():
    """A hit outside k is not a hit."""
    assert ndcg_at_k(["x", "y", "a"], ["a"], 2) == 0.0
    assert ndcg_at_k(["x", "y", "a"], ["a"], 3) > 0.0


def test_no_gold_is_vacuously_perfect():
    # Consistent with recall_at_k's convention for an empty gold set.
    assert ndcg_at_k(["a"], [], 10) == 1.0


def test_empty_retrieval_scores_zero():
    assert ndcg_at_k([], ["a"], 10) == 0.0


def test_duplicate_gold_ids_do_not_inflate_idcg():
    """A malformed qrel listing the same doc twice must not deflate the score."""
    assert ndcg_at_k(["a", "b"], ["a", "a"], 10) == 1.0
