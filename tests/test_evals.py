from ragevallab.cli import run_eval
from ragevallab.data import EVAL_SET, SAMPLE_DOCS
from ragevallab.evals import (
    FAITHFULNESS_THRESHOLD,
    evaluate,
    faithfulness,
    precision_at_k,
    recall_at_k,
)
from ragevallab.pipeline import RagPipeline


def test_precision_and_recall_math():
    assert precision_at_k(["a", "b", "c"], ["a"], k=3) == 1 / 3
    assert precision_at_k([], ["a"], k=3) == 0.0
    assert recall_at_k(["a", "b"], ["a", "c"], k=2) == 0.5
    assert recall_at_k(["a"], [], k=1) == 1.0


def test_faithfulness_grounded_vs_hallucinated():
    context = ["Venus is the hottest planet with a thick carbon dioxide atmosphere."]
    grounded = faithfulness("Venus is the hottest planet.", context)
    hallucinated = faithfulness("Neptune erupts with volcanic geysers.", context)
    assert grounded > hallucinated
    assert grounded >= FAITHFULNESS_THRESHOLD
    assert hallucinated < FAITHFULNESS_THRESHOLD


def test_evaluate_on_grounded_pipeline_scores_well():
    pipe = RagPipeline().ingest(SAMPLE_DOCS)
    run = evaluate(EVAL_SET, lambda q: pipe.answer(q, k=4), k=4)
    assert run.metrics["n_cases"] == float(len(EVAL_SET))
    # Extractive answers are drawn from context -> perfectly faithful.
    assert run.metrics["faithfulness"] == 1.0
    assert run.metrics["citation_rate"] == 1.0
    # Every question's gold chunk is retrieved.
    assert run.metrics["recall@k"] == 1.0
    assert run.metrics["flagged_cases"] == 0.0


def test_planted_hallucination_is_flagged(tmp_path):
    out = tmp_path / "eval_run.json"
    run = run_eval(k=4, out=str(out))
    assert out.exists()
    planted = [c for c in run.cases if "PLANTED" in c.note]
    assert len(planted) == 1
    assert planted[0].flagged is True
    assert planted[0].scores["faithfulness"] < FAITHFULNESS_THRESHOLD
    # The harness surfaces exactly the one bad case.
    assert run.metrics["flagged_cases"] == 1.0
