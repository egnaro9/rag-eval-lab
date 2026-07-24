"""A deterministic RAG evaluation harness.

No LLM-as-judge here — every metric is a closed-form function of the
retrieved chunk ids and the answer text, so results are reproducible and CI
never flakes. That is the whole idea behind the differential-oracle style of
testing: make the check something a machine can run the same way every time.

Metrics
-------
- ``precision_at_k`` / ``recall_at_k`` : retrieval quality vs. gold chunk ids.
- ``ndcg_at_k``                       : rank-aware retrieval quality — what the
                                         published benchmarks report, so the
                                         number can be compared to theirs.
- ``citation_present``                 : did the answer cite at least one source?
- ``faithfulness``                     : fraction of the answer's content tokens
                                         that are actually supported by the
                                         retrieved context (lexical grounding).
                                         A hallucinated answer scores low.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Sequence

# The grounding metric now lives in the shared gradecore engine — the same
# deterministic, no-LLM-judge core that model-drift and the crash-test platform
# use. It was extracted verbatim (identical tokenizer, stoplist and math), so the
# SciFact numbers below are unchanged; this repo just no longer keeps its own copy.
from gradecore.grounding import FAITHFULNESS_THRESHOLD, grounding_score


def precision_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    top = retrieved_ids[:k]
    if not top:
        return 0.0
    hits = sum(1 for i in top if i in set(gold_ids))
    return hits / len(top)


def recall_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    gold = set(gold_ids)
    if not gold:
        return 1.0
    top = set(retrieved_ids[:k])
    return len(top & gold) / len(gold)


def ndcg_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int = 10) -> float:
    """Normalised discounted cumulative gain — rank-aware retrieval quality.

    precision@k and recall@k can't tell a gold document at rank 1 from the same
    document at rank 10; both are "a hit in the top k". nDCG can, by discounting
    each hit by log2(rank + 1), and that difference is the whole user experience
    of a retriever.

    It's also the metric the published benchmarks report, which is the point of
    having it: a number nobody else measures is a number nobody can check. This
    is the standard binary-relevance formulation (BEIR's), so `ndcg@10` here
    means what `ndcg@10` means in a paper.

    IDCG uses min(len(gold), k) — the best achievable ordering given how many
    relevant documents actually exist. A query with one gold document scores 1.0
    for ranking it first, rather than being punished for the k-1 slots it had no
    way to fill.
    """
    if k <= 0:
        return 0.0
    gold = set(gold_ids)
    if not gold:
        return 1.0     # nothing to find; consistent with recall_at_k

    dcg = sum(1.0 / math.log2(rank + 2)
              for rank, doc_id in enumerate(retrieved_ids[:k])
              if doc_id in gold)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(min(len(gold), k)))
    return dcg / idcg if idcg else 0.0


def citation_present(citations: Sequence[str]) -> float:
    return 1.0 if citations else 0.0


def faithfulness(answer_text: str, contexts: Sequence[str]) -> float:
    """Fraction of the answer's content tokens grounded in the context.

    A thin alias for gradecore's shared ``grounding_score`` — kept as a local
    name so the rest of the harness (and its tests) stay unchanged.
    """
    return grounding_score(answer_text, contexts)


@dataclass
class CaseResult:
    q: str
    answer: str
    retrieved: List[str]
    citations: List[str]
    scores: Dict[str, float]
    flagged: bool
    note: str = ""


@dataclass
class EvalRun:
    run: str
    metrics: Dict[str, float]
    cases: List[CaseResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def evaluate(
    dataset: List[dict],
    answer_fn: Callable[[str], object],
    k: int = 4,
    run_name: str = "rag-eval-lab",
    faithfulness_threshold: float = FAITHFULNESS_THRESHOLD,
) -> EvalRun:
    """Run ``answer_fn`` over each ``{"q", "gold_ids"}`` case and score it.

    ``answer_fn(q)`` must return an object with ``.text``, ``.citations``,
    ``.retrieved`` (chunk ids) and ``.contexts`` (chunk texts) — i.e. a
    ``ragevallab.pipeline.Answer``.
    """
    cases: List[CaseResult] = []
    for item in dataset:
        q = item["q"]
        gold = item.get("gold_ids", [])
        ans = answer_fn(q)
        p = precision_at_k(ans.retrieved, gold, k)  # type: ignore[attr-defined]
        r = recall_at_k(ans.retrieved, gold, k)  # type: ignore[attr-defined]
        cite = citation_present(ans.citations)  # type: ignore[attr-defined]
        faith = faithfulness(ans.text, ans.contexts)  # type: ignore[attr-defined]
        flagged = faith < faithfulness_threshold
        cases.append(
            CaseResult(
                q=q,
                answer=ans.text,  # type: ignore[attr-defined]
                retrieved=list(ans.retrieved),  # type: ignore[attr-defined]
                citations=list(ans.citations),  # type: ignore[attr-defined]
                scores={
                    "precision@k": round(p, 3),
                    "recall@k": round(r, 3),
                    "citation": cite,
                    "faithfulness": round(faith, 3),
                },
                flagged=flagged,
                note=item.get("note", ""),
            )
        )

    def _mean(key: str) -> float:
        vals = [c.scores[key] for c in cases]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    metrics = {
        "precision@k": _mean("precision@k"),
        "recall@k": _mean("recall@k"),
        "citation_rate": _mean("citation"),
        "faithfulness": _mean("faithfulness"),
        "flagged_cases": float(sum(1 for c in cases if c.flagged)),
        "n_cases": float(len(cases)),
    }
    return EvalRun(run=run_name, metrics=metrics, cases=cases)
