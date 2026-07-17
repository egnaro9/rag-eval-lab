"""Command-line entry point.

    python -m ragevallab.cli eval [--k 4] [--out eval_run.json]

Runs the offline RAG pipeline over the demo eval set, appends one *planted*
hallucination case to prove the harness bites, and writes a JSON report that
the companion eval-dashboard can render.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List

from .data import EVAL_SET, PLANTED, SAMPLE_DOCS
from .evals import CaseResult, EvalRun, evaluate, faithfulness, precision_at_k, recall_at_k
from .pipeline import Answer, RagPipeline


def _build_planted_case(pipe: RagPipeline, k: int, threshold: float) -> CaseResult:
    """Retrieve real context but return the hallucinated answer, then score it."""
    real = pipe.answer(PLANTED["q"], k=k)
    faith = faithfulness(PLANTED["hallucinated_answer"], real.contexts)
    return CaseResult(
        q=PLANTED["q"],
        answer=PLANTED["hallucinated_answer"],
        retrieved=real.retrieved,
        citations=real.citations,
        scores={
            "precision@k": round(precision_at_k(real.retrieved, PLANTED["gold_ids"], k), 3),
            "recall@k": round(recall_at_k(real.retrieved, PLANTED["gold_ids"], k), 3),
            "citation": 1.0 if real.citations else 0.0,
            "faithfulness": round(faith, 3),
        },
        flagged=faith < threshold,
        note=PLANTED["note"],
    )


def run_eval(k: int = 4, out: str = "eval_run.json") -> EvalRun:
    pipe = RagPipeline().ingest(SAMPLE_DOCS)
    run = evaluate(EVAL_SET, lambda q: pipe.answer(q, k=k), k=k)
    # Append the planted hallucination and refresh the aggregate metrics.
    run.cases.append(_build_planted_case(pipe, k, threshold=0.6))
    run.metrics["n_cases"] = float(len(run.cases))
    run.metrics["flagged_cases"] = float(sum(1 for c in run.cases if c.flagged))
    for key, mkey in (
        ("faithfulness", "faithfulness"),
        ("precision@k", "precision@k"),
        ("recall@k", "recall@k"),
        ("citation", "citation_rate"),
    ):
        vals = [c.scores[key] for c in run.cases]
        run.metrics[mkey] = round(sum(vals) / len(vals), 3) if vals else 0.0
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(run.to_dict(), fh, indent=2)
    return run


def _print_summary(run: EvalRun) -> None:
    print(f"run: {run.run}")
    for key, val in run.metrics.items():
        print(f"  {key:>14}: {val}")
    flagged = [c for c in run.cases if c.flagged]
    if flagged:
        print(f"\n{len(flagged)} flagged case(s):")
        for c in flagged:
            print(f"  ! {c.q}")
            print(f"    answer: {c.answer}")
            print(f"    faithfulness={c.scores['faithfulness']}  ({c.note})")


def run_bench(data_dir: str, k: int = 10, split: str = "test", out: str | None = None) -> dict:
    """Score the retriever on a public benchmark and print it next to the references."""
    from .benchmark import load_beir, run_benchmark

    data = load_beir(data_dir, split=split)
    print(f"loaded {len(data.docs)} docs, {len(data.queries)} judged queries ({split})")

    def tick(n: int, total: int) -> None:
        print(f"  {n}/{total}", end="\r", flush=True)

    result = run_benchmark(data, k=k, progress=tick)
    print(" " * 24, end="\r")

    print(f"\nSciFact-style benchmark — {result['n_queries']} queries over "
          f"{result['n_docs']} docs ({result['n_chunks']} chunks)")
    for key in (f"ndcg@{k}", f"precision@{k}", f"recall@{k}"):
        print(f"  {key:>14}: {result[key]}")
    # State the comparison the number exists for. A benchmark score with nothing
    # to compare against isn't a measurement, it's a decoration.
    print("\n  for reference, published on SciFact:")
    print("    BM25            ndcg@10  0.665")
    print("    dense retrievers ndcg@10 ~0.65-0.70")

    if out:
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print(f"\nwrote {out}")
    return result


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ragevallab", description="RAG eval lab")
    sub = parser.add_subparsers(dest="cmd")
    ev = sub.add_parser("eval", help="run the eval suite and write a JSON report")
    ev.add_argument("--k", type=int, default=4)
    ev.add_argument("--out", default="eval_run.json")

    bm = sub.add_parser("benchmark", help="score the retriever on a BEIR-format dataset")
    bm.add_argument("--data", required=True, help="dataset dir (corpus.jsonl, queries.jsonl, qrels/)")
    bm.add_argument("--k", type=int, default=10, help="nDCG@k (10 is what BEIR reports)")
    bm.add_argument("--split", default="test")
    bm.add_argument("--out", default=None, help="write metrics as JSON")

    args = parser.parse_args(argv)

    if args.cmd == "eval":
        run = run_eval(k=args.k, out=args.out)
        _print_summary(run)
        print(f"\nwrote {args.out}")
        return 0
    if args.cmd == "benchmark":
        run_bench(args.data, k=args.k, split=args.split, out=args.out)
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
