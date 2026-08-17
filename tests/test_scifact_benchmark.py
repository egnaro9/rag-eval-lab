"""The README's most quotable claim, as an enforced test.

"A from-scratch BM25 matches the published baseline (0.664 vs 0.665)" was, until
2026-08-17, the only number in this repo with no guard: no dataset on the dev
machine, no CI job running `cli benchmark`, and test_benchmark.py exercising a
tiny inline fixture. Everything else here is CI-enforced; the line most likely
to be quoted in an interview rested on one hand run.

This test runs the real thing on the real SciFact test split. It is skipped
when the dataset directory is absent, EXCEPT under CI, where the workflow
downloads SciFact first and absence is a hard error: a benchmark claim whose
proof silently skipped is exactly the vacuous-pass failure the rest of this
portfolio exists to catch.

Dataset: BEIR SciFact (CC BY-SA 4.0), 5,183 abstracts, 300 judged test claims.
The bound asserted is two-sided: high enough to pin the published-baseline
match, and an upper bound too, because a from-scratch BM25 scoring far above
the published number would mean the scorer is broken in our favor, which is
the worse direction.
"""
import os
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCIFACT = ROOT / "scifact"

if not SCIFACT.is_dir() and os.environ.get("CI"):
    raise RuntimeError(
        "scifact/ is missing but CI is set. The benchmark test would skip and "
        "the suite would report green with the README's headline number "
        "unchecked. The workflow must download SciFact before pytest runs; "
        "see .github/workflows/ci.yml.")

needs_scifact = pytest.mark.skipif(
    not SCIFACT.is_dir(),
    reason="SciFact dataset not present (local dev only; absence is a hard "
           "error under CI). Fetch: BEIR scifact.zip -> ./scifact")


@needs_scifact
def test_bm25_matches_the_published_scifact_baseline():
    from ragevallab.benchmark import load_beir, run_benchmark

    data = load_beir(SCIFACT, split="test")
    assert len(data.qrels) == 300, "judged-query count changed; wrong dataset?"

    r = run_benchmark(data, k=10, strategy="bm25")
    ndcg = r["ndcg@10"]

    # Published BM25 on SciFact: 0.665. Ours measured 0.6644 on 2026-08-17.
    # Window is deliberately tight: a drop below 0.66 means the retriever or
    # scorer regressed; above 0.68 means the scorer started flattering us.
    assert 0.66 <= ndcg <= 0.68, (
        f"BM25 ndcg@10 = {ndcg}; README claims 0.664 vs published 0.665. "
        "If this moved, the README number is now false: fix the code or the "
        "claim, never the assertion window alone.")
