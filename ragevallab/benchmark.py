"""Measure the retriever against a public benchmark, in BEIR's format.

The bundled eval suite has six questions. It proves the harness catches a
planted hallucination; it proves nothing about whether the retriever is any
good, because six hand-written questions can be quietly tuned into agreeing
with you.

A public benchmark can't be. SciFact is 5,183 scientific abstracts and 300 test
claims with human relevance judgements, and BM25 scores nDCG@10 ≈ 0.665 on it —
a number published by people who have never heard of this repo. Running the same
metric on the same data makes the retriever's quality checkable instead of
asserted.

    python -m ragevallab.cli benchmark --data ./scifact

Reads BEIR's layout, which is what you get from the public download:

    corpus.jsonl      {"_id", "title", "text"}
    queries.jsonl     {"_id", "text"}
    qrels/test.tsv    query-id \\t corpus-id \\t score   (header row)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional


@dataclass
class BenchmarkData:
    """A loaded BEIR-format dataset."""

    docs: List[dict]                  # {"id", "text"} — title and body joined
    queries: Dict[str, str]           # query id -> text
    qrels: Dict[str, List[str]]       # query id -> relevant doc ids

    def __repr__(self) -> str:  # keeps a REPL honest about size
        return (f"BenchmarkData(docs={len(self.docs)}, "
                f"queries={len(self.queries)}, judged={len(self.qrels)})")


def _jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_beir(root: str | Path, split: str = "test") -> BenchmarkData:
    """Load a BEIR-format dataset from disk.

    Only the judged queries are returned. BEIR ships every query in
    `queries.jsonl` (SciFact: 1,109) but judges a subset per split (test: 300).
    Scoring the unjudged ones would silently average in zeros and report a
    retriever as three times worse than it is.
    """
    root = Path(root)
    corpus_path, queries_path = root / "corpus.jsonl", root / "queries.jsonl"
    qrels_path = root / "qrels" / f"{split}.tsv"
    for p in (corpus_path, queries_path, qrels_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found — is {root} a BEIR dataset?")

    docs = []
    for d in _jsonl(corpus_path):
        title, text = (d.get("title") or "").strip(), (d.get("text") or "").strip()
        # Title carries real signal in SciFact; a lexical retriever should see it.
        docs.append({"id": str(d["_id"]), "text": f"{title}. {text}".strip(". ")})

    qrels: Dict[str, List[str]] = {}
    with qrels_path.open(encoding="utf-8") as fh:
        header = next(fh, "")
        if not header.lower().startswith("query-id"):
            fh.seek(0)                       # some dumps have no header
        for line in fh:
            parts = line.split()
            if len(parts) < 3:
                continue
            qid, doc_id, score = parts[0], parts[1], parts[2]
            if int(score) > 0:
                qrels.setdefault(qid, []).append(doc_id)

    all_queries = {str(q["_id"]): q["text"] for q in _jsonl(queries_path)}
    queries = {qid: all_queries[qid] for qid in qrels if qid in all_queries}

    return BenchmarkData(docs=docs, queries=queries, qrels=qrels)


def docs_from_chunks(chunk_ids: Iterable[str]) -> List[str]:
    """Chunk ids -> the document ranking they imply.

    The pipeline retrieves *chunks* (`31715818#0`); SciFact judges *documents*
    (`31715818`). Scoring chunk ids against doc ids would report a flat zero and
    look like a broken retriever rather than a units mismatch.

    A document's rank is its best chunk's rank, and duplicates collapse — three
    chunks of one abstract is one document found, not three. Order is preserved,
    which is the part nDCG cares about.
    """
    seen, out = set(), []
    for cid in chunk_ids:
        doc_id = cid.split("#", 1)[0]
        if doc_id not in seen:
            seen.add(doc_id)
            out.append(doc_id)
    return out


def run_benchmark(data: BenchmarkData, k: int = 10, max_sentences: int = 2,
                  progress: Optional[Callable[[int, int], None]] = None) -> dict:
    """Score the real pipeline over a benchmark. Returns BEIR-comparable metrics.

    Retrieves `k * 4` chunks so that, after collapsing to documents, there are
    still k documents to score — otherwise several chunks of one abstract would
    crowd the top-k and depress nDCG for a reason that has nothing to do with
    retrieval quality.
    """
    from .evals import ndcg_at_k, precision_at_k, recall_at_k
    from .pipeline import RagPipeline

    pipe = RagPipeline().ingest({d["id"]: d["text"] for d in data.docs},
                                max_sentences=max_sentences)

    ndcgs, precs, recalls = [], [], []
    items = list(data.queries.items())
    for n, (qid, text) in enumerate(items, 1):
        gold = data.qrels.get(qid, [])
        chunks = [r.id for r in pipe.retrieve(text, k=k * 4)]
        ranked = docs_from_chunks(chunks)[:k]
        ndcgs.append(ndcg_at_k(ranked, gold, k))
        precs.append(precision_at_k(ranked, gold, k))
        recalls.append(recall_at_k(ranked, gold, k))
        if progress and n % 25 == 0:
            progress(n, len(items))

    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else 0.0  # noqa: E731
    return {
        f"ndcg@{k}": mean(ndcgs),
        f"precision@{k}": mean(precs),
        f"recall@{k}": mean(recalls),
        "n_queries": len(items),
        "n_docs": len(data.docs),
        "n_chunks": len(pipe.store),
    }
