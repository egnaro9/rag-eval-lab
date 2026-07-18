"""Retrieval strategies — the part that decides *which* chunks come back.

The pipeline's default is TF-IDF cosine, which finds the right document about
three-quarters of the time on SciFact but misranks it (recall@10 0.73, nDCG@10
0.58). The gap is ranking, and ranking is exactly what these add:

- **BM25** — the standard lexical baseline, with the term-frequency saturation
  and document-length normalisation that plain TF-IDF cosine lacks. This is the
  thing that beat the cosine retriever on the benchmark, and it's the published
  reference number, reimplemented here rather than cited.
- **Hybrid (reciprocal rank fusion)** — BM25 and TF-IDF disagree on different
  queries; RRF combines their *rankings* (not their incomparable scores) so a
  document both rate highly rises, with no tuning knob to overfit.
- **A lexical reranker** — a cheap second pass over the top candidates that
  rescores on how completely and how tightly the query's terms appear, catching
  the case where the right passage has all the query words but scattered.

Everything here is stdlib-only and deterministic, like the rest of the lab: the
same input yields the same ranking, so a benchmark number is reproducible and CI
never flakes. A neural cross-encoder reranker would drop in behind the same
`Reranker` interface; it's left out on purpose so the default needs no model.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from .embedder import tokenize


# ─────────────────────────────── BM25 ────────────────────────────────
@dataclass
class BM25Index:
    """Okapi BM25 over pre-tokenised documents.

    k1 controls term-frequency saturation (how fast repeated terms stop helping);
    b controls length normalisation (how much a long document is penalised).
    The 1.5 / 0.75 defaults are the standard ones the reference scores use.
    """

    k1: float = 1.5
    b: float = 0.75
    ids: List[str] = field(default_factory=list)
    _docs: List[List[str]] = field(default_factory=list)
    _idf: Dict[str, float] = field(default_factory=dict)
    _tf: List[Counter] = field(default_factory=list)
    _avg_len: float = 0.0
    _postings: Dict[str, List[int]] = field(default_factory=dict)

    def fit(self, docs: Dict[str, str]) -> "BM25Index":
        self.ids = list(docs)
        self._docs = [tokenize(t) for t in docs.values()]
        self._tf = [Counter(d) for d in self._docs]
        n = len(self._docs) or 1
        self._avg_len = sum(len(d) for d in self._docs) / n

        df: Counter = Counter()
        self._postings = defaultdict(list)
        for i, toks in enumerate(self._docs):
            for term in set(toks):
                df[term] += 1
                self._postings[term].append(i)
        # BM25's idf; the +0.5/+0.5 form, floored at 0 so a term in almost every
        # document can't push a score negative.
        self._idf = {t: max(0.0, math.log((n - c + 0.5) / (c + 0.5) + 1.0)) for t, c in df.items()}
        return self

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        q_terms = [t for t in tokenize(query) if t in self._idf]
        if not q_terms:
            return []
        # Only score documents that contain at least one query term.
        candidates = set()
        for t in q_terms:
            candidates.update(self._postings.get(t, ()))

        scores: List[Tuple[str, float]] = []
        for i in candidates:
            tf, dl = self._tf[i], len(self._docs[i])
            norm = self.k1 * (1 - self.b + self.b * dl / (self._avg_len or 1))
            s = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if f:
                    s += self._idf[term] * (f * (self.k1 + 1)) / (f + norm)
            if s > 0:
                scores.append((self.ids[i], s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


# ──────────────────────── hybrid: rank fusion ────────────────────────
def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = 10, c: int = 60
) -> List[Tuple[str, float]]:
    """Fuse several ranked id-lists into one. RRF score = Σ 1/(c + rank).

    Ranks, not scores, because BM25 and cosine live on different scales and
    averaging them is meaningless. `c` (60, the standard) damps how much the very
    top ranks dominate. A document ranked well by *both* retrievers wins.
    """
    agg: Dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            agg[doc_id] += 1.0 / (c + rank)
    fused = sorted(agg.items(), key=lambda x: x[1], reverse=True)
    return fused[:k]


# ─────────────────────────── the reranker ────────────────────────────
def _bigrams(toks: Sequence[str]) -> set:
    return set(zip(toks, toks[1:]))


class LexicalReranker:
    """Rescore a candidate set on how completely and tightly the query matches.

    Retrieval scores reward the presence of query terms; this rewards their
    *completeness* (how many distinct query terms the passage covers) and their
    *adjacency* (shared query bigrams), which is what separates a passage that is
    actually about the query from one that merely mentions its words in passing.
    A blend, not a replacement — the retriever's own ranking is kept as a prior so
    a strong lexical match can't be dislodged by a keyword-stuffed passage.
    """

    def __init__(self, coverage: float = 1.0, phrase: float = 0.5, prior: float = 0.3) -> None:
        self.coverage, self.phrase, self.prior = coverage, phrase, prior

    def rerank(
        self, query: str, candidates: Sequence[Tuple[str, str]], k: int = 10
    ) -> List[Tuple[str, float]]:
        """candidates: (id, text) in retrieval order. Returns (id, score), reranked."""
        q_toks = tokenize(query)
        q_terms = set(q_toks)
        q_bigrams = _bigrams(q_toks)
        if not q_terms:
            return [(cid, 0.0) for cid, _ in candidates[:k]]

        n = len(candidates)
        out: List[Tuple[str, float]] = []
        for rank, (cid, text) in enumerate(candidates):
            toks = tokenize(text)
            terms = set(toks)
            coverage = len(q_terms & terms) / len(q_terms)
            phrase = (len(q_bigrams & _bigrams(toks)) / len(q_bigrams)) if q_bigrams else 0.0
            prior = 1.0 - rank / n  # retrieval already thought this was good
            score = self.coverage * coverage + self.phrase * phrase + self.prior * prior
            out.append((cid, score))
        out.sort(key=lambda x: x[1], reverse=True)
        return out[:k]
