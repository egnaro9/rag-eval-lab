"""Deterministic text embedders.

The default `TfidfEmbedder` is pure-Python (stdlib only): it fits an
IDF table on a corpus and produces sparse TF-IDF vectors represented as
``dict[str, float]``. It is fully deterministic, which is exactly what an
*eval* harness wants — the same input always yields the same vector, so
metric numbers are reproducible and CI never flakes.

`OpenAIEmbedder` is an optional dense embedder used only when
``OPENAI_API_KEY`` is set and the ``openai`` package is installed. Nothing
in the test suite touches it.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase word tokenizer used everywhere in the lab."""
    return _TOKEN_RE.findall(text.lower())


class TfidfEmbedder:
    """A tiny, deterministic TF-IDF vectorizer.

    Vectors are sparse ``dict[str, float]``; cosine similarity is defined in
    :mod:`ragevallab.store`. No numpy, no model download.
    """

    def __init__(self) -> None:
        self.idf: Dict[str, float] = {}
        self._fitted = False

    def fit(self, corpus: Iterable[str]) -> "TfidfEmbedder":
        docs = [tokenize(d) for d in corpus]
        n = len(docs) or 1
        df: Counter = Counter()
        for toks in docs:
            for term in set(toks):
                df[term] += 1
        # smoothed idf, always positive
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}
        self._fitted = True
        return self

    def transform(self, text: str) -> Dict[str, float]:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder.transform called before fit()")
        toks = tokenize(text)
        if not toks:
            return {}
        tf: Counter = Counter(toks)
        length = len(toks)
        vec: Dict[str, float] = {}
        for term, count in tf.items():
            idf = self.idf.get(term)
            if idf is None:
                # unseen term still carries a little signal
                idf = 1.0
            vec[term] = (count / length) * idf
        return vec

    def fit_transform(self, corpus: List[str]) -> List[Dict[str, float]]:
        self.fit(corpus)
        return [self.transform(d) for d in corpus]


class OpenAIEmbedder:  # pragma: no cover - optional, never used in CI
    """Dense embeddings via OpenAI. Only importable path when a key is set."""

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        import os

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for OpenAIEmbedder")
        from openai import OpenAI  # type: ignore

        self._client = OpenAI()
        self.model = model

    def fit(self, corpus):  # noqa: D401 - dense embedders need no fit
        return self

    def transform(self, text: str):
        resp = self._client.embeddings.create(model=self.model, input=text)
        return resp.data[0].embedding
