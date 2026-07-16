"""The RAG pipeline: ingest -> chunk -> embed -> store -> retrieve -> answer.

The default answerer is *extractive* and deterministic (it returns the best
retrieved chunk with a citation), so the whole pipeline runs offline. Set
``RAG_LLM=openai`` with ``OPENAI_API_KEY`` to swap in a generative answerer.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .embedder import TfidfEmbedder, tokenize
from .store import InMemoryVectorStore, Record


@dataclass
class Chunk:
    id: str
    text: str
    doc_id: str


@dataclass
class Answer:
    query: str
    text: str
    citations: List[str] = field(default_factory=list)      # chunk ids
    retrieved: List[str] = field(default_factory=list)       # chunk ids in rank order
    contexts: List[str] = field(default_factory=list)        # chunk texts in rank order


def chunk_document(doc_id: str, text: str, max_sentences: int = 2) -> List[Chunk]:
    """Split a document into small overlapping-ish chunks by sentence groups."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    chunks: List[Chunk] = []
    step = max(1, max_sentences)
    for i in range(0, len(sentences), step):
        group = sentences[i : i + max_sentences]
        if group:
            chunks.append(
                Chunk(id=f"{doc_id}#{len(chunks)}", text=" ".join(group), doc_id=doc_id)
            )
    return chunks


class RagPipeline:
    def __init__(self, embedder: Optional[TfidfEmbedder] = None) -> None:
        self.embedder = embedder or TfidfEmbedder()
        self.store = InMemoryVectorStore()
        self.chunks: Dict[str, Chunk] = {}

    def ingest(self, docs: Dict[str, str], max_sentences: int = 2) -> "RagPipeline":
        all_chunks: List[Chunk] = []
        for doc_id, text in docs.items():
            all_chunks.extend(chunk_document(doc_id, text, max_sentences))
        # Fit IDF on the chunk texts so retrieval scoring is corpus-aware.
        self.embedder.fit([c.text for c in all_chunks])
        for c in all_chunks:
            self.chunks[c.id] = c
            self.store.add(c.id, c.text, self.embedder.transform(c.text), {"doc_id": c.doc_id})
        return self

    def retrieve(self, query: str, k: int = 4) -> List[Record]:
        qvec = self.embedder.transform(query)
        return [rec for rec, _score in self.store.search(qvec, k=k)]

    def answer(self, query: str, k: int = 4) -> Answer:
        records = self.retrieve(query, k=k)
        contexts = [r.text for r in records]
        retrieved_ids = [r.id for r in records]
        backend = os.environ.get("RAG_LLM", "extractive")
        if backend == "openai" and records:  # pragma: no cover - real path
            text = _openai_answer(query, contexts)
            citations = retrieved_ids[:2]
        elif records:
            # Deterministic extractive answer: best chunk, cited.
            text = records[0].text
            citations = [records[0].id]
        else:
            text = "I don't have information to answer that."
            citations = []
        return Answer(
            query=query,
            text=text,
            citations=citations,
            retrieved=retrieved_ids,
            contexts=contexts,
        )


def _openai_answer(query: str, contexts: List[str]) -> str:  # pragma: no cover
    from openai import OpenAI  # type: ignore

    client = OpenAI()
    ctx = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts))
    msg = (
        "Answer the question using ONLY the context. Cite sources like [0].\n\n"
        f"Context:\n{ctx}\n\nQuestion: {query}\nAnswer:"
    )
    resp = client.chat.completions.create(
        model=os.environ.get("RAG_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": msg}],
        temperature=0,
    )
    return resp.choices[0].message.content or ""
