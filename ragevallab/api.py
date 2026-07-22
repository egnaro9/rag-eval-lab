"""A thin FastAPI service over the RAG pipeline.

This is an *optional* layer — the lab's core stays dependency-free, and the API
lives behind the ``api`` extra (``pip install -e ".[api]"``), so importing the
pipeline never drags in a web framework.

    POST /query  {query, k}   -> retrieve + extractive answer, with citations
    POST /eval   {k}          -> run the offline eval set (plus the planted
                                 hallucination) and return the metrics
    GET  /healthz             -> liveness

The pipeline is ingested once at startup and reused, so /query is cheap; /eval
runs the whole demo suite, so it's a few hundred milliseconds.
"""
from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import __version__
from .cli import compute_eval_run
from .data import SAMPLE_DOCS
from .pipeline import RagPipeline


class QueryIn(BaseModel):
    query: str = Field(min_length=1, description="the question to answer from the corpus")
    k: int = Field(default=4, ge=1, le=20, description="how many chunks to retrieve")


class EvalIn(BaseModel):
    k: int = Field(default=4, ge=1, le=20)


def create_app() -> FastAPI:
    app = FastAPI(
        title="rag-eval-lab",
        version=__version__,
        summary="Retrieve, answer, and evaluate — the RAG lab as a service.",
        description="A dependency-free RAG pipeline (from-scratch BM25/TF-IDF, "
                    "extractive answerer) with a deterministic eval harness that "
                    "catches a planted hallucination. This wraps it in HTTP.",
    )
    # Ingest the demo corpus once; the extractive answerer is deterministic, so
    # the same query always returns the same answer.
    pipe = RagPipeline().ingest(SAMPLE_DOCS)

    @app.get("/healthz", tags=["ops"])
    def healthz() -> dict:
        """Liveness — deliberately touches nothing."""
        return {"status": "ok", "version": __version__}

    @app.post("/query", tags=["rag"])
    def query(body: QueryIn) -> dict:
        """Retrieve the top-k chunks and return the extractive answer + citations."""
        return asdict(pipe.answer(body.query, k=body.k))

    @app.post("/eval", tags=["rag"])
    def run_eval(body: EvalIn) -> dict:
        """Run the eval set (+ the planted hallucination) and return the run,
        the same shape eval-history ingests and eval-dashboard renders."""
        return compute_eval_run(body.k).to_dict()

    return app


# Module-level app for ``uvicorn ragevallab.api:app``.
app = create_app()
