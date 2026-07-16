"""rag-eval-lab: a small, dependency-light Retrieval-Augmented Generation
pipeline with a deterministic evaluation harness.

Everything in the default path runs offline with no API keys and no external
services, so `pytest` is green on a clean clone. A real-embeddings /
real-LLM / pgvector path exists behind optional env vars and dependencies.
"""

__version__ = "0.1.0"

from .embedder import TfidfEmbedder
from .store import InMemoryVectorStore
from .pipeline import RagPipeline, Chunk
from .evals import evaluate, faithfulness, precision_at_k

__all__ = [
    "TfidfEmbedder",
    "InMemoryVectorStore",
    "RagPipeline",
    "Chunk",
    "evaluate",
    "faithfulness",
    "precision_at_k",
]
