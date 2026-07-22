"""Vector stores.

`InMemoryVectorStore` is the default, pure-Python backend used by the tests:
sparse cosine similarity over ``dict[str, float]`` vectors.

`PgVectorStore` is the "real" path — it persists dense vectors in Postgres
using the pgvector extension. It is imported lazily and is never exercised
by CI (see docker-compose.yml to run it locally). Its presence is the point:
it shows the same pipeline swapping an in-memory index for a real vector DB.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple, Union

Vector = Union[Dict[str, float], Sequence[float]]


def _cosine_sparse(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    # iterate the smaller dict
    if len(a) > len(b):
        a, b = b, a
    dot = sum(w * b.get(t, 0.0) for t, w in a.items())
    na = math.sqrt(sum(w * w for w in a.values()))
    nb = math.sqrt(sum(w * w for w in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _cosine_dense(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def cosine(a: Vector, b: Vector) -> float:
    if isinstance(a, dict):
        return _cosine_sparse(a, b)  # type: ignore[arg-type]
    return _cosine_dense(a, b)  # type: ignore[arg-type]


@dataclass
class Record:
    id: str
    text: str
    vector: Vector
    meta: dict = field(default_factory=dict)


class InMemoryVectorStore:
    """Brute-force cosine search. Fine for demos and eval datasets."""

    def __init__(self) -> None:
        self._records: List[Record] = []

    def add(self, id: str, text: str, vector: Vector, meta: dict | None = None) -> None:
        self._records.append(Record(id=id, text=text, vector=vector, meta=meta or {}))

    def __len__(self) -> int:
        return len(self._records)

    def search(self, query_vec: Vector, k: int = 4) -> List[Tuple[Record, float]]:
        scored = [(r, cosine(query_vec, r.vector)) for r in self._records]
        scored.sort(key=lambda rs: rs[1], reverse=True)
        return scored[: max(0, k)]


# Bump when the chunks-table shape changes. Recorded in schema_meta so an old
# database is detectable rather than silently mismatching a new code path — the
# zero-framework version of a migration version, exercised by the pgvector CI job.
SCHEMA_VERSION = 1


class PgVectorStore:
    """Postgres + pgvector backend. Requires ``psycopg`` and a live DB.

    Run ``docker compose up -d db`` (see docker-compose.yml) then set
    ``DATABASE_URL=postgresql://rag:rag@localhost:5432/rag``. Exercised by the
    ``pgvector`` CI job against a real pgvector-enabled Postgres.
    """

    def __init__(self, dsn: str, dim: int, table: str = "chunks") -> None:
        import psycopg  # type: ignore

        self._conn = psycopg.connect(dsn, autocommit=True)
        self.table = table
        self.dim = dim
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {table} ("
                f"  id TEXT PRIMARY KEY, text TEXT, meta JSONB,"
                f"  embedding vector({dim}))"
            )
            cur.execute(
                "CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value INT)"
            )
            cur.execute(
                "INSERT INTO schema_meta (key, value) VALUES ('version', %s) "
                "ON CONFLICT (key) DO NOTHING",
                (SCHEMA_VERSION,),
            )

    def schema_version(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute("SELECT value FROM schema_meta WHERE key = 'version'")
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def add(self, id: str, text: str, vector: Sequence[float], meta: dict | None = None) -> None:
        import json

        with self._conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {self.table} (id, text, meta, embedding) "
                f"VALUES (%s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
                (id, text, json.dumps(meta or {}), list(vector)),
            )

    def search(self, query_vec: Sequence[float], k: int = 4):
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT id, text, meta, 1 - (embedding <=> %s) AS score "
                f"FROM {self.table} ORDER BY embedding <=> %s LIMIT %s",
                (list(query_vec), list(query_vec), k),
            )
            rows = cur.fetchall()
        return [
            (Record(id=r[0], text=r[1], vector=[], meta=r[2] or {}), float(r[3]))
            for r in rows
        ]
