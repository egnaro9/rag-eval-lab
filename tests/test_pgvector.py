"""Integration test for the pgvector backend.

Skipped unless a Postgres with the pgvector extension is reachable via
DATABASE_URL. The `pgvector` CI job provides one (the `pgvector/pgvector` image);
a normal local run has neither the driver nor the database, so it skips cleanly
rather than pretending to check the real path.
"""
import os

import pytest

pytest.importorskip("psycopg")                       # no driver -> skip the module
DSN = os.environ.get("DATABASE_URL")
pytestmark = pytest.mark.skipif(not DSN, reason="no DATABASE_URL / pgvector DB reachable")

from ragevallab.store import SCHEMA_VERSION, PgVectorStore  # noqa: E402


def test_pgvector_roundtrip_and_ranking():
    store = PgVectorStore(DSN, dim=3, table="test_chunks")
    store.add("a", "alpha", [1.0, 0.0, 0.0], {"doc": "1"})
    store.add("b", "beta", [0.0, 1.0, 0.0], {"doc": "2"})

    # A query nearest the 'a' vector must come back ranked first — this is the
    # real pgvector `<=>` distance ordering, not the in-memory store's.
    results = store.search([0.9, 0.1, 0.0], k=2)
    assert [r.id for r, _score in results][0] == "a"
    assert results[0][1] > results[1][1]             # 'a' scores higher than 'b'


def test_pgvector_records_its_schema_version():
    store = PgVectorStore(DSN, dim=3, table="test_chunks")
    assert store.schema_version() == SCHEMA_VERSION
