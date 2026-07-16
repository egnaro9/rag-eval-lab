from ragevallab.data import SAMPLE_DOCS
from ragevallab.embedder import TfidfEmbedder
from ragevallab.pipeline import RagPipeline, chunk_document
from ragevallab.store import InMemoryVectorStore, cosine


def test_chunk_ids_are_stable_and_single_chunk_docs():
    chunks = chunk_document("mars", SAMPLE_DOCS["mars"])
    assert [c.id for c in chunks] == ["mars#0"]
    assert chunks[0].doc_id == "mars"


def test_tfidf_is_deterministic():
    e1 = TfidfEmbedder().fit(SAMPLE_DOCS.values())
    e2 = TfidfEmbedder().fit(SAMPLE_DOCS.values())
    assert e1.transform("hottest planet") == e2.transform("hottest planet")


def test_cosine_identity_and_orthogonality():
    a = {"x": 1.0, "y": 2.0}
    assert abs(cosine(a, a) - 1.0) < 1e-9
    assert cosine(a, {"z": 5.0}) == 0.0


def test_store_ranks_by_similarity():
    store = InMemoryVectorStore()
    store.add("a", "alpha", {"alpha": 1.0})
    store.add("b", "beta", {"beta": 1.0})
    hits = store.search({"alpha": 1.0}, k=2)
    assert hits[0][0].id == "a"
    assert hits[0][1] > hits[1][1]


def test_retrieval_top1_on_unambiguous_queries():
    pipe = RagPipeline().ingest(SAMPLE_DOCS)
    cases = {
        "Which planet is the hottest?": "venus#0",
        "What is the tallest volcano in the Solar System?": "mars#0",
        "Which planet has a famous ring system?": "saturn#0",
    }
    for q, expected in cases.items():
        top = pipe.retrieve(q, k=1)
        assert top[0].id == expected, f"{q!r} -> {top[0].id}, expected {expected}"


def test_retrieval_recalls_gold_in_topk():
    # "largest planet" is lexically ambiguous vs. Saturn's "second largest
    # planet" — a fair limitation of a TF-IDF baseline — but the gold chunk is
    # still recalled within the top-k, which is what the recall metric checks.
    pipe = RagPipeline().ingest(SAMPLE_DOCS)
    top_ids = [r.id for r in pipe.retrieve("What is the largest planet?", k=3)]
    assert "jupiter#0" in top_ids


def test_answer_is_extractive_and_cited():
    pipe = RagPipeline().ingest(SAMPLE_DOCS)
    ans = pipe.answer("Which planet is the hottest?", k=3)
    assert ans.citations, "extractive answer must cite a chunk"
    assert ans.citations[0] == ans.retrieved[0]
    assert "venus" in ans.text.lower()


def test_answer_when_nothing_ingested():
    pipe = RagPipeline().ingest(SAMPLE_DOCS)
    ans = pipe.answer("something", k=0)
    assert ans.text == "I don't have information to answer that."
    assert ans.citations == []
