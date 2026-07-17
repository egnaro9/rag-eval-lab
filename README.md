# rag-eval-lab

[![ci](https://github.com/egnaro9/rag-eval-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/egnaro9/rag-eval-lab/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![live demo](https://img.shields.io/badge/demo-run%20it%20in%20your%20browser-f2a53c)](https://egnaro9.github.io/rag-eval-lab/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**A dependency-free RAG pipeline with a deterministic evaluation harness that catches hallucinations — and proves it in CI.**

The interesting part of a Retrieval-Augmented Generation system isn't the retrieval; it's *knowing whether the answer is grounded in what was retrieved*. This repo is a small, readable reference for that: a full ingest → chunk → embed → store → retrieve → answer pipeline, plus an eval harness whose metrics are closed-form (no LLM-as-judge), so the same input always produces the same score and **CI fails if a planted hallucination stops being flagged.**

- **Zero runtime dependencies.** The core is stdlib-only pure Python — `python -m ragevallab.cli eval` runs anywhere, no install, no API key, no model download.
- **Deterministic evals.** TF-IDF vectors + lexical faithfulness → reproducible numbers, so the eval is a *test*, not a vibe check.
- **Swappable "real" backends.** Drop in OpenAI embeddings/answers (`OPENAI_API_KEY`) or a Postgres + **pgvector** store — same pipeline, one env var.

### ▶ [Run it in your browser](https://egnaro9.github.io/rag-eval-lab/)

No install. The demo page **pip-installs this package's actual wheel into [Pyodide](https://pyodide.org)** and runs it client-side — the same stdlib-only code that runs in CI, executing in your tab. Run the eval suite, then **try to fool the harness**: write any answer you like and watch it score the grounding word by word.

**The two halves compose.** This library *produces* an `eval_run.json`; the companion [eval-dashboard](https://egnaro9.github.io/eval-dashboard/) *renders* one. Run the eval in the browser and hit **"Open this run in the dashboard →"** — the run you just generated is handed straight over (same origin, no server in the middle):

```
rag-eval-lab  ──►  eval_run.json  ──►  eval-dashboard
 (produces)         (the contract)        (renders)
```

---

## How it works

```mermaid
flowchart LR
    D[Documents] --> C[Chunk<br/>by sentence]
    C --> E[Embed<br/>TF-IDF · sparse]
    E --> S[(Vector store<br/>in-memory · pgvector)]
    Q[Query] --> QE[Embed] --> R[Retrieve top-k<br/>cosine similarity]
    S --> R
    R --> A[Answer<br/>extractive · or OpenAI]
    A --> EV{Eval harness}
    R --> EV
    EV --> M[precision@k · recall@k<br/>citation · faithfulness]
    M --> F[🚩 flag if<br/>faithfulness &lt; 0.6]
```

The four metrics, each a pure function of the retrieved chunk ids and the answer text:

| Metric | What it measures | Why it's deterministic |
| --- | --- | --- |
| `precision@k` / `recall@k` | Retrieval quality vs. gold chunk ids | set math over ranked ids |
| `citation_present` | Did the answer cite a source at all? | boolean |
| `faithfulness` | Fraction of the answer's *content* tokens grounded in the retrieved context | lexical overlap, stoplist-filtered |

`faithfulness` is the anti-hallucination check: an answer that names entities absent from its own retrieved context scores low and gets **flagged**.

---

## Verify it yourself

```bash
git clone https://github.com/egnaro9/rag-eval-lab && cd rag-eval-lab
python -m ragevallab.cli eval        # no install needed — stdlib only
```

Output (abridged):

```
run: rag-eval-lab
   faithfulness: 0.917
   flagged_cases: 1.0
        n_cases: 6.0

1 flagged case(s):
  ! Which planet is the hottest in the Solar System?
    answer: Neptune is the hottest planet because of its volcanic geysers.
    faithfulness=0.5  (PLANTED hallucination — answer is not supported by the retrieved context.)
```

The last case is a **planted hallucination**: retrieval correctly returns the Venus chunk, but the answer claims *Neptune* erupts with *volcanic geysers* — words that appear nowhere in the retrieved context. Faithfulness drops to 0.5 and the harness flags it. [`tests/test_evals.py`](tests/test_evals.py) asserts this flagging holds, and [CI](.github/workflows/ci.yml) re-checks it on every push — so a regression that silently stops catching hallucinations turns the build red.

```bash
pip install -e ".[dev]" && pytest -q        # 30 tests
docker compose up eval                       # or run it containerized
```

The full machine-readable report is [`eval_run.example.json`](eval_run.example.json) — this is the schema the companion [eval-dashboard](https://github.com/egnaro9) renders.

---

## Measured against a public benchmark

Six hand-written questions prove the harness bites. They prove nothing about whether the retriever is any *good* — a suite you wrote yourself can drift, without meaning to, into agreeing with you. So it also runs on **SciFact**: 5,183 scientific abstracts, 300 test claims, human relevance judgements, and published scores from people who have never heard of this repo.

```bash
python -m ragevallab.cli benchmark --data ./scifact     # ~20s, stdlib only
```

| retriever | nDCG@10 |
| --- | --- |
| dense models *(published)* | ~0.65 – 0.70 |
| BM25 *(published)* | 0.665 |
| **this repo — TF-IDF cosine, pure stdlib** | **0.581** |

**It loses to BM25, and how it loses is the interesting part.** recall@10 is **0.728** — it *finds* the right abstract nearly three quarters of the time, then fails to rank it first. That gap is the diagnosis: term saturation and length normalisation are precisely what TF-IDF cosine lacks and precisely what BM25 adds, and they fix *ranking*, not *finding*. A number with a named cause beats a number that flatters.

precision@10 is 0.08 because SciFact averages ~1.1 relevant documents per query — about 0.11 is the ceiling. It's reported next to nDCG rather than hidden, because it's the clearest illustration of why nDCG is the metric this task calls for.

Scoring uses the same pipeline the demo uses: chunks collapse to documents by best rank, and only the **300 judged** queries are scored, not all 1,109 in the file. Both are easy to get wrong in the direction that yields a plausible number instead of an error — so both are [tested](tests/test_benchmark.py).

---

## Using the "real" backends

The pipeline is backend-agnostic; the offline defaults exist so the repo is trivially runnable, not because it's toy-only.

**OpenAI answers + dense embeddings**
```bash
pip install -e ".[openai]"
export OPENAI_API_KEY=sk-...
export RAG_LLM=openai        # generative, cited answers instead of extractive
```

**Postgres + pgvector store**
```bash
docker compose up -d db
pip install -e ".[pgvector]"
export DATABASE_URL=postgresql://rag:rag@localhost:5432/rag
```
`ragevallab.store.PgVectorStore` creates the `vector` extension, stores dense embeddings in a `vector(dim)` column, and retrieves with the `<=>` cosine-distance operator — the same `add`/`search` interface as the in-memory store.

---

## Design notes

- **Why pure Python / no numpy?** Determinism and portability. The eval harness's value is that its numbers don't move; stdlib TF-IDF over sparse `dict[str, float]` vectors is exact and installs nowhere. It also means CI needs no wheels and no secrets.
- **Why closed-form metrics instead of an LLM judge?** An LLM judge is itself nondeterministic and unfalsifiable in CI. Lexical faithfulness is a weaker signal but a *checkable* one — you can assert on it. (The two compose: use this as the fast gate, an LLM judge as a slower offline layer.)
- **Honest limitation, tested as such.** A TF-IDF baseline has no stemmer, so *"largest planet"* is lexically ambiguous against Saturn's *"second largest planet."* [`test_retrieval_recalls_gold_in_topk`](tests/test_pipeline.py) encodes exactly that: the gold chunk is recalled within top-k even when it isn't rank-1. Swap in `OpenAIEmbedder` to close the gap.

## Layout

```
ragevallab/
  embedder.py   TfidfEmbedder (stdlib) · OpenAIEmbedder (optional)
  store.py      InMemoryVectorStore (cosine) · PgVectorStore (pgvector)
  pipeline.py   chunk → embed → retrieve → extractive/OpenAI answer
  evals.py      precision@k · recall@k · citation · faithfulness · evaluate()
  data.py       demo corpus + eval set + the planted hallucination
  cli.py        `python -m ragevallab.cli eval`
tests/          30 tests — pipeline behavior + the hallucination-flag guarantee
```

---

Built by [Erik Hill](https://egnaro9.github.io) · MIT licensed.
