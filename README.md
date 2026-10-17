# Personal Knowledge Base Agent

A multi-agent RAG system for querying personal documents in natural language. Upload PDFs or YouTube videos, then ask questions, request summaries, or generate study questions across everything you've added. Answers are grounded in your documents and returned with their sources.

## What it does

- Ingest PDFs (upload or drop into `data/pdfs/`) and YouTube transcripts
- Ask questions in plain English; get answers with the documents they came from
- Request summaries or generate study questions for revision
- Cache repeated questions in Redis and persist conversation history to a database
- Expose per-request and per-stage latency for monitoring

## How it works

```
PDF / YouTube URL
      ↓
Ingestion (800-char chunks, 120 overlap, source-tagged) → ChromaDB vector store
      ↓
User query → Redis cache ──hit──→ answer (~4 ms)
                 │ miss
                 ↓
        LangGraph router → retrieve (top-5 chunks) → LLM (Q&A / summary / study questions)
                 ↓
        answer + sources → cached in Redis, saved to DB
```

- **Router:** sends each request to the Q&A, summarisation or study-question prompt
- **Grounding:** the Q&A prompt answers only from retrieved context and says "I don't have that information" otherwise
- **Cache invalidation:** ingesting a new document clears cached answers so they never go stale
- **Graceful degradation:** if Redis is unavailable, requests are answered without the cache instead of failing

## Tech stack

| Layer | Tool |
|---|---|
| Agent framework | LangGraph |
| LLM | Groq (`openai/gpt-oss-20b`, configurable) |
| Vector store | ChromaDB |
| Embeddings | all-MiniLM-L6-v2 (ChromaDB default) |
| Cache | Redis |
| Database | PostgreSQL / SQLite via SQLAlchemy |
| API | FastAPI, slowapi rate limiting |
| Deployment | Docker, Railway |

## API

| Endpoint | Method | What it does |
|---|---|---|
| `/ingest` | POST | Upload a PDF for ingestion |
| `/ask` | POST | Ask a question |
| `/history` | GET | Recent conversation history |

All endpoints require an `X-API-Key` header.

Example `/ask` response:

```json
{
  "question": "What is the CGPA mentioned in the resume?",
  "answer": "The CGPA mentioned in the resume is 7.77/10.",
  "sources": ["Resume_test.pdf"],
  "cached": false
}
```

Every response also carries timing headers:

```
X-Cache: MISS
X-Process-Time-ms: 1385.2
Server-Timing: cache_lookup;dur=0.8, retrieval;dur=226.7, llm;dur=1138.7, total;dur=1385.2
```

## Performance

Benchmarked with `benchmark_latency.py` on 40 questions (40/40 succeeded), first uncached and then cached. Server-side times:

| | Median | p95 |
|---|---|---|
| Uncached, total | 1,385 ms | 1,887 ms |
| ↳ LLM generation | 1,139 ms | 1,650 ms |
| ↳ Retrieval | 227 ms | 261 ms |
| Cached, total | 3.6 ms | 4.4 ms |

- LLM generation is ~83% of uncached latency, so caching gives a ~385× speedup on repeat questions.
- Retrieval time is dominated by embedding the query on CPU, not by the vector search.

Reproduce:

```bash
python benchmark_latency.py --url http://127.0.0.1:8000/ask --api-key YOUR_KEY \
    --field question --redis-url redis://localhost:6379/0 --cache-prefix "kb:" --sleep 10
```

## Lessons learned

- **Chunk size matters more than expected.** With 100-character chunks, a one-page resume became 63 fragments; retrieval lost context and the model answered with a hallucinated name. Moving to 800-character chunks with overlap fixed it.
- **Caches need invalidation.** Keying answers by question meant re-ingested documents still returned old answers until the cache was cleared on ingest.
- **Measure before optimising.** Stage-level timing showed the LLM, not retrieval or the database, is the bottleneck.

## Run locally

**Prerequisites:** Python 3.11, Docker

```bash
git clone https://github.com/Ved130/Knowledge_Base
cd Knowledge_Base
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

Start Redis (and optionally PostgreSQL):

```bash
docker run -d --name kb-redis -p 6379:6379 redis:7
```

Create a `.env` file:

```
GROQ_API_KEY=your_groq_key
GROQ_MODEL=openai/gpt-oss-20b
API_KEY=your_secret_key
REDIS_URL=redis://localhost:6379/0
DATABASE_URL=sqlite:///./local.db
RATE_LIMIT=5/minute
```

For PostgreSQL, use `DATABASE_URL=postgresql://user:password@host:port/dbname` instead.

## Deployment

Previously deployed on Railway; the public instance is currently offline to avoid hosting costs. The app runs unchanged in any Docker-capable host with the environment variables above.

## Future work

- Stream responses to cut time-to-first-token
- Semantic caching so paraphrased questions also hit the cache
- Faster query embeddings (model loaded once / GPU) to reduce retrieval time
- Automated retrieval and faithfulness evaluation (hit@k, exact-match on factual questions)
