# Personal Knowledge Base Agent

A RAG-based multi-agent system for querying personal documents through natural language. Upload lecture notes, research papers, or YouTube videos and ask questions across all of them in one place.

## What it does

- Upload PDFs or paste a YouTube URL to add content to the knowledge base
- Ask questions in plain English across everything uploaded
- Request summaries of topics or generate study questions for exam prep
- Maintains conversation context within a session

## How it works

Uploaded documents are chunked and embedded into a ChromaDB vector store. When a query comes in, a LangGraph agent determines the request type — Q&A, summarisation, or study question generation — retrieves the relevant chunks, and passes them to an LLM to generate the response.

```
PDF / YouTube URL
      ↓
  Ingestion pipeline (chunking + embeddings)
      ↓
  ChromaDB vector store
      ↓
User query → LangGraph router → retriever → LLM → response
```

Repeated queries are cached in Redis. Conversation history is persisted to PostgreSQL.

## Tech stack

| Layer | Tool |
|---|---|
| Agent framework | LangGraph |
| LLM | Groq (Llama 3.1 8B) |
| Vector store | ChromaDB |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Cache | Redis |
| Database | PostgreSQL + SQLAlchemy |
| API | FastAPI |
| Deployment | Railway |

## Live API

```
https://knowledgebase-production-95be.up.railway.app/docs
```

Interactive Swagger UI — upload a PDF and start querying it directly from the browser.

## Running locally

**Prerequisites:** Python 3.11, Docker

```bash
git clone https://github.com/Ved130/Knowledge_Base
cd Knowledge_Base
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

Start Redis and PostgreSQL:
```bash
docker run -d --name redis -p 6379:6379 redis:alpine
docker run -d --name postgres -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=password -e POSTGRES_DB=knowledge_base -e POSTGRES_HOST_AUTH_METHOD=trust -p 5433:5432 postgres:alpine
```

Create a `.env` file:
```
GROQ_API_KEY=your_groq_key
API_KEY=your_secret_key
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql://postgres:password@127.0.0.1:5433/knowledge_base
```

Run the API:
```bash
uvicorn api:app --reload
```

Then go to `http://localhost:8000/docs`.

## API endpoints

| Endpoint | Method | What it does |
|---|---|---|
| `/ingest` | POST | Upload a PDF for ingestion |
| `/ask` | POST | Ask a question |
| `/history` | GET | Get recent conversation history |




