# Personal Knowledge Base Agent

A multi-agent RAG system that lets you query your personal documents (PDFs and YouTube videos) using natural language.

## Features
- PDF and YouTube video ingestion
- Multi-agent routing (Q&A, summarisation, study questions)
- Semantic search with ChromaDB
- Conversational memory
- Redis caching
- PostgreSQL conversation history
- API key authentication
- Rate limiting

## Tech Stack
- **Agent Framework:** LangGraph
- **LLM:** Groq (Llama 3.1)
- **Vector DB:** ChromaDB
- **Cache:** Redis
- **Database:** PostgreSQL
- **API:** FastAPI
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)

## Architecture
PDF/YouTube → Ingestion Pipeline → ChromaDB
↓
User Query → FastAPI → LangGraph Agent → Retriever → LLM → Response
                                                              ↓
                                                        Router decides:
                                                        Q&A
                                                        Summarise
                                                        Study Quest
                                                        
## Setup
1. Clone the repo
2. Create a virtual environment and install dependencies
3. Add your API keys to `.env`
4. Run Redis and PostgreSQL via Docker
5. Run `uvicorn api:app --reload`

## Environment Variables