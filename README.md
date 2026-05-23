# DocuRAG Chatbot

A retrieval-augmented generation (RAG) chatbot that answers questions grounded strictly in your own documents or websites. Upload PDFs, text files, or paste a URL — then chat with your content.

---

## Features

- **Multi-source ingestion** — PDF, plain text, Markdown, or any public URL
- **Two-stage retrieval** — broad vector search (ChromaDB) followed by cross-encoder reranking for precision
- **LLM-as-judge validation** — every answer is automatically checked for groundedness and usefulness before being returned; retries up to 2 times if it fails
- **Source transparency** — every answer surfaces the source documents and a 0-1 relevance confidence score
- **URL input validation** — URL ingestion validates against private/loopback IPs before fetching

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend API | FastAPI |
| Agent Pipeline | LangGraph |
| Vector Store | ChromaDB (persistent) |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | OpenAI `gpt-4o-mini` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Containerisation | Docker Compose |

---

## Quickstart

### Prerequisites

- Docker & Docker Compose
- An OpenAI API key

### 1. Clone the repo

```bash
git clone https://github.com/nathan-limjw/docurag_chatbot.git
cd docurag_chatbot
```

### 2. Set up environment variables

```bash
cp .env.example .env
```

Open `.env` and add your OpenAI API key:

```env
OPENAI_API_KEY=sk-...
```

### 3. Start the app

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI backend | http://localhost:8000 |
| API docs (Swagger) | http://localhost:8000/docs |

---

## Usage

1. Open the UI at `http://localhost:8501`
2. Use the sidebar to upload a file (PDF / TXT / MD) or paste a public URL
3. Click **Ingest** — you'll see the chunk count update in the sidebar
4. Ask questions in the chat input at the bottom
5. Answers include source badges and rerank scores for traceability

To start fresh, click **Reset & Clear All** in the sidebar.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/status` | Check chunk count and collection status |
| `POST` | `/ingest/file` | Upload a PDF, TXT, or MD file |
| `POST` | `/ingest/url` | Scrape and ingest a public URL |
| `POST` | `/chat` | Submit a query; returns answer + sources |
| `DELETE` | `/reset` | Wipe the vector store |
---

## Project Structure

```
docurag_chatbot/
├── app.py                        # Streamlit frontend
├── src/
│   ├── main.py                   # FastAPI app & endpoints
│   ├── database/
│   │   ├── store.py              # Ingestion, chunking, retrieval
│   │   └── reranker.py           # Cross-encoder reranking
│   ├── graph/
│   │   ├── pipeline.py           # LangGraph graph definition
│   │   ├── state.py              # AgentState TypedDict
│   │   ├── routing.py            # Conditional edge logic
│   │   └── nodes/
│   │       ├── retrieve.py       # Retrieve + rerank node
│   │       ├── generate.py       # LLM generation node
│   │       ├── validate.py       # LLM-as-judge node
│   │       └── increment.py      # Retry counter node
│   ├── schemas/
│   │   └── llm_output.py         # Pydantic request/response schemas
│   └── utils/
│       ├── config.py             # Settings via pydantic-settings
│       └── logger.py             # Structured logging
├── Dockerfile.api
├── Dockerfile.ui
├── docker-compose.yml
├── DESIGN.md                     # Architecture & implementation plan
├── chat_transcript.md            # Demo conversation with the chatbot
└── requirements.txt
```

---

## Configuration

All settings are loaded from `.env` via `pydantic-settings`. Key options:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required |
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM for generation and validation |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `RETRIEVAL_K` | `10` | Candidates fetched from ChromaDB |
| `RERANK_TOP_N` | `4` | Final docs after reranking |
| `MAX_RETRIES` | `2` | Max answer regeneration attempts |

---

## Documentation

See [DESIGN.md](./DESIGN.md) for the full architecture, prompt engineering decisions, chunking strategy, and retrieval design.

See [CHAT_TRANSCRIPT.md](./CHAT_TRANSCRIPT.md) for a demo conversation with the chatbot.