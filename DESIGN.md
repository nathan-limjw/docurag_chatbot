# DocuRAG Chatbot — Design & Implementation Plan

## Overview

DocuRAG is a Retrieval-Augmented Generation (RAG) chatbot that answers user questions grounded strictly in user-supplied documents or web pages. It exposes a FastAPI backend, a Streamlit frontend, and is fully containerised via Docker Compose.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI (app.py)                │
│  Upload PDF/TXT/MD │ Ingest URL │ Chat Interface        │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTP (httpx)
                           ▼
┌────────────────────────────────────────────────────────-─┐
│                  FastAPI Backend (src/main.py)           │
│                                                          │
│  POST /ingest/file   POST /ingest/url   POST /chat       │
│  GET  /status        DELETE /reset                       │
└──────┬───────────────────────────┬───────────────────────┘
       │                           │
       ▼                           ▼
┌─────────────────┐     ┌──────────────────────────────────-┐
│  Ingestion Layer│     │       LangGraph Pipeline          │
│  (store.py)     │     │                                   │
│                 │     │  START                            │
│  Load → Split   │     │    │                              │
│  → Embed →      │     │  [retrieve]  ← ChromaDB +         │
│  ChromaDB       │     │    │           Cross-Encoder      │
└────────┬────────┘     │  [generate]  ← OpenAI GPT-4o-mini │
         │              │    │                              │
         ▼              │  [validate]  ← LLM-as-Judge       │
┌─────────────────┐     │    │                              │
│  ChromaDB       │     │  grounded     retry               │
│  (Persistent    │     │  & useful? ──────────────────►    │
│   Vector Store) │     │    │          [increment_retry]   │
└─────────────────┘     │    ▼                              │
                        │   END                             │
                        └──────────────────────────────────-┘
```

### Component Summary

| Component | Technology | Role |
|---|---|---|
| Frontend | Streamlit | File/URL ingestion UI, chat interface |
| Backend API | FastAPI | REST endpoints, orchestration |
| Vector Store | ChromaDB (persistent) | Stores and retrieves document embeddings |
| Embeddings | OpenAI `text-embedding-3-small` | Converts text chunks to vectors |
| LLM | OpenAI `gpt-4o-mini` | Answer generation and validation |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Precision re-scoring of retrieved candidates |
| Agent Pipeline | LangGraph | Stateful graph with retry logic |
| Memory | LangGraph `MemorySaver` | Per-thread conversation history |
| Containerisation | Docker Compose | Separate `api` and `ui` services |

---

## Data Ingestion Pipeline

### Supported Sources

- **PDF** — loaded via `PyPDFLoader` (page-level documents)
- **Plain text / Markdown** — loaded via `TextLoader` (UTF-8)
- **Web URL** — fetched with `requests`, cleaned with `BeautifulSoup` (strips `<script>`, `<style>`, `<nav>`, `<header>`, `<footer>` tags), and wrapped into a single `Document`

### URL Security

Before fetching any URL, `validate_public_url` enforces:
- Only `http`/`https` schemes
- DNS resolution must succeed
- Resolved IP addresses must not be private, loopback, link-local, or reserved (SSRF protection)

### Chunking Strategy

Chunking is handled by `RecursiveCharacterTextSplitter` with the following configuration:

| Parameter | Value | Rationale |
|---|---|---|
| `chunk_size` | 800 chars | Large enough for meaningful context, small enough for precise retrieval |
| `chunk_overlap` | 100 chars | Preserves sentence continuity across chunk boundaries |
| `separators` | `["\n\n", "\n", ". ", " ", ""]` | Respects natural language boundaries — paragraphs first, then sentences, then words |

Each chunk is tagged with:
- `source` — original filename or URL
- `chunk_id` — UUID for deduplication and traceability

---

## Embeddings & Vector Storage

### Embedding Model

`text-embedding-3-small` (OpenAI) was chosen for:
- Strong semantic retrieval performance
- Low cost per token
- 1536-dimensional vectors with good separation for document Q&A tasks

### Vector Store

**ChromaDB** is used as the vector store with a persistent local directory (`./chroma_db`), mounted as a Docker volume (`chroma_data`) so data survives container restarts.

Retrieval uses cosine similarity search (`similarity_search`), over-fetching `k=10` candidates to give the reranker a broad pool to work with.

---

## Retrieval Strategy: Two-Stage Retrieval

A two-stage pipeline is used to balance recall and precision:

### Stage 1 — Vector Similarity Search (Broad)
- Embeds the user query and retrieves the top `k=10` chunks from ChromaDB
- Fast approximate nearest-neighbour search
- Optimises for **recall** — casts a wide net

### Stage 2 — Cross-Encoder Reranking (Precise)
- Passes each `(query, chunk)` pair through `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Unlike bi-encoders, cross-encoders see both query and document simultaneously — producing more accurate relevance scores
- Top `n=4` chunks are selected for context
- Rerank score is stored in chunk metadata and surfaced to the user in the UI

This approach is a well-established pattern: bi-encoder for fast candidate retrieval, cross-encoder for precise final selection.

---

## LangGraph Agent Pipeline

The pipeline is a directed state graph compiled with `MemorySaver` (enabling multi-turn conversation per `thread_id`).

### Nodes

| Node | Responsibility |
|---|---|
| `retrieve` | Vector search + cross-encoder reranking |
| `generate` | Builds context string, assembles messages, calls GPT-4o-mini |
| `validate` | LLM-as-judge: checks `is_grounded` and `is_useful` |
| `increment_retry` | Increments `retry_count`, clears stale answer |

### Routing Logic (`route_after_validate`)

```
if is_grounded AND is_useful  →  END
elif retry_count < max_retries  →  retry (back to generate)
else  →  END (return best effort answer)
```

`max_retries` defaults to `2`, capping total generation attempts at 3.

### State Schema (`AgentState`)

```python
messages         # full conversation history (LangGraph add_messages reducer)
query            # current user question
retrieved_docs   # top-k from ChromaDB
reranked_docs    # top-n after cross-encoder
answer           # current generated answer
is_grounded      # whether the answer can be found in the document
is_useful        # whether the answer is helpful in answering the query
retry_count      # number of regeneration attempts
error            # error message if any
```

---

## Prompt Engineering

### Generation Prompt (System)

```
You are a helpful assistant that answers questions based STRICTLY on the provided context documents.

Rules:
1. Answer ONLY using information from the provided context. DO NOT USE outside knowledge.
2. If the context does not contain enough information to answer, say:
   "I don't have enough information in the provided documents to answer that."
3. Be concise but complete. Use bullet points or numbered lists where it helps with clarity.
4. NEVER fabricate facts, statistics or sources.
```

**Rationale:**
- Explicit grounding rule prevents hallucination by restricting the model to provided context only
- The prescribed fallback phrase (`"I don't have enough information..."`) is detectable by the validator's `is_useful` check, creating a clean signal for the retry loop
- Low temperature (`0.1`) reduces creative deviation while keeping answers readable

### Context Injection

Context is built from the top reranked documents, formatted as:

```
[Document 1 | Source: filename.pdf]
<chunk content>

---

[Document 2 | Source: https://example.com]
<chunk content>
```

A `max_context_chars` limit (6000) prevents exceeding model context windows. Source labels are included so the model can reference them accurately.

### Conversation History

The last `max_history_turns=6` messages are prepended before the context + query, enabling coherent multi-turn dialogue without unbounded context growth.

### Validation Prompt (LLM-as-Judge)

A separate zero-temperature LLM call with structured output (`AnswerVerdict`) evaluates:
- `is_grounded`: every claim traceable to the provided context
- `is_useful`: answer substantively addresses the question (not a fallback)

Using structured output (`with_structured_output`) ensures deterministic parsing of the verdict.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/status` | Returns collection name and total chunk count |
| `POST` | `/ingest/file` | Upload PDF, TXT, or MD file for ingestion |
| `POST` | `/ingest/url` | Scrape and ingest a public URL |
| `POST` | `/chat` | Submit a query; returns answer, sources, grounding status |
| `DELETE` | `/reset` | Wipe the entire vector store collection |

---

## Configuration

All settings are managed via `pydantic-settings` and loaded from `.env`:

| Setting | Default | Description |
|---|---|---|
| `openai_api_key` | — | Required |
| `openai_model` | `gpt-4o-mini` | Generation and validation LLM |
| `openai_embedding_model` | `text-embedding-3-small` | Embedding model |
| `chunk_size` | `800` | Characters per chunk |
| `chunk_overlap` | `100` | Overlap between chunks |
| `retrieval_k` | `10` | Candidates fetched from ChromaDB |
| `rerank_top_n` | `4` | Final docs after reranking |
| `reranker_model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker |
| `llm_temperature` | `0.1` | Generation temperature |
| `max_context_chars` | `6000` | Context window cap |
| `max_history_turns` | `6` | Conversation turns to retain |
| `max_retries` | `2` | Max regeneration attempts |

---

## Deployment

```bash
# 1. Copy and populate environment variables
cp .env.example .env

# 2. Build and start both services
docker compose up --build

# API available at: http://localhost:8000
# UI  available at: http://localhost:8501
```

The `chroma_data` Docker volume persists the vector store across container restarts.

---

## Project Structure

```
DocuRAG/
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
└── requirements.txt
```