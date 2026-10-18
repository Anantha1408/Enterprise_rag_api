# Multi-Agent Enterprise Research & RAG API

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.50+-blue.svg)](https://langchain-ai.github.io/langgraph/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%20+%20pgvector-336791.svg?logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![Celery](https://img.shields.io/badge/Celery-5.4+-37814A.svg?logo=celery&logoColor=white)](https://docs.celeryq.dev)
[![Redis](https://img.shields.io/badge/Redis-7-DC382D.svg?logo=redis&logoColor=white)](https://redis.io)

An enterprise-grade, asynchronous Retrieval-Augmented Generation (RAG) API. Heavy document ingestion is decoupled into an asynchronous **Celery + Redis** queue, while queries are dynamically processed by a cyclic **LangGraph** multi-agent workflow (Router &rarr; Retriever &rarr; Synthesizer) backed by **PostgreSQL (`pgvector`)** with HNSW indexing.

Includes a built-in **Enterprise RAG Studio** web interface served directly from the application.

---

## Architecture Overview

```
                      +-----------------------------+
                      |   Client / Web Studio UI    |
                      |   (http://localhost:8000)   |
                      +--------------+--------------+
                                     |
             +-----------------------+-----------------------+
             |                                               |
             v [POST /api/v1/ingest]                         v [POST /api/v1/ask]
   +---------------------+                         +---------------------+
   |   FastAPI Gateway   |                         |  LangGraph Agent    |
   | (Immediate 202 Ack) |                         | - Router Node       |
   +----------+----------+                         | - Retriever Node    |
              |                                    | - Synthesizer Node  |
              v (Task Event)                       +----------+----------+
   +---------------------+                                    |
   |   Redis 7 Broker    |                                    | (Cosine Distance <=>)
   +----------+----------+                                    v
              |                                    +---------------------+
              v (Consume Task)                     |  PostgreSQL 16      |
   +---------------------+                         |  - documents table  |
   |   Celery Worker     |                         |  - pgvector HNSW    |
   | - TextSplitter      |------------------------>|    (384 dimensions) |
   | - SentenceTransform | (Persist Chunks)        +---------------------+
   +---------------------+
```

---

## Prerequisites

Before running the project, ensure you have:
- **Python 3.10+**
- **Docker Desktop** (running, for PostgreSQL and Redis)
- **Git**

---

## Method 1: Local Development Run (Recommended)

This method runs PostgreSQL (`pgvector`) and Redis inside Docker while running FastAPI and Celery directly on your machine for fast startup and code auto-reload.

### 1. Clone the Repository
```bash
git clone https://github.com/Anantha1408/Enterprise_rag_api.git
cd Enterprise_rag_api
```

### 2. Set Up Virtual Environment & Dependencies
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy the example environment configuration:
```bash
# Windows (PowerShell)
Copy-Item .env.example .env

# Linux / macOS
cp .env.example .env
```

### 4. Start Infrastructure Containers (Postgres + Redis)
```bash
docker compose up -d postgres redis
```
Verify the containers are healthy:
```bash
docker ps
```
You should see `enterprise-rag-postgres` (port 5432) and `enterprise-rag-redis` (port 6379) in `healthy` status.

---

### 5. Start the Background Celery Worker
Open **Terminal 1** with the virtual environment activated:

```bash
# Windows (Requires -P solo because Windows does not support fork)
.\.venv\Scripts\celery -A app.tasks.celery_app worker --loglevel=info -P solo

# Linux / macOS
celery -A app.tasks.celery_app worker --loglevel=info
```
You will see:
```text
celery@your-host ready.
[tasks]
  . process_document
```

---

### 6. Start the FastAPI Web Server
Open **Terminal 2** with the virtual environment activated:

```bash
# Windows
.\.venv\Scripts\uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Linux / macOS
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
You will see:
```text
INFO: Database initialized (pgvector extension + documents table).
INFO: Application startup complete.
INFO: Uvicorn running on http://0.0.0.0:8000
```

---

## Method 2: Full Docker Stack (Containerized)

To build and run all 4 services (`api`, `worker`, `postgres`, `redis`) simultaneously inside Docker:

```bash
# Build and start the entire stack
docker compose up --build

# Stop the stack
docker compose down
```

---

## Using the Application

### 1. Web Studio UI (Browser)
Open your browser and navigate to:
**[http://localhost:8000/](http://localhost:8000/)**

- **Ingestion Desk**: Enter custom documents or click **"Load Sample"**, then click **"Enqueue Ingestion"**.
- **Research Desk**: Enter natural language questions (or click prompt chips), view the LangGraph routing badge (`Retriever` vs `Direct`), read the synthesized answer, and inspect retrieved passages with cosine similarity scores.

### 2. Interactive API Documentation (Swagger)
Open Swagger UI to test endpoints with live schema validation:
**[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## API Reference & Examples

### Health Check
```bash
curl -X GET http://localhost:8000/health
```
```json
{
  "status": "ok",
  "service": "enterprise-rag-api",
  "version": "0.1.0"
}
```

---

### Ingest Document (`POST /api/v1/ingest`)
Enqueues document ingestion to Celery and returns immediately with `202 Accepted`.

**PowerShell**:
```powershell
$body = @{
    text = "Acme Corp Policy 2026: Remote work stipend is 150 USD per month. Customer logs are retained for 18 months."
    metadata = @{ department = "HR"; title = "Q3 Policy" }
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8000/api/v1/ingest -Method Post -Body $body -ContentType "application/json"
```

**cURL**:
```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Acme Corp Policy 2026: Remote work stipend is 150 USD per month. Customer logs are retained for 18 months.",
    "metadata": {"department": "HR", "title": "Q3 Policy"}
  }'
```

**Response** (`202 Accepted`):
```json
{
  "task_id": "ee4974db-8437-41d6-a897-31e4032bd359",
  "status": "accepted",
  "source": "data/uploads/2beaca33-e5e5-4512-b9b9-798d9dad7d5b.txt"
}
```

---

### Ask Query (`POST /api/v1/ask`)
Executes the LangGraph multi-agent research workflow.

**PowerShell**:
```powershell
$body = @{
    query = "According to the ingested document, what is the remote work stipend?"
    top_k = 3
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8000/api/v1/ask -Method Post -Body $body -ContentType "application/json"
```

**cURL**:
```bash
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "According to the ingested document, what is the remote work stipend?",
    "top_k": 3
  }'
```

**Response** (`200 OK`):
```json
{
  "answer": "Based on the retrieved knowledge-base context: The remote work stipend is 150 USD per month for all full-time employees.",
  "query": "According to the ingested document, what is the remote work stipend?",
  "context": [
    "[source=.../2beaca33.txt similarity=0.744]\nAcme Corp Policy 2026: Remote work stipend is 150 USD per month..."
  ],
  "used_retrieval": true,
  "route": "retriever"
}
```

---

## Configuration Reference (`.env`)

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `APP_NAME` | `enterprise-rag-api` | Service identifier |
| `DATABASE_URL` | `postgresql+asyncpg://rag:rag@localhost:5432/enterprise_rag` | Async database URL for FastAPI |
| `DATABASE_SYNC_URL` | `postgresql+psycopg://rag:rag@localhost:5432/enterprise_rag` | Sync database URL for Celery |
| `REDIS_URL` | `redis://localhost:6379/0` | Celery broker and result store URL |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | SentenceTransformer model (384 dims) |
| `EMBEDDING_DIMENSION` | `384` | Dimension of vectors in pgvector column |
| `LLM_BACKEND` | `mock` | `mock` (zero-weights test) or `llama_cpp` (local GGUF) |
| `LLAMA_MODEL_PATH` | `""` | Path to `.gguf` weight file if using `llama_cpp` |
| `CHUNK_SIZE` | `1000` | Target chunk character count |
| `CHUNK_OVERLAP` | `200` | Overlap between adjacent chunks |
| `RETRIEVAL_TOP_K` | `5` | Default number of nearest neighbors retrieved |
| `UPLOAD_DIR` | `data/uploads` | Local directory for raw ingested files |

---

## Project Structure

```
enterprise-rag-api/
├── app/
│   ├── __init__.py
│   ├── agent.py            # LangGraph multi-agent research graph & vector search
│   ├── config.py           # Pydantic Settings loaded from .env
│   ├── database.py         # Dual engines (asyncpg for API, psycopg for Celery)
│   ├── embeddings.py       # SentenceTransformers singleton (all-MiniLM-L6-v2)
│   ├── llm.py              # LLM backend factory (Mock or ChatLlamaCpp)
│   ├── main.py             # FastAPI entrypoint, CORS, routes & UI handler
│   ├── models.py           # SQLAlchemy DocumentChunk model with pgvector HNSW
│   ├── schemas.py          # Pydantic request/response schemas
│   ├── static/
│   │   └── index.html      # Enterprise RAG Studio web dashboard
│   └── tasks.py            # Celery application & asynchronous ingestion task
├── data/
│   └── uploads/            # Ingested text files
├── docker/
│   └── init.sql            # Postgres bootstrap (CREATE EXTENSION vector)
├── .env.example            # Environment template
├── .gitignore              # Ignores .venv, .env, data/uploads, caches
├── ARCHITECTURE_AND_INTERVIEW_GUIDE.md # Comprehensive system design & interview guide
├── docker-compose.yml      # Multi-container orchestration (postgres, redis, api, worker)
├── Dockerfile              # Container definition for API and Worker
├── requirements.txt        # Pinned Python package dependencies
└── README.md               # Quickstart and runbook documentation
```

---

## Troubleshooting

1. **Celery fails on Windows**:
   - Ensure you run Celery with the `-P solo` pool flag:
     `celery -A app.tasks.celery_app worker --loglevel=info -P solo`
2. **Database connection error**:
   - Ensure Docker is running and verify the Postgres container is healthy:
     `docker compose ps`
3. **HTTP 422 Unprocessable Entity**:
   - Ensure your JSON payload provides either non-empty `"text"` or a valid `"file_path"`.
