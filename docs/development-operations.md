# Development and Operations

This document covers local development, configuration, test commands, smoke checks, and deployment files for the current backend.

## Requirements

Python dependencies are listed in `requirements.txt`.

Core stack:

- Python
- FastAPI
- Pydantic / pydantic-settings
- SQLAlchemy async
- asyncpg
- PostgreSQL
- pgvector
- BeautifulSoup
- httpx / requests
- pytest

## Local Setup

Create a virtual environment and install dependencies:

```powershell
cd D:\dev\Polihack\polihack-backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
$env:PYTHONUTF8="1"
pip install -r requirements.txt
```

Start local PostgreSQL:

```powershell
docker compose up -d postgres
```

Set local DB URL:

```powershell
$env:DATABASE_URL="postgresql://lexai:lexai@127.0.0.1:5432/lexai"
```

Run migrations:

```powershell
psql "$env:DATABASE_URL" -f apps/api/app/db/migrations/0001_h08_d2_legal_import_tables.sql
psql "$env:DATABASE_URL" -f apps/api/app/db/migrations/0002_h08_d3_embeddings_pgvector.sql
```

Start the API:

```powershell
python -m uvicorn apps.api.app.main:app --host 127.0.0.1 --port 8010 --reload
```

## Environment Variables

Settings are defined in `apps/api/app/config.py`.

Core:

```text
APP_ENV=development
DATABASE_URL=postgresql://lexai:lexai@127.0.0.1:5432/lexai
API_CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
RAW_RETRIEVAL_BASE_URL=
ADMIN_INGEST_SECRET=
```

Optional LLM query decomposition:

```text
ENABLE_LLM_QUERY_DECOMPOSER=false
LLM_QUERY_DECOMPOSER_BASE_URL=
LLM_QUERY_DECOMPOSER_API_KEY=
LLM_QUERY_DECOMPOSER_MODEL=
LLM_QUERY_DECOMPOSER_TIMEOUT_SECONDS=5
```

Optional query embedding:

```text
QUERY_EMBEDDING_ENABLED=false
OLLAMA_BASE_URL=http://127.0.0.1:11434
QUERY_EMBEDDING_MODEL=qwen3-embedding:4b
QUERY_EMBEDDING_TIMEOUT_SECONDS=20
```

Embedding debug endpoint:

```text
EMBEDDING_BASE_URL=
EMBEDDING_MODEL=
EMBEDDING_DIM=2560
```

Never commit `.env`, credentials, API keys, or URLs containing passwords.

## Health Checks

```powershell
curl.exe -s http://127.0.0.1:8010/api/health
curl.exe -s http://127.0.0.1:8010/api/health/config
curl.exe -s http://127.0.0.1:8010/api/health/db
```

Demo checks:

```powershell
curl.exe -s -X POST http://127.0.0.1:8010/api/health/retrieval-demo
curl.exe -s -X POST http://127.0.0.1:8010/api/health/query-demo
curl.exe -s -X POST http://127.0.0.1:8010/api/health/query-graph-demo
```

Live smoke script:

```powershell
python scripts/smoke_demo_query.py --base-url http://127.0.0.1:8010
```

## Test Commands

Full suite:

```powershell
pytest -q
```

API tests:

```powershell
pytest tests/api/test_health.py -q
pytest tests/api/test_query.py -q
pytest tests/api/test_query_persistence.py -q
pytest tests/api/test_retrieve_raw.py -q
pytest tests/api/test_embeddings_debug.py -q
```

Query/RAG services:

```powershell
pytest tests/test_query_understanding.py -q
pytest tests/test_exact_citation_detector.py -q
pytest tests/test_query_frame.py -q
pytest tests/test_raw_retriever_client.py -q
pytest tests/test_retrieval_scoring.py -q
pytest tests/test_graph_expansion_policy.py -q
pytest tests/test_legal_ranker.py -q
pytest tests/test_evidence_pack_compiler.py -q
pytest tests/test_requirement_backfill.py -q
pytest tests/test_generation_adapter.py -q
pytest tests/test_citation_verifier.py -q
pytest tests/test_answer_repair.py -q
pytest tests/test_live_demo_regression.py -q
```

Ingestion and import:

```powershell
pytest tests/ingestion -q
```

Embedding service:

```powershell
pytest tests/test_query_embedding_service.py -q
pytest tests/ingestion/test_embeddings.py -q
pytest tests/ingestion/test_embeddings_job.py -q
```

## Common Workflows

### Generate a Canonical Bundle

```powershell
python scripts/run_parser_pipeline.py `
  --url "https://legislatie.just.ro/Public/DetaliiDocument/123456" `
  --law-id "ro.example" `
  --law-title "Example law" `
  --out-dir "ingestion/output/example_bundle" `
  --write-debug
```

### Import a Bundle

```powershell
python scripts/import_db_bundle.py `
  --source-dir ingestion/output/codul_muncii `
  --mode apply `
  --pretty
```

### Query the API

```powershell
@'
{
  "question": "Poate angajatorul sa-mi scada salariul fara act aditional?",
  "jurisdiction": "RO",
  "date": "current",
  "mode": "strict_citations",
  "debug": true
}
'@ | Set-Content -Encoding UTF8 body.json

curl.exe -s `
  -X POST "http://127.0.0.1:8010/api/query" `
  -H "Content-Type: application/json; charset=utf-8" `
  --data-binary "@body.json" `
  -o response.json
```

### Decode Windows UTF-8 Output

```powershell
$json = [System.Text.Encoding]::UTF8.GetString(
  [System.IO.File]::ReadAllBytes("$PWD\response.json")
)
```

## Deployment Files

`docker-compose.yml`
: Local `pgvector/pgvector:pg16` PostgreSQL service.

`db-init/001_enable_vector.sql`
: Enables pgvector extension for new Docker volumes.

`nixpacks.toml`
: Nixpacks/Railway build hint.

`railway.toml`
: Railway start command/config.

## Operational Constraints

- Do not run destructive DB/file commands unless explicitly requested.
- Keep optional external providers fallback-safe.
- Make route handlers thin and keep behavior in services.
- Preserve public API contracts unless a task explicitly changes them.
- Every legal claim must be supported by EvidencePack and citation verification.
- `raw_text` is legal truth; do not cite `retrieval_text` or embeddings.
- Keep debug payloads inspectable but avoid secrets and large vectors.

## Current Known Gaps

- No durable query response persistence.
- No default DB graph-neighbor expansion client.
- No Qwen reranker integration in current code.
- Some platform service files are empty placeholders.
- Some API surfaces are legacy/file-based rather than DB-backed.
- Migrations are SQL files, not Alembic revisions.
- Corpus coverage is demo-oriented.

## Suggested Development Order

For backend work after documentation:

1. Clean EvidencePack role/noise issues for demo query.
2. Add optional Qwen reranker if the Ollama/Railway provider is stable.
3. Wire real DB graph neighbors into `GraphExpansionPolicy`.
4. Broaden legal issue frames beyond the current demo-heavy labor paths.
5. Add durable query persistence if frontend history/debug replay requires it.
6. Build a larger eval set for 50 to 100 Romanian legal questions.
