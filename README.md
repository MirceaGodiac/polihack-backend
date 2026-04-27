# LexAI Backend

Backend repository for LexAI, a Romanian Legal Operating System. The backend ingests Romanian legislation, turns it into atomic legal units, imports legal truth into PostgreSQL/pgvector, retrieves evidence, ranks it, compiles an EvidencePack, generates grounded answers, verifies citations, repairs unsafe answers, and exposes inspectable API/debug payloads for Ask Mode and graph-oriented Explore Mode.

LexAI is not a generic legal chatbot. The LLM is never the legal source of truth. Citable legal text must come from `LegalUnit.raw_text` / `legal_units.raw_text`; retrieval text, embeddings, chunks, summaries, reranker output, and generated prose are only aids.

## Documentation

The full backend documentation set starts at [docs/README.md](docs/README.md):

- [Backend Architecture](docs/backend-architecture.md): layers, boundaries, package map, current implementation status.
- [RAG Pipeline](docs/rag-pipeline.md): `/api/query` orchestration from query understanding through repair.
- [Query API](docs/query-api.md): endpoints, request/response contracts, debug payloads, and API limitations.
- [Data Ingestion and Storage](docs/data-ingestion-storage.md): parser bundles, DB schema, import, embeddings, fixtures.
- [Development and Operations](docs/development-operations.md): setup, environment, tests, smoke checks, deployment notes.

## Current System Flow

```text
Romanian legal source
-> HTML cleaner / structural parser
-> canonical LegalUnit + LegalEdge bundle
-> optional embeddings input/output
-> PostgreSQL + pgvector import
-> /api/query
-> QueryUnderstanding + QueryFrame
-> optional query embedding
-> RawRetriever exact citation + FTS/lexical + dense fallback
-> GraphExpansionPolicy
-> LegalRanker
-> EvidencePackCompiler
-> RequirementBackfillService
-> GenerationAdapter
-> CitationVerifier
-> AnswerRepair
-> Query graph enrichment
```

The demo query used throughout the repo is:

```text
Poate angajatorul sa-mi scada salariul fara act aditional?
```

For the demo, the expected verified citations are:

- `ro.codul_muncii.art_41.alin_1`
- `ro.codul_muncii.art_41.alin_3`

## Repository Map

```text
apps/api/app/
  main.py                     FastAPI app factory and router wiring
  config.py                   Pydantic settings loaded from environment/.env
  db/                         SQLAlchemy async session, models, SQL migrations
  routes/                     API route modules
  schemas/                    Pydantic API contracts
  services/                   Query, retrieval, ranking, evidence, generation, verification

ingestion/
  pipeline.py                 URL -> canonical bundle pipeline
  html_cleaner.py             Conservative HTML cleanup
  structural_parser.py        Article/paragraph/letter parser
  exporters.py                Canonical bundle writer
  chunks.py                   Retrieval chunks and embeddings input
  embeddings.py               Embedding job helpers/providers
  import_repository.py        PostgreSQL import repository
  output/                     Versioned/demo generated bundles

scripts/
  run_parser_pipeline.py      Run single-source parser pipeline
  run_batch_pipeline.py       Run configured source batch
  plan_db_import.py           Validate/build DB import plan
  import_db_bundle.py         Dry-run/apply canonical bundle import
  generate_embeddings.py      Generate embeddings JSONL
  smoke_demo_query.py         Live API smoke test for demo query
  evaluate_*.py               Retrieval/ranker evaluation helpers

contracts/                    JSON schemas for frontend/backend contracts
docs/                         Design notes for ingestion and query API
tests/                        Unit, API, ingestion, regression, and eval tests
db-init/                      Docker init SQL for pgvector extension
```

`legal_units.json` and `legal_edges.json` in the repo root are legacy small artifacts. The canonical corpus bundles live under `ingestion/output/` and `tests/fixtures/corpus/`.

## Runtime API

The FastAPI application is `apps.api.app.main:app`. It mounts routes under `/api` and keeps a backward-compatible `/health` alias.

Implemented endpoints:

- `GET /api/health` and `GET /health`: service liveness.
- `GET /api/health/config`: safe configuration visibility.
- `GET /api/health/db`: PostgreSQL, extension, and table health.
- `POST /api/health/retrieval-demo`: checks retrieval for the demo query.
- `POST /api/health/query-demo`: checks answer/citation verification for the demo query.
- `POST /api/health/query-graph-demo`: checks graph output for the demo query.
- `POST /api/query`: full Ask Mode orchestration.
- `GET /api/query/{query_id}`: returns the in-memory stored response for a previous query.
- `GET /api/query/{query_id}/graph`: returns the enriched graph response for a previous query.
- `POST /api/retrieve/raw`: DB-backed raw retrieval over `legal_units` and `legal_embeddings`.
- `POST /api/ingest/`: background single-URL file-based ingestion into `ingestion/output/...`; it does not import into DB.
- `GET /api/legal-units/{corpus_id}`: reads file-based legal units from `ingestion/output/{corpus_id}/legal_units.json`.
- `POST /api/admin/ingest/batch`: protected batch ingestion over `ingestion/sources/demo_sources.yaml`.
- `GET /api/admin/ingest/batch/{run_id}`: protected in-memory batch job status.
- `GET /api/debug/embeddings-health`: protected/limited embedding provider health check.

Reserved or currently empty route modules exist for future corpus/domain/explore/search work; they are not fully wired as product APIs yet.

### Query Example

```powershell
curl.exe -s `
  -X POST "http://127.0.0.1:8010/api/query" `
  -H "Content-Type: application/json; charset=utf-8" `
  --data-binary "{`"question`":`"Poate angajatorul sa-mi scada salariul fara act aditional?`",`"jurisdiction`":`"RO`",`"date`":`"current`",`"mode`":`"strict_citations`",`"debug`":true}"
```

`debug=true` includes query understanding, query frame, query embedding status, retrieval debug, graph expansion, ranking rows, evidence selection, requirement backfill, generation, verifier, and answer repair data. Large embeddings are not returned; debug only reports presence and dimension.

## Query Pipeline Details

`QueryOrchestrator` owns the `/api/query` flow:

1. `QueryUnderstanding` normalizes the question, detects legal domain, query types, temporal context, safety flags, exact citations, retrieval filters, and expansion policy.
2. `QueryFrameBuilder` maps the question into deterministic legal issue frames. Optional `LLMQueryDecomposer` can enrich retrieval-only query decomposition, but it is guarded against answers, citations, article IDs, law IDs, and source URLs.
3. `QueryEmbeddingService` optionally calls Ollama `/api/embed` or legacy `/api/embeddings`. Failure emits warnings and falls back to lexical retrieval.
4. `RawRetrieverClient` uses the internal DB retriever by default when `RAW_RETRIEVAL_BASE_URL` is not set, or calls an external `/api/retrieve/raw` if configured.
5. `RawRetriever` combines exact citation lookup, PostgreSQL FTS/lexical search, optional dense pgvector search, and intent governing rule lookup. RRF participates in the weighted raw retrieval score.
6. `GraphExpansionPolicy` expands from retrieval seeds when a neighbors client exists. In the default app wiring there is no DB neighbors client yet, so graph expansion falls back to seed candidates and explicit debug reasons.
7. `LegalRanker` merges raw and expanded candidates, then applies V1 feature weighting or V2 query-frame-gated scoring.
8. `EvidencePackCompiler` selects diverse evidence with MMR and role classification (`direct_basis`, `condition`, `exception`, `definition`, `sanction`, `procedure`, `context`).
9. `RequirementBackfillService` repairs missing required evidence concepts from the real candidate pool before generation.
10. `GenerationAdapter` creates deterministic grounded answer drafts from EvidencePack; it is not a free-form legal authority.
11. `CitationVerifier` checks generated legal claims against EvidencePack and citations.
12. `AnswerRepair` removes, tempers, or refuses unsafe output if support is weak or citations fail.
13. `QueryGraphEnricher` adds query, answer, citation, and support nodes/edges for frontend graph inspection.

## Ingestion and Corpus Bundles

The ingestion side is file-based and deterministic. It fetches or reads legal HTML/text, cleans it, parses structure, exports canonical records, and validates import readiness.

Canonical bundle artifacts:

- `legal_units.json`: citable legal units. `raw_text` is legal truth.
- `legal_edges.json`: structural graph edges, primarily `contains`.
- `reference_candidates.json`: extracted but unresolved references; not authoritative graph edges.
- `legal_chunks.json`: retrieval chunks derived from legal units; not citable.
- `embeddings_input.jsonl`: deterministic inputs for an embedding job.
- `corpus_manifest.json`: provenance, counts, hashes, and warnings.
- `validation_report.json`: quality metrics and import gates.

Run a single URL pipeline:

```powershell
python scripts/run_parser_pipeline.py `
  --url "https://legislatie.just.ro/Public/DetaliiDocument/123456" `
  --law-id "ro.example" `
  --law-title "Example law" `
  --out-dir "ingestion/output/example_bundle" `
  --write-debug
```

Run all configured demo sources:

```powershell
python scripts/run_batch_pipeline.py --sources-file ingestion/sources/demo_sources.yaml --write-debug
```

Validate a bundle before DB import:

```powershell
python scripts/plan_db_import.py --source-dir ingestion/output/codul_muncii --pretty
```

## Database and Import

Local PostgreSQL is provided by Docker Compose:

```powershell
docker compose up -d postgres
```

The local DSN is:

```text
postgresql://lexai:lexai@127.0.0.1:5432/lexai
```

`db-init/001_enable_vector.sql` enables the `vector` extension on a fresh Docker volume. Table migrations live in `apps/api/app/db/migrations/` and should be applied with `psql` or your DB migration runner:

```powershell
psql "$env:DATABASE_URL" -f apps/api/app/db/migrations/0001_h08_d2_legal_import_tables.sql
psql "$env:DATABASE_URL" -f apps/api/app/db/migrations/0002_h08_d3_embeddings_pgvector.sql
```

Dry-run a canonical bundle import:

```powershell
python scripts/import_db_bundle.py `
  --source-dir ingestion/output/codul_muncii `
  --mode dry_run `
  --pretty
```

Apply an import:

```powershell
python scripts/import_db_bundle.py `
  --source-dir ingestion/output/codul_muncii `
  --mode apply `
  --pretty
```

Import with embeddings when `embeddings_output.jsonl` and manifest files exist:

```powershell
python scripts/import_db_bundle.py `
  --source-dir ingestion/output/demo_corpus_v1 `
  --mode apply `
  --with-embeddings `
  --embedding-dim 2560 `
  --pretty
```

The import scripts redact secrets and raw legal text from debug error output where possible. They are designed to validate before applying and to roll back failed DB writes.

## Embeddings

`scripts/generate_embeddings.py` can generate JSONL output from `embeddings_input.jsonl` using either a deterministic fake provider or an OpenAI-compatible embedding endpoint.

Fake provider for tests/dev:

```powershell
python scripts/generate_embeddings.py `
  --input ingestion/output/demo_corpus_v1/embeddings_input.jsonl `
  --output ingestion/output/demo_corpus_v1/embeddings.import.jsonl `
  --provider fake `
  --model fake-2560 `
  --expected-dim 2560
```

OpenAI-compatible provider:

```powershell
python scripts/generate_embeddings.py `
  --input ingestion/output/demo_corpus_v1/embeddings_input.jsonl `
  --output ingestion/output/demo_corpus_v1/embeddings.import.jsonl `
  --provider openai-compatible `
  --base-url "$env:EMBEDDING_BASE_URL" `
  --api-key-env EMBEDDING_API_KEY `
  --model "$env:EMBEDDING_MODEL" `
  --expected-dim 2560
```

At query time, `QueryEmbeddingService` can call Ollama when enabled. If Ollama is down or misconfigured, `/api/query` continues without dense retrieval and emits `query_embedding_unavailable` or `query_embedding_not_configured`.

## Configuration

Settings are loaded from environment variables and `.env` via `apps/api/app/config.py`.

Core variables:

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

Optional query embeddings:

```text
QUERY_EMBEDDING_ENABLED=false
OLLAMA_BASE_URL=http://127.0.0.1:11434
QUERY_EMBEDDING_MODEL=qwen3-embedding:4b
QUERY_EMBEDDING_TIMEOUT_SECONDS=20
```

Embedding debug endpoint variables:

```text
EMBEDDING_BASE_URL=
EMBEDDING_MODEL=
EMBEDDING_DIM=2560
```

Do not commit `.env`, secrets, credentials, or URLs containing passwords.

## Local Development

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
$env:PYTHONUTF8="1"
pip install -r requirements.txt
```

Start the API:

```powershell
python -m uvicorn apps.api.app.main:app --host 127.0.0.1 --port 8010 --reload
```

Check health:

```powershell
curl.exe -s http://127.0.0.1:8010/api/health
curl.exe -s http://127.0.0.1:8010/api/health/db
```

Run the live demo smoke test against a running API:

```powershell
python scripts/smoke_demo_query.py --base-url http://127.0.0.1:8010
```

On Windows, `Invoke-RestMethod` can display UTF-8 incorrectly. For decisive API checks, prefer `curl.exe -o response.json` and decode bytes as UTF-8 in PowerShell.

## Tests

Run the full suite:

```powershell
pytest -q
```

Common targeted suites:

```powershell
pytest tests/api/test_query.py -q
pytest tests/api/test_retrieve_raw.py -q
pytest tests/test_live_demo_regression.py -q
pytest tests/test_requirement_backfill.py -q
pytest tests/test_query_embedding_service.py -q
pytest tests/ingestion -q
```

API and service tests use fake stores/providers where practical. DB-backed behavior requires `DATABASE_URL`, migrations, and imported data.

## Contracts

Frontend-facing contracts are represented in two places:

- Pydantic schemas under `apps/api/app/schemas/`.
- JSON schema files under `contracts/`.

`/api/query` returns:

- `answer`
- `citations`
- `evidence_units`
- `verifier`
- `graph`
- optional `debug`
- `warnings`

Evidence units flatten LegalUnit fields and evidence metadata into one object. Citation labels and quotes must be backed by EvidencePack and verified against `raw_text`.

## Deployment Files

- `docker-compose.yml`: local PostgreSQL/pgvector service.
- `nixpacks.toml`: Railway/Nixpacks Python startup config.
- `railway.toml`: Railway deployment config.

The API can run without DB configuration, but query/retrieval behavior degrades to empty evidence with explicit warnings.

## Current Limitations

- DB graph neighbor expansion is not wired into the default `GraphExpansionPolicy`; it falls back to seed candidates unless a neighbors client is injected.
- Some route modules are placeholders or legacy surfaces.
- Qwen reranking through Ollama is not implemented in the current repo state.
- Query embeddings are optional and disabled by default.
- The generated answer path is deterministic/template-based; external LLM generation is not the legal authority.
- `GET /api/query/{query_id}` and `/graph` use an in-memory response store, not durable persistence.
- `/api/legal-units/{corpus_id}` reads file bundles from `ingestion/output`, while `/api/retrieve/raw` reads PostgreSQL.
- Corpus coverage is demo-oriented and depends on imported bundles.
- SQL migrations are plain SQL files; Alembic is not wired.

## Legal Safety Rules

- Never invent laws, article numbers, unit IDs, `source_url`, citations, or legal conclusions.
- Keep citable legal text in `raw_text`.
- Keep retrieval/chunk/embedding text separate from citation text.
- If EvidencePack is missing or weak, the backend must warn, repair, or refuse.
- Any optional model call must be timeout-bound, fallback-safe, and visible in debug.
- Romanian text encoding must preserve diacritics in stored `raw_text` and generated user-facing answers.
