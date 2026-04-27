# Query API and Runtime Endpoints

This document describes the current FastAPI surface in `apps/api/app/routes/`. All registered routers are mounted under `/api` by `apps/api/app/main.py`, except the compatibility alias `GET /health`.

## App Entry Point

Run the API with:

```powershell
python -m uvicorn apps.api.app.main:app --host 127.0.0.1 --port 8010 --reload
```

The app is created by `create_app()` and includes:

- CORS configured from `API_CORS_ORIGINS`;
- health routes;
- query routes;
- ingest/admin/debug routes;
- legal-unit file routes;
- raw retrieval route.

## Endpoint Summary

| Method | Path | Owner | Purpose |
| --- | --- | --- | --- |
| `GET` | `/api/health` | `routes/health.py` | Basic service liveness. |
| `GET` | `/health` | `routes/health.py` | Compatibility liveness alias. |
| `GET` | `/api/health/config` | `routes/health.py` | Safe config visibility. |
| `GET` | `/api/health/db` | `routes/health.py` | DB reachability, extension, and table status. |
| `POST` | `/api/health/retrieval-demo` | `routes/health.py` | Demo retrieval readiness check. |
| `POST` | `/api/health/query-demo` | `routes/health.py` | Demo answer/citation readiness check. |
| `POST` | `/api/health/query-graph-demo` | `routes/health.py` | Demo graph readiness check. |
| `POST` | `/api/query` | `routes/query.py` | Full Ask Mode query pipeline. |
| `GET` | `/api/query/{query_id}` | `routes/query.py` | Retrieve recent in-memory query response. |
| `GET` | `/api/query/{query_id}/graph` | `routes/query.py` | Retrieve enriched graph response for recent query. |
| `POST` | `/api/retrieve/raw` | `routes/retrieve_raw.py` | DB-backed raw retrieval. |
| `POST` | `/api/ingest/` | `routes/ingest.py` | Background single URL parser pipeline. |
| `GET` | `/api/legal-units/{corpus_id}` | `routes/legal_units.py` | File-based legal units from `ingestion/output`. |
| `POST` | `/api/admin/ingest/batch` | `routes/admin.py` | Protected batch parser pipeline. |
| `GET` | `/api/admin/ingest/batch/{run_id}` | `routes/admin.py` | Protected in-memory batch status. |
| `GET` | `/api/admin/debug` | `routes/admin.py` | Protected debug path info. |
| `GET` | `/api/debug/embeddings-health` | `routes/debug.py` | Protected embedding provider health check. |

Empty or placeholder route modules exist for future `corpus`, `domains`, `explore`, and `search` APIs. They do not currently expose registered product endpoints.

## `POST /api/query`

`/api/query` runs `QueryOrchestrator` and returns the full user-facing answer package.

Request schema: `apps/api/app/schemas/query.py::QueryRequest`

```json
{
  "question": "Poate angajatorul sa-mi scada salariul fara act aditional?",
  "jurisdiction": "RO",
  "date": "current",
  "mode": "strict_citations",
  "debug": true
}
```

Constraints:

- `question`: 10 to 4000 characters.
- `jurisdiction`: currently only `RO`.
- `date`: arbitrary string, defaults to `current`.
- `mode`: currently only `strict_citations`.
- `debug`: when true, includes inspectable internals.

Response schema: `QueryResponse`

```json
{
  "query_id": "...",
  "question": "...",
  "answer": {
    "short_answer": "...",
    "detailed_answer": "...",
    "confidence": 0.0,
    "not_legal_advice": true,
    "refusal_reason": null
  },
  "citations": [],
  "evidence_units": [],
  "verifier": {
    "groundedness_score": 0.0,
    "claims_total": 0,
    "claims_supported": 0,
    "claims_weakly_supported": 0,
    "claims_unsupported": 0,
    "citations_checked": 0,
    "verifier_passed": false,
    "claim_results": [],
    "warnings": [],
    "repair_applied": false,
    "refusal_reason": null
  },
  "graph": {
    "nodes": [],
    "edges": []
  },
  "debug": null,
  "warnings": []
}
```

`answer.confidence` is currently conservative. The verifier and warnings are more important than the answer confidence number.

## Query Debug Shape

When `debug=true`, `debug` includes:

- `orchestrator`
- `evidence_service`
- `retrieval_mode`
- `query_understanding`
- `query_frame`
- `query_decomposer`
- `query_embedding`
- `retrieval`
- `graph_expansion`
- `legal_ranker`
- `evidence_pack`
- `requirement_backfill`
- `generation`
- `verifier`
- `answer_repair`
- evidence/citation/graph counts
- notes

The debug payload is meant for development and frontend inspection. It should not expose large vectors or secrets. Query embeddings are summarized as present/dimension rather than returned in full.

## Query Response Store

`GET /api/query/{query_id}` and `GET /api/query/{query_id}/graph` read from `QueryResponseStore`, an in-memory bounded store created in `routes/query.py`.

Implications:

- results disappear on process restart;
- the store is local to one API process;
- this is not durable persistence;
- missing IDs return `404` with `query_not_found`.

## `POST /api/retrieve/raw`

`/api/retrieve/raw` runs the raw retrieval subsystem directly. It is useful for retrieval tests and debugging before full answer generation.

Request schema: `RawRetrievalRequest`

```json
{
  "question": "Ce spune art. 41 alin. (1) din Codul muncii?",
  "retrieval_filters": {
    "legal_domain": "munca",
    "status": "active"
  },
  "exact_citations": [
    {
      "law_id": "ro.codul_muncii",
      "article_number": "41",
      "paragraph_number": "1"
    }
  ],
  "query_embedding": null,
  "top_k": 50,
  "debug": true
}
```

Response schema: `RawRetrievalResponse`

```json
{
  "candidates": [
    {
      "unit_id": "ro.codul_muncii.art_41.alin_1",
      "rank": 1,
      "retrieval_score": 0.93,
      "score_breakdown": {
        "bm25": 0.8,
        "dense": 0.0,
        "rrf": 1.0,
        "exact_citation_boost": 1.0
      },
      "matched_terms": [],
      "why_retrieved": "exact_citation",
      "unit": {}
    }
  ],
  "retrieval_methods": ["exact_citation", "fts"],
  "warnings": [],
  "debug": {}
}
```

If `DATABASE_URL` is missing or unreachable, route dependency setup yields `EmptyRawRetrievalStore` and returns no candidates with database warnings instead of crashing.

## Health Endpoints

`GET /api/health`

```json
{
  "status": "ok",
  "service": "lexai-api"
}
```

`GET /api/health/config` reports whether important settings are configured, without revealing secrets.

`GET /api/health/db` checks:

- `DATABASE_URL` configured;
- DB reachable;
- `SELECT 1`;
- server version;
- installed extensions;
- existence/counts for `legal_units`, `legal_edges`, and `legal_embeddings`.

Demo health endpoints run real internal pipeline checks for the standard labor salary modification question.

## Ingestion and Admin Endpoints

`POST /api/ingest/` starts a background single-URL file-based ingestion job:

```json
{
  "url": "https://legislatie.just.ro/Public/DetaliiDocument/123456",
  "law_id": "ro.example",
  "law_title": "Example law",
  "out_dir": "ingestion/output/auto_ingest"
}
```

This endpoint runs `scripts/run_parser_pipeline.py`. It writes a bundle to disk and does not import to DB.

Admin endpoints require `X-Admin-Secret` matching `ADMIN_INGEST_SECRET`:

- `POST /api/admin/ingest/batch`
- `GET /api/admin/ingest/batch/{run_id}`
- `GET /api/admin/debug`

The admin batch status is also in-memory and process-local.

## Legal Units File Endpoint

`GET /api/legal-units/{corpus_id}?skip=0&limit=100` reads:

```text
ingestion/output/{corpus_id}/legal_units.json
```

This is a file-bundle endpoint, not a DB legal-unit endpoint. It uses legacy schemas in `apps/api/app/schemas/legal.py`, which differ from the main `QueryResponse` LegalUnit schema.

## Embedding Debug Endpoint

`GET /api/debug/embeddings-health?secret=...` checks an OpenAI-compatible embeddings provider using:

- `EMBEDDING_BASE_URL`
- `EMBEDDING_MODEL`
- `EMBEDDING_DIM`
- optional `ADMIN_INGEST_SECRET`

In production, `ADMIN_INGEST_SECRET` is required.

## Windows UTF-8 Check

PowerShell `Invoke-RestMethod` can display UTF-8 text incorrectly. For decisive checks, write bytes with `curl.exe` and decode manually:

```powershell
curl.exe -s `
  -X POST "http://127.0.0.1:8010/api/query" `
  -H "Content-Type: application/json; charset=utf-8" `
  --data-binary "@body.json" `
  -o response.json

$json = [System.Text.Encoding]::UTF8.GetString(
  [System.IO.File]::ReadAllBytes("$PWD\response.json")
)
```

## Current API Limitations

- No durable query-response store.
- No default DB graph-neighbor endpoint/client.
- Some route modules are empty placeholders.
- `/api/legal-units/{corpus_id}` is file-based and legacy-shaped.
- `/api/ingest/` writes bundles but does not import them into PostgreSQL.
- Admin job tracking is in-memory.
- External model integrations are optional and must be treated as retrieval aids, not legal authorities.
