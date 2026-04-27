# Data Ingestion and Storage

This document explains how legal text enters the repo, how it becomes canonical data, and how that data is imported into PostgreSQL/pgvector.

## Source-to-Bundle Pipeline

The ingestion pipeline is deterministic and file-based. It lives under `ingestion/` and can be run through `scripts/run_parser_pipeline.py` or `scripts/run_batch_pipeline.py`.

```mermaid
flowchart TD
    A["Source URL / local fixture"] --> B["HTML scraper"]
    B --> C["HTML cleaner"]
    C --> D["StructuralParser"]
    D --> E["Canonical exporter"]
    E --> F["legal_units.json"]
    E --> G["legal_edges.json"]
    E --> H["reference_candidates.json"]
    E --> I["legal_chunks.json"]
    E --> J["embeddings_input.jsonl"]
    E --> K["corpus_manifest.json"]
    E --> L["validation_report.json"]
```

The parser does not interpret law. It preserves legal text, extracts structure, and produces validation reports.

## Important Ingestion Modules

| Module | Purpose |
| --- | --- |
| `ingestion/html_scraper.py` | Fetches source HTML. |
| `ingestion/html_cleaner.py` | Removes navigation/noise while preserving legal markers and text. |
| `ingestion/structural_parser.py` | Parses articles, paragraphs, letters, and points. |
| `ingestion/legal_ids.py` | Builds deterministic legal unit IDs. |
| `ingestion/normalizer.py` | Normalizes derived text and repairs deterministic Romanian mojibake. |
| `ingestion/reference_extractor.py` | Extracts reference candidates from raw legal text. |
| `ingestion/chunks.py` | Builds retrieval chunks and embedding input records. |
| `ingestion/exporters.py` | Writes canonical bundle artifacts and validation report. |
| `ingestion/bundle_loader.py` | Loads canonical bundles for local/dev use. |
| `ingestion/local_retriever.py` | Deterministic local bundle retriever for tests/dev. |
| `ingestion/imports.py` | Builds and validates import plans. |
| `ingestion/import_repository.py` | Applies import plans to PostgreSQL. |

## Canonical Bundle Artifacts

Every import-ready bundle should contain:

| Artifact | Meaning | Citable? |
| --- | --- | --- |
| `legal_units.json` | Atomic LegalUnit records with IDs, hierarchy, metadata, and `raw_text`. | Yes, via `raw_text`. |
| `legal_edges.json` | Graph edges, currently mostly structural `contains` edges. | No. |
| `reference_candidates.json` | Extracted unresolved/candidate references. | No. |
| `legal_chunks.json` | Retrieval chunks derived from LegalUnits. | No. |
| `embeddings_input.jsonl` | Deterministic embedding job input. | No. |
| `corpus_manifest.json` | Provenance, source descriptors, hashes, and counts. | No. |
| `validation_report.json` | Quality metrics, warnings, and import gates. | No. |

Only `legal_units.json[*].raw_text` is citable legal text.

## LegalUnit Shape

Canonical LegalUnit records are designed to be compatible with query evidence, DB import, and frontend display.

Important fields:

- `id`: stable legal unit ID such as `ro.codul_muncii.art_41.alin_1`.
- `canonical_id`: compact internal ID.
- `law_id`, `law_title`: owning act.
- `status`: `active`, `historical`, `repealed`, or `unknown`.
- `hierarchy_path`: display/navigation path.
- `article_number`, `paragraph_number`, `letter_number`, `point_number`: legal location.
- `raw_text`: citable legal text.
- `normalized_text`: retrieval helper, not citable.
- `legal_domain`: conservative domain such as `munca`.
- `legal_concepts`: conservative concept list.
- `source_url`: source provenance where known.
- `parent_id`: structural parent.
- `parser_warnings`: explicit unknowns or parser concerns.

## ID Strategy

IDs are deterministic. Examples:

```text
ro.codul_muncii
ro.codul_muncii.art_41
ro.codul_muncii.art_41.alin_1
ro.codul_muncii.art_41.alin_3.lit_e
```

This lets ingestion, DB import, retrieval, evidence, citation verification, and graph output use the same identity.

## Chunks and Embeddings Input

`legal_chunks.json` and `embeddings_input.jsonl` are retrieval artifacts.

Rules:

- chunk `text` comes from LegalUnit `raw_text`;
- `retrieval_context` adds deterministic context such as law title, hierarchy, domain, parent, and unresolved references;
- `retrieval_text` is context plus raw text;
- `retrieval_text` is never citation text;
- embedding records preserve hashes for import validation.

## Running Ingestion

Single URL:

```powershell
python scripts/run_parser_pipeline.py `
  --url "https://legislatie.just.ro/Public/DetaliiDocument/123456" `
  --law-id "ro.example" `
  --law-title "Example law" `
  --out-dir "ingestion/output/example_bundle" `
  --write-debug
```

Batch configured sources:

```powershell
python scripts/run_batch_pipeline.py `
  --sources-file ingestion/sources/demo_sources.yaml `
  --write-debug
```

Historical daily helper:

```powershell
python scripts/run_daily_ingestion_pipeline.py
```

## DB Schema

Plain SQL migrations live in `apps/api/app/db/migrations/`.

Tables:

### `legal_units`

Stores canonical LegalUnits for citable legal text and retrieval metadata.

Key columns:

- `id text primary key`
- `law_id`
- `law_title`
- `status`
- `hierarchy_path jsonb`
- `article_number`
- `paragraph_number`
- `letter_number`
- `point_number`
- `raw_text`
- `normalized_text`
- `legal_domain`
- `legal_concepts jsonb`
- `source_url`
- `parent_id`
- `parser_warnings jsonb`

### `legal_edges`

Stores graph edges.

Key columns:

- `id text primary key`
- `source_id`
- `target_id`
- `type`
- `weight`
- `confidence`
- `metadata jsonb`

### `import_runs`

Stores import audit summaries.

### `legal_embeddings`

Stores pgvector embeddings.

Key columns:

- `record_id`
- `legal_unit_id`
- `chunk_id`
- `model_name`
- `embedding_dim`
- `text_hash`
- `embedding vector(2560)`
- `metadata jsonb`
- `source_path`

## Local DB Setup

Start local Postgres/pgvector:

```powershell
docker compose up -d postgres
```

Local DSN:

```text
postgresql://lexai:lexai@127.0.0.1:5432/lexai
```

Apply migrations:

```powershell
psql "$env:DATABASE_URL" -f apps/api/app/db/migrations/0001_h08_d2_legal_import_tables.sql
psql "$env:DATABASE_URL" -f apps/api/app/db/migrations/0002_h08_d3_embeddings_pgvector.sql
```

`db-init/001_enable_vector.sql` enables `vector` when the Docker volume is first created.

## Import Workflow

Validate or dry-run:

```powershell
python scripts/plan_db_import.py `
  --source-dir ingestion/output/codul_muncii `
  --pretty

python scripts/import_db_bundle.py `
  --source-dir ingestion/output/codul_muncii `
  --mode dry_run `
  --pretty
```

Apply:

```powershell
python scripts/import_db_bundle.py `
  --source-dir ingestion/output/codul_muncii `
  --mode apply `
  --pretty
```

Apply with embeddings:

```powershell
python scripts/import_db_bundle.py `
  --source-dir ingestion/output/demo_corpus_v1 `
  --mode apply `
  --with-embeddings `
  --embedding-dim 2560 `
  --pretty
```

The import path:

1. discovers bundle artifacts;
2. validates bundle and optional embeddings readiness;
3. builds an idempotent import plan;
4. ensures DB schema;
5. upserts LegalUnits, LegalEdges, and optional embeddings;
6. records import run status and counts;
7. rolls back failed transactions.

## Embedding Job Workflow

Fake deterministic embeddings:

```powershell
python scripts/generate_embeddings.py `
  --input ingestion/output/demo_corpus_v1/embeddings_input.jsonl `
  --output ingestion/output/demo_corpus_v1/embeddings.import.jsonl `
  --provider fake `
  --model fake-2560 `
  --expected-dim 2560
```

OpenAI-compatible embeddings:

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

Validation helpers:

```powershell
python scripts/validate_embeddings_output.py --help
python scripts/validate_embeddings_pair.py --help
python scripts/write_embeddings_manifest.py --help
```

## Fixture Strategy

Tracked fixtures under `tests/fixtures/corpus/` support deterministic integration tests without requiring a live DB or network.

Important fixture families:

- Codul muncii canonical fixture files.
- Mini Codul muncii expected bundle.
- Legacy units for parser/export tests.
- Retrieval and ranker eval cases.

Generated demo bundles under `ingestion/output/` are also tracked for local development and handoff workflows.

## Current Storage Limitations

- SQL migrations are plain SQL, not Alembic.
- Runtime `legal_units` list endpoint is file-based and legacy-shaped.
- DB graph-neighbor retrieval is not wired into the default query orchestrator.
- Embeddings are optional and depend on prepared `embeddings_output.jsonl`.
- The repo contains demo-oriented corpora, not a complete production legal corpus.
