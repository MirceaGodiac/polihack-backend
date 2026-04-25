# Dev Setup & Per-Module Instructions

## One-time setup (everyone)

```bash
git clone <repo>
cd polihack-backend

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Open .env and paste the DATABASE_PUBLIC_URL from Railway:
# Railway → ParadeDB service → Variables → DATABASE_PUBLIC_URL
# DATABASE_URL=postgresql://postgres:...@...proxy.rlwy.net:5432/railway
```

Verify the DB is reachable:

```bash
uvicorn apps.api.app.main:app --reload
# then in another terminal:
curl localhost:8000/health
# should return {"status":"ok","postgres_version":"18.x ...","extensions":[...,"pg_search","vector",...]}
```

---

## Running all tests

```bash
pytest
```

---

## Parser teammate

**Your modules:** `ingestion/structural_parser.py`, `ingestion/legal_ids.py`,
`ingestion/validators.py`, `ingestion/manifest.py`, `ingestion/parser_rules.py`

**Your tests:**

```bash
pytest tests/test_structural_parser.py tests/test_validation.py -v
```

**Smoke-run the parser against real fixture files:**

```bash
python verify_parser.py
# Parses ingestion/fixtures/codul_muncii.txt and ingestion/fixtures/constitutia.txt
# Outputs legal_units.json and legal_edges.json in the repo root
```

**What to build / acceptance criteria (from tests):**
- `StructuralParser(corpus_id).parse(lines)` returns `(units, edges)`
- `make_unit_id` normalises `^` → `_` and builds hierarchical IDs like `ro.codul_muncii.art_41.alin_1.lit_a`
- `validate_corpus(units)` raises `ValueError("Duplicate ID found")` on duplicate IDs
- Parser resets article state correctly when a new `Art. N` is encountered

**Source fixtures:** `ingestion/fixtures/codul_muncii.txt`, `ingestion/fixtures/constitutia.txt`
**Source list:** `ingestion/sources/demo_sources.yaml`

---

## Graph teammate

**Your modules:** `ingestion/reference_extractor.py`, `ingestion/reference_resolver.py`

**Your tests:**

```bash
pytest tests/test_references.py -v
```

**What to build / acceptance criteria (from tests):**
- `extract_references(unit)` → list of candidate dicts with keys:
  `source_unit_id`, `raw_reference`, `target_article`, `target_paragraph`, `target_law_hint`
- `target_law_hint` is `"same_act"` for *prezenta lege/cod*, `"external"` otherwise
- `resolve_references(candidates, all_units)` → `(resolved, edges)`
  - Resolved to paragraph → `confidence ≥ 0.85`, status `resolved_high_confidence`
  - Paragraph missing, article found → `confidence ≥ 0.60`, status `resolved_medium_confidence`
  - Nothing found → status `unresolved`, no edge emitted
  - External hint → status `external_unresolved`, no edge emitted
- Edge shape: `{"source_id": ..., "target_id": ..., "type": "references", "confidence": float}`

**Tip:** Run the parser first to get `legal_units.json`, then test resolution against real data:

```bash
python verify_parser.py          # generates legal_units.json
python -c "
import json
from ingestion.reference_extractor import extract_references
from ingestion.reference_resolver import resolve_references

units = json.load(open('legal_units.json'))
candidates = [c for u in units for c in extract_references(u)]
resolved, edges = resolve_references(candidates, units)
print(f'{len(edges)} edges resolved from {len(candidates)} candidates')
"
```

---

## Vectorial Embedding teammate

**Your modules:** `ingestion/embedding_service.py`

**Your tests:**

```bash
pytest tests/test_embeddings.py -v
```

**What to build / acceptance criteria (from tests):**
- `build_embedding_text(unit, corpus_metadata)` → a string that includes:
  - `Domain: <legal_domain>`
  - `Law: <law_title>`
  - `Path: A -> B -> ...` (from `unit["hierarchy_path"]`)
  - The raw text of the unit
- `MockEmbeddingProvider(dimension=1024).embed_texts(texts)` → list of float vectors, each of length `dimension`
- The real provider should target **`vector` (pgvector)** — the extension is already installed in the Railway DB

**Tip:** After generating `legal_units.json` with the parser, test embedding text construction at scale:

```bash
python -c "
import json
from ingestion.embedding_service import build_embedding_text

units = json.load(open('legal_units.json'))
corpus_metadata = {
    'ro.codul_muncii': {'law_title': 'Codul Muncii', 'legal_domain': 'dreptul_muncii'},
    'ro.constitutia':  {'law_title': 'Constitutia Romaniei', 'legal_domain': 'drept_constitutional'},
}
samples = [build_embedding_text(u, corpus_metadata) for u in units[:5]]
for s in samples:
    print(s[:120])
    print('---')
"
```

The DB already has `vector` extension active — when you're ready to persist embeddings,
the target table will be in `pgvector` format (`VECTOR(1024)` column).

---

## File map

```
ingestion/
  structural_parser.py   ← parser
  legal_ids.py           ← parser
  validators.py          ← parser
  manifest.py            ← parser
  parser_rules.py        ← parser
  reference_extractor.py ← graph
  reference_resolver.py  ← graph
  embedding_service.py   ← embedder
  fixtures/              ← sample Romanian law text files
  sources/               ← YAML list of law sources

apps/api/app/
  main.py                ← FastAPI entry point
  db.py                  ← SQLAlchemy async engine
  settings.py            ← env config (DATABASE_URL)
  services/retrieval_scoring.py ← hybrid BM25+vector scoring

tests/
  test_structural_parser.py
  test_validation.py
  test_references.py
  test_embeddings.py
  test_retrieval_scoring.py
```
