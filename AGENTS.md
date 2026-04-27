# LexAI — Codex Ruleset & Prompt Pack

Document de lucru pentru repo-ul `polihack-backend`.

Scop:

1. ruleset complet pentru `AGENTS.md`;
2. template-uri de prompt pentru taskuri viitoare;
3. prompturi concrete pentru următoarele milestone-uri LexAI;
4. reguli de output pentru workflow-ul ping-pong cu Codex + ChatGPT.

---

# 1. AGENTS.md — LexAI Backend Working Rules

> Copiază secțiunea de mai jos în `AGENTS.md` în root-ul repo-ului.

````md
# AGENTS.md — LexAI Backend Working Rules

## Project identity

This repository is `polihack-backend`, backend for LexAI.

LexAI is a Romanian Legal Operating System, not a generic legal chatbot.

The system ingests Romanian legislation, converts it into Atomic Legal Units, stores legal truth in PostgreSQL, supports hybrid retrieval, graph expansion, LegalRanker, EvidencePack, grounded generation, citation verification, and inspectable debug output.

Core product flow:

```txt
Romanian legal corpus
→ Atomic Legal Units
→ PostgreSQL + pgvector
→ hybrid retrieval
→ graph expansion
→ LegalRanker
→ EvidencePack
→ requirement coverage / backfill
→ grounded generation
→ CitationVerifier
→ AnswerRepair/refusal
→ Ask Mode + Explore Mode
```
````

## Non-negotiable legal safety rules

1. The LLM is never the legal source of truth.
2. Citable legal text must come from `LegalUnit.raw_text` / `legal_units.raw_text`.
3. `embedding_text`, `retrieval_text`, chunks, summaries, embeddings and reranker outputs are retrieval aids, not legal truth.
4. Do not invent laws, articles, citations, unit IDs, `source_url`, or legal conclusions.
5. Any legal claim in a generated answer must be supported by EvidencePack and citations.
6. If evidence is missing or weak, the system must warn, repair, or refuse.
7. Keep legal text separate from AI interpretation.
8. Debug output must make retrieval/ranking/evidence/verifier decisions inspectable.
9. The graph must support reasoning and evidence navigation, not just visualization.
10. Never “fix” legal text semantically. Encoding repair is allowed only as deterministic mojibake/diacritics repair.

## Official stack

Backend:

- Python
- FastAPI
- PostgreSQL
- pgvector
- SQLAlchemy / SQLModel where applicable
- Alembic where applicable
- Pydantic
- pytest

Corpus / ingestion:

- `legislatie.just.ro`
- BeautifulSoup
- regex structural parser
- Article → Paragraph → Letter → Point
- Romanian normalization
- no heavy NLP dependencies in initial parser unless explicitly requested

Retrieval:

- PostgreSQL full-text search
- pgvector dense search
- exact citation lookup
- lexical fallback
- RRF
- graph expansion
- LegalRanker
- optional Qwen reranker as a feature, not final authority

Frontend contracts:

- Ask Mode receives answer, citations, evidence units, verifier status, graph, debug, warnings.
- Explore Mode receives graph nodes/edges and highlighted legal path.

## Architecture boundaries

Backend AI/RAG owns:

- `/api/query`
- query understanding
- domain routing
- exact citation detection
- raw retrieval orchestration
- graph expansion policy
- LegalRanker
- EvidencePackCompiler
- RequirementBackfillService
- GenerationAdapter
- CitationVerifier
- AnswerRepair
- eval/debug endpoints

Backend Platform owns:

- DB models
- migrations
- import into PostgreSQL
- legal unit APIs
- graph APIs
- corpus stats
- raw retrieval storage substrate

Corpus/Ingestion owns:

- scraper/fetcher
- HTML cleaner
- structural parser
- canonical LegalUnit/LegalEdge bundle
- validation report
- legal chunks
- embeddings input
- reference candidates
- import manifests

## Current important state

The demo query is:

```txt
Poate angajatorul să-mi scadă salariul fără act adițional?
```

Expected legal evidence:

- `ro.codul_muncii.art_41.alin_1`
- `ro.codul_muncii.art_41.alin_3`
- optionally `ro.codul_muncii.art_41.alin_3.lit_e` if available

Correct citation behavior:

- cite art. 41 alin. (1)
- cite art. 41 alin. (3)
- do not cite generic salary distractors

Known implemented pieces:

- QueryUnderstanding
- ExactCitationDetector
- RawRetriever
- GraphExpansionPolicy
- LegalRanker
- EvidencePackCompiler
- RequirementBackfillService
- GenerationAdapter deterministic demo answer
- CitationVerifier
- AnswerRepair
- diacritics/mojibake repair path
- corpus diacritics diagnostic script

Recent fixes:

- `contract_modification_salary_scope` is answer-required.
- Art. 41 alin. (3)-like evidence is `condition` or `direct_basis`, not disposable `context`.
- Requirement backfill runs between EvidencePackCompiler and GenerationAdapter.
- Backfill repairs missing salary scope from real candidate pool.
- Citations for demo are verified and limited to art. 41 alin. (1) and alin. (3).
- Diacritics were repaired for parser/export/import/raw retrieval/generation path.
- For Windows debugging, use `curl.exe -o response.json` and decode UTF-8 manually when `Invoke-RestMethod` displays mojibake.

## Coding rules

1. Inspect existing files before editing.
2. Preserve public API contracts unless the task explicitly asks for contract changes.
3. Keep route handlers thin; put logic in services.
4. Prefer deterministic, testable services.
5. Do not introduce global side effects.
6. Do not hardcode legal conclusions into retrieval or verifier logic.
7. Hardcoded demo intent profiles are acceptable only as explicit V1 legal issue frames, not as hidden answer hacks.
8. Keep `LegalUnit.raw_text` citable and faithful.
9. `normalized_text` may be normalized for retrieval.
10. `retrieval_text` / `embedding_text` must never become citation text.
11. Any fallback must emit warnings/debug.
12. Any external model call must be optional, timeout-bound, and safely skipped if unavailable.
13. Never make the system fail just because Qwen reranker/embedder is unavailable.
14. Do not commit secrets, tokens, URLs with credentials, or local `.env` content.
15. Do not run destructive DB/file commands unless explicitly asked.

## Mathematical retrieval/ranking rules

Raw retrieval target formula, when dense is available:

```txt
S_raw =
0.30 * RRF
+ 0.25 * BM25
+ 0.20 * Dense
+ 0.10 * ExactCitation
+ 0.08 * DomainMatch
+ 0.04 * MetadataValidity
+ 0.03 * IntentPhraseMatch
```

Fallback without dense:

```txt
S_raw =
0.40 * RRF
+ 0.35 * BM25
+ 0.10 * ExactCitation
+ 0.08 * DomainMatch
+ 0.04 * MetadataValidity
+ 0.03 * IntentPhraseMatch
```

LegalRanker V1 target formula:

```txt
S_rank =
0.13 * bm25_score
+ 0.13 * dense_score
+ 0.14 * qwen_rerank_score
+ 0.10 * exact_citation_match
+ 0.09 * domain_match
+ 0.09 * graph_proximity
+ 0.07 * concept_overlap
+ 0.06 * legal_term_overlap
+ 0.07 * temporal_validity
+ 0.04 * source_reliability
+ 0.04 * parent_relevance
+ 0.02 * is_exception
+ 0.01 * is_definition
+ 0.01 * is_sanction
```

If Qwen reranker is not implemented/enabled, redistribute or leave `qwen_rerank_score = 0.0` safely.

Requirement backfill formula:

```txt
S_backfill(u, r) =
0.30 * RequirementMatch(u, r)
+ 0.20 * SameArticle(u, seed)
+ 0.15 * SameLaw(u, seed)
+ 0.15 * PhraseProximity(u)
+ 0.10 * UnitSpecificity(u)
+ 0.05 * ExistingRankQuality(u)
+ 0.05 * MetadataValidity(u)
- 0.20 * DistractorPenalty(u)
```

Evidence selection uses MMR:

```txt
MMR(u) =
0.75 * S_rank(u)
- 0.25 * max_similarity(u, selected)
```

Citation support:

```txt
Support =
0.40 * LexicalOverlap
+ 0.25 * ConceptOverlap
+ 0.25 * EmbeddingSimilarity
+ 0.10 * CitationConfidence
```

Thresholds:

```txt
>= 0.75 strongly_supported
0.60–0.75 supported
0.45–0.60 weakly_supported
< 0.45 unsupported
```

## Diacritics / encoding rules

Romanian diacritics must be preserved in `raw_text` and generated answers.

Correct:

- muncă
- părților
- Excepțiile
- situațiile
- unităților
- adițional

Incorrect:

- muncÄ
- pÄrÈilor
- ExcepÈiile
- situatiile
- unitatilor
- aditional, if displayed to user

`repair_romanian_mojibake()` is allowed for deterministic encoding repair.

Use:

- `encoding="utf-8"` for reading/writing files.
- `ensure_ascii=False` for JSON exports.
- `PYTHONUTF8=1` when running locally if needed.

Windows API test note:
`Invoke-RestMethod` may display UTF-8 incorrectly. For decisive verification, use:

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

## Local commands

Use the project venv.

```powershell
cd D:\dev\polihack-backend
.\.venv\Scripts\Activate.ps1
$env:PYTHONUTF8="1"
python -m uvicorn apps.api.app.main:app --host 127.0.0.1 --port 8010 --reload
```

Prefer `python -m uvicorn`, not global `uvicorn`.

Run tests:

```powershell
pytest tests/api/test_query.py -q
pytest tests/test_generation_adapter.py -q
pytest tests/test_requirement_backfill.py -q
pytest tests/test_live_demo_regression.py -q
pytest tests/api/test_query_persistence.py -q
pytest -q
```

For corpus diacritics:

```powershell
python scripts/check_corpus_diacritics.py
python scripts/check_corpus_diacritics.py tests/fixtures/corpus/codul_muncii_legal_units.json
```

## Expected task response format

After every implementation task, respond with:

```txt
1. Summary of changes
2. Files changed
3. Root cause addressed
4. Algorithm / architecture details
5. API/contract changes, if any
6. Tests run and results
7. Known limitations
8. Suggested next step
```

## Testing discipline

Every meaningful change needs tests.

Prefer:

- unit tests for pure services;
- API regression tests for `/api/query`;
- fixture-based tests for retrieval/evidence;
- negative tests for missing evidence/refusal;
- debug payload assertions where useful.

Do not report success without listing actual commands run and results.

## Git discipline

Use feature branches.

Do not push unless explicitly asked.

Recommended branch format:

```txt
feat/<short-task-name>
fix/<short-bug-name>
chore/<maintenance-name>
```

Before final response, check:

- no unrelated files changed;
- no generated junk unless intended;
- no secrets;
- tests pass;
- known warnings are documented.

````

---

# 2. Generic Codex task prompt template

```text
Lucrăm în repo `D:\dev\polihack-backend`, proiectul LexAI.

Respectă `AGENTS.md`.

Task: [scrie aici taskul exact]

Context curent:
- LexAI este Romanian Legal Operating System, nu legal chatbot.
- `LegalUnit.raw_text` / `legal_units.raw_text` este singura sursă citabilă.
- `/api/query` trebuie să producă răspuns grounded, evidence units, citations, verifier, graph, debug, warnings.
- Demo query: „Poate angajatorul să-mi scadă salariul fără act adițional?”
- Citările corecte pentru demo sunt `ro.codul_muncii.art_41.alin_1` și `ro.codul_muncii.art_41.alin_3`, verified.
- Nu inventa text legal, unit IDs, source URLs sau concluzii juridice.

Ce vreau să faci:
1. Inspectează codul existent înainte de modificări.
2. Propune intern soluția minimă robustă.
3. Implementează incremental.
4. Adaugă teste relevante.
5. Rulează testele specifice și, dacă are sens, full pytest.
6. Păstrează contractele API backward-compatible, cu debug opțional pentru câmpuri noi.
7. Nu folosi LLM ca sursă de adevăr juridic.
8. Nu schimba semantic `raw_text`.

Cerințe tehnice:
- [scrie cerințe concrete]
- [scrie fișiere probabile, dacă știi]
- [scrie endpoint-uri afectate]
- [scrie formula/scorul/algoritmul, dacă există]

Definition of Done:
- [criteriu 1]
- [criteriu 2]
- [criteriu 3]
- testele relevante trec
- output final include summary, files changed, tests run, limitations

La final răspunde exact cu:
1. Summary of changes
2. Files changed
3. Root cause addressed
4. Algorithm / architecture details
5. API/contract changes, if any
6. Tests run and results
7. Known limitations
8. Suggested next step
````

---

# 3. Prompt imediat recomandat — cleanup EvidencePack noise

````text
Lucrăm în repo `D:\dev\polihack-backend`, proiectul LexAI.

Respectă `AGENTS.md`.

Task: curăță zgomotul din EvidencePack pentru demo query fără să strici citările corecte.

Context:
După ultimele fixuri, `/api/query` pentru întrebarea:
„Poate angajatorul să-mi scadă salariul fără act adițional?”
produce citări corecte și verified:
- `ro.codul_muncii.art_41.alin_1`
- `ro.codul_muncii.art_41.alin_3`

Problema rămasă:
`EvidencePack` include prea multe unități contextuale și poate marca greșit unele unități, de exemplu `art_17.alin_3.lit_b`, ca `direct_basis`.

Obiectiv:
Pentru intentul `labor_contract_modification`, `direct_basis` trebuie să fie strict:
- agreement rule;
- contract modification salary scope;
- eventual salary-only child relevant, dar nu ca principal dacă alin. (3) are regula completă.

Cerințe:
1. Inspectează `legal_issue_frame.py`, `evidence_pack_compiler.py`, `generation_adapter.py`, testele demo.
2. Ajustează role classification astfel încât unitățile de tip art. 17 să nu fie `direct_basis` decât dacă acoperă explicit requirement-ul legal cerut.
3. Pentru demo query, limitează evidence pack-ul la un set mai compact, ideal 6–8 unități, fără să elimini:
   - `art_41.alin_1`
   - `art_41.alin_3`
4. Păstrează `art_42.alin_1` ca `exception` sau context dacă este util, dar nu îl cita ca bază principală.
5. Nu modifica citările corecte.
6. Nu hardcoda exclusiv unit IDs; folosește requirement matching, role scoring, article/scope signals.

Regulă de scoring:
Penalizează context generic:

```txt
S_evidence_final =
S_mmr
+ 0.20 * required_requirement_coverage
+ 0.10 * same_article_as_core
+ 0.08 * support_role_priority
- 0.15 * generic_context_penalty
- 0.20 * distractor_penalty
````

Support role priority:

```txt
direct_basis = 1.0
condition = 0.85
exception = 0.65
definition = 0.55
context = 0.35
```

Pentru acest intent, generic context penalty este mare dacă unitatea:

- conține doar termeni generici despre muncă/salariu;
- nu conține modificare contract / acordul părților / poate privi salariul;
- este din alt articol decât art. 41 și nu acoperă un requirement explicit.

Teste obligatorii:

1. API demo regression:
   - citations rămân exact art. 41 alin. (1) și art. 41 alin. (3), verified.
   - evidence_units include art. 41 alin. (1) și alin. (3).
   - art. 17 alin. 3 lit. b nu este `direct_basis`.
   - total evidence units pentru demo este <= 8, dacă nu afectează coverage.
2. Negative test:
   - dacă art. 41 alin. (3) lipsește complet, refusal/backfill behavior rămâne corect.
3. Rulează:
   - `pytest tests/api/test_query.py -q`
   - `pytest tests/test_live_demo_regression.py -q`
   - `pytest tests/test_requirement_backfill.py -q`
   - `pytest -q`

Definition of Done:

- EvidencePack demo este mai curat.
- Citările nu se schimbă greșit.
- `direct_basis` nu este atribuit unităților care nu acoperă requirement-uri directe.
- Debug rămâne inspectabil.

````

---

# 4. Prompt — Qwen reranker pe Ollama

```text
Lucrăm în repo `D:\dev\polihack-backend`, proiectul LexAI.

Respectă `AGENTS.md`.

Task: integrează un Qwen reranker opțional prin Ollama ca feature semantic în LegalRanker.

Context:
Avem deja Qwen embedder pe Ollama. Vrem să folosim Qwen reranker tot prin Ollama/Railway, dar doar ca semnal de ranking, nu ca autoritate juridică.

Model țintă:
- `dengcao/Qwen3-Reranker-0.6B:Q8_0`
sau modelul configurat prin env.

Ollama nu are endpoint nativ `/rerank`, deci V1 poate folosi `/api/generate` cu prompt care cere JSON score. Reranker-ul trebuie să fie optional, timeout-bound și fallback-safe.

Pipeline țintă:

```txt
RawRetriever
→ GraphExpansionPolicy
→ OllamaQwenReranker
→ LegalRanker
→ EvidencePackCompiler
→ RequirementBackfill
→ Generation
````

Cerințe:

1. Creează client nou:
   `apps/api/app/services/ollama_qwen_reranker.py`
2. Env vars:
   - `QWEN_RERANKER_ENABLED`
   - `OLLAMA_BASE_URL`
   - `QWEN_RERANKER_MODEL`
   - `QWEN_RERANKER_TOP_K`
   - `QWEN_RERANKER_TIMEOUT_SECONDS`
3. Clientul primește:
   - query;
   - listă candidați cu `unit_id` și document construit din `law_title`, `hierarchy_path`, article/paragraph labels, `raw_text`.
4. Nu trimite `embedding_text` ca sursă principală.
5. Returnează `dict[unit_id, score]` și warnings.
6. Dacă Ollama pică, returnează warning, dar pipeline-ul continuă.
7. Injectează scorul în `score_breakdown["qwen_rerank_score"]`.
8. Adaugă feature în LegalRanker.
9. Greutatea inițială pentru `qwen_rerank_score` trebuie să fie 0.10–0.14, nu mai mult.

Formula țintă:

```txt
S_rank =
0.13 * bm25_score
+ 0.13 * dense_score
+ 0.14 * qwen_rerank_score
+ 0.10 * exact_citation_match
+ 0.09 * domain_match
+ 0.09 * graph_proximity
+ 0.07 * concept_overlap
+ 0.06 * legal_term_overlap
+ 0.07 * temporal_validity
+ 0.04 * source_reliability
+ 0.04 * parent_relevance
+ 0.02 * is_exception
+ 0.01 * is_definition
+ 0.01 * is_sanction
```

Prompt reranker:

```txt
You are a Romanian legal retrieval reranker for LexAI.
Given a Romanian legal question and one candidate Romanian legal provision, score whether the provision directly helps answer the legal question.
Prefer governing rules, required conditions, exceptions, sanctions, definitions, or procedural steps.
Penalize provisions that only share generic words but do not answer the legal issue.
Do not infer law beyond the candidate text.
Return only valid JSON: {"score": number, "reason": "short reason"}.
```

Scoring:

```txt
1.0 = directly answers the legal issue
0.7 = useful condition/scope/exception/context needed for the answer
0.4 = weakly related
0.0 = generic or irrelevant lexical overlap
```

Tests:

1. Fake reranker provider:
   - art. 41 alin. (3) gets 0.92
   - generic salary article gets 0.25
   - LegalRanker uses qwen score and ranks correctly.
2. Unavailable Ollama:
   - warning emitted
   - pipeline still succeeds
   - qwen score defaults to 0.0
3. API debug includes qwen reranker summary when debug=true.
4. Run:
   - `pytest tests/api/test_query.py -q`
   - `pytest tests/test_live_demo_regression.py -q`
   - `pytest -q`

Definition of Done:

- Qwen reranker is optional.
- No failure when Ollama is down.
- Qwen score is visible in debug/score_breakdown.
- LegalRanker remains final decision layer.
- EvidencePack/CitationVerifier still gate legal answer.

````

---

# 5. Prompt — DB / corpus diacritics check

```text
Lucrăm în repo `D:\dev\polihack-backend`, proiectul LexAI.

Respectă `AGENTS.md`.

Task: adaugă un script DB-only pentru verificarea diacriticelor în `legal_units.raw_text`.

Context:
Am reparat diacriticele în parser/export/import/raw retriever/generation și avem `scripts/check_corpus_diacritics.py` pentru bundle JSON. Dar vrem și verificare directă a DB-ului, ca să știm dacă datele persistate în PostgreSQL sunt curate sau doar reparate la hidratare.

Obiectiv:
Creează script:
`scripts/check_db_diacritics.py`

Cerințe:
1. Se conectează la DB folosind `DATABASE_URL` / config existent.
2. Citește `id`, `law_id`, `article_number`, `paragraph_number`, `raw_text` din `legal_units`.
3. Detectează mojibake românesc folosind `contains_romanian_mojibake()`.
4. Afișează:
   - unit_id
   - law_id
   - location article/paragraph/letter
   - fragment corupt
5. Exit code:
   - 0 dacă nu găsește probleme;
   - 1 dacă găsește mojibake;
   - 2 dacă DB nu e configurat/conectabil.
6. Să fie safe read-only.
7. Să nu repare DB-ul. Doar diagnostic.

Teste:
- unit test cu fake rows dacă există infrastructură ușoară;
- sau test pentru funcția de formatting/detection.
- Rulează `pytest tests/ingestion/test_normalizer.py -q`.
- Rulează `pytest -q` dacă modificarea atinge config/db.

Definition of Done:
- `python scripts/check_db_diacritics.py` raportează clar starea DB.
- Nu modifică DB.
- Folosește detectorul central din `ingestion.normalizer`.
````

---

# 6. Prompt — raw retrieval RRF scoring upgrade

````text
Lucrăm în repo `D:\dev\polihack-backend`, proiectul LexAI.

Respectă `AGENTS.md`.

Task: fă RRF un semnal real în raw retrieval score, nu doar tie-breaker/debug.

Context:
`RawRetriever` calculează RRF peste exact citation, FTS, dense și domain-filtered rankings, dar `weighted_retrieval_score()` nu include RRF în formula principală. Vrem ca o unitate care apare bine în mai multe metode să fie ridicată mai robust.

Obiectiv:
Modifică `apps/api/app/services/retrieval_scoring.py` și testele relevante astfel încât `rrf` să fie inclus în `S_raw`.

Formula cu dense:

```txt
S_raw =
0.30 * RRF
+ 0.25 * BM25
+ 0.20 * Dense
+ 0.10 * ExactCitation
+ 0.08 * DomainMatch
+ 0.04 * MetadataValidity
+ 0.03 * IntentPhraseMatch
````

Dacă nu există `IntentPhraseMatch` în contract încă, fie:

- adaugă opțional în ScoreBreakdown cu default 0.0;
- fie redistribuie temporar greutatea de 0.03 către BM25, dar documentează.

Formula fără dense:

```txt
S_raw =
0.40 * RRF
+ 0.35 * BM25
+ 0.10 * ExactCitation
+ 0.08 * DomainMatch
+ 0.04 * MetadataValidity
+ 0.03 * IntentPhraseMatch
```

Cerințe:

1. Păstrează scorul în [0, 1].
2. Normalizează RRF dacă este necesar. Nu lăsa valori mici brute de tip 0.016 să fie irelevante.
3. Debug să afișeze `rrf`.
4. Testează că un candidat prezent în BM25 + dense + exact are scor mai mare decât unul prezent doar în BM25.
5. Testează fallback fără dense.
6. Rulează:
   - `pytest tests/api/test_retrieve_raw.py -q`
   - `pytest tests/api/test_query.py -q`
   - `pytest -q`

Definition of Done:

- RRF afectează real ranking-ul.
- Nu se strică schema `score_breakdown`.
- Demo query păstrează art. 41 alin. (1)/(3) în top/evidence.

````

---

# 7. Prompt — query embedding integration

```text
Lucrăm în repo `D:\dev\polihack-backend`, proiectul LexAI.

Respectă `AGENTS.md`.

Task: integrează query embedding în `/api/query`, astfel încât dense retrieval să fie real, nu skipped.

Context:
`RawRetrievalRequest` acceptă `query_embedding`, iar `RawRetriever` are `dense_search()`, dar `RawRetrieverClient.build_request()` nu trimite query embedding. În debug apare des `dense_retrieval_skipped_no_query_embedding`.

Obiectiv:
Adaugă un `QueryEmbeddingService` opțional care produce embedding pentru întrebare și îl trimite către `RawRetrieverClient`.

Cerințe:
1. Creează service:
   `apps/api/app/services/query_embedding_service.py`
2. Env/config:
   - `QUERY_EMBEDDING_ENABLED`
   - `OLLAMA_BASE_URL`
   - `QUERY_EMBEDDING_MODEL`
   - `QUERY_EMBEDDING_TIMEOUT_SECONDS`
3. Folosește Ollama `/api/embed` sau endpointul deja folosit pentru Qwen embedder.
4. Dacă embedding service pică:
   - continuă fără dense;
   - adaugă warning `query_embedding_unavailable`;
   - nu bloca `/api/query`.
5. `RawRetrieverClient.build_request()` trebuie să poată primi `query_embedding`.
6. Debug:
   - model;
   - dimensiune;
   - enabled/skipped/unavailable.
7. Nu trimite vectorul în răspuns public/debug complet dacă e mare. Doar dimensiune și status.
8. Verifică dimensiunea cu `legal_embeddings.embedding_dim` unde este posibil.

Teste:
1. Fake embedding service returnează vector.
2. RawRetrieverClient trimite `query_embedding`.
3. Dacă service e unavailable, warning și fallback lexical.
4. Rulează:
   - `pytest tests/api/test_query.py -q`
   - `pytest tests/api/test_retrieve_raw.py -q`
   - `pytest -q`

Definition of Done:
- `/api/query` poate rula dense retrieval când embedding service este disponibil.
- Dense fallback este safe.
- Debug arată clar dacă dense a rulat sau nu.
````

---

# 8. Prompt — graph neighbors real from DB

```text
Lucrăm în repo `D:\dev\polihack-backend`, proiectul LexAI.

Respectă `AGENTS.md`.

Task: conectează `GraphExpansionPolicy` la neighbors reali din DB.

Context:
`GraphExpansionPolicy` este implementat, dar dacă nu are `neighbors_client`, cade pe seed-only fallback. Vrem ca pentru un seed precum `ro.codul_muncii.art_41.alin_1`, sistemul să poată aduce parent/siblings/children/reference edges din `legal_units` și `legal_edges`.

Obiectiv:
Implementează un `GraphNeighborsClient` / `PostgresGraphNeighborsStore` care citește din DB și alimentează `GraphExpansionPolicy`.

Cerințe:
1. Creează service:
   `apps/api/app/services/graph_neighbors_client.py`
2. Pentru un `unit_id`, returnează records compatibile cu `GraphExpansionPolicy._neighbor_records()`.
3. Include:
   - parent via `parent_id`;
   - children via `parent_id = unit_id`;
   - legal_edges unde `source_id = unit_id` sau `target_id = unit_id`;
   - edge types: contains_parent, contains_child, references, defines, exception_to, sanctions.
4. Limite:
   - max_depth respectat;
   - max nodes per seed;
   - no infinite loops.
5. Pentru demo:
   - seed `art_41.alin_1` trebuie să poată ajunge la `art_41` și/sau siblings relevante;
   - ideal `art_41.alin_3` și `art_41.alin_3.lit_e`, dacă există în DB.
6. Fallback safe dacă DB nu e disponibil.
7. Debug include:
   - seed count;
   - neighbor count;
   - edge types used;
   - fallback reason dacă nu merge.

Teste:
1. Unit test cu fake DB rows:
   - parent/child expansion.
2. Integration-ish test cu fixtures:
   - art_41.alin_1 expands to art_41.alin_3.
3. `/api/query` demo still cites art. 41 alin. (1)/(3).
4. Rulează:
   - `pytest tests/test_graph_expansion_policy.py -q`
   - `pytest tests/api/test_query.py -q`
   - `pytest -q`

Definition of Done:
- GraphExpansionPolicy nu mai e doar seed-only când DB are edges.
- Expansion îmbunătățește EvidencePack fără să inventeze unități.
- Debug e inspectabil.
```

---

# 9. Standard ping-pong workflow

Folosește așa:

1. Dai promptul către Codex.
2. Codex implementează și îți dă raport.
3. Trimiți raportul către ChatGPT.
4. ChatGPT verifică arhitectural:
   - dacă a respectat LexAI;
   - dacă există risc juridic;
   - dacă există overfitting;
   - dacă testele sunt suficiente;
   - ce urmează.
5. Dacă e nevoie, ChatGPT îți dă prompt de follow-up pentru Codex.

Răspunsul Codex trebuie să aibă mereu:

```txt
1. Summary of changes
2. Files changed
3. Root cause addressed
4. Algorithm / architecture details
5. API/contract changes, if any
6. Tests run and results
7. Known limitations
8. Suggested next step
```

Dacă Codex nu listează teste rulate, cere explicit testele înainte să consideri taskul închis.

---

# 10. Git helper commands

## Branch nou

```powershell
git checkout dev
git pull origin dev
git checkout -b feat/<task-name>
```

## Status

```powershell
git status
```

## Add/commit

```powershell
git add .
git commit -m "feat: <short description>"
```

## Push

```powershell
git push origin feat/<task-name>
```

## După merge în dev

```powershell
git checkout dev
git pull origin dev
```

---

# 11. Current recommended next order

Ordinea recomandată după stadiul actual:

```txt
1. Cleanup EvidencePack noise
2. Qwen reranker optional via Ollama
3. RRF scoring upgrade
4. Query embedding integration, if not already fully wired
5. Graph neighbors real from DB
6. DB diacritics diagnostic script
7. Broaden legal issue frames beyond labor contract modification
8. Eval benchmark with 50–100 legal questions
```

Prioritatea practică pentru demo:

1. EvidencePack cleanup.
2. Qwen reranker, dacă Ollama/Railway e stabil.
3. Graph expansion real DB.
