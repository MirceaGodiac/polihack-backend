# Arhitectura parserului LexAI

## 1. Overview

Parserul LexAI transformă surse locale sau HTML legal în artefacte JSON canonice, verificabile și pregătite pentru import. LexAI nu este un legal chatbot: parserul nu interpretează juridic legea, nu formulează concluzii și nu folosește un LLM ca sursă de adevăr juridic.

Fluxul curent este:

```text
raw/local input
  -> HTML cleaner
  -> structural parser / legacy units
  -> LegalUnit canonic
  -> LegalEdge contains
  -> ReferenceCandidate
  -> LegalChunk
  -> embeddings_input.jsonl
  -> corpus_manifest.json
  -> validation_report.json
```

`LegalUnit.raw_text` este singura sursă citabilă. `LegalChunk.retrieval_text` este derivat pentru retrieval și nu trebuie citat ca text de lege. `ReferenceCandidate` nu este `LegalEdge`; referințele ambigue sau nerezolvate rămân candidate/unresolved.

Principiul de bază este: unknown stays unknown. Parserul nu inventează `source_url`, `source_id`, `status`, date, concepte juridice, referințe rezolvate sau concluzii juridice.

## 2. Ownership boundary

Parser / Handoff 06 deține:

- producerea bundle-urilor canonice file-based;
- extracția LegalUnit;
- strategia de ID-uri deterministe;
- ierarhia act -> articol -> alineat -> literă;
- HTML cleaner-ul;
- extracția de ReferenceCandidate;
- generarea LegalChunk și embeddings input;
- manifestul și validation report-ul.

Backend Platform / Handoff 04 deține:

- importul bundle-ului în PostgreSQL;
- schema DB, migrations și eventual Alembic;
- `legal_units` APIs;
- Explore APIs și graph APIs;
- search și raw retrieval;
- pgvector și importul vectorilor.

AI/RAG / Handoff 03 deține:

- consumul RetrievalCandidate;
- graph expansion peste unități și edges;
- LegalRanker;
- EvidencePackCompiler;
- GenerationAdapter;
- CitationVerifier.

Handoff 03 poate consuma LegalUnits reale, dar nu trebuie să trateze LLM-ul ca sursă de drept. Evidence și citările trebuie să se bazeze pe `LegalUnit.raw_text`.

## 3. File/folder map

`ingestion/contracts/__init__.py`
: Definește contractele ingestion-side: `ParserActMetadata`, `ParsedLegalUnit`, `ParsedLegalEdge`, `LegalChunk`, `ReferenceCandidate`, `CorpusManifest`, `ValidationReport`, `EmbeddingInputRecord`.

`ingestion/legal_ids.py`
: Conține strategia deterministă de ID-uri pentru law/art/alin/lit/pct și canonical IDs. Normalizează forme precum `41^1` și `41¹`.

`ingestion/legal_domains.py`
: Registry conservator pentru domenii juridice cunoscute, de exemplu `ro.codul_muncii -> munca`.

`ingestion/normalizer.py`
: Normalizează text derivat pentru matching/retrieval. Nu înlocuiește `raw_text`.

`ingestion/html_cleaner.py`
: Curăță HTML local: elimină script/style/nav/header/footer/search/menu/cookie-like noise și păstrează markerii juridici.

`ingestion/reference_extractor.py`
: Extractor rule-based pentru referințe legislative românești. Produce `ReferenceCandidate`, nu `LegalEdge`.

`ingestion/chunks.py`
: Generează `LegalChunk`, `retrieval_context`, `retrieval_text` și `EmbeddingInputRecord`.

`ingestion/exporters.py`
: Convertește legacy units în LegalUnits canonice, construiește contains edges, reference candidates, chunks, manifest și validation report.

`ingestion/validators.py`
: Conține validări legacy. Validarea canonică robustă este construită în `ingestion/exporters.py`.

`ingestion/bundle_loader.py`
: Loader file-based pentru bundle canonic. Încarcă artefactele, verifică fișierele obligatorii și construiește indexuri unit/chunk/adjacency.

`ingestion/local_retriever.py`
: Retriever local de dezvoltare/test. Scorarea folosește `LegalChunk.retrieval_text`, dar evidence text vine din `LegalUnit.raw_text`.

`scripts/export_canonical_bundle.py`
: CLI local pentru export din legacy units în bundle canonic. Scrie artefactele P8 și iese non-zero dacă `import_blocking_passed=false`.

`tests/fixtures/corpus/*`
: Fixture-uri canonice Codul Muncii și mini Codul Muncii. Sunt folosite pentru integrarea parserului cu Handoff 03 și pentru readiness checks.

`docs/ingestion-canonical-bundle.md`
: Document scurt despre bundle-ul canonic și consumul lui de Handoff 03/04.

## 4. Canonical data flow

```text
Raw HTML / local fixture
  -> clean_html_to_lines
  -> structural parser / legacy units
  -> ParsedLegalUnit
  -> LegalUnit dict
  -> contains LegalEdges
  -> ReferenceCandidate extraction
  -> LegalChunk generation
  -> EmbeddingInputRecord JSONL
  -> CorpusManifest
  -> ValidationReport
```

Transformări:

- HTML cleaner-ul extrage linii curate, fără navigație evidentă.
- Parserul structural sau legacy produce unități inițiale.
- Exporterul canonic convertește unitățile în `ParsedLegalUnit`.
- `ParsedLegalUnit.to_legal_unit_dict()` produce forma compatibilă cu backend `LegalUnit`.
- Parent-child produce `LegalEdge` de tip `contains`.
- `reference_extractor` citește lexical `raw_text` și produce candidați nerezolvați.
- `chunks.py` produce retrieval context determinist și `retrieval_text`.
- Manifestul conține metadate, counts și hash-uri.
- Validation report-ul decide dacă bundle-ul este import-ready.

## 5. LegalUnit

`LegalUnit` este unitatea juridică citabilă. Este puntea dintre parser, raw retrieval, evidence pack și citări. Un `EvidenceUnit` extinde `LegalUnit`, deci parserul trebuie să producă LegalUnits complete, nu dict-uri parțiale.

Câmpuri importante:

- `id`: ID stabil și citabil.
- `canonical_id`: ID intern compact, fără prefixul `ro.`.
- `law_id`: actul normativ.
- `law_title`: titlul actului.
- `hierarchy_path`: traseu complet pentru UI/RAG.
- `article_number`, `paragraph_number`, `letter_number`: localizare juridică.
- `raw_text`: textul juridic citabil.
- `normalized_text`: text derivat pentru matching, nu sursă citabilă.
- `legal_domain`: domeniu din registry sau `unknown`.
- `legal_concepts`: listă conservatoare, default `[]`.
- `parent_id`: unitatea părinte.
- `parser_warnings`: lipsuri sau decizii explicite de tip unknown/null.

Exemplu scurt:

```json
{
  "id": "ro.codul_muncii.art_41.alin_4",
  "canonical_id": "codul_muncii:art_41:alin_4",
  "law_id": "ro.codul_muncii",
  "law_title": "Codul muncii",
  "status": "unknown",
  "hierarchy_path": [
    "Legislatia Romaniei",
    "Munca",
    "Codul muncii",
    "Art. 41",
    "Alin. (4)"
  ],
  "article_number": "41",
  "paragraph_number": "4",
  "raw_text": "(4) Orice modificare a unuia dintre elementele prevazute la alin. (3), in timpul executarii contractului individual de munca, impune incheierea unui act aditional la contract...",
  "normalized_text": "(4) Orice modificare a unuia dintre elementele prevazute la alin. (3)...",
  "legal_domain": "munca",
  "legal_concepts": [],
  "parent_id": "ro.codul_muncii.art_41",
  "source_url": null,
  "parser_warnings": [
    "source_url_unknown",
    "status_unknown"
  ]
}
```

`raw_text` nu trebuie alterat semantic. Curățarea și normalizarea derivată trebuie să păstreze textul juridic citabil intact.

## 6. ID strategy

ID-urile sunt deterministe. `LegalUnit.id` nu folosește UUID.

Exemple:

- `ro.codul_muncii`
- `ro.codul_muncii.art_41`
- `ro.codul_muncii.art_41.alin_4`
- `ro.codul_muncii.art_17.alin_3.lit_k`
- `ro.codul_muncii.art_41_1`

Exemplu `canonical_id`:

- `codul_muncii:art_41:alin_4`

Normalizarea tratează forme precum:

- `Art. 41^1 -> art_41_1`
- `Art. 41¹ -> art_41_1`

Această strategie permite import DB, graph traversal, retrieval și evidence/citation pe aceeași cheie juridică stabilă.

## 7. HTML cleaner

`ingestion/html_cleaner.py` elimină conservator:

- `script`, `style`, `noscript`, `svg`;
- `nav`;
- `header`/`footer` când sunt clar navigaționale;
- breadcrumbs, search boxes, buttons, menus și cookie banners detectabile.

Păstrează:

- titlul actului;
- titluri/capitole/secțiuni;
- `Art.`, `Articolul`;
- alineate `(1)`;
- litere `k)`;
- puncte;
- diacritice.

Cleaner-ul produce warnings precum `legal_container_not_found`, `used_body_fallback`, `removed_navigation_blocks` sau `possible_navigation_residue`. Nu interpretează legea și nu rescrie semantic textul legal.

## 8. Reference extraction

`ingestion/reference_extractor.py` detectează lexical:

- `art. 41`, `art 41`, `articolul 41`;
- `Art. 41^1`, `Art. 41¹`;
- `alin. (3)`, `alineatul (3)`;
- `lit. k)`, `litera k)`;
- `pct. 2`, `punctul 2`;
- `teza a II-a`;
- `Legea nr. 53/2003`;
- `O.U.G. nr. 195/2002`, `OUG nr. 195/2002`;
- `O.G. nr. 2/2001`, `OG nr. 2/2001`;
- `H.G. nr. 1/2016`, `HG nr. 1/2016`;
- coduri numite, de exemplu `Codul muncii`, `Codul civil`;
- referințe locale precum `prezentul cod`.

Produce `ReferenceCandidate`. Nu produce `LegalEdge` de tip `references`. Rezolvarea completă este deferred.

Exemple:

```json
{
  "source_unit_id": "ro.codul_muncii.art_41.alin_4",
  "raw_reference": "alin. (3)",
  "reference_type": "paragraph",
  "target_law_hint": "same_act",
  "target_article": null,
  "target_paragraph": "3",
  "resolved_target_id": null,
  "resolution_status": "unresolved_needs_context"
}
```

```json
{
  "source_unit_id": "ro.demo.art_1",
  "raw_reference": "art. 17 alin. (3) lit. k)",
  "reference_type": "compound",
  "target_law_hint": "same_act",
  "target_article": "17",
  "target_paragraph": "3",
  "target_letter": "k",
  "resolved_target_id": null,
  "resolution_status": "candidate_only"
}
```

```json
{
  "source_unit_id": "ro.demo.art_1",
  "raw_reference": "Legea nr. 53/2003",
  "reference_type": "law",
  "target_law_hint": "ro.lege_53_2003",
  "resolved_target_id": null,
  "resolution_status": "candidate_only"
}
```

```json
{
  "source_unit_id": "ro.codul_muncii.art_41",
  "raw_reference": "prezentul cod",
  "reference_type": "same_act",
  "target_law_hint": "same_act",
  "resolved_target_id": null,
  "resolution_status": "unresolved_needs_context"
}
```

## 9. LegalEdges

În stadiul curent se produc doar `contains` edges sigure. Acestea conectează:

```text
act -> articol -> alineat -> literă
```

Exemplu:

```json
{
  "id": "edge.contains.ro.codul_muncii.art_41.ro.codul_muncii.art_41.alin_4",
  "source_id": "ro.codul_muncii.art_41",
  "target_id": "ro.codul_muncii.art_41.alin_4",
  "type": "contains",
  "weight": 1.0,
  "confidence": 1.0
}
```

`references` edges nu se creează din candidați ambigui. Handoff 03 și Handoff 04 pot folosi `contains` pentru graph expansion, children/parents și UI Explore.

## 10. LegalChunk + contextual retrieval

`LegalChunk` este derivat din `LegalUnit`. Nu este sursă juridică citabilă.

Reguli:

- `text` și `raw_text` vin din `LegalUnit.raw_text`;
- `retrieval_context` este determinist;
- contextul se bazează pe hierarchy, law title, legal domain, metadata părinte și reference candidates nerezolvate;
- `retrieval_text = retrieval_context + "\n\n" + raw_text`;
- `embeddings_input.jsonl` folosește `retrieval_text`;
- metadata marchează contextul ca non-citable.

Mecanismul este inspirat de contextual retrieval, dar implementat determinist pentru LexAI, fără LLM.

Exemplu:

```json
{
  "chunk_id": "chunk.ro.codul_muncii.art_41.alin_4.0",
  "legal_unit_id": "ro.codul_muncii.art_41.alin_4",
  "text": "(4) Orice modificare a unuia dintre elementele prevazute la alin. (3)...",
  "retrieval_context": "Unitate din Codul muncii, domeniul munca.\nLocalizare: Art. 41, Alin. (4).\nContext ierarhic: Legislatia Romaniei > Munca > Codul muncii > Art. 41 > Alin. (4).\nUnitate parinte: Art. 41.\nReferinte extrase nerezolvate: alin. (3).",
  "retrieval_text": "Unitate din Codul muncii, domeniul munca...\n\n(4) Orice modificare...",
  "context_generation_method": "deterministic_v1",
  "context_confidence": 0.75,
  "metadata": {
    "retrieval_context_is_citable": false,
    "text_source": "LegalUnit.raw_text"
  }
}
```

## 11. Canonical bundle output

`legal_units.json`
: Conține LegalUnits canonice. Consumatori: Handoff 03 evidence/retrieval, Handoff 04 import DB. Este sursa citabilă prin `raw_text`.

`legal_edges.json`
: Conține `contains` edges. Consumatori: Handoff 03 graph expansion, Handoff 04 graph APIs. Nu conține reference edges fragile.

`reference_candidates.json`
: Conține referințe detectate lexical și nerezolvate sau candidate-only. Consumatori: Handoff 04/P7+ pentru rezolvare ulterioară. Nu este sursă citabilă și nu este graph edge.

`legal_chunks.json`
: Conține chunks pentru retrieval/vector. Consumatori: Handoff 03 retrieval adapters, Handoff 04 staging pentru chunks. `retrieval_text` nu este citabil.

`embeddings_input.jsonl`
: JSONL determinist pentru job viitor de embeddings. Consumatori: Handoff 04/embedding job ulterior. Nu conține vectori reali.

`corpus_manifest.json`
: Conține parser version, generated_at, counts, input files, hashes, warnings. Consumatori: import/readiness checks.

`validation_report.json`
: Conține metrici, warnings, blocking errors, `import_blocking_passed`, `demo_path_passed`. Decide import readiness.

## 12. Storage boundary

Parser storage:

- bundle-uri canonice file-based;
- output local în `ingestion/output/<bundle_name>/`;
- fixture-uri mici versionate în `tests/fixtures/corpus/`;
- manifest, hash-uri și validation report.

Nu este responsabilitatea parserului:

- PostgreSQL;
- pgvector;
- Alembic;
- runtime API persistence;
- import DB real.

## 13. Validation gates

Validation report-ul include:

- `corpus_quality`: scor ponderat, nu cosmetizat;
- `import_blocking_passed`: true/false pentru import readiness;
- `demo_path_passed`: true/false/null pentru demo Codul Muncii;
- `blocking_errors`;
- `warnings`;
- `quality_metrics`.

Metrici importante:

- `unit_completeness`;
- `hierarchy_integrity`;
- `edge_endpoint_integrity`;
- `duplicate_free_score`;
- `source_url_coverage`;
- `text_cleanliness`;
- `reference_resolution_rate`;
- `chunk_coverage_rate`;
- `embedding_input_hash_integrity`.

Exemple de blocking:

- duplicate LegalUnit IDs;
- invalid edge endpoint;
- empty `raw_text` pentru unități citabile;
- `resolved_target_id` non-null dar inexistent;
- embedding hash greșit;
- chunk cu `legal_unit_id` invalid;
- `retrieval_text` gol;
- context cu interpretări juridice hardcodate.

`source_url=null` poate fi warning non-blocking pentru fixture-uri locale, dacă manifestul și validation report-ul explică sursa demo locală.

## 14. Local bundle loader and local retriever

`ingestion.bundle_loader`:

- încarcă bundle-ul canonic file-based;
- verifică required files;
- suportă fișiere standard și fixture-uri prefixate;
- construiește indexuri unit/chunk;
- construiește adjacency pentru `contains`.

`ingestion.local_retriever`:

- folosește `LegalChunk.retrieval_text` pentru scoring lexical determinist;
- aplică token overlap, domain boost și exact citation boost simplu;
- returnează candidate compatibil cu `RetrievalCandidate`;
- păstrează evidence text din `LegalUnit.raw_text`;
- este dev/test adapter, nu production DB retriever.

## 15. Integration with Handoff 03

Handoff 03 primește LegalUnits reale prin fixture-uri:

- fake raw retriever încarcă `tests/fixtures/corpus/codul_muncii_legal_units.json`;
- `GraphExpansionPolicy` folosește `codul_muncii_legal_edges.json`;
- `LegalRanker` rankează LegalUnits reale;
- `EvidencePackCompiler` produce `EvidenceUnit` flat cu `raw_text`;
- generator/verifier sunt încă mock/unverified.

Demo query:

```text
Poate angajatorul să-mi scadă salariul fără act adițional?
```

Unit IDs relevante:

- `ro.codul_muncii.art_41`
- `ro.codul_muncii.art_41.alin_3`
- `ro.codul_muncii.art_41.alin_4`
- `ro.codul_muncii.art_17.alin_3.lit_k`

Testul principal este `tests/test_handoff03_fixture_integration.py`.

## 16. Integration with Handoff 04

Platform owner trebuie să implementeze runtime/import real peste bundle-ul canonic:

- `POST /api/ingest/json`;
- tabel `legal_units`;
- tabel `legal_edges`;
- staging pentru chunks/embeddings;
- `GET /api/legal-units/{id}`;
- `GET /api/legal-units/{id}/neighbors`;
- `POST /api/retrieve/raw`;
- Explore APIs;
- pgvector import ulterior din `embeddings_input.jsonl`.

În repo, aceste rute/servicii runtime nu sunt complet implementate încă. Unele fișiere există ca schelet gol, de exemplu `apps/api/app/routes/retrieve_raw.py`, `apps/api/app/routes/explore.py`, `apps/api/app/services/import_service.py`, `apps/api/app/services/raw_retriever.py`, `apps/api/app/services/graph_store.py`.

## 17. How to run

Pipeline direct din URL `legislatie.just.ro` catre bundle canonic:

```powershell
py -3.13 scripts/run_parser_pipeline.py `
  --url "https://legislatie.just.ro/Public/DetaliiDocument/123456" `
  --out-dir ingestion/output/legislatie_codul_muncii `
  --law-id ro.codul_muncii `
  --law-title "Codul muncii" `
  --write-debug
```

Scriptul valideaza domeniul URL-ului, descarca HTML-ul, ruleaza cleaner-ul HTML,
trimite liniile in `StructuralParser`, apoi exporta bundle-ul canonic complet:
`legal_units.json`, `legal_edges.json`, `reference_candidates.json`,
`legal_chunks.json`, `embeddings_input.jsonl`, `corpus_manifest.json` si
`validation_report.json`. Parametrii `--law-id` si `--law-title` sunt recomandati
pentru ID-uri stabile; daca lipsesc, scriptul incearca inferenta conservatoare din
titlul HTML sau din document id-ul URL-ului.

Export bundle canonic din legacy units:

```powershell
py -3.13 scripts/export_canonical_bundle.py `
  --input tests/fixtures/corpus/codul_muncii_legacy_units.json `
  --out-dir ingestion/output/codul_muncii_canonical `
  --law-id ro.codul_muncii `
  --law-title "Codul muncii" `
  --generated-at 2026-04-25T00:00:00+00:00
```

Workflow-ul curent este local/test-first. Nu există încă un CLI complet pentru crawling oficial end-to-end în formă canonică production-ready.

Clarificare P11: acum exista un CLI pentru un singur URL `legislatie.just.ro`;
inca nu exista un crawler production-ready pentru corpusuri oficiale mari.

Teste relevante:

```powershell
py -3.13 -m pytest tests/test_bundle_loader.py
py -3.13 -m pytest tests/test_run_parser_pipeline.py
py -3.13 -m pytest tests/test_local_retriever.py
py -3.13 -m pytest tests/test_parser_integration_readiness.py
py -3.13 -m pytest tests/test_handoff03_fixture_integration.py
py -3.13 -m pytest tests/test_legal_chunks.py
py -3.13 -m pytest tests/test_canonical_bundle_export.py
py -3.13 -m pytest tests/test_validation.py
py -3.13 -m pytest
```

## 18. Current status

- Ultima verificare dupa pipeline URL: `244 passed, 1 warning`.
- Codul Muncii este integration-ready file-based.
- Ultima verificare locală: `240 passed, 1 existing warning`.
- Runtime API/DB integration nu este completă.
- Nu există embeddings reale.
- Nu există reference resolution complet.
- Nu există DB import real.
- `validation_report.import_blocking_passed=true` pentru fixture-ul Codul Muncii.
- `validation_report.demo_path_passed=true` pentru fixture-ul Codul Muncii.

## 19. Known limitations

- Corpus demo limitat.
- Nu există ingestie completă din sursă oficială pentru corpus mare.
- Referințele rămân unresolved/candidate-only.
- Nu există runtime DB import.
- Nu există pgvector import.
- Nu există embeddings reale.
- Nu există GenerationAdapter final.
- Nu există CitationVerifier final.
- Artefactele legacy root-level `legal_units.json` și `legal_edges.json` pot induce confuzie față de bundle-ul canonic.
- `apps/api/app/schemas/legal.py` păstrează o schemă legacy diferită de `apps/api/app/schemas/query.py::LegalUnit`.

## 20. Next recommended steps

1. Handoff 04: import canonic din bundle și implementare `/api/retrieve/raw`.
2. Handoff 04: endpoint-uri pentru `legal_units`, neighbors și Explore APIs peste DB.
3. Handoff 03: GenerationAdapter și CitationVerifier peste EvidencePack real.
4. Parser continuation: adăugare 1-2 acte demo prin același workflow canonic.
5. Cleanup: clarificarea sau mutarea artefactelor legacy root-level.
