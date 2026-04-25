# Arhitectura parserului LexAI

## 1. Overview

Parserul LexAI transformÄƒ surse locale sau HTML legal Ã®n artefacte JSON canonice, verificabile È™i pregÄƒtite pentru import. LexAI nu este un legal chatbot: parserul nu interpreteazÄƒ juridic legea, nu formuleazÄƒ concluzii È™i nu foloseÈ™te un LLM ca sursÄƒ de adevÄƒr juridic.

Fluxul curent este:

```text
raw/local input
  -> HTML cleaner
  -> structural parser / intermediate units
  -> LegalUnit canonic
  -> LegalEdge contains
  -> ReferenceCandidate
  -> LegalChunk
  -> embeddings_input.jsonl
  -> corpus_manifest.json
  -> validation_report.json
```

`LegalUnit.raw_text` este singura sursÄƒ citabilÄƒ. `LegalChunk.retrieval_text` este derivat pentru retrieval È™i nu trebuie citat ca text de lege. `ReferenceCandidate` nu este `LegalEdge`; referinÈ›ele ambigue sau nerezolvate rÄƒmÃ¢n candidate/unresolved.

Principiul de bazÄƒ este: unknown stays unknown. Parserul nu inventeazÄƒ `source_url`, `source_id`, `status`, date, concepte juridice, referinÈ›e rezolvate sau concluzii juridice.

## 2. Ownership boundary

Parser / Handoff 06 deÈ›ine:

- producerea bundle-urilor canonice file-based;
- extracÈ›ia LegalUnit;
- strategia de ID-uri deterministe;
- ierarhia act -> articol -> alineat -> literÄƒ;
- HTML cleaner-ul;
- extracÈ›ia de ReferenceCandidate;
- generarea LegalChunk È™i embeddings input;
- manifestul È™i validation report-ul.

Backend Platform / Handoff 04 deÈ›ine:

- importul bundle-ului Ã®n PostgreSQL;
- schema DB, migrations È™i eventual Alembic;
- `legal_units` APIs;
- Explore APIs È™i graph APIs;
- search È™i raw retrieval;
- pgvector È™i importul vectorilor.

AI/RAG / Handoff 03 deÈ›ine:

- consumul RetrievalCandidate;
- graph expansion peste unitÄƒÈ›i È™i edges;
- LegalRanker;
- EvidencePackCompiler;
- GenerationAdapter;
- CitationVerifier.

Handoff 03 poate consuma LegalUnits reale, dar nu trebuie sÄƒ trateze LLM-ul ca sursÄƒ de drept. Evidence È™i citÄƒrile trebuie sÄƒ se bazeze pe `LegalUnit.raw_text`.

## 3. File/folder map

`ingestion/contracts/__init__.py`
: DefineÈ™te contractele ingestion-side: `ParserActMetadata`, `ParsedLegalUnit`, `ParsedLegalEdge`, `LegalChunk`, `ReferenceCandidate`, `CorpusManifest`, `ValidationReport`, `EmbeddingInputRecord`.

`ingestion/legal_ids.py`
: ConÈ›ine strategia deterministÄƒ de ID-uri pentru law/art/alin/lit/pct È™i canonical IDs. NormalizeazÄƒ forme precum `41^1` È™i `41Â¹`.

`ingestion/legal_domains.py`
: Registry conservator pentru domenii juridice cunoscute, de exemplu `ro.codul_muncii -> munca`.

`ingestion/normalizer.py`
: NormalizeazÄƒ text derivat pentru matching/retrieval. Nu Ã®nlocuieÈ™te `raw_text`.

`ingestion/html_cleaner.py`
: CurÄƒÈ›Äƒ HTML local: eliminÄƒ script/style/nav/header/footer/search/menu/cookie-like noise È™i pÄƒstreazÄƒ markerii juridici.

`ingestion/reference_extractor.py`
: Extractor rule-based pentru referinÈ›e legislative romÃ¢neÈ™ti. Produce `ReferenceCandidate`, nu `LegalEdge`.

`ingestion/chunks.py`
: GenereazÄƒ `LegalChunk`, `retrieval_context`, `retrieval_text` È™i `EmbeddingInputRecord`.

`ingestion/exporters.py`
: ConverteÈ™te unitÄƒÈ›i structurale intermediare Ã®n LegalUnits canonice, construieÈ™te contains edges, reference candidates, chunks, manifest È™i validation report.

`ingestion/validators.py`
: ConÈ›ine validÄƒri legacy. Validarea canonicÄƒ robustÄƒ este construitÄƒ Ã®n `ingestion/exporters.py`.

`ingestion/bundle_loader.py`
: Loader file-based pentru bundle canonic. ÃŽncarcÄƒ artefactele, verificÄƒ fiÈ™ierele obligatorii È™i construieÈ™te indexuri unit/chunk/adjacency.

`ingestion/local_retriever.py`
: Retriever local de dezvoltare/test. Scorarea foloseÈ™te `LegalChunk.retrieval_text`, dar evidence text vine din `LegalUnit.raw_text`.

`scripts/export_canonical_bundle.py`
: CLI local pentru export din unitÄƒÈ›i structurale intermediare Ã®n bundle canonic. Scrie artefactele P8 È™i iese non-zero dacÄƒ `import_blocking_passed=false`.

`ingestion/pipeline.py`
: Pipeline reutilizabil pentru URL HTML: fetch, HTML cleaner, `StructuralParser`, export bundle canonic complet.

`scripts/run_parser_pipeline.py`
: CLI subtire peste `ingestion.pipeline.run_pipeline`.

`tests/fixtures/corpus/*`
: Fixture-uri canonice Codul Muncii È™i mini Codul Muncii. Sunt folosite pentru integrarea parserului cu Handoff 03 È™i pentru readiness checks.

`docs/ingestion-canonical-bundle.md`
: Document scurt despre bundle-ul canonic È™i consumul lui de Handoff 03/04.

## 4. Canonical data flow

```text
Raw HTML / local fixture
  -> clean_html_to_lines
  -> structural parser / intermediate units
  -> ParsedLegalUnit
  -> LegalUnit dict
  -> contains LegalEdges
  -> ReferenceCandidate extraction
  -> LegalChunk generation
  -> EmbeddingInputRecord JSONL
  -> CorpusManifest
  -> ValidationReport
```

TransformÄƒri:

- HTML cleaner-ul extrage linii curate, fÄƒrÄƒ navigaÈ›ie evidentÄƒ.
- Parserul structural sau legacy produce unitÄƒÈ›i iniÈ›iale.
- Exporterul canonic converteÈ™te unitÄƒÈ›ile Ã®n `ParsedLegalUnit`.
- `ParsedLegalUnit.to_legal_unit_dict()` produce forma compatibilÄƒ cu backend `LegalUnit`.
- Parent-child produce `LegalEdge` de tip `contains`.
- `reference_extractor` citeÈ™te lexical `raw_text` È™i produce candidaÈ›i nerezolvaÈ›i.
- `chunks.py` produce retrieval context determinist È™i `retrieval_text`.
- Manifestul conÈ›ine metadate, counts È™i hash-uri.
- Validation report-ul decide dacÄƒ bundle-ul este import-ready.

## 5. LegalUnit

`LegalUnit` este unitatea juridicÄƒ citabilÄƒ. Este puntea dintre parser, raw retrieval, evidence pack È™i citÄƒri. Un `EvidenceUnit` extinde `LegalUnit`, deci parserul trebuie sÄƒ producÄƒ LegalUnits complete, nu dict-uri parÈ›iale.

CÃ¢mpuri importante:

- `id`: ID stabil È™i citabil.
- `canonical_id`: ID intern compact, fÄƒrÄƒ prefixul `ro.`.
- `law_id`: actul normativ.
- `law_title`: titlul actului.
- `hierarchy_path`: traseu complet pentru UI/RAG.
- `article_number`, `paragraph_number`, `letter_number`: localizare juridicÄƒ.
- `raw_text`: textul juridic citabil.
- `normalized_text`: text derivat pentru matching, nu sursÄƒ citabilÄƒ.
- `legal_domain`: domeniu din registry sau `unknown`.
- `legal_concepts`: listÄƒ conservatoare, default `[]`.
- `parent_id`: unitatea pÄƒrinte.
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

`raw_text` nu trebuie alterat semantic. CurÄƒÈ›area È™i normalizarea derivatÄƒ trebuie sÄƒ pÄƒstreze textul juridic citabil intact.

## 6. ID strategy

ID-urile sunt deterministe. `LegalUnit.id` nu foloseÈ™te UUID.

Exemple:

- `ro.codul_muncii`
- `ro.codul_muncii.art_41`
- `ro.codul_muncii.art_41.alin_4`
- `ro.codul_muncii.art_17.alin_3.lit_k`
- `ro.codul_muncii.art_41_1`

Exemplu `canonical_id`:

- `codul_muncii:art_41:alin_4`

Normalizarea trateazÄƒ forme precum:

- `Art. 41^1 -> art_41_1`
- `Art. 41Â¹ -> art_41_1`

AceastÄƒ strategie permite import DB, graph traversal, retrieval È™i evidence/citation pe aceeaÈ™i cheie juridicÄƒ stabilÄƒ.

## 7. HTML cleaner

`ingestion/html_cleaner.py` eliminÄƒ conservator:

- `script`, `style`, `noscript`, `svg`;
- `nav`;
- `header`/`footer` cÃ¢nd sunt clar navigaÈ›ionale;
- breadcrumbs, search boxes, buttons, menus È™i cookie banners detectabile.

PÄƒstreazÄƒ:

- titlul actului;
- titluri/capitole/secÈ›iuni;
- `Art.`, `Articolul`;
- alineate `(1)`;
- litere `k)`;
- puncte;
- diacritice.

Cleaner-ul produce warnings precum `legal_container_not_found`, `used_body_fallback`, `removed_navigation_blocks` sau `possible_navigation_residue`. Nu interpreteazÄƒ legea È™i nu rescrie semantic textul legal.

## 8. Reference extraction

`ingestion/reference_extractor.py` detecteazÄƒ lexical:

- `art. 41`, `art 41`, `articolul 41`;
- `Art. 41^1`, `Art. 41Â¹`;
- `alin. (3)`, `alineatul (3)`;
- `lit. k)`, `litera k)`;
- `pct. 2`, `punctul 2`;
- `teza a II-a`;
- `Legea nr. 53/2003`;
- `O.U.G. nr. 195/2002`, `OUG nr. 195/2002`;
- `O.G. nr. 2/2001`, `OG nr. 2/2001`;
- `H.G. nr. 1/2016`, `HG nr. 1/2016`;
- coduri numite, de exemplu `Codul muncii`, `Codul civil`;
- referinÈ›e locale precum `prezentul cod`.

Produce `ReferenceCandidate`. Nu produce `LegalEdge` de tip `references`. Rezolvarea completÄƒ este deferred.

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

ÃŽn stadiul curent se produc doar `contains` edges sigure. Acestea conecteazÄƒ:

```text
act -> articol -> alineat -> literÄƒ
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

`references` edges nu se creeazÄƒ din candidaÈ›i ambigui. Handoff 03 È™i Handoff 04 pot folosi `contains` pentru graph expansion, children/parents È™i UI Explore.

## 10. LegalChunk + contextual retrieval

`LegalChunk` este derivat din `LegalUnit`. Nu este sursÄƒ juridicÄƒ citabilÄƒ.

Reguli:

- `text` È™i `raw_text` vin din `LegalUnit.raw_text`;
- `retrieval_context` este determinist;
- contextul se bazeazÄƒ pe hierarchy, law title, legal domain, metadata pÄƒrinte È™i reference candidates nerezolvate;
- `retrieval_text = retrieval_context + "\n\n" + raw_text`;
- `embeddings_input.jsonl` foloseÈ™te `retrieval_text`;
- metadata marcheazÄƒ contextul ca non-citable.

Mecanismul este inspirat de contextual retrieval, dar implementat determinist pentru LexAI, fÄƒrÄƒ LLM.

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
: ConÈ›ine LegalUnits canonice. Consumatori: Handoff 03 evidence/retrieval, Handoff 04 import DB. Este sursa citabilÄƒ prin `raw_text`.

`legal_edges.json`
: ConÈ›ine `contains` edges. Consumatori: Handoff 03 graph expansion, Handoff 04 graph APIs. Nu conÈ›ine reference edges fragile.

`reference_candidates.json`
: ConÈ›ine referinÈ›e detectate lexical È™i nerezolvate sau candidate-only. Consumatori: Handoff 04/P7+ pentru rezolvare ulterioarÄƒ. Nu este sursÄƒ citabilÄƒ È™i nu este graph edge.

`legal_chunks.json`
: ConÈ›ine chunks pentru retrieval/vector. Consumatori: Handoff 03 retrieval adapters, Handoff 04 staging pentru chunks. `retrieval_text` nu este citabil.

`embeddings_input.jsonl`
: JSONL determinist pentru job viitor de embeddings. Consumatori: Handoff 04/embedding job ulterior. Nu conÈ›ine vectori reali.

`corpus_manifest.json`
: ConÈ›ine parser version, generated_at, counts, input files, hashes, warnings. Consumatori: import/readiness checks.

`validation_report.json`
: ConÈ›ine metrici, warnings, blocking errors, `import_blocking_passed`, `demo_path_passed`. Decide import readiness.

## 12. Storage boundary

Parser storage:

- bundle-uri canonice file-based;
- output local Ã®n `ingestion/output/<bundle_name>/`;
- fixture-uri mici versionate Ã®n `tests/fixtures/corpus/`;
- manifest, hash-uri È™i validation report.

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
- empty `raw_text` pentru unitÄƒÈ›i citabile;
- `resolved_target_id` non-null dar inexistent;
- embedding hash greÈ™it;
- chunk cu `legal_unit_id` invalid;
- `retrieval_text` gol;
- context cu interpretÄƒri juridice hardcodate.

`source_url=null` poate fi warning non-blocking pentru fixture-uri locale, dacÄƒ manifestul È™i validation report-ul explicÄƒ sursa demo localÄƒ.

## 14. Local bundle loader and local retriever

`ingestion.bundle_loader`:

- Ã®ncarcÄƒ bundle-ul canonic file-based;
- verificÄƒ required files;
- suportÄƒ fiÈ™iere standard È™i fixture-uri prefixate;
- construieÈ™te indexuri unit/chunk;
- construieÈ™te adjacency pentru `contains`.

`ingestion.local_retriever`:

- foloseÈ™te `LegalChunk.retrieval_text` pentru scoring lexical determinist;
- aplicÄƒ token overlap, domain boost È™i exact citation boost simplu;
- returneazÄƒ candidate compatibil cu `RetrievalCandidate`;
- pÄƒstreazÄƒ evidence text din `LegalUnit.raw_text`;
- este dev/test adapter, nu production DB retriever.

## 15. Integration with Handoff 03

Handoff 03 primeÈ™te LegalUnits reale prin fixture-uri:

- fake raw retriever Ã®ncarcÄƒ `tests/fixtures/corpus/codul_muncii_legal_units.json`;
- `GraphExpansionPolicy` foloseÈ™te `codul_muncii_legal_edges.json`;
- `LegalRanker` rankeazÄƒ LegalUnits reale;
- `EvidencePackCompiler` produce `EvidenceUnit` flat cu `raw_text`;
- generator/verifier sunt Ã®ncÄƒ mock/unverified.

Demo query:

```text
Poate angajatorul sÄƒ-mi scadÄƒ salariul fÄƒrÄƒ act adiÈ›ional?
```

Unit IDs relevante:

- `ro.codul_muncii.art_41`
- `ro.codul_muncii.art_41.alin_3`
- `ro.codul_muncii.art_41.alin_4`
- `ro.codul_muncii.art_17.alin_3.lit_k`

Testul principal este `tests/test_handoff03_fixture_integration.py`.

## 16. Integration with Handoff 04

Platform owner trebuie sÄƒ implementeze runtime/import real peste bundle-ul canonic:

- `POST /api/ingest/json`;
- tabel `legal_units`;
- tabel `legal_edges`;
- staging pentru chunks/embeddings;
- `GET /api/legal-units/{id}`;
- `GET /api/legal-units/{id}/neighbors`;
- `POST /api/retrieve/raw`;
- Explore APIs;
- pgvector import ulterior din `embeddings_input.jsonl`.

ÃŽn repo, aceste rute/servicii runtime nu sunt complet implementate Ã®ncÄƒ. Unele fiÈ™iere existÄƒ ca schelet gol, de exemplu `apps/api/app/routes/retrieve_raw.py`, `apps/api/app/routes/explore.py`, `apps/api/app/services/import_service.py`, `apps/api/app/services/raw_retriever.py`, `apps/api/app/services/graph_store.py`.

## 17. How to run

Pipeline direct din URL HTML catre bundle canonic:

```powershell
py -3.13 scripts/run_parser_pipeline.py `
  --url "https://legislatie.just.ro/Public/DetaliiDocument/123456" `
  --out-dir ingestion/output/legislatie_codul_muncii `
  --law-id ro.codul_muncii `
  --law-title "Codul muncii" `
  --write-debug
```

Scriptul valideaza ca URL-ul este `http`/`https`, descarca HTML-ul, ruleaza cleaner-ul HTML,
trimite liniile in `StructuralParser`, apoi exporta bundle-ul canonic complet:
`legal_units.json`, `legal_edges.json`, `reference_candidates.json`,
`legal_chunks.json`, `embeddings_input.jsonl`, `corpus_manifest.json` si
`validation_report.json`. Parametrii `--law-id` si `--law-title` sunt recomandati
pentru ID-uri stabile; daca lipsesc, scriptul incearca inferenta conservatoare din
titlul HTML sau din document id-ul URL-ului.

Export bundle canonic din unitÄƒÈ›i structurale intermediare:

```powershell
py -3.13 scripts/export_canonical_bundle.py `
  --input tests/fixtures/corpus/codul_muncii_legacy_units.json `
  --out-dir ingestion/output/codul_muncii_canonical `
  --law-id ro.codul_muncii `
  --law-title "Codul muncii" `
  --generated-at 2026-04-25T00:00:00+00:00
```

Workflow-ul curent este local/test-first. Exista un CLI pentru un singur URL HTML,
dar nu exista inca un crawler production-ready pentru corpusuri oficiale mari.

Teste relevante:

```powershell
py -3.13 -m pytest tests/ingestion/test_bundle_loader.py
py -3.13 -m pytest tests/ingestion/test_run_parser_pipeline.py
py -3.13 -m pytest tests/ingestion/test_local_retriever.py
py -3.13 -m pytest tests/ingestion/test_parser_integration_readiness.py
py -3.13 -m pytest tests/test_handoff03_fixture_integration.py
py -3.13 -m pytest tests/ingestion/test_legal_chunks.py
py -3.13 -m pytest tests/ingestion/test_canonical_bundle_export.py
py -3.13 -m pytest tests/ingestion/test_validation.py
py -3.13 -m pytest
```

## 18. Current status

- Ultima verificare dupa reorganizarea pipeline/teste: `214 passed, 1 warning`.
- Codul Muncii este integration-ready file-based.
- Runtime API/DB integration nu este completÄƒ.
- Nu existÄƒ embeddings reale.
- Nu existÄƒ reference resolution complet.
- Nu existÄƒ DB import real.
- `validation_report.import_blocking_passed=true` pentru fixture-ul Codul Muncii.
- `validation_report.demo_path_passed=true` pentru fixture-ul Codul Muncii.

## 19. Known limitations

- Corpus demo limitat.
- Nu existÄƒ ingestie completÄƒ din sursÄƒ oficialÄƒ pentru corpus mare.
- ReferinÈ›ele rÄƒmÃ¢n unresolved/candidate-only.
- Nu existÄƒ runtime DB import.
- Nu existÄƒ pgvector import.
- Nu existÄƒ embeddings reale.
- Nu existÄƒ GenerationAdapter final.
- Nu existÄƒ CitationVerifier final.
- Artefactele legacy root-level `legal_units.json` È™i `legal_edges.json` pot induce confuzie faÈ›Äƒ de bundle-ul canonic.
- `apps/api/app/schemas/legal.py` pÄƒstreazÄƒ o schemÄƒ legacy diferitÄƒ de `apps/api/app/schemas/query.py::LegalUnit`.

## 20. Next recommended steps

1. Handoff 04: import canonic din bundle È™i implementare `/api/retrieve/raw`.
2. Handoff 04: endpoint-uri pentru `legal_units`, neighbors È™i Explore APIs peste DB.
3. Handoff 03: GenerationAdapter È™i CitationVerifier peste EvidencePack real.
4. Parser continuation: adÄƒugare 1-2 acte demo prin acelaÈ™i workflow canonic.
5. Cleanup: clarificarea sau mutarea artefactelor legacy root-level.
