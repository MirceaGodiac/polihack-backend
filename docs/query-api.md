# Query API

Phase 3 exposes the `/api/query` contract, deterministic QueryUnderstanding,
DomainRouter debug data, ExactCitationDetector output, and a deterministic mock
EvidencePack only. It is intended for frontend and API integration work before
real retrieval and answer generation are available.

Not implemented yet:

- database-backed retrieval
- graph expansion
- LegalRanker
- answer generation
- citation verification

The query understanding layer is rule-based and inspectable. It does not call an
LLM and it does not retrieve legal text. Exact citations are parsed only into
future lookup hints; they are not resolved against `legal_units` yet.

## Request

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Poate angajatorul sa-mi scada salariul fara act aditional?",
    "jurisdiction": "RO",
    "date": "current",
    "mode": "strict_citations",
    "debug": true
  }'
```

## Response Excerpt

```json
{
  "query_id": "9f8d85df-8d8b-59d5-b905-f76c351c7f10",
  "question": "Poate angajatorul sa-mi scada salariul fara act aditional?",
  "answer": {
    "short_answer": "Phase 1 mock response only. No verified legal conclusion is provided; the evidence below is deterministic placeholder data for API contract testing.",
    "detailed_answer": null,
    "confidence": 0.0,
    "not_legal_advice": true,
    "refusal_reason": "mock_evidence_pack_not_verified"
  },
  "citations": [
    {
      "citation_id": "mock-citation-1",
      "evidence_id": "mock-evidence-1",
      "legal_unit_id": "mock:ro:codul-muncii:art-17",
      "label": "Codul muncii art. 17 (mock, unverified)",
      "quote": "Mock excerpt for art. 17. This placeholder was not retrieved from an official source.",
      "source_url": null,
      "verified": false
    }
  ],
  "verifier": {
    "groundedness_score": 0.0,
    "claims_total": 0,
    "claims_supported": 0,
    "claims_weakly_supported": 0,
    "claims_unsupported": 0,
    "citations_checked": 0,
    "verifier_passed": false,
    "claim_results": [],
    "warnings": [
      "mock_unverified_evidence_pack: Phase 1 returns deterministic mock evidence only; retrieval, LegalRanker, generation, and citation verification have not run."
    ],
    "repair_applied": false,
    "refusal_reason": "mock_evidence_pack_not_verified"
  }
}
```

When `debug` is `true`, the response includes a `debug` object with mock service
counts, notes, and `query_understanding`. When `debug` is `false`, `debug` is
`null`.

Example debug excerpt:

```json
{
  "debug": {
    "query_understanding": {
      "legal_domain": "muncă",
      "domain_confidence": 1.0,
      "query_types": ["right", "prohibition", "obligation"],
      "exact_citations": [],
      "temporal_context": "current",
      "retrieval_filters": {
        "legal_domain": "muncă",
        "status": "active",
        "date_context": "current"
      },
      "expansion_policy": {
        "max_depth": 2,
        "max_expanded_nodes": 80
      }
    }
  }
}
```

Exact citation example:

```json
{
  "debug": {
    "query_understanding": {
      "legal_domain": "muncă",
      "exact_citations": [
        {
          "raw_text": "art. 41 alin. (1) din Codul muncii",
          "citation_type": "compound",
          "article": "41",
          "paragraph": "1",
          "letter": null,
          "act_hint": "Codul muncii",
          "law_id_hint": "ro.codul_muncii",
          "confidence": 0.98,
          "is_relative": false,
          "needs_resolution": false,
          "lookup_filters": {
            "law_id": "ro.codul_muncii",
            "article_number": "41",
            "paragraph_number": "1",
            "status": "active"
          }
        }
      ]
    }
  }
}
```
