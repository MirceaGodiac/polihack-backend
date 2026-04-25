# Query API

Phase 7 exposes the `/api/query` contract, deterministic QueryUnderstanding,
DomainRouter debug data, ExactCitationDetector output, RawRetrieverClient
request construction, GraphExpansionPolicy debug data, LegalRanker debug data,
and EvidencePackCompiler MMR selection. It is intended for frontend and API
integration work before real retrieval, graph traversal, answer generation, and
citation verification are available.

Not implemented yet:

- database-backed retrieval
- `/api/retrieve/raw`, which is owned by Handoff 04
- graph/neighbors endpoints, which are owned by Handoff 04
- database-backed graph expansion
- answer generation
- citation verification

The query understanding layer is rule-based and inspectable. It does not call an
LLM and it does not retrieve legal text. Exact citations are parsed only into
future lookup hints; they are not resolved against `legal_units` yet.
RawRetrieverClient prepares the future raw retrieval payload. GraphExpansionPolicy
turns raw retrieval candidates into graph expansion seeds and policy metadata.
LegalRanker reranks raw and expanded candidates deterministically when they
exist. EvidencePackCompiler selects and diversifies ranked candidates with MMR
when candidates include LegalUnit text. If raw retrieval or graph neighbors are
not configured, `/api/query` returns a safe fallback with empty evidence,
`confidence: 0.0`, `verifier_passed: false`, and explicit warnings.

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
    "short_answer": "Phase 1 mock response only. No verified legal conclusion is provided; Phase 7 may compile evidence units separately, but generation is not configured.",
    "detailed_answer": null,
    "confidence": 0.0,
    "not_legal_advice": true,
    "refusal_reason": "mock_evidence_pack_not_verified"
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
    "warnings": [
      "mock_unverified_answer: Phase 7 keeps answer generation and citation verification mocked; compiled evidence, if present, is not converted into a verified legal conclusion."
    ],
    "repair_applied": false,
    "refusal_reason": "mock_evidence_pack_not_verified"
  }
}
```

When ranked candidates contain LegalUnit text, `evidence_units` use a flat
LegalUnit-plus-evidence shape. The LegalUnit fields are not nested under
`legal_unit`.

```json
{
  "evidence_units": [
    {
      "id": "ro.codul_muncii.art_41",
      "law_id": "ro.codul_muncii",
      "law_title": "Codul muncii",
      "status": "active",
      "hierarchy_path": ["Codul muncii", "art. 41"],
      "article_number": "41",
      "paragraph_number": null,
      "raw_text": "Contractul individual de munca poate fi modificat prin acordul partilor.",
      "normalized_text": null,
      "legal_domain": "munca",
      "legal_concepts": ["contract", "salariu"],
      "source_url": "https://legislatie.just.ro/test",
      "evidence_id": "evidence:ro.codul_muncii.art_41",
      "excerpt": "Contractul individual de munca poate fi modificat prin acordul partilor.",
      "rank": 1,
      "relevance_score": 0.86,
      "retrieval_method": "raw_retrieval",
      "retrieval_score": 0.78,
      "rerank_score": 0.86,
      "mmr_score": 0.645,
      "support_role": "direct_basis",
      "why_selected": ["domain_match:munca", "selected_by_mmr"],
      "score_breakdown": {
        "bm25_score": 0.9,
        "dense_score": 0.7,
        "domain_match": 1.0,
        "graph_proximity": 1.0
      },
      "warnings": []
    }
  ]
}
```

When `debug` is `true`, the response includes a `debug` object with mock service
counts, notes, `query_understanding`, `retrieval`, `graph_expansion`,
`legal_ranker`, and `evidence_pack`. When `debug` is `false`, `debug` is
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

LegalRanker debug excerpt when no candidates are available:

```json
{
  "debug": {
    "legal_ranker": {
      "fallback_used": true,
      "input_candidate_count": 0,
      "ranked_candidate_count": 0,
      "weights": {
        "bm25_score": 0.16,
        "dense_score": 0.16,
        "exact_citation_match": 0.1,
        "domain_match": 0.1,
        "graph_proximity": 0.1,
        "concept_overlap": 0.08,
        "legal_term_overlap": 0.07,
        "temporal_validity": 0.07,
        "source_reliability": 0.05,
        "parent_relevance": 0.05,
        "is_exception": 0.03,
        "is_definition": 0.02,
        "is_sanction": 0.01
      },
      "ranked_candidates": [],
      "rows": [],
      "warnings": ["legal_ranker_no_candidates"]
    }
  }
}
```

EvidencePackCompiler debug excerpt when no ranked candidates are available:

```json
{
  "debug": {
    "evidence_pack": {
      "fallback_used": true,
      "input_ranked_candidate_count": 0,
      "candidate_pool_size": 0,
      "selected_evidence_count": 0,
      "lambda": 0.75,
      "target_evidence_units": 12,
      "max_evidence_units": 14,
      "selected_units": [],
      "warnings": ["evidence_pack_no_ranked_candidates"]
    }
  }
}
```

EvidencePackCompiler debug excerpt with ranked fixture candidates:

```json
{
  "debug": {
    "evidence_pack": {
      "fallback_used": false,
      "input_ranked_candidate_count": 1,
      "candidate_pool_size": 1,
      "selected_evidence_count": 1,
      "selected_units": [
        {
          "unit_id": "ro.codul_muncii.art_41",
          "rerank_score": 0.86,
          "mmr_score": 0.645,
          "support_role": "direct_basis",
          "why_selected": [
            "domain_match:munca",
            "high_bm25_score",
            "selected_by_mmr"
          ]
        }
      ],
      "warnings": ["evidence_pack_partial"]
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

Raw retrieval debug excerpt when `/api/retrieve/raw` is not configured:

```json
{
  "debug": {
    "retrieval": {
      "request_payload": {
        "question": "Ce spune art. 41 alin. (1) din Codul muncii?",
        "retrieval_filters": {
          "legal_domain": "muncă",
          "exact_citation_filters": [
            {
              "law_id": "ro.codul_muncii",
              "article_number": "41",
              "paragraph_number": "1",
              "status": "active"
            }
          ]
        },
        "exact_citations": [
          {
            "article": "41",
            "paragraph": "1",
            "act_hint": "Codul muncii"
          }
        ],
        "top_k": 50,
        "debug": true
      },
      "response_summary": {
        "candidate_count": 0,
        "retrieval_methods": [],
        "warnings": ["raw_retrieval_not_configured"]
      },
      "fallback_used": true
    }
  }
}
```

Graph expansion debug excerpt when raw retrieval has no seed candidates:

```json
{
  "debug": {
    "graph_expansion": {
      "fallback_used": true,
      "reason": "graph expansion has no seed candidates",
      "policy": {
        "max_depth": 2,
        "max_expanded_nodes": 80,
        "lambda_decay": 0.7,
        "allowed_edge_types": [
          "contains_parent",
          "contains_child",
          "references",
          "defines",
          "exception_to",
          "sanctions",
          "creates_obligation",
          "creates_right",
          "creates_prohibition",
          "procedure_step"
        ],
        "priority_edge_types": [
          "exception_to",
          "sanctions",
          "creates_obligation",
          "creates_prohibition",
          "creates_right"
        ]
      },
      "seed_candidate_count": 0,
      "expanded_candidate_count": 0,
      "expanded_candidates": [],
      "graph_node_count": 0,
      "graph_edge_count": 0,
      "warnings": ["graph_expansion_no_seed_candidates"]
    }
  }
}
```
