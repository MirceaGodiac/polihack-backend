from pathlib import Path

from ingestion.bundle_loader import CanonicalBundle
from ingestion.local_retriever import LocalBundleRetriever


FIXTURE_DIR = Path("tests/fixtures/corpus")
DEMO_QUERY = "Poate angajatorul sa-mi scada salariul fara act aditional?"


def test_local_retriever_recovers_codul_muncii_demo_units():
    retriever = LocalBundleRetriever.from_path(FIXTURE_DIR)

    candidates = retriever.retrieve(DEMO_QUERY, top_k=5)

    top_ids = [candidate.unit_id for candidate in candidates]
    assert "ro.codul_muncii.art_41" in top_ids
    assert "ro.codul_muncii.art_41.alin_4" in top_ids
    assert any(
        unit_id == "ro.codul_muncii.art_17.alin_3.lit_k"
        for unit_id in top_ids
    )
    assert candidates[0].scored_text_source == "LegalChunk.retrieval_text"
    assert candidates[0].evidence_text_source == "LegalUnit.raw_text"
    assert candidates[0].evidence_text == candidates[0].unit["raw_text"]
    assert candidates[0].score_breakdown["scored_text_retrieval_text"] == 1.0


def test_local_retriever_candidate_dict_is_retrieval_candidate_compatible():
    retriever = LocalBundleRetriever.from_path(FIXTURE_DIR)

    payload = retriever.retrieve(DEMO_QUERY, top_k=1)[0].to_retrieval_candidate_dict()

    assert payload["unit_id"]
    assert payload["rank"] == 1
    assert payload["retrieval_score"] > 0
    assert payload["score_breakdown"]["lexical_overlap"] > 0
    assert payload["unit"]["raw_text"]
    assert "retrieval_context" not in payload["unit"]


def test_local_retriever_scores_retrieval_text_but_keeps_raw_text_for_evidence():
    unit = {
        "id": "ro.test.art_1",
        "law_id": "ro.test",
        "law_title": "Test",
        "raw_text": "Text citabil fara termenul de scoring.",
        "legal_domain": "munca",
    }
    chunk = {
        "chunk_id": "chunk.ro.test.art_1.0",
        "legal_unit_id": "ro.test.art_1",
        "retrieval_text": "Context derivat pentru magicneedle.",
        "text": unit["raw_text"],
    }
    bundle = CanonicalBundle(
        root_path=Path("."),
        artifact_paths={},
        legal_units=[unit],
        legal_edges=[],
        reference_candidates=[],
        legal_chunks=[chunk],
        embeddings_input=[],
        corpus_manifest={},
        validation_report={},
    )

    candidate = LocalBundleRetriever(bundle).retrieve("magicneedle", top_k=1)[0]

    assert candidate.unit_id == "ro.test.art_1"
    assert candidate.matched_terms == ["magicneedle"]
    assert candidate.scored_text_source == "LegalChunk.retrieval_text"
    assert candidate.evidence_text_source == "LegalUnit.raw_text"
    assert candidate.evidence_text == unit["raw_text"]
    assert "magicneedle" not in candidate.evidence_text
