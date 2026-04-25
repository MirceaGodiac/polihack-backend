from apps.api.app.schemas import (
    GraphExpansionResult,
    RankedCandidate,
    RankerFeatureBreakdown,
)
from apps.api.app.services.evidence_pack_compiler import (
    EVIDENCE_PACK_NO_RANKED_CANDIDATES,
    EvidencePackCompiler,
)


def unit(unit_id: str, text: str, **overrides):
    data = {
        "id": unit_id,
        "law_id": "ro.codul_muncii",
        "law_title": "Codul muncii",
        "status": "active",
        "hierarchy_path": ["Codul muncii"],
        "article_number": "41",
        "raw_text": text,
        "legal_domain": "munca",
        "legal_concepts": ["salariu", "contract"],
        "source_url": "https://legislatie.just.ro/test",
        "type": "articol",
    }
    data.update(overrides)
    return data


def ranked(
    unit_id: str,
    *,
    rank: int = 1,
    score: float = 0.8,
    text: str = "salariu contract act aditional",
    why_ranked: list[str] | None = None,
    unit_overrides: dict | None = None,
) -> RankedCandidate:
    return RankedCandidate(
        unit_id=unit_id,
        rank=rank,
        rerank_score=score,
        retrieval_score=score - 0.1,
        unit=unit(unit_id, text, **(unit_overrides or {})),
        score_breakdown=RankerFeatureBreakdown(
            bm25_score=score,
            dense_score=score,
            domain_match=1.0,
            graph_proximity=1.0,
        ),
        why_ranked=why_ranked or ["domain_match:munca"],
        source="raw_retrieval",
    )


def test_no_ranked_candidates_returns_safe_fallback():
    result = EvidencePackCompiler().compile(
        ranked_candidates=[],
        debug=True,
    )

    assert result.evidence_units == []
    assert result.graph_nodes == []
    assert result.graph_edges == []
    assert result.warnings == [EVIDENCE_PACK_NO_RANKED_CANDIDATES]
    assert result.debug["fallback_used"] is True
    assert result.debug["input_ranked_candidate_count"] == 0
    assert result.debug["selected_evidence_count"] == 0


def test_mmr_penalizes_redundant_text_and_diversifies_selection():
    compiler = EvidencePackCompiler(target_evidence_units=2, max_evidence_units=2)
    first = ranked(
        "ro.codul_muncii.art_41",
        rank=1,
        score=0.9,
        text="salariu contract act aditional angajator",
    )
    redundant = ranked(
        "ro.codul_muncii.art_42",
        rank=2,
        score=0.88,
        text="salariu contract act aditional angajator",
    )
    diverse = ranked(
        "ro.codul_muncii.art_260",
        rank=3,
        score=0.7,
        text="amenda contraventie inspectorat sanctiune",
    )

    result = compiler.compile(
        ranked_candidates=[first, redundant, diverse],
        debug=True,
    )

    assert [item.id for item in result.evidence_units] == [
        "ro.codul_muncii.art_41",
        "ro.codul_muncii.art_260",
    ]
    assert result.debug["selected_units"][1]["unit_id"] == "ro.codul_muncii.art_260"


def test_duplicate_unit_ids_are_selected_once():
    result = EvidencePackCompiler(target_evidence_units=3).compile(
        ranked_candidates=[
            ranked("ro.codul_muncii.art_41", rank=1, score=0.7),
            ranked("ro.codul_muncii.art_41", rank=2, score=0.9),
        ],
        debug=True,
    )

    assert len(result.evidence_units) == 1
    assert result.evidence_units[0].id == "ro.codul_muncii.art_41"


def test_support_roles_are_classified_deterministically():
    compiler = EvidencePackCompiler(target_evidence_units=4, max_evidence_units=4)
    result = compiler.compile(
        ranked_candidates=[
            ranked(
                "ro.codul_muncii.exception",
                rank=1,
                score=0.9,
                text="Cu exceptia cazurilor prevazute de lege.",
                why_ranked=["exception_candidate"],
            ),
            ranked(
                "ro.codul_muncii.definition",
                rank=2,
                score=0.8,
                text="In sensul prezentei legi, salariatul se intelege ca persoana fizica.",
                why_ranked=["definition_candidate"],
            ),
            ranked(
                "ro.codul_muncii.sanction",
                rank=3,
                score=0.7,
                text="Constituie contraventie si se sanctioneaza cu amenda.",
                why_ranked=["sanction_candidate"],
            ),
            ranked(
                "ro.codul_muncii.condition",
                rank=4,
                score=0.6,
                text="Modificarea se poate face numai daca partile sunt de acord.",
            ),
        ],
        debug=True,
    )

    roles = {
        evidence.id: evidence.support_role
        for evidence in result.evidence_units
    }
    assert roles["ro.codul_muncii.exception"] == "exception"
    assert roles["ro.codul_muncii.definition"] == "definition"
    assert roles["ro.codul_muncii.sanction"] == "sanction"
    assert roles["ro.codul_muncii.condition"] == "condition"


def test_parent_context_is_included_only_when_parent_exists_in_input():
    compiler = EvidencePackCompiler(target_evidence_units=1, max_evidence_units=2)
    child = ranked(
        "ro.codul_muncii.art_41.alin_1",
        rank=1,
        score=0.9,
        unit_overrides={
            "paragraph_number": "1",
            "parent_id": "ro.codul_muncii.art_41",
            "type": "alineat",
        },
    )
    parent = ranked(
        "ro.codul_muncii.art_41",
        rank=2,
        score=0.2,
        text="Articolul 41 reglementeaza modificarea contractului.",
        unit_overrides={"type": "articol"},
    )

    result = compiler.compile(
        ranked_candidates=[child, parent],
        graph_expansion=GraphExpansionResult(),
        debug=True,
    )

    assert [evidence.id for evidence in result.evidence_units] == [
        "ro.codul_muncii.art_41.alin_1",
        "ro.codul_muncii.art_41",
    ]
    assert result.evidence_units[1].support_role == "context"


def test_parent_context_is_not_included_when_parent_is_missing():
    compiler = EvidencePackCompiler(target_evidence_units=1, max_evidence_units=2)
    child = ranked(
        "ro.codul_muncii.art_41.alin_1",
        rank=1,
        score=0.9,
        unit_overrides={
            "paragraph_number": "1",
            "parent_id": "ro.codul_muncii.art_41",
            "type": "alineat",
        },
    )

    result = compiler.compile(ranked_candidates=[child], debug=True)

    assert [evidence.id for evidence in result.evidence_units] == [
        "ro.codul_muncii.art_41.alin_1",
    ]


def test_compiler_preserves_text_source_scores_and_selection_reasons():
    result = EvidencePackCompiler(target_evidence_units=1).compile(
        ranked_candidates=[
            ranked(
                "ro.codul_muncii.art_41",
                score=0.86,
                why_ranked=["domain_match:munca", "high_bm25_score"],
            )
        ],
        debug=True,
    )

    evidence = result.evidence_units[0]
    assert evidence.id == "ro.codul_muncii.art_41"
    assert evidence.excerpt == "salariu contract act aditional"
    assert evidence.source_url == "https://legislatie.just.ro/test"
    assert evidence.rerank_score == 0.86
    assert evidence.retrieval_score == 0.76
    assert evidence.score_breakdown["bm25_score"] == 0.86
    assert "high_bm25_score" in evidence.why_selected
    assert "selected_by_mmr" in evidence.why_selected
