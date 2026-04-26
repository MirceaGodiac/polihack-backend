import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from apps.api.app.main import app
from apps.api.app.routes import retrieve_raw as retrieve_raw_route
from apps.api.app.services.raw_retriever import (
    EmptyRawRetrievalStore,
    PostgresRawRetrievalStore,
    RawRetriever,
    _build_lexical_query,
    _detect_fallback_intent,
    _expanded_query_terms,
    _fallback_search_terms,
    _labor_contract_modification_governing_rule_match,
    _lexical_ilike_score_for_text,
    _lexical_scoring_strategy,
    _query_terms,
    _weighted_fallback_terms,
)
from apps.api.app.services.query_frame import QueryFrameBuilder
from apps.api.app.services.query_understanding import QueryUnderstanding
from apps.api.app.services.retrieval_scoring import (
    ScoreBreakdown,
    normalize_rrf_scores,
    weighted_retrieval_score,
)


def test_weighted_retrieval_score_uses_rrf_with_dense():
    base = ScoreBreakdown(
        bm25=0.5,
        dense=0.5,
        domain_match=1.0,
        metadata_validity=1.0,
        exact_citation_boost=0.0,
    )
    with_rrf = ScoreBreakdown(
        bm25=0.5,
        dense=0.5,
        rrf=1.0,
        domain_match=1.0,
        metadata_validity=1.0,
        exact_citation_boost=0.0,
    )

    assert weighted_retrieval_score(with_rrf, dense_available=True) == pytest.approx(
        weighted_retrieval_score(base, dense_available=True) + 0.30
    )
    assert 0.0 <= weighted_retrieval_score(with_rrf, dense_available=True) <= 1.0


def test_weighted_retrieval_score_uses_rrf_without_dense():
    base = ScoreBreakdown(
        bm25=0.5,
        domain_match=1.0,
        metadata_validity=1.0,
        exact_citation_boost=0.0,
    )
    with_rrf = ScoreBreakdown(
        bm25=0.5,
        rrf=1.0,
        domain_match=1.0,
        metadata_validity=1.0,
        exact_citation_boost=0.0,
    )

    assert weighted_retrieval_score(with_rrf, dense_available=False) == pytest.approx(
        weighted_retrieval_score(base, dense_available=False) + 0.40
    )
    assert 0.0 <= weighted_retrieval_score(with_rrf, dense_available=False) <= 1.0


def test_normalize_rrf_scores_scales_raw_values_to_unit_interval():
    normalized = normalize_rrf_scores({"a": 0.032, "b": 0.016})

    assert normalized == {"a": 1.0, "b": pytest.approx(0.5)}


def test_endpoint_schema_valid_with_dependency_override():
    async def override_retriever():
        return RawRetriever(FakeStore())

    app.dependency_overrides[retrieve_raw_route.get_raw_retriever] = override_retriever
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/retrieve/raw",
                json={
                    "question": "Poate angajatorul sa-mi scada salariul fara act aditional?",
                    "filters": {"legal_domain": "munca", "status": "active"},
                    "exact_citations": [
                        {"law_id": "ro.codul_muncii", "article_number": "41"}
                    ],
                    "top_k": 10,
                    "debug": True,
                },
            )
    finally:
        app.dependency_overrides.clear()

    payload = response.json()
    assert response.status_code == 200
    assert payload["candidates"]
    by_id = {candidate["unit_id"]: candidate for candidate in payload["candidates"]}
    assert "ro.codul_muncii.art_41" in by_id
    assert by_id["ro.codul_muncii.art_41"]["unit"]["raw_text"]
    assert payload["debug"]["candidate_count"] >= 1
    assert "query_embedding" not in payload


def test_endpoint_without_database_url_returns_warning(monkeypatch):
    import apps.api.app.db.session as db_module

    monkeypatch.setattr(db_module.settings, "database_url", None)
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_sessionmaker", None)

    with TestClient(app) as client:
        response = client.post(
            "/api/retrieve/raw",
            json={
                "question": "Poate angajatorul sa-mi scada salariul fara act aditional?",
                "top_k": 5,
                "debug": True,
            },
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["candidates"] == []
    assert any("database_unavailable" in warning for warning in payload["warnings"])


@pytest.mark.anyio
async def test_exact_citation_art_41_returns_matching_unit():
    response = await RawRetriever(FakeStore()).retrieve(
        _request(
            exact_citations=[{"law_id": "ro.codul_muncii", "article_number": "41"}],
            debug=True,
        )
    )

    assert response.candidates
    candidate = next(
        candidate
        for candidate in response.candidates
        if candidate.unit_id == "ro.codul_muncii.art_41"
    )
    assert candidate.unit_id == "ro.codul_muncii.art_41"
    assert candidate.unit["article_number"] == "41"
    assert candidate.score_breakdown["exact_citation_boost"] == 1.0
    assert "exact citation" in candidate.why_retrieved


@pytest.mark.anyio
async def test_exact_citation_art_41_returns_descendant_units_from_fixture_db():
    response = await RawRetriever(FakeStore()).retrieve(
        _request(
            exact_citations=[{"law_id": "ro.codul_muncii", "article_number": "41"}],
            top_k=10,
            debug=True,
        )
    )

    candidate_ids = {candidate.unit_id for candidate in response.candidates}
    assert {
        "ro.codul_muncii.art_41",
        "ro.codul_muncii.art_41.alin_1",
        "ro.codul_muncii.art_41.alin_2",
        "ro.codul_muncii.art_41.alin_3.lit_e",
    }.issubset(candidate_ids)
    assert all(
        candidate.unit["article_number"] == "41"
        for candidate in response.candidates
        if candidate.unit_id.startswith("ro.codul_muncii.art_41")
    )
    assert not any(
        warning.startswith("exact_citation_lookup_failed")
        for warning in response.warnings
    )


@pytest.mark.anyio
async def test_fts_query_returns_codul_muncii_candidates():
    response = await RawRetriever(FakeStore()).retrieve(
        _request(question="salariu act aditional", exact_citations=[])
    )

    assert response.candidates
    assert response.candidates[0].unit["law_id"] == "ro.codul_muncii"
    assert response.candidates[0].score_breakdown["bm25"] > 0.0
    assert "lexical match" in response.candidates[0].why_retrieved


def test_endpoint_natural_labor_query_without_exact_citations_uses_lexical_fallback():
    async def override_retriever():
        return RawRetriever(FallbackProbeStore())

    app.dependency_overrides[retrieve_raw_route.get_raw_retriever] = override_retriever
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/retrieve/raw",
                json={
                    "question": "Poate angajatorul să-mi scadă salariul fără act adițional?",
                    "filters": {"legal_domain": "munca", "status": "active"},
                    "top_k": 10,
                    "debug": True,
                },
            )
    finally:
        app.dependency_overrides.clear()

    payload = response.json()
    assert response.status_code == 200
    candidate_ids = {candidate["unit_id"] for candidate in payload["candidates"]}
    assert candidate_ids.intersection(
        {
            "ro.codul_muncii.art_41.alin_1",
            "ro.codul_muncii.art_41.alin_2",
            "ro.codul_muncii.art_41.alin_3",
            "ro.codul_muncii.art_41.alin_3.lit_e",
        }
    )
    assert payload["debug"]["fts_fallback_used"] is True
    assert "salariul" in payload["debug"]["lexical_terms"]
    assert "modificare contract" in payload["debug"]["expanded_terms"]
    assert "acordul partilor" in payload["debug"]["expanded_terms"]


@pytest.mark.anyio
async def test_fallback_lexical_only_runs_when_strict_fts_returns_no_candidates():
    strict_store = FallbackProbeStore(strict_rows=[{**_UNITS[0], "bm25_score": 1.0}])
    strict_rows = await strict_store.lexical_search(
        "salariu act aditional",
        filters={"legal_domain": "munca", "status": "active"},
        limit=10,
    )

    assert strict_rows[0]["id"] == "ro.codul_muncii.art_41"
    assert strict_store.strict_called is True
    assert strict_store.fallback_called is False
    assert strict_store.last_lexical_debug["fts_fallback_used"] is False

    fallback_store = FallbackProbeStore()
    fallback_rows = await fallback_store.lexical_search(
        "Poate angajatorul să-mi scadă salariul fără act adițional?",
        filters={"legal_domain": "munca", "status": "active"},
        limit=10,
    )

    assert fallback_rows
    assert fallback_store.strict_called is True
    assert fallback_store.fallback_called is True
    assert fallback_store.last_lexical_debug["fts_fallback_used"] is True


@pytest.mark.anyio
async def test_debug_includes_lexical_terms_expanded_terms_and_fallback_flag():
    response = await RawRetriever(FallbackProbeStore()).retrieve(
        _request(
            question="Poate angajatorul să-mi scadă salariul fără act adițional?",
            exact_citations=[],
            debug=True,
        )
    )

    assert response.debug is not None
    assert response.debug["fts_fallback_used"] is True
    assert response.debug["fallback_intent"] == "labor_contract_modification"
    assert response.debug["fallback_intent_source"] == "legacy"
    assert response.debug["scoring_strategy"] == "legacy_intent_grouped_lexical_fallback"
    assert "salariul" in response.debug["lexical_terms"]
    assert "modificare contract" in response.debug["expanded_terms"]
    assert "contract individual de munca" in response.debug["expanded_terms"]


@pytest.mark.anyio
async def test_registry_query_frame_drives_demo_expanded_terms():
    question = "Poate angajatorul sa-mi scada salariul fara act aditional?"
    query_frame = _query_frame_for(question)
    response = await RawRetriever(FallbackProbeStore()).retrieve(
        _request(
            question=question,
            exact_citations=[],
            query_frame=query_frame,
            debug=True,
        )
    )

    assert response.debug["fallback_intent"] == "labor_contract_modification"
    assert response.debug["fallback_intent_source"] == "query_frame_registry"
    assert response.debug["scoring_strategy"] == "registry_intent_grouped_lexical_fallback"
    assert "modificare contract" in response.debug["expanded_terms"]
    assert "contract individual de munca" in response.debug["expanded_terms"]
    assert "acordul partilor" in response.debug["expanded_terms"]
    assert "salariu" in response.debug["expanded_terms"]
    assert "salariul" in response.debug["expanded_terms"]
    assert "act aditional" in response.debug["expanded_terms"]
    assert response.debug["query_frame_intents"] == ["labor_contract_modification"]
    assert response.debug["query_frame_confidence"] >= 0.70
    assert response.debug["registry_expanded_terms"]


def test_intent_governing_rule_helper_matches_agreement_rule_text():
    match = _labor_contract_modification_governing_rule_match(
        "Contractul individual de munca poate fi modificat numai prin acordul partilor."
    )

    assert match.matched is True
    assert match.role == "agreement_rule"
    assert match.score == 1.0


def test_intent_governing_rule_helper_matches_salary_scope_text():
    match = _labor_contract_modification_governing_rule_match(
        "Modificarea contractului individual de munca poate privi salariul."
    )

    assert match.matched is True
    assert match.role == "salary_scope"
    assert match.score >= 0.90


def test_intent_governing_rule_helper_rejects_formare_profesionala_distractor():
    match = _labor_contract_modification_governing_rule_match(
        "Modalitatea concreta de formare profesionala, drepturile si obligatiile "
        "partilor, durata formarii profesionale, precum si orice alte aspecte "
        "legate de formarea profesionala fac obiectul unor acte aditionale la "
        "contractele individuale de munca."
    )

    assert match.matched is False
    assert match.score == 0.0


@pytest.mark.anyio
async def test_intent_governing_rule_lookup_surfaces_live_like_art_41_units():
    question = "Poate angajatorul sa-mi scada salariul fara act aditional?"
    query_frame = _query_frame_for(question)
    response = await RawRetriever(
        FallbackProbeStore(units=_live_like_units_for_raw_retriever())
    ).retrieve(
        _request(
            question=question,
            filters={"legal_domain": "munca", "status": "active"},
            exact_citations=[],
            query_frame=query_frame,
            top_k=5,
            debug=True,
        )
    )

    top_ids = [candidate.unit_id for candidate in response.candidates[:3]]
    assert "ro.codul_muncii.art_41.alin_1" in top_ids
    assert "ro.codul_muncii.art_41.alin_3" in top_ids
    assert "ro.codul_muncii.art_196.alin_2" not in top_ids
    assert response.debug["intent_governing_rule_candidate_count"] >= 2
    assert response.debug["rankings"]["intent_governing_rule_lookup"][:2] == [
        "ro.codul_muncii.art_41.alin_1",
        "ro.codul_muncii.art_41.alin_3",
    ]
    by_id = {candidate.unit_id: candidate for candidate in response.candidates}
    assert by_id["ro.codul_muncii.art_41.alin_1"].score_breakdown[
        "intent_governing_rule"
    ] == 1.0
    assert "intent_governing_rule_lookup:labor_contract_modification" in by_id[
        "ro.codul_muncii.art_41.alin_1"
    ].why_retrieved


@pytest.mark.anyio
async def test_intent_governing_rule_lookup_includes_salary_target_parent_context():
    question = "Poate angajatorul sa-mi scada salariul fara act aditional?"
    query_frame = _query_frame_for(question)
    units = _live_like_units_for_raw_retriever()
    for unit in units:
        if unit["id"] == "ro.codul_muncii.art_41.alin_3":
            unit["raw_text"] = (
                "Modificarea contractului individual de munca se refera la "
                "oricare dintre urmatoarele elemente:"
            )
            unit["normalized_text"] = (
                "modificarea contractului individual de munca se refera "
                "urmatoarele elemente"
            )

    response = await RawRetriever(FallbackProbeStore(units=units)).retrieve(
        _request(
            question=question,
            filters={"legal_domain": "munca", "status": "active"},
            exact_citations=[],
            query_frame=query_frame,
            top_k=6,
            debug=True,
        )
    )

    ranking = response.debug["rankings"]["intent_governing_rule_lookup"]
    child_index = ranking.index("ro.codul_muncii.art_41.alin_3.lit_e")
    assert ranking[child_index + 1] == "ro.codul_muncii.art_41.alin_3"
    assert response.debug["intent_governing_rule_parent_ids"] == [
        "ro.codul_muncii.art_41.alin_3"
    ]
    by_id = {candidate.unit_id: candidate for candidate in response.candidates}
    parent = by_id["ro.codul_muncii.art_41.alin_3"]
    assert parent.score_breakdown["intent_governing_rule_parent"] == 1.0
    assert (
        "intent_governing_rule_parent_context:labor_contract_modification"
        in parent.why_retrieved
    )


@pytest.mark.anyio
async def test_registry_query_frame_generalizes_contravention_challenge_fallback():
    question = "Cum contest o amenda contraventionala?"
    query_frame = _query_frame_for(question)
    response = await RawRetriever(FallbackProbeStore()).retrieve(
        _request(
            question=question,
            filters={"legal_domain": "contraventional", "status": "active"},
            exact_citations=[],
            query_frame=query_frame,
            debug=True,
        )
    )

    expanded_terms = response.debug["expanded_terms"]
    assert query_frame["intents"][0] == "contravention_challenge"
    assert response.debug["fallback_intent"] == "contravention_challenge"
    assert response.debug["fallback_intent_source"] == "query_frame_registry"
    assert response.debug["scoring_strategy"] == "registry_intent_grouped_lexical_fallback"
    assert {"plangere contraventionala", "proces verbal", "amenda", "contraventie"}.issubset(
        set(expanded_terms)
    )


def test_registry_query_frame_expands_civil_prescription_terms():
    question = "In cat timp se prescrie dreptul la actiune pentru o datorie civila?"
    query_frame = _query_frame_for(question)

    lexical_query = _build_lexical_query(
        question,
        {"legal_domain": "civil", "status": "active"},
        query_frame=query_frame,
    )

    assert lexical_query.fallback_intent is not None
    assert lexical_query.fallback_intent.name == "civil_prescription"
    assert lexical_query.fallback_intent_source == "query_frame_registry"
    assert "prescriptie" in lexical_query.expanded_terms
    assert "dreptul la actiune" in lexical_query.expanded_terms
    assert "termen" in lexical_query.expanded_terms
    assert "datorie civila" in lexical_query.expanded_terms


def test_unknown_query_does_not_fabricate_confident_fallback_intent():
    question = "Ce fac exact aici?"
    query_frame = _query_frame_for(question)

    lexical_query = _build_lexical_query(
        question,
        {},
        query_frame=query_frame,
    )

    assert lexical_query.fallback_intent is None
    assert _lexical_scoring_strategy(lexical_query, fallback_used=True) == "weighted_lexical_fallback"


def test_registry_expanded_terms_generated_for_confident_eval_frames():
    fixture_path = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "eval"
        / "query_frame_cases.json"
    )

    cases = json.loads(fixture_path.read_text(encoding="utf-8"))
    for case in cases:
        if case["id"] == "unknown_short":
            continue
        query_frame = _query_frame_for(case["question"])
        if query_frame["confidence"] < 0.70:
            continue
        lexical_query = _build_lexical_query(
            case["question"],
            {"legal_domain": query_frame["domain"]},
            query_frame=query_frame,
        )
        assert lexical_query.fallback_intent is not None, case["id"]
        assert lexical_query.registry_expanded_terms, case["id"]


def test_stopwords_are_removed_from_lexical_terms():
    terms = _query_terms("Poate angajatorul să-mi scadă salariul fără act adițional?")

    assert "poate" not in terms
    assert "sa" not in terms
    assert "mi" not in terms
    assert "fara" not in terms
    assert {"angajatorul", "scada", "salariul", "act", "aditional"}.issubset(terms)


@pytest.mark.parametrize("text", ["act adi\u021bional", "act adi\u0163ional"])
def test_query_terms_transliterates_romanian_diacritics(text):
    terms = _query_terms(text)

    assert "act" in terms
    assert "aditional" in terms
    assert "adi" not in terms
    assert "ional" not in terms


def test_query_terms_repairs_windows_replacement_mark_in_act_aditional():
    terms = _query_terms("act adi?ional")

    assert "act" in terms
    assert "aditional" in terms
    assert "adi" not in terms
    assert "ional" not in terms


def test_labor_query_expansion_is_conservative_and_domain_aware():
    expanded_terms = _expanded_query_terms(
        "Poate angajatorul să-mi scadă salariul fără act adițional?",
        {"legal_domain": "munca"},
    )

    assert "salariu" in expanded_terms
    assert "salariul" in expanded_terms
    assert "salarizare" in expanded_terms
    assert "modificare contract" in expanded_terms
    assert "acordul partilor" in expanded_terms
    assert "contract individual de munca" in expanded_terms
    assert "contractul individual de munca" in expanded_terms
    assert "modificarea contractului individual de munca" in expanded_terms
    assert "contractului individual de munca" in expanded_terms
    assert "numai prin acordul partilor" in expanded_terms
    assert "poate fi modificat" in expanded_terms
    assert "modificat numai prin acordul partilor" in expanded_terms


def test_labor_query_expansion_repairs_windows_replacement_mark():
    expanded_terms = _expanded_query_terms(
        "Poate angajatorul sa-mi scada salariul fara act adi?ional?",
        {"legal_domain": "munca"},
    )

    assert "modificare contract" in expanded_terms
    assert "contract individual de munca" in expanded_terms
    assert "contractul individual de munca" in expanded_terms
    assert "acordul partilor" in expanded_terms


@pytest.mark.anyio
async def test_fallback_weighted_phrase_scoring_ranks_relevant_text_above_generic_contract():
    base = {
        **_UNITS[0],
        "law_id": "ro.codul_muncii",
        "law_title": "Codul muncii",
        "status": "active",
        "legal_domain": "munca",
        "source_url": "https://legislatie.just.ro/test",
    }
    units = [
        {
            **base,
            "id": "fixture.generic_contract",
            "canonical_id": "fixture.generic_contract",
            "article_number": "10",
            "raw_text": "Contractul produce efecte intre parti potrivit actului semnat.",
            "normalized_text": "contract act acord parti",
        },
        {
            **base,
            "id": "fixture.relevant_contract_change",
            "canonical_id": "fixture.relevant_contract_change",
            "article_number": "41",
            "raw_text": "Modificarea contractului individual de munca poate privi salariul.",
            "normalized_text": "modificarea contractului individual de munca salariul",
        },
    ]
    response = await RawRetriever(FallbackProbeStore(units=units)).retrieve(
        _request(
            question="Poate angajatorul sa-mi scada salariul fara act aditional?",
            exact_citations=[],
            filters={"legal_domain": "munca", "status": "active"},
            top_k=2,
            debug=True,
        )
    )

    assert [candidate.unit_id for candidate in response.candidates] == [
        "fixture.relevant_contract_change",
        "fixture.generic_contract",
    ]
    assert response.candidates[0].score_breakdown["bm25"] > response.candidates[1].score_breakdown["bm25"]


@pytest.mark.anyio
async def test_fallback_central_contract_match_beats_many_salary_employer_matches():
    base = {
        **_UNITS[0],
        "law_id": "ro.codul_muncii",
        "law_title": "Codul muncii",
        "status": "active",
        "legal_domain": "munca",
        "source_url": "https://legislatie.just.ro/test",
    }
    units = [
        {
            **base,
            "id": "fixture.salary_employer_contract",
            "canonical_id": "fixture.salary_employer_contract",
            "article_number": "160",
            "raw_text": "Angajatorul stabileste salariul in contract si poate discuta salarizarea.",
            "normalized_text": "angajator angajatorul salariu salariul contract salarizare",
        },
        {
            **base,
            "id": "fixture.central_contract_change",
            "canonical_id": "fixture.central_contract_change",
            "article_number": "41",
            "raw_text": "Contractul individual de munca poate fi modificat numai prin acordul partilor.",
            "normalized_text": "contract individual munca poate fi modificat numai prin acord parti",
        },
    ]
    response = await RawRetriever(FallbackProbeStore(units=units)).retrieve(
        _request(
            question="Poate angajatorul sa-mi scada salariul fara act aditional?",
            exact_citations=[],
            filters={"legal_domain": "munca", "status": "active"},
            top_k=2,
            debug=True,
        )
    )

    assert [candidate.unit_id for candidate in response.candidates] == [
        "fixture.central_contract_change",
        "fixture.salary_employer_contract",
    ]
    assert response.candidates[0].score_breakdown["bm25"] > response.candidates[1].score_breakdown["bm25"]


@pytest.mark.anyio
async def test_fallback_lexical_respects_filters():
    units = [
        *_UNITS,
        {
            **_UNITS[2],
            "id": "ro.codul_muncii.art_41.repealed_fixture",
            "canonical_id": "ro.codul_muncii.art_41.repealed_fixture",
            "status": "repealed",
        },
        {
            **_UNITS[2],
            "id": "ro.codul_civil.art_1.fixture",
            "canonical_id": "ro.codul_civil.art_1.fixture",
            "law_id": "ro.codul_civil",
            "law_title": "Codul civil",
            "legal_domain": "civil",
        },
    ]
    response = await RawRetriever(FallbackProbeStore(units=units)).retrieve(
        _request(
            question="Poate angajatorul să-mi scadă salariul fără act adițional?",
            exact_citations=[],
            filters={"legal_domain": "munca", "status": "active"},
            debug=True,
        )
    )

    candidate_ids = {candidate.unit_id for candidate in response.candidates}
    assert "ro.codul_muncii.art_41.repealed_fixture" not in candidate_ids
    assert "ro.codul_civil.art_1.fixture" not in candidate_ids
    assert response.debug["fts_fallback_used"] is True


@pytest.mark.anyio
async def test_domain_filter_sets_domain_match():
    response = await RawRetriever(FakeStore()).retrieve(
        _request(filters={"legal_domain": "munca", "status": "active"})
    )

    assert response.candidates
    assert all(
        candidate.score_breakdown["domain_match"] == 1.0
        for candidate in response.candidates
    )


@pytest.mark.anyio
async def test_missing_or_empty_db_returns_warning_without_crash():
    response = await RawRetriever(
        EmptyRawRetrievalStore(),
        initial_warnings=["database_unavailable"],
    ).retrieve(_request())

    assert response.candidates == []
    assert "database_unavailable" in response.warnings


@pytest.mark.anyio
async def test_dense_retrieval_skipped_when_query_embedding_missing():
    response = await RawRetriever(FakeStore()).retrieve(_request())

    assert "dense_retrieval_skipped_no_query_embedding" in response.warnings
    assert all(candidate.score_breakdown["dense"] == 0.0 for candidate in response.candidates)


@pytest.mark.anyio
async def test_dense_retrieval_uses_store_when_query_embedding_is_present():
    store = FakeStore()
    response = await RawRetriever(store).retrieve(
        _request(query_embedding=[0.1, 0.2, 0.3], exact_citations=[])
    )

    assert store.dense_called is True
    assert response.candidates
    assert response.candidates[0].score_breakdown["dense"] > 0.0
    assert "dense similarity" in " ".join(candidate.why_retrieved or "" for candidate in response.candidates)
    assert "dense_retrieval_skipped_no_query_embedding" not in response.warnings


@pytest.mark.anyio
async def test_postgres_dense_search_uses_asyncpg_safe_vector_cast():
    session = CaptureSqlSession()
    store = PostgresRawRetrievalStore(session)
    store._unit_columns = set(_UNITS[0].keys())

    await store.dense_search(
        [0.1, 0.2],
        filters={"legal_domain": "munca", "status": "active"},
        limit=5,
    )

    sql = session.statements[-1]
    assert ":embedding::vector" not in sql
    assert "CAST(:embedding AS vector)" in sql
    assert "e.embedding <=> CAST(:embedding AS vector)" in sql
    assert "1.0 - (e.embedding <=> CAST(:embedding AS vector))" in sql
    assert session.params[-1]["embedding"] == "[0.1,0.2]"


@pytest.mark.anyio
async def test_rrf_boosts_candidate_present_in_bm25_dense_and_exact():
    response = await RawRetriever(MultiSignalStore()).retrieve(
        _request(
            query_embedding=[0.1, 0.2, 0.3],
            top_k=2,
            debug=True,
        )
    )

    assert [candidate.unit_id for candidate in response.candidates] == [
        "fixture.multi_signal",
        "fixture.lexical_only",
    ]
    by_id = {candidate.unit_id: candidate for candidate in response.candidates}
    assert by_id["fixture.multi_signal"].score_breakdown["rrf"] == 1.0
    assert (
        by_id["fixture.multi_signal"].score_breakdown["rrf"]
        > by_id["fixture.lexical_only"].score_breakdown["rrf"]
    )
    assert (
        by_id["fixture.multi_signal"].retrieval_score
        > by_id["fixture.lexical_only"].retrieval_score
    )
    assert response.debug["rrf_scores"]["fixture.multi_signal"] == 1.0
    assert response.debug["rrf_scores_raw"]["fixture.multi_signal"] < 1.0


@pytest.mark.anyio
async def test_rrf_fallback_without_dense_rewards_multiple_methods_over_lexical_only():
    response = await RawRetriever(MultiSignalStore()).retrieve(
        _request(
            query_embedding=None,
            top_k=2,
            debug=True,
        )
    )

    assert response.debug["dense_available"] is False
    assert [candidate.unit_id for candidate in response.candidates] == [
        "fixture.multi_signal",
        "fixture.lexical_only",
    ]
    by_id = {candidate.unit_id: candidate for candidate in response.candidates}
    assert by_id["fixture.lexical_only"].score_breakdown["bm25"] == 1.0
    assert by_id["fixture.multi_signal"].score_breakdown["bm25"] < 1.0
    assert (
        by_id["fixture.multi_signal"].retrieval_score
        > by_id["fixture.lexical_only"].retrieval_score
    )


@pytest.mark.anyio
async def test_score_breakdown_has_required_fields_and_top_k_is_respected():
    response = await RawRetriever(FakeStore()).retrieve(
        _request(top_k=1, exact_citations=[])
    )

    assert len(response.candidates) == 1
    breakdown = response.candidates[0].score_breakdown
    assert {
        "bm25",
        "dense",
        "domain_match",
        "metadata_validity",
        "exact_citation_boost",
        "rrf",
        "intent_phrase_match",
    }.issubset(breakdown)
    assert 0.0 <= breakdown["rrf"] <= 1.0


@pytest.mark.anyio
async def test_response_does_not_include_vectors():
    response = await RawRetriever(FakeStore()).retrieve(
        _request(query_embedding=[0.1, 0.2, 0.3])
    )

    payload = response.model_dump(mode="json")
    assert "embedding" not in str(payload)
    assert "0.1, 0.2, 0.3" not in str(payload)


@pytest.mark.anyio
async def test_debug_store_warning_includes_sanitized_exception_details():
    response = await RawRetriever(FailingExactStore()).retrieve(_request(debug=True))

    warning = next(
        warning
        for warning in response.warnings
        if warning.startswith("exact_citation_lookup_failed:")
    )
    assert warning.startswith("exact_citation_lookup_failed:RuntimeError:")
    assert "super-secret" not in warning
    assert "DATABASE_URL" not in warning
    assert "raw_text" not in warning
    assert "[0.1, 0.2, 0.3]" not in warning


@pytest.mark.anyio
async def test_non_debug_store_warning_keeps_generic_code():
    response = await RawRetriever(FailingExactStore()).retrieve(_request(debug=False))

    assert "exact_citation_lookup_failed" in response.warnings
    assert not any(
        warning.startswith("exact_citation_lookup_failed:")
        for warning in response.warnings
    )


def _request(**overrides):
    from apps.api.app.schemas import RawRetrievalRequest

    payload = {
        "question": "Poate angajatorul sa-mi scada salariul fara act aditional?",
        "filters": {"legal_domain": "munca", "status": "active"},
        "exact_citations": [
            {"law_id": "ro.codul_muncii", "article_number": "41"}
        ],
        "top_k": 10,
        "debug": False,
    }
    payload.update(overrides)
    return RawRetrievalRequest.model_validate(payload)


def _query_frame_for(question: str) -> dict:
    from apps.api.app.schemas import QueryRequest

    plan = QueryUnderstanding().build_plan(
        QueryRequest(
            question=question,
            jurisdiction="RO",
            date="current",
            mode="strict_citations",
            debug=True,
        )
    )
    return QueryFrameBuilder().build(question=question, plan=plan).model_dump(
        mode="json"
    )


def _live_like_units_for_raw_retriever():
    from tests.helpers.live_like_demo import LIVE_LIKE_UNITS

    base = {
        "canonical_id": None,
        "source_id": "fixture",
        "law_id": "ro.codul_muncii",
        "law_title": "Codul muncii",
        "act_type": "code",
        "act_number": None,
        "publication_date": None,
        "effective_date": None,
        "version_start": None,
        "version_end": None,
        "status": "active",
        "legal_domain": "munca",
        "legal_concepts": ["contract", "salariu"],
        "source_url": "https://legislatie.just.ro/test",
        "parser_warnings": [],
        "created_at": None,
        "updated_at": None,
        "type": "alineat",
    }
    units = []
    for unit in LIVE_LIKE_UNITS:
        units.append(
            {
                **base,
                **unit,
                "canonical_id": unit["id"],
                "parent_id": unit.get("parent_id"),
                "point_number": None,
                "normalized_text": unit["raw_text"].casefold(),
                "hierarchy_path": [
                    "Codul muncii",
                    f"Art. {unit['article_number']}",
                    f"Alin. ({unit['paragraph_number']})",
                ],
            }
        )
    return units


class CaptureSqlSession:
    def __init__(self):
        self.statements = []
        self.params = []

    def begin_nested(self):
        return _NoopNestedTransaction()

    async def execute(self, statement, params=None):
        self.statements.append(str(statement))
        self.params.append(dict(params or {}))
        return _MappingRows([])


class _NoopNestedTransaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None


class _MappingRows:
    def __init__(self, rows):
        self.rows = rows

    def mappings(self):
        return self

    def all(self):
        return self.rows


class FakeStore:
    def __init__(self):
        self.dense_called = False

    async def exact_citation_lookup(self, citations, *, filters, limit):
        return [
            dict(unit)
            for citation in citations
            for unit in _filtered_units(filters)
            if unit["law_id"] == citation.get("law_id")
            and unit["article_number"] == citation.get("article_number")
        ][:limit]

    async def lexical_search(self, question, *, filters, limit):
        terms = [term for term in question.lower().split() if len(term) >= 3]
        rows = []
        for unit in _filtered_units(filters):
            haystack = f"{unit['raw_text']} {unit['normalized_text']}".lower()
            matches = sum(1 for term in terms if term in haystack)
            if matches:
                rows.append({**unit, "bm25_score": matches / max(1, len(terms))})
        return sorted(rows, key=lambda row: row["bm25_score"], reverse=True)[:limit]

    async def dense_search(self, query_embedding, *, filters, limit):
        self.dense_called = True
        return [
            {**unit, "dense_score": 0.91 - (index * 0.1)}
            for index, unit in enumerate(_filtered_units(filters))
        ][:limit]


class MultiSignalStore:
    def __init__(self):
        self.units = _multi_signal_units()
        self.dense_called = False

    async def exact_citation_lookup(self, citations, *, filters, limit):
        rows = []
        for citation in citations:
            for unit in _filtered_units_from(self.units, filters):
                if (
                    unit["law_id"] == citation.get("law_id")
                    and unit["article_number"] == citation.get("article_number")
                ):
                    rows.append(dict(unit))
        return rows[:limit]

    async def lexical_search(self, question, *, filters, limit):
        rows = []
        for unit in _filtered_units_from(self.units, filters):
            if unit["id"] == "fixture.lexical_only":
                rows.append({**unit, "bm25_score": 1.0})
            elif unit["id"] == "fixture.multi_signal":
                rows.append({**unit, "bm25_score": 0.8})
        return rows[:limit]

    async def dense_search(self, query_embedding, *, filters, limit):
        self.dense_called = True
        return [
            {**unit, "dense_score": 0.9}
            for unit in _filtered_units_from(self.units, filters)
            if unit["id"] == "fixture.multi_signal"
        ][:limit]


class FallbackProbeStore(PostgresRawRetrievalStore):
    def __init__(self, strict_rows=None, units=None):
        self.strict_rows = strict_rows or []
        self.units = units or _UNITS
        self.strict_called = False
        self.fallback_called = False
        self.fallback_terms = []
        self.last_lexical_debug = {}
        self._unit_columns = set(_UNITS[0].keys())

    async def _get_unit_columns(self):
        return self._unit_columns

    async def _strict_fts_search(
        self,
        *,
        select_columns,
        search_document,
        clauses,
        params,
    ):
        self.strict_called = True
        return [dict(row) for row in self.strict_rows]

    async def _lexical_ilike_fallback(
        self,
        question,
        *,
        filters,
        limit,
        available_columns=None,
        terms=None,
        query_frame=None,
        lexical_query=None,
    ):
        self.fallback_called = True
        fallback_intent = (
            lexical_query.fallback_intent
            if lexical_query is not None
            else _detect_fallback_intent(question, filters, query_frame=query_frame)
        )
        self.fallback_terms = _fallback_search_terms(terms or [], fallback_intent)
        weighted_terms = _weighted_fallback_terms(self.fallback_terms)
        rows = []
        for unit in _filtered_units_from(self.units, filters):
            score = _lexical_ilike_score_for_text(
                f"{unit['raw_text']} {unit['normalized_text']}",
                weighted_terms,
                intent=fallback_intent,
            )
            if score > 0.0:
                rows.append(
                    {
                        **unit,
                        "bm25_score": score,
                    }
                )
        return sorted(rows, key=lambda row: row["bm25_score"], reverse=True)[:limit]


class FailingExactStore(FakeStore):
    async def exact_citation_lookup(self, citations, *, filters, limit):
        raise RuntimeError(
            "DATABASE_URL=postgresql://lexai:super-secret@localhost/lexai "
            "raw_text='SECRET_RAW_TEXT' embedding=[0.1, 0.2, 0.3]"
        )


def _filtered_units(filters):
    return _filtered_units_from(_UNITS, filters)


def _filtered_units_from(units, filters):
    rows = []
    for unit in units:
        if filters.get("legal_domain") and unit["legal_domain"] != filters["legal_domain"]:
            continue
        if filters.get("status") and unit["status"] != filters["status"]:
            continue
        rows.append(unit)
    return rows


def _multi_signal_units():
    base = {
        **_UNITS[0],
        "source_id": "fixture",
        "law_id": "ro.codul_muncii",
        "law_title": "Codul muncii",
        "status": "active",
        "legal_domain": "munca",
        "source_url": "https://legislatie.just.ro/test",
        "parser_warnings": [],
        "created_at": None,
        "updated_at": None,
    }
    return [
        {
            **base,
            "id": "fixture.lexical_only",
            "canonical_id": "fixture.lexical_only",
            "article_number": "160",
            "paragraph_number": None,
            "letter_number": None,
            "point_number": None,
            "raw_text": "Salariul si actul aditional sunt mentionate in contractul de munca.",
            "normalized_text": "salariu act aditional contract munca",
            "parent_id": None,
        },
        {
            **base,
            "id": "fixture.multi_signal",
            "canonical_id": "fixture.multi_signal",
            "article_number": "41",
            "paragraph_number": "1",
            "letter_number": None,
            "point_number": None,
            "raw_text": "Contractul individual de munca poate fi modificat numai prin acordul partilor.",
            "normalized_text": "contract individual munca modificat acord parti act aditional",
            "parent_id": None,
        },
    ]


_UNITS = [
    {
        "id": "ro.codul_muncii.art_41",
        "canonical_id": "ro.codul_muncii.art_41",
        "source_id": "fixture",
        "law_id": "ro.codul_muncii",
        "law_title": "Codul muncii",
        "act_type": "code",
        "act_number": None,
        "publication_date": None,
        "effective_date": None,
        "version_start": None,
        "version_end": None,
        "status": "active",
        "hierarchy_path": ["Codul muncii", "Art. 41"],
        "article_number": "41",
        "paragraph_number": None,
        "letter_number": None,
        "point_number": None,
        "raw_text": "Contractul individual de munca poate fi modificat numai prin acordul partilor, prin act aditional.",
        "normalized_text": "salariu contract munca act aditional acord parti",
        "legal_domain": "munca",
        "legal_concepts": ["contract", "salariu"],
        "source_url": "https://legislatie.just.ro/test",
        "parent_id": None,
        "parser_warnings": [],
        "created_at": None,
        "updated_at": None,
    },
    {
        "id": "ro.codul_muncii.art_160",
        "canonical_id": "ro.codul_muncii.art_160",
        "source_id": "fixture",
        "law_id": "ro.codul_muncii",
        "law_title": "Codul muncii",
        "act_type": "code",
        "act_number": None,
        "publication_date": None,
        "effective_date": None,
        "version_start": None,
        "version_end": None,
        "status": "active",
        "hierarchy_path": ["Codul muncii", "Art. 160"],
        "article_number": "160",
        "paragraph_number": None,
        "letter_number": None,
        "point_number": None,
        "raw_text": "Salariul cuprinde salariul de baza si alte adaosuri.",
        "normalized_text": "salariu baza adaosuri",
        "legal_domain": "munca",
        "legal_concepts": ["salariu"],
        "source_url": "https://legislatie.just.ro/test",
        "parent_id": None,
        "parser_warnings": [],
        "created_at": None,
        "updated_at": None,
    },
    {
        "id": "ro.codul_muncii.art_41.alin_1",
        "canonical_id": "ro.codul_muncii.art_41.alin_1",
        "source_id": "fixture",
        "law_id": "ro.codul_muncii",
        "law_title": "Codul muncii",
        "act_type": "code",
        "act_number": None,
        "publication_date": None,
        "effective_date": None,
        "version_start": None,
        "version_end": None,
        "status": "active",
        "hierarchy_path": ["Codul muncii", "Art. 41", "Alin. (1)"],
        "article_number": "41",
        "paragraph_number": "1",
        "letter_number": None,
        "point_number": None,
        "raw_text": "(1) Contractul individual de munca poate fi modificat numai prin acordul partilor.",
        "normalized_text": "contract individual munca modificat acord parti act aditional salariu",
        "legal_domain": "munca",
        "legal_concepts": ["contract", "salariu"],
        "source_url": "https://legislatie.just.ro/test",
        "parent_id": "ro.codul_muncii.art_41",
        "parser_warnings": [],
        "created_at": None,
        "updated_at": None,
    },
    {
        "id": "ro.codul_muncii.art_41.alin_2",
        "canonical_id": "ro.codul_muncii.art_41.alin_2",
        "source_id": "fixture",
        "law_id": "ro.codul_muncii",
        "law_title": "Codul muncii",
        "act_type": "code",
        "act_number": None,
        "publication_date": None,
        "effective_date": None,
        "version_start": None,
        "version_end": None,
        "status": "active",
        "hierarchy_path": ["Codul muncii", "Art. 41", "Alin. (2)"],
        "article_number": "41",
        "paragraph_number": "2",
        "letter_number": None,
        "point_number": None,
        "raw_text": "(2) Modificarea contractului individual de munca se refera la elementele contractuale.",
        "normalized_text": "modificare contract individual munca elemente contractuale",
        "legal_domain": "munca",
        "legal_concepts": ["contract"],
        "source_url": "https://legislatie.just.ro/test",
        "parent_id": "ro.codul_muncii.art_41",
        "parser_warnings": [],
        "created_at": None,
        "updated_at": None,
    },
    {
        "id": "ro.codul_muncii.art_41.alin_3",
        "canonical_id": "ro.codul_muncii.art_41.alin_3",
        "source_id": "fixture",
        "law_id": "ro.codul_muncii",
        "law_title": "Codul muncii",
        "act_type": "code",
        "act_number": None,
        "publication_date": None,
        "effective_date": None,
        "version_start": None,
        "version_end": None,
        "status": "active",
        "hierarchy_path": ["Codul muncii", "Art. 41", "Alin. (3)"],
        "article_number": "41",
        "paragraph_number": "3",
        "letter_number": None,
        "point_number": None,
        "raw_text": "(3) Modificarea contractului individual de munca poate privi durata, locul, felul muncii si salariul.",
        "normalized_text": "modificare contract individual munca durata loc fel salariu",
        "legal_domain": "munca",
        "legal_concepts": ["contract", "salariu"],
        "source_url": "https://legislatie.just.ro/test",
        "parent_id": "ro.codul_muncii.art_41",
        "parser_warnings": [],
        "created_at": None,
        "updated_at": None,
    },
    {
        "id": "ro.codul_muncii.art_41.alin_3.lit_e",
        "canonical_id": "ro.codul_muncii.art_41.alin_3.lit_e",
        "source_id": "fixture",
        "law_id": "ro.codul_muncii",
        "law_title": "Codul muncii",
        "act_type": "code",
        "act_number": None,
        "publication_date": None,
        "effective_date": None,
        "version_start": None,
        "version_end": None,
        "status": "active",
        "hierarchy_path": ["Codul muncii", "Art. 41", "Alin. (3)", "Lit. e)"],
        "article_number": "41",
        "paragraph_number": "3",
        "letter_number": "e",
        "point_number": None,
        "raw_text": "e) salariul;",
        "normalized_text": "salariu",
        "legal_domain": "munca",
        "legal_concepts": ["salariu"],
        "source_url": "https://legislatie.just.ro/test",
        "parent_id": "ro.codul_muncii.art_41.alin_3",
        "parser_warnings": [],
        "created_at": None,
        "updated_at": None,
    },
]
