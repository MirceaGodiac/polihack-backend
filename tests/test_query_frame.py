import json
from pathlib import Path

import pytest

from apps.api.app.schemas import QueryPlan, QueryRequest
from apps.api.app.services.query_frame import LegalIntentRegistry, QueryFrameBuilder
from apps.api.app.services.query_understanding import QueryUnderstanding, normalize_ro_text


FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "eval" / "query_frame_cases.json"
)


def build_plan(question: str) -> QueryPlan:
    if len(question) < 10:
        return QueryPlan(
            question=question,
            normalized_question=normalize_ro_text(question),
            legal_domain=None,
            domain_confidence=0.0,
            domain_scores={},
            ambiguity_flags=["low_domain_confidence"],
            retrieval_filters={"status": "active", "date_context": "current"},
        )
    return QueryUnderstanding().build_plan(
        QueryRequest(
            question=question,
            jurisdiction="RO",
            date="current",
            mode="strict_citations",
            debug=True,
        )
    )


def build_frame(question: str):
    plan = build_plan(question)
    return QueryFrameBuilder().build(question=question, plan=plan)


def assert_any(actual: list[str], expected: list[str] | None) -> None:
    if not expected:
        return
    assert set(actual).intersection(expected)


@pytest.mark.parametrize(
    "case",
    json.loads(FIXTURE_PATH.read_text(encoding="utf-8")),
    ids=lambda case: case["id"],
)
def test_query_frame_eval_baseline(case):
    frame = build_frame(case["question"])

    if "expected_domain" in case:
        assert frame.domain == case["expected_domain"]
    assert_any(frame.intents, case.get("expected_intents_any"))
    assert_any(frame.meta_intents, case.get("expected_meta_intents_any"))
    assert_any(frame.targets, case.get("expected_targets_any"))
    assert_any(frame.actors, case.get("expected_actors_any"))
    assert_any(frame.qualifiers, case.get("expected_qualifiers_any"))
    if "expected_requires_clarification" in case:
        assert frame.requires_clarification is case["expected_requires_clarification"]
    if not frame.requires_clarification:
        assert frame.confidence > 0.70


def test_demo_query_frame_keeps_expected_labor_contract_fields():
    frame = build_frame("Poate angajatorul sa-mi scada salariul fara act aditional?")

    assert frame.domain == "munca"
    assert "labor_contract_modification" in frame.intents
    assert "salary" in frame.targets
    assert "employer" in frame.actors
    assert "employee" in frame.actors
    assert {
        "without_addendum",
        "without_agreement",
    }.intersection(frame.qualifiers)
    assert frame.requires_clarification is False


def test_demo_query_frame_tolerates_replacement_mark_in_aditional():
    frame = build_frame("Poate angajatorul sa-mi scada salariul fara act adi?ional?")

    assert frame.domain == "munca"
    assert "labor_contract_modification" in frame.intents
    assert "without_addendum" in frame.qualifiers
    assert frame.requires_clarification is False


@pytest.mark.parametrize(
    ("question", "intent_id"),
    [
        ("Ce pot face daca angajatorul intarzie plata salariului?", "labor_salary_payment"),
        ("Angajatorul trebuie sa imi dea preaviz inainte de concediere?", "labor_dismissal"),
        ("Cate ore suplimentare poate cere angajatorul?", "labor_working_time"),
        ("Cum se aproba concediul de odihna?", "labor_leave"),
        ("Ce sanctiune contraventionala se poate aplica?", "contravention_sanction"),
        ("Cum contest o amenda contraventionala?", "contravention_challenge"),
        ("Care este termenul de plata a amenzii contraventionale?", "contravention_payment_deadline"),
        ("Cand este un contract civil valabil?", "civil_contract_validity"),
        ("Cand pot cere despagubiri pentru prejudiciu?", "civil_liability"),
        ("In cat timp se prescrie dreptul la actiune pentru o datorie civila?", "civil_prescription"),
        ("Cand platesc impozitul datorat?", "tax_payment_obligation"),
        ("Cand trebuie depusa declaratia fiscala la ANAF?", "tax_declaration_obligation"),
        ("Ce penalitati fiscale se aplica pentru intarziere?", "tax_penalty_interest"),
    ],
)
def test_each_registered_v1_intent_is_detectable(question, intent_id):
    frame = build_frame(question)

    assert intent_id in frame.intents
    assert frame.confidence > 0.70
    assert frame.requires_clarification is False


def test_unknown_short_query_does_not_create_confident_artificial_intent():
    frame = build_frame("Ce fac?")

    assert frame.intents == []
    assert frame.confidence < 0.35
    assert frame.requires_clarification is True


def test_registry_has_no_article_or_unit_id_specific_rules():
    forbidden_fragments = ("art.", "article_number", "unit_id", "ro.codul")

    for intent in LegalIntentRegistry().all():
        payload = intent.model_dump(mode="json")
        flattened = json.dumps(payload, ensure_ascii=False).casefold()
        assert not any(fragment in flattened for fragment in forbidden_fragments)
