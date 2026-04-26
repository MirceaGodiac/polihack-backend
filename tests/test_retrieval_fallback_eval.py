import copy
import json
from pathlib import Path

from scripts.evaluate_retrieval_fallback import (
    DEFAULT_FIXTURE_PATH,
    DEMO_CASE_ID,
    evaluate_case,
    evaluate_cases,
    load_cases,
    main,
)


REQUIRED_CASE_FIELDS = {
    "id",
    "fixture_scope",
    "question",
    "expected_intent",
    "expected_expanded_terms_any",
    "expected_top_any",
    "forbidden_top_3",
    "candidate_units",
}
REQUIRED_UNIT_FIELDS = {
    "id",
    "law_id",
    "law_title",
    "legal_domain",
    "status",
    "raw_text",
    "normalized_text",
    "hierarchy_path",
}


def test_retrieval_fallback_eval_fixture_loads():
    cases = load_cases(DEFAULT_FIXTURE_PATH)

    assert len(cases) >= 6
    assert {case["id"] for case in cases} >= {
        "labor_salary_reduction_without_addendum",
        "labor_salary_payment_delay",
        "labor_dismissal_notice",
        "contravention_fine_challenge",
        "civil_prescription",
        "tax_declaration_obligation",
    }


def test_retrieval_fallback_eval_fixture_cases_have_required_fields():
    for case in load_cases(DEFAULT_FIXTURE_PATH):
        assert REQUIRED_CASE_FIELDS <= set(case)
        assert case["fixture_scope"] == "synthetic_retrieval_fallback_eval"
        assert case["candidate_units"]
        for unit in case["candidate_units"]:
            assert REQUIRED_UNIT_FIELDS <= set(unit)


def test_retrieval_fallback_eval_runner_returns_metrics():
    report = evaluate_cases(load_cases(DEFAULT_FIXTURE_PATH))

    summary = report["summary"]
    assert summary["cases_total"] >= 6
    assert 0.0 <= summary["expansion_hit_rate_avg"] <= 1.0
    assert 0.0 <= summary["hit_rate_at_1"] <= 1.0
    assert 0.0 <= summary["hit_rate_at_3"] <= 1.0
    assert "cases" in report
    assert len(report["cases"]) == summary["cases_total"]


def test_retrieval_fallback_eval_demo_case_passes():
    demo_case = next(
        case for case in load_cases(DEFAULT_FIXTURE_PATH) if case["id"] == DEMO_CASE_ID
    )

    result = evaluate_case(demo_case)

    assert result["passed"] is True
    assert result["fallback_intent"] == "labor_contract_modification"
    assert result["fallback_intent_source"] == "query_frame_registry"
    assert result["scoring_strategy"] == "registry_intent_grouped_lexical_fallback"
    assert result["expected_expanded_terms_hit_rate"] >= 0.60
    assert result["top_1_unit_id"] in {
        "ro.codul_muncii.art_41.alin_1",
        "ro.codul_muncii.art_41.alin_3",
    }
    assert "ro.codul_muncii.art_264.lit_a" not in result["top_3_unit_ids"]


def test_retrieval_fallback_eval_detects_forbidden_top_3_hit():
    case = copy.deepcopy(load_cases(DEFAULT_FIXTURE_PATH)[0])
    baseline = evaluate_case(case)
    case["forbidden_top_3"] = [baseline["top_1_unit_id"]]

    result = evaluate_case(case)

    assert result["passed"] is False
    assert result["forbidden_top_3_hits"] == [baseline["top_1_unit_id"]]
    assert "forbidden_top_3_hit" in result["failures"]


def test_retrieval_fallback_eval_script_returns_zero_and_writes_report(
    tmp_path: Path,
):
    output_path = tmp_path / "retrieval_fallback_eval.json"

    exit_code = main(["--fixture", str(DEFAULT_FIXTURE_PATH), "--output", str(output_path)])

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["summary"]["failed_cases"] == 0
    assert payload["summary"]["passed_cases"] == payload["summary"]["cases_total"]
