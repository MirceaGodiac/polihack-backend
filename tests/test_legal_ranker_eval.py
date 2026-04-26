import copy
import json
from pathlib import Path

from scripts.evaluate_legal_ranker import (
    DEFAULT_FIXTURE_PATH,
    DEMO_CASE_ID,
    evaluate_case,
    evaluate_cases,
    load_cases,
    main,
)


REQUIRED_CASE_FIELDS = {
    "id",
    "question",
    "expected_top_any",
    "expected_in_top_3_any",
    "forbidden_top_3",
    "forbidden_top_5",
    "candidates",
}
REQUIRED_CANDIDATE_FIELDS = {
    "unit_id",
    "rank",
    "retrieval_score",
    "score_breakdown",
    "unit",
}
REQUIRED_UNIT_FIELDS = {
    "id",
    "law_id",
    "law_title",
    "status",
    "legal_domain",
    "raw_text",
    "normalized_text",
    "legal_concepts",
}


def test_legal_ranker_eval_fixture_loads():
    cases = load_cases(DEFAULT_FIXTURE_PATH)

    assert len(cases) >= 5
    assert {case["id"] for case in cases} >= {
        "labor_salary_reduction_without_addendum",
        "labor_salary_payment_delay",
        "labor_dismissal_notice",
        "contravention_fine_challenge",
        "civil_prescription",
    }


def test_legal_ranker_eval_fixture_cases_have_required_fields():
    for case in load_cases(DEFAULT_FIXTURE_PATH):
        assert REQUIRED_CASE_FIELDS <= set(case)
        assert case["fixture_scope"] == "synthetic_ranker_eval"
        assert case["candidates"]
        for candidate in case["candidates"]:
            assert REQUIRED_CANDIDATE_FIELDS <= set(candidate)
            assert REQUIRED_UNIT_FIELDS <= set(candidate["unit"])


def test_legal_ranker_eval_runner_returns_metrics():
    report = evaluate_cases(load_cases(DEFAULT_FIXTURE_PATH))

    summary = report["summary"]
    assert summary["cases_total"] >= 5
    assert 0.0 <= summary["hit_rate_at_1"] <= 1.0
    assert 0.0 <= summary["hit_rate_at_3"] <= 1.0
    assert 0.0 <= summary["hit_rate_at_5"] <= 1.0
    assert 0.0 <= summary["mrr"] <= 1.0
    assert "cases" in report
    assert len(report["cases"]) == summary["cases_total"]


def test_legal_ranker_eval_demo_case_passes():
    demo_case = next(
        case for case in load_cases(DEFAULT_FIXTURE_PATH) if case["id"] == DEMO_CASE_ID
    )

    result = evaluate_case(demo_case)

    assert result["passed"] is True
    assert result["expected_top_any_found_rank"] in (1, 2)
    assert "ro.codul_muncii.art_264.lit_a" not in result["top_3_unit_ids"]
    assert result["query_frame"]["intents"] == ["labor_contract_modification"]
    assert result["query_frame"]["confidence"] >= 0.60


def test_legal_ranker_eval_script_returns_zero_and_writes_report(tmp_path: Path):
    output_path = tmp_path / "legal_ranker_eval.json"

    exit_code = main(["--fixture", str(DEFAULT_FIXTURE_PATH), "--output", str(output_path)])

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["summary"]["failed_cases"] == 0
    assert payload["summary"]["passed_cases"] == payload["summary"]["cases_total"]


def test_legal_ranker_eval_detects_forbidden_top_3_hit():
    case = copy.deepcopy(load_cases(DEFAULT_FIXTURE_PATH)[0])
    baseline = evaluate_case(case)
    case["forbidden_top_3"] = [baseline["top_1_unit_id"]]

    result = evaluate_case(case)

    assert result["passed"] is False
    assert result["forbidden_top_3_hits"] == [baseline["top_1_unit_id"]]
    assert "forbidden_top_3_hit" in result["failures"]
