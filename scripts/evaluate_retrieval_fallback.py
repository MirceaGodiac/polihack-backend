from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.api.app.schemas import QueryRequest, RawRetrievalRequest  # noqa: E402
from apps.api.app.services.query_frame import QueryFrameBuilder  # noqa: E402
from apps.api.app.services.query_understanding import QueryUnderstanding  # noqa: E402
from apps.api.app.services.raw_retriever import (  # noqa: E402
    evaluate_fallback_candidates,
)

DEFAULT_FIXTURE_PATH = (
    REPO_ROOT / "tests" / "fixtures" / "eval" / "retrieval_fallback_cases.json"
)
DEFAULT_MIN_QUERY_FRAME_CONFIDENCE = 0.60
DEMO_CASE_ID = "labor_salary_reduction_without_addendum"


def load_cases(path: Path = DEFAULT_FIXTURE_PATH) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    case_results = [evaluate_case(case) for case in cases]
    cases_total = len(case_results)
    hit_at_1 = sum(1 for result in case_results if result["hit_top_1"])
    hit_at_3 = sum(1 for result in case_results if result["hit_top_3"])
    forbidden_top_3_total = sum(
        len(result["forbidden_top_3_hits"]) for result in case_results
    )
    expansion_hit_rate_total = sum(
        result["expected_expanded_terms_hit_rate"] for result in case_results
    )
    passed_cases = sum(1 for result in case_results if result["passed"])
    failed_case_ids = [
        result["id"] for result in case_results if not result["passed"]
    ]
    return {
        "summary": {
            "cases_total": cases_total,
            "expansion_hit_rate_avg": _rate(expansion_hit_rate_total, cases_total),
            "hit_rate_at_1": _rate(hit_at_1, cases_total),
            "hit_rate_at_3": _rate(hit_at_3, cases_total),
            "forbidden_top_3_total": forbidden_top_3_total,
            "passed_cases": passed_cases,
            "failed_cases": cases_total - passed_cases,
            "failed_case_ids": failed_case_ids,
        },
        "cases": case_results,
    }


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    question = case["question"]
    plan = QueryUnderstanding().build_plan(_query_request(question))
    query_frame = QueryFrameBuilder().build(question=question, plan=plan)
    raw_request = RawRetrievalRequest(
        question=question,
        filters=plan.retrieval_filters,
        retrieval_filters=plan.retrieval_filters,
        query_frame=query_frame.model_dump(mode="json"),
        exact_citations=[
            citation.model_dump(mode="json") for citation in plan.exact_citations
        ],
        top_k=50,
        debug=True,
    )
    eval_result = evaluate_fallback_candidates(
        question=raw_request.question,
        filters=raw_request.retrieval_filters,
        query_frame=raw_request.query_frame,
        candidate_units=_filtered_units(
            case.get("candidate_units", []),
            raw_request.retrieval_filters,
        ),
    )

    ranked = eval_result["ranked"]
    ranked_ids = [row["unit_id"] for row in ranked]
    top_1_unit_id = ranked_ids[0] if ranked_ids else None
    top_3_unit_ids = ranked_ids[:3]
    expected_top_any = case.get("expected_top_any", [])
    expected_rank = _first_rank(ranked_ids, expected_top_any)
    hit_top_1 = expected_rank == 1
    hit_top_3 = expected_rank is not None and expected_rank <= 3
    forbidden_top_3_hits = _hits(top_3_unit_ids, case.get("forbidden_top_3", []))
    expected_expanded_terms = case.get("expected_expanded_terms_any", [])
    expanded_term_hits = _expected_term_hits(
        expected_expanded_terms,
        [
            *eval_result["expanded_terms"],
            *eval_result["registry_expanded_terms"],
        ],
    )
    expanded_hit_rate = _rate(len(expanded_term_hits), len(expected_expanded_terms))
    expected_intent = case.get("expected_intent")
    intent_pass = expected_intent in query_frame.intents or (
        expected_intent == eval_result["fallback_intent"]
    )
    confidence_pass = query_frame.confidence >= case.get(
        "min_query_frame_confidence",
        DEFAULT_MIN_QUERY_FRAME_CONFIDENCE,
    )

    expected_rank_pass = hit_top_3
    strategy_pass = True
    if case["id"] == DEMO_CASE_ID:
        expected_rank_pass = expected_rank is not None and expected_rank <= 2
        strategy_pass = eval_result["scoring_strategy"] in {
            "registry_intent_grouped_lexical_fallback",
            "legacy_intent_grouped_lexical_fallback",
        } and eval_result["fallback_intent_source"] in {
            "query_frame_registry",
            "legacy",
        }

    failures = []
    if not intent_pass:
        failures.append("expected_intent_missing")
    if expanded_hit_rate < 0.60:
        failures.append("expanded_terms_below_threshold")
    if not expected_rank_pass:
        failures.append("expected_top_any_not_in_required_rank")
    if forbidden_top_3_hits:
        failures.append("forbidden_top_3_hit")
    if not confidence_pass:
        failures.append("query_frame_confidence_below_threshold")
    if not strategy_pass:
        failures.append("demo_scoring_strategy_not_registry")

    return {
        "id": case["id"],
        "question": question,
        "passed": not failures,
        "failures": failures,
        "query_frame": {
            "domain": query_frame.domain,
            "intents": query_frame.intents,
            "confidence": query_frame.confidence,
        },
        "fallback_intent": eval_result["fallback_intent"],
        "fallback_intent_source": eval_result["fallback_intent_source"],
        "scoring_strategy": eval_result["scoring_strategy"],
        "lexical_terms": eval_result["lexical_terms"],
        "expanded_terms": eval_result["expanded_terms"],
        "registry_expanded_terms": eval_result["registry_expanded_terms"],
        "expected_expanded_terms_hit_rate": expanded_hit_rate,
        "expected_expanded_terms_hits": expanded_term_hits,
        "top_1_unit_id": top_1_unit_id,
        "top_3_unit_ids": top_3_unit_ids,
        "hit_top_1": hit_top_1,
        "hit_top_3": hit_top_3,
        "forbidden_top_3_hits": forbidden_top_3_hits,
        "group_scores_by_top_candidate": {
            row["unit_id"]: row["group_scores"] for row in ranked[:3]
        },
        "ranked": [
            {
                "unit_id": row["unit_id"],
                "rank": row["rank"],
                "fallback_score": row["fallback_score"],
                "group_scores": row["group_scores"],
            }
            for row in ranked
        ],
    }


def print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("Registry-driven retrieval fallback offline eval")
    print(f"cases_total: {summary['cases_total']}")
    print(f"expansion_hit_rate_avg: {summary['expansion_hit_rate_avg']:.3f}")
    print(f"hit_rate_at_1: {summary['hit_rate_at_1']:.3f}")
    print(f"hit_rate_at_3: {summary['hit_rate_at_3']:.3f}")
    print(f"forbidden_top_3_total: {summary['forbidden_top_3_total']}")
    print(f"passed_cases: {summary['passed_cases']}")
    print(f"failed_cases: {summary['failed_cases']}")
    for case in report["cases"]:
        status = "PASS" if case["passed"] else "FAIL"
        failures = ",".join(case["failures"]) if case["failures"] else "-"
        top_3 = ", ".join(case["top_3_unit_ids"])
        print(
            f"{status} {case['id']}: "
            f"intent={case['fallback_intent']} "
            f"strategy={case['scoring_strategy']} "
            f"expansion={case['expected_expanded_terms_hit_rate']:.2f} "
            f"top1={case['top_1_unit_id']} "
            f"top3=[{top_3}] "
            f"failures={failures}"
        )


def write_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate registry-driven retrieval fallback fixtures."
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=DEFAULT_FIXTURE_PATH,
        help="Path to retrieval_fallback_cases.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report output path.",
    )
    args = parser.parse_args(argv)

    report = evaluate_cases(load_cases(args.fixture))
    print_summary(report)
    if args.output:
        write_report(report, args.output)
    return 0 if report["summary"]["failed_cases"] == 0 else 1


def _query_request(question: str) -> QueryRequest:
    return QueryRequest(
        question=question,
        jurisdiction="RO",
        date="current",
        mode="strict_citations",
        debug=True,
    )


def _filtered_units(
    units: list[dict[str, Any]],
    filters: dict[str, Any],
) -> list[dict[str, Any]]:
    filtered = []
    for unit in units:
        if filters.get("legal_domain") and unit.get("legal_domain") != filters["legal_domain"]:
            continue
        if filters.get("status") and unit.get("status") != filters["status"]:
            continue
        filtered.append(unit)
    return filtered


def _expected_term_hits(
    expected_terms: list[str],
    expanded_terms: list[str],
) -> list[str]:
    expanded = set(expanded_terms)
    return [term for term in expected_terms if term in expanded]


def _first_rank(ranked_ids: list[str], expected_ids: list[str]) -> int | None:
    expected = set(expected_ids)
    for index, unit_id in enumerate(ranked_ids, start=1):
        if unit_id in expected:
            return index
    return None


def _hits(ranked_ids: list[str], forbidden_ids: list[str]) -> list[str]:
    forbidden = set(forbidden_ids)
    return [unit_id for unit_id in ranked_ids if unit_id in forbidden]


def _rate(numerator: float, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


if __name__ == "__main__":
    raise SystemExit(main())
