from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.api.app.schemas import (  # noqa: E402
    GraphExpansionResult,
    QueryRequest,
    RawRetrievalResponse,
    RetrievalCandidate,
)
from apps.api.app.services.legal_ranker import LegalRanker  # noqa: E402
from apps.api.app.services.query_frame import QueryFrameBuilder  # noqa: E402
from apps.api.app.services.query_understanding import QueryUnderstanding  # noqa: E402

DEFAULT_FIXTURE_PATH = (
    REPO_ROOT / "tests" / "fixtures" / "eval" / "legal_ranker_cases.json"
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
    hit_at_5 = sum(1 for result in case_results if result["hit_top_5"])
    reciprocal_rank_total = sum(result["reciprocal_rank"] for result in case_results)
    forbidden_top_3_total = sum(
        result["forbidden_top_3_count"] for result in case_results
    )
    forbidden_top_5_total = sum(
        result["forbidden_top_5_count"] for result in case_results
    )
    passed_cases = sum(1 for result in case_results if result["passed"])
    failed_case_ids = [
        result["id"] for result in case_results if not result["passed"]
    ]

    return {
        "summary": {
            "cases_total": cases_total,
            "hit_rate_at_1": _rate(hit_at_1, cases_total),
            "hit_rate_at_3": _rate(hit_at_3, cases_total),
            "hit_rate_at_5": _rate(hit_at_5, cases_total),
            "mrr": _rate(reciprocal_rank_total, cases_total),
            "forbidden_top_3_total": forbidden_top_3_total,
            "forbidden_top_5_total": forbidden_top_5_total,
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
    retrieval_response = _retrieval_response(case)
    graph_expansion = _graph_expansion(case)
    result = LegalRanker().rank(
        question=question,
        plan=plan,
        retrieval_response=retrieval_response,
        graph_expansion=graph_expansion,
        query_frame=query_frame,
        debug=True,
    )
    ranked = result.ranked_candidates
    ranked_ids = [candidate.unit_id for candidate in ranked]
    top_1_unit_id = ranked_ids[0] if ranked_ids else None
    top_3_unit_ids = ranked_ids[:3]
    top_5_unit_ids = ranked_ids[:5]
    expected_top_any = case.get("expected_top_any", [])
    expected_rank = _first_rank(ranked_ids, expected_top_any)
    forbidden_top_3_hits = _hits(top_3_unit_ids, case.get("forbidden_top_3", []))
    forbidden_top_5_hits = _hits(top_5_unit_ids, case.get("forbidden_top_5", []))
    min_confidence = case.get(
        "min_query_frame_confidence",
        DEFAULT_MIN_QUERY_FRAME_CONFIDENCE,
    )

    hit_top_1 = expected_rank == 1
    hit_top_3 = expected_rank is not None and expected_rank <= 3
    hit_top_5 = expected_rank is not None and expected_rank <= 5
    confidence_pass = query_frame.confidence >= min_confidence
    expected_rank_pass = hit_top_3
    if case.get("id") == DEMO_CASE_ID:
        expected_rank_pass = expected_rank is not None and expected_rank <= 2

    failures = []
    if not expected_rank_pass:
        failures.append("expected_top_any_not_in_required_rank")
    if forbidden_top_3_hits:
        failures.append("forbidden_top_3_hit")
    if forbidden_top_5_hits:
        failures.append("forbidden_top_5_hit")
    if not confidence_pass:
        failures.append("query_frame_confidence_below_threshold")

    return {
        "id": case["id"],
        "passed": not failures,
        "failures": failures,
        "top_1_unit_id": top_1_unit_id,
        "top_3_unit_ids": top_3_unit_ids,
        "top_5_unit_ids": top_5_unit_ids,
        "reciprocal_rank": 1.0 / expected_rank if expected_rank else 0.0,
        "hit_top_1": hit_top_1,
        "hit_top_3": hit_top_3,
        "hit_top_5": hit_top_5,
        "forbidden_top_3_count": len(forbidden_top_3_hits),
        "forbidden_top_5_count": len(forbidden_top_5_hits),
        "forbidden_top_3_hits": forbidden_top_3_hits,
        "forbidden_top_5_hits": forbidden_top_5_hits,
        "expected_top_any_found_rank": expected_rank,
        "query_frame_intents": query_frame.intents,
        "query_frame_confidence": query_frame.confidence,
        "query_frame": {
            "domain": query_frame.domain,
            "intents": query_frame.intents,
            "confidence": query_frame.confidence,
        },
        "ranked": [
            {
                "unit_id": candidate.unit_id,
                "rank": candidate.rank,
                "rerank_score": candidate.rerank_score,
                "score_breakdown": candidate.score_breakdown.model_dump(mode="json"),
            }
            for candidate in ranked
        ],
        "debug": result.debug,
    }


def print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("LegalRanker V2 offline eval")
    print(f"cases_total: {summary['cases_total']}")
    print(f"hit_rate_at_1: {summary['hit_rate_at_1']:.3f}")
    print(f"hit_rate_at_3: {summary['hit_rate_at_3']:.3f}")
    print(f"hit_rate_at_5: {summary['hit_rate_at_5']:.3f}")
    print(f"mrr: {summary['mrr']:.3f}")
    print(f"forbidden_top_3_total: {summary['forbidden_top_3_total']}")
    print(f"forbidden_top_5_total: {summary['forbidden_top_5_total']}")
    print(f"passed_cases: {summary['passed_cases']}")
    print(f"failed_cases: {summary['failed_cases']}")
    for case in report["cases"]:
        status = "PASS" if case["passed"] else "FAIL"
        failures = ",".join(case["failures"]) if case["failures"] else "-"
        top_3 = ", ".join(case["top_3_unit_ids"])
        print(
            f"{status} {case['id']}: "
            f"top1={case['top_1_unit_id']} "
            f"rr={case['reciprocal_rank']:.3f} "
            f"qf_conf={case['query_frame_confidence']:.2f} "
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
    parser = argparse.ArgumentParser(description="Evaluate LegalRanker V2 fixtures.")
    parser.add_argument(
        "--fixture",
        type=Path,
        default=DEFAULT_FIXTURE_PATH,
        help="Path to legal_ranker_cases.json.",
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


def _retrieval_response(case: dict[str, Any]) -> RawRetrievalResponse:
    return RawRetrievalResponse(
        candidates=[
            RetrievalCandidate(**candidate) for candidate in case.get("candidates", [])
        ],
        retrieval_methods=["fixture"],
    )


def _graph_expansion(case: dict[str, Any]) -> GraphExpansionResult:
    graph_data = case.get("graph_expansion")
    if not graph_data:
        return GraphExpansionResult()
    return GraphExpansionResult(**graph_data)


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
