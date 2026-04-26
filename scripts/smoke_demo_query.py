from __future__ import annotations

import argparse
import sys
from typing import Any

import requests


DEMO_QUESTION = "Poate angajatorul sa-mi scada salariul fara act aditional?"
FORBIDDEN_CITATION_FRAGMENTS = (
    "art_16",
    "art_196",
    "art_42",
    "art_254",
    "art_264",
    "art_21",
    "art_92",
    "art_81",
    "art_35",
    "art_260",
    "art_68",
)
REQUIRED_CITATION_IDS = (
    "ro.codul_muncii.art_41.alin_1",
    "ro.codul_muncii.art_41.alin_3",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke test the LexAI demo query against a running API."
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8010",
        help="API base URL, for example http://127.0.0.1:8010",
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    try:
        query_payload = post_query(base_url)
        graph_payload = get_graph(base_url, query_payload["query_id"])
    except requests.RequestException as exc:
        print(f"FAIL request_error: {exc}")
        return 2

    failures = validate_query_payload(query_payload)
    failures.extend(validate_graph_payload(graph_payload))

    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        print_diagnostics(query_payload, graph_payload)
        return 1

    citation_ids = [citation["legal_unit_id"] for citation in query_payload["citations"]]
    edge_types = [edge["type"] for edge in graph_payload["graph"]["edges"]]
    print("PASS demo query smoke")
    print(f"query_id: {query_payload['query_id']}")
    print(f"citations: {citation_ids}")
    print(f"graph_edge_types: {sorted(set(edge_types))}")
    print(f"verifier_passed: {query_payload['verifier']['verifier_passed']}")
    return 0


def post_query(base_url: str) -> dict[str, Any]:
    response = requests.post(
        f"{base_url}/api/query",
        json={
            "question": DEMO_QUESTION,
            "jurisdiction": "RO",
            "date": "current",
            "mode": "strict_citations",
            "debug": True,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def get_graph(base_url: str, query_id: str) -> dict[str, Any]:
    response = requests.get(f"{base_url}/api/query/{query_id}/graph", timeout=30)
    response.raise_for_status()
    return response.json()


def validate_query_payload(payload: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    citation_ids = [citation["legal_unit_id"] for citation in payload["citations"]]
    answer_text = " ".join(
        str(value or "")
        for value in (
            payload["answer"].get("short_answer"),
            payload["answer"].get("detailed_answer"),
        )
    )
    answer_lower = answer_text.casefold()

    for required_id in REQUIRED_CITATION_IDS:
        if required_id not in citation_ids:
            failures.append(f"missing_required_citation:{required_id}")

    for citation_id in citation_ids:
        if any(fragment in citation_id for fragment in FORBIDDEN_CITATION_FRAGMENTS):
            failures.append(f"forbidden_citation:{citation_id}")

    if "art. 41" not in answer_lower:
        failures.append("answer_missing_art_41")
    for forbidden in ("art. 16", "art. 196", "art. 42", "art. 254", "art. 264"):
        if forbidden in answer_lower:
            failures.append(f"answer_contains_forbidden_reference:{forbidden}")

    verifier = payload["verifier"]
    if verifier.get("claims_unsupported", 0) > 0:
        failures.append("verifier_has_unsupported_claims")
    if not verifier.get("verifier_passed") and verifier.get("claims_unsupported", 0) > 0:
        failures.append("verifier_failed_with_unsupported_claims")
    return failures


def validate_graph_payload(payload: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    edge_types = {edge["type"] for edge in payload["graph"]["edges"]}
    if "cited_in_answer" not in edge_types:
        failures.append("graph_missing_cited_in_answer_edges")
    if not payload.get("highlighted_edge_ids"):
        failures.append("graph_highlighted_edge_ids_empty")
    for required_id in REQUIRED_CITATION_IDS:
        if required_id not in payload["cited_unit_ids"]:
            failures.append(f"graph_missing_cited_unit:{required_id}")
    return failures


def print_diagnostics(
    query_payload: dict[str, Any],
    graph_payload: dict[str, Any],
) -> None:
    answer = query_payload.get("answer") or {}
    print("diagnostics.short_answer:")
    print(answer.get("short_answer"))
    print("diagnostics.citation_unit_ids:")
    print([citation.get("legal_unit_id") for citation in query_payload.get("citations", [])])
    print("diagnostics.first_10_evidence:")
    for evidence in query_payload.get("evidence_units", [])[:10]:
        print(
            {
                "id": evidence.get("id"),
                "support_role": evidence.get("support_role"),
                "rerank_score": evidence.get("rerank_score"),
                "retrieval_score": evidence.get("retrieval_score"),
            }
        )
    print("diagnostics.graph_highlighted_edge_ids:")
    print(graph_payload.get("highlighted_edge_ids"))


if __name__ == "__main__":
    sys.exit(main())
