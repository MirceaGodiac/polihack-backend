"""
validators.py
-------------
Phase 1: validate_corpus – duplicate ID guard (blocking).
Phase 3: build_validation_report – full quality report for a processed corpus.
"""

import json
from typing import Any


# ---------------------------------------------------------------------------
# Phase 1 – Blocking duplicate ID check
# ---------------------------------------------------------------------------

def validate_corpus(units: list) -> bool:
    """
    Checks for duplicate IDs in the generated legal units.
    Raises ValueError (blocking) if duplicates are found.
    """
    seen_ids: set = set()
    for unit in units:
        unit_id = unit.get("id")
        if not unit_id:
            raise ValueError("Unit missing ID field")
        if unit_id in seen_ids:
            raise ValueError(f"Blocking Error: Duplicate ID found: {unit_id}")
        seen_ids.add(unit_id)
    return True


# ---------------------------------------------------------------------------
# Phase 3 – Full validation report
# ---------------------------------------------------------------------------

def build_validation_report(
    units: list[dict[str, Any]],
    contains_edges: list[dict[str, Any]],
    ref_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build a structured validation report for a processed corpus.

    Fields:
        duplicate_ids           – list of IDs that appear more than once (must be empty, else FAIL).
        status                  – "PASS" | "FAIL"
        total_units             – total LegalUnit count.
        total_contains_edges    – total structural (contains) edges.
        total_ref_candidates    – total extracted reference candidates.
        resolved_candidates     – candidates with status resolved_high/medium_confidence.
        ReferenceResolutionRate – float in [0, 1]; resolved / total (0 when no candidates).
        orphan_units            – list of unit IDs that appear as target in no contains edge
                                  AND are not the root (warning only, not blocking).
        warnings                – list of human-readable warning strings.
    """
    report: dict[str, Any] = {}

    # --- Duplicate check (blocking) ---
    seen: dict[str, int] = {}
    for unit in units:
        uid = unit.get("id", "")
        seen[uid] = seen.get(uid, 0) + 1
    duplicates = [uid for uid, count in seen.items() if count > 1]
    report["duplicate_ids"] = duplicates
    report["status"] = "FAIL" if duplicates else "PASS"

    # --- Counts ---
    report["total_units"] = len(units)
    report["total_contains_edges"] = len(contains_edges)
    report["total_ref_candidates"] = len(ref_candidates)

    resolved_statuses = {"resolved_high_confidence", "resolved_medium_confidence"}
    resolved_count = sum(
        1 for c in ref_candidates if c.get("status") in resolved_statuses
    )
    report["resolved_candidates"] = resolved_count

    total_candidates = len(ref_candidates)
    report["ReferenceResolutionRate"] = (
        round(resolved_count / total_candidates, 4) if total_candidates > 0 else 0.0
    )

    # --- Orphan detection (warning) ---
    target_ids = {e["target_id"] for e in contains_edges}
    all_unit_ids = {u["id"] for u in units}
    # An orphan is any unit that is NOT the target of any contains edge
    # (i.e., has no parent). Root-level units (titles, top articles) are expected orphans.
    orphans = sorted(all_unit_ids - target_ids)
    report["orphan_units"] = orphans

    warnings = []
    if orphans:
        warnings.append(
            f"{len(orphans)} orphan unit(s) detected (no parent contains-edge). "
            "These are likely top-level structural nodes."
        )
    if duplicates:
        warnings.append(f"BLOCKING: {len(duplicates)} duplicate ID(s) found.")

    report["warnings"] = warnings

    return report


def save_validation_report(report: dict[str, Any], path: str) -> None:
    """Serialize the validation report to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
