"""
reference_resolver.py
---------------------
Resolves ReferenceCandidate dicts against a set of known LegalUnit IDs and
generates LegalEdge dicts of type "references".

Resolution strategy (MVP – intra-act only):
  1. Build a lookup set from the provided unit list.
  2. For each candidate with target_law_hint == "same_act":
       - Derive the corpus_id from the source_unit_id.
       - Construct candidate target IDs (most-specific first).
       - If a constructed ID exists in the lookup → resolved_high_confidence (≥ 0.90).
       - If only the article-level ID exists (paragraph missing) → resolved_medium_confidence (0.70).
       - Otherwise → unresolved (no edge created).
  3. External-law candidates are skipped with status "external_unresolved".
"""

from typing import List, Dict, Any, Set
from .legal_ids import normalize_number


# ---------------------------------------------------------------------------
# Confidence thresholds
# ---------------------------------------------------------------------------
HIGH_CONF_THRESHOLD   = 0.90
MEDIUM_CONF_THRESHOLD = 0.70


def _corpus_id_from_unit_id(unit_id: str) -> str:
    """Extract corpus prefix, e.g. 'ro.codul_muncii' from any hierarchical unit ID."""
    parts = unit_id.split(".")
    # corpus_id is always the first two dot-segments (ro.<name>)
    return ".".join(parts[:2]) if len(parts) >= 2 else parts[0]


def _find_target_id(
    known_ids: set,
    corpus_id: str,
    article: str,
    paragraph: str | None,
) -> tuple[str | None, str | None]:
    """
    Search known_ids for a unit whose ID belongs to corpus_id and whose
    last path segment(s) match the referenced article (and optional paragraph).

    Returns (target_id, status) where status is:
        'resolved_high_confidence'   – article+paragraph matched
        'resolved_medium_confidence' – article only matched
        None                         – no match
    """
    norm_art  = normalize_number(article)
    art_suffix  = f".art_{norm_art}"

    if paragraph:
        norm_para = normalize_number(paragraph)
        para_suffix = f".alin_{norm_para}"

        # Try to find a unit: <corpus>.<...>.art_N.alin_M
        for uid in known_ids:
            if uid.startswith(corpus_id) and uid.endswith(art_suffix + para_suffix):
                return uid, "resolved_high_confidence"

    # Fallback: article only – find <corpus>.<...>.art_N  (must end exactly there)
    for uid in known_ids:
        if uid.startswith(corpus_id) and uid.endswith(art_suffix):
            # Make sure nothing follows (i.e., uid ends at art_N, not art_N.alin_1)
            tail = uid[len(corpus_id):]
            if tail.endswith(art_suffix) and not any(
                uid.endswith(art_suffix + f".{x}") for x in ["alin", "lit"]
            ):
                return uid, "resolved_medium_confidence"

    return None, None


def resolve_references(
    candidates: List[Dict[str, Any]],
    units: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Resolve a list of ReferenceCandidate dicts against a unit list.

    Returns:
        resolved_candidates – candidates enriched with resolution metadata
        edges               – LegalEdge dicts (type="references") for high/medium confidence
    """
    # Build fast lookup: set of known IDs
    known_ids: Set[str] = {u["id"] for u in units}

    resolved_candidates: List[Dict[str, Any]] = []
    edges:               List[Dict[str, Any]] = []

    for cand in candidates:
        cand = dict(cand)  # shallow copy – don't mutate caller's data

        if cand.get("target_law_hint") == "external":
            cand["status"]     = "external_unresolved"
            cand["confidence"] = 0.0
            cand["target_id"]  = None
            resolved_candidates.append(cand)
            continue

        # ---- intra-act resolution ----
        corpus_id = _corpus_id_from_unit_id(cand["source_unit_id"])
        article   = cand.get("target_article")
        paragraph = cand.get("target_paragraph")

        if not article:
            cand["status"]     = "unresolved"
            cand["confidence"] = 0.0
            cand["target_id"]  = None
            resolved_candidates.append(cand)
            continue

        # Hierarchical ID search – works regardless of chapter/title nesting
        target_id, status = _find_target_id(known_ids, corpus_id, article, paragraph)

        if target_id is None:
            cand["status"]     = "unresolved"
            cand["confidence"] = 0.0
            cand["target_id"]  = None
            resolved_candidates.append(cand)
            continue

        confidence = HIGH_CONF_THRESHOLD if status == "resolved_high_confidence" else MEDIUM_CONF_THRESHOLD

        cand["target_id"]  = target_id
        cand["confidence"] = confidence
        cand["status"]     = status
        resolved_candidates.append(cand)

        # Emit edge for high or medium confidence
        edges.append({
            "source_id": cand["source_unit_id"],
            "target_id": target_id,
            "type":      "references",
            "confidence": confidence,
        })

    return resolved_candidates, edges
