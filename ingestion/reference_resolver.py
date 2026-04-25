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
    """Extract corpus prefix, e.g. 'ro.codul_muncii' from 'ro.codul_muncii.art_41'."""
    parts = unit_id.split(".")
    # corpus_id is always the first two dot-segments (ro.<name>)
    return ".".join(parts[:2]) if len(parts) >= 2 else parts[0]


def _build_target_id(corpus_id: str, article: str, paragraph: str | None) -> str:
    """Construct the target LegalUnit ID from components."""
    norm_art = normalize_number(article)
    base = f"{corpus_id}.art_{norm_art}"
    if paragraph:
        norm_para = normalize_number(paragraph)
        return f"{base}.alin_{norm_para}"
    return base


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

        # Try most-specific ID first (with paragraph), then fall back to article only
        full_id    = _build_target_id(corpus_id, article, paragraph)
        article_id = _build_target_id(corpus_id, article, None)

        if full_id in known_ids:
            confidence = HIGH_CONF_THRESHOLD
            target_id  = full_id
            status     = "resolved_high_confidence"
        elif article_id in known_ids:
            confidence = MEDIUM_CONF_THRESHOLD
            target_id  = article_id
            status     = "resolved_medium_confidence"
        else:
            cand["status"]     = "unresolved"
            cand["confidence"] = 0.0
            cand["target_id"]  = None
            resolved_candidates.append(cand)
            continue

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
