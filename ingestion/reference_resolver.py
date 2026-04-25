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
    # Build fast lookup index: O(N)
    # Maps (corpus_id, art, alin, lit) -> unit_id
    known_index = {}
    for u in units:
        uid = u["id"]
        parts = uid.split(".")
        corpus = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
        
        art = alin = lit = None
        for p in parts:
            if p.startswith("art_"):
                art = p.split("_")[1]
            elif p.startswith("alin_"):
                alin = p.split("_")[1]
            elif p.startswith("lit_"):
                lit = p.split("_")[1]
                
        last_part = parts[-1]
        if last_part.startswith("art_"):
            known_index[(corpus, art, None, None)] = uid
        elif last_part.startswith("alin_"):
            known_index[(corpus, art, alin, None)] = uid
        elif last_part.startswith("lit_"):
            known_index[(corpus, art, alin, lit)] = uid

    resolved_candidates: List[Dict[str, Any]] = []
    edges:               List[Dict[str, Any]] = []

    for cand in candidates:
        cand = dict(cand)  # shallow copy

        if cand.get("target_law_hint") == "external":
            cand["status"]     = "external_unresolved"
            cand["confidence"] = 0.0
            cand["target_id"]  = None
            resolved_candidates.append(cand)
            continue

        corpus_id = _corpus_id_from_unit_id(cand["source_unit_id"])
        article   = cand.get("target_article")
        paragraph = cand.get("target_paragraph")
        letter    = cand.get("target_letter")

        if not article:
            cand["status"]     = "unresolved"
            cand["confidence"] = 0.0
            cand["target_id"]  = None
            resolved_candidates.append(cand)
            continue

        # Normalize 
        norm_art = normalize_number(article)
        norm_para = normalize_number(paragraph) if paragraph else None
        norm_lit = letter.lower().strip() if letter else None

        target_id = None
        status = None

        # O(1) resolution logic
        if norm_lit:
            exact = known_index.get((corpus_id, norm_art, norm_para, norm_lit))
            if exact:
                target_id, status = exact, "resolved_high_confidence"
            else:
                fallback = known_index.get((corpus_id, norm_art, norm_para, None))
                if fallback:
                    target_id, status = fallback, "resolved_medium_confidence"
                else:
                    fallback_art = known_index.get((corpus_id, norm_art, None, None))
                    if fallback_art:
                        target_id, status = fallback_art, "resolved_medium_confidence"
        elif norm_para:
            exact = known_index.get((corpus_id, norm_art, norm_para, None))
            if exact:
                target_id, status = exact, "resolved_high_confidence"
            else:
                fallback_art = known_index.get((corpus_id, norm_art, None, None))
                if fallback_art:
                    target_id, status = fallback_art, "resolved_medium_confidence"
        else:
            exact = known_index.get((corpus_id, norm_art, None, None))
            if exact:
                target_id, status = exact, "resolved_high_confidence"

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

        edges.append({
            "source_id": cand["source_unit_id"],
            "target_id": target_id,
            "type":      "references",
            "confidence": confidence,
        })

    return resolved_candidates, edges
