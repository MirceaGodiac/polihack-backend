"""
reference_extractor.py
----------------------
Parses raw_text of a LegalUnit and returns a list of ReferenceCandidate dicts.

Supports:
  - REF_ART_RE   – standalone article references: "art. 17"
  - REF_ALIN_RE  – paragraph references: "alin. (2)"
  - REF_LOCAL_RE – local-act hints: "prezenta lege", "prezentul cod"
  - Combined pattern: "art. 17 alin. (1)"

Does NOT handle cross-act resolution (delegated to reference_resolver.py).
"""

import re
from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# Compiled regexes
# ---------------------------------------------------------------------------

# Matches: art. 17  |  art. 17^1
REF_ART_RE = re.compile(
    r'art\.\s*(\d+(?:\^\d+)?)',
    re.IGNORECASE
)

# Matches: alin. (1)  |  alin.(2)
REF_ALIN_RE = re.compile(
    r'alin\.\s*\((\d+)\)',
    re.IGNORECASE
)

# Matches local-act self-references
REF_LOCAL_RE = re.compile(
    r'(prezenta\s+lege|prezentul\s+(?:cod|regulament|ordin))',
    re.IGNORECASE
)

# Combined: "art. 17 alin. (1)"  – the most common citation pattern
REF_COMBINED_RE = re.compile(
    r'art\.\s*(\d+(?:\^\d+)?)(?:\s+alin\.\s*\((\d+)\))?',
    re.IGNORECASE
)

# Hint for external law references – e.g. "din Codul Civil", "din Legea 53/2003"
# When detected we mark the candidate as external
REF_EXTERNAL_HINT_RE = re.compile(
    r'din\s+(Legea|Codul|Ordonanța|OUG|OG|HG)\s+',
    re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_references(unit: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Given a LegalUnit dict, scan its raw_text and return a list of
    ReferenceCandidate dicts.

    Each candidate has:
        source_unit_id  – ID of the unit that contains the citation
        raw_reference   – the matched substring (for debugging)
        target_article  – str | None  (e.g. "17" or "41^1")
        target_paragraph – str | None (e.g. "1")
        target_law_hint  – "same_act" | "external" | None
    """
    source_id = unit.get("id", "")
    raw_text  = unit.get("raw_text", "")
    candidates: List[Dict[str, Any]] = []

    for match in REF_COMBINED_RE.finditer(raw_text):
        art_val  = match.group(1)          # always present when REF_COMBINED_RE fires
        alin_val = match.group(2) or None  # may be absent

        raw_ref = match.group(0)

        # Determine law hint: look at the surrounding context (up to 60 chars after)
        start, end = match.span()
        context_after = raw_text[end:end + 80]

        is_local    = bool(REF_LOCAL_RE.search(raw_text[max(0, start - 60):end + 60]))
        is_external = bool(REF_EXTERNAL_HINT_RE.search(context_after))

        if is_external:
            law_hint = "external"
        elif is_local:
            law_hint = "same_act"
        else:
            # Default for intra-act references when no explicit hint
            law_hint = "same_act"

        candidates.append({
            "source_unit_id":   source_id,
            "raw_reference":    raw_ref,
            "target_article":   art_val,
            "target_paragraph": alin_val,
            "target_law_hint":  law_hint,
        })

    return candidates
