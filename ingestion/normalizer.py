from __future__ import annotations

import re
import unicodedata


ROMANIAN_MOJIBAKE_MARKERS = (
    "Äƒ",
    "Ä\x83",
    "Ä‚",
    "Ä\x82",
    "Ã¢",
    "Ã\xa2",
    "Ã‚",
    "Ã\x82",
    "Ã®",
    "Ã\xae",
    "ÃŽ",
    "È™",
    "È\x99",
    "È˜",
    "È\x98",
    "È›",
    "È\x9b",
    "Èš",
    "È\x9a",
    "ÅŸ",
    "Åž",
    "Å£",
    "Å¢",
    "muncÄ",
    "pÄ",
    "pÄrÈ",
)
ROMANIAN_MOJIBAKE_REPLACEMENTS = (
    ("Äƒ", "ă"),
    ("Ä\x83", "ă"),
    ("Ä‚", "Ă"),
    ("Ä\x82", "Ă"),
    ("Ã¢", "â"),
    ("Ã\xa2", "â"),
    ("Ã‚", "Â"),
    ("Ã\x82", "Â"),
    ("Ã®", "î"),
    ("Ã\xae", "î"),
    ("ÃŽ", "Î"),
    ("È™", "ș"),
    ("È\x99", "ș"),
    ("È˜", "Ș"),
    ("È\x98", "Ș"),
    ("ÅŸ", "ș"),
    ("Åž", "Ș"),
    ("È›", "ț"),
    ("È\x9b", "ț"),
    ("Èš", "Ț"),
    ("È\x9a", "Ț"),
    ("Å£", "ț"),
    ("Å¢", "Ț"),
)
ROMANIAN_TRUNCATED_WORD_REPAIRS = (
    (re.compile(r"\bmuncÄ(?=[\W]|$)"), "muncă"),
    (re.compile(r"\bpÄr(?:È(?:›)?|Å£)ilor\b"), "părților"),
    (re.compile(r"\bpÄr(?:È(?:›)?|Å£)i\b"), "părți"),
)
ROMANIAN_MOJIBAKE_DETECTION_RE = re.compile(
    r"(Äƒ|Ä\x83|Ä‚|Ä\x82|Ã¢|Ã\xa2|Ã‚|Ã\x82|Ã®|Ã\xae|ÃŽ|È™|È\x99|È˜|È\x98|È›|È\x9b|Èš|È\x9a|ÅŸ|Åž|Å£|Å¢|\bmuncÄ\b|\bpÄrÈ|\bpÄ)",
)
ORPHAN_ARTIFACT_REPAIRS = (
    re.compile(r"(?<=\s)Â(?=\s)"),
    re.compile(r"(?<=\s)Â(?=[,.;:!?])"),
)


def contains_romanian_mojibake(text: str | None) -> bool:
    if not text:
        return False
    return bool(ROMANIAN_MOJIBAKE_DETECTION_RE.search(text))


def repair_romanian_mojibake(text: str | None) -> str | None:
    if text is None or text == "":
        return text

    repaired = str(text).replace("\u00a0", " ")
    original = repaired

    for broken, fixed in ROMANIAN_MOJIBAKE_REPLACEMENTS:
        repaired = repaired.replace(broken, fixed)
    for pattern, replacement in ROMANIAN_TRUNCATED_WORD_REPAIRS:
        repaired = pattern.sub(replacement, repaired)

    if contains_romanian_mojibake(repaired):
        repaired = _repair_romanian_roundtrip_mojibake(repaired)
        for broken, fixed in ROMANIAN_MOJIBAKE_REPLACEMENTS:
            repaired = repaired.replace(broken, fixed)
        for pattern, replacement in ROMANIAN_TRUNCATED_WORD_REPAIRS:
            repaired = pattern.sub(replacement, repaired)

    for pattern in ORPHAN_ARTIFACT_REPAIRS:
        repaired = pattern.sub("", repaired)

    if repaired == original:
        return original
    return repaired


def normalize_legal_text(raw_text: str | None) -> str | None:
    """Derive retrieval-friendly text without replacing the source raw_text."""
    if raw_text is None:
        return None

    normalized = unicodedata.normalize("NFC", repair_romanian_mojibake(raw_text) or "")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized or None


def _repair_romanian_roundtrip_mojibake(text: str) -> str:
    original_score = _romanian_mojibake_score(text)
    best = text
    best_score = original_score
    for encoding in ("cp1252", "latin-1"):
        try:
            candidate = text.encode(encoding).decode("utf-8")
        except UnicodeError:
            continue
        candidate_score = _romanian_mojibake_score(candidate)
        if candidate_score < best_score:
            best = candidate
            best_score = candidate_score
    return best


def _romanian_mojibake_score(text: str) -> int:
    return sum(text.count(marker) for marker in ROMANIAN_MOJIBAKE_MARKERS)
