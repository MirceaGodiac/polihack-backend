from __future__ import annotations

import re
import unicodedata

# Common Romanian mojibake mappings (UTF-8 bytes misread as Windows-1252/Latin-1)
_ROMANIAN_MOJIBAKE_MAP: dict[str, str] = {
    "Ä\u0083": "ă",
    "Ã¢": "â",
    "Ã®": "î",
    "È™": "ș",
    "È›": "ț",
    "Ä‚": "Ă",
    "Ã‚": "Â",
    "ÃŽ": "Î",
    "È˜": "Ș",
    "Èš": "Ț",
    "Å£": "ț",
    "Å¢": "Ț",
    "ÅŸ": "ș",
    "Åž": "Ș",
    "ãƒ": "ă",
    "Äƒ": "ă",
}

_MOJIBAKE_RE = re.compile("|".join(re.escape(k) for k in sorted(_ROMANIAN_MOJIBAKE_MAP, key=len, reverse=True)))


def repair_romanian_mojibake(text: str | None) -> str | None:
    """Replace common Romanian mojibake sequences with correct diacritics."""
    if text is None:
        return None
    repaired = _MOJIBAKE_RE.sub(lambda m: _ROMANIAN_MOJIBAKE_MAP[m.group()], text)
    return unicodedata.normalize("NFC", repaired) or None


def normalize_legal_text(raw_text: str | None) -> str | None:
    """Derive retrieval-friendly text without replacing the source raw_text."""
    if raw_text is None:
        return None

    normalized = unicodedata.normalize("NFC", raw_text)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized or None
