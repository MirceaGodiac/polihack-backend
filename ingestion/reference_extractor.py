from __future__ import annotations

import re
import unicodedata
from typing import Any, Mapping

from ingestion.contracts import ReferenceCandidate
from ingestion.legal_ids import make_law_id, normalize_number


ARTICLE_NUMBER = r"\d+(?:\^\d+|[\u2070\u00b9\u00b2\u00b3\u2074-\u2079]+)?"
LETTER = r"[a-zA-Z]"
ROMAN = r"[IVXLCDM]+"

REF_ART_RE = re.compile(
    rf"\b(?:art\.?|articolul)\s*({ARTICLE_NUMBER})",
    re.IGNORECASE,
)
REF_ALIN_RE = re.compile(
    r"\b(?:alin\.?|alineatul)\s*\(?(\d+)\)?",
    re.IGNORECASE,
)
REF_LOCAL_RE = re.compile(
    r"\b("
    r"prezentul\s+cod|"
    r"prezenta\s+lege|"
    r"prezentul\s+act\s+normativ|"
    r"prezenta\s+ordonan(?:t|ț|ţ)(?:a|ă)|"
    r"prezenta\s+hot(?:a|ă)r(?:a|â)re"
    r")\b",
    re.IGNORECASE,
)

REF_COMPOUND_RE = re.compile(
    rf"\b(?:art\.?|articolul)\s*(?P<article>{ARTICLE_NUMBER})"
    r"(?:\s*,?\s*(?:alin\.?|alineatul)\s*\(?(?P<paragraph>\d+)\)?)?"
    rf"(?:\s*,?\s*(?:lit\.?|litera)\s*(?P<letter>{LETTER})\))?"
    r"(?:\s*,?\s*(?:pct\.?|punctul)\s*(?P<point>\d+))?"
    rf"(?:\s*,?\s*(?P<thesis>teza\s+a\s+(?P<thesis_value>{ROMAN})-a))?",
    re.IGNORECASE,
)
REF_LETTER_RE = re.compile(
    rf"\b(?:lit\.?|litera)\s*({LETTER})\)",
    re.IGNORECASE,
)
REF_POINT_RE = re.compile(
    r"\b(?:pct\.?|punctul)\s*(\d+)\b",
    re.IGNORECASE,
)
REF_THESIS_RE = re.compile(
    rf"\bteza\s+a\s+({ROMAN})-a\b",
    re.IGNORECASE,
)
REF_NUMBERED_ACT_RE = re.compile(
    r"\b(?P<label>"
    r"Legea|"
    r"O\.?\s*U\.?\s*G\.?|"
    r"O\.?\s*G\.?|"
    r"H\.?\s*G\.?"
    r")\s+(?:nr\.?\s*)?(?P<number>\d+)\s*/\s*(?P<year>\d{4})\b",
    re.IGNORECASE,
)
REF_CODE_RE = re.compile(
    r"\b("
    r"Codul\s+de\s+procedur(?:a|ă)\s+civil(?:a|ă)|"
    r"Codul\s+de\s+procedur(?:a|ă)\s+penal(?:a|ă)|"
    r"Codul\s+muncii|"
    r"Codul\s+civil|"
    r"Codul\s+fiscal|"
    r"Codul\s+penal"
    r")\b",
    re.IGNORECASE,
)


def extract_references(unit: Mapping[str, Any]) -> list[dict[str, Any]]:
    source_unit_id = str(unit.get("id") or "")
    raw_text = str(unit.get("raw_text") or "")
    if not raw_text.strip():
        return []
    return extract_references_from_text(raw_text, source_unit_id=source_unit_id)


def extract_references_from_units(units: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for unit in units:
        candidates.extend(extract_references(unit))
    return sorted(candidates, key=_candidate_sort_key)


def extract_references_from_text(
    raw_text: str,
    *,
    source_unit_id: str,
) -> list[dict[str, Any]]:
    candidates: list[tuple[int, dict[str, Any]]] = []
    structural_spans: list[tuple[int, int]] = []

    for match in REF_COMPOUND_RE.finditer(raw_text):
        if _is_self_article_heading(raw_text, match, source_unit_id):
            continue
        candidate = _compound_candidate(raw_text, match, source_unit_id)
        candidates.append((match.start(), candidate))
        structural_spans.append(match.span())

        sibling_letter = _sibling_letter_candidate(raw_text, match, source_unit_id)
        if sibling_letter is not None:
            candidates.append((match.start(), sibling_letter))

    for match in REF_ALIN_RE.finditer(raw_text):
        if _overlaps(match.span(), structural_spans):
            continue
        candidates.append(
            (
                match.start(),
                _candidate(
                    source_unit_id=source_unit_id,
                    raw_reference=match.group(0),
                    reference_type="paragraph",
                    target_law_hint=_law_hint_near(raw_text, match) or "same_act",
                    target_paragraph=normalize_number(match.group(1)),
                    resolution_status="unresolved_needs_context",
                    resolver_notes=["paragraph_reference_without_article"],
                ),
            )
        )

    for regex, reference_type, field_name in (
        (REF_LETTER_RE, "letter", "target_letter"),
        (REF_POINT_RE, "point", "target_point"),
        (REF_THESIS_RE, "thesis", "target_thesis"),
    ):
        for match in regex.finditer(raw_text):
            if _overlaps(match.span(), structural_spans):
                continue
            value = _normalize_field(reference_type, match.group(1))
            candidates.append(
                (
                    match.start(),
                    _candidate(
                        source_unit_id=source_unit_id,
                        raw_reference=match.group(0),
                        reference_type=reference_type,
                        target_law_hint=_law_hint_near(raw_text, match) or "same_act",
                        resolution_status="unresolved_needs_context",
                        resolver_notes=[f"{reference_type}_reference_without_article"],
                        **{field_name: value},
                    ),
                )
            )

    for match in REF_NUMBERED_ACT_RE.finditer(raw_text):
        if _is_whole_text_reference(raw_text, match):
            continue
        candidates.append(
            (
                match.start(),
                _candidate(
                    source_unit_id=source_unit_id,
                    raw_reference=match.group(0),
                    reference_type=_numbered_reference_type(match.group("label")),
                    target_law_hint=_numbered_law_hint(
                        match.group("label"),
                        match.group("number"),
                        match.group("year"),
                    ),
                    resolver_notes=["act_reference_candidate_only"],
                ),
            )
        )

    for match in REF_CODE_RE.finditer(raw_text):
        if _is_whole_text_reference(raw_text, match):
            continue
        candidates.append(
            (
                match.start(),
                _candidate(
                    source_unit_id=source_unit_id,
                    raw_reference=match.group(0),
                    reference_type="code",
                    target_law_hint=_code_law_hint(match.group(0)),
                    resolver_notes=["code_reference_candidate_only"],
                ),
            )
        )

    for match in REF_LOCAL_RE.finditer(raw_text):
        candidates.append(
            (
                match.start(),
                _candidate(
                    source_unit_id=source_unit_id,
                    raw_reference=match.group(0),
                    reference_type="same_act",
                    target_law_hint="same_act",
                    resolution_status="unresolved_needs_context",
                    resolver_notes=["same_act_reference_requires_context"],
                ),
            )
        )

    return _dedupe_in_text_order(candidates)


def _compound_candidate(
    raw_text: str,
    match: re.Match[str],
    source_unit_id: str,
) -> dict[str, Any]:
    has_child = any(
        match.group(name)
        for name in ("paragraph", "letter", "point", "thesis_value")
    )
    return _candidate(
        source_unit_id=source_unit_id,
        raw_reference=match.group(0),
        reference_type="compound" if has_child else "article",
        target_law_hint=_law_hint_near(raw_text, match) or "same_act",
        target_article=normalize_number(match.group("article")),
        target_paragraph=normalize_number(match.group("paragraph"))
        if match.group("paragraph")
        else None,
        target_letter=normalize_number(match.group("letter"))
        if match.group("letter")
        else None,
        target_point=normalize_number(match.group("point"))
        if match.group("point")
        else None,
        target_thesis=_normalize_thesis(match.group("thesis_value")),
        resolver_notes=["compound_reference_candidate_only"]
        if has_child
        else ["article_reference_candidate_only"],
    )


def _sibling_letter_candidate(
    raw_text: str,
    match: re.Match[str],
    source_unit_id: str,
) -> dict[str, Any] | None:
    if not match.group("letter"):
        return None
    tail = raw_text[match.end() : match.end() + 12]
    tail_match = re.match(r"\s*(?:și|si)\s*([a-zA-Z])\)", tail, re.IGNORECASE)
    if not tail_match:
        return None
    return _candidate(
        source_unit_id=source_unit_id,
        raw_reference=f"{match.group(0)}{tail_match.group(0)}",
        reference_type="compound",
        target_law_hint=_law_hint_near(raw_text, match) or "same_act",
        target_article=normalize_number(match.group("article")),
        target_paragraph=normalize_number(match.group("paragraph"))
        if match.group("paragraph")
        else None,
        target_letter=normalize_number(tail_match.group(1)),
        target_point=normalize_number(match.group("point"))
        if match.group("point")
        else None,
        target_thesis=_normalize_thesis(match.group("thesis_value")),
        resolver_notes=["compound_sibling_letter_reference_candidate_only"],
    )


def _candidate(
    *,
    source_unit_id: str,
    raw_reference: str,
    reference_type: str,
    target_law_hint: str | None = None,
    target_article: str | None = None,
    target_paragraph: str | None = None,
    target_letter: str | None = None,
    target_point: str | None = None,
    target_thesis: str | None = None,
    resolution_status: str = "candidate_only",
    resolver_notes: list[str] | None = None,
) -> dict[str, Any]:
    return ReferenceCandidate(
        source_unit_id=source_unit_id,
        raw_reference=raw_reference,
        reference_type=reference_type,
        target_law_hint=target_law_hint,
        target_article=target_article,
        target_paragraph=target_paragraph,
        target_letter=target_letter,
        target_point=target_point,
        target_thesis=target_thesis,
        resolved_target_id=None,
        resolution_status=resolution_status,
        resolution_confidence=0.0,
        resolver_notes=resolver_notes or ["reference_resolution_deferred_to_p7"],
    ).model_dump()


def _law_hint_near(raw_text: str, match: re.Match[str]) -> str | None:
    context = raw_text[max(0, match.start() - 80) : min(len(raw_text), match.end() + 120)]
    same_act = REF_LOCAL_RE.search(context)
    if same_act:
        return "same_act"
    code = REF_CODE_RE.search(context)
    if code:
        return _code_law_hint(code.group(0))
    numbered = REF_NUMBERED_ACT_RE.search(context)
    if numbered:
        return _numbered_law_hint(
            numbered.group("label"),
            numbered.group("number"),
            numbered.group("year"),
        )
    return None


def _numbered_reference_type(label: str) -> str:
    normalized = _normalize_ascii(label)
    if "u" in normalized and normalized.replace(" ", "").replace(".", "") == "oug":
        return "oug"
    compact = normalized.replace(" ", "").replace(".", "")
    if compact == "og":
        return "og"
    if compact == "hg":
        return "hg"
    return "law"


def _numbered_law_hint(label: str, number: str, year: str) -> str:
    prefix = _numbered_reference_type(label)
    if prefix == "law":
        prefix = "lege"
    return f"ro.{prefix}_{normalize_number(number)}_{normalize_number(year)}"


def _code_law_hint(raw_reference: str) -> str:
    normalized = _normalize_ascii(raw_reference)
    mapping = {
        "codul muncii": "ro.codul_muncii",
        "codul civil": "ro.codul_civil",
        "codul fiscal": "ro.codul_fiscal",
        "codul penal": "ro.codul_penal",
        "codul de procedura civila": "ro.codul_de_procedura_civila",
        "codul de procedura penala": "ro.codul_de_procedura_penala",
    }
    return mapping.get(normalized, make_law_id(raw_reference))


def _normalize_field(reference_type: str, value: str) -> str:
    if reference_type == "thesis":
        return _normalize_thesis(value) or value.upper()
    return normalize_number(value)


def _normalize_thesis(value: str | None) -> str | None:
    if not value:
        return None
    return value.upper()


def _is_self_article_heading(
    raw_text: str,
    match: re.Match[str],
    source_unit_id: str,
) -> bool:
    line_start = raw_text.rfind("\n", 0, match.start()) + 1
    prefix = raw_text[line_start : match.start()]
    if prefix.strip():
        return False
    article = normalize_number(match.group("article"))
    return source_unit_id.endswith(f".art_{article}") or f".art_{article}." in source_unit_id


def _is_whole_text_reference(raw_text: str, match: re.Match[str]) -> bool:
    return raw_text.strip() == match.group(0).strip()


def _overlaps(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(start < existing_end and end > existing_start for existing_start, existing_end in spans)


def _dedupe_in_text_order(candidates: list[tuple[int, dict[str, Any]]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    deduped: list[tuple[int, dict[str, Any]]] = []
    for index, candidate in candidates:
        key = _candidate_dedupe_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((index, candidate))
    return [
        candidate
        for _, candidate in sorted(
            deduped,
            key=lambda item: (item[0], _candidate_sort_key(item[1])),
        )
    ]


def _candidate_dedupe_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        candidate.get("source_unit_id"),
        candidate.get("raw_reference"),
        candidate.get("reference_type"),
        candidate.get("target_law_hint"),
        candidate.get("target_article"),
        candidate.get("target_paragraph"),
        candidate.get("target_letter"),
        candidate.get("target_point"),
        candidate.get("target_thesis"),
    )


def _candidate_sort_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        candidate.get("source_unit_id") or "",
        candidate.get("raw_reference") or "",
        candidate.get("reference_type") or "",
        candidate.get("target_article") or "",
        candidate.get("target_paragraph") or "",
        candidate.get("target_letter") or "",
        candidate.get("target_point") or "",
        candidate.get("target_thesis") or "",
    )


def _normalize_ascii(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value.casefold())
    stripped = "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    )
    return " ".join(stripped.replace(".", " ").split())
