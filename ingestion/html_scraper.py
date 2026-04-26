from __future__ import annotations

import re
from typing import Optional

import requests
from bs4 import BeautifulSoup

from ingestion.normalizer import (
    contains_romanian_mojibake,
    repair_romanian_mojibake,
)


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}
META_CHARSET_RE = re.compile(
    br"charset\s*=\s*['\"]?\s*([A-Za-z0-9._:-]+)",
    re.IGNORECASE,
)
HEADER_CHARSET_RE = re.compile(
    r"charset\s*=\s*['\"]?\s*([^;,\s'\"]+)",
    re.IGNORECASE,
)


def scrape_html_source(url: str) -> Optional[str]:
    """
    Fetch raw HTML source from the given URL.

    This helper only fetches HTML. Canonical parsing happens in
    scripts/run_parser_pipeline.py via html_cleaner + StructuralParser.
    """
    try:
        response = requests.get(url, headers=DEFAULT_HEADERS, timeout=30)
        response.raise_for_status()
        content = getattr(response, "content", None)
        if isinstance(content, bytes | bytearray):
            return decode_html_content(
                bytes(content),
                headers=getattr(response, "headers", None),
            )
        text = getattr(response, "text", None)
        return repair_romanian_mojibake(text) if isinstance(text, str) else None
    except requests.exceptions.RequestException as exc:
        print(f"[scraper] Error fetching URL {url}: {exc}")
        return None


def scrape_with_soup(url: str) -> Optional[BeautifulSoup]:
    html = scrape_html_source(url)
    if html is None:
        return None
    return BeautifulSoup(html, "html.parser")


def decode_html_content(content: bytes, *, headers: object | None = None) -> str:
    if not content:
        return ""

    decoded_candidates: list[tuple[int, int, str]] = []
    for order, encoding in enumerate(_candidate_encodings(content, headers=headers)):
        try:
            decoded = content.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
        repaired = repair_romanian_mojibake(decoded) or ""
        decoded_candidates.append((_decoded_text_score(repaired), order, repaired))

    if decoded_candidates:
        decoded_candidates.sort(key=lambda item: (item[0], item[1]))
        return decoded_candidates[0][2]
    return repair_romanian_mojibake(content.decode("utf-8", errors="replace")) or ""


def _candidate_encodings(content: bytes, *, headers: object | None) -> list[str]:
    candidates = [
        _charset_from_headers(headers),
        _charset_from_meta(content),
        "utf-8",
        "cp1252",
        "latin-1",
    ]
    return _dedupe_encodings(candidates)


def _charset_from_headers(headers: object | None) -> str | None:
    if not headers:
        return None
    content_type = None
    get = getattr(headers, "get", None)
    if callable(get):
        content_type = get("content-type") or get("Content-Type")
    if not content_type:
        return None
    match = HEADER_CHARSET_RE.search(str(content_type))
    return match.group(1).strip() if match else None


def _charset_from_meta(content: bytes) -> str | None:
    match = META_CHARSET_RE.search(content[:4096])
    if not match:
        return None
    return match.group(1).decode("ascii", errors="ignore").strip()


def _dedupe_encodings(encodings: list[str | None]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for encoding in encodings:
        if not encoding:
            continue
        normalized = encoding.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _decoded_text_score(text: str) -> int:
    score = 20 * text.count("\ufffd")
    if contains_romanian_mojibake(text):
        score += 10
    return score
