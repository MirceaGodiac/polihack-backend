from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from ingestion.exporters import export_canonical_bundle
from ingestion.html_cleaner import clean_html_to_lines
from ingestion.html_scraper import scrape_html_source
from ingestion.legal_ids import make_law_id
from ingestion.structural_parser import StructuralParser


DEFAULT_OUTPUT_DIR = Path("ingestion/output/url_parser_bundle")


@dataclass(frozen=True)
class ParserPipelineResult:
    artifact_paths: dict[str, Path]
    law_id: str
    law_title: str
    intermediate_units_count: int
    cleaner_warnings: list[str]
    import_blocking_passed: bool


def run_pipeline(
    *,
    url: str,
    out_dir: str | Path = DEFAULT_OUTPUT_DIR,
    law_id: str | None = None,
    law_title: str | None = None,
    source_id: str | None = None,
    status: str = "unknown",
    parser_version: str = "0.1.0",
    generated_at: str | None = None,
    write_debug: bool = False,
) -> ParserPipelineResult:
    """
    Fetch an HTML URL and export a canonical parser bundle.

    This is a file-based integration pipeline: it does not write to DB, does
    not generate vectors, and does not treat retrieval text as citable law.
    """
    _validate_http_url(url)

    html = scrape_html_source(url)
    if not html:
        raise RuntimeError(f"Could not fetch HTML from {url}")

    page_metadata = _extract_metadata(html)
    clean_result = clean_html_to_lines(html)
    if not clean_result.lines:
        raise RuntimeError("HTML cleaner produced no legal text lines")

    resolved_law_title, title_warning = _resolve_law_title(
        explicit_title=law_title,
        page_metadata=page_metadata,
        cleaned_lines=clean_result.lines,
    )
    resolved_law_id, id_warning = _resolve_law_id(
        explicit_law_id=law_id,
        law_title=resolved_law_title,
        url=url,
    )

    structural_parser = StructuralParser(corpus_id=resolved_law_id)
    intermediate_units, _intermediate_edges = structural_parser.parse(clean_result.lines)
    if not intermediate_units:
        raise RuntimeError("Structural parser produced no legal units")

    output_dir = Path(out_dir)
    generated_at_value = generated_at or datetime.now(timezone.utc).isoformat()
    additional_warnings = [f"html_cleaner:{warning}" for warning in clean_result.warnings]
    if title_warning:
        additional_warnings.append(title_warning)
    if id_warning:
        additional_warnings.append(id_warning)

    metadata = {
        "law_id": resolved_law_id,
        "law_title": resolved_law_title,
        "source_id": source_id,
        "source_url": url,
        "status": status,
    }
    source_descriptor = {
        "law_id": resolved_law_id,
        "law_title": resolved_law_title,
        "source_type": _source_type_from_url(url),
        "source_url": url,
        "source_id": source_id,
    }
    artifact_paths = export_canonical_bundle(
        intermediate_units,
        metadata,
        output_dir,
        parser_version=parser_version,
        generated_at=generated_at_value,
        input_files=[url],
        source_descriptors=[source_descriptor],
        additional_warnings=additional_warnings,
    )

    if write_debug:
        _write_debug_files(
            output_dir,
            clean_lines=clean_result.lines,
            cleaner_report={
                "warnings": clean_result.warnings,
                "removed_blocks_count": clean_result.removed_blocks_count,
                "selected_container": clean_result.selected_container,
                "text_hash": clean_result.text_hash,
            },
            intermediate_units=intermediate_units,
        )

    validation_report = _load_json(artifact_paths["validation_report"])
    return ParserPipelineResult(
        artifact_paths=artifact_paths,
        law_id=resolved_law_id,
        law_title=resolved_law_title,
        intermediate_units_count=len(intermediate_units),
        cleaner_warnings=clean_result.warnings,
        import_blocking_passed=bool(validation_report.get("import_blocking_passed")),
    )


def _validate_http_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("URL must use http or https")
    if not parsed.hostname:
        raise ValueError("URL must include a host")


def _source_type_from_url(url: str) -> str:
    host = (urlparse(url).hostname or "unknown").lower()
    return f"url:{host}"


def _resolve_law_title(
    *,
    explicit_title: str | None,
    page_metadata: dict,
    cleaned_lines: list[str] | None = None,
) -> tuple[str, str | None]:
    if explicit_title and explicit_title.strip():
        return explicit_title.strip(), None

    title = _clean_html_title(str(page_metadata.get("title") or "").strip())
    if title and not _is_generic_html_title(title):
        return title, "law_title_inferred_from_html_title"

    for line in (cleaned_lines or [])[:20]:
        candidate = _clean_html_title(line.strip())
        if _looks_like_act_title(candidate):
            return candidate, "law_title_inferred_from_cleaned_text"

    return "unknown", "law_title_unknown"


def _resolve_law_id(
    *,
    explicit_law_id: str | None,
    law_title: str,
    url: str,
) -> tuple[str, str | None]:
    if explicit_law_id and explicit_law_id.strip():
        return explicit_law_id.strip(), None

    if law_title and law_title != "unknown":
        return _make_law_id_from_title(law_title), "law_id_inferred_from_law_title"

    document_id = _extract_url_document_id(url)
    if document_id:
        return f"ro.url_document_{document_id}", "law_id_inferred_from_url_document_id"

    return f"ro.url_document_{_url_hash(url)}", "law_id_inferred_from_url_hash"


def _extract_url_document_id(url: str) -> str | None:
    path = urlparse(url).path
    match = re.search(r"/(?:detaliidocument|detaliidocumentafis)/(\d+)", path, re.IGNORECASE)
    if match:
        return match.group(1)
    numeric_segments = re.findall(r"(?:^|/)(\d{4,})(?:/|$)", path)
    return numeric_segments[-1] if numeric_segments else None


def _url_hash(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]


def _extract_metadata(html: str | None) -> dict[str, str]:
    if not html:
        return {}
    soup = BeautifulSoup(html, "html.parser")
    metadata: dict[str, str] = {}
    title = soup.find("title")
    if title:
        metadata["title"] = title.get_text(strip=True)
    description = soup.find("meta", attrs={"name": "description"})
    if description and description.get("content"):
        metadata["description"] = str(description["content"]).strip()
    return metadata


def _make_law_id_from_title(title: str) -> str:
    normalized = _ascii_lower(title)
    match = re.search(
        r"\b(lege|legea|oug|o\.u\.g\.|og|o\.g\.|hg|h\.g\.)\s+"
        r"(?:nr\.?\s*)?(\d+)"
        r"(?:\s*/\s*(\d{4})|\s+din\s+.*?\b((?:19|20)\d{2})\b)",
        normalized,
    )
    if match:
        prefix = _numbered_act_prefix(match.group(1))
        year = match.group(3) or match.group(4)
        return f"ro.{prefix}_{match.group(2)}_{year}"
    return make_law_id(title)


def _numbered_act_prefix(raw_prefix: str) -> str:
    compact = raw_prefix.replace(".", "")
    if compact in {"lege", "legea"}:
        return "lege"
    return compact


def _clean_html_title(title: str) -> str:
    title = re.sub(r"\s*[-|]\s*portal legislativ\s*$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def _is_generic_html_title(title: str) -> bool:
    normalized = _ascii_lower(title)
    return normalized in {
        "portal legislativ",
        "legislatie just",
        "legislatie.just.ro",
        "untitled",
    }


def _looks_like_act_title(text: str) -> bool:
    if not text:
        return False
    normalized = _ascii_lower(text)
    return bool(
        re.match(
            r"^(codul|constitutia|legea?\s+nr\.?|oug|o\.u\.g\.|og|o\.g\.|hg|h\.g\.|"
            r"ordonanta|hotararea)",
            normalized,
        )
    )


def _ascii_lower(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    without_marks = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return without_marks.lower()


def _write_debug_files(
    output_dir: Path,
    *,
    clean_lines: list[str],
    cleaner_report: dict,
    intermediate_units: list[dict],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cleaned_lines.txt").write_text(
        "\n".join(clean_lines) + "\n",
        encoding="utf-8",
    )
    (output_dir / "cleaner_report.json").write_text(
        json.dumps(cleaner_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "intermediate_units.json").write_text(
        json.dumps(intermediate_units, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
