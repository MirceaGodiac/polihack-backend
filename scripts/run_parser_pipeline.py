from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ingestion.exporters import export_canonical_bundle
from ingestion.html_cleaner import clean_html_to_lines
from ingestion.legal_ids import make_law_id
from ingestion.parser.html_parser import extract_metadata
from ingestion.parser.html_scraper import scrape_html_source
from ingestion.structural_parser import StructuralParser


DEFAULT_OUTPUT_DIR = Path("ingestion/output/legislatie_just_ro_bundle")
LEGISLATIE_HOST = "legislatie.just.ro"


@dataclass(frozen=True)
class ParserPipelineResult:
    artifact_paths: dict[str, Path]
    law_id: str
    law_title: str
    legacy_units_count: int
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
    Fetch a legislatie.just.ro page and export a canonical parser bundle.

    This is a file-based integration pipeline: it does not write to DB, does
    not generate vectors, and does not treat retrieval text as citable law.
    """
    _validate_legislatie_url(url)

    html = scrape_html_source(url)
    if not html:
        raise RuntimeError(f"Could not fetch HTML from {url}")

    page_metadata = extract_metadata(html)
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
    legacy_units, _legacy_edges = structural_parser.parse(clean_result.lines)
    if not legacy_units:
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
        "source_type": "legislatie.just.ro_url",
        "source_url": url,
        "source_id": source_id,
    }
    artifact_paths = export_canonical_bundle(
        legacy_units,
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
            legacy_units=legacy_units,
        )

    validation_report = _load_json(artifact_paths["validation_report"])
    return ParserPipelineResult(
        artifact_paths=artifact_paths,
        law_id=resolved_law_id,
        law_title=resolved_law_title,
        legacy_units_count=len(legacy_units),
        cleaner_warnings=clean_result.warnings,
        import_blocking_passed=bool(validation_report.get("import_blocking_passed")),
    )


def _validate_legislatie_url(url: str) -> None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("URL must use http or https")
    if host != LEGISLATIE_HOST and not host.endswith(f".{LEGISLATIE_HOST}"):
        raise ValueError("URL must point to legislatie.just.ro")


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

    document_id = _extract_legislatie_document_id(url)
    if document_id:
        return f"ro.legislatie_document_{document_id}", "law_id_inferred_from_url_document_id"

    return "ro.legislatie_unknown", "law_id_unknown_fallback"


def _extract_legislatie_document_id(url: str) -> str | None:
    path = urlparse(url).path
    match = re.search(r"/(?:detaliidocument|detaliidocumentafis)/(\d+)", path, re.IGNORECASE)
    return match.group(1) if match else None


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
    legacy_units: list[dict],
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
    (output_dir / "legacy_units.json").write_text(
        json.dumps(legacy_units, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the canonical LexAI parser pipeline for a legislatie.just.ro URL",
    )
    parser.add_argument("url", nargs="?", help="Source URL from legislatie.just.ro")
    parser.add_argument("--url", dest="url_option", default=None, help="Source URL from legislatie.just.ro")
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for the canonical bundle",
    )
    parser.add_argument("--law-id", default=None, help="Optional canonical law id")
    parser.add_argument("--law-title", default=None, help="Optional legal act title")
    parser.add_argument("--source-id", default=None, help="Optional upstream source id")
    parser.add_argument(
        "--status",
        default="unknown",
        choices=["active", "historical", "repealed", "unknown"],
    )
    parser.add_argument("--parser-version", default="0.1.0")
    parser.add_argument("--generated-at", default=None)
    parser.add_argument(
        "--write-debug",
        action="store_true",
        help="Write cleaned lines, cleaner report, and legacy units next to the bundle",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    url = args.url_option or args.url
    if not url:
        parser.error("a legislatie.just.ro URL is required")
    try:
        result = run_pipeline(
            url=url,
            out_dir=args.out_dir,
            law_id=args.law_id,
            law_title=args.law_title,
            source_id=args.source_id,
            status=args.status,
            parser_version=args.parser_version,
            generated_at=args.generated_at,
            write_debug=args.write_debug,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"[parser-pipeline] {exc}", file=sys.stderr)
        sys.exit(2)

    print(f"law_id: {result.law_id}")
    print(f"law_title: {result.law_title}")
    print(f"legacy_units_count: {result.legacy_units_count}")
    if result.cleaner_warnings:
        print(f"cleaner_warnings: {', '.join(result.cleaner_warnings)}")
    for artifact, path in result.artifact_paths.items():
        print(f"{artifact}: {path}")

    if not result.import_blocking_passed:
        print(
            "import_blocking_passed=false; canonical bundle is not import-ready",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
