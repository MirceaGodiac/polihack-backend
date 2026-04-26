from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ingestion.normalizer import contains_romanian_mojibake


DEFAULT_BUNDLE = (
    REPO_ROOT
    / "ingestion"
    / "output"
    / "codul_muncii"
    / "legal_units.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report legal units whose raw_text still contains Romanian mojibake.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_BUNDLE),
        help="Path to legal_units.json or to a bundle directory containing it.",
    )
    args = parser.parse_args()

    units_path = _resolve_units_path(Path(args.path))
    units = json.loads(units_path.read_text(encoding="utf-8"))
    if not isinstance(units, list):
        raise ValueError("legal_units payload must be a JSON array")

    broken_units: list[dict[str, str]] = []
    for unit in units:
        if not isinstance(unit, dict):
            continue
        raw_text = str(unit.get("raw_text") or "")
        if not contains_romanian_mojibake(raw_text):
            continue
        broken_units.append(
            {
                "unit_id": str(unit.get("id") or ""),
                "law_id": str(unit.get("law_id") or ""),
                "snippet": _snippet(raw_text),
            }
        )

    if not broken_units:
        print(f"OK: no Romanian mojibake found in {units_path}")
        return 0

    print(f"FOUND {len(broken_units)} unit(s) with Romanian mojibake in {units_path}")
    for row in broken_units:
        print(f"- {row['unit_id']} | law_id={row['law_id']} | {row['snippet']}")
    return 1


def _resolve_units_path(path: Path) -> Path:
    if path.is_dir():
        path = path / "legal_units.json"
    if not path.is_file():
        raise FileNotFoundError(f"legal_units.json not found at {path}")
    return path


def _snippet(raw_text: str, limit: int = 120) -> str:
    compact = " ".join(raw_text.split())
    return compact if len(compact) <= limit else compact[: limit - 3].rstrip() + "..."


if __name__ == "__main__":
    sys.exit(main())
