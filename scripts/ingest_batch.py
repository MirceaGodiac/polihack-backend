"""
scripts/ingest_batch.py
-----------------------
Batch ingestion orchestrator for the AntiGravity pipeline.

Usage:
    python scripts/ingest_batch.py \
        --manifest ingestion/sources/demo_sources.yaml \
        --out-dir  ingestion/output/demo_corpus_v1

For each source in the YAML:
  1. Read the local fixture .txt file.
  2. Parse it through StructuralParser.
  3. Extract references from every unit.
  4. Resolve intra-act references.
  5. Aggregate units + edges across all sources.
  6. Run the corpus validator.
  7. Emit legal_units.json, legal_edges.json, validation_report.json,
     and corpus_manifest.json into --out-dir.
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running as `python scripts/ingest_batch.py` from the repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import yaml

from ingestion.structural_parser import StructuralParser
from ingestion.reference_extractor import extract_references
from ingestion.reference_resolver import resolve_references
from ingestion.validators import build_validation_report, save_validation_report, validate_corpus
from ingestion.manifest import build_manifest, save_manifest


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_fixture(fixture_path: str, base_dir: Path) -> list[str]:
    """Read a local .txt fixture and return its lines."""
    p = Path(fixture_path)
    if not p.is_absolute():
        p = base_dir / p
    with open(p, "r", encoding="utf-8") as f:
        return f.readlines()


def process_source(source: dict, base_dir: Path) -> tuple[list, list, list, list]:
    """
    Process a single source entry.

    Returns:
        units, contains_edges, ref_candidates, ref_edges
    """
    law_id = source["law_id"]
    fixture = source["fixture_file"]

    lines = read_fixture(fixture, base_dir)

    # --- Structural parsing ---
    parser = StructuralParser(corpus_id=law_id)
    units, contains_edges = parser.parse(lines)

    # --- Reference extraction ---
    all_candidates = []
    for unit in units:
        candidates = extract_references(unit)
        all_candidates.extend(candidates)

    # --- Reference resolution (intra-act) ---
    resolved_candidates, ref_edges = resolve_references(all_candidates, units)

    print(f"  [{law_id}] {len(units)} units | "
          f"{len(contains_edges)} contains-edges | "
          f"{len(all_candidates)} ref-candidates | "
          f"{len(ref_edges)} ref-edges resolved")

    return units, contains_edges, resolved_candidates, ref_edges


def run(manifest_path: str, out_dir: str) -> None:
    manifest_path = Path(manifest_path)
    out_dir = Path(out_dir)
    base_dir = REPO_ROOT  # fixture paths are relative to repo root

    print(f"\n=== AntiGravity Batch Ingestion ===")
    print(f"Manifest : {manifest_path}")
    print(f"Output   : {out_dir}\n")

    config = load_yaml(manifest_path)
    sources = config.get("sources", [])

    all_units: list         = []
    all_contains_edges: list = []
    all_ref_candidates: list = []
    all_ref_edges: list      = []

    for source in sources:
        print(f"Processing: {source['law_title']} ({source['law_id']})")
        units, contains_edges, ref_candidates, ref_edges = process_source(source, base_dir)
        all_units.extend(units)
        all_contains_edges.extend(contains_edges)
        all_ref_candidates.extend(ref_candidates)
        all_ref_edges.extend(ref_edges)

    # --- Blocking duplicate check ---
    try:
        validate_corpus(all_units)
    except ValueError as exc:
        print(f"\n[FATAL] {exc}")
        sys.exit(1)

    # --- Build validation report ---
    report = build_validation_report(all_units, all_contains_edges, all_ref_candidates)
    print(f"\nValidation status       : {report['status']}")
    print(f"ReferenceResolutionRate : {report['ReferenceResolutionRate']}")
    if report["warnings"]:
        for w in report["warnings"]:
            print(f"  [WARN] {w}")

    # --- Write output files ---
    out_dir.mkdir(parents=True, exist_ok=True)

    all_edges = all_contains_edges + all_ref_edges

    with open(out_dir / "legal_units.json", "w", encoding="utf-8") as f:
        json.dump(all_units, f, indent=2, ensure_ascii=False)

    with open(out_dir / "legal_edges.json", "w", encoding="utf-8") as f:
        json.dump(all_edges, f, indent=2, ensure_ascii=False)

    save_validation_report(report, str(out_dir / "validation_report.json"))

    # Derive batch_id from the out_dir name
    batch_id = out_dir.name
    manifest = build_manifest(batch_id, sources, out_dir)
    save_manifest(manifest, out_dir / "corpus_manifest.json")

    print(f"\nOutput bundle written to: {out_dir.resolve()}")
    print("  legal_units.json")
    print("  legal_edges.json")
    print("  validation_report.json")
    print("  corpus_manifest.json")
    print("\n=== Done ===\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="AntiGravity batch ingestion pipeline")
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to the YAML sources manifest (e.g. ingestion/sources/demo_sources.yaml)",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        dest="out_dir",
        help="Output directory for generated JSON files",
    )
    args = parser.parse_args()
    run(args.manifest, args.out_dir)


if __name__ == "__main__":
    main()
