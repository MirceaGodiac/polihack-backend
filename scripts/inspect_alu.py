import argparse
import json
import sys
from pathlib import Path

def inspect_alu(corpus_dir: Path, unit_id: str):
    units_file = corpus_dir / "legal_units.json"
    edges_file = corpus_dir / "legal_edges.json"
    candidates_file = corpus_dir / "reference_candidates.json"

    if not units_file.exists():
        print(f"Error: {units_file} not found.")
        return

    print(f"Loading units from {units_file}...")
    with open(units_file, "r", encoding="utf-8") as f:
        units = json.load(f)

    unit = next((u for u in units if u["id"] == unit_id), None)
    if not unit:
        print(f"Error: Unit '{unit_id}' not found in the corpus.")
        available_ids = [u["id"] for u in units[:5]]
        print(f"Sample available IDs: {available_ids}")
        return

    # Load Edges
    edges = []
    if edges_file.exists():
        with open(edges_file, "r", encoding="utf-8") as f:
            edges = json.load(f)

    # Load Candidates
    candidates = []
    if candidates_file.exists():
        with open(candidates_file, "r", encoding="utf-8") as f:
            candidates = json.load(f)

    incoming_edges = [e for e in edges if e["target_id"] == unit_id]
    outgoing_edges = [e for e in edges if e["source_id"] == unit_id]
    unit_candidates = [c for c in candidates if c.get("source_unit_id") == unit_id]

    # Print Summary
    print("\n" + "="*80)
    print(f" ALU DEBUG INSPECTOR: {unit['id']}")
    print("="*80)
    print(f" Law/Corpus ID: {unit.get('corpus_id', 'N/A')}")
    print(f" Unit Type:     {unit['type']}")
    print(f" Hierarchy:     {' > '.join(unit.get('hierarchy_path', []))}")
    print("-" * 80)
    print(" RAW TEXT:")
    print(unit['raw_text'])
    print("-" * 80)
    
    print(f" EDGES (Total: {len(incoming_edges) + len(outgoing_edges)})")
    if not incoming_edges and not outgoing_edges:
        print("  (No edges found)")
    else:
        for e in outgoing_edges:
            print(f"  [OUT] ({e['type']}) -> {e['target_id']}")
        for e in incoming_edges:
            print(f"  [IN ] ({e['type']}) <- {e['source_id']}")

    print("-" * 80)
    print(f" REFERENCE CANDIDATES (Total: {len(unit_candidates)})")
    if not unit_candidates:
        print("  (No reference candidates extracted)")
    else:
        for c in unit_candidates:
            status = "RESOLVED" if c.get("target_id") else "UNRESOLVED"
            target = c.get("target_id", "???")
            print(f"  [{status:10}] '{c['raw_reference']}' -> {target}")
    
    print("="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Debug CLI to inspect a specific Legal Unit (ALU)")
    parser.add_argument("--dir", required=True, help="Path to the corpus output directory (e.g. ingestion/output/demo_corpus_v1)")
    parser.add_argument("--id", required=True, help="Target unit_id to inspect")
    args = parser.parse_args()

    corpus_dir = Path(args.dir)
    if not corpus_dir.is_dir():
        print(f"Error: {args.dir} is not a directory.")
        sys.exit(1)

    inspect_alu(corpus_dir, args.id)

if __name__ == "__main__":
    main()
