import argparse
import json
import hashlib
from pathlib import Path
import sys

# Allow running as `python scripts/generate_embeddings.py` from the repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ingestion.embedding_service import MockEmbeddingProvider, build_embedding_text

def get_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for legal units")
    parser.add_argument("--input", required=True, help="Path to legal_units.json")
    parser.add_argument("--out", required=True, help="Path to output embeddings.import.jsonl")
    parser.add_argument("--model", required=True, help="Model name")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    out_path = Path(args.out)
    
    # Try to load manifest for metadata
    manifest_path = input_path.parent / "corpus_manifest.json"
    corpus_metadata = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
            for source in manifest.get("sources", []):
                corpus_metadata[source["law_id"]] = {
                    "law_title": source.get("law_title", ""),
                    "legal_domain": source.get("legal_domain", "")
                }
    else:
        print(f"Warning: Manifest not found at {manifest_path}. Metadata will be missing.")
    
    with open(input_path, "r", encoding="utf-8") as f:
        units = json.load(f)
        
    provider = MockEmbeddingProvider(dimension=1024)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    batch_size = 100
    
    with open(out_path, "w", encoding="utf-8") as out_f:
        for i in range(0, len(units), batch_size):
            batch_units = units[i:i+batch_size]
            texts = [build_embedding_text(u, corpus_metadata) for u in batch_units]
            embeddings = provider.embed_texts(texts)
            
            for j, unit in enumerate(batch_units):
                text_hash = get_text_hash(texts[j])
                record = {
                    "unit_id": unit["id"],
                    "model_name": args.model,
                    "embedding_dim": 1024,
                    "text_hash": text_hash,
                    "embedding": embeddings[j]
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
    print(f"Generated {len(units)} embeddings to {out_path}")

if __name__ == "__main__":
    main()
