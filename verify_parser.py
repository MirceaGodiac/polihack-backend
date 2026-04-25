from ingestion.structural_parser import StructuralParser
from ingestion.validators import validate_corpus
import json

def verify():
    parser = StructuralParser("ro.codul_muncii")
    mock_text = [
        "TITLUL I",
        "Dispoziţii generale",
        "CAPITOLUL I",
        "Domeniul de aplicare",
        "Art. 1",
        "Prezentul cod reglementează domeniul raporturilor de muncă.",
        "Art. 2",
        "Dispoziţiile prezentului cod se aplică:",
        "a) cetăţenilor români cu domiciliul în România;",
        "b) cetăţenilor străini sau apatrizilor."
    ]
    
    units, edges = parser.parse(mock_text)
    
    # Validate
    validate_corpus(units)
    
    # Save
    parser.save("legal_units.json", "legal_edges.json")
    
    print(f"Generated {len(units)} units and {len(edges)} edges.")
    print("\nSample Units:")
    for unit in units[:5]:
        print(f"ID: {unit['id']} | Type: {unit['type']}")
    
    print("\nSample Edges:")
    for edge in edges[:5]:
        print(f"{edge['source_id']} -> {edge['target_id']}")

if __name__ == "__main__":
    verify()
