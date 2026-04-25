from fastapi import APIRouter, HTTPException
from typing import List
from pathlib import Path
import json
from ..schemas.legal import LegalUnit, LegalUnitList

router = APIRouter(prefix="/legal-units", tags=["legal-units"])

OUTPUT_BASE = Path("ingestion/output")

@router.get("/{corpus_id}", response_model=LegalUnitList)
async def get_legal_units(corpus_id: str, skip: int = 0, limit: int = 100):
    """
    Get the atomic units for a specific corpus.
    """
    units_file = OUTPUT_BASE / corpus_id / "legal_units.json"
    if not units_file.exists():
        raise HTTPException(status_code=404, detail="Corpus not found")

    try:
        with open(units_file, "r", encoding="utf-8") as f:
            all_units = json.load(f)
        
        # Paginate
        paginated = all_units[skip : skip + limit]
        
        return {
            "units": paginated,
            "total": len(all_units)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading units: {str(e)}")
