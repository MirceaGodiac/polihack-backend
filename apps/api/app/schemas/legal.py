from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ReferenceSchema(BaseModel):
    raw_reference: str
    target_article: Optional[str] = None
    target_paragraph: Optional[str] = None
    target_letter: Optional[str] = None
    target_id: Optional[str] = None

class LegalUnitBase(BaseModel):
    id: str
    path: str
    type: str
    text: str
    references: List[ReferenceSchema] = []

class LegalUnit(LegalUnitBase):
    metadata: Dict[str, Any] = {}

    class Config:
        from_attributes = True

class LegalUnitList(BaseModel):
    units: List[LegalUnit]
    total: int
