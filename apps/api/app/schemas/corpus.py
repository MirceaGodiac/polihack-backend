from pydantic import BaseModel
from typing import List, Optional

class CorpusBase(BaseModel):
    id: str
    title: str
    source_url: Optional[str] = None

class CorpusCreate(CorpusBase):
    pass

class Corpus(CorpusBase):
    unit_count: int = 0
    
    class Config:
        from_attributes = True

class CorpusList(BaseModel):
    items: List[Corpus]
