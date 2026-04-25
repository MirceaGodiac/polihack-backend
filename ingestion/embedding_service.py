from typing import Protocol, List
import random

class EmbeddingProvider(Protocol):
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        ...

class MockEmbeddingProvider:
    def __init__(self, dimension: int = 1024):
        self.dimension = dimension

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        # Returns random float arrays of dimension 1024
        return [[random.random() for _ in range(self.dimension)] for _ in texts]

def build_embedding_text(unit: dict, corpus_metadata: dict = None) -> str:
    """
    Concatenate the domain, law title, hierarchy path, and raw text into a single string.
    corpus_metadata should map corpus_id to dict with 'law_title' and 'legal_domain'.
    """
    corpus_id = unit.get("corpus_id", "")
    metadata = {}
    if corpus_metadata and corpus_id in corpus_metadata:
        metadata = corpus_metadata[corpus_id]
        
    domain = metadata.get("legal_domain", "Unknown Domain")
    law_title = metadata.get("law_title", "Unknown Law")
    hierarchy = " -> ".join(unit.get("hierarchy_path", []))
    raw_text = unit.get("raw_text", "")
    
    return f"Domain: {domain}\nLaw: {law_title}\nPath: {hierarchy}\n\n{raw_text}"
