import pytest
from ingestion.embedding_service import build_embedding_text, MockEmbeddingProvider

def test_build_embedding_text():
    unit = {
        "id": "ro.codul_muncii.art_1",
        "type": "articol",
        "raw_text": "Art. 1\nMunca este libera.",
        "hierarchy_path": ["I", "1"],
        "corpus_id": "ro.codul_muncii"
    }
    corpus_metadata = {
        "ro.codul_muncii": {
            "law_title": "Codul Muncii",
            "legal_domain": "dreptul_muncii"
        }
    }
    
    text = build_embedding_text(unit, corpus_metadata)
    
    assert "Domain: dreptul_muncii" in text
    assert "Law: Codul Muncii" in text
    assert "Path: I -> 1" in text
    assert "Munca este libera." in text

def test_mock_provider():
    provider = MockEmbeddingProvider(dimension=1024)
    embeddings = provider.embed_texts(["test text 1", "test text 2"])
    
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 1024
    assert isinstance(embeddings[0][0], float)
