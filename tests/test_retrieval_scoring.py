import pytest
from apps.api.app.services.retrieval_scoring import (
    ScoreBreakdown, 
    weighted_retrieval_score, 
    reciprocal_rank_fusion, 
    domain_match, 
    exact_citation_boost
)

def test_weighted_retrieval_score_dense():
    s = ScoreBreakdown(
        bm25=1.0, 
        dense=1.0, 
        domain_match=1.0, 
        metadata_validity=1.0, 
        exact_citation_boost=1.0
    )
    # 0.35 + 0.30 + 0.15 + 0.10 + 0.10 = 1.0
    assert weighted_retrieval_score(s, dense_available=True) == pytest.approx(1.0)
    
    s2 = ScoreBreakdown(bm25=0.5, dense=0.8)
    # 0.35*0.5 + 0.30*0.8 = 0.175 + 0.24 = 0.415
    assert weighted_retrieval_score(s2, dense_available=True) == pytest.approx(0.415)

def test_weighted_retrieval_score_no_dense():
    s = ScoreBreakdown(
        bm25=1.0, 
        domain_match=1.0, 
        metadata_validity=1.0, 
        exact_citation_boost=1.0
    )
    # 0.60 + 0.20 + 0.10 + 0.10 = 1.0
    assert weighted_retrieval_score(s, dense_available=False) == pytest.approx(1.0)
    
    s2 = ScoreBreakdown(bm25=0.7, domain_match=0.5)
    # 0.60*0.7 + 0.20*0.5 = 0.42 + 0.10 = 0.52
    assert weighted_retrieval_score(s2, dense_available=False) == pytest.approx(0.52)

def test_reciprocal_rank_fusion():
    rankings = {
        'bm25': ['A', 'B', 'C'],
        'dense': ['B', 'A', 'D']
    }
    k = 60
    fused = reciprocal_rank_fusion(rankings, k=k)
    
    # A: rank 1 in bm25 (1/(60+1)), rank 2 in dense (1/(60+2))
    score_a = 1.0/61 + 1.0/62
    # B: rank 2 in bm25 (1/(60+2)), rank 1 in dense (1/(60+1))
    score_b = 1.0/62 + 1.0/61
    # C: rank 3 in bm25 (1/(60+3))
    score_c = 1.0/63
    # D: rank 3 in dense (1/(60+3))
    score_d = 1.0/63
    
    assert fused['A'] == pytest.approx(score_a)
    assert fused['B'] == pytest.approx(score_b)
    assert fused['C'] == pytest.approx(score_c)
    assert fused['D'] == pytest.approx(score_d)
    
    # Verify relative ordering
    assert fused['A'] > fused['C']
    assert fused['B'] > fused['D']

def test_domain_match():
    assert domain_match("civil", "civil") == 1.0
    assert domain_match("civil", "penal") == 0.0
    assert domain_match(None, "civil") == 0.5

def test_exact_citation_boost():
    query_citations = [{'article': '17'}, {'article': '20', 'paragraph': '2'}]
    
    # Match article 17
    unit1 = {'article': '17', 'type': 'articol'}
    assert exact_citation_boost(query_citations, unit1) == 1.0
    
    # Match article 20, paragraph 2
    unit2 = {'article': '20', 'paragraph': '2'}
    assert exact_citation_boost(query_citations, unit2) == 1.0
    
    # Nested metadata support
    unit2_nested = {'metadata': {'article': '20', 'paragraph': '2'}}
    assert exact_citation_boost(query_citations, unit2_nested) == 1.0
    
    # No match
    unit3 = {'article': '21'}
    assert exact_citation_boost(query_citations, unit3) == 0.0
    
    # Partial match for multi-field citation
    unit4 = {'article': '20', 'paragraph': '1'}
    assert exact_citation_boost(query_citations, unit4) == 0.0
