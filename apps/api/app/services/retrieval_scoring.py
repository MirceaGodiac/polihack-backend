from dataclasses import dataclass

@dataclass
class ScoreBreakdown:
    bm25: float = 0.0
    dense: float = 0.0
    rrf: float = 0.0
    domain_match: float = 0.0
    metadata_validity: float = 0.0
    exact_citation_boost: float = 0.0

def weighted_retrieval_score(s: ScoreBreakdown, dense_available: bool = True) -> float:
    """
    Calculates the final hybrid score based on the project's weighted formula.
    """
    if dense_available:
        return (0.35 * s.bm25 + 
                0.30 * s.dense + 
                0.15 * s.domain_match + 
                0.10 * s.metadata_validity + 
                0.10 * s.exact_citation_boost)
    else:
        return (0.60 * s.bm25 + 
                0.20 * s.domain_match + 
                0.10 * s.metadata_validity + 
                0.10 * s.exact_citation_boost)

def reciprocal_rank_fusion(rankings: dict[str, list[str]], k: int = 60) -> dict[str, float]:
    """
    Implements Reciprocal Rank Fusion (RRF) algorithm.
    rankings: dict mapping method name (e.g., 'bm25', 'dense') to a list of unit_ids in ranked order.
    k: Constant to mitigate the impact of high rankings.
    Returns mapping of unit_id -> fused RRF score.
    """
    fused_scores = {}
    for method, ids in rankings.items():
        for rank_zero, unit_id in enumerate(ids):
            rank = rank_zero + 1
            score = 1.0 / (k + rank)
            fused_scores[unit_id] = fused_scores.get(unit_id, 0.0) + score
    return fused_scores

def domain_match(query_domain: str | None, unit_domain: str) -> float:
    """
    Calculates domain relevance.
    Returns 1.0 for exact match, 0.5 if no query domain specified, 0.0 otherwise.
    """
    if query_domain is None:
        return 0.5
    return 1.0 if query_domain == unit_domain else 0.0

def exact_citation_boost(query_citations: list[dict], unit_dict: dict) -> float:
    """
    Calculates boost for units matching exact citations extracted from the query.
    Matches are based on dictionary comparison of provided fields (article, paragraph, etc.).
    """
    for cit in query_citations:
        match = True
        for key, val in cit.items():
            # Check both root level and nested 'metadata' if it exists
            unit_val = unit_dict.get(key)
            if unit_val is None and 'metadata' in unit_dict:
                unit_val = unit_dict['metadata'].get(key)
                
            if unit_val is None or str(unit_val) != str(val):
                match = False
                break
        if match:
            return 1.0
            
    return 0.0
