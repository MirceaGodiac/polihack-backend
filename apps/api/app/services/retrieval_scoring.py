from dataclasses import dataclass

@dataclass
class ScoreBreakdown:
    bm25: float = 0.0
    dense: float = 0.0
    rrf: float = 0.0
    domain_match: float = 0.0
    metadata_validity: float = 0.0
    exact_citation_boost: float = 0.0
    intent_phrase_match: float = 0.0

def weighted_retrieval_score(s: ScoreBreakdown, dense_available: bool = True) -> float:
    """
    Calculates the final hybrid score based on the project's weighted formula.
    """
    if dense_available:
        score = (
            0.30 * _clamp01(s.rrf)
            + 0.25 * _clamp01(s.bm25)
            + 0.20 * _clamp01(s.dense)
            + 0.10 * _clamp01(s.exact_citation_boost)
            + 0.08 * _clamp01(s.domain_match)
            + 0.04 * _clamp01(s.metadata_validity)
            + 0.03 * _clamp01(s.intent_phrase_match)
        )
    else:
        score = (
            0.40 * _clamp01(s.rrf)
            + 0.35 * _clamp01(s.bm25)
            + 0.10 * _clamp01(s.exact_citation_boost)
            + 0.08 * _clamp01(s.domain_match)
            + 0.04 * _clamp01(s.metadata_validity)
            + 0.03 * _clamp01(s.intent_phrase_match)
        )
    return _clamp01(score)

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

def normalize_rrf_scores(rrf_scores: dict[str, float]) -> dict[str, float]:
    """
    Max-normalizes raw RRF values into [0, 1] so they can participate in
    weighted scoring instead of acting as a near-zero tie-breaker.
    """
    if not rrf_scores:
        return {}
    max_score = max(rrf_scores.values())
    if max_score <= 0.0:
        return {unit_id: 0.0 for unit_id in rrf_scores}
    return {
        unit_id: _clamp01(score / max_score)
        for unit_id, score in rrf_scores.items()
    }

def _clamp01(value: float) -> float:
    if value != value:
        return 0.0
    return max(0.0, min(1.0, value))

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
