import math

def calculate_recall_at_k(expected_ids: set[str], retrieved_ids: list[str], k: int) -> float:
    """
    Calculates the proportion of relevant documents found in the top k results.
    """
    if not expected_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = len(expected_ids.intersection(set(top_k)))
    return hits / len(expected_ids)

def calculate_mrr(expected_ids: set[str], retrieved_ids: list[str]) -> float:
    """
    Calculates Mean Reciprocal Rank. 
    Returns 1.0 / rank of the first relevant document found.
    """
    if not expected_ids:
        return 0.0
    for i, unit_id in enumerate(retrieved_ids):
        if unit_id in expected_ids:
            return 1.0 / (i + 1)
    return 0.0

def calculate_ndcg_at_k(expected_ids: set[str], retrieved_ids: list[str], k: int) -> float:
    """
    Calculates Normalized Discounted Cumulative Gain at k for binary relevance.
    Formula: DCG / IDCG, where DCG = sum(1 / log2(rank + 1)) for each hit.
    """
    if not expected_ids:
        return 0.0
    
    # Binary relevance DCG
    dcg = 0.0
    top_k = retrieved_ids[:k]
    for i, unit_id in enumerate(top_k):
        if unit_id in expected_ids:
            # rank i is (i+1), log2(rank+1) is log2(i+2)
            dcg += 1.0 / math.log2(i + 2)
            
    # Ideal DCG: all expected items (up to k) are at the very top
    idcg = 0.0
    num_possible_hits = min(len(expected_ids), k)
    for i in range(num_possible_hits):
        idcg += 1.0 / math.log2(i + 2)
        
    if idcg == 0.0:
        return 0.0
        
    return dcg / idcg
