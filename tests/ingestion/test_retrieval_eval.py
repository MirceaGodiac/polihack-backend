import pytest
import os
import json
from ingestion.retrieval_eval import calculate_recall_at_k, calculate_mrr, calculate_ndcg_at_k
from ingestion.eval_parser import load_eval_cases

def test_calculate_recall_at_k():
    expected = {"unit1", "unit2"}
    
    # 1. Full hit
    retrieved = ["unit1", "unit2", "noise"]
    assert calculate_recall_at_k(expected, retrieved, 2) == 1.0
    
    # 2. Partial hit
    retrieved = ["noise", "unit1", "other_noise"]
    assert calculate_recall_at_k(expected, retrieved, 5) == 0.5
    
    # 3. Hit outside k
    assert calculate_recall_at_k(expected, retrieved, 1) == 0.0
    
    # 4. Zero hits
    assert calculate_recall_at_k(expected, ["noise"], 10) == 0.0

def test_calculate_mrr():
    expected = {"unit_target"}
    
    # 1. First position
    assert calculate_mrr(expected, ["unit_target", "noise"]) == 1.0
    
    # 2. Second position
    assert calculate_mrr(expected, ["noise", "unit_target"]) == 0.5
    
    # 3. Third position
    assert calculate_mrr(expected, ["n1", "n2", "unit_target"]) == 1/3
    
    # 4. Not found
    assert calculate_mrr(expected, ["n1", "n2"]) == 0.0

def test_calculate_ndcg_at_k():
    expected = {"A", "B"}
    
    # 1. Perfect Ranking (A and B at top)
    # DCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.6309 = 1.6309
    # IDCG = 1.6309
    assert calculate_ndcg_at_k(expected, ["A", "B"], 2) == pytest.approx(1.0)
    
    # 2. Swapped/Suboptimal Ranking (A at 2, B at 10)
    # DCG = 1/log2(3) + 1/log2(11) = 0.6309 + 0.2891 = 0.9200
    # IDCG = 1.6309
    # Ratio = 0.5641
    retrieved = ["noise"] * 10
    retrieved[1] = "A"
    retrieved[9] = "B"
    assert calculate_ndcg_at_k(expected, retrieved, 10) == pytest.approx(0.5641, abs=1e-3)
    
    # 3. Hit outside k (should be ignored)
    assert calculate_ndcg_at_k(expected, ["noise", "noise", "A"], 2) == 0.0

def test_load_eval_cases(tmp_path):
    # Create a temporary valid JSON file
    d = tmp_path / "eval"
    d.mkdir()
    f = d / "test_cases.json"
    
    valid_data = [
        {
            "id": "T1",
            "question": "test?",
            "expected_domain": "test",
            "expected_units": ["U1"]
        }
    ]
    f.write_text(json.dumps(valid_data))
    
    cases = list(load_eval_cases(str(f)))
    assert len(cases) == 1
    assert cases[0]["id"] == "T1"

def test_load_eval_cases_invalid(tmp_path):
    f = tmp_path / "invalid.json"
    
    # Missing field
    invalid_data = [{"id": "T1", "question": "missing fields"}]
    f.write_text(json.dumps(invalid_data))
    
    with pytest.raises(ValueError, match="missing required field"):
        list(load_eval_cases(str(f)))
