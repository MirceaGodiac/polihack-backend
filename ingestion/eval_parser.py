import json
import os
from typing import TypedDict, List, Generator

class EvalCase(TypedDict):
    id: str
    question: str
    expected_domain: str
    expected_units: List[str]

def load_eval_cases(file_path: str) -> Generator[EvalCase, None, None]:
    """
    Reads evaluation cases from a JSON file and validates their structure.
    Yields validated EvalCase objects.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Evaluation cases file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in evaluation file: {e}")
        
    if not isinstance(data, list):
        raise ValueError("Evaluation cases must be provided as a JSON list.")
        
    required_fields = ["id", "question", "expected_domain", "expected_units"]
    
    for i, case in enumerate(data):
        for field in required_fields:
            if field not in case:
                raise ValueError(f"Case at index {i} is missing required field: '{field}'")
        
        if not isinstance(case["expected_units"], list):
            raise ValueError(f"Case '{case.get('id', i)}' must have 'expected_units' as a list.")
            
        yield case
