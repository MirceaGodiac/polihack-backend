import argparse
import json
import os
import random
import sys

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ingestion.eval_parser import load_eval_cases
from ingestion.retrieval_eval import calculate_recall_at_k, calculate_mrr, calculate_ndcg_at_k

def run_dummy_retrieval(query: str, expected_units: list[str]) -> list[str]:
    """
    Simulates a retrieval engine for testing purposes.
    Returns a list of unit IDs with some randomness to exercise metrics.
    """
    rand = random.random()
    retrieved = []
    
    # Simulate different scenarios
    if rand < 0.4:
        # Scenario 1: Perfect retrieval (Expected units at the top)
        retrieved = expected_units + [f"noise_{i}" for i in range(15)]
    elif rand < 0.7:
        # Scenario 2: Relevant items found but not at the very top
        noise_pre = [f"noise_{i}" for i in range(3)]
        retrieved = noise_pre + expected_units[:1] + [f"noise_{i}" for i in range(3, 15)]
    elif rand < 0.9:
        # Scenario 3: Partial match
        retrieved = [f"noise_{i}" for i in range(5)] + expected_units[1:] + [f"noise_{i}" for i in range(5, 20)]
    else:
        # Scenario 4: Total miss
        retrieved = [f"miss_{i}" for i in range(20)]
        
    return retrieved

def main():
    parser = argparse.ArgumentParser(description="AntiGravity Retrieval Evaluation CLI")
    parser.add_argument("--eval-cases", required=True, help="Path to the JSON file containing evaluation cases")
    parser.add_argument("--out", required=True, help="Path where the final report JSON will be saved")
    args = parser.parse_args()

    print(f"Loading evaluation cases from {args.eval_cases}...")
    try:
        cases = list(load_eval_cases(args.eval_cases))
    except Exception as e:
        print(f"Error loading cases: {e}")
        sys.exit(1)

    print(f"Running benchmark on {len(cases)} cases...")
    
    case_results = []
    for case in cases:
        retrieved_ids = run_dummy_retrieval(case["question"], case["expected_units"])
        expected_set = set(case["expected_units"])
        
        metrics = {
            "case_id": case["id"],
            "question": case["question"],
            "recall_10": calculate_recall_at_k(expected_set, retrieved_ids, 10),
            "mrr": calculate_mrr(expected_set, retrieved_ids),
            "ndcg_10": calculate_ndcg_at_k(expected_set, retrieved_ids, 10)
        }
        case_results.append(metrics)

    # Calculate overall aggregates
    num_cases = len(case_results)
    avg_recall_10 = sum(r["recall_10"] for r in case_results) / num_cases
    avg_mrr = sum(r["mrr"] for r in case_results) / num_cases
    avg_ndcg_10 = sum(r["ndcg_10"] for r in case_results) / num_cases

    summary = {
        "avg_recall_10": round(avg_recall_10, 4),
        "avg_mrr": round(avg_mrr, 4),
        "avg_ndcg_10": round(avg_ndcg_10, 4),
        "total_cases": num_cases
    }

    report = {
        "summary": summary,
        "details": case_results
    }

    # Save output
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print("\n--- Evaluation Report ---")
    print(f"Total Cases: {num_cases}")
    print(f"Avg Recall@10: {summary['avg_recall_10']}")
    print(f"Avg MRR:       {summary['avg_mrr']}")
    print(f"Avg nDCG@10:   {summary['avg_ndcg_10']}")
    print(f"------------------------")
    print(f"Full report saved to {args.out}")

if __name__ == "__main__":
    main()
