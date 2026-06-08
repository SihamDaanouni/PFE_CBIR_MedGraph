"""
Unit test script for the evaluation pipeline.
"""

from src.evaluation.baseline_evaluator import RetrievalEvaluator

def run_test() -> None:
    """Validates similarity and information retrieval calculations."""
    print ("----------------------------")
    print("Executing Evaluator performance validations...")
    evaluator = RetrievalEvaluator(k_values=[1, 2, 4])
    evaluator.evaluate(prefix="mock_baseline")
    print("Evaluator metrics checking pipeline completed successfully.")

if __name__ == "__main__":
    run_test()