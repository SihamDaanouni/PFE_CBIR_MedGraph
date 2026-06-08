"""
Main execution pipeline for the Baseline CBIR system.
Orchestrates data loading, ResNet50 feature extraction, and performance evaluation.
"""

import argparse
import sys

from src.data.dataset_loader import get_baseline_loader
from src.features.resnet_extractor import ResNetBaselineExtractor
from src.evaluation.baseline_evaluator import RetrievalEvaluator

def run_baseline_pipeline(csv_filename: str, img_dirname: str) -> None:
    """
    Executes the full baseline pipeline: Extraction -> Saving -> Evaluation.
    """
    print("==================================================")
    print("       CBIR BASELINE PIPELINE (ResNet50)          ")
    print("==================================================\n")

    try:
        # 1. Initialize Data Loader
        print("[1/3] Initializing DataLoader...")
        loader = get_baseline_loader(csv_name=csv_filename, img_dir=img_dirname)
        
        # 2. Feature Extraction
        print("\n[2/3] Starting Feature Extraction...")
        extractor = ResNetBaselineExtractor()
        embeddings, labels = extractor.extract_from_loader(loader)
        
        # Save embeddings to disk
        extractor.save_embeddings(embeddings, labels, filename_prefix="mock_baseline")
        
        # 3. Evaluation
        print("\n[3/3] Evaluating Retrieval Performance...")
        evaluator = RetrievalEvaluator(k_values=[1, 2, 4])
        evaluator.evaluate(prefix="mock_baseline")

        print("\nBaseline pipeline execution completed successfully.")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Ensure the dataset is correctly placed in the data/raw/ directory.")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the CBIR Baseline Pipeline.")
    parser.add_argument(
        "--csv", 
        type=str, 
        default="mock_metadata.csv", 
        help="Name of the ground truth CSV file in data/raw/"
    )
    parser.add_argument(
        "--img_dir", 
        type=str, 
        default="mock_images", 
        help="Name of the image directory in data/raw/"
    )
    
    args = parser.parse_args()
    run_baseline_pipeline(csv_filename=args.csv, img_dirname=args.img_dir)