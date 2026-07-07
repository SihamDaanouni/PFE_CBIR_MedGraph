"""
Main execution pipeline for Ablation Level 1: BioCLIP only.
Orchestrates multimodal data loading, BioCLIP feature extraction, and performance evaluation.
"""

import argparse
import sys

from src.data.dataset_loader import get_bioclip_loader
from src.features.bioclip_extractor import BioCLIPExtractor
from src.evaluation.baseline_evaluator import RetrievalEvaluator


def run_bioclip_pipeline(csv_filename: str, img_dirname: str) -> None:
    """
    Executes the Level 1 ablation pipeline: Multimodal Extraction -> Saving -> Evaluation.
    """
    print("==================================================")
    print("       ABLATION LEVEL 1: BioCLIP ONLY             ")
    print("==================================================\n")

    try:
        # 1. Initialize Extractor and extract internal preprocessing transforms
        print("[1/3] Initializing BioCLIP Extractor and DataLoader...")
        extractor = BioCLIPExtractor()
        bioclip_transform = extractor.get_transform()
        
        loader = get_bioclip_loader(
            csv_name=csv_filename, 
            img_dir=img_dirname, 
            bioclip_transform=bioclip_transform
        )
        
        # 2. Multimodal Feature Extraction
        print("\n[2/3] Starting Multimodal Feature Extraction...")
        embeddings, labels = extractor.extract_from_loader(loader)
        
        prefix = "isic_bioclip"
        extractor.save_embeddings(embeddings, labels, filename_prefix=prefix)
        
        # 3. Evaluation using identical SemVisIR K values
        print("\n[3/3] Evaluating Retrieval Performance...")
        k_standard_values = [5, 10, 15, 20, 30, 100]
        evaluator = RetrievalEvaluator(k_values=k_standard_values)
        evaluator.evaluate(prefix=prefix)

        print("\nLevel 1 execution completed successfully.")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Ensure the dataset is correctly placed in the data/raw/ directory.")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the BioCLIP Ablation Pipeline.")
    parser.add_argument(
        "--csv", 
        type=str, 
        default="isic_balanced_multimodal.csv", 
        help="Name of the ground truth CSV file in data/raw/"
    )
    parser.add_argument(
        "--img_dir", 
        type=str, 
        default="isic_balanced_images", 
        help="Name of the image directory in data/raw/"
    )
    
    args = parser.parse_args()
    run_bioclip_pipeline(csv_filename=args.csv, img_dirname=args.img_dir)