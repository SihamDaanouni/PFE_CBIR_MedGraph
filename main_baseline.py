"""
Main execution pipeline for the Official Baseline (Generic CLIP).
"""

import argparse
import sys

# We reuse the bioclip loader function because it accepts custom transforms (like CLIP's preprocess)
from src.data.dataset_loader import get_bioclip_loader
from src.features.clip_extractor import GenericCLIPExtractor
from src.evaluation.baseline_evaluator import RetrievalEvaluator


def run_baseline_pipeline(csv_filename: str, img_dirname: str) -> None:
    print("==================================================")
    print("        BASELINE (Generic CLIP)           ")
    print("==================================================\n")

    try:
        print("[1/3] Initializing Generic CLIP Extractor and DataLoader...")
        extractor = GenericCLIPExtractor()
        clip_transform = extractor.get_transform()
        
        loader = get_bioclip_loader(
            csv_name=csv_filename, 
            img_dir=img_dirname, 
            bioclip_transform=clip_transform
        )
        
        print("\n[2/3] Starting Feature Extraction...")
        embeddings, labels = extractor.extract_from_loader(loader)
        
        prefix = "isic_clip_baseline"
        extractor.save_embeddings(embeddings, labels, filename_prefix=prefix)
        
        print("\n[3/3] Evaluating Retrieval Performance...")
        k_standard_values = [5, 10, 15, 20, 30, 100]
        evaluator = RetrievalEvaluator(k_values=k_standard_values)
        evaluator.evaluate(prefix=prefix)

        print("\nBaseline execution completed successfully.")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Generic CLIP Baseline Pipeline.")
    parser.add_argument("--csv", type=str, default="isic_balanced_multimodal.csv")
    parser.add_argument("--img_dir", type=str, default="isic_balanced_images")
    args = parser.parse_args()
    
    run_baseline_pipeline(csv_filename=args.csv, img_dirname=args.img_dir)