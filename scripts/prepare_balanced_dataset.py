"""
Dataset preparation script for the ISIC 2019 Archive.
Merges ground truth labels with clinical metadata, applies hybrid balancing,
and extracts the final dataset for multimodal feature extraction.
"""

import argparse
import shutil
import pandas as pd
from pathlib import Path

# --- DIRECTORY PATHS ---
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

def create_balanced_multimodal_subset(
    gt_csv: str, 
    meta_csv: str, 
    img_dir: str, 
    target_per_class: int, 
    output_prefix: str
) -> None:
    """
    Reads ground truth and metadata, merges them, balances the classes, 
    and extracts the corresponding physical images.
    """
    path_gt = RAW_DATA_DIR / gt_csv
    path_meta = RAW_DATA_DIR / meta_csv
    path_img = RAW_DATA_DIR / img_dir
    
    if not path_gt.exists():
        raise FileNotFoundError(f"Ground Truth file not found: {path_gt}")
    if not path_meta.exists():
        raise FileNotFoundError(f"Metadata file not found: {path_meta}")
    if not path_img.exists():
        raise FileNotFoundError(f"Image directory not found: {path_img}")

    output_img_dir = RAW_DATA_DIR / f"{output_prefix}_images"
    output_csv_path = RAW_DATA_DIR / f"{output_prefix}_multimodal.csv"

    if output_img_dir.exists():
        print(f"Cleaning existing output directory: {output_img_dir}")
        shutil.rmtree(output_img_dir)
    output_img_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    df_gt = pd.read_csv(path_gt)
    df_meta = pd.read_csv(path_meta)

    # Merge Ground Truth and Metadata on the 'image' column
    print("Merging Ground Truth and Clinical Metadata...")
    df_merged = pd.merge(df_gt, df_meta, on='image', how='inner')

    # Identify diagnostic columns (one-hot encoded in ISIC 2019)
    # Exclude standard metadata columns and 'UNK' (Unknown class)
    excluded_cols = ['image', 'lesion_id', 'age_approx', 'anatom_site_general', 'sex', 'UNK']
    class_columns = [col for col in df_merged.columns if col not in excluded_cols]
    
    # Assign a single label column for stratification
    df_merged['diagnostic_class'] = df_merged[class_columns].idxmax(axis=1)
    
    balanced_dfs = []
    
    print("\n--- Executing Hybrid Class Balancing ---")
    for class_name in class_columns:
        class_subset = df_merged[df_merged['diagnostic_class'] == class_name]
        available_count = len(class_subset)
        
        if available_count == 0:
            continue
            
        # Oversample if insufficient data, undersample if abundant
        needs_replacement = available_count < target_per_class
        
        sampled_subset = class_subset.sample(
            n=target_per_class, 
            replace=needs_replacement, 
            random_state=42
        )
        
        balanced_dfs.append(sampled_subset)
        
        action = "Oversampled" if needs_replacement else "Undersampled"
        print(f"Class {class_name:<5} | Available: {available_count:<5} | Action: {action} to {target_per_class}")

    final_balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    final_balanced_df = final_balanced_df.drop(columns=['diagnostic_class'])

    print("\nBeginning physical image extraction and renaming...")
    copied_count = 0
    missing_images = 0
    
    for idx, row in final_balanced_df.iterrows():
        src_filename = f"{row['image']}.jpg"
        src_file = path_img / src_filename
        
        # Assign a unique filename to handle duplicated rows from oversampling
        dst_filename = f"{row['image']}_{copied_count}.jpg"
        dst_file = output_img_dir / dst_filename
        
        if src_file.exists():
            shutil.copy2(src_file, dst_file)
            # Update dataframe with the new unique identifier
            final_balanced_df.at[idx, 'image'] = f"{row['image']}_{copied_count}"
            copied_count += 1
        else:
            missing_images += 1
            
        if copied_count > 0 and copied_count % 1000 == 0:
            print(f"   Processed {copied_count} images...")

    # Save the final unified CSV
    final_balanced_df.to_csv(output_csv_path, index=False)

    print("\nDataset generation complete.")
    print(f"Total synchronized records saved to: {output_csv_path}")
    print(f"Total physical images copied: {copied_count}")
    
    if missing_images > 0:
        print(f"Warning: {missing_images} images could not be located in the source directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a balanced, multimodal ISIC dataset.")
    parser.add_argument("--gt", type=str, default="ISIC_2019_Training_GroundTruth.csv", help="Ground truth CSV")
    parser.add_argument("--meta", type=str, default="ISIC_2019_Training_Metadata.csv", help="Metadata CSV")
    parser.add_argument("--img_dir", type=str, default="ISIC_2019_Training_Input", help="Source image directory")
    parser.add_argument("--target", type=int, default=625, help="Target samples per class")
    parser.add_argument("--prefix", type=str, default="isic_balanced", help="Output prefix")
    
    args = parser.parse_args()
    
    create_balanced_multimodal_subset(
        gt_csv=args.gt,
        meta_csv=args.meta,
        img_dir=args.img_dir,
        target_per_class=args.target,
        output_prefix=args.prefix
    )