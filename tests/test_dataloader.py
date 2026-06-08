"""
Unit test script for the dataset loader component.
Generates temporary mock dataset assets and validates data loader shapes.
"""

import pandas as pd
import numpy as np
from PIL import Image
from src.config import RAW_DATA_DIR
from src.data.dataset_loader import get_baseline_loader

def generate_mock_assets() -> None:
    """Generates localized sample data for verification."""
    print("Generating isolated test assets...")
    mock_img_dir = RAW_DATA_DIR / "mock_images"
    mock_img_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = RAW_DATA_DIR / "mock_metadata.csv"
    mock_data = {
        "image": [f"mock_lesion_{i}" for i in range(5)],
        "diagnosis": ["melanoma", "nevus", "melanoma", "nevus", "seborrheic_keratosis"]
    }
    pd.DataFrame(mock_data).to_csv(csv_path, index=False)
    
    for img_name in mock_data["image"]:
        img_path = mock_img_dir / f"{img_name}.jpg"
        random_array = np.random.rand(224, 224, 3) * 255
        Image.fromarray(random_array.astype('uint8')).save(img_path)

def run_test() -> None:
    """Validates the structure and formatting of loaded batches."""
    generate_mock_assets()
    print("Executing DataLoader validations...")
    loader = get_baseline_loader(csv_name="mock_metadata.csv", img_dir="mock_images")
    
    for images, labels, prompts in loader:
        assert images.shape == (5, 3, 224, 224), f"Incorrect shape detected: {images.shape}"
        print(f"DataLoader validation passed successfully. Shape: {images.shape}")
        break

if __name__ == "__main__":
    run_test()