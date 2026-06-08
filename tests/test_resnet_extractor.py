"""
Unit test script for the ResNet50 feature extraction component.
"""

from src.data.dataset_loader import get_baseline_loader
from src.features.resnet_extractor import ResNetBaselineExtractor

def run_test() -> None:
    """Validates embedding dimensionality against expected model architecture."""
    print("Executing Feature Extractor validations...")
    loader = get_baseline_loader(csv_name="mock_metadata.csv", img_dir="mock_images")
    extractor = ResNetBaselineExtractor()
    
    embeddings, labels = extractor.extract_from_loader(loader)
    assert embeddings.shape == (5, 2048), f"Incorrect embedding shape: {embeddings.shape}"
    
    extractor.save_embeddings(embeddings, labels, filename_prefix="mock_baseline")
    print(f"Feature Extractor validation passed successfully. Shape: {embeddings.shape}")

if __name__ == "__main__":
    run_test()