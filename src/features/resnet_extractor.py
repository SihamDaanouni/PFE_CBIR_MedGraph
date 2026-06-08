"""
Feature extraction module for the Baseline CBIR system.
Uses a pre-trained ResNet50 to extract deep visual embeddings from medical images.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
import numpy as np
from src.config import PROCESSED_DATA_DIR

class ResNetBaselineExtractor:
    """
    Extracts visual embeddings using a pre-trained ResNet50 model.
    The classification head is removed to output raw, high-dimensional feature vectors.
    """

    def __init__(self, device: str = None):
        """
        Initializes the ResNet50 model for feature extraction.
        
        Args:
            device (str, optional): Device to run the model on ('cuda', 'mps', or 'cpu').
        """
        # Automatically detect hardware accelerator to optimize inference speed
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            
        print(f"Initializing ResNet50 Extractor on hardware: {self.device}")
        
        # Load standard pre-trained ResNet50 weights (ImageNet)
        weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=weights)
        
        # Remove the final fully connected layer (classifier) to retrieve embeddings (dim 2048)
        self.model.fc = nn.Identity()
        
        # Move model to the selected hardware and lock it in evaluation mode (no gradient updates)
        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def extract_from_loader(self, dataloader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        """
        Iterates over the dataloader and passes images through the model to extract embeddings.

        Args:
            dataloader (DataLoader): The PyTorch DataLoader containing batched images.

        Returns:
            tuple: A tuple containing:
                - embeddings (np.ndarray): The extracted feature vectors (Shape: N, 2048).
                - labels (np.ndarray): The corresponding ground-truth labels.
        """
        all_embeddings = []
        all_labels = []


        print(f"Starting embedding extraction for {len(dataloader.dataset)} images...")
        
        for batch_idx, (images, labels, _) in enumerate(dataloader):
            # Transfer images to the same device as the model
            images = images.to(self.device)
            
            # Forward pass to get deep features
            features = self.model(images)
            
            # Move data back to CPU memory and convert to Numpy for FAISS compatibility later
            all_embeddings.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
            
            print(f"   [Batch {batch_idx + 1}/{len(dataloader)}] processed.")

        # Concatenate all batches into single continuous arrays
        embeddings_array = np.vstack(all_embeddings)
        labels_array = np.concatenate(all_labels)
        
        return embeddings_array, labels_array

    def save_embeddings(self, embeddings: np.ndarray, labels: np.ndarray, filename_prefix: str = "baseline"):
        """
        Saves the extracted numpy arrays to disk in the processed data directory.
        """
        emb_path = PROCESSED_DATA_DIR / f"{filename_prefix}_embeddings.npy"
        lbl_path = PROCESSED_DATA_DIR / f"{filename_prefix}_labels.npy"
        
        np.save(emb_path, embeddings)
        np.save(lbl_path, labels)
        
        print(f"Successfully saved embeddings to: {emb_path}")
        print(f"Successfully saved labels to: {lbl_path}")

'''# --- Quick Unit Test Execution ---
if __name__ == "__main__":
    from src.data.dataset_loader import get_baseline_loader
    
    print("--- Testing ResNet50 Pipeline ---")
    
    # Initialize the loader using the mock data generated in the previous step
    test_loader = get_baseline_loader(csv_name="mock_metadata.csv", img_dir="mock_images")
    
    # Initialize extractor and process the data
    extractor = ResNetBaselineExtractor()
    test_embeddings, test_labels = extractor.extract_from_loader(test_loader)
    
    print(f"\n✅ Extraction Complete!")
    print(f"   Final Embeddings Matrix Shape: {test_embeddings.shape} (Expected: 5, 2048)")
    
    # Validate the saving mechanism
    extractor.save_embeddings(test_embeddings, test_labels, filename_prefix="mock_baseline")
'''
