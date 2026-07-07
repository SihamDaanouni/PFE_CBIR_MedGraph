"""
Feature extraction module using Generic CLIP (OpenAI).
Used as the official baseline for the CBIR pipeline.
"""

import torch
import numpy as np
import open_clip
from torch.utils.data import DataLoader

from src.config import PROCESSED_DATA_DIR


class GenericCLIPExtractor:
    """
    Extracts embeddings using the standard OpenAI CLIP model.
    Serves as the baseline to prove the value of domain-specific (BioMedCLIP) models.
    """

    def __init__(self, device: str = None):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            
        print(f"Initializing Generic CLIP Extractor on device: {self.device}")
        
        # Load standard OpenAI CLIP
        model_name = 'ViT-B-32'
        pretrained = 'openai'
        print(f"Downloading/Loading {model_name} weights...")
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        self.model = self.model.to(self.device)
        self.model.eval()

    def get_transform(self):
        return self.preprocess

    @torch.no_grad()
    def extract_from_loader(self, dataloader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        all_embeddings = []
        all_labels = []

        print(f"Starting Generic CLIP extraction for {len(dataloader.dataset)} images...")
        
        for batch_idx, (images, labels, prompts) in enumerate(dataloader):
            images = images.to(self.device)
            texts = self.tokenizer(prompts).to(self.device)
            
            # Extract features
            image_features, text_features, _ = self.model(images, texts)
            
            # For the pure baseline, we normalize the visual features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            all_embeddings.append(image_features.cpu().numpy())
            all_labels.append(labels.numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"   Batch {batch_idx + 1}/{len(dataloader)} processed.")

        embeddings_array = np.vstack(all_embeddings)
        labels_array = np.concatenate(all_labels)
        
        return embeddings_array, labels_array

    def save_embeddings(self, embeddings: np.ndarray, labels: np.ndarray, filename_prefix: str = "clip_baseline"):
        emb_path = PROCESSED_DATA_DIR / f"{filename_prefix}_embeddings.npy"
        lbl_path = PROCESSED_DATA_DIR / f"{filename_prefix}_labels.npy"
        
        np.save(emb_path, embeddings)
        np.save(lbl_path, labels)
        print(f"Saved Generic CLIP embeddings to: {emb_path}")