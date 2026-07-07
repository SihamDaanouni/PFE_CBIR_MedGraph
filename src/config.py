"""
Configuration module for the CBIR Medical Graph Project.
This file centralizes all paths, hyperparameters, and constants.
"""

import os
from pathlib import Path

# --- DIRECTORY PATHS ---
# Build paths dynamically based on the project root (2 levels up from this file)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Ensure essential directories exist at runtime
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- EXPERIMENT SETTINGS ---
# ISIC 2019 dataset specific settings
IMAGE_TARGET_SIZE = (224, 224)  # Standard input size for ResNet50 and ViT
BATCH_SIZE = 32
RANDOM_SEED = 42