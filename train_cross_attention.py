"""
Training and Evaluation for Cross-Attention Fusion (Table 2 - Ligne 3).
Strictly implements Train/Test split to prevent data leakage and overfitting.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import open_clip
import time
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from src.data.dataset_loader import get_bioclip_loader
from src.evaluation.baseline_evaluator import RetrievalEvaluator
from src.models.multimodal_attention import CrossAttentionFusion

def run_cross_attention_pipeline(csv_filename: str, img_dirname: str):
    print("==================================================")
    print("    TRAINING CROSS-ATTENTION FUSION (PROPOSÉ)     ")
    print("==================================================\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1/5] Loading BioMedCLIP on {device}...")
    model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    model, _, preprocess = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device)
    model.eval()

    loader = get_bioclip_loader(csv_filename, img_dirname, preprocess)
    img_features_list, txt_features_list, labels_list = [], [], []
    inference_times = []

    print("\n[2/5] Extracting frozen features from BioMedCLIP...")
    with torch.no_grad():
        for batch_idx, (images, labels, prompts) in enumerate(loader):
            images, texts = images.to(device), tokenizer(prompts).to(device)
            start_time = time.time()
            img_f, txt_f, _ = model(images, texts)
            batch_time = time.time() - start_time
            inference_times.append(batch_time / images.shape[0])
            img_features_list.append(img_f.cpu())
            txt_features_list.append(txt_f.cpu())
            labels_list.append(labels)
            if (batch_idx + 1) % 20 == 0:
                print(f"   Batch {batch_idx + 1}/{len(loader)} processed.")

    avg_extraction_time = np.mean(inference_times)
    
    # 3. Train/Test Split (The crucial fix for Data Leakage)
    print("\n[3/5] Splitting Data (80% Train, 20% Test) to prevent leakage...")
    X_img_all = torch.cat(img_features_list).numpy()
    X_txt_all = torch.cat(txt_features_list).numpy()
    Y_all = torch.cat(labels_list).numpy()

    X_img_train, X_img_test, X_txt_train, X_txt_test, Y_train, Y_test = train_test_split(
        X_img_all, X_txt_all, Y_all, test_size=0.2, random_state=42, stratify=Y_all
    )
    
    # Convert to Tensors for training
    t_X_img_train = torch.tensor(X_img_train).to(device)
    t_X_txt_train = torch.tensor(X_txt_train).to(device)
    t_Y_train = torch.tensor(Y_train).to(device)

    print(f"Training on {len(Y_train)} images. Testing on {len(Y_test)} images.")

    fusion_module = CrossAttentionFusion(embed_dim=512).to(device)
    classifier_head = nn.Linear(512, 8).to(device) 
    optimizer = optim.Adam(list(fusion_module.parameters()) + list(classifier_head.parameters()), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    train_dataset = TensorDataset(t_X_img_train, t_X_txt_train, t_Y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    print(f"\n[4/5] Training Cross-Attention on TRAIN SET for 15 epochs...")
    fusion_module.train()
    for epoch in range(15):
        epoch_loss = 0.0
        for b_img, b_txt, b_labels in train_loader:
            optimizer.zero_grad()
            fused_vectors = fusion_module(b_img, b_txt)
            predictions = classifier_head(fused_vectors)
            loss = criterion(predictions, b_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch+1}/15 - Loss: {epoch_loss/len(train_loader):.4f}")

    print("\n[5/5] Generating embeddings for TEST SET and Evaluating...")
    fusion_module.eval()
    
    t_X_img_test = torch.tensor(X_img_test).to(device)
    t_X_txt_test = torch.tensor(X_txt_test).to(device)
    
    t0 = time.time()
    with torch.no_grad():
        final_test_embeddings = fusion_module(t_X_img_test, t_X_txt_test).cpu().numpy()
    fusion_time = (time.time() - t0) / len(Y_test)

    total_inference_time = avg_extraction_time + fusion_time

    # Evaluate ONLY on the strictly unseen Test Set
    evaluator = RetrievalEvaluator(k_values=[5, 10, 20, 30, 100])
    similarity_matrix = cosine_similarity(final_test_embeddings)
    mean_p, mean_r, mean_map, mean_f1 = evaluator.compute_metrics(similarity_matrix, Y_test)

    print("\n>>> UNBIASED RESULTS FOR CROSS-ATTENTION (TEST SET ONLY) <<<")
    print(f"Average Inference Time per image: {total_inference_time:.5f} seconds")
    for k in [10, 20]:
        print(f"P@{k}: {mean_p[k]:.4f} | MAP@{k}: {mean_map[k]:.4f} | F1@{k}: {mean_f1[k]:.4f}")

if __name__ == "__main__":
    run_cross_attention_pipeline("isic_balanced_multimodal.csv", "isic_balanced_images")