"""
Ablation Study for Multimodal Fusion (Table 2).
Evaluates Zero-Shot fusion mechanisms: Concatenation and Fixed Beta Weighting.
"""

import torch
import numpy as np
import open_clip
import time
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity

from src.data.dataset_loader import get_bioclip_loader
from src.evaluation.baseline_evaluator import RetrievalEvaluator

def run_fusion_ablation(csv_filename: str, img_dirname: str):
    print("==================================================")
    print("       TABLE 2: MULTIMODAL FUSION ABLATION        ")
    print("==================================================\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1/4] Loading BioMedCLIP on {device}...")
    
    model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    model, _, preprocess = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device)
    model.eval()

    loader = get_bioclip_loader(csv_filename, img_dirname, preprocess)
    
    img_features_list, txt_features_list, labels_list = [], [], []
    inference_times = []

    print("\n[2/4] Extracting isolated Image and Text features (simulating inference)...")
    with torch.no_grad():
        for batch_idx, (images, labels, prompts) in enumerate(loader):
            images, texts = images.to(device), tokenizer(prompts).to(device)
            
            start_time = time.time()
            img_f, txt_f, _ = model(images, texts)
            batch_time = time.time() - start_time
            inference_times.append(batch_time / images.shape[0])
            
            img_f /= img_f.norm(dim=-1, keepdim=True)
            txt_f /= txt_f.norm(dim=-1, keepdim=True)
            
            img_features_list.append(img_f.cpu().numpy())
            txt_features_list.append(txt_f.cpu().numpy())
            labels_list.append(labels.numpy())

            if (batch_idx + 1) % 20 == 0:
                print(f"   Batch {batch_idx + 1}/{len(loader)} processed.")

    img_features = np.vstack(img_features_list)
    txt_features = np.vstack(txt_features_list)
    all_labels = np.concatenate(labels_list)
    avg_extraction_time = np.mean(inference_times)

    print("\n[3/4] Applying Fusion Mechanisms...")
    t0 = time.time()
    concat_fusion = np.concatenate((img_features, txt_features), axis=1)
    time_concat = (time.time() - t0) / len(all_labels)
    
    t0 = time.time()
    beta = 0.75
    beta_fusion = (beta * img_features) + ((1 - beta) * txt_features)
    beta_fusion = beta_fusion / np.linalg.norm(beta_fusion, axis=1, keepdims=True)
    time_beta = (time.time() - t0) / len(all_labels)

    total_time_concat = avg_extraction_time + time_concat
    total_time_beta = avg_extraction_time + time_beta

    print("\n[4/4] Evaluating Retrieval Performance...")
    evaluator = RetrievalEvaluator(k_values=[5, 10, 20, 30, 100])
    
    print("\n>>> RESULTS FOR CONCATENATION SIMPLE <<<")
    print(f"Average Inference Time per image: {total_time_concat:.5f} seconds")
    sim_concat = cosine_similarity(concat_fusion)
    mean_p_c, mean_r_c, mean_map_c, mean_f1_c = evaluator.compute_metrics(sim_concat, all_labels)
    for k in [10, 20]:
        print(f"P@{k}: {mean_p_c[k]:.4f} | MAP@{k}: {mean_map_c[k]:.4f} | F1@{k}: {mean_f1_c[k]:.4f}")

    print("\n>>> RESULTS FOR FUSION PONDEREE (Beta=0.75) <<<")
    print(f"Average Inference Time per image: {total_time_beta:.5f} seconds")
    sim_beta = cosine_similarity(beta_fusion)
    mean_p_b, mean_r_b, mean_map_b, mean_f1_b = evaluator.compute_metrics(sim_beta, all_labels)
    for k in [10, 20]:
        print(f"P@{k}: {mean_p_b[k]:.4f} | MAP@{k}: {mean_map_b[k]:.4f} | F1@{k}: {mean_f1_b[k]:.4f}")

if __name__ == "__main__":
    run_fusion_ablation(csv_filename="isic_balanced_multimodal.csv", img_dirname="isic_balanced_images")