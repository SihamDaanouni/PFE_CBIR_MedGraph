"""
Evaluation module for the CBIR pipeline.
Computes Information Retrieval metrics: Precision@K, Recall@K, and mAP@K.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, List, Dict

from src.config import PROCESSED_DATA_DIR


class RetrievalEvaluator:
    """
    Evaluates the performance of image embeddings for content-based retrieval
    using standard metrics: Precision, Recall, and Mean Average Precision (mAP).
    """

    def __init__(self, k_values: List[int] = [5, 10, 15, 20, 30, 100]):
        self.k_values = k_values

    def load_data(self, prefix: str) -> Tuple[np.ndarray, np.ndarray]:
        emb_path = PROCESSED_DATA_DIR / f"{prefix}_embeddings.npy"
        lbl_path = PROCESSED_DATA_DIR / f"{prefix}_labels.npy"

        if not emb_path.exists() or not lbl_path.exists():
            raise FileNotFoundError(f"Missing data files for prefix '{prefix}'.")

        embeddings = np.load(emb_path)
        labels = np.load(lbl_path)

        return embeddings, labels

    def compute_metrics(self, similarities: np.ndarray, labels: np.ndarray) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
        num_queries = similarities.shape[0]
        
        mean_precision = {k: 0.0 for k in self.k_values}
        mean_recall = {k: 0.0 for k in self.k_values}
        mean_map = {k: 0.0 for k in self.k_values}

        for i in range(num_queries):
            query_label = labels[i]
            
            total_relevant = np.sum(labels == query_label) - 1
            if total_relevant == 0:
                continue
            
            # Exclude the query image itself (index 0)
            sorted_indices = np.argsort(similarities[i])[::-1][1:]

            for k in self.k_values:
                top_k_indices = sorted_indices[:k]
                retrieved_labels = labels[top_k_indices]
                relevant_matches = (retrieved_labels == query_label)

                precision = np.sum(relevant_matches) / k
                mean_precision[k] += precision

                recall = np.sum(relevant_matches) / total_relevant
                mean_recall[k] += recall

                ap = 0.0
                relevant_count = 0
                for rank, is_relevant in enumerate(relevant_matches):
                    if is_relevant:
                        relevant_count += 1
                        ap += relevant_count / (rank + 1)
                
                mean_map[k] += ap / total_relevant

        for k in self.k_values:
            mean_precision[k] /= num_queries
            mean_recall[k] /= num_queries
            mean_map[k] /= num_queries

        return mean_precision, mean_recall, mean_map

    def evaluate(self, prefix: str):
        print(f"Starting comprehensive evaluation for: {prefix}")
        
        embeddings, labels = self.load_data(prefix)
        similarity_matrix = cosine_similarity(embeddings)
        
        print("\nInformation Retrieval Metrics:")
        print("-" * 55)
        print(f"{'K':<5} | {'Precision@K':<13} | {'Recall@K':<10} | {'mAP@K':<10}")
        print("-" * 55)
        
        mean_precision, mean_recall, mean_map = self.compute_metrics(similarity_matrix, labels)
        
        for k in self.k_values:
            print(f"{k:<5} | {mean_precision[k]:<13.4f} | {mean_recall[k]:<10.4f} | {mean_map[k]:<10.4f}")
            
        print("-" * 55)
        print("Evaluation complete.\n")