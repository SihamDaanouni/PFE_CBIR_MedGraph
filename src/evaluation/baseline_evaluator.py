"""
Evaluation module for the CBIR Baseline.
Computes Cosine Similarity and Information Retrieval metrics (Precision@K) 
using the extracted deep features.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, List

from src.config import PROCESSED_DATA_DIR


class RetrievalEvaluator:
    """
    Evaluates the performance of image embeddings for content-based retrieval.
    """

    def __init__(self, k_values: List[int] = [1, 3, 5]):
        """
        Initializes the evaluator.

        Args:
            k_values (List[int]): A list of K values for which to compute Precision@K.
        """
        self.k_values = k_values

    def load_data(self, prefix: str = "baseline") -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads the precomputed embeddings and labels from disk.

        Args:
            prefix (str): The prefix used when saving the files.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The embeddings and labels arrays.
        """
        emb_path = PROCESSED_DATA_DIR / f"{prefix}_embeddings.npy"
        lbl_path = PROCESSED_DATA_DIR / f"{prefix}_labels.npy"

        if not emb_path.exists() or not lbl_path.exists():
            raise FileNotFoundError(f"Missing data files. Ensure extraction has been run for prefix '{prefix}'.")

        embeddings = np.load(emb_path)
        labels = np.load(lbl_path)

        return embeddings, labels

    def compute_precision_at_k(self, similarities: np.ndarray, labels: np.ndarray, k: int) -> float:
        """
        Computes the average Precision@K across all queries.

        Args:
            similarities (np.ndarray): Pairwise cosine similarity matrix (N x N).
            labels (np.ndarray): Array of ground truth labels (N).
            k (int): The number of top results to consider.

        Returns:
            float: The mean Precision@K value.
        """
        num_queries = similarities.shape[0]
        total_precision = 0.0

        for i in range(num_queries):
            query_label = labels[i]
            
            # Get the indices of the most similar images (descending order)
            # We exclude the first index because the most similar image to a query is the query itself
            sorted_indices = np.argsort(similarities[i])[::-1]
            top_k_indices = sorted_indices[1:k + 1]

            # Count how many retrieved images share the same label as the query
            retrieved_labels = labels[top_k_indices]
            relevant_matches = np.sum(retrieved_labels == query_label)

            # Avoid division by zero in case k is larger than the available dataset size
            actual_k = len(top_k_indices)
            if actual_k > 0:
                total_precision += (relevant_matches / actual_k)

        return total_precision / num_queries

    def evaluate(self, prefix: str = "baseline"):
        """
        Executes the full evaluation pipeline.
        """
        print(f"Starting evaluation for configuration: {prefix}")
        
        embeddings, labels = self.load_data(prefix)
        print(f"Data loaded. Computing pairwise cosine similarities for {embeddings.shape[0]} samples...")
        
        # Compute the NxN similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        print("\nRetrieval Performance Metrics:")
        print("-" * 30)
        
        # Compute metrics for each specified K
        for k in self.k_values:
            # For our mock data of 5 images, requesting K >= 5 will bound to the available items
            precision = self.compute_precision_at_k(similarity_matrix, labels, k)
            print(f"Precision@{k:<2} : {precision:.4f}")
            
        print("-" * 30)
        print("Evaluation complete.\n")