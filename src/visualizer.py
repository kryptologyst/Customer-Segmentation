import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from loguru import logger


class ClusterVisualizer:
    """Visualization utilities for clustering."""
    
    @staticmethod
    def plot_clusters(
        X: np.ndarray,
        labels: np.ndarray,
        centers: np.ndarray,
        save_path: Optional[Path] = None
    ) -> None:
        """Plot clusters with centers."""
        plt.figure(figsize=(10, 6))
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=100, alpha=0.6, edgecolors='black')
        plt.scatter(centers[:, 0], centers[:, 1], c='red', s=300, marker='X', 
                   label='Centroids', edgecolors='black', linewidths=2)
        plt.xlabel("Annual Income ($1000s)", fontsize=12)
        plt.ylabel("Spending Score", fontsize=12)
        plt.title("Customer Segmentation using K-Means Clustering", fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Cluster plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_elbow_curve(
        k_values: range,
        inertias: list,
        save_path: Optional[Path] = None
    ) -> None:
        """Plot elbow curve for optimal k selection."""
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, inertias, marker='o', linewidth=2, markersize=8)
        plt.xlabel("Number of Clusters (k)", fontsize=12)
        plt.ylabel("Inertia", fontsize=12)
        plt.title("Elbow Method for Optimal k", fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Elbow curve saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_silhouette_scores(
        k_values: range,
        scores: list,
        save_path: Optional[Path] = None
    ) -> None:
        """Plot silhouette scores."""
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, scores, marker='o', linewidth=2, markersize=8, color='green')
        plt.xlabel("Number of Clusters (k)", fontsize=12)
        plt.ylabel("Silhouette Score", fontsize=12)
        plt.title("Silhouette Analysis", fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Silhouette plot saved to {save_path}")
        
        plt.show()
