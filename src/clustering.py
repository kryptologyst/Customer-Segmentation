import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Dict, Any
from loguru import logger

from .config import settings


class CustomerSegmentation:
    """K-Means clustering for customer segmentation."""
    
    def __init__(self, n_clusters: int = 3):
        self.n_clusters = n_clusters
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=settings.random_state,
            n_init=10
        )
        self.is_fitted = False
        
    def fit(self, X: np.ndarray) -> None:
        """Fit the K-Means model."""
        logger.info(f"Fitting K-Means with {self.n_clusters} clusters...")
        self.model.fit(X)
        self.is_fitted = True
        logger.success("Clustering completed")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def get_labels(self) -> np.ndarray:
        """Get cluster labels from fitted model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.model.labels_
    
    def get_cluster_centers(self) -> np.ndarray:
        """Get cluster centers."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.model.cluster_centers_
    
    def get_metrics(self, X: np.ndarray) -> Dict[str, Any]:
        """Calculate clustering metrics."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
            
        labels = self.get_labels()
        
        metrics = {
            "inertia": self.model.inertia_,
            "silhouette_score": silhouette_score(X, labels),
            "n_clusters": self.n_clusters,
            "n_samples": len(X)
        }
        
        return metrics
