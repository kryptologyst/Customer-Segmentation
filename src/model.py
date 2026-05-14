import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from loguru import logger
from typing import Optional


class CustomerSegmentation:
    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            max_iter=300,
        )
        self.labels_: Optional[np.ndarray] = None
        self.cluster_centers_: Optional[np.ndarray] = None
        self.inertia_: float = 0.0

    def fit(self, X: np.ndarray) -> "CustomerSegmentation":
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.labels_ = self.model.labels_
        self.cluster_centers_ = self.scaler.inverse_transform(
            self.model.cluster_centers_
        )
        self.inertia_ = self.model.inertia_
        logger.info(
            f"K-Means fitted: {self.n_clusters} clusters, "
            f"inertia={self.inertia_:.2f}"
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def get_labels(self) -> np.ndarray:
        if self.labels_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.labels_

    def get_cluster_centers(self) -> np.ndarray:
        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.cluster_centers_

    def get_metrics(self, X: np.ndarray) -> dict:
        if self.labels_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        X_scaled = self.scaler.transform(X)
        return {
            "inertia": float(self.inertia_),
            "silhouette_score": float(
                silhouette_score(X_scaled, self.labels_)
            ),
            "davies_bouldin_score": float(
                davies_bouldin_score(X_scaled, self.labels_)
            ),
            "calinski_harabasz_score": float(
                calinski_harabasz_score(X_scaled, self.labels_)
            ),
        }

    def get_cluster_profiles(self, X: np.ndarray, feature_names: list) -> pd.DataFrame:
        if self.labels_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        df = pd.DataFrame(X, columns=feature_names)
        df["Cluster"] = self.labels_
        profiles = df.groupby("Cluster").agg(["mean", "std", "count"]).round(2)
        return profiles

    @staticmethod
    def find_optimal_k(
        X: np.ndarray, max_k: int = 10, random_state: int = 42
    ) -> dict:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        inertias = []
        silhouette_scores = []
        db_scores = []
        ch_scores = []

        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = km.fit_predict(X_scaled)
            inertias.append(km.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, labels))
            db_scores.append(davies_bouldin_score(X_scaled, labels))
            ch_scores.append(calinski_harabasz_score(X_scaled, labels))

        return {
            "k_values": list(range(2, max_k + 1)),
            "inertias": inertias,
            "silhouette_scores": silhouette_scores,
            "davies_bouldin_scores": db_scores,
            "calinski_harabasz_scores": ch_scores,
        }
