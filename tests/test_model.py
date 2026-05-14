import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import CustomerSegmentation
from src.data import generate_customer_data, save_data, load_data


class TestCustomerSegmentation:
    @pytest.fixture
    def sample_data(self):
        X, names = generate_customer_data()
        return X, names

    def test_fit_returns_self(self, sample_data):
        X, _ = sample_data
        model = CustomerSegmentation(n_clusters=3)
        result = model.fit(X)
        assert result is model

    def test_labels_shape(self, sample_data):
        X, _ = sample_data
        model = CustomerSegmentation(n_clusters=3)
        model.fit(X)
        labels = model.get_labels()
        assert len(labels) == len(X)
        assert set(labels).issubset({0, 1, 2})

    def test_predict(self, sample_data):
        X, _ = sample_data
        model = CustomerSegmentation(n_clusters=3)
        model.fit(X)
        preds = model.predict(X[:5])
        assert len(preds) == 5

    def test_metrics_range(self, sample_data):
        X, _ = sample_data
        model = CustomerSegmentation(n_clusters=3)
        model.fit(X)
        metrics = model.get_metrics(X)
        assert -1 <= metrics["silhouette_score"] <= 1
        assert metrics["inertia"] > 0
        assert metrics["davies_bouldin_score"] > 0

    def test_find_optimal_k(self, sample_data):
        X, _ = sample_data
        results = CustomerSegmentation.find_optimal_k(X, max_k=5)
        assert len(results["k_values"]) == 4
        assert len(results["inertias"]) == 4
        assert len(results["silhouette_scores"]) == 4
        assert results["inertias"] == sorted(results["inertias"], reverse=True)

    def test_unfitted_raises(self):
        model = CustomerSegmentation()
        with pytest.raises(ValueError):
            model.get_labels()
        with pytest.raises(ValueError):
            model.get_metrics(np.array([[1.0, 2.0]]))

    def test_save_and_load(self, sample_data, tmp_path):
        X, _ = sample_data
        model = CustomerSegmentation(n_clusters=3)
        model.fit(X)
        labels = model.get_labels()
        csv_path = tmp_path / "test.csv"
        save_data(X, labels, csv_path)
        X2, labels2 = load_data(csv_path)
        assert X2.shape == X.shape
        assert len(labels2) == len(labels)
