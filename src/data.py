import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from loguru import logger
import io
import urllib.request


MALL_CUSTOMERS_URL = (
    "https://raw.githubusercontent.com/SteffiPeTaffy/machineLearningAZ/master/"
    "Machine%20Learning%20A-Z%20Template%20Folder/Part%204%20-%20Clustering/"
    "Section%2024%20-%20K-Means%20Clustering/Mall_Customers.csv"
)

FEATURE_COLS = ["Annual Income (k$)", "Spending Score (1-100)"]


def _fetch_mall_customers() -> pd.DataFrame:
    logger.info("Downloading Mall Customers dataset...")
    try:
        with urllib.request.urlopen(MALL_CUSTOMERS_URL, timeout=10) as resp:
            df = pd.read_csv(io.BytesIO(resp.read()))
        logger.info(f"Downloaded {len(df)} records")
        return df
    except Exception:
        logger.warning("Download failed, using generated data")
        return _generate_fallback_data()


def _generate_fallback_data(n_samples: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    income = np.concatenate([
        rng.normal(25, 8, 50), rng.normal(55, 10, 50),
        rng.normal(85, 12, 50), rng.normal(40, 15, 25),
        rng.normal(90, 8, 25),
    ])
    spending = np.concatenate([
        rng.normal(20, 10, 50), rng.normal(50, 12, 50),
        rng.normal(80, 8, 50), rng.normal(75, 10, 25),
        rng.normal(15, 8, 25),
    ])
    income = np.clip(income, 10, 140)
    spending = np.clip(spending, 1, 100)
    return pd.DataFrame({
        "Annual Income (k$)": income,
        "Spending Score (1-100)": spending,
    })


def generate_customer_data(
    filepath: Optional[Path] = None,
) -> Tuple[np.ndarray, list]:
    if filepath and filepath.exists():
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
    else:
        df = _fetch_mall_customers()
    X = df[FEATURE_COLS].values.astype(np.float64)
    logger.info(f"Loaded {len(X)} samples with {X.shape[1]} features")
    return X, FEATURE_COLS


def save_data(X: np.ndarray, labels: np.ndarray, filepath: Path) -> None:
    df = pd.DataFrame(X, columns=FEATURE_COLS)
    df["Cluster"] = labels
    df.to_csv(filepath, index=False)
    logger.info(f"Data saved to {filepath}")


def load_data(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(filepath)
    X = df[FEATURE_COLS].values
    labels = df["Cluster"].values
    logger.info(f"Loaded {len(X)} samples from {filepath}")
    return X, labels
