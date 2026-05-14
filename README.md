# Customer Segmentation

K-Means clustering applied to the **Mall Customers** dataset to segment shoppers by annual income and spending score.

## Overview

Unsupervised learning project that:
- Downloads the real Mall Customers dataset (200 records, 5 features)
- Finds optimal cluster count via **elbow method**, **silhouette analysis**, **Davies-Bouldin**, and **Calinski-Harabasz** scores
- Profiles each segment with mean/std statistics
- Exposes both a **CLI** and an interactive **Streamlit dashboard**

## Quick Start

```bash
pip install -r requirements.txt

# Streamlit web app
streamlit run app.py

# CLI
python -m src.main cluster --n-clusters 5
python -m src.main optimize --max-clusters 10

# Tests
pytest tests/ -v
```

## Docker

```bash
docker compose up --build
# Open http://localhost:8501
```

## Project Structure

```
0003 Customer Segmentation/
├── app.py              # Streamlit dashboard
├── Dockerfile
├── docker-compose.yml
├── src/
│   ├── model.py        # K-Means with optimal-k search
│   ├── data.py         # Mall Customers dataset loader
│   ├── visualizer.py   # Cluster, elbow, silhouette plots
│   ├── main.py         # Typer CLI
│   └── config.py       # Pydantic settings
├── tests/
│   └── test_model.py   # 8 unit tests
└── outputs/
    ├── plots/
    └── models/
```

## Key Metrics Reported

| Metric | Description |
|---|---|
| Inertia | Sum of squared distances to centroids |
| Silhouette Score | Cluster separation quality (-1 to 1) |
| Davies-Bouldin | Average similarity between clusters (lower is better) |
| Calinski-Harabasz | Ratio of between-cluster to within-cluster dispersion |

## License

MIT
# Customer-Segmentation
