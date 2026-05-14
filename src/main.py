import typer
import sys
from loguru import logger

from .config import settings
from .data import generate_customer_data, save_data
from .model import CustomerSegmentation
from .visualizer import ClusterVisualizer

app = typer.Typer(help="Customer Segmentation with K-Means CLI")

logger.remove()
logger.add(sys.stderr, level=settings.log_level)


@app.command()
def cluster(
    n_clusters: int = typer.Option(3, help="Number of clusters"),
    visualize: bool = typer.Option(True, help="Generate visualizations")
):
    """Perform customer segmentation clustering."""
    logger.info(f"Starting K-Means clustering with {n_clusters} clusters...")
    
    X, feature_names = generate_customer_data()
    
    segmentation = CustomerSegmentation(n_clusters=n_clusters)
    segmentation.fit(X)
    
    labels = segmentation.get_labels()
    centers = segmentation.get_cluster_centers()
    metrics = segmentation.get_metrics(X)
    
    logger.info(f"Inertia: {metrics['inertia']:.2f}")
    logger.info(f"Silhouette Score: {metrics['silhouette_score']:.3f}")
    
    logger.info("\nCluster Centers:")
    for i, center in enumerate(centers):
        logger.info(f"Cluster {i+1}: Income=${center[0]:.0f}k, Spending Score={center[1]:.0f}")
    
    if visualize:
        vis = ClusterVisualizer()
        vis.plot_clusters(
            X, labels, centers,
            save_path=settings.plots_dir / "clusters.png"
        )
    
    save_data(X, labels, settings.data_dir / "segmented_customers.csv")
    logger.success("Clustering completed!")


@app.command()
def optimize(max_clusters: int = typer.Option(10, help="Maximum clusters to test")):
    """Find optimal number of clusters using elbow method."""
    logger.info("Finding optimal number of clusters...")
    
    X, _ = generate_customer_data()
    
    inertias = []
    silhouette_scores = []
    
    for k in range(2, max_clusters + 1):
        seg = CustomerSegmentation(n_clusters=k)
        seg.fit(X)
        metrics = seg.get_metrics(X)
        inertias.append(metrics['inertia'])
        silhouette_scores.append(metrics['silhouette_score'])
        logger.info(f"k={k}: Inertia={metrics['inertia']:.2f}, Silhouette={metrics['silhouette_score']:.3f}")
    
    vis = ClusterVisualizer()
    vis.plot_elbow_curve(
        range(2, max_clusters + 1), inertias,
        save_path=settings.plots_dir / "elbow_curve.png"
    )
    vis.plot_silhouette_scores(
        range(2, max_clusters + 1), silhouette_scores,
        save_path=settings.plots_dir / "silhouette_scores.png"
    )
    
    logger.success("Optimization completed!")


if __name__ == "__main__":
    app()
