import streamlit as st
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data import generate_customer_data
from src.model import CustomerSegmentation
from src.visualizer import ClusterVisualizer

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="",
    layout="wide",
)

st.title("Customer Segmentation — K-Means Clustering")
st.markdown("Segment customers by **Annual Income** and **Spending Score** using K-Means clustering.")

X, feature_names = generate_customer_data()

tab1, tab2, tab3 = st.tabs(["Optimal K", "Cluster", "Profiles"])

with tab1:
    st.subheader("Find Optimal Number of Clusters")
    max_k = st.slider("Max K to test", 3, 15, 10, key="max_k")
    if st.button("Run Elbow Analysis", type="primary"):
        with st.spinner("Testing k=2 to k={}...".format(max_k)):
            results = CustomerSegmentation.find_optimal_k(X, max_k=max_k)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Best Silhouette K", str(
                results["k_values"][np.argmax(results["silhouette_scores"])]
            ))
        with col2:
            st.metric("Best CH Score K", str(
                results["k_values"][np.argmax(results["calinski_harabasz_scores"])]
            ))
        chart_data = pd.DataFrame({
            "K": results["k_values"],
            "Inertia": results["inertias"],
            "Silhouette": results["silhouette_scores"],
            "Davies-Bouldin": results["davies_bouldin_scores"],
        })
        st.line_chart(chart_data.set_index("K")[["Inertia"]], y_label="Inertia")
        st.line_chart(chart_data.set_index("K")[["Silhouette"]], y_label="Silhouette Score")

with tab2:
    st.subheader("Run Clustering")
    n_clusters = st.slider("Number of Clusters", 2, 10, 5, key="n_clust")
    if st.button("Segment Customers", type="primary", key="btn_segment"):
        with st.spinner("Clustering..."):
            model = CustomerSegmentation(n_clusters=n_clusters)
            model.fit(X)
            labels = model.get_labels()
            metrics = model.get_metrics(X)
        col1, col2, col3 = st.columns(3)
        col1.metric("Silhouette Score", f"{metrics['silhouette_score']:.3f}")
        col2.metric("Davies-Bouldin", f"{metrics['davies_bouldin_score']:.3f}")
        col3.metric("Inertia", f"{metrics['inertia']:.0f}")
        df_plot = pd.DataFrame(X, columns=feature_names)
        df_plot["Cluster"] = labels.astype(str)
        st.scatter_chart(
            df_plot,
            x="Annual Income (k$)",
            y="Spending Score (1-100)",
            color="Cluster",
            size=60,
        )
        st.caption("Cluster Centers:")
        centers = model.get_cluster_centers()
        st.dataframe(
            pd.DataFrame(centers, columns=feature_names)
            .assign(Cluster=lambda d: [f"Cluster {i}" for i in range(len(d))])
            .set_index("Cluster"),
            use_container_width=True,
        )

with tab3:
    st.subheader("Cluster Profiles")
    n_prof = st.slider("Clusters for profiling", 2, 10, 5, key="n_prof")
    if st.button("Generate Profiles", type="primary", key="btn_prof"):
        model = CustomerSegmentation(n_clusters=n_prof)
        model.fit(X)
        profiles = model.get_cluster_profiles(X, feature_names)
        st.dataframe(profiles, use_container_width=True)
        st.download_button(
            "Download CSV",
            profiles.to_csv().encode("utf-8"),
            "cluster_profiles.csv",
            "text/csv",
        )
