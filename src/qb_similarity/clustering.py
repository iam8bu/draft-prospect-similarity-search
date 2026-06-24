"""Dimensionality checks, clustering models, and cluster-comparison helpers."""

import numpy as np
import pandas as pd
import umap
from sklearn.cluster import AgglomerativeClustering, OPTICS, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_distances


def compute_eigengap_table(cluster_data: pd.DataFrame) -> pd.DataFrame:
    """Eigenvalues/eigengaps of the feature covariance matrix, used to eyeball a plausible cluster count."""
    covariance = np.cov(np.transpose(cluster_data))
    eigenvalues = np.linalg.eigvals(covariance)
    eigengaps = -np.diff(eigenvalues)
    eigengap_ratio = np.divide(eigengaps, eigenvalues[:-1])
    nan = [np.nan]
    return pd.DataFrame(
        {
            "Eigenvalues": eigenvalues,
            "Eigengaps": np.concatenate((eigengaps, nan), axis=0),
            "Eigengap_Eigenvalue_ratio": np.concatenate((eigengap_ratio, nan), axis=0),
        }
    )


def umap_embedding(cluster_data: pd.DataFrame, seed: int = 4365) -> pd.DataFrame:
    """2D UMAP projection of the feature matrix, for visualizing cluster structure."""
    np.random.seed(seed)
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(cluster_data)
    return pd.DataFrame(embedding, columns=["component1", "component2"])


def run_agglomerative(
    cluster_data: pd.DataFrame, n_clusters: int, metric: str = "manhattan", linkage: str = "complete", seed: int = 2356
) -> pd.DataFrame:
    """Fit agglomerative clustering and return cluster_data with a `cluster_assignments` column."""
    np.random.seed(seed)
    cluster_data = cluster_data.drop(columns="cluster_assignments", errors="ignore").copy()
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, metric=metric, compute_full_tree=False)
    cluster_data["cluster_assignments"] = model.fit_predict(cluster_data)
    return cluster_data


def find_max_diameter(df: pd.DataFrame, metric: str = "manhattan") -> float:
    """Largest pairwise distance within any cluster (used to gauge cluster tightness)."""
    max_diameters = []
    for label in df["cluster_assignments"].unique():
        members = df[df["cluster_assignments"] == label].drop(columns="cluster_assignments")
        distances = pairwise_distances(members, metric=metric)
        max_diameters.append(np.amax(distances))
    return max(max_diameters)


def evaluate_agglomerative_clusters(
    df: pd.DataFrame, metric: str = "manhattan", linkage: str = "complete", nclusts: tuple[int, int] = (4, 18)
) -> pd.DataFrame:
    """Silhouette score and max cluster diameter across a range of agglomerative cluster counts."""
    df = df.drop(columns="cluster_assignments", errors="ignore").copy()
    silhouette_coefficients = []
    max_diameters = []
    for k in range(nclusts[0], nclusts[1] + 1):
        model = AgglomerativeClustering(n_clusters=k, metric=metric, linkage=linkage)
        cluster_labels = model.fit_predict(df)
        df["cluster_assignments"] = cluster_labels
        silhouette_coefficients.append(silhouette_score(df.drop(columns="cluster_assignments"), cluster_labels, metric=metric))
        max_diameters.append(find_max_diameter(df, metric=metric))

    return pd.DataFrame(
        {
            "NumberClusters": range(nclusts[0], nclusts[1] + 1),
            "ClusterDiameter": max_diameters,
            "SilhouetteCoefficient": silhouette_coefficients,
        }
    )


def run_spectral_clustering(
    cluster_data: pd.DataFrame, n_clusters: int = 15, n_neighbors: int = 50, random_state: int = 0, seed: int = 7788
) -> pd.DataFrame:
    """Fit spectral clustering and return cluster_data with a `cluster_assignments` column."""
    np.random.seed(seed)
    cluster_data = cluster_data.drop(columns="cluster_assignments", errors="ignore").copy()
    model = SpectralClustering(
        assign_labels="discretize", n_clusters=n_clusters, affinity="nearest_neighbors", n_neighbors=n_neighbors, random_state=random_state
    )
    cluster_data["cluster_assignments"] = model.fit_predict(cluster_data)
    return cluster_data


def run_optics(cluster_data: pd.DataFrame, min_samples: int = 10, p: int = 2, seed: int = 4512) -> pd.DataFrame:
    """Fit OPTICS and return cluster_data with a `cluster_assignments` column (-1 = noise)."""
    np.random.seed(seed)
    cluster_data = cluster_data.drop(columns="cluster_assignments", errors="ignore").copy()
    model = OPTICS(p=p, min_samples=min_samples)
    model.fit(cluster_data)
    cluster_data["cluster_assignments"] = model.labels_
    return cluster_data


def cluster_feature_differences(
    cluster_data: pd.DataFrame,
    target_cluster,
    exclude_columns: tuple[str, ...] = ("cluster_assignments", "nfl_success", "athlete_id"),
    threshold: float = 0.15,
) -> pd.DataFrame:
    """Compare mean feature values of `target_cluster` against every other cluster.

    Returns the features whose absolute mean difference exceeds `threshold`, sorted descending.
    """
    target = cluster_data[cluster_data["cluster_assignments"] == target_cluster]
    other = cluster_data[cluster_data["cluster_assignments"] != target_cluster]

    comparison = pd.concat(
        [
            target.drop(columns=list(exclude_columns), errors="ignore").mean().rename("target_cluster"),
            other.drop(columns=list(exclude_columns), errors="ignore").mean().rename("other_clusters"),
        ],
        axis=1,
    )
    comparison["difference"] = comparison["target_cluster"] - comparison["other_clusters"]
    important = comparison[comparison["difference"].abs() >= threshold]
    return important.sort_values(by="difference", ascending=False)
