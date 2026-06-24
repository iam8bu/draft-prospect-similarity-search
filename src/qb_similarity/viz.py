"""Plotting helpers shared by the analysis notebook and the Streamlit app.

Every function returns a matplotlib Figure (via `fig.show()` in a notebook, or
`st.pyplot(fig)` in Streamlit) instead of calling `plt.show()` directly.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_years_played_distribution(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots()
    ax.hist(df["years_played"], bins=14, edgecolor="black")
    ax.set_title("Distribution of Years Played")
    ax.set_xlabel("Years Played")
    ax.set_ylabel("Number of QBs")
    return fig


def plot_av_per_year_distribution(df: pd.DataFrame) -> plt.Figure:
    filtered = df["av_per_year"][df["av_per_year"] != 0]
    fig, ax = plt.subplots()
    ax.hist(filtered, bins=20, edgecolor="black")
    ax.set_title("Distribution of AV per Year")
    ax.set_xlabel("AV per Year")
    ax.set_ylabel("Number of QBs")
    return fig


def plot_passer_rating_vs_av(df: pd.DataFrame) -> plt.Figure:
    filtered = df[df["av_per_year"] != 0]
    fig, ax = plt.subplots()
    ax.scatter(filtered["passer_rating"], filtered["av_per_year"])
    ax.set_title("College Passer Rating vs NFL AV per Year")
    ax.set_xlabel("College Passer Rating")
    ax.set_ylabel("AV per Year")
    return fig


def plot_qb_comparison(df: pd.DataFrame, name1: str, name2: str, metrics: list[str]) -> plt.Figure:
    """Bar chart comparing two QBs (matched by Clean_Name) across the given metric columns."""
    row1 = df[df["Clean_Name"] == name1].iloc[0]
    row2 = df[df["Clean_Name"] == name2].iloc[0]

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - width / 2, [row1[m] for m in metrics], width, label=name1.title())
    ax.bar(x + width / 2, [row2[m] for m in metrics], width, label=name2.title())
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title("QB Comparison")
    ax.legend()
    return fig


def plot_feature_distributions(df: pd.DataFrame, columns: list[str]) -> list[plt.Figure]:
    """One histogram+KDE per numeric column, for spotting skew before transforming."""
    figs = []
    for col in columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(data=df, x=col, kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        fig.tight_layout()
        figs.append(fig)
    return figs


def plot_cluster_assignments(embedding_df: pd.DataFrame, cluster_assignments: pd.Series) -> plt.Figure:
    """UMAP scatter colored by cluster, with marker style indicating NFL success."""
    df = embedding_df.drop(columns=["nfl_success"], errors="ignore").copy()
    # Index-aligned assignment (not `.values`): callers sometimes pass a cluster_assignments
    # Series already filtered down to a subset of rows (e.g. dropping OPTICS noise points),
    # and rows with no match should show up as NaN rather than raising a length mismatch.
    df["cluster_assignments"] = cluster_assignments.astype("category")
    df["nfl_success"] = embedding_df["nfl_success"]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(
        data=df, x="component1", y="component2",
        hue="cluster_assignments", style="nfl_success",
        markers=["o", "^"], s=10, alpha=0.1, ax=ax,
    )
    legend = ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.set_title("UMAP projection of cluster assignments")
    return fig


def plot_agglomerative_evaluation(evaluation_df: pd.DataFrame) -> plt.Figure:
    """Side-by-side max cluster diameter and silhouette score vs. number of clusters."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(evaluation_df["NumberClusters"], evaluation_df["ClusterDiameter"])
    axes[0].set_xlabel("Number of clusters")
    axes[0].set_ylabel("Maximum cluster diameter")
    axes[0].set_title("Maximum cluster diameter vs. number of clusters")
    axes[1].plot(evaluation_df["NumberClusters"], evaluation_df["SilhouetteCoefficient"])
    axes[1].set_xlabel("Number of clusters")
    axes[1].set_ylabel("Silhouette Coefficients")
    axes[1].set_title("Silhouette coefficient vs. number of clusters")
    return fig


def plot_clusters_by_factor(df: pd.DataFrame, factor: str, kind: str = "violin") -> plt.Figure:
    """Violin (or box) plot of one feature across clusters, split by NFL success."""
    fig, ax = plt.subplots(figsize=(10, 4))
    if kind == "violin":
        sns.violinplot(x="cluster_assignments", y=factor, hue="nfl_success", data=df, dodge=True, ax=ax)
    else:
        sns.boxplot(x="cluster_assignments", y=factor, hue="nfl_success", data=df, ax=ax)
    ax.set_title(f"{factor} by cluster number")
    ax.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    return fig
