"""Latent space visualization: t-SNE and UMAP of EEG feature space.

Generates publication-ready plots showing class separability in the
feature space extracted from EEG windows.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE

from thoughtlink.data.loader import CLASS_NAMES


CLASS_COLORS = {
    "Right Fist": "#3b82f6",
    "Left Fist": "#eab308",
    "Both Fists": "#22c55e",
    "Tongue Tapping": "#f97316",
    "Relax": "#6b7280",
}

COLOR_LIST = [CLASS_COLORS[name] for name in CLASS_NAMES]


def plot_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: list[str] | None = None,
    perplexity: float = 30.0,
    random_state: int = 42,
    title: str = "t-SNE of EEG Feature Space",
    ax: plt.Axes | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
) -> plt.Figure | None:
    """Plot t-SNE embedding of the feature space, color-coded by class.

    Args:
        features: Shape (n_samples, n_features), e.g. (n, 42).
        labels: Shape (n_samples,), integer class labels.
        class_names: Class name list matching label indices.
        perplexity: t-SNE perplexity parameter.
        random_state: Random seed for reproducibility.
        title: Plot title.
        ax: Optional matplotlib Axes. Creates new figure if None.
        show: Whether to call plt.show().
        save_path: Optional path to save the figure.

    Returns:
        Figure if ax was None, else None.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state,
                n_iter=1000, learning_rate="auto", init="pca")
    embedding = tsne.fit_transform(features)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    for idx, name in enumerate(class_names):
        mask = labels == idx
        if mask.sum() == 0:
            continue
        color = CLASS_COLORS.get(name, "#888")
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=color, label=name, s=15, alpha=0.6, edgecolors="none",
        )

    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=8, markerscale=2)

    if fig:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_umap(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: list[str] | None = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    title: str = "UMAP of EEG Feature Space",
    ax: plt.Axes | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
) -> plt.Figure | None:
    """Plot UMAP embedding of the feature space, color-coded by class.

    Falls back to t-SNE if umap-learn is not installed.

    Args:
        features: Shape (n_samples, n_features).
        labels: Shape (n_samples,), integer class labels.
        class_names: Class name list matching label indices.
        n_neighbors: UMAP n_neighbors parameter.
        min_dist: UMAP min_dist parameter.
        random_state: Random seed.
        title: Plot title.
        ax: Optional matplotlib Axes.
        show: Whether to call plt.show().
        save_path: Optional path to save the figure.

    Returns:
        Figure if ax was None, else None.
    """
    try:
        import umap
        reducer = umap.UMAP(
            n_components=2, n_neighbors=n_neighbors,
            min_dist=min_dist, random_state=random_state,
        )
        embedding = reducer.fit_transform(features)
    except ImportError:
        print("umap-learn not installed, falling back to t-SNE")
        return plot_tsne(
            features, labels, class_names=class_names,
            title=title.replace("UMAP", "t-SNE (UMAP unavailable)"),
            ax=ax, show=show, save_path=save_path,
        )

    if class_names is None:
        class_names = CLASS_NAMES

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    for idx, name in enumerate(class_names):
        mask = labels == idx
        if mask.sum() == 0:
            continue
        color = CLASS_COLORS.get(name, "#888")
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=color, label=name, s=15, alpha=0.6, edgecolors="none",
        )

    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(fontsize=8, markerscale=2)

    if fig:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_feature_importance(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: list[str] | None = None,
    n_top: int = 10,
    title: str = "Top Features by Inter-Class Variance",
    ax: plt.Axes | None = None,
    show: bool = True,
) -> plt.Figure | None:
    """Bar plot of top features ranked by inter-class variance (F-ratio proxy).

    Args:
        features: Shape (n_samples, n_features).
        labels: Shape (n_samples,), integer class labels.
        class_names: Class name list.
        n_top: Number of top features to display.
        title: Plot title.
        ax: Optional matplotlib Axes.
        show: Whether to call plt.show().

    Returns:
        Figure if ax was None, else None.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    unique_labels = np.unique(labels)
    n_features = features.shape[1]

    # Compute inter-class variance per feature
    grand_mean = features.mean(axis=0)
    between_var = np.zeros(n_features)
    for lbl in unique_labels:
        mask = labels == lbl
        class_mean = features[mask].mean(axis=0)
        between_var += mask.sum() * (class_mean - grand_mean) ** 2
    between_var /= len(labels)

    # Feature names (band power + Hjorth)
    bands = ["theta", "mu", "beta", "gamma"]
    channels = ["AFF6", "AFp2", "AFp1", "AFF5", "FCz", "CPz"]
    hjorth_params = ["activity", "mobility", "complexity"]

    feat_names = []
    for band in bands:
        for ch in channels:
            feat_names.append(f"{band}_{ch}")
    for param in hjorth_params:
        for ch in channels:
            feat_names.append(f"{param}_{ch}")
    # Pad if features have more columns than expected names
    while len(feat_names) < n_features:
        feat_names.append(f"feat_{len(feat_names)}")

    top_idx = np.argsort(between_var)[::-1][:n_top]

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    names = [feat_names[i] for i in top_idx]
    values = between_var[top_idx]
    ax.barh(range(n_top), values[::-1], color="#3b82f6", alpha=0.8)
    ax.set_yticks(range(n_top))
    ax.set_yticklabels(names[::-1], fontsize=8)
    ax.set_xlabel("Inter-Class Variance")
    ax.set_title(title)

    if fig:
        fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_latent_report(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: list[str] | None = None,
    title: str = "ThoughtLink -- Feature Space Analysis",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Generate combined 2-panel latent space report.

    Panels:
        1. t-SNE scatter plot (class separability)
        2. Top features by inter-class variance

    Args:
        features: Shape (n_samples, n_features).
        labels: Shape (n_samples,), integer class labels.
        class_names: Class name list.
        title: Super-title for the figure.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    plot_tsne(features, labels, class_names=class_names, ax=axes[0], show=False)
    plot_feature_importance(features, labels, class_names=class_names, ax=axes[1], show=False)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
