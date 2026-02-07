"""Temporal stability visualization: action vs time color-coded timeline.

Generates publication-ready plots showing how decoded actions evolve over time,
highlighting the effect of the stability pipeline (confidence + hysteresis +
debounce + majority voting).
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from thoughtlink.bridge.brain_policy import StepResult


ACTION_COLORS = {
    "RIGHT": "#3b82f6",
    "LEFT": "#eab308",
    "FORWARD": "#22c55e",
    "STOP": "#ef4444",
}

ACTION_ORDER = ["STOP", "LEFT", "FORWARD", "RIGHT"]


def plot_action_timeline(
    results: list[StepResult],
    title: str = "Action Timeline",
    ground_truth: str | None = None,
    ax: plt.Axes | None = None,
    show: bool = True,
) -> plt.Figure | None:
    """Plot color-coded action vs time scatter.

    Args:
        results: List of StepResult from BrainPolicy.
        title: Plot title.
        ground_truth: Expected class label (for annotation).
        ax: Optional matplotlib Axes. Creates new figure if None.
        show: Whether to call plt.show().

    Returns:
        Figure if ax was None, else None.
    """
    if not results:
        return None

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 2.5))

    times = [r.timestamp_s for r in results]
    actions = [r.action for r in results]
    action_to_y = {a: i for i, a in enumerate(ACTION_ORDER)}
    ys = [action_to_y.get(a, 0) for a in actions]
    colors = [ACTION_COLORS.get(a, "#888") for a in actions]

    ax.scatter(times, ys, c=colors, s=30, edgecolors="white", linewidths=0.3)
    ax.set_yticks(range(len(ACTION_ORDER)))
    ax.set_yticklabels(ACTION_ORDER)
    ax.set_xlabel("Time (s)")
    ax.set_title(title)

    if ground_truth:
        ax.annotate(
            f"GT: {ground_truth}",
            xy=(0.98, 0.95), xycoords="axes fraction",
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
        )

    patches = [
        mpatches.Patch(color=ACTION_COLORS[a], label=a)
        for a in ACTION_ORDER
    ]
    ax.legend(handles=patches, loc="upper left", fontsize=7, ncol=4)

    if fig:
        fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_confidence_trace(
    results: list[StepResult],
    threshold: float = 0.6,
    title: str = "Confidence Over Time",
    ax: plt.Axes | None = None,
    show: bool = True,
) -> plt.Figure | None:
    """Plot confidence values over time with threshold line.

    Args:
        results: List of StepResult from BrainPolicy.
        threshold: Confidence threshold to draw as reference.
        title: Plot title.
        ax: Optional matplotlib Axes.
        show: Whether to call plt.show().

    Returns:
        Figure if ax was None, else None.
    """
    if not results:
        return None

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 2.5))

    times = [r.timestamp_s for r in results]
    confs = [r.confidence for r in results]
    colors = [ACTION_COLORS.get(r.action, "#888") for r in results]

    ax.scatter(times, confs, c=colors, s=15, alpha=0.7)
    ax.axhline(threshold, color="red", ls="--", lw=1, label=f"threshold={threshold}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Confidence")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(fontsize=8)

    if fig:
        fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_probability_heatmap(
    results: list[StepResult],
    class_names: list[str] | None = None,
    title: str = "Class Probabilities Over Time",
    ax: plt.Axes | None = None,
    show: bool = True,
) -> plt.Figure | None:
    """Plot heatmap of class probabilities over time.

    Args:
        results: List of StepResult from BrainPolicy.
        class_names: Class labels for y-axis.
        title: Plot title.
        ax: Optional matplotlib Axes.
        show: Whether to call plt.show().

    Returns:
        Figure if ax was None, else None.
    """
    if not results:
        return None

    from thoughtlink.data.loader import CLASS_NAMES as DEFAULT_NAMES
    if class_names is None:
        class_names = DEFAULT_NAMES

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))

    times = [r.timestamp_s for r in results]
    prob_matrix = np.array([r.probs for r in results]).T  # (n_classes, n_steps)

    ax.imshow(
        prob_matrix,
        aspect="auto",
        cmap="YlOrRd",
        extent=[times[0], times[-1], len(class_names) - 0.5, -0.5],
        vmin=0, vmax=1,
    )
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_title(title)

    if fig:
        fig.colorbar(ax.images[0], ax=ax, label="P(class)")
        fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_full_report(
    results: list[StepResult],
    ground_truth: str | None = None,
    threshold: float = 0.6,
    title: str = "ThoughtLink — Temporal Stability Report",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Generate a combined 3-panel stability report.

    Panels:
        1. Action timeline (scatter)
        2. Confidence trace
        3. Probability heatmap

    Args:
        results: List of StepResult.
        ground_truth: Expected class label.
        threshold: Confidence threshold.
        title: Super-title.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    plot_action_timeline(results, title="Action Timeline", ground_truth=ground_truth, ax=axes[0], show=False)
    plot_confidence_trace(results, threshold=threshold, ax=axes[1], show=False)
    plot_probability_heatmap(results, ax=axes[2], show=False)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
