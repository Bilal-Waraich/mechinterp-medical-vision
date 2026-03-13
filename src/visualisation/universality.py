"""Feature universality analysis across datasets/modalities.

Provides tools to measure how universally a SAE feature fires across
different datasets or imaging modalities, and to compute co-activation
correlations between features.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


# ──────────────────────────────────────────────────────────────────────────
# 1. Universality profile
# ──────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_universality_profile(
    feature_idx: int,
    stats: dict,
    dataset_labels: list[str],
    top_k: int = 50,
) -> dict[str, dict[str, float]]:
    """Compute how a feature's top activating images are distributed across datasets.

    For the *top_k* most-activating images of the given feature, tallies
    the fraction belonging to each dataset and the mean activation within
    that dataset subset.  Also computes a universality score defined as
    ``1 - max_fraction``, where higher values indicate the feature fires
    more evenly across datasets.

    Args:
        feature_idx: Index into the SAE dictionary (column of feature_max).
        stats: Output of :func:`collector.compute_feature_stats`.  Must
            contain ``feature_max`` [n_images, d_sae] and ``n_images``.
        dataset_labels: List of length ``n_images`` assigning each image to
            a dataset/modality name, e.g. ``['brain_tumor', 'semicol', ...]``.
        top_k: Number of top-activating images to consider (default 50).

    Returns:
        A dict mapping each dataset name to a sub-dict with keys:

        * ``'fraction'`` (float): fraction of the top-k images belonging to
          this dataset.
        * ``'mean_activation'`` (float): mean ``feature_max`` value across
          the top-k images that belong to this dataset.

        Additionally the returned dict has a top-level key
        ``'universality_score'`` (float) equal to ``1 - max_fraction``.
    """
    feature_max: torch.Tensor = stats["feature_max"]  # [n_images, d_sae]
    n_images = feature_max.shape[0]

    assert len(dataset_labels) == n_images, (
        f"dataset_labels length {len(dataset_labels)} != n_images {n_images}"
    )

    col = feature_max[:, feature_idx]  # [n_images]
    k_actual = min(top_k, n_images)
    top_indices = torch.topk(col, k_actual).indices.tolist()  # list[int]

    # Collect activations per dataset
    dataset_acts: dict[str, list[float]] = {}
    for img_idx in top_indices:
        label = dataset_labels[img_idx]
        if label not in dataset_acts:
            dataset_acts[label] = []
        dataset_acts[label].append(float(col[img_idx].item()))

    result: dict[str, dict[str, float]] = {}
    for ds_name, acts in dataset_acts.items():
        result[ds_name] = {
            "fraction": len(acts) / k_actual,
            "mean_activation": float(np.mean(acts)),
        }

    # Universality score: 1 - fraction of most-dominant dataset
    max_frac = max((v["fraction"] for v in result.values()), default=0.0)
    result["universality_score"] = 1.0 - max_frac  # type: ignore[assignment]

    return result


# ──────────────────────────────────────────────────────────────────────────
# 2. Universality dashboard
# ──────────────────────────────────────────────────────────────────────────

def render_universality_dashboard(
    feature_indices: list[int],
    stats: dict,
    dataset_labels: list[str],
    top_k: int = 50,
    save_path: str | None = None,
) -> plt.Figure:
    """Render a horizontal stacked-bar chart of dataset distribution per feature.

    Each feature appears as one horizontal bar.  Bar segments are coloured
    by dataset using the ``tab10`` colormap.  Features are sorted by
    universality score descending (most universal at top).  The universality
    score is annotated at the right end of each bar.

    Args:
        feature_indices: SAE feature indices to include.
        stats: Output of :func:`collector.compute_feature_stats`.
        dataset_labels: List of length ``n_images`` assigning each image to
            a named dataset/modality.
        top_k: Number of top-activating images considered per feature
            (passed to :func:`compute_universality_profile`).
        save_path: If not ``None``, save the figure to this path at 150 dpi.
            Parent directories are created automatically.

    Returns:
        The rendered ``matplotlib.figure.Figure``.
    """
    # 1. Collect profiles --------------------------------------------------
    profiles: list[tuple[int, dict]] = []
    for fidx in tqdm(feature_indices, desc="Universality profiles"):
        profile = compute_universality_profile(fidx, stats, dataset_labels, top_k=top_k)
        profiles.append((fidx, profile))

    # 2. Sort by universality score descending (most universal at top) ------
    profiles.sort(key=lambda x: x[1].get("universality_score", 0.0), reverse=True)

    # 3. Collect all dataset names and assign colours ----------------------
    all_datasets: list[str] = []
    for _, profile in profiles:
        for k in profile:
            if k != "universality_score" and k not in all_datasets:
                all_datasets.append(k)

    cmap = plt.get_cmap("tab10")
    dataset_colors: dict[str, tuple] = {
        ds: cmap(i % 10) for i, ds in enumerate(all_datasets)
    }

    # 4. Draw figure -------------------------------------------------------
    n = len(profiles)
    fig_h = max(6, n * 0.4)
    fig, ax = plt.subplots(figsize=(12, fig_h))

    y_positions = list(range(n))
    bar_height = 0.6

    for row_i, (fidx, profile) in enumerate(profiles):
        y = y_positions[row_i]
        x_start = 0.0
        u_score = profile.get("universality_score", 0.0)

        for ds_name in all_datasets:
            if ds_name not in profile:
                continue
            frac = profile[ds_name]["fraction"]
            ax.barh(
                y, frac, left=x_start,
                height=bar_height,
                color=dataset_colors[ds_name],
                edgecolor="white",
                linewidth=0.5,
            )
            x_start += frac

        # Annotate universality score
        ax.text(
            1.02, y,
            f"u={u_score:.2f}",
            va="center", ha="left",
            fontsize=7, color="black",
        )

    # 5. Axes cosmetics ----------------------------------------------------
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"F{fidx}" for fidx, _ in profiles], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of top-activating images", fontsize=9)
    ax.set_title(
        "Feature Universality Dashboard",
        fontsize=11, fontweight="bold",
    )

    # Legend
    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, fc=dataset_colors[ds], label=ds)
        for ds in all_datasets
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower right",
        fontsize=7,
        framealpha=0.8,
        title="Dataset",
        title_fontsize=7,
    )

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ──────────────────────────────────────────────────────────────────────────
# 3. Co-activation matrix
# ──────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_co_activation_matrix(
    feature_indices: list[int],
    stats: dict,
) -> torch.Tensor:
    """Compute pairwise Pearson correlations between feature max-activations.

    For each pair of features (i, j) in *feature_indices*, computes the
    Pearson correlation between their ``feature_max`` vectors across all
    images.

    Args:
        feature_indices: List of SAE feature indices to include.
        stats: Output of :func:`collector.compute_feature_stats`.  Must
            contain ``feature_max`` [n_images, d_sae].

    Returns:
        Float32 tensor of shape ``[len(feature_indices), len(feature_indices)]``
        where entry ``[i, j]`` is the Pearson correlation between
        ``feature_max[:, feature_indices[i]]`` and
        ``feature_max[:, feature_indices[j]]``.
    """
    feature_max: torch.Tensor = stats["feature_max"]  # [n_images, d_sae]
    n = len(feature_indices)

    # Extract the relevant columns: [n_images, n_features]
    idx_tensor = torch.tensor(feature_indices, dtype=torch.long)
    sub = feature_max[:, idx_tensor].float()  # [n_images, n_features]

    # Pearson correlation = normalised covariance
    # Centre each feature
    sub_centered = sub - sub.mean(dim=0, keepdim=True)  # [n_images, n_features]
    std = sub_centered.std(dim=0, unbiased=False)  # [n_features]
    std = std.clamp(min=1e-8)

    normed = sub_centered / std.unsqueeze(0)  # [n_images, n_features]
    n_images = sub.shape[0]
    corr = (normed.T @ normed) / n_images  # [n_features, n_features]

    return corr.float()


# ──────────────────────────────────────────────────────────────────────────
# 4. Co-activation heatmap
# ──────────────────────────────────────────────────────────────────────────

def render_co_activation_heatmap(
    feature_indices: list[int],
    stats: dict,
    feature_labels: list[str],
    save_path: str | None = None,
) -> plt.Figure:
    """Render a hierarchically-clustered heatmap of feature co-activations.

    Uses :func:`seaborn.clustermap` with complete-linkage hierarchical
    clustering.  The diverging colormap ``RdBu_r`` is centred at 0 so that
    negative correlations appear red and positive correlations appear blue.

    Args:
        feature_indices: SAE feature indices whose co-activations to plot.
        stats: Output of :func:`collector.compute_feature_stats`.
        feature_labels: Human-readable label for each feature in
            *feature_indices* (must be the same length).
        save_path: If not ``None``, save the figure to this path at 150 dpi.
            Parent directories are created automatically.

    Returns:
        The ``matplotlib.figure.Figure`` from the clustermap (accessed via
        ``clustermap_obj.fig``).
    """
    import seaborn as sns  # noqa: PLC0415

    corr = compute_co_activation_matrix(feature_indices, stats)
    corr_np = corr.numpy()

    n = len(feature_indices)
    fig_size = max(8, 0.3 * n)

    cg = sns.clustermap(
        corr_np,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        xticklabels=feature_labels,
        yticklabels=feature_labels,
        figsize=(fig_size, fig_size),
        linewidths=0.0,
        method="complete",
    )

    cg.ax_heatmap.set_xlabel("Feature", fontsize=8)
    cg.ax_heatmap.set_ylabel("Feature", fontsize=8)
    cg.ax_heatmap.set_title(
        "Feature Co-activation Matrix (Pearson r)",
        fontsize=10, fontweight="bold", pad=10,
    )

    # Rotate tick labels for readability
    plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=90, fontsize=7)
    plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0, fontsize=7)

    fig = cg.fig

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ── Smoke tests ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("  universality.py — smoke tests")
    print("=" * 60)

    torch.manual_seed(0)
    n_images = 50
    d_sae = 2048
    dataset_names = ["brain_tumor", "semicol", "chest_xray"]

    # Fake stats
    stats = {
        "feature_max": torch.rand(n_images, d_sae),
        "feature_argmax": torch.randint(0, 196, (n_images, d_sae)),
        "feature_mean": torch.rand(n_images, d_sae),
        "global_max": torch.rand(d_sae),
        "n_images": n_images,
    }

    # Assign dataset labels
    rng = np.random.default_rng(42)
    labels = [dataset_names[i % len(dataset_names)] for i in range(n_images)]
    rng.shuffle(labels)

    feature_indices = [0, 1, 5, 10, 100]

    # 1. compute_universality_profile
    profile = compute_universality_profile(0, stats, labels, top_k=50)
    assert "universality_score" in profile
    u_score = profile["universality_score"]
    assert 0.0 <= u_score <= 1.0
    print(f"[OK] compute_universality_profile: universality_score={u_score:.3f}")
    for ds in [k for k in profile if k != "universality_score"]:
        print(f"     {ds}: fraction={profile[ds]['fraction']:.2f}, "
              f"mean_act={profile[ds]['mean_activation']:.3f}")

    # 2. render_universality_dashboard
    with tempfile.TemporaryDirectory() as tmpdir:
        save_p = str(Path(tmpdir) / "universality_dashboard.png")
        fig = render_universality_dashboard(
            feature_indices=feature_indices,
            stats=stats,
            dataset_labels=labels,
            save_path=save_p,
        )
        plt.close(fig)
        assert Path(save_p).exists()
        print(f"[OK] render_universality_dashboard: saved")

    # 3. compute_co_activation_matrix
    corr = compute_co_activation_matrix(feature_indices, stats)
    assert corr.shape == (len(feature_indices), len(feature_indices))
    assert corr.dtype == torch.float32
    # Diagonal should be 1.0
    diag = corr.diagonal()
    assert (diag - 1.0).abs().max() < 1e-4, f"diagonal not 1: {diag}"
    print(f"[OK] compute_co_activation_matrix: shape={corr.shape}, diag≈1")

    # 4. render_co_activation_heatmap
    feat_labels = [f"F{i}" for i in feature_indices]
    with tempfile.TemporaryDirectory() as tmpdir:
        save_p = str(Path(tmpdir) / "coact_heatmap.png")
        fig = render_co_activation_heatmap(
            feature_indices=feature_indices,
            stats=stats,
            feature_labels=feat_labels,
            save_path=save_p,
        )
        plt.close(fig)
        assert Path(save_p).exists()
        print(f"[OK] render_co_activation_heatmap: saved")

    print("\nAll universality smoke tests passed ✓")