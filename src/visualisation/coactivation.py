"""Co-activation analysis for Phase 3 visualisation."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def compute_coactivation_matrix(
    stats: dict,
    top_k: int = 50,
    feature_indices: torch.Tensor | list[int] | None = None,
) -> torch.Tensor:
    """Compute Jaccard similarity of top-K image sets between features.

    Parameters
    ----------
    stats : dict
        Output from collector.compute_feature_stats.
    top_k : int
        Number of most active images to consider for each feature's "active set".
    feature_indices : Tensor or list[int] or None
        Which features to compute the matrix for. If None, computes for all.

    Returns
    -------
    Tensor [n_features, n_features]
        The coactivation matrix, where elements are Jaccard similarities in [0, 1].
    """
    feature_max = stats["feature_max"]  # [n_images, d_sae]

    if feature_indices is None:
        feature_indices = list(range(feature_max.shape[1]))
    elif isinstance(feature_indices, torch.Tensor):
        feature_indices = feature_indices.tolist()

    n_feats = len(feature_indices)
    coactivation = torch.zeros((n_feats, n_feats), dtype=torch.float32)

    # Precompute active sets
    # We take the indices of the top_k images
    active_sets = []
    for feat_idx in feature_indices:
        col = feature_max[:, feat_idx]
        if col.max() == 0:
            active_sets.append(set())
        else:
            n_pick = min(top_k, col.shape[0])
            top_imgs = torch.topk(col, n_pick).indices.tolist()
            # Only consider images where activation > 0
            valid_imgs = [img for img in top_imgs if col[img] > 0]
            active_sets.append(set(valid_imgs))

    for i in range(n_feats):
        set_i = active_sets[i]
        len_i = len(set_i)
        coactivation[i, i] = 1.0
        
        for j in range(i + 1, n_feats):
            set_j = active_sets[j]
            len_j = len(set_j)
            
            if len_i == 0 and len_j == 0:
                score = 0.0
            elif len_i == 0 or len_j == 0:
                score = 0.0
            else:
                inter = len(set_i & set_j)
                union = len_i + len_j - inter
                score = inter / union if union > 0 else 0.0
                
            coactivation[i, j] = score
            coactivation[j, i] = score

    return coactivation


def render_coactivation_heatmap(
    coactivation: torch.Tensor,
    feature_indices: list[int],
    save_path: str | None = None,
) -> plt.Figure:
    """Render the coactivation matrix as a heatmap.

    Parameters
    ----------
    coactivation : Tensor [n, n]
    feature_indices : list[int]
        Labels for the axes.
    save_path : str, optional

    Returns
    -------
    Figure
    """
    import numpy as np
    
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(coactivation.numpy(), cmap="viridis", vmin=0.0, vmax=1.0)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    # Use feature_indices for labels if not too many
    labels = [f"F{idx}" for idx in feature_indices]
    n_show = len(labels)
    
    if n_show <= 50:
        ax.set_xticks(np.arange(n_show))
        ax.set_yticks(np.arange(n_show))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
    else:
        # Show fewer ticks if too many features
        step = max(1, n_show // 20)
        ticks = np.arange(0, n_show, step)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([labels[i] for i in ticks], rotation=90, fontsize=8)
        ax.set_yticklabels([labels[i] for i in ticks], fontsize=8)

    ax.set_title("Feature co-activation (Jaccard similarity of top-50 image sets)", fontsize=11)
    fig.tight_layout()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def find_coactivating_pairs(
    coactivation: torch.Tensor,
    feature_indices: list[int],
    threshold: float = 0.5,
) -> list[tuple[int, int, float]]:
    """Find pairs of features with high co-activation.

    Parameters
    ----------
    coactivation : Tensor [n, n]
    feature_indices : list[int]
    threshold : float
        Minimum Jaccard similarity.

    Returns
    -------
    list[tuple[int, int, float]]
        Sorted list of (feat_i, feat_j, score).
    """
    n = len(feature_indices)
    pairs = []
    
    for i in range(n):
        for j in range(i + 1, n):
            score = float(coactivation[i, j])
            if score >= threshold:
                pairs.append((feature_indices[i], feature_indices[j], score))
                
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs
