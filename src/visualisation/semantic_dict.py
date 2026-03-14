"""Semantic-dictionary renderer for Phase 3 visualisation.

For each SAE feature, crops the original images at the most-activating
patch to produce a visual "dictionary" showing what the feature detects.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for servers / CI
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure ``src/`` is importable
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from sae_models import BatchTopKSAE  # noqa: E402
from visualisation.vis_config import VisConfig  # noqa: E402
from visualisation.collector import get_top_images_per_feature  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────
# Patch cropping
# ──────────────────────────────────────────────────────────────────────────

def get_patch_crop(
    img: Image.Image,
    patch_idx: int,
    spatial_h: int,
    spatial_w: int,
    context_patches: int = 1,
) -> Image.Image:
    """Crop *img* to the region covered by *patch_idx*, with context.

    Parameters
    ----------
    img : PIL.Image
        Original image (any resolution).
    patch_idx : int
        Flat index in ``0 .. spatial_h * spatial_w - 1``.
    spatial_h, spatial_w : int
        Patch-grid dimensions (e.g. 14×14 for Swin stage-2).
    context_patches : int
        Extra patches to include on every side (clamped to image bounds).

    Returns
    -------
    PIL.Image
        Cropped region.
    """
    patch_row = patch_idx // spatial_w
    patch_col = patch_idx % spatial_w

    img_w, img_h = img.size
    cell_h = img_h / spatial_h
    cell_w = img_w / spatial_w

    # Expand by context, clamp to grid bounds
    r0 = max(patch_row - context_patches, 0)
    r1 = min(patch_row + context_patches + 1, spatial_h)
    c0 = max(patch_col - context_patches, 0)
    c1 = min(patch_col + context_patches + 1, spatial_w)

    # Pixel coordinates (left, upper, right, lower)
    left = int(c0 * cell_w)
    upper = int(r0 * cell_h)
    right = int(c1 * cell_w)
    lower = int(r1 * cell_h)

    return img.crop((left, upper, right, lower))


# ──────────────────────────────────────────────────────────────────────────
# Single-feature semantic dictionary
# ──────────────────────────────────────────────────────────────────────────

def render_semantic_dictionary(
    feature_idx: int,
    stats: dict,
    image_paths: list[Path],
    config: VisConfig,
    feature_label: str | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Render a 3×3 grid of top-activating patch crops for one feature.

    Parameters
    ----------
    feature_idx : int
        Index into the SAE dictionary.
    stats : dict
        Output of :func:`collector.compute_feature_stats`.
    image_paths : list[Path]
        Sorted image paths (same order as activations).
    config : VisConfig
        Supplies ``spatial_h``, ``spatial_w``, ``top_k_images``.
    feature_label : str, optional
        Human-readable label appended to the figure title.
    save_path : str, optional
        If given, saves the figure to this path at 150 dpi.

    Returns
    -------
    matplotlib.figure.Figure
    """
    k = config.top_k_images  # default 9

    # Top-k image indices for this feature  [k]
    top_imgs = get_top_images_per_feature(stats, k=k)[:, feature_idx]  # [k]

    feature_max = stats["feature_max"]       # [n_images, d_sae]
    feature_argmax = stats["feature_argmax"]  # [n_images, d_sae]

    n_show = min(k, len(top_imgs))
    ncols = 3
    nrows = max(1, (n_show + ncols - 1) // ncols)

    fig = plt.figure(figsize=(3 * ncols, 3 * nrows + 3.0))
    gs = gridspec.GridSpec(nrows + 1, ncols, figure=fig, hspace=0.4, wspace=0.3)

    for i in range(nrows * ncols):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        if i >= n_show:
            ax.axis("off")
            continue

        img_idx = int(top_imgs[i].item())
        patch_idx = int(feature_argmax[img_idx, feature_idx].item())
        act_val = float(feature_max[img_idx, feature_idx].item())

        img = Image.open(image_paths[img_idx]).convert("RGB")
        crop = get_patch_crop(
            img,
            patch_idx,
            config.spatial_h,
            config.spatial_w,
            context_patches=2,
        )

        ax.imshow(crop)
        ax.set_title(f"act={act_val:.3f}", fontsize=8)
        ax.axis("off")

    # Histogram below the grid
    ax_hist = fig.add_subplot(gs[nrows, :])
    col = feature_max[:, feature_idx].float().cpu().numpy()
    gmax = float(stats["global_max"][feature_idx].item())

    if gmax == 0.0:
        ax_hist.text(0.5, 0.5, "dead feature", ha="center", va="center", fontsize=12, color="red")
        ax_hist.axis("off")
    else:
        ax_hist.hist(col, bins=40, color="steelblue", edgecolor="none")
        ax_hist.set_xlabel("max activation per image", fontsize=9)
        ax_hist.set_ylabel("count", fontsize=9)
        mean_val = float(col.mean())
        std_val = float(col.std())
        ax_hist.set_title(f"F{feature_idx} activation distribution  (mean={mean_val:.2f}, std={std_val:.2f})", fontsize=10)
        ax_hist.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label="mean")
        ax_hist.legend(fontsize=8)

    # Figure title
    title = f"Feature {feature_idx}"
    if feature_label:
        title += f"  |  {feature_label}"
    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.01)

    # Caption
    fig.text(
        0.5, -0.01,
        "top patch crops · context=2 patches",
        ha="center", fontsize=7, color="grey",
    )

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ──────────────────────────────────────────────────────────────────────────
# Batch dashboard
# ──────────────────────────────────────────────────────────────────────────

def render_feature_dashboard(
    feature_indices: list[int],
    stats: dict,
    image_paths: list[Path],
    config: VisConfig,
    feature_labels: dict[int, str] | None = None,
    output_dir: str | None = None,
) -> None:
    """Render semantic-dictionary pages for multiple features.

    Parameters
    ----------
    feature_indices : list[int]
        Features to render.
    stats : dict
        Output of :func:`collector.compute_feature_stats`.
    image_paths : list[Path]
        Sorted image paths.
    config : VisConfig
        Visualisation settings.
    feature_labels : dict, optional
        Mapping ``feature_idx → human label``.
    output_dir : str, optional
        If given, each figure is saved as
        ``{output_dir}/feature_{idx:05d}.png``.
    """
    labels = feature_labels or {}

    for feat_idx in tqdm(feature_indices, desc="Rendering features"):
        sp = None
        if output_dir:
            sp = str(Path(output_dir) / f"feature_{feat_idx:05d}.png")

        fig = render_semantic_dictionary(
            feature_idx=feat_idx,
            stats=stats,
            image_paths=image_paths,
            config=config,
            feature_label=labels.get(feat_idx),
            save_path=sp,
        )
        plt.close(fig)

    n = len(feature_indices)
    if output_dir:
        print(f"Rendered {n} feature dashboards to {output_dir}")
    else:
        print(f"Rendered {n} feature dashboards (not saved — no output_dir)")


# ──────────────────────────────────────────────────────────────────────────
# render_semantic_dictionary_v2
# ──────────────────────────────────────────────────────────────────────────

def render_semantic_dictionary_v2(
    feature_idx: int,
    stats: dict,
    image_paths: list[Path],
    sae: BatchTopKSAE,
    config: VisConfig,
    feature_label: str = "",
    save_path: str | None = None,
) -> plt.Figure:
    """Render an enhanced semantic dictionary with histogram, spatial map, and similarity bars.

    Layout (2 rows):

    * **Row 1**: 3×3 grid of top-9 activating patch crops (same as
      :func:`render_semantic_dictionary`).
    * **Row 2 (a)**: Activation histogram of ``feature_max[:, feature_idx]``
      with a log-scale y-axis.  A vertical dashed line marks the top-9
      activation threshold.  Subtitle reports percentage of images where
      the feature fires (value > 0).
    * **Row 2 (b)**: Global spatial frequency map — 14×14 heatmap (nearest-
      neighbour upsampled to 112×112) showing the fraction of images where
      ``feature_max[img, feature_idx] > 0`` for each spatial patch position.
    * **Row 2 (c)**: Horizontal bar chart of the top-10 most cosine-similar
      decoder atoms to ``W_dec[feature_idx]``, excluding self.

    Args:
        feature_idx: Index into the SAE dictionary.
        stats: Output of :func:`collector.compute_feature_stats`.
        image_paths: Sorted image paths (same order as activations).
        sae: Trained :class:`BatchTopKSAE`.  Needs ``.W_dec`` and ``.W_enc``.
        config: Supplies ``spatial_h``, ``spatial_w``, ``top_k_images``.
        feature_label: Human-readable label appended to the figure title.
        save_path: If not ``None``, save the figure to this path at 150 dpi.

    Returns:
        The rendered ``matplotlib.figure.Figure``.
    """
    k = config.top_k_images  # default 9

    feature_max: torch.Tensor = stats["feature_max"]       # [n_images, d_sae]
    feature_argmax: torch.Tensor = stats["feature_argmax"]  # [n_images, d_sae]
    n_images = feature_max.shape[0]

    # Top-k image indices for this feature [k]
    top_imgs = get_top_images_per_feature(stats, k=k)[:, feature_idx]  # [k]
    n_show = min(k, len(top_imgs), len(image_paths))

    # ── Figure layout ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 8))
    gs_outer = gridspec.GridSpec(
        2, 1, figure=fig, hspace=0.45, height_ratios=[3, 2]
    )

    # Row 1: 3×3 patch grid
    gs_top = gridspec.GridSpecFromSubplotSpec(
        3, 3, subplot_spec=gs_outer[0], hspace=0.3, wspace=0.2
    )
    for i in range(9):
        ax = fig.add_subplot(gs_top[i // 3, i % 3])
        if i >= n_show or i >= len(image_paths):
            ax.axis("off")
            continue

        img_idx = int(top_imgs[i].item())
        patch_idx = int(feature_argmax[img_idx, feature_idx].item())
        act_val = float(feature_max[img_idx, feature_idx].item())

        img = Image.open(image_paths[img_idx]).convert("RGB")
        crop = get_patch_crop(
            img, patch_idx,
            config.spatial_h, config.spatial_w,
            context_patches=2,
        )
        ax.imshow(crop)
        ax.set_title(f"act={act_val:.3f}", fontsize=7)
        ax.axis("off")

    # Row 2: three panels
    gs_bot = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_outer[1], hspace=0.3, wspace=0.4
    )

    # ── (a) Activation histogram ──────────────────────────────────────────
    ax_hist = fig.add_subplot(gs_bot[0])
    col = feature_max[:, feature_idx].float().cpu().numpy()
    pct_firing = float((col > 0).mean() * 100)

    ax_hist.hist(col[col > 0], bins=30, color="steelblue", edgecolor="none")
    ax_hist.set_yscale("log")

    # Threshold line: lowest activation in top-k
    if n_show > 0:
        thresh_val = float(feature_max[top_imgs[n_show - 1], feature_idx].item())
        ax_hist.axvline(thresh_val, color="red", linestyle="--", linewidth=1.0,
                        label=f"top-{n_show} thresh")
        ax_hist.legend(fontsize=6)

    ax_hist.set_xlabel("Activation", fontsize=8)
    ax_hist.set_ylabel("Count (log)", fontsize=8)
    ax_hist.set_title(
        f"Activation histogram\nfires on {pct_firing:.1f}% of images",
        fontsize=8,
    )
    ax_hist.tick_params(labelsize=7)

    # ── (b) Spatial frequency map ─────────────────────────────────────────
    ax_spat = fig.add_subplot(gs_bot[1])

    # feature_argmax tells us the patch with max activation per image.
    # Build fraction-of-images-firing per spatial position.
    fires = (feature_max[:, feature_idx] > 0)  # [n_images] bool
    n_patches = config.n_patches
    spatial_h, spatial_w = config.spatial_h, config.spatial_w

    freq_map = torch.zeros(n_patches, dtype=torch.float32)
    if fires.any():
        argmax_col = feature_argmax[:, feature_idx]  # [n_images] patch index
        for img_i in range(n_images):
            if fires[img_i]:
                freq_map[argmax_col[img_i]] += 1.0
        freq_map = freq_map / n_images

    freq_2d = freq_map.reshape(spatial_h, spatial_w)  # [H, W]

    # Nearest-neighbour upsample to 112×112
    freq_up = F.interpolate(
        freq_2d[None, None].float(),
        size=(112, 112),
        mode="nearest",
    ).squeeze().numpy()

    im = ax_spat.imshow(freq_up, cmap="viridis", interpolation="none",
                        vmin=0)
    ax_spat.set_title("Spatial firing frequency\n(fraction of images)", fontsize=8)
    ax_spat.axis("off")
    fig.colorbar(im, ax=ax_spat, fraction=0.046, pad=0.04)

    # ── (c) Decoder cosine similarity ─────────────────────────────────────
    ax_sim = fig.add_subplot(gs_bot[2])

    with torch.no_grad():
        W_dec = sae.W_dec  # [d_sae, d_model]
        query = W_dec[feature_idx]  # [d_model]
        normed = F.normalize(W_dec, dim=-1)  # [d_sae, d_model]
        query_norm = F.normalize(query.unsqueeze(0), dim=-1)  # [1, d_model]
        sims = (query_norm @ normed.T).squeeze(0)  # [d_sae]
        sims[feature_idx] = -2.0  # exclude self
        n_top = min(10, sims.shape[0])
        top10 = torch.topk(sims, n_top)

    top10_vals = top10.values.cpu().numpy()
    top10_idxs = top10.indices.cpu().numpy()
    labels_bar = [f"F{int(i)}" for i in top10_idxs]

    # Flip so highest similarity is at top
    ax_sim.barh(
        list(range(n_top))[::-1], top10_vals,
        color="coral", edgecolor="none",
    )
    ax_sim.set_yticks(list(range(n_top))[::-1])
    ax_sim.set_yticklabels(labels_bar, fontsize=7)
    ax_sim.set_xlabel("Cosine similarity", fontsize=8)
    ax_sim.set_title(f"Top-{n_top} similar decoder atoms\n(cosine sim)", fontsize=8)
    ax_sim.tick_params(labelsize=7)

    # ── Super-title ───────────────────────────────────────────────────────
    title = f"Feature {feature_idx}"
    if feature_label:
        title += f"  |  {feature_label}"
    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.01)

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ──────────────────────────────────────────────────────────────────────────
# Feature card (Neuronpedia-style)
# ──────────────────────────────────────────────────────────────────────────

def render_feature_card(
    feature_idx: int,
    stats: dict,
    image_paths: list[Path],
    sae: BatchTopKSAE,
    config: VisConfig,
    dataset_labels: list[str] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Render a Neuronpedia-style single-feature summary card (A4 figure).

    The card is a 2×2 grid of panels:

    * **Top-left**: 3×3 semantic dictionary grid (top-9 patch crops).
    * **Top-right**: Global activation frequency map (14×14 heatmap of how
      often the feature fires across all images) plus an activation histogram
      side by side.
    * **Bottom-left**: Decoder cosine-similarity bar chart — top-15 most
      similar features to ``W_dec[feature_idx]`` (excluding self).
    * **Bottom-right**: Dataset universality stacked bar (from
      :mod:`visualisation.universality`).  Rendered only when
      *dataset_labels* is provided; otherwise a placeholder is shown.

    The figure title reports the feature index, its global max activation,
    and the percentage of images on which it fires.

    Args:
        feature_idx: Index into the SAE dictionary.
        stats: Output of :func:`collector.compute_feature_stats`.
        image_paths: Sorted image paths (same order as activations).
        sae: Trained :class:`BatchTopKSAE`.  Needs ``.W_dec`` and ``.W_enc``.
        config: Visualisation settings (``spatial_h``, ``spatial_w``,
            ``top_k_images``).
        dataset_labels: Optional list of length ``n_images`` assigning each
            image to a named dataset/modality.  If ``None``, the bottom-right
            panel displays a "no dataset labels" placeholder.
        save_path: If not ``None``, save the figure at 150 dpi.

    Returns:
        The rendered ``matplotlib.figure.Figure``.
    """
    from visualisation.universality import (  # noqa: PLC0415
        compute_universality_profile,
    )

    feature_max: torch.Tensor = stats["feature_max"]       # [n_images, d_sae]
    feature_argmax: torch.Tensor = stats["feature_argmax"]  # [n_images, d_sae]
    global_max: torch.Tensor = stats["global_max"]          # [d_sae]
    n_images = feature_max.shape[0]

    gmax_val = float(global_max[feature_idx].item())
    col = feature_max[:, feature_idx].float().cpu().numpy()
    pct_firing = float((col > 0).mean() * 100)

    k = config.top_k_images
    top_imgs = get_top_images_per_feature(stats, k=k)[:, feature_idx]  # [k]
    n_show = min(k, len(top_imgs), len(image_paths))

    # ── Figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 20))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── TOP LEFT: 3×3 patch grid ──────────────────────────────────────────
    gs_tl = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs[0, 0],
                                             hspace=0.25, wspace=0.15)
    for i in range(9):
        ax = fig.add_subplot(gs_tl[i // 3, i % 3])
        if i >= n_show or not image_paths:
            ax.axis("off")
            continue
        img_idx = int(top_imgs[i].item())
        patch_idx = int(feature_argmax[img_idx, feature_idx].item())
        act_val = float(feature_max[img_idx, feature_idx].item())
        img = Image.open(image_paths[img_idx]).convert("RGB")
        crop = get_patch_crop(img, patch_idx,
                              config.spatial_h, config.spatial_w,
                              context_patches=2)
        ax.imshow(crop)
        ax.set_title(f"act={act_val:.3f}", fontsize=7)
        ax.axis("off")

    # ── TOP RIGHT: freq map + histogram ──────────────────────────────────
    gs_tr = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 1],
                                             wspace=0.4)

    # (a) Spatial frequency map
    spatial_h, spatial_w = config.spatial_h, config.spatial_w
    n_patches = config.n_patches
    fires = (feature_max[:, feature_idx] > 0)  # [n_images] bool
    freq_map = torch.zeros(n_patches, dtype=torch.float32)
    if fires.any():
        argmax_col = feature_argmax[:, feature_idx]
        for img_i in range(n_images):
            if fires[img_i]:
                freq_map[argmax_col[img_i]] += 1.0
        freq_map = freq_map / n_images
    freq_2d = freq_map.reshape(spatial_h, spatial_w)
    freq_up = F.interpolate(
        freq_2d[None, None].float(), size=(112, 112), mode="nearest",
    ).squeeze().numpy()

    ax_freq = fig.add_subplot(gs_tr[0])
    im_freq = ax_freq.imshow(freq_up, cmap="viridis", interpolation="none", vmin=0)
    ax_freq.set_title("Firing frequency\n(frac. of images)", fontsize=8)
    ax_freq.axis("off")
    fig.colorbar(im_freq, ax=ax_freq, fraction=0.046, pad=0.04)

    # (b) Activation histogram
    ax_hist = fig.add_subplot(gs_tr[1])
    firing_vals = col[col > 0]
    if len(firing_vals) > 0:
        ax_hist.hist(firing_vals, bins=30, color="steelblue", edgecolor="none")
    ax_hist.set_yscale("log")
    if n_show > 0:
        thresh_val = float(feature_max[top_imgs[n_show - 1], feature_idx].item())
        ax_hist.axvline(thresh_val, color="red", linestyle="--",
                        linewidth=1.0, label=f"top-{n_show}")
        ax_hist.legend(fontsize=6)
    ax_hist.set_xlabel("Activation", fontsize=8)
    ax_hist.set_ylabel("Count (log)", fontsize=8)
    ax_hist.set_title(f"Activation histogram\nfires {pct_firing:.1f}%", fontsize=8)
    ax_hist.tick_params(labelsize=7)

    # ── BOTTOM LEFT: decoder cosine similarity bar chart (top-15) ─────────
    ax_sim = fig.add_subplot(gs[1, 0])

    with torch.no_grad():
        W_dec = sae.W_dec  # [d_sae, d_model]
        query = W_dec[feature_idx]  # [d_model]
        normed_dec = F.normalize(W_dec, dim=-1)
        query_norm = F.normalize(query.unsqueeze(0), dim=-1)
        sims = (query_norm @ normed_dec.T).squeeze(0)  # [d_sae]
        sims[feature_idx] = -2.0  # exclude self
        n_top = min(15, sims.shape[0])
        top15 = torch.topk(sims, n_top)

    top15_vals = top15.values.cpu().numpy()
    top15_idxs = top15.indices.cpu().numpy()
    labels_bar = [f"F{int(i)}" for i in top15_idxs]

    y_pos = list(range(n_top))[::-1]
    ax_sim.barh(y_pos, top15_vals, color="coral", edgecolor="none")
    ax_sim.set_yticks(y_pos)
    ax_sim.set_yticklabels(labels_bar, fontsize=7)
    ax_sim.set_xlabel("Cosine similarity", fontsize=8)
    ax_sim.set_title(f"Top-{n_top} similar decoder atoms", fontsize=9, fontweight="bold")
    ax_sim.tick_params(labelsize=7)

    # ── BOTTOM RIGHT: universality stacked bar ────────────────────────────
    ax_univ = fig.add_subplot(gs[1, 1])

    if dataset_labels is not None:
        profile = compute_universality_profile(
            feature_idx, stats, dataset_labels, top_k=50
        )
        all_datasets = [k for k in profile if k != "universality_score"]
        u_score = profile.get("universality_score", 0.0)

        cmap_tab = plt.get_cmap("tab10")
        x_start = 0.0
        for ds_i, ds_name in enumerate(all_datasets):
            frac = profile[ds_name]["fraction"]
            ax_univ.barh(
                0, frac, left=x_start,
                height=0.5,
                color=cmap_tab(ds_i % 10),
                label=ds_name,
            )
            x_start += frac

        ax_univ.set_xlim(0, 1)
        ax_univ.set_yticks([])
        ax_univ.set_xlabel("Fraction of top-50 images", fontsize=8)
        ax_univ.set_title(
            f"Dataset universality\n(score={u_score:.2f})",
            fontsize=9, fontweight="bold",
        )
        ax_univ.legend(fontsize=7, loc="upper right")
        ax_univ.tick_params(labelsize=7)
    else:
        ax_univ.text(
            0.5, 0.5,
            "No dataset labels provided",
            ha="center", va="center",
            fontsize=9, color="grey",
            transform=ax_univ.transAxes,
        )
        ax_univ.axis("off")
        ax_univ.set_title("Dataset universality", fontsize=9, fontweight="bold")

    # ── Super title ───────────────────────────────────────────────────────
    fig.suptitle(
        f"Feature {feature_idx}  |  max_act={gmax_val:.2f}  |  "
        f"fires on {pct_firing:.1f}% of images",
        fontsize=13, fontweight="bold", y=1.005,
    )

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ── Smoke tests ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile
    import numpy as np

    print("=" * 60)
    print("  semantic_dict.py — smoke tests")
    print("=" * 60)

    # ── 1. get_patch_crop ─────────────────────────────────────────────
    # Create a 224×224 dummy image
    dummy_arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    dummy_img = Image.fromarray(dummy_arr)

    crop = get_patch_crop(dummy_img, patch_idx=0, spatial_h=14, spatial_w=14, context_patches=1)
    assert crop.size[0] > 0 and crop.size[1] > 0
    print(f"[OK] get_patch_crop corner (0): crop size = {crop.size}")

    crop_mid = get_patch_crop(dummy_img, patch_idx=7 * 14 + 7, spatial_h=14, spatial_w=14, context_patches=2)
    assert crop_mid.size[0] > 0
    print(f"[OK] get_patch_crop centre (105): crop size = {crop_mid.size}")

    # ── 2. render_semantic_dictionary ─────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 4 tiny images
        img_dir = Path(tmpdir) / "images"
        img_dir.mkdir()
        paths = []
        for i in range(4):
            p = img_dir / f"img_{i:03d}.png"
            arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            Image.fromarray(arr).save(p)
            paths.append(p)

        # Fake stats
        n_images, d_sae, n_patches = 4, 8, 196
        stats = {
            "feature_max": torch.rand(n_images, d_sae),
            "feature_argmax": torch.randint(0, n_patches, (n_images, d_sae)),
            "feature_mean": torch.rand(n_images, d_sae),
            "global_max": torch.rand(d_sae),
            "n_images": n_images,
        }

        cfg = VisConfig(spatial_h=14, spatial_w=14, top_k_images=4, device="cpu")

        save_p = str(Path(tmpdir) / "test_feature.png")
        fig = render_semantic_dictionary(
            feature_idx=0,
            stats=stats,
            image_paths=paths,
            config=cfg,
            feature_label="test-label",
            save_path=save_p,
        )
        plt.close(fig)
        assert Path(save_p).exists()
        print(f"[OK] render_semantic_dictionary: saved to {save_p}")

        # ── 3. render_feature_dashboard ───────────────────────────────
        dash_dir = str(Path(tmpdir) / "dashboard")
        render_feature_dashboard(
            feature_indices=[0, 1, 2],
            stats=stats,
            image_paths=paths,
            config=cfg,
            feature_labels={0: "edges", 2: "texture"},
            output_dir=dash_dir,
        )
        pngs = list(Path(dash_dir).glob("*.png"))
        assert len(pngs) == 3
        print(f"[OK] render_feature_dashboard: {len(pngs)} PNGs in {dash_dir}")

        # ── 4. render_semantic_dictionary_v2 ──────────────────────────
        sae = BatchTopKSAE(d_model=16, d_sae=d_sae, k=2)
        sae.eval()
        save_v2 = str(Path(tmpdir) / "test_feature_v2.png")
        fig_v2 = render_semantic_dictionary_v2(
            feature_idx=0,
            stats=stats,
            image_paths=paths,
            sae=sae,
            config=cfg,
            feature_label="v2-test",
            save_path=save_v2,
        )
        plt.close(fig_v2)
        assert Path(save_v2).exists()
        print(f"[OK] render_semantic_dictionary_v2: saved to {save_v2}")

        # ── 5. render_feature_card ─────────────────────────────────────
        ds_labels = ["brain", "lung", "colon", "brain"]
        save_card = str(Path(tmpdir) / "test_card.png")
        fig_card = render_feature_card(
            feature_idx=0,
            stats=stats,
            image_paths=paths,
            sae=sae,
            config=cfg,
            dataset_labels=ds_labels,
            save_path=save_card,
        )
        plt.close(fig_card)
        assert Path(save_card).exists()
        print(f"[OK] render_feature_card (with labels): saved to {save_card}")

        # Without dataset labels
        fig_card2 = render_feature_card(
            feature_idx=0,
            stats=stats,
            image_paths=paths,
            sae=sae,
            config=cfg,
            dataset_labels=None,
        )
        plt.close(fig_card2)
        print("[OK] render_feature_card (no labels): created")

    print("\nAll semantic_dict smoke tests passed ✓")
