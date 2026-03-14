"""Spatial activation-map renderer for Phase 3 visualisation.

Slices the flat harvested activations for a single image, encodes them
through the SAE, and produces per-feature heatmaps that can be overlaid
on the original image to show *where* in the image a feature fires.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from sae_models import BatchTopKSAE  # noqa: E402
from visualisation.vis_config import VisConfig  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────
# 1. Per-image SAE encoding
# ──────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_image_activations(
    image_idx: int,
    activations: torch.Tensor,
    sae: BatchTopKSAE,
    config: VisConfig,
) -> torch.Tensor:
    """Encode the patches of a single image through the SAE.

    Parameters
    ----------
    image_idx : int
        Which image (0-indexed, matching harvest order).
    activations : Tensor [N_eff, d_model]
        Full flat activation tensor.
    sae : BatchTopKSAE
        Trained SAE in eval mode.
    config : VisConfig
        Supplies ``n_patches``.

    Returns
    -------
    Tensor [n_patches, d_sae]
        Per-sample top-k sparse codes for every patch.
    """
    n_patches = config.n_patches
    start = image_idx * n_patches
    end = start + n_patches
    patch_acts = activations[start:end].to(sae.W_enc.device)  # [n_patches, d_model]
    return sae.encode(patch_acts).cpu()  # [n_patches, d_sae]


# ──────────────────────────────────────────────────────────────────────────
# 2. Spatial activation map
# ──────────────────────────────────────────────────────────────────────────

def get_spatial_activation_map(
    f: torch.Tensor,
    feature_idx: int,
    spatial_h: int,
    spatial_w: int,
) -> torch.Tensor:
    """Reshape a single feature's activations into a 2-D spatial map.

    Parameters
    ----------
    f : Tensor [n_patches, d_sae]
    feature_idx : int
    spatial_h, spatial_w : int

    Returns
    -------
    Tensor [spatial_h, spatial_w]  (raw values, not normalised)
    """
    spatial = f[:, feature_idx]  # [n_patches]
    return spatial.reshape(spatial_h, spatial_w)


# ──────────────────────────────────────────────────────────────────────────
# 3. Upsampling + normalisation
# ──────────────────────────────────────────────────────────────────────────

def upsample_and_normalise(
    heatmap: torch.Tensor,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    """Bilinear-upsample and min-max normalise a heatmap to [0, 1].

    Parameters
    ----------
    heatmap : Tensor [H', W']
    target_h, target_w : int
        Output spatial size (typically the original image dimensions).

    Returns
    -------
    Tensor [target_h, target_w]  on CPU, values in [0, 1].
    """
    up = F.interpolate(
        heatmap[None, None].float(),
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze()  # [target_h, target_w]

    lo, hi = up.min(), up.max()
    normed = (up - lo) / (hi - lo + 1e-8)
    return normed.cpu()


# ──────────────────────────────────────────────────────────────────────────
# 4. Overlay figure
# ──────────────────────────────────────────────────────────────────────────

def overlay_heatmap(
    img: Image.Image,
    heatmap_norm: torch.Tensor,
    alpha: float = 0.5,
    cmap: str = "hot",
) -> plt.Figure:
    """Three-panel figure: original | heatmap | overlay.

    Parameters
    ----------
    img : PIL.Image
        Original image (RGB or L).
    heatmap_norm : Tensor [H, W]
        Normalised heatmap in [0, 1] on CPU.
    alpha : float
        Blending weight for the overlay.
    cmap : str
        Matplotlib colour-map name.

    Returns
    -------
    matplotlib.figure.Figure
    """
    img_arr = np.array(img)
    hm = heatmap_norm.numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # [0] Original
    if img_arr.ndim == 2:
        axes[0].imshow(img_arr, cmap="gray")
    else:
        axes[0].imshow(img_arr)
    axes[0].set_title("Original", fontsize=9)
    axes[0].axis("off")

    # [1] Heatmap
    im1 = axes[1].imshow(hm, cmap=cmap, interpolation="none")
    axes[1].set_title("Heatmap", fontsize=9)
    axes[1].axis("off")

    # [2] Overlay
    if img_arr.ndim == 2:
        axes[2].imshow(img_arr, cmap="gray")
    else:
        axes[2].imshow(img_arr)
    im2 = axes[2].imshow(hm, cmap=cmap, alpha=alpha, interpolation="none")
    axes[2].set_title("Overlay", fontsize=9)
    axes[2].axis("off")

    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────
# 5. Multi-feature spatial panel
# ──────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def render_multi_feature_spatial(
    image_idx: int,
    feature_indices: list[int],
    activations: torch.Tensor,
    sae: BatchTopKSAE,
    image_paths: list[Path],
    config: VisConfig,
    feature_labels: dict[int, str] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Render spatial-overlay panels for several features on one image.

    Layout: one row per feature, three columns
    ``[original | heatmap | overlay]``.

    Parameters
    ----------
    image_idx : int
        Image index (0-based).
    feature_indices : list[int]
        SAE features to visualise.
    activations : Tensor [N_eff, d_model]
    sae : BatchTopKSAE  (eval mode)
    image_paths : list[Path]
    config : VisConfig
    feature_labels : dict, optional
        ``{feature_idx: label}``.
    save_path : str, optional
        If given, figure is saved at 150 dpi.

    Returns
    -------
    matplotlib.figure.Figure
    """
    labels = feature_labels or {}
    n_feats = len(feature_indices)

    # Encode patches once
    f = get_image_activations(image_idx, activations, sae, config)

    # Load image
    img = Image.open(image_paths[image_idx]).convert("RGB")
    img_w, img_h = img.size
    img_arr = np.array(img)

    fig, axes = plt.subplots(n_feats, 3, figsize=(12, 4 * n_feats))
    if n_feats == 1:
        axes = axes[np.newaxis, :]  # ensure 2-D

    for row, feat_idx in enumerate(feature_indices):
        heatmap = get_spatial_activation_map(
            f, feat_idx, config.spatial_h, config.spatial_w
        )
        hm_norm = upsample_and_normalise(heatmap, img_h, img_w)
        hm_np = hm_norm.numpy()

        # Col 0: original
        axes[row, 0].imshow(img_arr)
        axes[row, 0].axis("off")

        # Col 1: heatmap
        axes[row, 1].imshow(hm_np, cmap="hot", interpolation="none")
        axes[row, 1].axis("off")

        # Col 2: overlay
        axes[row, 2].imshow(img_arr)
        im = axes[row, 2].imshow(hm_np, cmap="hot", alpha=0.5, interpolation="none")
        axes[row, 2].axis("off")
        fig.colorbar(im, ax=axes[row, 2], fraction=0.046, pad=0.04)

        # Row label
        lbl = f"Feature {feat_idx}"
        if feat_idx in labels:
            lbl += f"  |  {labels[feat_idx]}"
        axes[row, 0].set_title(lbl, fontsize=9, fontweight="bold", loc="left")

    # Column headers on top row
    for col, title in enumerate(["Original", "Heatmap", "Overlay"]):
        axes[0, col].set_title(
            axes[0, col].get_title() + f"\n{title}" if col == 0 else title,
            fontsize=9,
        )

    fig.suptitle(
        f"Image {image_idx}  —  spatial activation maps",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ── Smoke tests ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("  spatial_maps.py — smoke tests")
    print("=" * 60)

    d_model, d_sae, k = 16, 32, 4
    spatial_h, spatial_w = 2, 2
    n_patches = spatial_h * spatial_w
    n_images = 3
    N_eff = n_images * n_patches

    sae = BatchTopKSAE(d_model=d_model, d_sae=d_sae, k=k)
    sae.eval()
    cfg = VisConfig(spatial_h=spatial_h, spatial_w=spatial_w, device="cpu")
    acts = torch.randn(N_eff, d_model)

    # 1. get_image_activations
    f = get_image_activations(0, acts, sae, cfg)
    assert f.shape == (n_patches, d_sae), f"got {f.shape}"
    print(f"[OK] get_image_activations: shape={f.shape}")

    # 2. get_spatial_activation_map
    hm = get_spatial_activation_map(f, feature_idx=0, spatial_h=spatial_h, spatial_w=spatial_w)
    assert hm.shape == (spatial_h, spatial_w)
    print(f"[OK] get_spatial_activation_map: shape={hm.shape}")

    # 3. upsample_and_normalise
    hm_up = upsample_and_normalise(hm, 64, 64)
    assert hm_up.shape == (64, 64)
    assert 0.0 <= hm_up.min() and hm_up.max() <= 1.0 + 1e-6
    print(f"[OK] upsample_and_normalise: shape={hm_up.shape}, range=[{hm_up.min():.3f}, {hm_up.max():.3f}]")

    # 4. overlay_heatmap
    dummy_img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    fig = overlay_heatmap(dummy_img, hm_up, alpha=0.5)
    plt.close(fig)
    print("[OK] overlay_heatmap: figure created")

    # Test with grayscale image
    gray_img = Image.fromarray(np.random.randint(0, 255, (64, 64), dtype=np.uint8), mode="L")
    fig = overlay_heatmap(gray_img, hm_up)
    plt.close(fig)
    print("[OK] overlay_heatmap (grayscale): figure created")

    # 5. render_multi_feature_spatial
    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir = Path(tmpdir) / "images"
        img_dir.mkdir()
        paths = []
        for i in range(n_images):
            p = img_dir / f"img_{i:03d}.png"
            Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(p)
            paths.append(p)

        save_p = str(Path(tmpdir) / "multi_spatial.png")
        fig = render_multi_feature_spatial(
            image_idx=0,
            feature_indices=[0, 1],
            activations=acts,
            sae=sae,
            image_paths=paths,
            config=cfg,
            feature_labels={0: "edges"},
            save_path=save_p,
        )
        plt.close(fig)
        assert Path(save_p).exists()
        print(f"[OK] render_multi_feature_spatial: saved to {save_p}")

    print("\nAll spatial_maps smoke tests passed ✓")
