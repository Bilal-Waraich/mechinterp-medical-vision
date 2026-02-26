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
import torch
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure ``src/`` is importable
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

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

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows + 0.8))
    axes = axes.flat if hasattr(axes, "flat") else [axes]

    for i, ax in enumerate(axes):
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

    print("\nAll semantic_dict smoke tests passed ✓")
