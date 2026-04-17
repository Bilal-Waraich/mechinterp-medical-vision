"""Step 8 — Qualitative inspection of confirmed universal latents.

For each confirmed universal latent (from universality_eval.py), renders:
  1. Semantic dictionary grid — top-K activating patch crops per dataset,
     side-by-side across all datasets in a single figure.
  2. Patch ablation maps — deletion-mode ablation on the top-1 activating
     image per dataset, showing which patches causally drive the latent.

Target: ≤15 confirmed latents.  All outputs written to --output_dir.

Input
-----
  --confirmed_json   universality_summary.json  (or confirmed_latent_frequencies.json)
                     produced by universality_eval.py
  --activations      PATH:LABEL  (repeat per dataset, same format as universality_eval.py)
  --images_dir       PATH:LABEL  (repeat per dataset — directory of original images)
  --checkpoint_dir   SAE checkpoint (sae_weights.safetensors + cfg.json)

Usage
-----
python src/scripts/inspect_universal_latents.py \\
    --confirmed_json  /deep_learning/output/.../universality/universality_summary.json \\
    --checkpoint_dir  /deep_learning/output/.../checkpoints/step_0200000 \\
    --activations     /deep_learning/output/.../activations_chexpert.safetensors:chexpert \\
    --activations     /deep_learning/output/.../activations_rsna.safetensors:rsna \\
    --activations     /deep_learning/output/.../activations_siim.safetensors:siim_covid \\
    --activations     /deep_learning/output/.../activations_tb.safetensors:tb \\
    --images_dir      /deep_learning/input/data/mmm/chexpert/CheXpert-v1.0/valid:chexpert \\
    --images_dir      /deep_learning/input/data/mmm/rsna-pneumonia-detection-challenge/stage_2_train_images:rsna \\
    --images_dir      /deep_learning/input/data/mmm/siim-covid19-detection/train:siim_covid \\
    --images_dir      /deep_learning/input/data/mmm/tuberculosis-chest-xrays-shenzhen/images/images:tb \\
    --output_dir      /deep_learning/output/.../universality/inspection \\
    --top_k           8 \\
    --max_latents     15 \\
    --device          cpu

See also
--------
docs/methodology.md §1.7 Step 8 — qualitative inspection
src/scripts/universality_eval.py — produces confirmed_indices input
src/visualisation/ablation.py    — patch deletion/insertion primitives
src/visualisation/semantic_dict.py — patch cropping utilities
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from safetensors.torch import load_file
from visualisation.loaders import load_sae
from visualisation.ablation import run_patch_ablation, render_ablation_figure
from visualisation.semantic_dict import get_patch_crop


# ── Image loading ─────────────────────────────────────────────────────────────

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".dcm"}

def load_image_paths(images_dir: str) -> list[Path]:
    """Return sorted non-DICOM image paths; falls back to DICOM if nothing else."""
    root = Path(images_dir)
    paths = sorted(p for p in root.iterdir()
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not paths:
        # Try one level deeper (some datasets nest by patient)
        paths = sorted(p for sub in root.iterdir() if sub.is_dir()
                       for p in sub.iterdir()
                       if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    return paths


def open_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


# ── Feature-max extraction ────────────────────────────────────────────────────

@torch.no_grad()
def get_feature_max(
    acts: torch.Tensor,
    sae,
    n_patches: int,
    batch_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return feature_max [n_images, d_sae] and feature_argmax [n_images, d_sae]."""
    N = acts.shape[0]
    n_images = N // n_patches
    d_sae = sae.d_sae
    feat_max    = torch.zeros(n_images, d_sae)
    feat_argmax = torch.zeros(n_images, d_sae, dtype=torch.long)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        codes = sae.encode(acts[start:end].to(next(sae.parameters()).device)).cpu()
        # Align to image boundaries
        img_start = start // n_patches
        img_end   = min((end + n_patches - 1) // n_patches, n_images)
        # Trim codes to exact image-aligned chunk
        n_tokens_chunk = (img_end - img_start) * n_patches
        codes = codes[:n_tokens_chunk].reshape(img_end - img_start, n_patches, d_sae)
        chunk_max, chunk_argmax = codes.max(dim=1)
        feat_max[img_start:img_end]    = chunk_max
        feat_argmax[img_start:img_end] = chunk_argmax

    return feat_max, feat_argmax


# ── Semantic dictionary grid ──────────────────────────────────────────────────

def render_semantic_grid(
    lat_idx: int,
    datasets: list[tuple[str, torch.Tensor]],
    image_dirs: dict[str, list[Path]],
    sae,
    n_patches: int,
    spatial_h: int,
    spatial_w: int,
    top_k: int,
    save_path: Path,
    batch_size: int = 4096,
) -> None:
    """Render top-K activating patch crops per dataset in a side-by-side grid."""
    ds_names = [name for name, _ in datasets]
    n_ds = len(ds_names)
    fig = plt.figure(figsize=(3 * top_k, 3.5 * n_ds))
    gs  = gridspec.GridSpec(n_ds, top_k, figure=fig, wspace=0.05, hspace=0.3)

    for row, (ds_name, acts) in enumerate(datasets):
        feat_max, feat_argmax = get_feature_max(acts, sae, n_patches, batch_size)
        col_max = feat_max[:, lat_idx]

        k_actual = min(top_k, col_max.shape[0])
        top_imgs = torch.topk(col_max, k_actual).indices.tolist()

        img_paths = image_dirs.get(ds_name, [])

        for col, img_idx in enumerate(top_imgs):
            ax = fig.add_subplot(gs[row, col])
            ax.axis("off")

            patch_idx = int(feat_argmax[img_idx, lat_idx].item())
            act_val   = float(col_max[img_idx].item())

            if img_idx < len(img_paths):
                try:
                    img = open_image(img_paths[img_idx])
                    crop = get_patch_crop(img, patch_idx, spatial_h, spatial_w,
                                         context_patches=1)
                    ax.imshow(np.array(crop))
                except Exception as e:
                    ax.text(0.5, 0.5, f"err\n{e}", ha="center", va="center",
                            fontsize=6, transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, f"img {img_idx}\n(no path)",
                        ha="center", va="center", fontsize=7,
                        transform=ax.transAxes, color="grey")

            title = f"act={act_val:.2f}" if col == 0 else f"{act_val:.2f}"
            ax.set_title(title, fontsize=7, pad=2)

        # Row label
        fig.text(0.01, (n_ds - row - 0.5) / n_ds, ds_name,
                 va="center", ha="left", fontsize=10, fontweight="bold",
                 rotation=90)

    plt.suptitle(f"Latent F{lat_idx} — top-{top_k} activating patches per dataset",
                 fontsize=13, fontweight="bold", y=1.01)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Patch ablation panel ──────────────────────────────────────────────────────

def render_ablation_panel(
    lat_idx: int,
    datasets: list[tuple[str, torch.Tensor]],
    image_dirs: dict[str, list[Path]],
    sae,
    n_patches: int,
    spatial_h: int,
    spatial_w: int,
    save_path: Path,
    batch_size: int = 4096,
) -> None:
    """Run deletion-mode patch ablation on the top-1 activating image per dataset.

    Renders one row per dataset: [original + important patches | delta heatmap].
    """
    ds_names = [name for name, _ in datasets]
    n_ds = len(ds_names)
    fig, axes = plt.subplots(n_ds, 2, figsize=(10, 4 * n_ds))
    if n_ds == 1:
        axes = [axes]

    for row, (ds_name, acts) in enumerate(datasets):
        feat_max, _ = get_feature_max(acts, sae, n_patches, batch_size)
        col_max = feat_max[:, lat_idx]
        top_img_idx = int(col_max.argmax().item())

        result = run_patch_ablation(
            feature_idx=lat_idx,
            image_idx=top_img_idx,
            sae=sae,
            flat_activations=acts,
            n_patches=n_patches,
            spatial_h=spatial_h,
            spatial_w=spatial_w,
            mode="deletion",
        )

        img_paths = image_dirs.get(ds_name, [])
        ax_img, ax_heat = axes[row][0], axes[row][1]

        delta = result["delta_map"]

        # Load original image if available
        bg = None
        if top_img_idx < len(img_paths):
            try:
                bg = np.array(open_image(img_paths[top_img_idx]))
            except Exception:
                pass

        # Panel A: original with top-5 patch boxes
        if bg is not None:
            ax_img.imshow(bg)
        else:
            ax_img.set_facecolor("#222")
        cell_h = (bg.shape[0] if bg is not None else 224) / spatial_h
        cell_w = (bg.shape[1] if bg is not None else 224) / spatial_w
        colours = ["red", "orange", "yellow", "lime", "cyan"]
        import matplotlib.patches as mpatches
        for rank, p in enumerate(result["most_important_patches"][:5]):
            r, c = p // spatial_w, p % spatial_w
            rect = mpatches.Rectangle(
                (c * cell_w, r * cell_h), cell_w, cell_h,
                linewidth=2, edgecolor=colours[rank], facecolor="none"
            )
            ax_img.add_patch(rect)
        ax_img.set_title(f"{ds_name} — img {top_img_idx}\nbaseline act={result['baseline_act']:.2f}",
                         fontsize=9)
        ax_img.axis("off")

        # Panel B: delta heatmap
        if bg is not None:
            ax_heat.imshow(bg)
        try:
            import cv2
            delta_up = cv2.resize(delta.astype(np.float32),
                                  (bg.shape[1] if bg is not None else 224,
                                   bg.shape[0] if bg is not None else 224),
                                  interpolation=cv2.INTER_NEAREST)
        except ImportError:
            from PIL import Image as PILImage
            d_norm = ((delta - delta.min()) / max(delta.ptp(), 1e-8) * 255).astype(np.uint8)
            d_pil  = PILImage.fromarray(d_norm).resize(
                (bg.shape[1] if bg is not None else 224,
                 bg.shape[0] if bg is not None else 224), PILImage.NEAREST)
            delta_up = np.array(d_pil) / 255.0 * (delta.max() - delta.min()) + delta.min()

        vmax = max(abs(delta_up.min()), abs(delta_up.max()), 1e-6)
        ax_heat.imshow(delta_up, cmap="RdBu_r", alpha=0.65, vmin=-vmax, vmax=vmax)
        ax_heat.set_title("Deletion delta (red=important)", fontsize=9)
        ax_heat.axis("off")

    plt.suptitle(f"Patch Ablation — Latent F{lat_idx}", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Activation frequency summary bar ─────────────────────────────────────────

def render_frequency_bar(
    lat_idx: int,
    freq_entry: dict[str, float],
    save_path: Path,
) -> None:
    """Small horizontal bar chart of per-dataset activation frequency."""
    ds_names = list(freq_entry.keys())
    freqs    = [freq_entry[d] for d in ds_names]
    colours  = plt.get_cmap("tab10")(np.arange(len(ds_names)) / 10)

    fig, ax = plt.subplots(figsize=(5, 2))
    bars = ax.barh(ds_names, freqs, color=colours)
    ax.axvline(0.01, color="black", linestyle="--", linewidth=0.8,
               label="1% threshold")
    ax.set_xlabel("Activation frequency (fraction of tokens)", fontsize=8)
    ax.set_title(f"F{lat_idx} — per-dataset token frequency", fontsize=9,
                 fontweight="bold")
    ax.legend(fontsize=7)
    for bar, v in zip(bars, freqs):
        ax.text(v + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{v:.3%}", va="center", fontsize=7)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Qualitative inspection of confirmed universal latents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--confirmed_json", required=True,
                    help="universality_summary.json or confirmed_latent_frequencies.json "
                         "from universality_eval.py.")
    ap.add_argument("--checkpoint_dir", required=True,
                    help="SAE checkpoint directory.")
    ap.add_argument("--activations", action="append", required=True,
                    metavar="PATH:LABEL",
                    help="Per-dataset activation safetensors and label. Repeat per dataset.")
    ap.add_argument("--images_dir", action="append", default=[],
                    metavar="PATH:LABEL",
                    help="Directory of original images for a dataset. Repeat per dataset. "
                         "Optional — patch crops are skipped if not provided.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--top_k",      type=int, default=8,
                    help="Top-K activating images to show per dataset in the semantic grid.")
    ap.add_argument("--max_latents", type=int, default=15,
                    help="Maximum number of confirmed latents to inspect.")
    ap.add_argument("--spatial_h",  type=int, default=7,
                    help="Patch grid height (7 for pm_index=2, 14 for pm_index=1).")
    ap.add_argument("--spatial_w",  type=int, default=7)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--device",
                    default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip_ablation", action="store_true",
                    help="Skip patch ablation (faster; use when no image paths provided).")
    return ap.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_patches = args.spatial_h * args.spatial_w

    print("=" * 65)
    print("  Universal Latent Inspection")
    print("=" * 65)

    # ── Load confirmed latent list ─────────────────────────────────────────
    with open(args.confirmed_json) as f:
        data = json.load(f)

    # Accept both universality_summary.json and confirmed_latent_frequencies.json
    if "confirmed_indices" in data:
        confirmed = data["confirmed_indices"]
        freq_map: dict[str, dict[str, float]] = {}
    else:
        # confirmed_latent_frequencies.json: keys are latent indices (strings)
        confirmed = [int(k) for k in data.keys()]
        freq_map  = {k: v for k, v in data.items()}

    confirmed = confirmed[: args.max_latents]
    print(f"  Inspecting {len(confirmed)} confirmed latents: {confirmed}")

    # Load per-latent frequencies if available (for bar charts)
    freq_json_path = Path(args.confirmed_json).parent / "confirmed_latent_frequencies.json"
    if freq_json_path.exists() and not freq_map:
        with open(freq_json_path) as f:
            freq_map = json.load(f)

    # ── Load SAE ──────────────────────────────────────────────────────────
    print(f"\nLoading SAE …")
    sae = load_sae(args.checkpoint_dir, device=str(device))
    sae.eval()
    print(f"  d_model={sae.d_model}  d_sae={sae.d_sae}  k={sae.k}")

    # ── Load activations ─────────────────────────────────────────────────
    datasets: list[tuple[str, torch.Tensor]] = []
    for entry in args.activations:
        path, label = entry.rsplit(":", 1)
        raw  = load_file(str(path))
        acts = raw["activations"].float()
        datasets.append((label, acts))
        print(f"  {label}: {acts.shape[0]:,} tokens")

    # ── Load image directories ────────────────────────────────────────────
    image_dirs: dict[str, list[Path]] = {}
    for entry in (args.images_dir or []):
        path, label = entry.rsplit(":", 1)
        paths = load_image_paths(path)
        image_dirs[label] = paths
        print(f"  {label}: {len(paths)} images found")

    # ── Render per-latent outputs ─────────────────────────────────────────
    print(f"\nRendering {len(confirmed)} latents …")
    for lat_idx in tqdm(confirmed, desc="Latents"):
        lat_dir = out_dir / f"latent_{lat_idx:05d}"
        lat_dir.mkdir(parents=True, exist_ok=True)

        # 1. Frequency bar chart
        if str(lat_idx) in freq_map:
            render_frequency_bar(
                lat_idx,
                freq_map[str(lat_idx)],
                lat_dir / "frequency_bar.png",
            )

        # 2. Semantic dictionary grid (top-K patches per dataset)
        render_semantic_grid(
            lat_idx=lat_idx,
            datasets=datasets,
            image_dirs=image_dirs,
            sae=sae,
            n_patches=n_patches,
            spatial_h=args.spatial_h,
            spatial_w=args.spatial_w,
            top_k=args.top_k,
            save_path=lat_dir / "semantic_grid.png",
            batch_size=args.batch_size,
        )

        # 3. Patch ablation panel (one row per dataset)
        if not args.skip_ablation:
            render_ablation_panel(
                lat_idx=lat_idx,
                datasets=datasets,
                image_dirs=image_dirs,
                sae=sae,
                n_patches=n_patches,
                spatial_h=args.spatial_h,
                spatial_w=args.spatial_w,
                save_path=lat_dir / "patch_ablation.png",
                batch_size=args.batch_size,
            )

    # ── Index page ────────────────────────────────────────────────────────
    index_lines = [
        "# Universal Latent Inspection Index\n",
        f"Confirmed latents: {confirmed}\n\n",
    ]
    for lat_idx in confirmed:
        index_lines.append(
            f"## Latent F{lat_idx}\n"
            f"- [Frequency bar](latent_{lat_idx:05d}/frequency_bar.png)\n"
            f"- [Semantic grid](latent_{lat_idx:05d}/semantic_grid.png)\n"
            f"- [Patch ablation](latent_{lat_idx:05d}/patch_ablation.png)\n\n"
        )
    (out_dir / "index.md").write_text("".join(index_lines))

    print(f"\n{'=' * 65}")
    print(f"  Done. Outputs written to: {out_dir}")
    print(f"  Index: {out_dir / 'index.md'}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
