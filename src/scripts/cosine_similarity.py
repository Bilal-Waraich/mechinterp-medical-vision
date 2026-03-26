"""Cross-SAE cosine similarity analysis for universality detection.

Loads two trained SAEs (e.g. breast and colon, layer 2) and computes
pairwise cosine similarity between their decoder weight vectors (W_dec rows).
Each row of W_dec is a learned feature direction in activation space —
high cosine similarity means two features from different SAEs point in
the same direction, suggesting a universal concept.

Outputs
-------
1. CSV of all colon SAE features (index, norm, top-5 activation stats)
2. CSV of all breast SAE features (index, norm, top-5 activation stats)
3. CSV similarity matrix (all pairs above threshold)
4. Side-by-side visualisation of matched feature pairs (semantic dicts)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from sae_models import BatchTopKSAE  # noqa: E402
from visualisation.loaders import load_sae, load_activations  # noqa: E402
from visualisation.vis_config import VisConfig  # noqa: E402
from visualisation.collector import compute_feature_stats, get_top_images_per_feature  # noqa: E402
from visualisation.semantic_dict import get_patch_crop  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────
# 1. Cosine similarity matrix
# ──────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_decoder_cosine_similarity(
    sae_a: BatchTopKSAE,
    sae_b: BatchTopKSAE,
) -> torch.Tensor:
    """Compute cosine similarity between all pairs of decoder features.

    Parameters
    ----------
    sae_a, sae_b : BatchTopKSAE
        Two trained SAEs. Their W_dec must have the same d_model.

    Returns
    -------
    Tensor [d_sae_a, d_sae_b]
        Cosine similarity matrix.
    """
    # W_dec shape: [d_sae, d_model] — each row is a feature direction
    assert sae_a.d_model == sae_b.d_model, (
        f"d_model mismatch: {sae_a.d_model} vs {sae_b.d_model}"
    )

    W_a = F.normalize(sae_a.W_dec.data.float(), dim=1)  # [d_sae_a, d_model]
    W_b = F.normalize(sae_b.W_dec.data.float(), dim=1)  # [d_sae_b, d_model]

    # Cosine similarity = dot product of unit vectors
    return W_a @ W_b.T  # [d_sae_a, d_sae_b]


# ──────────────────────────────────────────────────────────────────────────
# 2. Feature CSV export
# ──────────────────────────────────────────────────────────────────────────

def export_feature_csv(
    sae: BatchTopKSAE,
    output_path: Path,
    label: str,
) -> None:
    """Write a CSV listing all features in an SAE.

    Columns: feature_idx, dataset, decoder_norm, encoder_norm
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    W_dec_norms = sae.W_dec.data.float().norm(dim=1).cpu().numpy()
    W_enc_norms = sae.W_enc.data.float().norm(dim=0).cpu().numpy()  # [d_sae]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_idx", "dataset", "decoder_norm", "encoder_norm"])
        for i in range(sae.d_sae):
            writer.writerow([i, label, f"{W_dec_norms[i]:.6f}", f"{W_enc_norms[i]:.6f}"])

    print(f"  Wrote {sae.d_sae} features to {output_path}")


# ──────────────────────────────────────────────────────────────────────────
# 3. Similarity matrix CSV export
# ──────────────────────────────────────────────────────────────────────────

def export_similarity_csv(
    sim_matrix: torch.Tensor,
    output_path: Path,
    threshold: float,
    label_a: str = "colon",
    label_b: str = "breast",
) -> list[tuple[int, int, float]]:
    """Write CSV of all feature pairs above threshold.

    Returns list of (idx_a, idx_b, similarity) sorted descending.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Find pairs above threshold
    mask = sim_matrix >= threshold
    pairs_a, pairs_b = torch.where(mask)
    sims = sim_matrix[mask]

    # Sort by similarity descending
    order = sims.argsort(descending=True)
    pairs_a = pairs_a[order].cpu().numpy()
    pairs_b = pairs_b[order].cpu().numpy()
    sims = sims[order].cpu().numpy()

    results = []
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            f"{label_a}_feature_idx",
            f"{label_b}_feature_idx",
            "cosine_similarity",
        ])
        for a, b, s in zip(pairs_a, pairs_b, sims):
            writer.writerow([int(a), int(b), f"{s:.6f}"])
            results.append((int(a), int(b), float(s)))

    print(f"  Wrote {len(results)} pairs (threshold >= {threshold}) to {output_path}")
    return results


# ──────────────────────────────────────────────────────────────────────────
# 4. Side-by-side semantic dict visualisation
# ──────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def render_matched_pair(
    feature_idx_a: int,
    feature_idx_b: int,
    cosine_sim: float,
    sae_a: BatchTopKSAE,
    sae_b: BatchTopKSAE,
    activations_a: torch.Tensor,
    activations_b: torch.Tensor,
    image_paths_a: list[Path],
    image_paths_b: list[Path],
    config_a: VisConfig,
    config_b: VisConfig,
    label_a: str = "Colon",
    label_b: str = "Breast",
    top_k: int = 9,
    save_path: str | None = None,
) -> plt.Figure:
    """Render a side-by-side comparison of two matched features.

    Left panel:  3x3 grid of top-activating crops from SAE A
    Right panel: 3x3 grid of top-activating crops from SAE B
    Title: feature indices and cosine similarity
    """
    grid_side = int(np.ceil(np.sqrt(top_k)))

    fig = plt.figure(figsize=(14, 7))
    gs = gridspec.GridSpec(1, 2, wspace=0.3)

    for panel_idx, (feat_idx, sae, acts, img_paths, config, label) in enumerate([
        (feature_idx_a, sae_a, activations_a, image_paths_a, config_a, label_a),
        (feature_idx_b, sae_b, activations_b, image_paths_b, config_b, label_b),
    ]):
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            grid_side, grid_side, subplot_spec=gs[panel_idx], wspace=0.05, hspace=0.15
        )

        # Encode all activations to get per-image feature max
        n_patches = config.n_patches
        N_eff = acts.shape[0]
        n_images = N_eff // n_patches

        # Compute feature_max for this single feature efficiently
        device = sae.W_enc.device
        feature_max = torch.zeros(n_images)
        feature_argmax = torch.zeros(n_images, dtype=torch.long)
        batch_imgs = max(1, 4096 // n_patches)

        for img_start in range(0, n_images, batch_imgs):
            img_end = min(img_start + batch_imgs, n_images)
            chunk = acts[img_start * n_patches : img_end * n_patches].to(device)
            codes = sae.encode(chunk).cpu()  # [B*n_patches, d_sae]
            codes = codes.reshape(img_end - img_start, n_patches, -1)
            feat_codes = codes[:, :, feat_idx]  # [B, n_patches]
            max_vals, max_idxs = feat_codes.max(dim=1)
            feature_max[img_start:img_end] = max_vals
            feature_argmax[img_start:img_end] = max_idxs

        # Get top-k images
        k_actual = min(top_k, n_images)
        top_indices = torch.topk(feature_max, k_actual).indices.tolist()

        for cell_idx in range(grid_side * grid_side):
            ax = fig.add_subplot(inner_gs[cell_idx])
            ax.axis("off")

            if cell_idx >= k_actual:
                continue

            img_idx = top_indices[cell_idx]
            patch_idx = int(feature_argmax[img_idx].item())

            try:
                img = Image.open(img_paths[img_idx]).convert("RGB")
                crop = get_patch_crop(
                    img, patch_idx,
                    config.spatial_h, config.spatial_w,
                    context_patches=1,
                )
                ax.imshow(np.array(crop))
                act_val = feature_max[img_idx].item()
                ax.set_title(f"{act_val:.2f}", fontsize=6, pad=2)
            except Exception as e:
                ax.text(0.5, 0.5, "err", ha="center", va="center",
                        fontsize=7, transform=ax.transAxes)

        # Panel title
        side_label = "LEFT" if panel_idx == 0 else "RIGHT"
        fig.text(
            0.25 + 0.5 * panel_idx, 0.92,
            f"{label} — Feature {feat_idx}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    fig.suptitle(
        f"Cosine Similarity: {cosine_sim:.4f}",
        fontsize=13, fontweight="bold", y=0.98,
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ──────────────────────────────────────────────────────────────────────────
# 5. Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cross-SAE cosine similarity analysis for universality detection."
    )

    # SAE A (colon)
    parser.add_argument("--checkpoint_a", type=str, required=True,
                        help="Checkpoint dir for SAE A (colon)")
    parser.add_argument("--activations_a", type=str, required=True,
                        help="Activations .safetensors for SAE A")
    parser.add_argument("--images_a", type=str, required=True,
                        help="Images directory for SAE A")
    parser.add_argument("--label_a", type=str, default="colon",
                        help="Label for SAE A (default: colon)")

    # SAE B (breast)
    parser.add_argument("--checkpoint_b", type=str, required=True,
                        help="Checkpoint dir for SAE B (breast)")
    parser.add_argument("--activations_b", type=str, required=True,
                        help="Activations .safetensors for SAE B")
    parser.add_argument("--images_b", type=str, required=True,
                        help="Images directory for SAE B")
    parser.add_argument("--label_b", type=str, default="breast",
                        help="Label for SAE B (default: breast)")

    # Similarity settings
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Minimum cosine similarity threshold (default: 0.7)")
    parser.add_argument("--top_k_vis", type=int, default=50,
                        help="Number of top pairs to visualise (default: 50, 0=all)")

    # Output
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for CSVs and visualisations")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (default: cpu)")

    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load SAEs ──────────────────────────────────────────────────────
    print(f"Loading SAE A ({args.label_a}) from {args.checkpoint_a}")
    sae_a = load_sae(args.checkpoint_a, device=args.device)
    print(f"  d_model={sae_a.d_model}, d_sae={sae_a.d_sae}, k={sae_a.k}")

    print(f"Loading SAE B ({args.label_b}) from {args.checkpoint_b}")
    sae_b = load_sae(args.checkpoint_b, device=args.device)
    print(f"  d_model={sae_b.d_model}, d_sae={sae_b.d_sae}, k={sae_b.k}")

    # ── Export feature CSVs ────────────────────────────────────────────
    print("\nExporting feature CSVs...")
    export_feature_csv(sae_a, out / f"features_{args.label_a}.csv", args.label_a)
    export_feature_csv(sae_b, out / f"features_{args.label_b}.csv", args.label_b)

    # ── Compute cosine similarity ──────────────────────────────────────
    print("\nComputing cosine similarity matrix...")
    sim_matrix = compute_decoder_cosine_similarity(sae_a, sae_b)
    print(f"  Shape: {sim_matrix.shape}")
    print(f"  Max similarity: {sim_matrix.max().item():.4f}")
    print(f"  Pairs >= {args.threshold}: {(sim_matrix >= args.threshold).sum().item()}")

    # ── Export similarity CSV ──────────────────────────────────────────
    print("\nExporting similarity matrix CSV...")
    matched_pairs = export_similarity_csv(
        sim_matrix, out / "similarity_matrix.csv",
        threshold=args.threshold,
        label_a=args.label_a, label_b=args.label_b,
    )

    if not matched_pairs:
        print(f"\nNo pairs found above threshold {args.threshold}. Done.")
        return

    # ── Visualise matched pairs ────────────────────────────────────────
    n_vis = len(matched_pairs) if args.top_k_vis == 0 else min(args.top_k_vis, len(matched_pairs))
    print(f"\nVisualising top {n_vis} matched pairs...")

    # Load activations and images for visualisation
    print(f"Loading activations A from {args.activations_a}")
    acts_a, meta_a = load_activations(args.activations_a, device="cpu")
    config_a = VisConfig.from_checkpoint(args.checkpoint_a, device=args.device)
    if "spatial_h" in meta_a:
        config_a.spatial_h = int(meta_a["spatial_h"])
    if "spatial_w" in meta_a:
        config_a.spatial_w = int(meta_a["spatial_w"])

    print(f"Loading activations B from {args.activations_b}")
    acts_b, meta_b = load_activations(args.activations_b, device="cpu")
    config_b = VisConfig.from_checkpoint(args.checkpoint_b, device=args.device)
    if "spatial_h" in meta_b:
        config_b.spatial_h = int(meta_b["spatial_h"])
    if "spatial_w" in meta_b:
        config_b.spatial_w = int(meta_b["spatial_w"])

    # Load image paths
    from visualisation.loaders import load_images
    img_paths_a = load_images(args.images_a)
    img_paths_b = load_images(args.images_b)
    print(f"  Images A: {len(img_paths_a)}, Images B: {len(img_paths_b)}")

    vis_dir = out / "matched_pairs"
    vis_dir.mkdir(parents=True, exist_ok=True)

    for rank, (idx_a, idx_b, sim) in enumerate(tqdm(matched_pairs[:n_vis], desc="Rendering pairs")):
        save_path = str(vis_dir / f"rank_{rank:04d}_{args.label_a}_F{idx_a}_{args.label_b}_F{idx_b}_sim{sim:.3f}.png")
        fig = render_matched_pair(
            feature_idx_a=idx_a,
            feature_idx_b=idx_b,
            cosine_sim=sim,
            sae_a=sae_a, sae_b=sae_b,
            activations_a=acts_a, activations_b=acts_b,
            image_paths_a=img_paths_a, image_paths_b=img_paths_b,
            config_a=config_a, config_b=config_b,
            label_a=args.label_a, label_b=args.label_b,
            save_path=save_path,
        )
        plt.close(fig)

    print(f"\nDone! All outputs in {out}/")
    print(f"  - features_{args.label_a}.csv")
    print(f"  - features_{args.label_b}.csv")
    print(f"  - similarity_matrix.csv ({len(matched_pairs)} pairs)")
    print(f"  - matched_pairs/ ({n_vis} visualisations)")


if __name__ == "__main__":
    main()
