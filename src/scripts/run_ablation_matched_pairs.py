"""Run patch ablation on the top-N reciprocal-best-match feature pairs
from the colon–breast cosine-similarity analysis.

For each matched pair (colon_F, breast_F) we run deletion ablation on
the top-activating images in each respective dataset, producing spatial
importance maps and figures.

Usage
-----
    python -m src.scripts.run_ablation_matched_pairs \
        --top_n 10 \
        --n_images 5 \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from visualisation.loaders import load_sae, load_activations
from visualisation.ablation import run_patch_ablation, render_ablation_figure

# ── Paths ─────────────────────────────────────────────────────────────────
BASE = Path(r"\\fileserver\share\deep_learning\output\UMed-SAE\batch-topk\layer-2")

COLON_CKPT = BASE / "SemiCOL" / "SemiCOL_v1" / "track_a" / "checkpoints" / "step_0050000"
COLON_ACTS = BASE / "SemiCOL" / "SemiCOL_v1" / "activations_track_a.safetensors"
COLON_PATCHES = BASE / "SemiCOL" / "SemiCOL_v1" / "patches_track_a"
COLON_STATS = BASE / "SemiCOL" / "SemiCOL_v1" / "track_a" / "vis_outputs" / "stats.pt"

BREAST_CKPT = BASE / "tcga-breast" / "tcga-breast_v1" / "checkpoints" / "step_0050000"
BREAST_ACTS = BASE / "tcga-breast" / "tcga-breast_v1" / "activations.safetensors"
BREAST_PATCHES = BASE / "tcga-breast" / "tcga-breast_v1" / "patches"

MATCHES_CSV = BASE / "cosine-similarity-colon-breast-v2" / "reciprocal_best_matches.csv"

OUTPUT_DIR = BASE / "cosine-similarity-colon-breast-v2" / "ablation_results"

# ── Spatial config (layer-2: 7×7 patches) ────────────────────────────────
SPATIAL_H = 7
SPATIAL_W = 7
N_PATCHES = SPATIAL_H * SPATIAL_W  # 49


def load_matches(path: Path, top_n: int) -> list[dict]:
    """Read the top_n rows from reciprocal_best_matches.csv."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= top_n:
                break
            rows.append({
                "rank": i,
                "colon_idx": int(row["colon_feature_idx"]),
                "breast_idx": int(row["breast_feature_idx"]),
                "cosine_sim": float(row["cosine_similarity"]),
            })
    return rows


def find_top_images(
    flat_activations: torch.Tensor,
    sae,
    feature_idx: int,
    n_images_total: int,
    n_top: int,
) -> list[int]:
    """Find the images where `feature_idx` has the highest mean activation."""
    mean_acts = torch.zeros(n_images_total)
    with torch.no_grad():
        for img_i in range(n_images_total):
            acts = flat_activations[img_i * N_PATCHES : (img_i + 1) * N_PATCHES]
            encoded = sae.encode(acts)
            mean_acts[img_i] = encoded[:, feature_idx].mean()
    top_indices = torch.topk(mean_acts, k=min(n_top, n_images_total)).indices.tolist()
    return top_indices


def get_image_paths(patches_dir: Path) -> list[Path]:
    """Sorted list of image paths in patches_dir."""
    exts = {".png", ".jpg", ".jpeg"}
    return sorted(p for p in patches_dir.iterdir() if p.suffix.lower() in exts)


def run_pair_ablation(
    rank: int,
    colon_feat: int,
    breast_feat: int,
    cosine_sim: float,
    colon_sae,
    breast_sae,
    colon_acts: torch.Tensor,
    breast_acts: torch.Tensor,
    colon_images: list[Path],
    breast_images: list[Path],
    n_colon_images: int,
    n_breast_images: int,
    n_top_images: int,
    output_dir: Path,
    device: str,
) -> dict:
    """Run ablation for one matched pair on both datasets."""

    pair_dir = output_dir / f"rank_{rank:04d}_colon_F{colon_feat}_breast_F{breast_feat}"
    pair_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "rank": rank,
        "colon_feature": colon_feat,
        "breast_feature": breast_feat,
        "cosine_similarity": cosine_sim,
        "colon_ablations": [],
        "breast_ablations": [],
    }

    # --- Colon side ---
    colon_top = find_top_images(colon_acts, colon_sae, colon_feat, n_colon_images, n_top_images)
    for img_idx in colon_top:
        abl = run_patch_ablation(
            feature_idx=colon_feat,
            image_idx=img_idx,
            sae=colon_sae,
            flat_activations=colon_acts,
            n_patches=N_PATCHES,
            spatial_h=SPATIAL_H,
            spatial_w=SPATIAL_W,
            mode="deletion",
        )
        # Save figure if image exists
        if img_idx < len(colon_images):
            fig_path = pair_dir / f"colon_F{colon_feat}_img{img_idx}_deletion.png"
            render_ablation_figure(
                abl, colon_images[img_idx],
                spatial_h=SPATIAL_H, spatial_w=SPATIAL_W,
                save_path=str(fig_path),
            )
            import matplotlib.pyplot as plt
            plt.close("all")

        results["colon_ablations"].append({
            "image_idx": img_idx,
            "baseline_act": abl["baseline_act"],
            "top_patches": abl["most_important_patches"],
            "delta_range": [float(abl["delta_map"].min()), float(abl["delta_map"].max())],
        })

    # --- Breast side ---
    breast_top = find_top_images(breast_acts, breast_sae, breast_feat, n_breast_images, n_top_images)
    for img_idx in breast_top:
        abl = run_patch_ablation(
            feature_idx=breast_feat,
            image_idx=img_idx,
            sae=breast_sae,
            flat_activations=breast_acts,
            n_patches=N_PATCHES,
            spatial_h=SPATIAL_H,
            spatial_w=SPATIAL_W,
            mode="deletion",
        )
        if img_idx < len(breast_images):
            fig_path = pair_dir / f"breast_F{breast_feat}_img{img_idx}_deletion.png"
            render_ablation_figure(
                abl, breast_images[img_idx],
                spatial_h=SPATIAL_H, spatial_w=SPATIAL_W,
                save_path=str(fig_path),
            )
            import matplotlib.pyplot as plt
            plt.close("all")

        results["breast_ablations"].append({
            "image_idx": img_idx,
            "baseline_act": abl["baseline_act"],
            "top_patches": abl["most_important_patches"],
            "delta_range": [float(abl["delta_map"].min()), float(abl["delta_map"].max())],
        })

    # Save per-pair JSON summary
    with open(pair_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Ablation on matched colon-breast feature pairs")
    parser.add_argument("--top_n", type=int, default=10, help="Number of top matched pairs to ablate")
    parser.add_argument("--n_images", type=int, default=5, help="Top-activating images per feature")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading matched pairs...")
    matches = load_matches(MATCHES_CSV, args.top_n)
    print(f"  {len(matches)} pairs loaded (top cosine sim: {matches[0]['cosine_sim']:.3f})")

    print("Loading colon SAE + activations...")
    colon_sae = load_sae(str(COLON_CKPT), device=args.device)
    colon_acts, colon_meta = load_activations(str(COLON_ACTS), device=args.device)
    n_colon_images = colon_meta.get("n_images", colon_acts.shape[0] // N_PATCHES)
    colon_images = get_image_paths(COLON_PATCHES) if COLON_PATCHES.exists() else []
    print(f"  Colon: {n_colon_images} images, acts shape {colon_acts.shape}")

    print("Loading breast SAE + activations...")
    breast_sae = load_sae(str(BREAST_CKPT), device=args.device)
    breast_acts, breast_meta = load_activations(str(BREAST_ACTS), device=args.device)
    n_breast_images = breast_meta.get("n_images", breast_acts.shape[0] // N_PATCHES)
    breast_images = get_image_paths(BREAST_PATCHES) if BREAST_PATCHES.exists() else []
    print(f"  Breast: {n_breast_images} images, acts shape {breast_acts.shape}")

    all_results = []
    for match in tqdm(matches, desc="Ablating matched pairs"):
        result = run_pair_ablation(
            rank=match["rank"],
            colon_feat=match["colon_idx"],
            breast_feat=match["breast_idx"],
            cosine_sim=match["cosine_sim"],
            colon_sae=colon_sae,
            breast_sae=breast_sae,
            colon_acts=colon_acts,
            breast_acts=breast_acts,
            colon_images=colon_images,
            breast_images=breast_images,
            n_colon_images=n_colon_images,
            n_breast_images=n_breast_images,
            n_top_images=args.n_images,
            output_dir=output_dir,
            device=args.device,
        )
        all_results.append(result)

    # Save global summary
    summary_path = output_dir / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDone. Results saved to {output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
