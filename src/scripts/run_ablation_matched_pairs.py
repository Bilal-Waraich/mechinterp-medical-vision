"""Run patch ablation on reciprocal-best-match feature pairs from a
cosine-similarity analysis, with automatic positional feature filtering.

A feature is classified as *positional* if its top-5 important patches are
identical across all test images (i.e., it always fires on the same grid
locations regardless of content). These are filtered out so only semantic
features remain.

Usage
-----
    python -m src.scripts.run_ablation_matched_pairs \
        --checkpoint_a  /path/to/sae_a/step_0050000 \
        --activations_a /path/to/activations_a.safetensors \
        --images_a      /path/to/patches_a \
        --label_a       colon \
        --checkpoint_b  /path/to/sae_b/step_0050000 \
        --activations_b /path/to/activations_b.safetensors \
        --images_b      /path/to/patches_b \
        --label_b       breast \
        --matches_csv   /path/to/reciprocal_best_matches.csv \
        --output_dir    /path/to/output \
        --top_n         50 \
        --n_images      5 \
        --device        cuda:0
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from visualisation.loaders import load_sae, load_activations
from visualisation.ablation import run_patch_ablation, render_ablation_figure

# ── Spatial config (layer-2: 7×7 patches) ────────────────────────────────
SPATIAL_H = 7
SPATIAL_W = 7
N_PATCHES = SPATIAL_H * SPATIAL_W  # 49


# ── Positional feature detection ─────────────────────────────────────────

def is_positional_feature(
    sae,
    feature_idx: int,
    flat_activations: torch.Tensor,
    image_indices: list[int],
    n_patches: int = N_PATCHES,
    spatial_h: int = SPATIAL_H,
    spatial_w: int = SPATIAL_W,
) -> tuple[bool, float]:
    """Check if a feature is positional by measuring top-patch consistency.

    Runs deletion ablation on each image and checks whether the top-5
    important patches are the same across all images.

    Returns
    -------
    is_positional : bool
        True if the Jaccard similarity of top-5 patches across all image
        pairs exceeds 0.8 (i.e., the feature always fires on the same
        grid positions).
    mean_jaccard : float
        Mean pairwise Jaccard similarity of top-5 patch sets.
    """
    top_patch_sets = []
    for img_idx in image_indices:
        result = run_patch_ablation(
            feature_idx=feature_idx,
            image_idx=img_idx,
            sae=sae,
            flat_activations=flat_activations,
            n_patches=n_patches,
            spatial_h=spatial_h,
            spatial_w=spatial_w,
            mode="deletion",
        )
        top_patch_sets.append(set(result["most_important_patches"]))

    if len(top_patch_sets) < 2:
        return False, 0.0

    jaccards = []
    for i in range(len(top_patch_sets)):
        for j in range(i + 1, len(top_patch_sets)):
            intersection = len(top_patch_sets[i] & top_patch_sets[j])
            union = len(top_patch_sets[i] | top_patch_sets[j])
            jaccards.append(intersection / union if union > 0 else 0.0)

    mean_jaccard = float(np.mean(jaccards))
    return mean_jaccard > 0.5, mean_jaccard


# ── Helpers ───────────────────────────────────────────────────────────────

def load_matches(path: Path, top_n: int, label_a: str, label_b: str) -> list[dict]:
    """Read the top_n rows from reciprocal_best_matches.csv.

    Auto-detects column names from the CSV header. Expects columns like
    ``{label_a}_feature_idx``, ``{label_b}_feature_idx``, ``cosine_similarity``.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []

        # Find the feature index columns
        col_a = next((c for c in cols if label_a in c.lower() and "feature" in c.lower()), None)
        col_b = next((c for c in cols if label_b in c.lower() and "feature" in c.lower()), None)
        col_sim = next((c for c in cols if "cosine" in c.lower() or "similarity" in c.lower()), None)

        if not (col_a and col_b and col_sim):
            raise ValueError(
                f"Cannot find columns for labels '{label_a}'/'{label_b}' in {cols}. "
                f"Expected columns containing the label name + 'feature' and a similarity column."
            )

        for i, row in enumerate(reader):
            if i >= top_n:
                break
            rows.append({
                "rank": i,
                "feat_a": int(row[col_a]),
                "feat_b": int(row[col_b]),
                "cosine_sim": float(row[col_sim]),
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


def run_single_side_ablation(
    sae,
    feature_idx: int,
    flat_activations: torch.Tensor,
    image_indices: list[int],
    image_paths: list[Path],
    label: str,
    pair_dir: Path,
) -> list[dict]:
    """Run ablation on one side (dataset) and save figures."""
    ablation_results = []
    for img_idx in image_indices:
        abl = run_patch_ablation(
            feature_idx=feature_idx,
            image_idx=img_idx,
            sae=sae,
            flat_activations=flat_activations,
            n_patches=N_PATCHES,
            spatial_h=SPATIAL_H,
            spatial_w=SPATIAL_W,
            mode="deletion",
        )
        if img_idx < len(image_paths):
            fig_path = pair_dir / f"{label}_F{feature_idx}_img{img_idx}_deletion.png"
            render_ablation_figure(
                abl, image_paths[img_idx],
                spatial_h=SPATIAL_H, spatial_w=SPATIAL_W,
                save_path=str(fig_path),
            )
            plt.close("all")

        ablation_results.append({
            "image_idx": img_idx,
            "baseline_act": abl["baseline_act"],
            "top_patches": abl["most_important_patches"],
            "delta_range": [float(abl["delta_map"].min()), float(abl["delta_map"].max())],
        })
    return ablation_results


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ablation on matched feature pairs with positional filtering"
    )
    parser.add_argument("--checkpoint_a", type=str, required=True, help="SAE checkpoint dir for dataset A")
    parser.add_argument("--activations_a", type=str, required=True, help="Activations .safetensors for A")
    parser.add_argument("--images_a", type=str, required=True, help="Patches directory for A")
    parser.add_argument("--label_a", type=str, required=True, help="Label for dataset A (e.g. 'colon')")
    parser.add_argument("--checkpoint_b", type=str, required=True, help="SAE checkpoint dir for dataset B")
    parser.add_argument("--activations_b", type=str, required=True, help="Activations .safetensors for B")
    parser.add_argument("--images_b", type=str, required=True, help="Patches directory for B")
    parser.add_argument("--label_b", type=str, required=True, help="Label for dataset B (e.g. 'breast')")
    parser.add_argument("--matches_csv", type=str, required=True, help="Path to reciprocal_best_matches.csv")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--top_n", type=int, default=50, help="Max matched pairs to scan (before filtering)")
    parser.add_argument("--max_semantic", type=int, default=20, help="Stop after this many semantic (non-positional) pairs")
    parser.add_argument("--n_images", type=int, default=5, help="Top-activating images per feature")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading matched pairs...")
    matches = load_matches(Path(args.matches_csv), args.top_n, args.label_a, args.label_b)
    print(f"  {len(matches)} pairs loaded (top cosine sim: {matches[0]['cosine_sim']:.3f})")

    print(f"Loading {args.label_a} SAE + activations...")
    sae_a = load_sae(args.checkpoint_a, device=args.device)
    acts_a, meta_a = load_activations(args.activations_a, device=args.device)
    n_images_a = meta_a.get("n_images", acts_a.shape[0] // N_PATCHES)
    images_a = get_image_paths(Path(args.images_a)) if Path(args.images_a).exists() else []
    print(f"  {args.label_a}: {n_images_a} images, acts shape {acts_a.shape}")

    print(f"Loading {args.label_b} SAE + activations...")
    sae_b = load_sae(args.checkpoint_b, device=args.device)
    acts_b, meta_b = load_activations(args.activations_b, device=args.device)
    n_images_b = meta_b.get("n_images", acts_b.shape[0] // N_PATCHES)
    images_b = get_image_paths(Path(args.images_b)) if Path(args.images_b).exists() else []
    print(f"  {args.label_b}: {n_images_b} images, acts shape {acts_b.shape}")

    semantic_results = []
    positional_log = []
    semantic_rank = 0

    for match in tqdm(matches, desc="Scanning matched pairs"):
        if len(semantic_results) >= args.max_semantic:
            break

        feat_a, feat_b = match["feat_a"], match["feat_b"]
        rank = match["rank"]

        # Find top images for positional check
        top_imgs_a = find_top_images(acts_a, sae_a, feat_a, n_images_a, args.n_images)
        top_imgs_b = find_top_images(acts_b, sae_b, feat_b, n_images_b, args.n_images)

        # Check if either side is positional
        pos_a, jaccard_a = is_positional_feature(sae_a, feat_a, acts_a, top_imgs_a)
        pos_b, jaccard_b = is_positional_feature(sae_b, feat_b, acts_b, top_imgs_b)

        if pos_a or pos_b:
            entry = {
                "original_rank": rank,
                f"{args.label_a}_feature": feat_a,
                f"{args.label_b}_feature": feat_b,
                "cosine_similarity": match["cosine_sim"],
                f"{args.label_a}_jaccard": round(jaccard_a, 3),
                f"{args.label_b}_jaccard": round(jaccard_b, 3),
                "reason": "positional" if (pos_a and pos_b) else (
                    f"{args.label_a}_positional" if pos_a else f"{args.label_b}_positional"
                ),
            }
            positional_log.append(entry)
            print(f"  [SKIP] rank {rank}: {args.label_a}_F{feat_a}/{args.label_b}_F{feat_b} "
                  f"— positional (jaccard: {jaccard_a:.2f}/{jaccard_b:.2f})")
            continue

        # Semantic feature — run full ablation
        print(f"  [KEEP] rank {rank}: {args.label_a}_F{feat_a}/{args.label_b}_F{feat_b} "
              f"— semantic (jaccard: {jaccard_a:.2f}/{jaccard_b:.2f})")

        pair_dir = output_dir / (
            f"semantic_{semantic_rank:04d}_orig{rank:04d}_"
            f"{args.label_a}_F{feat_a}_{args.label_b}_F{feat_b}"
        )
        pair_dir.mkdir(parents=True, exist_ok=True)

        abl_a = run_single_side_ablation(
            sae_a, feat_a, acts_a, top_imgs_a, images_a, args.label_a, pair_dir,
        )
        abl_b = run_single_side_ablation(
            sae_b, feat_b, acts_b, top_imgs_b, images_b, args.label_b, pair_dir,
        )

        result = {
            "semantic_rank": semantic_rank,
            "original_rank": rank,
            f"{args.label_a}_feature": feat_a,
            f"{args.label_b}_feature": feat_b,
            "cosine_similarity": match["cosine_sim"],
            f"{args.label_a}_jaccard": round(jaccard_a, 3),
            f"{args.label_b}_jaccard": round(jaccard_b, 3),
            f"{args.label_a}_ablations": abl_a,
            f"{args.label_b}_ablations": abl_b,
        }

        with open(pair_dir / "summary.json", "w") as f:
            json.dump(result, f, indent=2)

        semantic_results.append(result)
        semantic_rank += 1

    # Save global outputs
    with open(output_dir / "ablation_summary_semantic.json", "w") as f:
        json.dump(semantic_results, f, indent=2)

    with open(output_dir / "positional_features_filtered.json", "w") as f:
        json.dump(positional_log, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done. Scanned {len(semantic_results) + len(positional_log)} pairs total.")
    print(f"  Semantic (kept):     {len(semantic_results)}")
    print(f"  Positional (filtered): {len(positional_log)}")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
