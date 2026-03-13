"""
Cross-dataset evaluation: compute SAE feature universality across multiple datasets.

Processes each dataset's activations through the trained SAE separately (memory-efficient),
then concatenates per-image feature stats and renders the universality dashboard.

Usage
-----
python src/scripts/cross_dataset_eval.py \\
    --checkpoint_dir /deep_learning/output/.../colon-sae/track_b/checkpoints/step_0200000 \\
    --activations /deep_learning/output/.../colon-sae/track_b/activations_track_b.safetensors:semicol_b \\
    --activations /deep_learning/output/.../colon-sae/activations_track_a.safetensors:semicol_a \\
    --activations /deep_learning/output/.../brain-sae/activations.safetensors:brain_tumor \\
    --activations /deep_learning/output/.../sae-tcga-breast/activations.safetensors:tcga_breast \\
    --output_dir /deep_learning/output/.../universality \\
    --top_features 50 \\
    --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file
from tqdm.auto import tqdm

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from visualisation.loaders import load_sae
from visualisation.universality import (
    render_universality_dashboard,
    render_co_activation_heatmap,
    compute_co_activation_matrix,
)
from visualisation.collector import save_stats


# ── Per-dataset feature-max extraction ────────────────────────────────────────

@torch.no_grad()
def extract_feature_max(
    activations_path: str,
    sae,
    device: torch.device,
    batch_size: int = 4096,
) -> tuple[torch.Tensor, int]:
    """Run SAE encoder over a safetensors file, return per-image feature_max.

    Args:
        activations_path: Path to a ``activations_*.safetensors`` file produced
            by ``harvest_activations.py``.
        sae: Loaded BatchTopKSAE (already on ``device``).
        device: Torch device.
        batch_size: Token-level batch size for SAE inference.

    Returns:
        feature_max: float32 tensor of shape ``[n_images, d_sae]``.
        n_images: number of images in this dataset.
    """
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] Loading {activations_path} …")

    raw = load_file(str(activations_path))
    acts: torch.Tensor = raw["activations"].to(torch.float32)
    meta = {k: v for k, v in raw.items() if k != "activations"}

    n_eff, d_model = acts.shape
    n_patches = int(meta.get("spatial_h", b"14")) * int(meta.get("spatial_w", b"14"))
    # metadata values are byte-strings in safetensors
    try:
        n_patches = int(meta["spatial_h"]) * int(meta["spatial_w"])
    except (KeyError, ValueError):
        n_patches = 196  # default: Stage 2→3, 14×14

    n_images = n_eff // n_patches
    d_sae = sae.d_sae

    print(f"       shape={list(acts.shape)}  n_images={n_images}  n_patches={n_patches}")

    # Process in token-level batches to bound GPU memory
    feature_acts = torch.empty(n_eff, d_sae, dtype=torch.float32, device="cpu")

    for start in tqdm(range(0, n_eff, batch_size), desc="  SAE encode", leave=False):
        end = min(start + batch_size, n_eff)
        x = acts[start:end].to(device)
        enc = sae.encode(x)          # [batch, d_sae]  (top-k sparse)
        feature_acts[start:end] = enc.cpu()

    # Reshape to [n_images, n_patches, d_sae], take max over patches
    feature_acts = feature_acts.reshape(n_images, n_patches, d_sae)
    feature_max = feature_acts.max(dim=1).values   # [n_images, d_sae]

    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}]   → feature_max shape: {list(feature_max.shape)}")
    return feature_max, n_images


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Cross-dataset SAE feature universality evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--checkpoint_dir", required=True,
        help="Path to a step_XXXXXXX checkpoint directory (contains sae_weights.safetensors).",
    )
    ap.add_argument(
        "--activations", action="append", metavar="PATH:LABEL", required=True,
        help="Activations safetensors path and dataset label separated by ':'. "
             "Repeat for each dataset, e.g. --activations colon.safetensors:semicol_b.",
    )
    ap.add_argument("--output_dir", required=True, help="Directory to save all outputs.")
    ap.add_argument("--top_features", type=int, default=50,
                    help="Number of top features (by global max activation) to visualise.")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=4096,
                    help="Token-level batch size for SAE inference per dataset.")
    args = ap.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Parse activations list ─────────────────────────────────────────────
    datasets: list[tuple[str, str]] = []
    for entry in args.activations:
        if ":" not in entry:
            ap.error(f"--activations must be PATH:LABEL, got: {entry!r}")
        path, label = entry.rsplit(":", 1)
        datasets.append((path, label))

    print("=" * 60)
    print("Cross-Dataset SAE Universality Evaluation")
    print("=" * 60)
    print(f"Checkpoint : {args.checkpoint_dir}")
    print(f"Datasets   : {[label for _, label in datasets]}")
    print(f"Output     : {args.output_dir}")
    print(f"Device     : {device}")
    print()

    # ── Load SAE ──────────────────────────────────────────────────────────
    print(f"Loading SAE from {args.checkpoint_dir} …")
    sae = load_sae(args.checkpoint_dir, device=str(device))
    sae.eval()
    print(f"  d_model={sae.d_model}  d_sae={sae.d_sae}  k={sae.k}")

    # ── Per-dataset inference ──────────────────────────────────────────────
    all_feature_max: list[torch.Tensor] = []
    all_labels: list[str] = []

    for acts_path, label in datasets:
        print(f"\n── Dataset: {label} ──")
        fm, n_img = extract_feature_max(acts_path, sae, device, batch_size=args.batch_size)
        all_feature_max.append(fm)
        all_labels.extend([label] * n_img)

    # ── Combine ───────────────────────────────────────────────────────────
    combined_max = torch.cat(all_feature_max, dim=0)   # [n_total, d_sae]
    n_total = combined_max.shape[0]

    print(f"\nCombined feature_max: {list(combined_max.shape)}  "
          f"({n_total} images across {len(datasets)} datasets)")

    global_max = combined_max.max(dim=0).values        # [d_sae]

    # Save combined stats for downstream use
    stats = {
        "feature_max":  combined_max,
        "feature_mean": combined_max,   # approximate: max≈mean for universality
        "global_max":   global_max,
        "n_images":     n_total,
    }
    stats_path = str(out_dir / "cross_dataset_stats.pt")
    save_stats(stats, stats_path)
    print(f"  ✓ Saved combined stats → {stats_path}")

    # Save dataset labels alongside stats
    labels_path = out_dir / "dataset_labels.json"
    labels_path.write_text(json.dumps(all_labels, indent=2))
    print(f"  ✓ Saved dataset labels → {labels_path}")

    # ── Top features ──────────────────────────────────────────────────────
    top_indices = global_max.argsort(descending=True)[: args.top_features].tolist()
    print(f"\nTop-{args.top_features} features: {top_indices[:10]} …")

    # ── Universality dashboard ─────────────────────────────────────────────
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{ts}] Rendering universality dashboard …")
    dash_path = str(out_dir / "universality_dashboard.png")
    render_universality_dashboard(
        feature_indices=top_indices,
        stats=stats,
        dataset_labels=all_labels,
        save_path=dash_path,
    )
    print(f"  ✓ Saved → {dash_path}")

    # ── Co-activation heatmap ──────────────────────────────────────────────
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] Computing co-activation matrix for top-{args.top_features} features …")
    heatmap_path = str(out_dir / "co_activation_heatmap.png")
    feature_labels = [f"F{i}" for i in top_indices]
    render_co_activation_heatmap(
        feature_indices=top_indices,
        stats=stats,
        feature_labels=feature_labels,
        save_path=heatmap_path,
    )
    print(f"  ✓ Saved → {heatmap_path}")

    print(f"\n{'=' * 60}")
    print(f"  Done. All outputs in: {out_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
