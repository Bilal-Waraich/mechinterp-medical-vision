"""Activation Patching — Cross-SAE Feature Interchangeability (Evidence Layer 5).

Tests whether matched SAE features are *causally interchangeable* by swapping
decoder directions at the SAE hook point (pm_index=2) and measuring downstream
representation change through Stage 4 of the shared Swin Transformer backbone.

Procedure per matched pair (F_A in SAE_A, F_B in SAE_B):
  1. Load top-N images for F_A from patches_dir_A
  2. Forward each image through the encoder; hook captures raw activation
     at pm_index=2  (B, 7, 7, 1024)
  3. Normalize activation (Anthropic-style), encode with SAE_A → get F_A magnitude
  4. Compute delta: magnitude × (W_dec_B[F_B] − W_dec_A[F_A])
  5. Map delta from normalized → raw space, add to original activation
  6. Forward both original and patched activation through Stage 4 + LayerNorm
  7. Cosine similarity of the two Stage 4 outputs = "patching score"

Score interpretation:
  1.000 = perfectly interchangeable features (no downstream change)
  <0.99 = features cause measurably different downstream representations

This script is fully self-contained.  It reads from existing outputs
(SAE checkpoints, stats.pt, patches_dir, reciprocal_best_matches.csv)
and writes a single JSON output.  It does NOT modify any existing files
or data.

Usage:
  python src/scripts/run_activation_patching.py \\
      --checkpoint  /path/to/encoder.pt \\
      --sae_a       /path/to/sae_a_checkpoint \\
      --sae_b       /path/to/sae_b_checkpoint \\
      --stats_a     /path/to/stats_a.pt \\
      --patches_dir /path/to/patches_a \\
      --matches_csv /path/to/reciprocal_best_matches.csv \\
      --label_a colon --label_b breast \\
      --output      /path/to/activation_patching.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# ── Ensure src/ and project root are importable ──────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent
_PROJECT_DIR = _SRC_DIR.parent
for _p in [str(_SRC_DIR), str(_PROJECT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from visualisation.loaders import load_sae  # noqa: E402

EPS = 1e-8
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ─────────────────────────────────────────────────────────────────────────
# Model loading (mirrors harvest_activations, kept standalone for isolation)
# ─────────────────────────────────────────────────────────────────────────

def _setup_mmm():
    """Ensure UmedPT packages are importable."""
    candidates = [
        Path("/app"),
        Path(__file__).resolve().parent.parent.parent,
        Path.cwd(),
    ]
    for root in candidates:
        for pkg in ("UmedPT", "UmedPT-sdk"):
            p = str(root / pkg)
            if Path(p).exists() and p not in sys.path:
                sys.path.insert(0, p)
    os.environ.setdefault("MMM_LICENSE_ACCEPTED", "i accept")


def load_encoder(checkpoint: str, device: str):
    """Load PyramidEncoder from a .pt file."""
    _setup_mmm()
    ckpt = Path(checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Encoder not found: {ckpt}")
    print(f"[Encoder] Loading from {ckpt}")
    encoder = torch.load(str(ckpt), map_location=device, weights_only=False)
    encoder.eval()
    print(f"[Encoder] Loaded ({type(encoder).__name__}) → {device}")
    return encoder


# ─────────────────────────────────────────────────────────────────────────
# Hook infrastructure
# ─────────────────────────────────────────────────────────────────────────

def _find_pm_layers(encoder):
    """Return [(feat_idx, module), ...] for all PatchMerging layers."""
    from torchvision.models.swin_transformer import PatchMergingV2
    features = encoder.model.wrapped_model.features
    pm_layers = [
        (i, m) for i, m in enumerate(features)
        if isinstance(m, PatchMergingV2)
    ]
    # Add the final element (LayerNorm / last stage) as pm_index=3
    pm_layers.append((len(features) - 1, features[-1]))
    return pm_layers


def register_hook(encoder, pm_index: int = 2):
    """Attach a forward hook at the given PatchMerging layer."""
    pm_layers = _find_pm_layers(encoder)
    _, pm_module = pm_layers[pm_index]
    captured = {}

    def _hook(module, inp, out):
        captured["feat"] = out.detach()

    handle = pm_module.register_forward_hook(_hook)
    return handle, captured


def get_downstream_modules(encoder, pm_index: int = 2):
    """Return (features_sequential, start_index) for Stage 4+ modules."""
    pm_layers = _find_pm_layers(encoder)
    pm_feat_idx = pm_layers[pm_index][0]
    features = encoder.model.wrapped_model.features
    downstream_start = pm_feat_idx + 1
    return features, downstream_start


def downstream_forward(features_seq, start_idx: int, x: torch.Tensor):
    """Forward through downstream modules (Stage 4 + LayerNorm)."""
    for i in range(start_idx, len(features_seq)):
        x = features_seq[i](x)
    return x


# ─────────────────────────────────────────────────────────────────────────
# Image loading
# ─────────────────────────────────────────────────────────────────────────

def load_sorted_image_paths(patches_dir: Path) -> list[Path]:
    """Return sorted list of image paths in the patches directory."""
    exts = {".png", ".jpg", ".jpeg"}
    return sorted(p for p in patches_dir.iterdir() if p.suffix.lower() in exts)


def load_and_normalize_image(
    path: Path, device: torch.device,
) -> torch.Tensor:
    """Load a PNG, convert to [1, 3, 224, 224] ImageNet-normalised tensor."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3] in [0,1]
    t = torch.from_numpy(arr).permute(2, 0, 1)     # [3, H, W]
    t = (t - _IMAGENET_MEAN) / _IMAGENET_STD        # ImageNet normalise
    return t.unsqueeze(0).to(device)                 # [1, 3, H, W]


# ─────────────────────────────────────────────────────────────────────────
# Match loading
# ─────────────────────────────────────────────────────────────────────────

def load_matches(
    csv_path: Path, label_a: str, label_b: str, top_n: int,
) -> list[dict]:
    """Read reciprocal_best_matches.csv, auto-detecting column names."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        col_a = next(
            (c for c in cols if label_a in c.lower() and "feature" in c.lower()),
            None,
        )
        col_b = next(
            (c for c in cols if label_b in c.lower() and "feature" in c.lower()),
            None,
        )
        col_sim = next(
            (c for c in cols if "cosine" in c.lower() or "similarity" in c.lower()),
            None,
        )
        if not (col_a and col_b and col_sim):
            raise ValueError(
                f"Cannot find columns for '{label_a}'/'{label_b}' in {cols}."
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


# ─────────────────────────────────────────────────────────────────────────
# Core patching logic
# ─────────────────────────────────────────────────────────────────────────

def compute_patching_scores(
    encoder,
    features_seq,
    downstream_start: int,
    sae_a,
    sae_b,
    feature_a: int,
    feature_b: int,
    image_paths: list[Path],
    top_img_indices: list[int],
    device: torch.device,
    n_images: int = 5,
    pm_index: int = 2,
) -> list[dict]:
    """Measure downstream change when swapping decoder directions.

    For each top image:
      1. Forward through encoder → capture raw pm_index=2 activation
      2. Normalise, encode with SAE_A → get F_A magnitude per spatial position
      3. Compute delta = magnitude × (W_dec_B[F_B] − W_dec_A[F_A])
      4. Map delta to raw space, add to original
      5. Forward original + patched through Stage 4
      6. cosine_similarity(y_original, y_patched)

    Returns list of per-image result dicts.
    """
    hook_handle, captured = register_hook(encoder, pm_index)

    # Decoder directions
    W_dec_A = sae_a.W_dec[feature_a].detach().to(device)  # [d_model]
    W_dec_B = sae_b.W_dec[feature_b].detach().to(device)  # [d_model]
    decoder_delta = (W_dec_B - W_dec_A)                    # [d_model]

    scores = []

    with torch.inference_mode():
        for rank, img_idx in enumerate(top_img_indices[:n_images]):
            if img_idx >= len(image_paths):
                scores.append({
                    "img_idx": int(img_idx), "rank": rank,
                    "cos_sim": None, "note": "img_idx_out_of_range",
                })
                continue

            img_tensor = load_and_normalize_image(image_paths[img_idx], device)

            # Full encoder forward — hook captures pm_index=2 output
            encoder(img_tensor)
            x_raw = captured["feat"]  # (1, H, W, C) on device
            B, H, W, C = x_raw.shape

            # ── Original downstream output ────────────────────────────
            y_orig = downstream_forward(features_seq, downstream_start, x_raw)

            # ── Flatten and Anthropic-normalize ───────────────────────
            x_flat = x_raw.reshape(-1, C).float()     # (H*W, C)
            rms = x_flat.pow(2).mean(dim=-1, keepdim=True).sqrt() + EPS
            d_model = C
            x_norm = x_flat / rms * math.sqrt(d_model)  # (H*W, C)

            # ── SAE encode → get F_A magnitude per spatial position ───
            z = sae_a.encode(x_norm)       # (H*W, d_sae) sparse
            m = z[:, feature_a]            # (H*W,) magnitude

            n_firing = (m > 0).sum().item()
            if n_firing == 0:
                scores.append({
                    "img_idx": int(img_idx), "rank": rank,
                    "cos_sim": 1.0, "n_firing": 0,
                    "note": "feature_not_firing",
                })
                continue

            # ── Delta in NORMALISED space ─────────────────────────────
            delta_norm = m.unsqueeze(-1) * decoder_delta.unsqueeze(0)  # (H*W, C)

            # ── Convert delta to RAW space ────────────────────────────
            #   x_norm = x_raw / rms × √d  →  x_raw = x_norm × rms / √d
            #   delta_raw = delta_norm × rms / √d
            delta_raw = delta_norm * rms / math.sqrt(d_model)  # (H*W, C)

            # ── Patched activation ────────────────────────────────────
            x_patched_flat = x_flat + delta_raw
            x_patched = x_patched_flat.reshape(B, H, W, C)

            # ── Patched downstream output ─────────────────────────────
            y_patch = downstream_forward(
                features_seq, downstream_start, x_patched,
            )

            # ── Cosine similarity ─────────────────────────────────────
            cos = F.cosine_similarity(
                y_orig.reshape(1, -1).float(),
                y_patch.reshape(1, -1).float(),
            ).item()

            scores.append({
                "img_idx": int(img_idx),
                "rank": rank,
                "cos_sim": cos,
                "n_firing": n_firing,
                "mean_magnitude": float(m[m > 0].mean().item()),
            })

    hook_handle.remove()
    return scores


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Activation patching for cross-SAE feature interchangeability",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--checkpoint", required=True,
                    help="Path to encoder.pt (shared backbone)")
    ap.add_argument("--sae_a", required=True,
                    help="SAE checkpoint directory for dataset A")
    ap.add_argument("--sae_b", required=True,
                    help="SAE checkpoint directory for dataset B")
    ap.add_argument("--stats_a", required=True,
                    help="stats.pt for dataset A (used to find top images)")
    ap.add_argument("--patches_dir", required=True,
                    help="Patches directory for dataset A (224×224 PNGs)")
    ap.add_argument("--matches_csv", required=True,
                    help="reciprocal_best_matches.csv")
    ap.add_argument("--label_a", required=True,
                    help="Label for dataset A (e.g. 'colon')")
    ap.add_argument("--label_b", required=True,
                    help="Label for dataset B (e.g. 'breast')")
    ap.add_argument("--output", required=True,
                    help="Output JSON path")
    ap.add_argument("--top_n", type=int, default=20,
                    help="Number of top matched pairs to test")
    ap.add_argument("--n_images", type=int, default=5,
                    help="Top-activating images per pair")
    ap.add_argument("--pm_index", type=int, default=2,
                    help="PatchMerging index (2 = Stage 3→4 boundary)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device)

    # ── Load encoder ──────────────────────────────────────────────────
    encoder = load_encoder(args.checkpoint, args.device)
    features_seq, downstream_start = get_downstream_modules(encoder, args.pm_index)
    n_downstream = len(features_seq) - downstream_start
    print(f"[Downstream] {n_downstream} module(s) after pm_index={args.pm_index} "
          f"(features[{downstream_start}:{len(features_seq)}])")

    # ── Load SAEs ─────────────────────────────────────────────────────
    print(f"\n[SAE A] Loading from {args.sae_a}")
    sae_a = load_sae(args.sae_a, device=args.device)
    print(f"  d_sae={sae_a.d_sae}, d_model={sae_a.d_model}")

    print(f"[SAE B] Loading from {args.sae_b}")
    sae_b = load_sae(args.sae_b, device=args.device)
    print(f"  d_sae={sae_b.d_sae}, d_model={sae_b.d_model}")

    # ── Load stats for top-image selection ────────────────────────────
    stats = torch.load(args.stats_a, map_location="cpu")
    feature_max = stats["feature_max"]  # [n_images, d_sae]
    print(f"\n[Stats] feature_max shape: {feature_max.shape}")

    # ── Load image paths ──────────────────────────────────────────────
    patches_dir = Path(args.patches_dir)
    image_paths = load_sorted_image_paths(patches_dir)
    print(f"[Images] {len(image_paths)} images in {patches_dir}")

    # ── Load matched pairs ────────────────────────────────────────────
    matches = load_matches(
        Path(args.matches_csv), args.label_a, args.label_b, args.top_n,
    )
    print(f"\n[Matches] {len(matches)} pairs loaded "
          f"(top cosine: {matches[0]['cosine_sim']:.3f})")

    # ── Run patching ──────────────────────────────────────────────────
    all_results = []

    for match in tqdm(matches, desc="Activation patching"):
        fa, fb = match["feat_a"], match["feat_b"]

        # Find top-n images for F_A
        n_avail = min(feature_max.shape[0], len(image_paths))
        top_indices = torch.topk(
            feature_max[:n_avail, fa], min(args.n_images, n_avail)
        ).indices.tolist()

        scores = compute_patching_scores(
            encoder, features_seq, downstream_start,
            sae_a, sae_b, fa, fb,
            image_paths, top_indices,
            device,
            n_images=args.n_images,
            pm_index=args.pm_index,
        )

        valid_scores = [s["cos_sim"] for s in scores if s["cos_sim"] is not None]
        mean_cos = sum(valid_scores) / len(valid_scores) if valid_scores else None

        result = {
            "rank": match["rank"],
            "feat_a": fa,
            "feat_b": fb,
            "cosine_sim": match["cosine_sim"],
            "mean_patching_score": mean_cos,
            "per_image": scores,
        }
        all_results.append(result)

        tqdm.write(
            f"  F{fa}↔F{fb}  cos={match['cosine_sim']:.3f}  "
            f"patch_score={mean_cos:.6f}" if mean_cos else
            f"  F{fa}↔F{fb}  cos={match['cosine_sim']:.3f}  patch_score=N/A"
        )

    # ── Summary ───────────────────────────────────────────────────────
    valid_means = [
        r["mean_patching_score"]
        for r in all_results
        if r["mean_patching_score"] is not None
    ]

    summary = {
        "label_a": args.label_a,
        "label_b": args.label_b,
        "n_pairs": len(all_results),
        "n_images_per_pair": args.n_images,
        "pm_index": args.pm_index,
        "overall_mean_patching_score": (
            sum(valid_means) / len(valid_means) if valid_means else None
        ),
        "overall_min": min(valid_means) if valid_means else None,
        "overall_max": max(valid_means) if valid_means else None,
        "pairs": all_results,
    }

    # ── Save ──────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to {output_path}")
    print(f"Overall mean patching score: "
          f"{summary['overall_mean_patching_score']:.6f}")
    print(f"Range: [{summary['overall_min']:.6f}, {summary['overall_max']:.6f}]")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
