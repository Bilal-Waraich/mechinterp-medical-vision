"""Feature-statistics collector for Phase 3 visualisation.

Encodes the flat harvested activations through a trained SAE, reshapes
to [n_images, n_patches, d_sae], and computes per-feature / per-image
summary statistics used by all downstream renderers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure ``src/`` is importable regardless of working directory
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from sae_models import BatchTopKSAE  # noqa: E402
from visualisation.vis_config import VisConfig  # noqa: E402
from visualisation.loaders import load_sae, load_activations  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────
# Core statistics
# ──────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_feature_stats(
    activations: torch.Tensor,
    sae: BatchTopKSAE,
    config: VisConfig,
    batch_size: int = 4096,
) -> dict:
    """Encode every activation row and collect per-image feature statistics.

    Parameters
    ----------
    activations : Tensor [N_eff, d_model]
        Flat harvested activations (``N_eff = n_images * n_patches``).
    sae : BatchTopKSAE
        Trained SAE in **eval** mode.
    config : VisConfig
        Must supply ``spatial_h``, ``spatial_w`` so we know ``n_patches``.
    batch_size : int
        Number of rows to encode at once (controls GPU memory usage).

    Returns
    -------
    dict with keys:
        ``feature_max``    : Tensor [n_images, d_sae]
        ``feature_argmax`` : Tensor [n_images, d_sae]  (patch index 0..n_patches-1)
        ``feature_mean``   : Tensor [n_images, d_sae]
        ``global_max``     : Tensor [d_sae]
        ``n_images``       : int
    """
    n_patches = config.n_patches
    N_eff = activations.shape[0]
    n_images = N_eff // n_patches
    assert N_eff == n_images * n_patches, (
        f"N_eff={N_eff} is not divisible by n_patches={n_patches}"
    )

    d_sae = sae.d_sae

    # Pre-allocate output tensors on CPU — avoids materialising N_eff×d_sae
    feature_max    = torch.zeros(n_images, d_sae)
    feature_argmax = torch.zeros(n_images, d_sae, dtype=torch.long)
    feature_mean   = torch.zeros(n_images, d_sae)

    # Process in image-aligned batches so we never hold more than
    # images_per_batch × n_patches × d_sae floats in memory at once.
    images_per_batch = max(1, batch_size // n_patches)

    for img_start in tqdm(range(0, n_images, images_per_batch), desc="Encoding"):
        img_end  = min(img_start + images_per_batch, n_images)
        tok_start = img_start * n_patches
        tok_end   = img_end   * n_patches

        chunk = activations[tok_start:tok_end].to(sae.W_enc.device)  # [B*n_patches, d_model]
        codes = sae.encode(chunk).cpu()                            # [B*n_patches, d_sae]
        codes = codes.reshape(img_end - img_start, n_patches, d_sae)

        img_max, img_argmax = codes.max(dim=1)           # [B, d_sae]
        img_mean            = codes.mean(dim=1)          # [B, d_sae]

        feature_max   [img_start:img_end] = img_max
        feature_argmax[img_start:img_end] = img_argmax
        feature_mean  [img_start:img_end] = img_mean

    global_max = feature_max.max(dim=0).values           # [d_sae]

    return {
        "feature_max": feature_max,
        "feature_argmax": feature_argmax,
        "feature_mean": feature_mean,
        "global_max": global_max,
        "n_images": n_images,
    }


# ──────────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────────

def save_stats(stats: dict, path: str) -> None:
    """Persist stats dict via :func:`torch.save`.  Creates parent dirs."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, str(out))


def load_stats(path: str, device: str = "cpu") -> dict:
    """Load a stats dict and move all tensors to *device*."""
    stats: dict = torch.load(path, map_location=device, weights_only=False)
    for k, v in stats.items():
        if isinstance(v, torch.Tensor):
            stats[k] = v.to(device)
    return stats


# ──────────────────────────────────────────────────────────────────────────
# Query helpers
# ──────────────────────────────────────────────────────────────────────────

def get_top_images_per_feature(stats: dict, k: int = 9) -> torch.Tensor:
    """Return top-*k* image indices per feature, ranked by max activation.

    Parameters
    ----------
    stats : dict
        Output of :func:`compute_feature_stats`.
    k : int
        Number of images per feature.

    Returns
    -------
    Tensor [k, d_sae]
        Each column holds the *k* image indices with the highest
        ``feature_max`` for that feature.
    """
    feature_max = stats["feature_max"]  # [n_images, d_sae]
    k = min(k, feature_max.shape[0])
    return torch.topk(feature_max, k, dim=0).indices  # [k, d_sae]


def get_dead_features(
    stats: dict,
    threshold: float = 0.0,
) -> torch.Tensor:
    """Return indices of features whose ``global_max <= threshold``.

    Returns
    -------
    Tensor [n_dead]
        Sorted feature indices.
    """
    global_max = stats["global_max"]  # [d_sae]
    dead_mask = global_max <= threshold
    return torch.where(dead_mask)[0]


def get_most_active_features(
    stats: dict,
    n: int = 20,
) -> torch.Tensor:
    """Return the *n* most-active feature indices (dead features excluded).

    Activity is measured as the mean of ``feature_max`` across images.

    Returns
    -------
    Tensor [n]
        Feature indices, most-active first.
    """
    feature_max = stats["feature_max"]  # [n_images, d_sae]
    global_max = stats["global_max"]    # [d_sae]

    mean_max = feature_max.mean(dim=0)  # [d_sae]

    # Mask out dead features (global_max == 0)
    alive_mask = global_max > 0.0
    mean_max[~alive_mask] = -1.0  # ensure dead features sort last

    n = min(n, int(alive_mask.sum().item()))
    return torch.topk(mean_max, n).indices


# ──────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute per-feature statistics from a trained SAE checkpoint."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to the SAE checkpoint directory (must contain cfg.json + sae_weights.safetensors).",
    )
    parser.add_argument(
        "--activations_path",
        type=str,
        required=True,
        help="Path to the .safetensors activations file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/vis_outputs/feature_stats.pt",
        help="Path for the output .pt stats file.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4096,
        help="Encoding batch size (default 4096).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run on.",
    )

    args = parser.parse_args()

    # Build config from checkpoint
    cfg = VisConfig.from_checkpoint(
        args.checkpoint_dir,
        activations_path=args.activations_path,
        device=args.device,
    )

    # Load artefacts
    print(f"Loading SAE from {args.checkpoint_dir} …")
    sae = load_sae(args.checkpoint_dir, device=cfg.device)

    print(f"Loading activations from {args.activations_path} …")
    acts, meta = load_activations(args.activations_path, device=cfg.device)

    # Override spatial layout from activation metadata if present
    if "spatial_h" in meta:
        cfg.spatial_h = int(meta["spatial_h"])
    if "spatial_w" in meta:
        cfg.spatial_w = int(meta["spatial_w"])

    print(f"  shape = {acts.shape},  n_patches = {cfg.n_patches}")
    print(f"  n_images = {acts.shape[0] // cfg.n_patches}")

    # Compute
    stats = compute_feature_stats(acts, sae, cfg, batch_size=args.batch_size)

    # Quick summary
    n_dead = get_dead_features(stats).numel()
    top_feats = get_most_active_features(stats, n=10)
    print(f"\n  n_images       = {stats['n_images']}")
    print(f"  d_sae          = {stats['global_max'].shape[0]}")
    print(f"  dead features  = {n_dead}")
    print(f"  top-10 active  = {top_feats.tolist()}")

    # Save
    save_stats(stats, args.output)
    print(f"\nStats saved to {args.output}")
