"""Phase 3 Visualisation — top-level CLI entry-point.

Usage examples
--------------
  python src/visualise.py collect   --checkpoint_dir ... --activations_path ... --stats_output ...
  python src/visualise.py layer1    --checkpoint_dir ... --activations_path ... --images_dir ... --stats_path ... --output_dir ...
  python src/visualise.py layer2    --checkpoint_dir ... --activations_path ... --images_dir ... --image_idx 7 --feature_indices 0 14 203 --output_dir ...
  python src/visualise.py layer3    --checkpoint_dir ... --activations_path ... --images_dir ... --image_idx 7 --output_dir ...
  python src/visualise.py all       --checkpoint_dir ... --activations_path ... --images_dir ... --output_dir ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from visualisation.vis_config import VisConfig
from visualisation.loaders import load_sae, load_activations, load_images
from visualisation.collector import (
    compute_feature_stats,
    save_stats,
    load_stats,
    get_dead_features,
    get_most_active_features,
    get_top_images_per_feature,
)
from visualisation.semantic_dict import (
    render_semantic_dictionary,
    render_feature_dashboard,
)
from visualisation.spatial_maps import render_multi_feature_spatial
from visualisation.graph_utils import build_graph_data
from visualisation.attribution_graph import (
    render_attribution_graph,
    render_graph_with_spatial_panel,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────

def validate_setup(
    config: VisConfig,
    sae,
    activations: torch.Tensor,
    image_paths: list[Path],
) -> None:
    """Print a summary and raise on any mismatch."""
    d_model = sae.d_model
    d_sae = sae.d_sae
    k = sae.k
    N_eff, act_d = activations.shape
    n_patches = config.n_patches
    n_images = N_eff // n_patches

    print()
    print(f"  ✓ SAE loaded: d_model={d_model}, d_sae={d_sae}, k={k}")
    print(f"  ✓ Activations: shape={list(activations.shape)}, n_images={n_images}")
    print(f"  ✓ Images: {len(image_paths)} images found in {config.images_dir}")

    # Checks
    if act_d != d_model:
        raise ValueError(
            f"Activation d_model ({act_d}) != SAE d_model ({d_model}). "
            f"Check that the activations file matches the checkpoint."
        )
    if N_eff % n_patches != 0:
        raise ValueError(
            f"N_eff ({N_eff}) is not divisible by n_patches ({n_patches}). "
            f"Check spatial_h={config.spatial_h}, spatial_w={config.spatial_w}."
        )
    if image_paths and len(image_paths) != n_images:
        raise ValueError(
            f"Number of images ({len(image_paths)}) != n_images derived from "
            f"activations ({n_images}). Images must match harvest order."
        )

    print(f"  ✓ Consistency checks passed")
    print()


# ──────────────────────────────────────────────────────────────────────────
# Common setup helper
# ──────────────────────────────────────────────────────────────────────────

def _common_setup(args):
    """Shared boilerplate: build config, load SAE + activations + images."""
    device = getattr(args, "device", "cuda:0")

    cfg = VisConfig.from_checkpoint(
        args.checkpoint_dir,
        activations_path=args.activations_path,
        images_dir=getattr(args, "images_dir", ""),
        device=device,
    )

    print(f"Loading SAE from {args.checkpoint_dir} …")
    sae = load_sae(args.checkpoint_dir, device=cfg.device)

    print(f"Loading activations from {args.activations_path} …")
    acts, meta = load_activations(args.activations_path, device="cpu")

    # Override spatial layout from activation metadata
    if "spatial_h" in meta:
        cfg.spatial_h = int(meta["spatial_h"])
    if "spatial_w" in meta:
        cfg.spatial_w = int(meta["spatial_w"])

    image_paths: list[Path] = []
    if cfg.images_dir:
        image_paths = load_images(cfg.images_dir)

    return cfg, sae, acts, image_paths


# ──────────────────────────────────────────────────────────────────────────
# Subcommand: collect
# ──────────────────────────────────────────────────────────────────────────

def cmd_collect(args):
    cfg, sae, acts, image_paths = _common_setup(args)
    validate_setup(cfg, sae, acts, image_paths)

    print("Computing feature statistics …")
    stats = compute_feature_stats(acts, sae, cfg, batch_size=args.batch_size)

    n_dead = get_dead_features(stats).numel()
    print(f"  ✓ Dead features: {n_dead} / {stats['global_max'].shape[0]}")

    save_stats(stats, args.stats_output)
    print(f"  ✓ Stats saved to {args.stats_output}")


# ──────────────────────────────────────────────────────────────────────────
# Subcommand: layer1 (semantic dictionaries)
# ──────────────────────────────────────────────────────────────────────────

def cmd_layer1(args):
    cfg, sae, acts, image_paths = _common_setup(args)
    validate_setup(cfg, sae, acts, image_paths)

    print(f"Loading stats from {args.stats_path} …")
    stats = load_stats(args.stats_path, device="cpu")

    if args.feature_idx is not None:
        # Single feature
        feature_indices = [args.feature_idx]
    else:
        # Top-N most active
        feature_indices = get_most_active_features(stats, n=args.top_n).tolist()
        print(f"  Top-{args.top_n} most active features: {feature_indices}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    render_feature_dashboard(
        feature_indices=feature_indices,
        stats=stats,
        image_paths=image_paths,
        config=cfg,
        output_dir=args.output_dir,
    )


# ──────────────────────────────────────────────────────────────────────────
# Subcommand: layer2 (spatial maps)
# ──────────────────────────────────────────────────────────────────────────

def cmd_layer2(args):
    cfg, sae, acts, image_paths = _common_setup(args)
    validate_setup(cfg, sae, acts, image_paths)

    save_path = str(
        Path(args.output_dir)
        / f"spatial_image_{args.image_idx:05d}.png"
    )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Rendering spatial maps for image {args.image_idx}, "
          f"features {args.feature_indices} …")
    fig = render_multi_feature_spatial(
        image_idx=args.image_idx,
        feature_indices=args.feature_indices,
        activations=acts,
        sae=sae,
        image_paths=image_paths,
        config=cfg,
        save_path=save_path,
    )
    plt.close(fig)
    print(f"  ✓ Saved to {save_path}")


# ──────────────────────────────────────────────────────────────────────────
# Subcommand: layer3 (attribution graph)
# ──────────────────────────────────────────────────────────────────────────

def cmd_layer3(args):
    cfg, sae, acts, image_paths = _common_setup(args)
    validate_setup(cfg, sae, acts, image_paths)

    img_idx = args.image_idx
    print(f"Building attribution graph for image {img_idx} …")
    gd = build_graph_data(img_idx, acts, sae, cfg)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Graph only
    graph_path = str(
        Path(args.output_dir) / f"graph_image_{img_idx:05d}.png"
    )
    fig1 = render_attribution_graph(
        gd,
        target_class=args.target_class,
        save_path=graph_path,
    )
    plt.close(fig1)
    print(f"  ✓ Graph saved to {graph_path}")

    # Graph + spatial companion panel
    combo_path = str(
        Path(args.output_dir) / f"graph_spatial_image_{img_idx:05d}.png"
    )
    fig2 = render_graph_with_spatial_panel(
        image_idx=img_idx,
        activations=acts,
        sae=sae,
        image_paths=image_paths,
        config=cfg,
        graph_data=gd,
        save_path=combo_path,
    )
    plt.close(fig2)
    print(f"  ✓ Graph + spatial panel saved to {combo_path}")


# ──────────────────────────────────────────────────────────────────────────
# Subcommand: all
# ──────────────────────────────────────────────────────────────────────────

def cmd_all(args):
    cfg, sae, acts, image_paths = _common_setup(args)
    validate_setup(cfg, sae, acts, image_paths)

    base = Path(args.output_dir)
    base.mkdir(parents=True, exist_ok=True)

    # ── 1. Collect ────────────────────────────────────────────────────
    stats_path = base / "stats.pt"
    if stats_path.exists():
        print(f"Stats file already exists at {stats_path}, loading …")
        stats = load_stats(str(stats_path), device="cpu")
    else:
        print("Computing feature statistics …")
        stats = compute_feature_stats(acts, sae, cfg, batch_size=4096)
        save_stats(stats, str(stats_path))
        print(f"  ✓ Stats saved to {stats_path}")

    n_dead = get_dead_features(stats).numel()
    d_sae_total = stats["global_max"].shape[0]
    print(f"  ✓ Dead features: {n_dead} / {d_sae_total}")

    # ── 2. Layer 1: top 20 semantic dictionaries ─────────────────────
    top_features = get_most_active_features(stats, n=20).tolist()
    layer1_dir = str(base / "semantic_dicts")
    print(f"\nRendering Layer 1 — top-20 semantic dictionaries …")
    render_feature_dashboard(
        feature_indices=top_features,
        stats=stats,
        image_paths=image_paths,
        config=cfg,
        output_dir=layer1_dir,
    )

    # ── 3. Layer 3: first 5 images with companion panel ──────────────
    n_graph_images = min(5, stats["n_images"])
    layer3_dir = base / "attribution_graphs"
    layer3_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nRendering Layer 3 — attribution graphs for {n_graph_images} images …")

    for img_idx in range(n_graph_images):
        gd = build_graph_data(img_idx, acts, sae, cfg)
        combo_path = str(layer3_dir / f"graph_spatial_image_{img_idx:05d}.png")
        fig = render_graph_with_spatial_panel(
            image_idx=img_idx,
            activations=acts,
            sae=sae,
            image_paths=image_paths,
            config=cfg,
            graph_data=gd,
            save_path=combo_path,
        )
        plt.close(fig)
        print(f"  ✓ Image {img_idx}: {combo_path}")

    print(f"\n{'═' * 60}")
    print(f"  All visualisations saved to {base}")
    print(f"{'═' * 60}")


# ──────────────────────────────────────────────────────────────────────────
# CLI argument parser
# ──────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="visualise",
        description="Phase 3 Visualisation CLI for BatchTopK SAE.",
    )
    p.add_argument("--device", type=str, default="cuda:0", help="Torch device.")
    sub = p.add_subparsers(dest="command", required=True)

    # ── shared args ───────────────────────────────────────────────────
    def _add_common(sp):
        sp.add_argument("--checkpoint_dir", type=str, required=True)
        sp.add_argument("--activations_path", type=str, required=True)

    # ── collect ───────────────────────────────────────────────────────
    sp_collect = sub.add_parser("collect", help="Compute and save feature statistics.")
    _add_common(sp_collect)
    sp_collect.add_argument("--images_dir", type=str, default="")
    sp_collect.add_argument("--stats_output", type=str, default="data/vis_outputs/stats.pt")
    sp_collect.add_argument("--batch_size", type=int, default=4096)
    sp_collect.set_defaults(func=cmd_collect)

    # ── layer1 ────────────────────────────────────────────────────────
    sp_l1 = sub.add_parser("layer1", help="Render semantic dictionaries (patch crops).")
    _add_common(sp_l1)
    sp_l1.add_argument("--images_dir", type=str, required=True)
    sp_l1.add_argument("--stats_path", type=str, required=True)
    sp_l1.add_argument("--feature_idx", type=int, default=None,
                       help="Single feature to render.")
    sp_l1.add_argument("--top_n", type=int, default=20,
                       help="Number of top features (if --feature_idx not given).")
    sp_l1.add_argument("--output_dir", type=str, default="data/vis_outputs/semantic_dicts")
    sp_l1.set_defaults(func=cmd_layer1)

    # ── layer2 ────────────────────────────────────────────────────────
    sp_l2 = sub.add_parser("layer2", help="Render spatial activation maps.")
    _add_common(sp_l2)
    sp_l2.add_argument("--images_dir", type=str, required=True)
    sp_l2.add_argument("--image_idx", type=int, required=True)
    sp_l2.add_argument("--feature_indices", type=int, nargs="+", required=True,
                       help="Space-separated feature indices.")
    sp_l2.add_argument("--output_dir", type=str, default="data/vis_outputs/spatial_maps")
    sp_l2.set_defaults(func=cmd_layer2)

    # ── layer3 ────────────────────────────────────────────────────────
    sp_l3 = sub.add_parser("layer3", help="Render attribution graph.")
    _add_common(sp_l3)
    sp_l3.add_argument("--images_dir", type=str, required=True)
    sp_l3.add_argument("--image_idx", type=int, required=True)
    sp_l3.add_argument("--target_class", type=int, default=None)
    sp_l3.add_argument("--output_dir", type=str, default="data/vis_outputs/attribution_graphs")
    sp_l3.set_defaults(func=cmd_layer3)

    # ── all ───────────────────────────────────────────────────────────
    sp_all = sub.add_parser("all", help="Run collect → layer1 → layer3 pipeline.")
    _add_common(sp_all)
    sp_all.add_argument("--images_dir", type=str, required=True)
    sp_all.add_argument("--output_dir", type=str, default="data/vis_outputs")
    sp_all.set_defaults(func=cmd_all)

    return p


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
