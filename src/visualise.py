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
from visualisation.loaders import load_sae, load_activations, load_images, load_images_multi
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
from visualisation.coactivation import (
    compute_coactivation_matrix,
    render_coactivation_heatmap,
    find_coactivating_pairs,
)
from visualisation.html_dashboard import save_dashboard

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
        # Allow mismatch when using multi-dir stem matching (some may be None)
        n_valid = sum(1 for p in image_paths if p is not None)
        if n_valid > 0 and n_valid != n_images:
            print(f"  ⚠ Image count ({n_valid} resolved) != n_images ({n_images}).")
            print(f"    Patch crops will be unavailable for unresolved images.")
        elif n_valid == 0:
            print(f"  ⚠ No images resolved — patch crop rendering will be skipped.")

    print(f"  ✓ Consistency checks passed")
    print()


# ──────────────────────────────────────────────────────────────────────────
# Common setup helper
# ──────────────────────────────────────────────────────────────────────────

def load_spatial_config(activations_path: str) -> tuple[int, int, int]:
    """
    Returns (n_patches, spatial_h, spatial_w).
    Reads from sidecar _metadata.json if present, else falls back to 196/14/14.
    """
    import json
    from pathlib import Path
    meta_path = Path(activations_path).parent / \
                (Path(activations_path).stem + "_metadata.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        n = meta["n_patches"]
        h = meta.get("spatial_h", int(n ** 0.5))
        w = meta.get("spatial_w", int(n ** 0.5))
        return n, h, w
    return 196, 14, 14

def _common_setup(args):
    """Shared boilerplate: build config, load SAE + activations + images."""
    device = getattr(args, "device", "cuda:0")

    # Determine images_dir for VisConfig (use first dir if multi-dir mode)
    images_dir_str = getattr(args, "images_dir", "") or ""
    images_dirs = getattr(args, "images_dirs", None)
    if not images_dir_str and images_dirs:
        # Use first entry's path for VisConfig (it only needs one for config)
        images_dir_str = images_dirs[0].rsplit(":", 1)[0]

    cfg = VisConfig.from_checkpoint(
        args.checkpoint_dir,
        activations_path=args.activations_path,
        images_dir=images_dir_str,
        device=device,
    )

    print(f"Loading SAE from {args.checkpoint_dir} …")
    sae = load_sae(args.checkpoint_dir, device=cfg.device)

    print(f"Loading activations from {args.activations_path} …")
    acts, meta = load_activations(args.activations_path, device="cpu")

    # Override spatial layout from sidecar metadata
    n_patches, h, w = load_spatial_config(args.activations_path)
    cfg.spatial_h = h
    cfg.spatial_w = w

    image_paths: list[Path] = []
    if images_dirs:
        # Multi-dataset mode: use image_stems from merged activations
        import json as _json
        image_stems_raw = meta.get("image_stems", "{}")
        if isinstance(image_stems_raw, str):
            image_stems_by_ds = _json.loads(image_stems_raw)
        else:
            image_stems_by_ds = image_stems_raw
        ds_names_raw = meta.get("dataset_names", "[]")
        if isinstance(ds_names_raw, str):
            dataset_names = _json.loads(ds_names_raw)
        else:
            dataset_names = ds_names_raw
        print(f"Loading images from {len(images_dirs)} directories …")
        image_paths = load_images_multi(images_dirs, image_stems_by_ds, dataset_names)
    elif cfg.images_dir:
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

    # Optionally load stats for dataset-colour node colouring
    stats = None
    img_dir = None
    if getattr(args, "stats_path", None):
        stats = load_stats(args.stats_path, device="cpu")
        print(f"  ✓ Stats loaded from {args.stats_path}")
    if getattr(args, "images_dir", None):
        img_dir = Path(args.images_dir)

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
        sae=sae,
        save_path=graph_path,
        stats=stats,
        images_dir=img_dir,
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
        stats=stats,
        images_dir=img_dir,
    )
    plt.close(fig2)
    print(f"  ✓ Graph + spatial panel saved to {combo_path}")


# ──────────────────────────────────────────────────────────────────────────
# Subcommand: label-views
# ──────────────────────────────────────────────────────────────────────────

def cmd_label_views(args):
    from visualisation.view_labeller import label_dataset_views
    label_dataset_views(Path(args.images_dir), save_path=Path(args.output))


# ──────────────────────────────────────────────────────────────────────────
# Subcommand: ablate
# ──────────────────────────────────────────────────────────────────────────

def cmd_ablate(args):
    from visualisation.ablation import run_patch_ablation, render_ablation_figure

    cfg, sae, acts, image_paths = _common_setup(args)

    n_patches, spatial_h, spatial_w = load_spatial_config(args.activations_path)

    result = run_patch_ablation(
        feature_idx=args.feature_idx,
        image_idx=args.image_idx,
        sae=sae,
        flat_activations=acts,
        n_patches=n_patches,
        spatial_h=spatial_h,
        spatial_w=spatial_w,
        mode=args.mode,
    )

    print(f"Baseline activation : {result['baseline_act']:.3f}")
    print(f"Most important patches: {result['most_important_patches']}")

    if image_paths:
        save_path = str(
            Path(args.output_dir)
            / f"ablation_F{args.feature_idx}_img{args.image_idx}_{args.mode}.png"
        )
        fig = render_ablation_figure(
            result, image_paths[args.image_idx],
            spatial_h=spatial_h, spatial_w=spatial_w,
            save_path=save_path,
        )
        plt.close(fig)
        print(f"  Saved to {save_path}")


# ──────────────────────────────────────────────────────────────────────────
# Subcommand: faithfulness
# ──────────────────────────────────────────────────────────────────────────

def cmd_faithfulness(args) -> None:
    import json
    from pathlib import Path
    from safetensors.torch import load_file as st_load
    from visualisation.ablation import (
        compute_feature_faithfulness_profile,
        render_faithfulness_figure,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load SAE
    import json as _json
    sae = load_sae(args.checkpoint_dir, device="cpu")
    sae.eval()

    # Load cfg.json for spatial dims
    _cfg_path = Path(args.checkpoint_dir) / "cfg.json"
    _cfg_dict = _json.loads(_cfg_path.read_text()) if _cfg_path.exists() else {}

    # Load activations
    acts_tensors = st_load(args.activations_path)
    key = next(iter(acts_tensors))
    flat_acts = acts_tensors[key]  # [N*n_patches, d_model]

    # Load stats
    stats = torch.load(args.stats_path, map_location="cpu")
    if isinstance(stats, dict) and all(isinstance(v, torch.Tensor) for v in stats.values()):
        stats = {k: v.numpy() for k, v in stats.items()}

    # Load spatial config
    spatial_h = int(_cfg_dict.get("spatial_h", 14))
    spatial_w = int(_cfg_dict.get("spatial_w", 14))
    n_patches = spatial_h * spatial_w

    # Load classifier weights (or use SAE proxy)
    if args.use_sae_proxy:
        # Self-faithfulness: treat feature activations as logits
        # Use W_dec as a [d_sae, d_model] proxy classifier
        classifier_weights = sae.W_dec.detach()   # [d_sae, d_model]
        classifier_bias = None
        print("Using SAE decoder as proxy classifier (--use_sae_proxy)")
    else:
        weights_dict = torch.load(args.classifier_weights_path, map_location="cpu")
        # Try common key names
        cw = None
        for key in ("weight", "W", "classifier.weight", "head.weight"):
            if key in weights_dict:
                cw = weights_dict[key]
                break
        if cw is None:
            # Fallback: use first tensor
            cw = next(iter(weights_dict.values()))
        classifier_weights = cw.float()

        cb = None
        for key in ("bias", "b", "classifier.bias", "head.bias"):
            if key in weights_dict:
                cb = weights_dict[key].float()
                break
        classifier_bias = cb

    # Compute profile
    profile = compute_feature_faithfulness_profile(
        feature_idx=args.feature_idx,
        stats=stats,
        sae=sae,
        flat_activations=flat_acts,
        classifier_weights=classifier_weights,
        classifier_bias=classifier_bias,
        n_patches=n_patches,
        spatial_h=spatial_h,
        spatial_w=spatial_w,
        top_k_patches=args.top_k_patches,
        n_images=args.n_images,
    )

    # Print summary
    print(f"F{args.feature_idx} Faithfulness Profile")
    print(f"  Evaluated on: {profile['n_evaluated']} images")
    print(f"  Mean faithfulness: {profile.get('mean_faithfulness', 0):.4f}")
    print(f"  Std faithfulness:  {profile.get('std_faithfulness', 0):.4f}")
    print(f"  Mean cosine sim:   {profile.get('mean_cosine_sim', 0):.4f}")
    print(f"  Mean NMSE:         {profile.get('mean_nmse', 0):.4f}")
    print(f"  Verdict: {profile.get('overall_interpretation', 'N/A')}")

    # Render figure
    fig_path = out_dir / f"faithfulness_F{args.feature_idx}.png"
    render_faithfulness_figure(profile, save_path=str(fig_path))
    print(f"  Figure saved: {fig_path}")

    # Save JSON
    json_path = out_dir / f"faithfulness_F{args.feature_idx}.json"
    json_path.write_text(json.dumps(profile, indent=2))
    print(f"  JSON saved:   {json_path}")


# ──────────────────────────────────────────────────────────────────────────
# Subcommand: coactivation
# ──────────────────────────────────────────────────────────────────────────

def cmd_coactivation(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Loading stats from {args.stats_path} …")
    try:
        stats = load_stats(args.stats_path, device="cpu")
    except Exception as e:
        print(f"Error loading stats file: {e}")
        sys.exit(1)
        
    top_feats = get_most_active_features(stats, n=args.top_features).tolist()
    
    print("Computing co-activation matrix …")
    coact_matrix = compute_coactivation_matrix(stats, top_k=50, feature_indices=top_feats)
    
    save_path = str(Path(args.output_dir) / "coactivation_heatmap.png")
    fig = render_coactivation_heatmap(coact_matrix, top_feats, save_path=save_path)
    plt.close(fig)
    print(f"  ✓ Saved heatmap to {save_path}")
    
    pairs = find_coactivating_pairs(coact_matrix, top_feats, threshold=0.0)
    print("\nTop 20 co-activating pairs:")
    for i in range(min(20, len(pairs))):
        f1, f2, score = pairs[i]
        print(f"  F{f1} & F{f2}: {score:.3f}")


# ──────────────────────────────────────────────────────────────────────────
# Subcommand: dashboard
# ──────────────────────────────────────────────────────────────────────────

def cmd_dashboard(args):
    cfg, sae, acts, image_paths = _common_setup(args)
    
    print(f"Loading stats from {args.stats_path} …")
    try:
        stats = load_stats(args.stats_path, device="cpu")
    except Exception as e:
        print(f"Error loading stats file: {e}")
        sys.exit(1)
        
    top_feats = get_most_active_features(stats, n=args.top_n).tolist()

    view_labels = None
    if getattr(args, "view_labels_path", None):
        import json
        view_labels = json.loads(Path(args.view_labels_path).read_text())

    out_path = Path(args.output)
    save_dashboard(
        feature_indices=top_feats,
        stats=stats,
        sae=sae,
        images_dir=Path(args.images_dir),
        output_path=out_path,
        spatial_h=cfg.spatial_h,
        spatial_w=cfg.spatial_w,
        context_patches=2,
        top_k_patches=9,
        n_coactivation_rows=10,
        coactivation_top_k=50,
        verbose=True,
        activations=acts,
        view_labels=view_labels,
    )

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

    # ── 4. HTML dashboard: top-50 features ───────────────────────────────
    dashboard_path = base / "dashboard.html"
    print(f"\nRendering HTML dashboard — top-50 features …")
    top50 = get_most_active_features(stats, n=50).tolist()
    save_dashboard(
        feature_indices=top50,
        stats=stats,
        sae=sae,
        images_dir=Path(cfg.images_dir) if cfg.images_dir else Path("."),
        output_path=dashboard_path,
        spatial_h=cfg.spatial_h,
        spatial_w=cfg.spatial_w,
        context_patches=2,
        top_k_patches=9,
        n_coactivation_rows=10,
        coactivation_top_k=50,
        verbose=True,
        activations=acts,
        image_paths=image_paths if image_paths else None,
    )

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
    sp_l3.add_argument("--stats_path", type=str, default=None,
                       help="Path to stats.pt (enables dataset-colour nodes).")
    sp_l3.add_argument("--output_dir", type=str, default="data/vis_outputs/attribution_graphs")
    sp_l3.set_defaults(func=cmd_layer3)

    # ── label-views ───────────────────────────────────────────────────
    sp_lv = sub.add_parser("label-views", help="Label brain MRI images by view orientation (axial/sagittal/coronal).")
    sp_lv.add_argument("--images_dir", type=str, required=True)
    sp_lv.add_argument("--output", type=str, required=True, help="Path to save labels JSON.")
    sp_lv.set_defaults(func=cmd_label_views)

    # ── ablate ────────────────────────────────────────────────────────
    sp_abl = sub.add_parser("ablate", help="Patch deletion/insertion ablation test for a single feature.")
    _add_common(sp_abl)
    sp_abl.add_argument("--images_dir", type=str, required=True)
    sp_abl.add_argument("--feature_idx", type=int, required=True)
    sp_abl.add_argument("--image_idx",   type=int, required=True)
    sp_abl.add_argument("--mode", type=str, default="deletion", choices=["deletion", "insertion"])
    sp_abl.add_argument("--output_dir", type=str, default="data/vis_outputs/ablation")
    sp_abl.set_defaults(func=cmd_ablate)

    # ── faithfulness ──────────────────────────────────────────────────
    sp_faith = sub.add_parser("faithfulness",
        help="Compute mechanistic faithfulness scores for a feature.")
    sp_faith.add_argument("--feature_idx",  type=int, required=True)
    sp_faith.add_argument("--checkpoint_dir",      required=True)
    sp_faith.add_argument("--activations_path",    required=True)
    sp_faith.add_argument("--stats_path",          required=True)
    sp_faith.add_argument("--classifier_weights_path", default=None)
    sp_faith.add_argument("--use_sae_proxy", action="store_true", default=False,
        help="Use SAE decoder as proxy classifier (when no head available).")
    sp_faith.add_argument("--n_images",    type=int, default=10)
    sp_faith.add_argument("--top_k_patches", type=int, default=5)
    sp_faith.add_argument("--output_dir",  required=True)
    sp_faith.set_defaults(func=cmd_faithfulness)

    # ── all ───────────────────────────────────────────────────────────
    sp_all = sub.add_parser("all", help="Run collect -> layer1 -> layer3 pipeline.")
    _add_common(sp_all)
    sp_all.add_argument("--images_dir", type=str, default="",
                        help="Single image directory (legacy single-dataset mode).")
    sp_all.add_argument("--images_dirs", type=str, nargs="+", default=None,
                        help="Multiple image dirs in path:label format.")
    sp_all.add_argument("--output_dir", type=str, default="data/vis_outputs")
    sp_all.set_defaults(func=cmd_all)

    # ── coactivation ──────────────────────────────────────────────────
    sp_coact = sub.add_parser("coactivation", help="Render feature co-activation heatmap.")
    sp_coact.add_argument("--stats_path", type=str, required=True)
    sp_coact.add_argument("--top_features", type=int, default=50)
    sp_coact.add_argument("--output_dir", type=str, required=True)
    sp_coact.set_defaults(func=cmd_coactivation)

    # ── dashboard ─────────────────────────────────────────────────────
    sp_dash = sub.add_parser("dashboard", help="Export interactive HTML dashboard.")
    _add_common(sp_dash)
    sp_dash.add_argument("--images_dir", type=str, required=True)
    sp_dash.add_argument("--stats_path", type=str, required=True)
    sp_dash.add_argument("--top_n", type=int, default=50)
    sp_dash.add_argument("--output", type=str, required=True)
    sp_dash.add_argument("--view_labels_path", type=str, default=None)
    sp_dash.set_defaults(func=cmd_dashboard)

    return p


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
