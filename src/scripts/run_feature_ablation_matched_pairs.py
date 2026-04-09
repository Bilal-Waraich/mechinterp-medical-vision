"""Feature-level ablation on reciprocal-best-match pairs across two SAEs.

For each matched semantic pair (A_feature, B_feature), zeros that feature in the
SAE latent space across all patches of the top-activating images, decodes back, and
measures the resulting reconstruction disruption.  The cross-dataset symmetry score
quantifies whether the two features carry equivalent functional weight in their
respective SAEs -- the key functional-equivalence claim for universality.

Symmetry score = 1 - |rel_A - rel_B| / max(rel_A, rel_B)
  - 1.0 = perfectly symmetric disruption
  - 0.0 = one feature disrupts reconstruction, the other does not

Usage
-----
    python -m src.scripts.run_feature_ablation_matched_pairs \\
        --checkpoint_a  /path/to/sae_a/step_0200000 \\
        --activations_a /path/to/activations_a.safetensors \\
        --label_a       colon \\
        --checkpoint_b  /path/to/sae_b/step_0200000 \\
        --activations_b /path/to/activations_b.safetensors \\
        --label_b       breast \\
        --matches_csv   /path/to/reciprocal_best_matches.csv \\
        --output_dir    /path/to/output \\
        --top_n         20 \\
        --n_images      5 \\
        --device        cuda:0
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from visualisation.loaders import load_sae, load_activations
from visualisation.ablation import compute_feature_ablation_profile

# ── Spatial config (layer-2: 7×7 patches) ────────────────────────────────
SPATIAL_H = 7
SPATIAL_W = 7
N_PATCHES = SPATIAL_H * SPATIAL_W  # 49


# ── Helpers ───────────────────────────────────────────────────────────────

def load_matches(path: Path, top_n: int, label_a: str, label_b: str) -> list[dict]:
    """Read top_n rows from reciprocal_best_matches.csv, auto-detecting column names."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        col_a = next((c for c in cols if label_a in c.lower() and "feature" in c.lower()), None)
        col_b = next((c for c in cols if label_b in c.lower() and "feature" in c.lower()), None)
        col_sim = next((c for c in cols if "cosine" in c.lower() or "similarity" in c.lower()), None)
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


def find_top_images(
    flat_activations: torch.Tensor,
    sae,
    feature_idx: int,
    n_images_total: int,
    n_top: int,
) -> list[int]:
    """Return indices of the n_top images with highest mean feature activation."""
    device = sae.W_enc.device
    mean_acts = torch.zeros(n_images_total)
    batch_imgs = max(1, 2048 // N_PATCHES)
    with torch.no_grad():
        for start_i in range(0, n_images_total, batch_imgs):
            end_i = min(start_i + batch_imgs, n_images_total)
            acts = flat_activations[start_i * N_PATCHES : end_i * N_PATCHES].to(device)
            encoded = sae.encode(acts)
            encoded = encoded.reshape(end_i - start_i, N_PATCHES, -1)
            mean_acts[start_i:end_i] = encoded[:, :, feature_idx].mean(dim=1).cpu()
    return torch.topk(mean_acts, k=min(n_top, n_images_total)).indices.tolist()


def symmetry_score(rel_a: float, rel_b: float) -> float:
    """1 - |rel_a - rel_b| / max(rel_a, rel_b).  Returns 1.0 if both are zero."""
    denom = max(rel_a, rel_b)
    if denom < 1e-10:
        return 1.0
    return float(1.0 - abs(rel_a - rel_b) / denom)


def render_summary_figure(
    results: list[dict],
    label_a: str,
    label_b: str,
    save_path: str,
) -> None:
    """3-panel figure summarising feature ablation across all matched pairs.

    Panel 1: relative disruption per feature, side-by-side bars for A and B.
    Panel 2: scatter of disruption_A vs disruption_B, coloured by symmetry score.
    Panel 3: distribution of symmetry scores.
    """
    feat_labels = [
        f"{label_a}_F{r['feat_a']}\n{label_b}_F{r['feat_b']}" for r in results
    ]
    rel_a = [r["mean_rel_disruption_a"] for r in results]
    rel_b = [r["mean_rel_disruption_b"] for r in results]
    sym = [r["symmetry_score"] for r in results]

    x = np.arange(len(results))
    width = 0.35

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: side-by-side disruption bars
    ax1.bar(x - width / 2, rel_a, width, label=label_a, color="steelblue", alpha=0.85)
    ax1.bar(x + width / 2, rel_b, width, label=label_b, color="coral", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(feat_labels, rotation=45, ha="right", fontsize=7)
    ax1.set_ylabel("Mean relative disruption (ΔL2 / baseline L2)")
    ax1.set_title(f"Feature Ablation Disruption\n{label_a} vs {label_b}")
    ax1.legend(fontsize=8)

    # Panel 2: scatter disruption A vs B coloured by symmetry
    sc = ax2.scatter(rel_a, rel_b, c=sym, cmap="RdYlGn", vmin=0, vmax=1,
                     s=80, edgecolors="k", linewidths=0.5)
    plt.colorbar(sc, ax=ax2, label="Symmetry score")
    max_val = max(max(rel_a, default=0), max(rel_b, default=0)) * 1.1 or 0.1
    ax2.plot([0, max_val], [0, max_val], "k--", alpha=0.3, label="y=x (perfect symmetry)")
    ax2.set_xlabel(f"Relative disruption — {label_a}")
    ax2.set_ylabel(f"Relative disruption — {label_b}")
    ax2.set_title("Cross-dataset disruption symmetry")
    ax2.legend(fontsize=8)
    for i, r in enumerate(results):
        ax2.annotate(
            f"F{r['feat_a']}", (rel_a[i], rel_b[i]),
            fontsize=6, xytext=(3, 3), textcoords="offset points",
        )

    # Panel 3: symmetry score distribution
    ax3.hist(sym, bins=10, range=(0, 1), color="mediumseagreen", edgecolor="white")
    ax3.axvline(np.mean(sym), color="red", linestyle="--",
                label=f"mean={np.mean(sym):.3f}")
    ax3.set_xlabel("Symmetry score")
    ax3.set_ylabel("Count")
    ax3.set_title("Symmetry score distribution")
    ax3.legend(fontsize=8)

    plt.suptitle(
        f"Feature Ablation — {label_a} × {label_b}  (n={len(results)} pairs)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Feature-level ablation on matched SAE pairs (functional equivalence test)."
    )
    parser.add_argument("--checkpoint_a", required=True, help="SAE checkpoint dir for dataset A")
    parser.add_argument("--activations_a", required=True, help="Activations .safetensors for A")
    parser.add_argument("--label_a", required=True, help="Label for dataset A (e.g. 'colon')")
    parser.add_argument("--checkpoint_b", required=True, help="SAE checkpoint dir for dataset B")
    parser.add_argument("--activations_b", required=True, help="Activations .safetensors for B")
    parser.add_argument("--label_b", required=True, help="Label for dataset B (e.g. 'breast')")
    parser.add_argument("--matches_csv", required=True, help="Path to reciprocal_best_matches.csv")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--top_n", type=int, default=20,
                        help="Number of top matched pairs to evaluate")
    parser.add_argument("--n_images", type=int, default=5,
                        help="Top-activating images per feature to average over")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────
    print(f"Loading matched pairs from {args.matches_csv} …")
    matches = load_matches(Path(args.matches_csv), args.top_n, args.label_a, args.label_b)
    print(f"  {len(matches)} pairs (top cosine: {matches[0]['cosine_sim']:.3f})")

    print(f"\nLoading {args.label_a} SAE + activations …")
    sae_a = load_sae(args.checkpoint_a, device=args.device)
    acts_a, meta_a = load_activations(args.activations_a, device="cpu")
    n_images_a = meta_a.get("n_images", acts_a.shape[0] // N_PATCHES)
    print(f"  {args.label_a}: {n_images_a} images, acts {acts_a.shape}, "
          f"d_sae={sae_a.d_sae}")

    print(f"\nLoading {args.label_b} SAE + activations …")
    sae_b = load_sae(args.checkpoint_b, device=args.device)
    acts_b, meta_b = load_activations(args.activations_b, device="cpu")
    n_images_b = meta_b.get("n_images", acts_b.shape[0] // N_PATCHES)
    print(f"  {args.label_b}: {n_images_b} images, acts {acts_b.shape}, "
          f"d_sae={sae_b.d_sae}")

    # ── Run feature ablation ──────────────────────────────────────────────
    all_results = []
    print()
    for match in tqdm(matches, desc="Feature ablation pairs"):
        feat_a, feat_b = match["feat_a"], match["feat_b"]

        top_imgs_a = find_top_images(acts_a, sae_a, feat_a, n_images_a, args.n_images)
        top_imgs_b = find_top_images(acts_b, sae_b, feat_b, n_images_b, args.n_images)

        profile_a = compute_feature_ablation_profile(
            feature_idx=feat_a,
            sae=sae_a,
            flat_activations=acts_a,
            image_indices=top_imgs_a,
            n_patches=N_PATCHES,
        )
        profile_b = compute_feature_ablation_profile(
            feature_idx=feat_b,
            sae=sae_b,
            flat_activations=acts_b,
            image_indices=top_imgs_b,
            n_patches=N_PATCHES,
        )

        rel_a = profile_a["mean_relative_disruption"]
        rel_b = profile_b["mean_relative_disruption"]
        sym = symmetry_score(rel_a, rel_b)

        result = {
            "rank":                    match["rank"],
            "feat_a":                  feat_a,
            "feat_b":                  feat_b,
            "cosine_similarity":       match["cosine_sim"],
            "mean_rel_disruption_a":   rel_a,
            "std_rel_disruption_a":    profile_a["std_relative_disruption"],
            "mean_rel_disruption_b":   rel_b,
            "std_rel_disruption_b":    profile_b["std_relative_disruption"],
            "symmetry_score":          round(sym, 4),
            "mean_feature_act_a":      profile_a["mean_feature_activation"],
            "mean_feature_act_b":      profile_b["mean_feature_activation"],
            f"{args.label_a}_profile": profile_a,
            f"{args.label_b}_profile": profile_b,
        }
        all_results.append(result)

        tqdm.write(
            f"  rank {match['rank']:3d}: "
            f"{args.label_a}_F{feat_a} rel={rel_a:.4f}  "
            f"{args.label_b}_F{feat_b} rel={rel_b:.4f}  "
            f"sym={sym:.3f}  cos={match['cosine_sim']:.3f}"
        )

    # ── Save outputs ──────────────────────────────────────────────────────
    summary_path = output_dir / "feature_ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved full results → {summary_path}")

    # CSV for quick inspection
    csv_path = output_dir / "feature_ablation_summary.csv"
    csv_keys = [
        "rank", "feat_a", "feat_b", "cosine_similarity",
        "mean_rel_disruption_a", "std_rel_disruption_a",
        "mean_rel_disruption_b", "std_rel_disruption_b",
        "symmetry_score", "mean_feature_act_a", "mean_feature_act_b",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_keys)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r[k] for k in csv_keys})
    print(f"Saved CSV summary    → {csv_path}")

    # Summary figure
    fig_path = str(output_dir / "feature_ablation_figure.png")
    render_summary_figure(all_results, args.label_a, args.label_b, fig_path)
    print(f"Saved figure         → {fig_path}")

    # Print aggregate stats
    sym_scores = [r["symmetry_score"] for r in all_results]
    rel_a_all  = [r["mean_rel_disruption_a"] for r in all_results]
    rel_b_all  = [r["mean_rel_disruption_b"] for r in all_results]
    print(f"\n{'='*60}")
    print(f"  Pairs evaluated: {len(all_results)}")
    print(f"  Mean symmetry score:        {np.mean(sym_scores):.4f} ± {np.std(sym_scores):.4f}")
    print(f"  Mean rel disruption {args.label_a}: {np.mean(rel_a_all):.4f}")
    print(f"  Mean rel disruption {args.label_b}: {np.mean(rel_b_all):.4f}")
    high_sym = sum(1 for s in sym_scores if s >= 0.8)
    print(f"  Pairs with symmetry ≥ 0.8:  {high_sym} / {len(all_results)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
