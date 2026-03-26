"""Cross-SAE cosine similarity analysis for universality detection.

Loads two trained SAEs (e.g. breast and colon, layer 2) and computes
pairwise cosine similarity between their decoder weight vectors (W_dec rows).
Each row of W_dec is a learned feature direction in activation space —
high cosine similarity means two features from different SAEs point in
the same direction, suggesting a universal concept.

Outputs
-------
1. CSV of all features per SAE (index, norms, encoder-decoder alignment)
2. CSV similarity matrix (all pairs above threshold)
3. Reciprocal best matching CSV (gold-standard matched pairs)
4. Per-SAE monosemanticity scores CSV
5. Per-SAE Jaccard uniqueness report CSV
6. UMAP projection of decoder weights (both SAEs overlaid)
7. Side-by-side visualisation of matched feature pairs (semantic dicts)

New analysis additions adapted from:
- sae-for-vlm (Pach et al. 2025): Monosemanticity Score, Jaccard uniqueness, UMAP
- semantic-lens: clarity_score concept (monosemanticity via pairwise cosine similarity)
- SAELens: encoder-decoder alignment metric
- sae_vis: cross-SAE activation correlation pattern
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
import json
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

def compute_enc_dec_alignment(sae: BatchTopKSAE) -> torch.Tensor:
    """Compute per-feature cosine similarity between encoder and decoder directions.

    Adapted from SAELens encoder_decoder_cosine_sim.
    A well-trained feature should have enc-dec alignment close to 1.0.
    Low alignment (< 0.5) indicates the feature encodes one direction
    but reconstructs another — a sign of poor training convergence.

    Returns
    -------
    Tensor [d_sae] — cosine similarity per feature.
    """
    # W_enc shape: [d_model, d_sae], W_dec shape: [d_sae, d_model]
    W_enc = F.normalize(sae.W_enc.data.float(), dim=0)  # normalise each column
    W_dec = F.normalize(sae.W_dec.data.float(), dim=1)  # normalise each row
    # Per-feature dot product: sum over d_model dimension
    return (W_enc.T * W_dec).sum(dim=1)  # [d_sae]


def export_feature_csv(
    sae: BatchTopKSAE,
    output_path: Path,
    label: str,
) -> torch.Tensor:
    """Write a CSV listing all features in an SAE.

    Columns: feature_idx, dataset, decoder_norm, encoder_norm, enc_dec_alignment

    Returns the enc_dec_alignment tensor for downstream filtering.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    W_dec_norms = sae.W_dec.data.float().norm(dim=1).cpu().numpy()
    W_enc_norms = sae.W_enc.data.float().norm(dim=0).cpu().numpy()  # [d_sae]
    enc_dec_align = compute_enc_dec_alignment(sae).cpu()

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_idx", "dataset", "decoder_norm", "encoder_norm", "enc_dec_alignment"])
        for i in range(sae.d_sae):
            writer.writerow([
                i, label,
                f"{W_dec_norms[i]:.6f}",
                f"{W_enc_norms[i]:.6f}",
                f"{enc_dec_align[i].item():.6f}",
            ])

    n_low = (enc_dec_align < 0.5).sum().item()
    print(f"  Wrote {sae.d_sae} features to {output_path}")
    print(f"    Enc-dec alignment: mean={enc_dec_align.mean():.4f}, "
          f"{n_low} features below 0.5")
    return enc_dec_align


# ──────────────────────────────────────────────────────────────────────────
# 3. Similarity matrix CSV export & Reciprocal best matching
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


@torch.no_grad()
def compute_reciprocal_best_matches(
    sim_matrix: torch.Tensor,
    output_path: Path,
    label_a: str = "colon",
    label_b: str = "breast",
    min_sim: float = 0.0,
) -> list[tuple[int, int, float]]:
    """Find reciprocal best matches between two SAEs.

    Inspired by orthology detection in comparative genomics:
    Feature A's best match is B, AND Feature B's best match is A.
    This eliminates false positives from one-directional thresholding.

    Parameters
    ----------
    sim_matrix : Tensor [d_sae_a, d_sae_b]
    output_path : where to save the CSV
    min_sim : minimum similarity to report (default: 0.0 = all reciprocals)

    Returns
    -------
    list of (idx_a, idx_b, similarity) sorted descending.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Best match for each A feature in B
    best_b_for_a = sim_matrix.argmax(dim=1)  # [d_sae_a]
    # Best match for each B feature in A
    best_a_for_b = sim_matrix.argmax(dim=0)  # [d_sae_b]

    reciprocal_pairs = []
    for a_idx in range(sim_matrix.shape[0]):
        b_idx = best_b_for_a[a_idx].item()
        # Check if B's best match points back to A
        if best_a_for_b[b_idx].item() == a_idx:
            sim_val = sim_matrix[a_idx, b_idx].item()
            if sim_val >= min_sim:
                reciprocal_pairs.append((a_idx, b_idx, sim_val))

    # Sort descending by similarity
    reciprocal_pairs.sort(key=lambda x: x[2], reverse=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            f"{label_a}_feature_idx",
            f"{label_b}_feature_idx",
            "cosine_similarity",
            "match_type",
        ])
        for a, b, s in reciprocal_pairs:
            writer.writerow([a, b, f"{s:.6f}", "reciprocal_best"])

    print(f"  Found {len(reciprocal_pairs)} reciprocal best matches "
          f"(min_sim={min_sim:.2f}) → {output_path}")
    return reciprocal_pairs


# ──────────────────────────────────────────────────────────────────────────
# 4. Monosemanticity score
# ──────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def compute_monosemanticity(
    sae: BatchTopKSAE,
    output_path: Path,
    label: str,
    top_k: int = 16,
) -> torch.Tensor:
    """Compute per-feature monosemanticity score using decoder weight geometry.

    Adapted from sae-for-vlm metric.py (Pach et al. 2025) and
    semantic-lens clarity_score (Dreyer et al.).

    Without a VLM, we use decoder weight rows as feature "embeddings".
    For each feature i, we compute its cosine similarity with every other
    feature, weighted by their decoder norm (proxy for importance).
    A monosemantic feature should have a few very high matches
    (semantically related) and mostly low matches.

    Here we use a simpler, more robust variant: for each decoder row,
    compute its mean cosine similarity with ALL other decoder rows.
    Low mean = unique/monosemantic direction. High mean = generic direction.
    We invert: score = 1 - mean_cosine_sim, so high = more monosemantic.

    Returns
    -------
    Tensor [d_sae] — monosemanticity score per feature (0–1, higher = better).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    W_dec = F.normalize(sae.W_dec.data.float(), dim=1)  # [d_sae, d_model]
    # Full pairwise cosine similarity within this SAE
    sim = W_dec @ W_dec.T  # [d_sae, d_sae]

    # Zero self-similarity diagonal
    sim.fill_diagonal_(0.0)
    # Mean cosine similarity with all other features
    mean_sim = sim.sum(dim=1) / (sim.shape[0] - 1)
    # Invert: high monosemanticity = low mean similarity to others
    mono_score = 1.0 - mean_sim.clamp(min=0.0)

    mono_cpu = mono_score.cpu()
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_idx", "dataset", "monosemanticity"])
        for i in range(sae.d_sae):
            writer.writerow([i, label, f"{mono_cpu[i].item():.6f}"])

    print(f"  Monosemanticity ({label}): mean={mono_cpu.mean():.4f}, "
          f"std={mono_cpu.std():.4f}, "
          f"{(mono_cpu > 0.7).sum().item()}/{sae.d_sae} above 0.7")
    return mono_score


# ──────────────────────────────────────────────────────────────────────────
# 5. Jaccard uniqueness (redundancy detection)
# ──────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def compute_jaccard_uniqueness(
    sae: BatchTopKSAE,
    activations: torch.Tensor,
    config: VisConfig,
    output_path: Path,
    label: str,
    top_k: int = 16,
    sample_pairs: int = 100_000,
) -> dict:
    """Compute Jaccard Index-based redundancy for features within one SAE.

    Adapted from sae-for-vlm uniqueness.py (Pach et al. 2025).
    For each pair of features, measure overlap in their top-K activating images.
    If two features share most of their top images, they are behaviourally redundant.

    Parameters
    ----------
    sae : the SAE to analyse
    activations : raw activations [N_total, d_model]
    config : VisConfig for spatial resolution
    top_k : number of top activating images per feature
    sample_pairs : max feature pairs to sample (full pairwise is d_sae^2)

    Returns
    -------
    dict with summary statistics.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = sae.W_enc.device
    n_patches = config.n_patches
    N_eff = activations.shape[0]
    n_images = N_eff // n_patches

    print(f"  Computing top-{top_k} activating images for {sae.d_sae} features...")
    # Compute per-image feature max for all features
    # Do this in batches to avoid OOM
    batch_imgs = max(1, 2048 // n_patches)
    feature_max = torch.zeros(n_images, sae.d_sae)

    for img_start in tqdm(range(0, n_images, batch_imgs), desc=f"Encoding {label}", leave=False):
        img_end = min(img_start + batch_imgs, n_images)
        chunk = activations[img_start * n_patches : img_end * n_patches].to(device)
        codes = sae.encode(chunk).cpu()  # [B*n_patches, d_sae]
        codes = codes.reshape(img_end - img_start, n_patches, -1)
        feature_max[img_start:img_end] = codes.max(dim=1).values  # [B, d_sae]

    # Get top-K image indices per feature
    k_actual = min(top_k, n_images)
    top_indices = torch.topk(feature_max, k_actual, dim=0).indices  # [k, d_sae]
    top_sets = [set(top_indices[:, i].tolist()) for i in range(sae.d_sae)]

    # Sample random pairs and compute Jaccard
    n_features = sae.d_sae
    total_pairs = n_features * (n_features - 1) // 2
    rng = np.random.default_rng(42)

    if total_pairs <= sample_pairs:
        # Compute all pairs
        from itertools import combinations
        pairs = list(combinations(range(n_features), 2))
    else:
        # Random sample
        pairs = []
        for _ in range(sample_pairs):
            i, j = rng.integers(0, n_features, size=2)
            while i == j:
                j = rng.integers(0, n_features)
            pairs.append((min(i, j), max(i, j)))
        pairs = list(set(pairs))  # deduplicate

    jaccard_scores = []
    for i, j in tqdm(pairs, desc=f"Jaccard {label}", leave=False):
        intersection = len(top_sets[i] & top_sets[j])
        union = len(top_sets[i] | top_sets[j])
        jaccard = intersection / union if union > 0 else 0.0
        jaccard_scores.append(jaccard)

    jaccard_arr = np.array(jaccard_scores)

    # Compute report
    report = {
        "dataset": label,
        "n_features": n_features,
        "n_pairs_sampled": len(pairs),
        "mean_jaccard": float(jaccard_arr.mean()),
        "std_jaccard": float(jaccard_arr.std()),
        "max_jaccard": float(jaccard_arr.max()),
        "pct_above_0.1": float((jaccard_arr > 0.1).mean() * 100),
        "pct_above_0.3": float((jaccard_arr > 0.3).mean() * 100),
        "pct_above_0.5": float((jaccard_arr > 0.5).mean() * 100),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"  Jaccard uniqueness ({label}): mean={report['mean_jaccard']:.4f}, "
          f"max={report['max_jaccard']:.4f}, "
          f"{report['pct_above_0.3']:.1f}% of pairs above 0.3")
    return report


# ──────────────────────────────────────────────────────────────────────────
# 6. UMAP visualisation of decoder weights
# ──────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def render_umap(
    sae_a: BatchTopKSAE,
    sae_b: BatchTopKSAE,
    label_a: str,
    label_b: str,
    save_path: Path,
    mono_scores_a: torch.Tensor | None = None,
    mono_scores_b: torch.Tensor | None = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> None:
    """Project decoder weights from both SAEs to 2D via UMAP.

    Adapted from sae-for-vlm dictionary_learning/interp.py feature_umap().
    Overlays both SAEs on the same plot, coloured by dataset.
    Optionally sizes points by monosemanticity score.

    Parameters
    ----------
    sae_a, sae_b : the two SAEs to compare
    save_path : output .png path
    mono_scores_a, mono_scores_b : optional monosemanticity scores for point sizing
    """
    try:
        import umap
    except ImportError:
        print("  ⚠ UMAP not installed (pip install umap-learn). Skipping UMAP plot.")
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)

    W_a = F.normalize(sae_a.W_dec.data.float(), dim=1).cpu().numpy()  # [d_sae_a, d_model]
    W_b = F.normalize(sae_b.W_dec.data.float(), dim=1).cpu().numpy()  # [d_sae_b, d_model]

    # Combine and project
    combined = np.vstack([W_a, W_b])
    labels = ([label_a] * W_a.shape[0]) + ([label_b] * W_b.shape[0])

    print(f"  Fitting UMAP on {combined.shape[0]} decoder directions...")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        n_components=2,
        random_state=42,
    )
    embedding = reducer.fit_transform(combined)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    n_a = W_a.shape[0]

    # Size by monosemanticity if available
    size_a = 2.0
    size_b = 2.0
    if mono_scores_a is not None:
        size_a = (mono_scores_a.cpu().numpy() * 10 + 1).clip(1, 15)
    if mono_scores_b is not None:
        size_b = (mono_scores_b.cpu().numpy() * 10 + 1).clip(1, 15)

    ax.scatter(
        embedding[:n_a, 0], embedding[:n_a, 1],
        s=size_a, c="#2196F3", alpha=0.4, label=label_a, rasterized=True,
    )
    ax.scatter(
        embedding[n_a:, 0], embedding[n_a:, 1],
        s=size_b, c="#F44336", alpha=0.4, label=label_b, rasterized=True,
    )

    ax.legend(fontsize=12, markerscale=5)
    ax.set_title(f"UMAP of Decoder Weights — {label_a} vs {label_b}", fontsize=14)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  UMAP saved to {save_path}")


# ──────────────────────────────────────────────────────────────────────────
# 7. Side-by-side semantic dict visualisation
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
# 8. Main
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

    # New analysis options
    parser.add_argument("--skip_jaccard", action="store_true",
                        help="Skip Jaccard uniqueness computation (slow for large SAEs)")
    parser.add_argument("--skip_umap", action="store_true",
                        help="Skip UMAP visualisation (requires umap-learn)")
    parser.add_argument("--skip_vis", action="store_true",
                        help="Skip rendering matched pair images")

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

    # ── [NEW] Export feature CSVs with enc-dec alignment ───────────────
    print("\n" + "="*60)
    print("STEP 1: Feature CSVs (with encoder-decoder alignment)")
    print("="*60)
    enc_dec_a = export_feature_csv(sae_a, out / f"features_{args.label_a}.csv", args.label_a)
    enc_dec_b = export_feature_csv(sae_b, out / f"features_{args.label_b}.csv", args.label_b)

    # ── [NEW] Monosemanticity scores ───────────────────────────────────
    print("\n" + "="*60)
    print("STEP 2: Monosemanticity scores")
    print("="*60)
    mono_a = compute_monosemanticity(
        sae_a, out / f"monosemanticity_{args.label_a}.csv", args.label_a,
    )
    mono_b = compute_monosemanticity(
        sae_b, out / f"monosemanticity_{args.label_b}.csv", args.label_b,
    )

    # ── Compute cosine similarity ──────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 3: Cross-SAE cosine similarity matrix")
    print("="*60)
    sim_matrix = compute_decoder_cosine_similarity(sae_a, sae_b)
    print(f"  Shape: {sim_matrix.shape}")
    print(f"  Max similarity: {sim_matrix.max().item():.4f}")
    print(f"  Pairs >= {args.threshold}: {(sim_matrix >= args.threshold).sum().item()}")

    # ── Export threshold-based similarity CSV ──────────────────────────
    print("\nExporting threshold-based similarity CSV...")
    matched_pairs = export_similarity_csv(
        sim_matrix, out / "similarity_matrix.csv",
        threshold=args.threshold,
        label_a=args.label_a, label_b=args.label_b,
    )

    # ── [NEW] Reciprocal best matching ─────────────────────────────────
    print("\n" + "="*60)
    print("STEP 4: Reciprocal best matching")
    print("="*60)
    reciprocal_pairs = compute_reciprocal_best_matches(
        sim_matrix, out / "reciprocal_best_matches.csv",
        label_a=args.label_a, label_b=args.label_b,
        min_sim=0.3,
    )
    # Also compute with higher threshold for thesis highlight
    strong_reciprocals = [p for p in reciprocal_pairs if p[2] >= args.threshold]
    print(f"    Of which {len(strong_reciprocals)} have cosine >= {args.threshold}")

    # ── Load activations for downstream analysis ──────────────────────
    print(f"\nLoading activations A from {args.activations_a}")
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

    # ── [NEW] Jaccard uniqueness ───────────────────────────────────────
    if not args.skip_jaccard:
        print("\n" + "="*60)
        print("STEP 5: Jaccard uniqueness (redundancy detection)")
        print("="*60)
        compute_jaccard_uniqueness(
            sae_a, acts_a, config_a,
            out / f"jaccard_uniqueness_{args.label_a}.json", args.label_a,
        )
        compute_jaccard_uniqueness(
            sae_b, acts_b, config_b,
            out / f"jaccard_uniqueness_{args.label_b}.json", args.label_b,
        )
    else:
        print("\n  Skipping Jaccard uniqueness (--skip_jaccard)")

    # ── [NEW] UMAP visualisation ──────────────────────────────────────
    if not args.skip_umap:
        print("\n" + "="*60)
        print("STEP 6: UMAP projection of decoder weights")
        print("="*60)
        render_umap(
            sae_a, sae_b,
            label_a=args.label_a, label_b=args.label_b,
            save_path=out / f"umap_{args.label_a}_vs_{args.label_b}.png",
            mono_scores_a=mono_a,
            mono_scores_b=mono_b,
        )
    else:
        print("\n  Skipping UMAP (--skip_umap)")

    # ── Visualise matched pairs ────────────────────────────────────────
    if not args.skip_vis:
        # Use reciprocal best matches for visualisation (higher quality)
        vis_pairs = reciprocal_pairs if reciprocal_pairs else matched_pairs
        n_vis = len(vis_pairs) if args.top_k_vis == 0 else min(args.top_k_vis, len(vis_pairs))
        print(f"\n" + "="*60)
        print(f"STEP 7: Visualising top {n_vis} matched pairs")
        print("="*60)

        # Load image paths
        from visualisation.loaders import load_images
        img_paths_a = load_images(args.images_a)
        img_paths_b = load_images(args.images_b)
        print(f"  Images A: {len(img_paths_a)}, Images B: {len(img_paths_b)}")

        vis_dir = out / "matched_pairs"
        vis_dir.mkdir(parents=True, exist_ok=True)

        for rank, (idx_a, idx_b, sim) in enumerate(tqdm(vis_pairs[:n_vis], desc="Rendering pairs")):
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
    else:
        print("\n  Skipping visualisation (--skip_vis)")

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n" + "="*60)
    print("DONE — All outputs:")
    print("="*60)
    print(f"  📁 {out}/")
    print(f"  ├── features_{args.label_a}.csv  (with enc-dec alignment)")
    print(f"  ├── features_{args.label_b}.csv  (with enc-dec alignment)")
    print(f"  ├── monosemanticity_{args.label_a}.csv")
    print(f"  ├── monosemanticity_{args.label_b}.csv")
    print(f"  ├── similarity_matrix.csv  ({len(matched_pairs)} pairs >= {args.threshold})")
    print(f"  ├── reciprocal_best_matches.csv  ({len(reciprocal_pairs)} reciprocal pairs)")
    if not args.skip_jaccard:
        print(f"  ├── jaccard_uniqueness_{args.label_a}.json")
        print(f"  ├── jaccard_uniqueness_{args.label_b}.json")
    if not args.skip_umap:
        print(f"  ├── umap_{args.label_a}_vs_{args.label_b}.png")
    if not args.skip_vis:
        print(f"  └── matched_pairs/  ({n_vis} images)")


if __name__ == "__main__":
    main()
