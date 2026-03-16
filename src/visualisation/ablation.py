import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Literal

AblationMode = Literal["deletion", "insertion"]

def run_patch_ablation(
    feature_idx: int,
    image_idx: int,
    sae,
    flat_activations: torch.Tensor,
    n_patches: int = 196,
    spatial_h: int = 14,
    spatial_w: int = 14,
    mode: AblationMode = "deletion",
    baseline_value: float = 0.0,
) -> dict:
    acts = flat_activations[image_idx*n_patches : (image_idx+1)*n_patches].clone()
    with torch.no_grad():
        f_baseline = sae.encode(acts)
        baseline_act = f_baseline[:, feature_idx].mean().item()
        ablation_map = np.zeros(n_patches)
        if mode == "deletion":
            for patch_pos in range(n_patches):
                acts_ablated = acts.clone()
                acts_ablated[patch_pos] = baseline_value
                f_ablated = sae.encode(acts_ablated)
                ablation_map[patch_pos] = f_ablated[:, feature_idx].mean().item()
        elif mode == "insertion":
            for patch_pos in range(n_patches):
                acts_insert = torch.full_like(acts, baseline_value)
                acts_insert[patch_pos] = acts[patch_pos]
                f_inserted = sae.encode(acts_insert)
                ablation_map[patch_pos] = f_inserted[:, feature_idx].mean().item()
    delta_map = ablation_map - baseline_act
    importance = np.abs(delta_map)
    top5 = np.argsort(importance)[::-1][:5].tolist()
    return {
        "feature_idx":            feature_idx,
        "image_idx":              image_idx,
        "mode":                   mode,
        "baseline_act":           baseline_act,
        "ablation_map":           ablation_map.reshape(spatial_h, spatial_w),
        "delta_map":              delta_map.reshape(spatial_h, spatial_w),
        "most_important_patches": top5,
    }

def render_ablation_figure(
    result: dict,
    image_path: Path,
    spatial_h: int = 14,
    spatial_w: int = 14,
    save_path: str | None = None,
) -> plt.Figure:
    bg = np.array(Image.open(image_path).convert("RGB"))
    delta = result["delta_map"]

    # Upsample delta map to image size
    try:
        import cv2
        delta_up = cv2.resize(delta, (bg.shape[1], bg.shape[0]), interpolation=cv2.INTER_NEAREST)
    except ImportError:
        from PIL import Image as PILImage
        delta_pil = PILImage.fromarray(
            ((delta - delta.min()) / (max(delta.ptp(), 1e-8)) * 255).astype(np.uint8)
        ).resize((bg.shape[1], bg.shape[0]), PILImage.NEAREST)
        delta_up = np.array(delta_pil) / 255.0 * (delta.max() - delta.min()) + delta.min()

    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.3)

    # Panel 1: original with important patch boxes
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(bg)
    cell_h = bg.shape[0] / spatial_h
    cell_w = bg.shape[1] / spatial_w
    colours = ["red", "orange", "yellow", "green", "blue"]
    import matplotlib.patches as mpatches
    for rank, p in enumerate(result["most_important_patches"][:5]):
        row, col = p // spatial_w, p % spatial_w
        rect = mpatches.Rectangle(
            (col * cell_w, row * cell_h), cell_w, cell_h,
            linewidth=2, edgecolor=colours[rank], facecolor="none"
        )
        ax1.add_patch(rect)
        ax1.text(col*cell_w + 2, row*cell_h + 10, f"#{rank+1}",
                 color=colours[rank], fontsize=8, fontweight="bold")
    ax1.set_title(f"F{result['feature_idx']} — top-5 important patches\n"
                  f"(image {result['image_idx']}, {result['mode']} mode)")
    ax1.axis("off")

    # Panel 2: delta heatmap overlay
    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(bg)
    vmax = max(abs(delta_up.min()), abs(delta_up.max()), 1e-6)
    ax2.imshow(delta_up, cmap="RdBu_r", alpha=0.6, vmin=-vmax, vmax=vmax)
    ax2.set_title(f"Activation delta map\n(baseline={result['baseline_act']:.2f})")
    ax2.axis("off")

    # Panel 3: top-10 importance bar chart
    ax3 = fig.add_subplot(gs[2])
    importance = np.abs(result["delta_map"]).ravel()
    top10_idx = np.argsort(importance)[::-1][:10]
    top10_vals = importance[top10_idx]
    ax3.barh(range(10), top10_vals[::-1], color="steelblue")
    ax3.set_yticks(range(10))
    ax3.set_yticklabels([f"P{i}" for i in top10_idx[::-1]], fontsize=8)
    ax3.set_xlabel("|Delta activation|")
    ax3.set_title("Patch importance ranking")

    plt.suptitle(
        f"Patch {result['mode'].capitalize()} Test — F{result['feature_idx']}",
        fontsize=13, fontweight="bold"
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def compute_faithfulness_score(
    feature_idx: int,
    image_idx: int,
    sae,
    flat_activations,
    classifier_weights: torch.Tensor,
    classifier_bias=None,
    n_patches: int = 196,
    spatial_h: int = 14,
    spatial_w: int = 14,
    top_k_patches: int = 5,
    target_class=None,
) -> dict:
    """
    Measures how much ablating the patches most important to feature_idx
    changes the classification output.

    Steps:
    1. Get image activations: acts [n_patches, d_model]
    2. Compute baseline logits via mean pooling + linear head:
         pooled = acts.mean(dim=0)
         logits = pooled @ classifier_weights.T + bias
    3. Identify feature top_k_patches via deletion ablation
    4. Zero out those patches in acts -> acts_ablated
    5. Compute ablated logits the same way
    6. Compute delta_logit for the target class
    7. Compute cosine similarity between baseline and ablated pooled vectors
    8. Compute NMSE
    """
    if isinstance(flat_activations, np.ndarray):
        flat_acts_t = torch.from_numpy(flat_activations).float()
    else:
        flat_acts_t = flat_activations.float()

    acts = flat_acts_t[image_idx * n_patches : (image_idx + 1) * n_patches].clone()

    with torch.no_grad():
        # Step 1: baseline logits via mean pooling
        pooled_baseline = acts.mean(dim=0)              # [d_model]
        logits_baseline = pooled_baseline @ classifier_weights.T
        if classifier_bias is not None:
            logits_baseline = logits_baseline + classifier_bias

        # Step 2: determine target class
        if target_class is None:
            target_class = int(logits_baseline.argmax().item())

        # Step 3: find top_k_patches via deletion ablation
        f_baseline = sae.encode(acts)                   # [n_patches, d_sae]
        baseline_feat_act = f_baseline[:, feature_idx].mean().item()

        patch_importance = torch.zeros(n_patches)
        for p in range(n_patches):
            acts_tmp = acts.clone()
            acts_tmp[p] = 0.0
            f_tmp = sae.encode(acts_tmp)
            patch_importance[p] = baseline_feat_act - f_tmp[:, feature_idx].mean().item()
        top_patches = torch.topk(patch_importance, k=min(top_k_patches, n_patches)).indices.tolist()

        # Step 4: zero out top patches
        acts_ablated = acts.clone()
        for p in top_patches:
            acts_ablated[p] = 0.0

        # Step 5: ablated logits
        pooled_ablated = acts_ablated.mean(dim=0)
        logits_ablated = pooled_ablated @ classifier_weights.T
        if classifier_bias is not None:
            logits_ablated = logits_ablated + classifier_bias

        # Step 6: scores
        delta_logit = (logits_ablated[target_class] - logits_baseline[target_class]).item()
        baseline_mag = abs(logits_baseline[target_class].item()) + 1e-8
        faithfulness = abs(delta_logit) / baseline_mag

        # Cosine similarity of pooled representations
        cos_sim = torch.nn.functional.cosine_similarity(
            pooled_baseline.unsqueeze(0),
            pooled_ablated.unsqueeze(0)
        ).item()

        # Normalised MSE
        nmse = (
            (pooled_baseline - pooled_ablated).pow(2).sum() /
            (pooled_baseline.pow(2).sum() + 1e-8)
        ).item()

        # Human-readable interpretation
        if faithfulness > 0.3:
            interpretation = "HIGH (>0.3) — feature causally mediates prediction"
        elif faithfulness > 0.1:
            interpretation = "MEDIUM (0.1-0.3) — feature partially mediates prediction"
        else:
            interpretation = "LOW (<0.1) — feature may be epiphenomenal"

    return {
        "feature_idx":               feature_idx,
        "image_idx":                 image_idx,
        "target_class":              target_class,
        "baseline_logits":           logits_baseline.tolist(),
        "ablated_logits":            logits_ablated.tolist(),
        "delta_logit":               round(delta_logit, 4),
        "baseline_magnitude":        round(baseline_mag, 4),
        "faithfulness_score":        round(faithfulness, 4),
        "representation_cosine_sim": round(cos_sim, 4),
        "nmse":                      round(nmse, 4),
        "top_ablated_patches":       top_patches,
        "interpretation":            interpretation,
    }


def compute_feature_faithfulness_profile(
    feature_idx: int,
    stats: dict,
    sae,
    flat_activations,
    classifier_weights: torch.Tensor,
    classifier_bias=None,
    n_patches: int = 196,
    spatial_h: int = 14,
    spatial_w: int = 14,
    top_k_patches: int = 5,
    n_images: int = 10,
) -> dict:
    """
    Computes faithfulness scores across the top-n images for a feature.
    """
    fmax = stats["feature_max"][:, feature_idx]
    if isinstance(fmax, np.ndarray):
        fmax_t = torch.from_numpy(fmax).float()
    else:
        fmax_t = fmax.float()

    top_image_indices = torch.topk(
        fmax_t, k=min(n_images, fmax_t.shape[0])
    ).indices.tolist()

    results = []
    for img_idx in top_image_indices:
        try:
            r = compute_faithfulness_score(
                feature_idx=feature_idx,
                image_idx=img_idx,
                sae=sae,
                flat_activations=flat_activations,
                classifier_weights=classifier_weights,
                classifier_bias=classifier_bias,
                n_patches=n_patches,
                spatial_h=spatial_h,
                spatial_w=spatial_w,
                top_k_patches=top_k_patches,
            )
            results.append(r)
        except Exception:
            continue

    if not results:
        return {"feature_idx": feature_idx, "n_evaluated": 0,
                "mean_faithfulness": 0.0, "error": "no valid images"}

    scores  = [r["faithfulness_score"] for r in results]
    cosines = [r["representation_cosine_sim"] for r in results]
    nmses   = [r["nmse"] for r in results]

    mean_f = float(np.mean(scores))
    if mean_f > 0.3:
        overall = "HIGH — strong causal mediator"
    elif mean_f > 0.1:
        overall = "MEDIUM — partial mediator"
    else:
        overall = "LOW — likely epiphenomenal"

    return {
        "feature_idx":            feature_idx,
        "n_evaluated":            len(results),
        "mean_faithfulness":      round(mean_f, 4),
        "std_faithfulness":       round(float(np.std(scores)), 4),
        "mean_cosine_sim":        round(float(np.mean(cosines)), 4),
        "mean_nmse":              round(float(np.mean(nmses)), 4),
        "per_image":              results,
        "overall_interpretation": overall,
    }


def render_faithfulness_figure(
    profile: dict,
    save_path=None,
) -> "plt.Figure":
    """
    2-panel figure:
      Left:  bar chart of faithfulness score per image, coloured by
             HIGH(green)/MEDIUM(orange)/LOW(red)
      Right: scatter plot of faithfulness_score vs representation_cosine_sim
    """
    import matplotlib.pyplot as plt

    per_image = profile.get("per_image", [])
    if not per_image:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    scores  = [r["faithfulness_score"] for r in per_image]
    cosines = [r["representation_cosine_sim"] for r in per_image]
    img_ids = [r["image_idx"] for r in per_image]

    colors = []
    for s in scores:
        if s > 0.3:
            colors.append("#4CAF50")
        elif s > 0.1:
            colors.append("#FF9800")
        else:
            colors.append("#F44336")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: per-image faithfulness bars
    ax1.barh(range(len(scores)), scores, color=colors)
    ax1.set_yticks(range(len(scores)))
    ax1.set_yticklabels([f"img {i}" for i in img_ids], fontsize=8)
    ax1.axvline(0.3, color="green",  linestyle="--", alpha=0.5, label="HIGH threshold")
    ax1.axvline(0.1, color="orange", linestyle="--", alpha=0.5, label="MEDIUM threshold")
    ax1.set_xlabel("Faithfulness score")
    ax1.set_title(
        f"F{profile['feature_idx']} Faithfulness\n"
        f"mean={profile['mean_faithfulness']:.3f} "
        f"({profile['overall_interpretation'].split('--')[0].strip()})"
    )
    ax1.legend(fontsize=8)

    # Panel 2: faithfulness vs cosine similarity scatter
    ax2.scatter(cosines, scores, c=colors, s=80, edgecolors="k", linewidths=0.5)
    for c, s, iid in zip(cosines, scores, img_ids):
        ax2.annotate(f"img{iid}", (c, s), fontsize=7,
                     xytext=(3, 3), textcoords="offset points")
    ax2.set_xlabel("Representation cosine similarity\n(1.0 = no change, lower = more disrupted)")
    ax2.set_ylabel("Faithfulness score")
    ax2.set_title("Faithfulness vs Representation Disruption")
    ax2.axhline(0.3, color="green",  linestyle="--", alpha=0.3)
    ax2.axhline(0.1, color="orange", linestyle="--", alpha=0.3)
    ax2.text(0.02, 0.95, "High faith +\nHigh disruption\n(ideal)",
             transform=ax2.transAxes, fontsize=7, color="green", va="top")
    ax2.text(0.75, 0.05, "Low faith +\nLow disruption\n(epiphenomenal)",
             transform=ax2.transAxes, fontsize=7, color="red", va="bottom")

    plt.suptitle(
        f"Mechanistic Faithfulness Profile -- F{profile['feature_idx']}  "
        f"(n={profile['n_evaluated']} images)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
