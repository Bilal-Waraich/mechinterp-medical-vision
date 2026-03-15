"""Attribution-graph utilities for Phase 3 visualisation.

Computes pairwise feature interactions via the linear approximation:
  interaction(i→j) = W_dec[i] · W_enc[:, j]

and weights them by prompt-specific (image-specific) mean activations
to produce a directed attribution graph between active SAE features.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from sae_models import BatchTopKSAE  # noqa: E402
from visualisation.vis_config import VisConfig  # noqa: E402
from visualisation.spatial_maps import get_image_activations  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────
# 1. Active feature detection
# ──────────────────────────────────────────────────────────────────────────

def get_active_features(
    f: torch.Tensor,
    threshold: float = 0.0,
    min_active_features: int = 5,
    max_active_features: int = 80,
) -> torch.Tensor:
    """Return indices of features that fired on at least one patch.

    Uses a mean-based threshold: a feature must fire above 10% of the
    max activation seen on this image to count as active.  The raw
    *threshold* parameter is kept for backward compatibility but is
    effectively superseded by the 10%-of-max rule.

    If more than *max_active_features* pass the threshold, only the
    strongest (by mean activation) are kept.

    Parameters
    ----------
    f : Tensor [n_patches, d_sae]
    threshold : float
        Legacy threshold (still applied as a floor).
    min_active_features : int
        Minimum number of features to return (uses top-k if fewer pass threshold).
    max_active_features : int
        Maximum number of features to return.

    Returns
    -------
    Tensor [n_active]  (int64 indices, sorted)
    """
    # 10%-of-image-max adaptive threshold
    image_max = f.max().item()
    adaptive_threshold = max(threshold, 0.1 * image_max)

    active_mask = (f > adaptive_threshold).any(dim=0)  # [d_sae]
    active_indices = active_mask.nonzero(as_tuple=False).squeeze(-1)

    # Handle edge cases (scalar or empty)
    if active_indices.ndim == 0:
        active_indices = active_indices.unsqueeze(0)

    # If too few features pass the adaptive threshold, fall back to top-k
    if len(active_indices) < min_active_features:
        strengths = f.mean(dim=0)  # [d_sae]
        k = min(min_active_features, strengths.shape[0])
        active_indices = torch.topk(strengths, k=k).indices

    # If too many features, keep only the strongest
    if len(active_indices) > max_active_features:
        strengths = f[:, active_indices].mean(dim=0)
        topk = torch.topk(strengths, k=max_active_features)
        active_indices = active_indices[topk.indices]

    # Return sorted
    return active_indices.sort().values


# ──────────────────────────────────────────────────────────────────────────
# 2. Linear interaction matrix
# ──────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_interaction_matrix(
    sae: BatchTopKSAE,
    active_indices: torch.Tensor,
) -> torch.Tensor:
    """Compute pairwise decoder cosine similarity among active features.

    ``interaction[i, j] = cosine_similarity(W_dec[active_i], W_dec[active_j])``

    Using decoder-decoder similarity rather than W_dec·W_enc because encoder
    columns in a trained BatchTopK SAE grow in the negative direction to
    suppress competing features, which causes W_dec·W_enc to produce nearly
    all-negative off-diagonal entries.  Decoder cosine similarity directly
    measures whether two features' output directions are aligned (+) or
    opposing (−) in activation space, and has a naturally balanced sign
    distribution.

    Diagonal (self-similarity = 1.0) entries are zeroed out.

    Parameters
    ----------
    sae : BatchTopKSAE
    active_indices : Tensor [n_active]

    Returns
    -------
    Tensor [n_active, n_active]  on CPU
    """
    W_dec_active = sae.W_dec[active_indices]  # [n_active, d_model]

    # W_dec rows are kept unit-norm during training; normalise defensively
    normed = torch.nn.functional.normalize(W_dec_active, p=2, dim=-1)
    interaction = normed @ normed.T  # [n_active, n_active], values in [-1, 1]

    print(f"[DEBUG graph] W_dec={sae.W_dec.shape}, active_indices={active_indices.shape}")
    print(f"[DEBUG graph] interaction: min={interaction.min():.3f}, max={interaction.max():.3f}, "
          f"frac_positive={(interaction > 0).float().mean():.2%}")

    interaction.fill_diagonal_(0.0)
    return interaction.cpu()


# ──────────────────────────────────────────────────────────────────────────
# 3. Prompt-specific (image-specific) attribution
# ──────────────────────────────────────────────────────────────────────────

def compute_prompt_attribution(
    f: torch.Tensor,
    active_indices: torch.Tensor,
    interaction: torch.Tensor,
) -> torch.Tensor:
    """Weight the interaction matrix by per-feature mean activation.

    ``attribution[i, j] = mean_activation[i] * interaction[i, j]``

    Parameters
    ----------
    f : Tensor [n_patches, d_sae]
    active_indices : Tensor [n_active]
    interaction : Tensor [n_active, n_active]

    Returns
    -------
    Tensor [n_active, n_active]
    """
    source_strengths = f[:, active_indices].mean(dim=0)  # [n_active]
    assert (source_strengths >= 0).all(), \
        "source_strengths has negative values — check SAE forward pass"
    attribution = source_strengths.unsqueeze(1) * interaction.to(source_strengths.device)
    return attribution


# ──────────────────────────────────────────────────────────────────────────
# 4. Logit attribution
# ──────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_logit_attribution(
    sae: BatchTopKSAE,
    classifier_weights: torch.Tensor | None,
    active_indices: torch.Tensor,
    f: torch.Tensor,
) -> torch.Tensor | None:
    """Attribute each active feature's contribution to each output class.

    ``logit_attr[i, c] = mean_act[i] * (W_dec[active_i] · classifier_weights[c])``

    Parameters
    ----------
    sae : BatchTopKSAE
    classifier_weights : Tensor [n_classes, d_model] or None
        Linear classification head.  If *None*, returns *None*.
    active_indices : Tensor [n_active]
    f : Tensor [n_patches, d_sae]

    Returns
    -------
    Tensor [n_active, n_classes] or None
    """
    if classifier_weights is None:
        return None

    W_dec_active = sae.W_dec[active_indices]                    # [n_active, d_model]
    source_strengths = f[:, active_indices].mean(dim=0)         # [n_active]
    logit_attr = W_dec_active @ classifier_weights.T            # [n_active, n_classes]
    logit_attr = logit_attr * source_strengths.unsqueeze(1)
    return logit_attr.cpu()


# ──────────────────────────────────────────────────────────────────────────
# 5. Graph pruning
# ──────────────────────────────────────────────────────────────────────────

def prune_graph(
    attribution: torch.Tensor,
    top_n: int = 50,
) -> torch.Tensor:
    """Keep only the *top_n* edges by absolute value; zero the rest.

    Parameters
    ----------
    attribution : Tensor [n_active, n_active]
    top_n : int

    Returns
    -------
    Tensor [n_active, n_active]  (sparse — most entries zero)
    """
    flat = attribution.flatten()
    if flat.numel() <= top_n:
        return attribution.clone()

    abs_flat = flat.abs()
    topk_vals = torch.topk(abs_flat, top_n)
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[topk_vals.indices] = True
    pruned = torch.where(mask, flat, torch.zeros_like(flat))
    return pruned.reshape(attribution.shape)


# ──────────────────────────────────────────────────────────────────────────
# 6. End-to-end builder
# ──────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def build_graph_data(
    image_idx: int,
    activations: torch.Tensor,
    sae: BatchTopKSAE,
    config: VisConfig,
    classifier_weights: torch.Tensor | None = None,
) -> dict:
    """Build the full attribution graph for a single image.

    Parameters
    ----------
    image_idx : int
    activations : Tensor [N_eff, d_model]
    sae : BatchTopKSAE  (eval mode)
    config : VisConfig
    classifier_weights : Tensor [n_classes, d_model] or None

    Returns
    -------
    dict with keys:
        ``active_indices``    : Tensor [n_active]
        ``attribution``       : Tensor [n_active, n_active]  (pruned, signed)
        ``edge_attributions`` : list[float]  signed attribution per non-zero edge,
                                in row-major order matching non-zero entries
        ``logit_attribution`` : Tensor [n_active, n_classes] or None
        ``feature_strengths`` : Tensor [n_active]
        ``n_active``          : int
        ``image_idx``         : int
    """
    f = get_image_activations(image_idx, activations, sae, config)

    active_indices = get_active_features(f, config.activation_threshold)
    interaction = compute_interaction_matrix(sae, active_indices)
    attribution = compute_prompt_attribution(f, active_indices, interaction)
    logit_attr = compute_logit_attribution(
        sae, classifier_weights, active_indices, f
    )
    pruned = prune_graph(attribution, config.top_n_edges)

    # Collect signed edge attributions in row-major order
    attr_np = pruned.cpu().numpy()
    edge_attributions: list[float] = []
    n_active = len(active_indices)
    for i in range(n_active):
        for j in range(n_active):
            v = float(attr_np[i, j])
            if v != 0.0:
                edge_attributions.append(v)

    # Diagnostic: excitatory vs inhibitory edge ratio (printed once per call)
    n_pos = (pruned > 0).sum().item()
    n_neg = (pruned < 0).sum().item()
    total_edges = n_pos + n_neg
    if total_edges > 0:
        print(f"Attribution matrix: {n_pos} excitatory, {n_neg} inhibitory edges "
              f"({100*n_pos/(total_edges):.0f}% excitatory)")

    return {
        "active_indices": active_indices.cpu(),
        "attribution": pruned.cpu(),
        "edge_attributions": edge_attributions,
        "logit_attribution": logit_attr,
        "feature_strengths": f[:, active_indices].mean(0).cpu(),
        "n_active": n_active,
        "image_idx": image_idx,
    }


@torch.no_grad()
def compute_decoder_cosine_similarity(
    sae: BatchTopKSAE,
    active_indices: torch.Tensor,
) -> torch.Tensor:
    """Compute pairwise cosine similarity between active decoder atoms.
    
    Self-similarity (diagonal) is zeroed out.
    
    Parameters
    ----------
    sae : BatchTopKSAE
    active_indices : Tensor [n_active]
    
    Returns
    -------
    Tensor [n_active, n_active] on CPU
    """
    W_dec_active = sae.W_dec[active_indices]  # [n_active, d_model]
    
    # Cosine similarity using a gram matrix on normalized vectors
    # W_dec rows should already be unit norm, but let's be safe
    W_dec_normalized = torch.nn.functional.normalize(W_dec_active, p=2, dim=-1)
    cos_sim = W_dec_normalized @ W_dec_normalized.T  # [n_active, n_active]
    
    cos_sim.fill_diagonal_(0.0)
    return cos_sim.cpu()


# ──────────────────────────────────────────────────────────────────────────
# 7. Hub consistency
# ──────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_hub_consistency(
    image_indices: list[int],
    activations: torch.Tensor,
    sae: BatchTopKSAE,
    config: VisConfig,
    top_n_hubs: int = 10,
) -> dict:
    """Measure how consistently the same SAE features are "hubs" across images.

    For each image in *image_indices*, encodes its patches and selects the
    ``top_n_hubs`` features by mean activation (feature_strengths).  Then
    computes pairwise Jaccard similarity of these hub sets across images.

    Parameters
    ----------
    image_indices : list[int]
        Image indices to analyse.
    activations : Tensor [N_eff, d_model]
        Full flat activation tensor.
    sae : BatchTopKSAE
        Trained SAE in eval mode.
    config : VisConfig
        Visualisation settings (needs ``n_patches``, ``activation_threshold``).
    top_n_hubs : int
        Number of top features to select per image.

    Returns
    -------
    dict
        ``'hub_overlap_matrix'``: numpy array [n_images, n_images], pairwise
            Jaccard similarity of top-hub sets.
        ``'consistent_hubs'``: list[int] features that are in the top_n_hubs
            set for at least 80% of images.
        ``'per_image_hubs'``: list[list[int]] top-hub feature indices per image.
    """
    n_images_req = len(image_indices)
    per_image_hubs: list[list[int]] = []

    for img_idx in image_indices:
        f = get_image_activations(img_idx, activations, sae, config)
        # Feature strengths: mean activation across patches
        strengths = f.mean(dim=0)  # [d_sae]
        n_pick = min(top_n_hubs, strengths.shape[0])
        top_feats = torch.topk(strengths, n_pick).indices.cpu().tolist()
        per_image_hubs.append(top_feats)

    # Pairwise Jaccard similarity
    import numpy as np  # noqa: PLC0415
    overlap_matrix = np.zeros((n_images_req, n_images_req), dtype=np.float32)
    for i in range(n_images_req):
        for j in range(n_images_req):
            set_i = set(per_image_hubs[i])
            set_j = set(per_image_hubs[j])
            union = len(set_i | set_j)
            inter = len(set_i & set_j)
            overlap_matrix[i, j] = inter / union if union > 0 else 0.0

    # Consistent hubs: in top_n_hubs for >= 80% of images
    threshold = 0.8 * n_images_req
    from collections import Counter  # noqa: PLC0415
    hub_counts: Counter = Counter()
    for hubs in per_image_hubs:
        hub_counts.update(hubs)
    consistent_hubs = [
        feat for feat, cnt in hub_counts.items() if cnt >= threshold
    ]
    consistent_hubs.sort()

    return {
        "hub_overlap_matrix": overlap_matrix,
        "consistent_hubs": consistent_hubs,
        "per_image_hubs": per_image_hubs,
    }


# ── Smoke tests ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  graph_utils.py — smoke tests")
    print("=" * 60)

    d_model, d_sae, k = 16, 32, 4
    spatial_h, spatial_w = 2, 2
    n_patches = spatial_h * spatial_w
    n_images = 3
    N_eff = n_images * n_patches

    sae = BatchTopKSAE(d_model=d_model, d_sae=d_sae, k=k)
    sae.eval()
    cfg = VisConfig(
        spatial_h=spatial_h, spatial_w=spatial_w,
        top_n_edges=5, activation_threshold=0.0, device="cpu",
    )
    acts = torch.randn(N_eff, d_model)

    # Encode one image to get f
    f = get_image_activations(0, acts, sae, cfg)

    # 1. get_active_features
    active = get_active_features(f, threshold=0.0)
    n_active = active.numel()
    assert n_active <= d_sae
    print(f"[OK] get_active_features: {n_active} / {d_sae} active")

    # 2. compute_interaction_matrix
    inter = compute_interaction_matrix(sae, active)
    assert inter.shape == (n_active, n_active)
    assert inter.diagonal().abs().sum() == 0.0, "diagonal should be zero"
    print(f"[OK] compute_interaction_matrix: shape={inter.shape}, diag=0")

    # 3. compute_prompt_attribution
    attr = compute_prompt_attribution(f, active, inter)
    assert attr.shape == (n_active, n_active)
    print(f"[OK] compute_prompt_attribution: shape={attr.shape}")

    # 4. compute_logit_attribution — with classifier
    clf_w = torch.randn(5, d_model)  # 5 classes
    logit = compute_logit_attribution(sae, clf_w, active, f)
    assert logit is not None
    assert logit.shape == (n_active, 5)
    print(f"[OK] compute_logit_attribution: shape={logit.shape}")

    # 4b. compute_logit_attribution — None
    assert compute_logit_attribution(sae, None, active, f) is None
    print("[OK] compute_logit_attribution: None when no classifier")

    # 5. prune_graph
    pruned = prune_graph(attr, top_n=5)
    assert pruned.shape == attr.shape
    n_nonzero = (pruned != 0).sum().item()
    assert n_nonzero <= 5, f"expected ≤5 edges, got {n_nonzero}"
    print(f"[OK] prune_graph: {n_nonzero} edges kept (top_n=5)")

    # 6. build_graph_data
    gd = build_graph_data(0, acts, sae, cfg, classifier_weights=clf_w)
    assert gd["n_active"] == n_active
    assert gd["attribution"].shape == (n_active, n_active)
    assert gd["logit_attribution"].shape == (n_active, 5)
    assert gd["feature_strengths"].shape == (n_active,)
    assert gd["image_idx"] == 0
    assert "edge_attributions" in gd
    assert isinstance(gd["edge_attributions"], list)
    # Verify signed values are preserved
    non_zero = (gd["attribution"] != 0).sum().item()
    assert len(gd["edge_attributions"]) == non_zero, (
        f"edge_attributions length {len(gd['edge_attributions'])} != "
        f"non-zero attribution entries {non_zero}"
    )
    print(f"[OK] build_graph_data: n_active={gd['n_active']}, keys={list(gd.keys())}, "
          f"edge_attributions={len(gd['edge_attributions'])}")

    # 6b. build_graph_data without classifier
    gd2 = build_graph_data(0, acts, sae, cfg, classifier_weights=None)
    assert gd2["logit_attribution"] is None
    print("[OK] build_graph_data: logit_attribution=None without classifier")

    # 7. compute_hub_consistency
    image_indices = list(range(n_images))
    hub_result = compute_hub_consistency(
        image_indices, acts, sae, cfg, top_n_hubs=3,
    )
    assert "hub_overlap_matrix" in hub_result
    assert "consistent_hubs" in hub_result
    assert "per_image_hubs" in hub_result

    mat = hub_result["hub_overlap_matrix"]
    assert mat.shape == (n_images, n_images)
    # Diagonal must be 1.0 (each image has Jaccard=1 with itself)
    import numpy as _np
    assert _np.allclose(mat.diagonal(), 1.0), f"diagonal not 1: {mat.diagonal()}"

    assert len(hub_result["per_image_hubs"]) == n_images
    for hubs in hub_result["per_image_hubs"]:
        assert len(hubs) <= 3

    print(f"[OK] compute_hub_consistency: overlap_matrix={mat.shape}, "
          f"consistent_hubs={hub_result['consistent_hubs']}, "
          f"per_image_hubs={hub_result['per_image_hubs']}")

    print("\nAll graph_utils smoke tests passed ✓")
