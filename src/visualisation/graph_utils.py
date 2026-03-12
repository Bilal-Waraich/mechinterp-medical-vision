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
) -> torch.Tensor:
    """Return indices of features that fired on at least one patch.

    Parameters
    ----------
    f : Tensor [n_patches, d_sae]
    threshold : float

    Returns
    -------
    Tensor [n_active]  (int64 indices, sorted)
    """
    active_mask = (f > threshold).any(dim=0)  # [d_sae]
    return active_mask.nonzero(as_tuple=False).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────
# 2. Linear interaction matrix
# ──────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_interaction_matrix(
    sae: BatchTopKSAE,
    active_indices: torch.Tensor,
) -> torch.Tensor:
    """Compute pairwise write→read interactions among active features.

    ``interaction[i, j] = W_dec[active_i] · W_enc[:, active_j]``

    Diagonal (self-interaction) entries are zeroed out.

    Parameters
    ----------
    sae : BatchTopKSAE
    active_indices : Tensor [n_active]

    Returns
    -------
    Tensor [n_active, n_active]  on CPU
    """
    W_dec_active = sae.W_dec[active_indices]       # [n_active, d_model]
    W_enc_active = sae.W_enc[:, active_indices]    # [d_model, n_active]
    interaction = W_dec_active @ W_enc_active       # [n_active, n_active]
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
        ``attribution``       : Tensor [n_active, n_active]  (pruned)
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

    return {
        "active_indices": active_indices.cpu(),
        "attribution": pruned.cpu(),
        "logit_attribution": logit_attr,
        "feature_strengths": f[:, active_indices].mean(0).cpu(),
        "n_active": len(active_indices),
        "image_idx": image_idx,
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
    print(f"[OK] build_graph_data: n_active={gd['n_active']}, keys={list(gd.keys())}")

    # 6b. build_graph_data without classifier
    gd2 = build_graph_data(0, acts, sae, cfg, classifier_weights=None)
    assert gd2["logit_attribution"] is None
    print("[OK] build_graph_data: logit_attribution=None without classifier")

    print("\nAll graph_utils smoke tests passed ✓")
