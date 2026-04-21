"""HTML dashboard exporter — X-Ray Conglomerate SAE.

Generates a self-contained single-file HTML dashboard with:
  - Sidebar: searchable feature list with universality score bars
  - Main panel: feature header, activation histogram, universality profile,
    top activating patches, co-activation table, feature circuit graph

Dataset palette (must stay in sync with DATASET_COLORS in init.js):
  chexpert   #3B82F6  (blue)
  rsna       #F59E0B  (amber)
  siim_covid #10B981  (emerald)
  tb         #8B5CF6  (violet)
"""

from __future__ import annotations

import base64
import io
import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from typing import Any
from tqdm import tqdm

from visualisation.semantic_dict import get_patch_crop
from visualisation.coactivation import compute_coactivation_matrix

INIT_JS_PATH   = Path(__file__).parent / "init.js"
STYLE_CSS_PATH = Path(__file__).parent / "style.css"

# ── Dataset configuration ──────────────────────────────────────────────────────
# Colours must match DATASET_COLORS in init.js.

DATASET_COLORS = {
    "chexpert":   "#3B82F6",
    "rsna":       "#F59E0B",
    "siim_covid": "#10B981",
    "tb":         "#8B5CF6",
    "unknown":    "#9CA3AF",
}

# Keywords used to infer dataset from image file path
_DATASET_KEYWORDS: list[tuple[str, str]] = [
    ("chexpert",            "chexpert"),
    ("rsna",                "rsna"),
    ("rsna",                "pneumonia-detection"),
    ("siim_covid",          "siim"),
    ("siim_covid",          "covid"),
    ("tb",                  "tuberculosis"),
    ("tb",                  "shenzhen"),
    ("tb",                  "tb_"),
    ("tb",                  "/tb/"),
]


def infer_dataset_name(image_path: Path) -> str:
    """Infer the dataset key from the image file path."""
    path_str = str(image_path).lower()
    for ds_key, keyword in _DATASET_KEYWORDS:
        if keyword in path_str:
            return ds_key
    return "unknown"


# ── Data builders ──────────────────────────────────────────────────────────────

def build_acts_histogram_data(
    feature_max_per_image: np.ndarray,
    feature_idx: int,
    n_bins: int = 50,
) -> dict:
    """Activation distribution histogram data for Plotly."""
    active_values = feature_max_per_image[feature_max_per_image > 0]
    if len(active_values) == 0:
        return {"y": [], "x": [], "ticks": [], "title": "DEAD FEATURE"}

    counts, edges = np.histogram(active_values, bins=n_bins)
    centres = (edges[:-1] + edges[1:]) / 2
    step = max(1, n_bins // 5)

    return {
        "y":     [round(float(v), 4) for v in counts],
        "x":     [round(float(v), 4) for v in centres],
        "ticks": [round(float(v), 4) for v in centres[::step]],
        "title": f"F{feature_idx} — activation distribution (n={len(active_values)} images)",
    }


def build_feature_tables_data(
    feature_idx: int,
    coactivation_matrix: np.ndarray,
    decoder_cossim_matrix: np.ndarray,
    n_rows: int = 10,
) -> dict:
    """Co-activation + decoder cosine-similarity rows for the related-features table."""
    scores = coactivation_matrix[feature_idx].copy()
    scores[feature_idx] = 0.0
    top_indices = np.argsort(scores)[::-1][:n_rows]
    rows = [
        {
            "index":       str(i),
            "value":       f"{scores[i]:+.3f}",
            "percentageL1": f"{decoder_cossim_matrix[feature_idx, i]:+.3f}",
        }
        for i in top_indices
    ]
    return {"correlatedFeatures": rows}


def build_top_patches_data(
    feature_idx: int,
    stats: dict,
    images_dir: Path,
    spatial_h: int = 7,
    spatial_w: int = 7,
    context_patches: int = 1,
    top_k: int = 9,
) -> list:
    """Top-k activating patch crops encoded as base64 PNGs."""
    fmax = stats["feature_max"][:, feature_idx]
    if isinstance(fmax, torch.Tensor):
        fmax = fmax.cpu().numpy()

    top_indices = torch.topk(torch.from_numpy(fmax), k=min(top_k, len(fmax))).indices.tolist()

    # Support both flat dirs and one-level-deep patient subdirs
    image_paths = sorted(
        p for p in images_dir.rglob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

    result = []
    for image_idx in top_indices:
        patch_idx = int(stats["feature_argmax"][image_idx, feature_idx])
        act_value = round(float(fmax[image_idx]), 3)
        if image_idx >= len(image_paths):
            continue
        image_path = image_paths[image_idx]

        try:
            img  = Image.open(image_path).convert("RGB")
            crop = get_patch_crop(img, patch_idx, spatial_h, spatial_w, context_patches)
            if max(crop.size) > 200:
                crop = crop.resize((200, 200), Image.LANCZOS)
            buf = io.BytesIO()
            crop.save(buf, format="PNG")
            b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            result.append({
                "img_b64":   b64,
                "act":       act_value,
                "dataset":   infer_dataset_name(image_path),
                "image_idx": image_idx,
                "patch_idx": patch_idx,
            })
        except Exception:
            continue

    return result


def build_universality_profile_data(
    feature_idx: int,
    stats: dict,
    images_dir: Path,
    top_k: int = 50,
) -> dict:
    """Fraction of top-k activating images per dataset (for the stacked bar)."""
    fmax = stats["feature_max"][:, feature_idx]
    if isinstance(fmax, torch.Tensor):
        fmax = fmax.cpu().numpy()

    top_indices = torch.topk(
        torch.from_numpy(fmax), k=min(top_k, len(fmax))
    ).indices.tolist()

    image_paths = sorted(
        p for p in images_dir.rglob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

    dataset_counts: dict[str, int] = {}
    total = 0
    for idx in top_indices:
        if float(fmax[idx]) <= 0 or idx >= len(image_paths):
            continue
        ds = infer_dataset_name(image_paths[idx])
        dataset_counts[ds] = dataset_counts.get(ds, 0) + 1
        total += 1

    bars = []
    if total > 0:
        for ds, count in dataset_counts.items():
            bars.append({
                "dataset":  ds,
                "fraction": round(count / total, 4),
                "color":    DATASET_COLORS.get(ds, DATASET_COLORS["unknown"]),
            })
    bars.sort(key=lambda x: x["fraction"], reverse=True)

    return {
        "bars":  bars,
        "title": f"F{feature_idx} — top-{top_k} image breakdown by dataset",
    }


def build_attribution_graph_data(
    feature_idx: int,
    stats: dict,
    sae: Any,
    activations: np.ndarray,
    n_patches: int = 49,
    top_k_images: int = 3,
    max_active: int = 60,
    activation_threshold_frac: float = 0.1,
) -> list[dict]:
    """Build force-graph data for the feature circuit panel (top-k images)."""
    fmax = stats["feature_max"][:, feature_idx]
    fmax_np = fmax.cpu().numpy() if isinstance(fmax, torch.Tensor) else np.asarray(fmax)

    fmax_t = torch.from_numpy(fmax_np).float()
    if fmax_t.max().item() <= 0:
        return []

    k = min(top_k_images, len(fmax_t))
    top_img_indices = torch.topk(fmax_t, k=k).indices.tolist()

    results: list[dict] = []

    for img_idx in top_img_indices:
        if float(fmax_np[img_idx]) <= 0:
            continue
        start, end = img_idx * n_patches, (img_idx + 1) * n_patches
        if end > len(activations):
            continue

        acts_t = torch.from_numpy(np.asarray(activations[start:end])).float()
        device = next(sae.parameters()).device
        acts_t = acts_t.to(device)

        with torch.no_grad():
            f = sae.encode(acts_t)

        image_max = f.max().item()
        if image_max <= 0:
            continue
        threshold = activation_threshold_frac * image_max
        active_mask = (f > threshold).any(dim=0)
        active_indices = active_mask.nonzero(as_tuple=False).squeeze(-1)
        if active_indices.ndim == 0:
            active_indices = active_indices.unsqueeze(0)

        if len(active_indices) > max_active:
            strengths_all = f[:, active_indices].mean(dim=0)
            topk_local = torch.topk(strengths_all, k=max_active)
            active_indices = active_indices[topk_local.indices].sort().values

        # Always include the queried feature
        if feature_idx not in active_indices.tolist():
            target_t = torch.tensor([feature_idx], dtype=active_indices.dtype, device=device)
            active_indices = torch.cat([active_indices, target_t]).sort().values

        if len(active_indices) < 2:
            continue

        feat_strengths = f[:, active_indices].mean(dim=0)

        W_dec = sae.W_dec[active_indices]
        W_enc = sae.W_enc[:, active_indices]
        interaction  = W_dec @ W_enc
        interaction.fill_diagonal_(0.0)
        attribution = feat_strengths.unsqueeze(1) * interaction

        flat = attribution.flatten()
        n_keep = min(30, flat.numel())
        if n_keep > 0:
            keep_idx = torch.topk(flat.abs(), k=n_keep).indices
            mask = torch.zeros_like(flat, dtype=torch.bool)
            mask[keep_idx] = True
            attribution = torch.where(mask, flat, torch.zeros_like(flat)).reshape(interaction.shape)

        # Assign dataset colour to each node via plurality vote
        node_colors: dict[int, str] = {}
        for fidx_t in active_indices:
            fidx = int(fidx_t.item())
            feat_fmax = stats["feature_max"][:, fidx]
            feat_fmax = feat_fmax.cpu() if isinstance(feat_fmax, torch.Tensor) else torch.from_numpy(np.asarray(feat_fmax))
            top5 = torch.topk(feat_fmax.float(), k=min(5, len(feat_fmax))).indices.tolist()
            node_colors[fidx] = DATASET_COLORS["unknown"]  # can't resolve paths here

        nodes = [
            {
                "id":           int(active_indices[i].item()),
                "strength":     round(float(feat_strengths[i].item()), 4),
                "is_target":    int(active_indices[i].item()) == feature_idx,
                "dataset_color": node_colors.get(int(active_indices[i].item()), DATASET_COLORS["unknown"]),
            }
            for i in range(len(active_indices))
        ]

        attr_np = attribution.detach().cpu().numpy()
        n_act   = len(active_indices)
        edges = [
            {
                "source":    int(active_indices[i].item()),
                "target":    int(active_indices[j].item()),
                "weight":    round(abs(float(attr_np[i, j])), 4),
                "excitatory": float(attr_np[i, j]) > 0,
            }
            for i in range(n_act) for j in range(n_act)
            if float(attr_np[i, j]) != 0.0
        ]

        from visualisation.graph_utils import compute_decoder_cosine_similarity
        cossim = compute_decoder_cosine_similarity(sae, active_indices)
        cossim_pairs = [
            {"i": int(active_indices[ii].item()), "j": int(active_indices[jj].item()), "cossim": round(float(cossim[ii, jj].item()), 3)}
            for ii in range(n_act) for jj in range(ii + 1, n_act)
            if float(cossim[ii, jj].item()) > 0.3
        ]

        results.append({
            "image_idx":   img_idx,
            "nodes":       nodes,
            "edges":       edges,
            "cossim_pairs": cossim_pairs,
        })

    return results


def build_data_for_feature(
    feature_idx: int,
    stats: dict,
    coactivation_matrix: np.ndarray,
    decoder_cossim_matrix: np.ndarray,
    images_dir: Path,
    spatial_h: int = 7,
    spatial_w: int = 7,
    context_patches: int = 1,
    top_k_patches: int = 9,
    n_coactivation_rows: int = 10,
    sae: Any | None = None,
    activations: np.ndarray | None = None,
    n_patches: int | None = None,
) -> dict:
    fmax = stats["feature_max"][:, feature_idx]
    if isinstance(fmax, torch.Tensor):
        fmax = fmax.cpu().numpy()

    data: dict = {
        "actsHistogram":       build_acts_histogram_data(np.array(fmax), feature_idx),
        "featureTables":       build_feature_tables_data(
                                   feature_idx, coactivation_matrix, decoder_cossim_matrix,
                                   n_rows=n_coactivation_rows),
        "topPatches":          build_top_patches_data(
                                   feature_idx, stats, images_dir,
                                   spatial_h, spatial_w, context_patches, top_k=top_k_patches),
        "universalityProfile": build_universality_profile_data(
                                   feature_idx, stats, images_dir, top_k=50),
    }

    if sae is not None and activations is not None:
        n_p = n_patches if n_patches is not None else spatial_h * spatial_w
        data["attributionMiniGraph"] = build_attribution_graph_data(
            feature_idx, stats, sae, activations, n_patches=n_p)

    return data


# ── HTML assembly ──────────────────────────────────────────────────────────────

def _html_template(
    data_json: str,
    metadata_json: str,
    start_key: str,
    style_css: str,
    init_js: str,
    n_features: int,
    checkpoint_label: str = "",
) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SAE Feature Dashboard — X-Ray Conglomerate</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<script src="https://d3js.org/d3.v7.min.js"></script>
<script src="https://cdn.plot.ly/plotly-2.27.1.min.js"></script>
<style>
{style_css}
</style>
</head>
<body>

<!-- ── Top header ────────────────────────────────────────────────────── -->
<header id="app-header">
  <div class="logo">
    <div class="logo-icon">S</div>
    SAE Feature Dashboard
  </div>
  <div class="header-meta">
    <span class="header-badge">UMedPT Swin · X-Ray Conglomerate</span>
    <span class="header-badge">{n_features} features</span>
    {f'<span class="header-badge">{checkpoint_label}</span>' if checkpoint_label else ""}
  </div>
</header>

<!-- ── App shell ─────────────────────────────────────────────────────── -->
<div id="app-shell">

  <!-- Sidebar -->
  <aside id="sidebar">
    <div id="sidebar-header">
      <h2>Features</h2>
      <input id="feature-search" type="search" placeholder="Search F…" autocomplete="off" />
    </div>
    <div id="feature-list"></div>
  </aside>

  <!-- Main panel -->
  <main id="main-panel">

    <!-- Feature header -->
    <div class="card" id="feature-header"></div>

    <!-- Histogram + Universality row -->
    <div class="row-2col">
      <div class="card">
        <div class="card-title">Activation Distribution</div>
        <div id="hist-container"></div>
      </div>
      <div class="card">
        <div class="card-title">Universality Profile</div>
        <div id="u-profile-container"></div>
      </div>
    </div>

    <!-- Top patches -->
    <div class="card">
      <div class="card-title">Top Activating Patches</div>
      <div id="patches-container"></div>
    </div>

    <!-- Co-activation + Circuit row -->
    <div class="row-2col">
      <div class="card">
        <div class="card-title">Correlated Features</div>
        <div id="coact-container"></div>
      </div>
      <div class="card">
        <div class="card-title">Feature Circuit</div>
        <div id="graph-container"></div>
      </div>
    </div>

  </main>
</div>

<script>
const START_KEY  = {json.dumps(start_key)};
const METADATA   = {metadata_json};

function defineData() {{ return {data_json}; }}
const DATA = defineData();

{init_js}
</script>
</body>
</html>"""


def save_dashboard(
    feature_indices: list[int],
    stats: dict,
    sae: Any,
    images_dir: Path,
    output_path: Path,
    spatial_h: int = 7,
    spatial_w: int = 7,
    context_patches: int = 1,
    top_k_patches: int = 9,
    n_coactivation_rows: int = 10,
    coactivation_top_k: int = 50,
    verbose: bool = True,
    activations: np.ndarray | None = None,
    checkpoint_label: str = "",
) -> None:
    """Build and write the self-contained HTML dashboard.

    Parameters
    ----------
    feature_indices:    SAE feature indices to include
    stats:              output of collector.compute_feature_stats()
    sae:                trained SAE in eval mode
    images_dir:         directory of original images (PNG/JPG, sorted order)
    output_path:        destination .html file
    spatial_h/w:        patch grid dimensions (7×7 for pm_index=2)
    context_patches:    extra patches of context around crop (1 for 7×7)
    top_k_patches:      number of top activating patches to show (default 9)
    n_coactivation_rows: rows in the co-activation table
    coactivation_top_k: top-k images used to compute Jaccard co-activation
    activations:        flat [N_eff, d_model] tensor for circuit graphs
    checkpoint_label:   short string shown in the header badge
    """
    # Co-activation matrix (Jaccard over top-k image sets)
    coact_sub  = compute_coactivation_matrix(
        stats, top_k=coactivation_top_k, feature_indices=feature_indices
    ).cpu().numpy()
    coact_full = np.zeros((sae.d_sae, sae.d_sae))
    for ii, fi in enumerate(feature_indices):
        for jj, fj in enumerate(feature_indices):
            coact_full[fi, fj] = coact_sub[ii, jj]

    # Decoder cosine similarity matrix
    W_dec_sub = sae.W_dec[feature_indices].cpu().float()
    normed    = W_dec_sub / W_dec_sub.norm(dim=1, keepdim=True).clamp(min=1e-8)
    cossim_sub = (normed @ normed.T).detach().numpy()
    np.fill_diagonal(cossim_sub, 0.0)
    cossim_full = np.zeros((sae.d_sae, sae.d_sae))
    for ii, fi in enumerate(feature_indices):
        for jj, fj in enumerate(feature_indices):
            cossim_full[fi, fj] = cossim_sub[ii, jj]

    # Build per-feature data
    n_p  = spatial_h * spatial_w
    DATA = {}
    for feat_idx in tqdm(feature_indices, desc="Building dashboard", disable=not verbose):
        DATA[str(feat_idx)] = build_data_for_feature(
            feature_idx          = feat_idx,
            stats                = stats,
            coactivation_matrix  = coact_full,
            decoder_cossim_matrix= cossim_full,
            images_dir           = images_dir,
            spatial_h            = spatial_h,
            spatial_w            = spatial_w,
            context_patches      = context_patches,
            top_k_patches        = top_k_patches,
            n_coactivation_rows  = n_coactivation_rows,
            sae                  = sae if activations is not None else None,
            activations          = activations,
            n_patches            = n_p,
        )

    metadata = {
        "spatial_h": spatial_h,
        "spatial_w": spatial_w,
        "n_features": len(feature_indices),
    }
    start_key = str(min(feature_indices))

    style_css = STYLE_CSS_PATH.read_text(encoding="utf-8") if STYLE_CSS_PATH.exists() else ""
    init_js   = INIT_JS_PATH.read_text(encoding="utf-8")   if INIT_JS_PATH.exists()   else ""

    html = _html_template(
        data_json       = json.dumps(DATA, separators=(",", ":")),
        metadata_json   = json.dumps(metadata),
        start_key       = start_key,
        style_css       = style_css,
        init_js         = init_js,
        n_features      = len(feature_indices),
        checkpoint_label= checkpoint_label,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    if verbose:
        size_mb = output_path.stat().st_size / 1e6
        print(f"Dashboard saved → {output_path}  ({len(feature_indices)} features, {size_mb:.1f} MB)")
