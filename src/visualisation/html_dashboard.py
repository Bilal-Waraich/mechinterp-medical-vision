"""HTML dashboard exporter for Phase 3 visualisation."""

import base64
import io
import json
import math
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

DATASET_COLORS = {
    "histology":  "#8B5E3C",
    "covid":      "#2E86AB",
    "pneumonia":  "#A23B72",
    "brain_mri":  "#F18F01",
}

def infer_dataset_name(image_path: Path) -> str:
    path_str = str(image_path).lower()
    if "semicol" in path_str or "crc" in path_str:
        return "histology"
    if "covid" in path_str:
        return "covid"
    if "pneumonia" in path_str:
        return "pneumonia"
    if "brain" in path_str or "tumor" in path_str:
        return "brain_mri"
    return image_path.parent.name

def build_acts_histogram_data(feature_max_per_image: np.ndarray, feature_idx: int, n_bins: int = 50) -> dict:
    values = feature_max_per_image
    active_values = values[values > 0]
    count = len(active_values)
    
    if count == 0:
        return {
            "y": [],
            "x": [],
            "ticks": [],
            "title": "DEAD FEATURE"
        }
        
    counts, edges = np.histogram(active_values, bins=n_bins)
    bar_centres = (edges[:-1] + edges[1:]) / 2
    
    step = max(1, n_bins // 5)
    ticks = bar_centres[::step].tolist()
    
    # round values
    y_rounded = [round(float(v), 4) for v in counts]
    x_rounded = [round(float(v), 4) for v in bar_centres]
    ticks_rounded = [round(float(v), 4) for v in ticks]
    
    return {
        "y": y_rounded,
        "x": x_rounded,
        "ticks": ticks_rounded,
        "title": f"F{feature_idx} — activation distribution (n={count} images)"
    }

def build_feature_tables_data(feature_idx: int, coactivation_matrix: np.ndarray, decoder_cossim_matrix: np.ndarray, n_rows: int = 10) -> dict:
    scores = coactivation_matrix[feature_idx].copy()
    scores[feature_idx] = 0.0
    
    top_indices = np.argsort(scores)[::-1][:n_rows]
    rows = []
    for i in top_indices:
        rows.append({
            "index": str(i),
            "value": f"{scores[i]:+.3f}",
            "percentageL1": f"{decoder_cossim_matrix[feature_idx, i]:+.3f}"
        })
        
    return {"correlatedFeatures": rows}

def build_top_patches_data(feature_idx: int, stats: dict, images_dir: Path, spatial_h: int = 14, spatial_w: int = 14, context_patches: int = 2, top_k: int = 9) -> list:
    fmax = stats["feature_max"][:, feature_idx]
    if isinstance(fmax, torch.Tensor):
        fmax = fmax.cpu().numpy()
        
    top_k_indices = torch.topk(torch.from_numpy(fmax), k=min(top_k, len(fmax))).indices.tolist()
    
    image_paths = sorted(images_dir.glob("*"))
    result = []
    
    for image_idx in top_k_indices:
        patch_idx = int(stats["feature_argmax"][image_idx, feature_idx])
        act_value = round(float(fmax[image_idx]), 3)
        image_path = image_paths[image_idx]
        
        try:
            img = Image.open(image_path).convert("RGB")
            crop = get_patch_crop(img, patch_idx, spatial_h, spatial_w, context_patches)
            if max(crop.size) > 200:
                crop = crop.resize((200, 200), Image.LANCZOS)
                
            buf = io.BytesIO()
            crop.save(buf, format="PNG")
            b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            
            dataset = infer_dataset_name(image_path)
            
            result.append({
                "img_b64": b64,
                "act": act_value,
                "dataset": dataset,
                "image_idx": image_idx,
                "patch_idx": patch_idx
            })
        except Exception:
            continue
            
    return result

def build_universality_profile_data(feature_idx: int, stats: dict, images_dir: Path, top_k: int = 50) -> dict:
    fmax = stats["feature_max"][:, feature_idx]
    if isinstance(fmax, torch.Tensor):
        fmax = fmax.cpu().numpy()
        
    top_k_indices = torch.topk(torch.from_numpy(fmax), k=min(top_k, len(fmax))).indices.tolist()
    image_paths = sorted(images_dir.glob("*"))
    
    dataset_counts = {}
    total_valid = 0
    
    for img_idx in top_k_indices:
        if float(fmax[img_idx]) <= 0: # only count active images
            continue
        ipath = image_paths[img_idx]
        dname = infer_dataset_name(ipath)
        dataset_counts[dname] = dataset_counts.get(dname, 0) + 1
        total_valid += 1
        
    bars = []
    if total_valid > 0:
        for dname, dcount in dataset_counts.items():
            frac = round(dcount / total_valid, 4)
            bars.append({
                "dataset": dname,
                "fraction": frac,
                "color": DATASET_COLORS.get(dname, "#888888")
            })
            
    bars.sort(key=lambda x: x["fraction"], reverse=True)
    
    return {
        "bars": bars,
        "title": f"F{feature_idx} — top-{top_k} image breakdown by dataset"
    }

def build_data_for_feature(
    feature_idx: int,
    stats: dict,
    coactivation_matrix: np.ndarray,
    decoder_cossim_matrix: np.ndarray,
    images_dir: Path,
    spatial_h: int = 14,
    spatial_w: int = 14,
    context_patches: int = 2,
    top_k_patches: int = 9,
    n_coactivation_rows: int = 10
) -> dict:
    fmax = stats["feature_max"][:, feature_idx]
    if isinstance(fmax, torch.Tensor):
        fmax = fmax.cpu().numpy()
        
    return {
        "actsHistogram": build_acts_histogram_data(np.array(fmax), feature_idx),
        "featureTables": build_feature_tables_data(feature_idx, coactivation_matrix, decoder_cossim_matrix, n_rows=n_coactivation_rows),
        "topPatches": build_top_patches_data(feature_idx, stats, images_dir, spatial_h, spatial_w, context_patches, top_k=top_k_patches),
        "universalityProfile": build_universality_profile_data(feature_idx, stats, images_dir, top_k=50)
    }

def build_metadata(height: int = 820) -> dict:
    return {
        "height": height,
        "layout": [
            ["featureTables", "universalityProfile"],
            ["actsHistogram"],
            ["topPatches"]
        ],
        "columnWidths": [300, 390, 640]
    }

def build_additional_js() -> str:
    return '''
function setupTopPatches(featureIdx, componentData, containerId) {
    const container = d3.select(`#${containerId}`);
    container.append("h4").text("TOP ACTIVATING PATCHES");

    const datasetColors = {
        "histology": "#8B5E3C",
        "covid":     "#2E86AB",
        "pneumonia": "#A23B72",
        "brain_mri": "#F18F01",
    };

    const grid = container.append("div")
        .style("display", "grid")
        .style("grid-template-columns", "repeat(3, 200px)")
        .style("gap", "8px")
        .style("margin-top", "8px");

    componentData.forEach(patch => {
        const cell = grid.append("div")
            .attr("class", "hover-text")
            .style("position", "relative");

        cell.append("img")
            .attr("src", patch.img_b64)
            .attr("width", "200px")
            .style("display", "block")
            .style("border-radius", "4px");

        cell.append("div")
            .style("font-size", "0.75em")
            .style("text-align", "center")
            .style("margin-top", "2px")
            .text(`act=${patch.act.toFixed(3)}`);

        cell.append("div")
            .style("position", "absolute")
            .style("top", "4px")
            .style("right", "4px")
            .style("background-color", datasetColors[patch.dataset] || "#888")
            .style("color", "white")
            .style("font-size", "0.65em")
            .style("padding", "1px 5px")
            .style("border-radius", "8px")
            .text(patch.dataset);

        const tooltipContainer = cell.append("div")
            .attr("class", "tooltip-container");
        tooltipContainer.append("div")
            .attr("class", "tooltip")
            .html(`
                <code>F${featureIdx}</code><br>
                act: <b>${patch.act.toFixed(3)}</b><br>
                dataset: <b>${patch.dataset}</b><br>
                image: ${patch.image_idx}, patch: ${patch.patch_idx}
            `);
    });
}

function setupUniversalityProfile(featureIdx, componentData, containerId) {
    const container = d3.select(`#${containerId}`);
    container.append("h4").text("UNIVERSALITY PROFILE");

    const bar = container.append("div")
        .style("display", "flex")
        .style("height", "18px")
        .style("border-radius", "4px")
        .style("overflow", "hidden")
        .style("margin-bottom", "8px");

    componentData.bars.forEach(b => {
        bar.append("div")
            .style("width", `${(b.fraction * 100).toFixed(1)}%`)
            .style("background-color", b.color)
            .attr("title", `${b.dataset}: ${(b.fraction * 100).toFixed(1)}%`);
    });

    const legend = container.append("div")
        .style("display", "flex")
        .style("flex-wrap", "wrap")
        .style("gap", "6px");

    componentData.bars.forEach(b => {
        const pill = legend.append("div")
            .style("display", "flex")
            .style("align-items", "center")
            .style("gap", "4px")
            .style("font-size", "0.75em");
        pill.append("div")
            .style("width", "10px")
            .style("height", "10px")
            .style("border-radius", "50%")
            .style("background-color", b.color);
        pill.append("span")
            .text(`${b.dataset} ${(b.fraction * 100).toFixed(0)}%`);
    });
}

componentMap["topPatches"] = setupTopPatches;
componentMap["universalityProfile"] = setupUniversalityProfile;
'''

def save_dashboard(
    feature_indices: list[int],
    stats: dict,
    sae: Any,
    images_dir: Path,
    output_path: Path,
    spatial_h: int = 14,
    spatial_w: int = 14,
    context_patches: int = 2,
    top_k_patches: int = 9,
    n_coactivation_rows: int = 10,
    coactivation_top_k: int = 50,
    verbose: bool = True
) -> None:
    coact_sub = compute_coactivation_matrix(stats, top_k=coactivation_top_k, feature_indices=feature_indices).cpu().numpy()
    coactivation_full = np.zeros((sae.d_sae, sae.d_sae))
    for ii, fi in enumerate(feature_indices):
        for jj, fj in enumerate(feature_indices):
            coactivation_full[fi, fj] = coact_sub[ii, jj]

    W_dec_sub = sae.W_dec[feature_indices].cpu().float()
    norms = W_dec_sub.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normed = W_dec_sub / norms
    cossim_sub = (normed @ normed.T).numpy()
    np.fill_diagonal(cossim_sub, 0.0)
    
    cossim_full = np.zeros((sae.d_sae, sae.d_sae))
    for ii, fi in enumerate(feature_indices):
        for jj, fj in enumerate(feature_indices):
            cossim_full[fi, fj] = cossim_sub[ii, jj]

    DATA = {}
    for feat_idx in tqdm(feature_indices, desc="Building dashboard", disable=not verbose):
        DATA[str(feat_idx)] = build_data_for_feature(
            feat_idx=feat_idx,
            stats=stats,
            coactivation_matrix=coactivation_full,
            decoder_cossim_matrix=cossim_full,
            images_dir=images_dir,
            spatial_h=spatial_h,
            spatial_w=spatial_w,
            context_patches=context_patches,
            top_k_patches=top_k_patches,
            n_coactivation_rows=n_coactivation_rows
        )

    metadata = build_metadata()
    start_key = str(min(feature_indices))
    
    init_js = INIT_JS_PATH.read_text(encoding="utf-8") if INIT_JS_PATH.exists() else ""
    style_css = STYLE_CSS_PATH.read_text(encoding="utf-8") if STYLE_CSS_PATH.exists() else ""
    extra_js = build_additional_js()

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>SAE Feature Dashboard — UMedPT</title>
<script src="https://d3js.org/d3.v6.min.js"></script>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
{style_css}
.topPatches {{ margin-bottom: 20px; }}
.topPatches img {{ border: 1px solid #e6e6e6; cursor: pointer; }}
.topPatches img:hover {{ border-color: #999; box-shadow: 0 2px 6px rgba(0,0,0,0.2); }}
.universalityProfile {{ margin-bottom: 20px; min-width: 280px; }}
</style>
</head>
<body>
<div id='dropdown-container'></div>
<div class='grid-container'></div>
<script>
const START_KEY = {json.dumps(start_key)};
const METADATA = {json.dumps(metadata)};
const DATA = defineData();
const PROMPT_DATA = definePromptData();

{init_js}

{extra_js}

function defineData() {{
    return {json.dumps(DATA, separators=(',', ':'))};
}}
function definePromptData() {{
    return {{}};
}}
</script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    if verbose:
        size_mb = output_path.stat().st_size / 1e6
        print(f"Dashboard saved → {output_path}  "
              f"({len(feature_indices)} features, {size_mb:.1f} MB)")
