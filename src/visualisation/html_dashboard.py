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

VIEW_COLORS = {
    "brain_mri/axial":    "#F18F01",
    "brain_mri/sagittal": "#E07800",
    "brain_mri/coronal":  "#FFB347",
}

def build_universality_profile_data(feature_idx: int, stats: dict, images_dir: Path, top_k: int = 50, view_labels: dict[str, str] | None = None) -> dict:
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
        if view_labels is not None and dname == "brain_mri":
            dname = "brain_mri/" + view_labels.get(ipath.name, "unknown")
        dataset_counts[dname] = dataset_counts.get(dname, 0) + 1
        total_valid += 1

    bars = []
    if total_valid > 0:
        for dname, dcount in dataset_counts.items():
            frac = round(dcount / total_valid, 4)
            color = VIEW_COLORS.get(dname, DATASET_COLORS.get(dname, "#888888"))
            bars.append({
                "dataset": dname,
                "fraction": frac,
                "color": color
            })

    bars.sort(key=lambda x: x["fraction"], reverse=True)

    return {
        "bars": bars,
        "title": f"F{feature_idx} — top-{top_k} image breakdown by dataset"
    }


def build_attribution_graph_data(
    feature_idx: int,
    stats: dict,
    sae: Any,                       # BatchTopKSAE
    activations: np.ndarray,        # flat [N*n_patches, d_model]
    n_patches: int = 196,
    top_k_images: int = 3,
    max_active: int = 60,
    activation_threshold_frac: float = 0.1,
) -> list[dict]:
    """Build attribution mini-graph data for the top-k images that
    *feature_idx* fires on most strongly.

    Returns a list of JSON-serialisable dicts (one per image) with
    ``nodes`` and ``edges`` for a D3 force-directed graph.
    Returns ``[]`` if the feature is dead or has zero activation.
    """
    fmax = stats["feature_max"][:, feature_idx]
    if isinstance(fmax, torch.Tensor):
        fmax_np = fmax.cpu().numpy()
    else:
        fmax_np = np.asarray(fmax)

    # Find top-k images by activation
    fmax_t = torch.from_numpy(fmax_np).float()
    if fmax_t.max().item() <= 0:
        return []  # feature is dead

    k = min(top_k_images, len(fmax_t))
    top_img_indices = torch.topk(fmax_t, k=k).indices.tolist()

    results: list[dict] = []
    images_dir_path = None  # will be set lazily if needed

    for img_idx in top_img_indices:
        if float(fmax_np[img_idx]) <= 0:
            continue

        # Slice activations for this image
        start = img_idx * n_patches
        end = start + n_patches
        if end > len(activations):
            continue
        acts_slice = activations[start:end]  # [n_patches, d_model]
        if isinstance(acts_slice, np.ndarray):
            acts_t = torch.from_numpy(acts_slice).float()
        else:
            acts_t = acts_slice.float()

        # Move to same device as SAE
        device = next(sae.parameters()).device
        acts_t = acts_t.to(device)

        # Encode through SAE
        with torch.no_grad():
            f = sae.encode(acts_t)  # [n_patches, d_sae]

        # Find active features with adaptive threshold
        image_max = f.max().item()
        if image_max <= 0:
            continue
        threshold = activation_threshold_frac * image_max
        active_mask = (f > threshold).any(dim=0)
        active_indices = active_mask.nonzero(as_tuple=False).squeeze(-1)
        if active_indices.ndim == 0:
            active_indices = active_indices.unsqueeze(0)

        # Cap at max_active
        if len(active_indices) > max_active:
            strengths = f[:, active_indices].mean(dim=0)
            topk = torch.topk(strengths, k=max_active)
            active_indices = active_indices[topk.indices]
            active_indices = active_indices.sort().values

        # Ensure the feature we're graphing is always present as a node
        if feature_idx not in active_indices:
            target_t = torch.tensor([feature_idx], dtype=active_indices.dtype, device=active_indices.device)
            active_indices = torch.cat([active_indices, target_t]).sort().values

        if len(active_indices) < 2:
            continue

        # Feature strengths
        feat_strengths = f[:, active_indices].mean(dim=0)  # [n_active]

        # Compute interaction & attribution
        W_dec_active = sae.W_dec[active_indices]
        W_enc_active = sae.W_enc[:, active_indices]
        interaction = W_dec_active @ W_enc_active
        interaction.fill_diagonal_(0.0)
        attribution = feat_strengths.unsqueeze(1) * interaction

        # Prune to top 30 edges by abs
        flat = attribution.flatten()
        n_keep = min(30, flat.numel())
        if n_keep > 0:
            topk_edges = torch.topk(flat.abs(), k=n_keep)
            mask = torch.zeros_like(flat, dtype=torch.bool)
            mask[topk_edges.indices] = True
            pruned = torch.where(mask, flat, torch.zeros_like(flat))
            attribution = pruned.reshape(interaction.shape)

        # Determine dataset colour for each active feature
        node_colors: dict[int, str] = {}
        for local_i in range(len(active_indices)):
            fidx = int(active_indices[local_i].item())
            # Simple majority vote from top-5 images
            feat_fmax = stats["feature_max"][:, fidx]
            if isinstance(feat_fmax, torch.Tensor):
                feat_fmax = feat_fmax.cpu()
            else:
                feat_fmax = torch.from_numpy(np.asarray(feat_fmax))
            top5 = torch.topk(feat_fmax.float(), k=min(5, len(feat_fmax))).indices.tolist()
            # We don't have images_dir here, use a default colour
            node_colors[fidx] = DATASET_COLORS.get("unknown", "#888888")

        # Build nodes list
        nodes = []
        for local_i in range(len(active_indices)):
            fidx = int(active_indices[local_i].item())
            nodes.append({
                "id": fidx,
                "strength": round(float(feat_strengths[local_i].item()), 4),
                "is_target": fidx == feature_idx,
                "dataset_color": node_colors.get(fidx, "#888888"),
            })

        # Build edges list
        edges = []
        n_active = len(active_indices)
        attr_np_local = attribution.detach().cpu().numpy()
        for i in range(n_active):
            for j in range(n_active):
                w = float(attr_np_local[i, j])
                if w == 0.0:
                    continue
                edges.append({
                    "source": int(active_indices[i].item()),
                    "target": int(active_indices[j].item()),
                    "weight": round(abs(w), 4),
                    "excitatory": w > 0,
                })

        # Compute pairwise decoder cosine similarities for supernode merging
        from visualisation.graph_utils import compute_decoder_cosine_similarity
        cossim_matrix = compute_decoder_cosine_similarity(sae, active_indices)
        cossim_pairs = []
        n_act = len(active_indices)
        for ii in range(n_act):
            for jj in range(ii + 1, n_act):
                val = float(cossim_matrix[ii, jj].item())
                if val > 0.3:
                    cossim_pairs.append({
                        "i": int(active_indices[ii].item()),
                        "j": int(active_indices[jj].item()),
                        "cossim": round(val, 3),
                    })

        results.append({
            "image_idx": img_idx,
            "nodes": nodes,
            "edges": edges,
            "cossim_pairs": cossim_pairs,
        })

    return results

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
    n_coactivation_rows: int = 10,
    sae: Any | None = None,
    activations: np.ndarray | None = None,
    n_patches: int | None = None,
    view_labels: dict[str, str] | None = None,
) -> dict:
    fmax = stats["feature_max"][:, feature_idx]
    if isinstance(fmax, torch.Tensor):
        fmax = fmax.cpu().numpy()

    data: dict = {
        "actsHistogram": build_acts_histogram_data(np.array(fmax), feature_idx),
        "featureTables": build_feature_tables_data(feature_idx, coactivation_matrix, decoder_cossim_matrix, n_rows=n_coactivation_rows),
        "topPatches": build_top_patches_data(feature_idx, stats, images_dir, spatial_h, spatial_w, context_patches, top_k=top_k_patches),
        "universalityProfile": build_universality_profile_data(feature_idx, stats, images_dir, top_k=50, view_labels=view_labels),
    }

    if sae is not None and activations is not None:
        n_p = n_patches if n_patches is not None else spatial_h * spatial_w
        data["attributionMiniGraph"] = build_attribution_graph_data(
            feature_idx, stats, sae, activations,
            n_patches=n_p,
        )

    return data

def build_metadata(height: int = 900) -> dict:
    return {
        "height": height,
        "layout": [
            ["featureTables", "universalityProfile"],
            ["actsHistogram"],
            ["topPatches"],
            ["attributionMiniGraph"],
        ],
        "columnWidths": [300, 390, 640, 390]
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

function setupAttributionMiniGraph(featureIdx, componentData, containerId) {
    if (!componentData || componentData.length === 0) return;

    const container = d3.select(`#${containerId}`);
    container.append("h4").text("FEATURE CIRCUITS (top-3 images)");

    // Supernode slider
    const sliderRow = container.append("div")
        .style("display", "flex")
        .style("align-items", "center")
        .style("gap", "8px")
        .style("margin-bottom", "10px")
        .style("font-size", "0.75em");

    sliderRow.append("span").text("Supernode threshold:");

    const sliderVal = sliderRow.append("span")
        .style("font-weight", "bold")
        .style("min-width", "32px")
        .text("OFF");

    const slider = sliderRow.append("input")
        .attr("type", "range")
        .attr("min", "0")
        .attr("max", "100")
        .attr("value", "101")
        .style("width", "120px");

    sliderRow.append("span")
        .style("color", "#888")
        .text("<- drag left to merge similar features");

    function renderGraph(graphData, svgEl, threshold) {
        const svg = d3.select(svgEl);
        svg.selectAll("*").remove();

        const rawNodes = graphData.nodes.map(d => ({...d}));
        const rawEdges = graphData.edges;
        const cossimPairs = graphData.cossim_pairs || [];

        // Union-find for supernode grouping
        const parent = {};
        rawNodes.forEach(n => { parent[n.id] = n.id; });

        function find(x) {
            if (parent[x] !== x) parent[x] = find(parent[x]);
            return parent[x];
        }
        function union(a, b) { parent[find(a)] = find(b); }

        if (threshold <= 1.0) {
            cossimPairs.forEach(p => {
                if (p.cossim >= threshold) union(p.i, p.j);
            });
        }

        // Build collapsed node list
        const superGroups = {};
        rawNodes.forEach(n => {
            const root = find(n.id);
            if (!superGroups[root]) {
                superGroups[root] = {
                    id:            root,
                    members:       [],
                    strength:      0,
                    is_target:     false,
                    dataset_color: n.dataset_color,
                };
            }
            superGroups[root].members.push(n.id);
            superGroups[root].strength += n.strength;
            if (n.is_target) superGroups[root].is_target = true;
        });

        const nodes = Object.values(superGroups);

        // Remap edges to supernode ids, deduplicate self-loops
        const edgeMap = {};
        rawEdges.forEach(e => {
            const src = find(e.source), tgt = find(e.target);
            if (src === tgt) return;
            const key = `${src}-${tgt}`;
            if (!edgeMap[key]) edgeMap[key] = {
                source: src, target: tgt,
                weight: 0, excitatory: e.excitatory
            };
            edgeMap[key].weight += e.weight;
        });
        const edges = Object.values(edgeMap);

        const width = 360, height = 240;

        svg.append("defs").append("marker")
            .attr("id", `arrow-${containerId}`)
            .attr("viewBox", "0 -3 6 6")
            .attr("refX", 12).attr("refY", 0)
            .attr("markerWidth", 4).attr("markerHeight", 4)
            .attr("orient", "auto")
            .append("path").attr("d", "M0,-3L6,0L0,3").attr("fill", "#aaa");

        const simulation = d3.forceSimulation(nodes)
            .force("link",      d3.forceLink(edges).id(d => d.id)
                                   .distance(60).strength(0.5))
            .force("charge",    d3.forceManyBody().strength(-80))
            .force("center",    d3.forceCenter(width/2, height/2))
            .force("collision", d3.forceCollide(d =>
                                   8 + d.members.length * 3));

        const link = svg.append("g").selectAll("line")
            .data(edges).enter().append("line")
            .attr("stroke", d => d.excitatory ? "#2196F3" : "#F44336")
            .attr("stroke-width", d => Math.min(4, 1 + Math.abs(d.weight) * 3))
            .attr("stroke-opacity", 0.7)
            .attr("marker-end", `url(#arrow-${containerId})`);

        const node = svg.append("g").selectAll("g")
            .data(nodes).enter().append("g")
            .attr("cursor", "pointer")
            .call(d3.drag()
                .on("start", (ev, d) => {
                    if (!ev.active) simulation.alphaTarget(0.3).restart();
                    d.fx = d.x; d.fy = d.y;
                })
                .on("drag",  (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
                .on("end",   (ev, d) => {
                    if (!ev.active) simulation.alphaTarget(0);
                    d.fx = null; d.fy = null;
                }));

        const r = d => d.members.length > 1 ? 10 + d.members.length * 1.5 : 7;

        node.append("circle")
            .attr("r", r)
            .attr("fill", d => d.dataset_color)
            .attr("stroke", d => d.is_target ? "#000" :
                                 d.members.length > 1 ? "#fff" : "#fff")
            .attr("stroke-width", d => d.is_target ? 2.5 :
                                       d.members.length > 1 ? 2 : 1)
            .attr("stroke-dasharray", d =>
                d.members.length > 1 ? "4,2" : "none");

        node.append("text")
            .attr("dy", "0.3em")
            .attr("text-anchor", "middle")
            .style("font-size", d => d.members.length > 1 ? "7px" : "6px")
            .style("fill", "white")
            .style("pointer-events", "none")
            .text(d => d.members.length > 1
                       ? `x${d.members.length}`
                       : `F${d.id}`);

        node.append("title")
            .text(d => d.members.length > 1
                ? `Supernode: F${d.members.join(", F")}  (${d.members.length} features)\\nstr=${d.strength.toFixed(2)}`
                : `F${d.id}  str=${d.strength.toFixed(2)}`);

        simulation.on("tick", () => {
            link
                .attr("x1", d => d.source.x).attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
            node.attr("transform", d => `translate(${d.x},${d.y})`);
        });
    }

    const svgElements = [];

    componentData.forEach((graphData, imageNum) => {
        container.append("div")
            .style("font-size", "0.75em")
            .style("color", "#666")
            .style("margin-bottom", "4px")
            .text(`Image ${graphData.image_idx} - ${graphData.nodes.length} features`);

        const svgEl = container.append("svg")
            .attr("width", 360).attr("height", 240)
            .style("background", "#fafafa")
            .style("border", "1px solid #e6e6e6")
            .style("border-radius", "4px")
            .style("margin-bottom", "12px")
            .node();

        svgElements.push({ svgEl, graphData });
        renderGraph(graphData, svgEl, 1.01);
    });

    slider.on("input", function() {
        const raw = +this.value;
        if (raw > 100) {
            sliderVal.text("OFF");
            svgElements.forEach(({svgEl, graphData}) =>
                renderGraph(graphData, svgEl, 1.01));
        } else {
            const threshold = raw / 100;
            sliderVal.text(threshold.toFixed(2));
            svgElements.forEach(({svgEl, graphData}) =>
                renderGraph(graphData, svgEl, threshold));
        }
    });
}

componentMap["attributionMiniGraph"] = setupAttributionMiniGraph;
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
    verbose: bool = True,
    activations: np.ndarray | None = None,
    view_labels: dict[str, str] | None = None,
) -> None:
    coact_sub = compute_coactivation_matrix(stats, top_k=coactivation_top_k, feature_indices=feature_indices).cpu().numpy()
    coactivation_full = np.zeros((sae.d_sae, sae.d_sae))
    for ii, fi in enumerate(feature_indices):
        for jj, fj in enumerate(feature_indices):
            coactivation_full[fi, fj] = coact_sub[ii, jj]

    W_dec_sub = sae.W_dec[feature_indices].cpu().float()
    norms = W_dec_sub.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normed = W_dec_sub / norms
    cossim_sub = (normed @ normed.T).detach().numpy()
    np.fill_diagonal(cossim_sub, 0.0)
    
    cossim_full = np.zeros((sae.d_sae, sae.d_sae))
    for ii, fi in enumerate(feature_indices):
        for jj, fj in enumerate(feature_indices):
            cossim_full[fi, fj] = cossim_sub[ii, jj]

    DATA = {}
    n_p = spatial_h * spatial_w
    for feat_idx in tqdm(feature_indices, desc="Building dashboard", disable=not verbose):
        DATA[str(feat_idx)] = build_data_for_feature(
            feature_idx=feat_idx,
            stats=stats,
            coactivation_matrix=coactivation_full,
            decoder_cossim_matrix=cossim_full,
            images_dir=images_dir,
            spatial_h=spatial_h,
            spatial_w=spatial_w,
            context_patches=context_patches,
            top_k_patches=top_k_patches,
            n_coactivation_rows=n_coactivation_rows,
            sae=sae if activations is not None else None,
            activations=activations,
            n_patches=n_p,
            view_labels=view_labels,
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
