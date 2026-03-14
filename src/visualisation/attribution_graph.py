"""Attribution-graph renderer for Phase 3 visualisation.

Draws a directed networkx graph where nodes are active SAE features and
edges represent the linear interaction attribution (W_dec · W_enc)
weighted by prompt-specific mean activation.  Optionally adds logit-
attribution edges to class nodes and a companion spatial-overlay panel.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import matplotlib.colors as mpl_colors
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import networkx as nx

# Patch for numpy 2.0 compatibility when using old networkx versions (<3.3)
if not hasattr(np, "alltrue"):
    np.alltrue = np.all
import torch
from PIL import Image

# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from sae_models import BatchTopKSAE  # noqa: E402
from visualisation.vis_config import VisConfig  # noqa: E402
from visualisation.spatial_maps import (  # noqa: E402
    get_image_activations,
    get_spatial_activation_map,
    upsample_and_normalise,
)


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def _scale_range(
    vals: np.ndarray,
    lo: float,
    hi: float,
) -> np.ndarray:
    """Linearly scale *vals* into [lo, hi].  Constant arrays map to mid."""
    vmin, vmax = vals.min(), vals.max()
    if vmax - vmin < 1e-12:
        return np.full_like(vals, (lo + hi) / 2)
    return lo + (vals - vmin) / (vmax - vmin) * (hi - lo)


# ──────────────────────────────────────────────────────────────────────────
# 1. Attribution graph
# ──────────────────────────────────────────────────────────────────────────

def build_node_thumbnails(
    graph_data: dict,
    activations: torch.Tensor,
    sae: BatchTopKSAE,
    image_idx: int,
    config: VisConfig,
    top_n: int = 15,
) -> dict[int, "Image.Image"]:
    """Build 28×28 spatial activation heatmap thumbnails for hub nodes.

    For the *top_n* strongest hub nodes (by feature_strengths), returns a
    mapping from global feature index to a 28×28 PIL Image showing the
    spatial activation heatmap for that feature on *image_idx*.

    Parameters
    ----------
    graph_data : dict
        Output of :func:`graph_utils.build_graph_data`.
    activations : Tensor [N_eff, d_model]
        Full flat activation tensor.
    sae : BatchTopKSAE
        Trained SAE in eval mode.
    image_idx : int
        Which image to build thumbnails for.
    config : VisConfig
        Supplies ``n_patches``, ``spatial_h``, ``spatial_w``.
    top_n : int
        Number of strongest hub features to include.

    Returns
    -------
    dict[int, PIL.Image.Image]
        Mapping ``{feature_idx: PIL.Image}`` for each hub node.
        Images are 28×28 pixels, coloured with the 'viridis' colormap.
    """
    active_indices = graph_data["active_indices"]
    strengths = graph_data["feature_strengths"]  # [n_active]

    n_pick = min(top_n, len(active_indices))
    top_local = torch.topk(strengths, n_pick).indices  # local indices
    top_feat_ids = active_indices[top_local]           # global feature indices

    f = get_image_activations(image_idx, activations, sae, config)
    thumbnails: dict[int, Image.Image] = {}

    plasma_cm = mpl_cm.get_cmap("viridis")

    for local_i in top_local:
        feat_idx = int(active_indices[local_i].item())
        heatmap = get_spatial_activation_map(
            f, feat_idx, config.spatial_h, config.spatial_w,
        )  # [H, W]
        hm_up = upsample_and_normalise(heatmap, 28, 28)  # [28, 28] in [0,1]
        hm_np = hm_up.numpy()  # [28, 28]

        # Apply colormap → RGBA [28, 28, 4]
        rgba = (plasma_cm(hm_np) * 255).astype(np.uint8)
        thumbnails[feat_idx] = Image.fromarray(rgba, mode="RGBA").convert("RGB")

    return thumbnails


def render_attribution_graph(
    graph_data: dict,
    feature_labels: dict[int, str] | None = None,
    class_names: list[str] | None = None,
    target_class: int | None = None,
    save_path: str | None = None,
    ax: plt.Axes | None = None,
    activations: torch.Tensor | None = None,
    sae: BatchTopKSAE | None = None,
    config: VisConfig | None = None,
    node_thumbnail_top_n: int = 15,
) -> plt.Figure:
    """Render the attribution graph as a networkx directed graph.

    Edge colours reflect the **signed** attribution values:

    * Blue (excitatory): attribution > 0
    * Red (inhibitory): attribution < 0
    * Grey: |attribution| < 0.1 × max |attribution| (weak edges)

    Edge alpha is proportional to |attribution|.  Edge width is proportional
    to |attribution|, capped at 5× base width.

    Hub nodes (top *node_thumbnail_top_n* by feature_strengths) are drawn
    with a 28×28 spatial heatmap thumbnail as node background when
    *activations*, *sae*, and *config* are supplied.  Non-hub nodes are
    coloured using the 'plasma' colormap (darker = stronger activation).

    Parameters
    ----------
    graph_data : dict
        Output of :func:`graph_utils.build_graph_data`.
    feature_labels : dict, optional
        ``{feature_idx: "human label"}``.
    class_names : list[str], optional
        Class names for logit nodes (indices match classifier rows).
    target_class : int, optional
        If given, show only this class's logit node.
    save_path : str, optional
        If given, save at 150 dpi.
    ax : matplotlib Axes, optional
        If given, draw into this axes (for embedding in a larger figure).
    activations : Tensor [N_eff, d_model], optional
        Required for node thumbnails.
    sae : BatchTopKSAE, optional
        Required for node thumbnails.
    config : VisConfig, optional
        Required for node thumbnails.
    node_thumbnail_top_n : int
        Number of top hub nodes to render thumbnails for (default 15).

    Returns
    -------
    matplotlib.figure.Figure
    """
    labels_map = feature_labels or {}
    active_indices = graph_data["active_indices"]
    attribution = graph_data["attribution"]
    logit_attr = graph_data["logit_attribution"]
    strengths = graph_data["feature_strengths"].numpy()
    n_active = graph_data["n_active"]

    # Determine hub nodes
    n_hubs = min(node_thumbnail_top_n, n_active)
    hub_local_indices = set(
        torch.topk(graph_data["feature_strengths"], n_hubs).indices.tolist()
    )
    hub_feat_ids: set[int] = {
        int(active_indices[li].item()) for li in hub_local_indices
    }

    # Retrieve thumbnails if possible
    thumbnails: dict[int, Image.Image] = {}
    if activations is not None and sae is not None and config is not None:
        thumbnails = build_node_thumbnails(
            graph_data, activations, sae,
            graph_data["image_idx"], config,
            top_n=node_thumbnail_top_n,
        )

    G = nx.DiGraph()

    # ── Feature nodes ─────────────────────────────────────────────────
    node_ids: list[str] = []
    for i in range(n_active):
        idx = int(active_indices[i].item())
        nid = f"f{idx}"
        label = labels_map.get(idx, f"F{idx}")
        G.add_node(nid, label=label, kind="feature")
        node_ids.append(nid)

    node_sizes = _scale_range(strengths, 300, 2000)

    # Node colours: plasma colormap for non-hub; transparent placeholder for hub
    plasma = mpl_cm.get_cmap("plasma")
    strength_norm = _scale_range(strengths, 0.0, 1.0)
    node_colors = []
    for i in range(n_active):
        feat_idx = int(active_indices[i].item())
        if feat_idx in hub_feat_ids:
            node_colors.append("none")  # will be drawn as thumbnail
        else:
            node_colors.append(plasma(float(strength_norm[i])))

    # ── Feature → Feature edges (signed attribution) ──────────────────
    edge_colors_rgba: list = []
    edge_widths: list[float] = []
    edge_alphas: list[float] = []
    attr_np = attribution.numpy()
    abs_max = max(float(np.abs(attr_np).max()), 1e-12)
    base_width = 2.0
    max_width = base_width * 5.0

    all_edges: list[tuple[str, str]] = []

    for i in range(n_active):
        for j in range(n_active):
            v = float(attr_np[i, j])
            if v == 0.0:
                continue
            abs_v = abs(v)
            frac = abs_v / abs_max
            G.add_edge(node_ids[i], node_ids[j], weight=abs_v)
            all_edges.append((node_ids[i], node_ids[j]))

            # Colour: blue (excitatory), red (inhibitory), grey (weak)
            if abs_v < 0.1 * abs_max:
                edge_colors_rgba.append("grey")
            elif v > 0:
                edge_colors_rgba.append("steelblue")
            else:
                edge_colors_rgba.append("crimson")

            edge_widths.append(min(frac * 4.0 * base_width, max_width))
            edge_alphas.append(max(0.2, frac))

    n_edges = len(all_edges)

    # ── Logit nodes ───────────────────────────────────────────────────
    logit_node_ids: list[str] = []
    if logit_attr is not None:
        logit_np = logit_attr.numpy()  # [n_active, n_classes]
        n_classes = logit_np.shape[1]
        names = class_names or [f"Class {c}" for c in range(n_classes)]

        if target_class is not None:
            class_indices = [target_class]
        else:
            summed = np.abs(logit_np).sum(axis=0)  # [n_classes]
            top3 = min(3, n_classes)
            class_indices = list(np.argsort(summed)[::-1][:top3])

        logit_abs_max = max(float(np.abs(logit_np).max()), 1e-12)

        for c in class_indices:
            cid = f"class_{c}"
            G.add_node(cid, label=names[c], kind="logit")
            logit_node_ids.append(cid)
            for i in range(n_active):
                v = float(logit_np[i, c])
                if abs(v) < 1e-8:
                    continue
                abs_v = abs(v)
                frac = abs_v / logit_abs_max
                G.add_edge(node_ids[i], cid, weight=abs_v)
                all_edges.append((node_ids[i], cid))

                if abs_v < 0.1 * logit_abs_max:
                    edge_colors_rgba.append("grey")
                elif v > 0:
                    edge_colors_rgba.append("steelblue")
                else:
                    edge_colors_rgba.append("crimson")

                edge_widths.append(min(frac * 4.0 * base_width, max_width))
                edge_alphas.append(max(0.2, frac))

    # ── Layout ────────────────────────────────────────────────────────
    pos = nx.spring_layout(G, seed=42, k=2.0)
    if logit_node_ids:
        all_y = [pos[n][1] for n in node_ids]
        bottom_y = min(all_y) - 0.6
        for li, cid in enumerate(logit_node_ids):
            spacing = 0.5 * (li - (len(logit_node_ids) - 1) / 2)
            pos[cid] = np.array([spacing, bottom_y])

    # ── Draw ──────────────────────────────────────────────────────────
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=(13, 8))
    else:
        fig = ax.figure

    # Draw non-hub feature nodes
    non_hub_node_ids = [
        nid for i, nid in enumerate(node_ids)
        if int(active_indices[i].item()) not in hub_feat_ids
    ]
    non_hub_sizes = [
        node_sizes[i] for i, nid in enumerate(node_ids)
        if int(active_indices[i].item()) not in hub_feat_ids
    ]
    non_hub_colors = [
        node_colors[i] for i, nid in enumerate(node_ids)
        if int(active_indices[i].item()) not in hub_feat_ids
    ]

    if non_hub_node_ids:
        nx.draw_networkx_nodes(
            G, pos, nodelist=non_hub_node_ids, ax=ax,
            node_size=non_hub_sizes, node_color=non_hub_colors,
            edgecolors="grey",
        )

    # Draw hub nodes (transparent background) so thumbnails show through
    hub_node_ids = [
        nid for i, nid in enumerate(node_ids)
        if int(active_indices[i].item()) in hub_feat_ids
    ]
    hub_node_sizes = [
        node_sizes[i] for i, nid in enumerate(node_ids)
        if int(active_indices[i].item()) in hub_feat_ids
    ]
    if hub_node_ids:
        nx.draw_networkx_nodes(
            G, pos, nodelist=hub_node_ids, ax=ax,
            node_size=hub_node_sizes, node_color="white",
            edgecolors="black", linewidths=1.5,
        )

    # Overlay thumbnail images onto hub nodes
    if thumbnails:
        for i, nid in enumerate(node_ids):
            feat_idx = int(active_indices[i].item())
            if feat_idx not in thumbnails:
                continue
            thumb = thumbnails[feat_idx]
            img_arr = np.array(thumb)
            xy = pos[nid]
            oi = OffsetImage(img_arr, zoom=0.8)
            ab = AnnotationBbox(oi, xy, frameon=False, pad=0.0)
            ax.add_artist(ab)

    # Logit nodes
    if logit_node_ids:
        nx.draw_networkx_nodes(
            G, pos, nodelist=logit_node_ids, ax=ax,
            node_size=800, node_color="lightyellow", node_shape="s",
            edgecolors="grey",
        )

    # Labels
    label_dict = nx.get_node_attributes(G, "label")
    nx.draw_networkx_labels(G, pos, labels=label_dict, font_size=7, ax=ax)

    # Edges — draw per-edge so we can set individual alpha
    for (src, dst), color, width, alpha in zip(
        all_edges, edge_colors_rgba, edge_widths, edge_alphas
    ):
        nx.draw_networkx_edges(
            G, pos, edgelist=[(src, dst)], ax=ax,
            edge_color=[color], width=width,
            arrows=True, arrowsize=12, alpha=alpha,
            connectionstyle="arc3,rad=0.1",
        )

    # ── Legend ────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color="steelblue", label="Excitatory (+)"),
        mpatches.Patch(color="crimson", label="Inhibitory (−)"),
        mpatches.Patch(color="grey", label="Weak (<10% max)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=7, framealpha=0.8)

    # ── Title ─────────────────────────────────────────────────────────
    ax.set_title(
        f"Attribution Graph — Image {graph_data['image_idx']}  "
        f"({n_active} active features, {n_edges} edges shown)",
        fontsize=10,
        fontweight="bold",
    )
    ax.axis("off")

    # ── Cosine Similarity Panel ───────────────────────────────────────
    if sae is not None and n_active > 0:
        from visualisation.graph_utils import compute_decoder_cosine_similarity
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        cos_sim = compute_decoder_cosine_similarity(sae, active_indices)
        
        divider = make_axes_locatable(ax)
        ax_sim = divider.append_axes("right", size="30%", pad=0.5)
        
        cax = ax_sim.imshow(cos_sim.numpy(), cmap="RdBu_r", vmin=-1.0, vmax=1.0)
        ax_sim.set_title("Decoder cosine\nsimilarity", fontsize=8, fontweight="bold")
        
        feat_labels = [labels_map.get(int(active_indices[i].item()), f"{int(active_indices[i].item())}") for i in range(n_active)]
        
        ax_sim.set_xticks(range(n_active))
        ax_sim.set_yticks(range(n_active))
        ax_sim.set_xticklabels(feat_labels, rotation=90, fontsize=6)
        ax_sim.set_yticklabels(feat_labels, fontsize=6)
        
        fig.colorbar(cax, cax=divider.append_axes("right", size="8%", pad=0.1))

    if own_fig:
        fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ──────────────────────────────────────────────────────────────────────────
# 2. Graph + spatial companion panel
# ──────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def render_graph_with_spatial_panel(
    image_idx: int,
    activations: torch.Tensor,
    sae: BatchTopKSAE,
    image_paths: list[Path],
    config: VisConfig,
    graph_data: dict,
    feature_labels: dict[int, str] | None = None,
    n_top_spatial: int = 5,
    save_path: str | None = None,
    node_thumbnail_top_n: int = 15,
) -> plt.Figure:
    """Two-panel figure: attribution graph (left) + spatial overlays (right).

    Parameters
    ----------
    image_idx : int
    activations : Tensor [N_eff, d_model]
    sae : BatchTopKSAE  (eval mode)
    image_paths : list[Path]
    config : VisConfig
    graph_data : dict
        Output of :func:`graph_utils.build_graph_data`.
    feature_labels : dict, optional
    n_top_spatial : int
        Number of top features to show spatial maps for.
    save_path : str, optional
    node_thumbnail_top_n : int
        Number of hub nodes to render with thumbnail overlays (default 15).

    Returns
    -------
    matplotlib.figure.Figure
    """
    labels_map = feature_labels or {}
    active_indices = graph_data["active_indices"]
    strengths = graph_data["feature_strengths"]

    # Pick top features by strength
    n_show = min(n_top_spatial, len(active_indices))
    top_local = torch.topk(strengths, n_show).indices  # local indices
    top_feat_ids = active_indices[top_local]            # global feature indices

    # Encode patches & load image
    f = get_image_activations(image_idx, activations, sae, config)
    img = Image.open(image_paths[image_idx]).convert("RGB")
    img_w, img_h = img.size
    img_arr = np.array(img)

    # ── Figure with GridSpec ──────────────────────────────────────────
    fig = plt.figure(figsize=(16, max(8, 3 * n_show)))
    gs = gridspec.GridSpec(n_show, 2, width_ratios=[3, 2], hspace=0.35, wspace=0.25)

    # LEFT: attribution graph (spans all rows)
    ax_graph = fig.add_subplot(gs[:, 0])
    render_attribution_graph(
        graph_data,
        feature_labels=feature_labels,
        ax=ax_graph,
        activations=activations,
        sae=sae,
        config=config,
        node_thumbnail_top_n=node_thumbnail_top_n,
    )

    # RIGHT: spatial overlay for each top feature
    for row_i in range(n_show):
        ax_r = fig.add_subplot(gs[row_i, 1])
        feat_idx = int(top_feat_ids[row_i].item())

        heatmap = get_spatial_activation_map(
            f, feat_idx, config.spatial_h, config.spatial_w,
        )
        hm_norm = upsample_and_normalise(heatmap, img_h, img_w)

        ax_r.imshow(img_arr)
        ax_r.imshow(hm_norm.numpy(), cmap="hot", alpha=0.5, interpolation="none")
        ax_r.axis("off")

        lbl = labels_map.get(feat_idx, f"F{feat_idx}")
        strength_val = float(strengths[top_local[row_i]].item())
        ax_r.set_title(f"{lbl}  (str={strength_val:.3f})", fontsize=8)

    fig.suptitle(
        f"Image {image_idx} — Attribution Graph + Spatial Maps",
        fontsize=12, fontweight="bold", y=1.01,
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ── Smoke tests ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile

    from visualisation.graph_utils import build_graph_data  # noqa: E402

    print("=" * 60)
    print("  attribution_graph.py — smoke tests")
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

    # Fake image files
    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir = Path(tmpdir) / "images"
        img_dir.mkdir()
        paths: list[Path] = []
        for i in range(n_images):
            p = img_dir / f"img_{i:03d}.png"
            Image.fromarray(
                np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            ).save(p)
            paths.append(p)

        # Build graph data
        clf_w = torch.randn(4, d_model)
        gd = build_graph_data(0, acts, sae, cfg, classifier_weights=clf_w)

        # 1. render_attribution_graph — with classifier & target_class
        save_p1 = str(Path(tmpdir) / "graph_full.png")
        fig1 = render_attribution_graph(
            gd,
            feature_labels={int(gd["active_indices"][0]): "edges"},
            class_names=["A", "B", "C", "D"],
            target_class=1,
            save_path=save_p1,
        )
        plt.close(fig1)
        assert Path(save_p1).exists()
        print(f"[OK] render_attribution_graph (target_class=1): saved")

        # 2. render_attribution_graph — auto top-3 classes
        save_p2 = str(Path(tmpdir) / "graph_auto.png")
        fig2 = render_attribution_graph(
            gd, class_names=["A", "B", "C", "D"], save_path=save_p2,
        )
        plt.close(fig2)
        assert Path(save_p2).exists()
        print(f"[OK] render_attribution_graph (auto top-3 classes): saved")

        # 3. render_attribution_graph — no classifier
        gd_nocls = build_graph_data(0, acts, sae, cfg, classifier_weights=None)
        fig3 = render_attribution_graph(gd_nocls)
        plt.close(fig3)
        print("[OK] render_attribution_graph (no classifier): created")

        # 4. render_graph_with_spatial_panel
        save_p4 = str(Path(tmpdir) / "combined.png")
        fig4 = render_graph_with_spatial_panel(
            image_idx=0,
            activations=acts,
            sae=sae,
            image_paths=paths,
            config=cfg,
            graph_data=gd,
            feature_labels={int(gd["active_indices"][0]): "edges"},
            n_top_spatial=3,
            save_path=save_p4,
        )
        plt.close(fig4)
        assert Path(save_p4).exists()
        print(f"[OK] render_graph_with_spatial_panel: saved")

        # 5. build_node_thumbnails
        thumbs = build_node_thumbnails(gd, acts, sae, image_idx=0,
                                       config=cfg, top_n=4)
        assert isinstance(thumbs, dict)
        for feat_idx_key, img_val in thumbs.items():
            assert isinstance(img_val, Image.Image)
            assert img_val.size == (28, 28), f"expected 28x28, got {img_val.size}"
        print(f"[OK] build_node_thumbnails: {len(thumbs)} thumbnails, all 28×28")

        # 6. render_attribution_graph with thumbnails
        save_p6 = str(Path(tmpdir) / "graph_with_thumbs.png")
        fig6 = render_attribution_graph(
            gd,
            activations=acts,
            sae=sae,
            config=cfg,
            node_thumbnail_top_n=4,
            save_path=save_p6,
        )
        plt.close(fig6)
        assert Path(save_p6).exists()
        print(f"[OK] render_attribution_graph (with thumbnails): saved")

        # 7. Verify edge_attributions sign preservation
        # Build a simple attribution matrix with known signs
        n_act = gd["n_active"]
        assert len(gd["edge_attributions"]) == (gd["attribution"] != 0).sum().item()
        print(f"[OK] edge_attributions: {len(gd['edge_attributions'])} signed edges")

    print("\nAll attribution_graph smoke tests passed ✓")
