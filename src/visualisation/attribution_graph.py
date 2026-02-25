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
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import networkx as nx
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

def render_attribution_graph(
    graph_data: dict,
    feature_labels: dict[int, str] | None = None,
    class_names: list[str] | None = None,
    target_class: int | None = None,
    save_path: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Render the attribution graph as a networkx directed graph.

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

    # ── Feature → Feature edges ───────────────────────────────────────
    edge_colors: list[str] = []
    edge_widths: list[float] = []
    attr_np = attribution.numpy()
    abs_max = max(np.abs(attr_np).max(), 1e-12)

    for i in range(n_active):
        for j in range(n_active):
            v = float(attr_np[i, j])
            if v == 0.0:
                continue
            G.add_edge(node_ids[i], node_ids[j], weight=abs(v))
            edge_colors.append("steelblue" if v > 0 else "crimson")
            edge_widths.append(abs(v) / abs_max * 4.0)

    n_edges = len(edge_colors)

    # ── Logit nodes ───────────────────────────────────────────────────
    logit_node_ids: list[str] = []
    if logit_attr is not None:
        logit_np = logit_attr.numpy()  # [n_active, n_classes]
        n_classes = logit_np.shape[1]
        names = class_names or [f"Class {c}" for c in range(n_classes)]

        if target_class is not None:
            class_indices = [target_class]
        else:
            # Top-3 by summed absolute attribution
            summed = np.abs(logit_np).sum(axis=0)  # [n_classes]
            top3 = min(3, n_classes)
            class_indices = list(np.argsort(summed)[::-1][:top3])

        logit_abs_max = max(np.abs(logit_np).max(), 1e-12)

        for c in class_indices:
            cid = f"class_{c}"
            G.add_node(cid, label=names[c], kind="logit")
            logit_node_ids.append(cid)
            for i in range(n_active):
                v = float(logit_np[i, c])
                if abs(v) < 1e-8:
                    continue
                G.add_edge(node_ids[i], cid, weight=abs(v))
                edge_colors.append("steelblue" if v > 0 else "crimson")
                edge_widths.append(abs(v) / logit_abs_max * 4.0)

    # ── Layout ────────────────────────────────────────────────────────
    pos = nx.spring_layout(G, seed=42, k=2.0)
    # Push logit nodes to the bottom
    if logit_node_ids:
        all_y = [pos[n][1] for n in node_ids]
        bottom_y = min(all_y) - 0.6
        for li, cid in enumerate(logit_node_ids):
            spacing = 0.5 * (li - (len(logit_node_ids) - 1) / 2)
            pos[cid] = np.array([spacing, bottom_y])

    # ── Draw ──────────────────────────────────────────────────────────
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    else:
        fig = ax.figure

    # Feature nodes
    nx.draw_networkx_nodes(
        G, pos, nodelist=node_ids, ax=ax,
        node_size=node_sizes, node_color="lightblue", edgecolors="grey",
    )
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

    # Edges
    edges = list(G.edges())
    if edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=edges, ax=ax,
            edge_color=edge_colors, width=edge_widths,
            arrows=True, arrowsize=12, alpha=0.7,
            connectionstyle="arc3,rad=0.1",
        )

    # ── Legend ────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color="steelblue", label="Excitatory (+)"),
        mpatches.Patch(color="crimson", label="Inhibitory (−)"),
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

    print("\nAll attribution_graph smoke tests passed ✓")
