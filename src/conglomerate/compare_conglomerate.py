"""Compare the conglomerate SAE against all individual per-dataset SAEs.

For each individual SAE (colon, brain, breast, pneumonia) computes:
  1. Cross-SAE cosine similarity matrix  (conglomerate vs individual)
  2. Reciprocal best matches
  3. Summary statistics per dataset pair

Then produces:
  - similarity_conglomerate_vs_<dataset>.csv   (all pairs above threshold)
  - reciprocal_conglomerate_vs_<dataset>.csv   (reciprocal best matches)
  - summary.json                               (per-pair stats)
  - umap_conglomerate_vs_all.png               (UMAP of all decoder weights)
  - monosemanticity_conglomerate.csv           (within-conglomerate geometry)

This is the primary analysis tool for studying whether the conglomerate SAE
learns features that generalise across medical-imaging domains.

Usage
-----
  python src/conglomerate/compare_conglomerate.py \\
      --conglomerate_checkpoint  data/conglomerate/checkpoints/step_0200000 \\
      --individual  data/colon/checkpoints/step_0200000:colon \\
      --individual  data/brain/checkpoints/step_0200000:brain_tumor \\
      --individual  data/breast/checkpoints/step_0200000:tcga_breast \\
      --individual  data/pneumonia/checkpoints/step_0200000:pneumonia \\
      --output_dir  data/conglomerate/comparison \\
      --threshold   0.7 \\
      --device      cpu
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
import torch.nn.functional as F
from tqdm import tqdm

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sae_models import BatchTopKSAE            # noqa: E402
from visualisation.loaders import load_sae     # noqa: E402


# ── Cosine similarity ──────────────────────────────────────────────────────────

@torch.no_grad()
def decoder_cosine_sim(sae_a: BatchTopKSAE, sae_b: BatchTopKSAE) -> torch.Tensor:
    """Return [d_sae_a, d_sae_b] cosine similarity between decoder rows."""
    assert sae_a.d_model == sae_b.d_model, (
        f"d_model mismatch: {sae_a.d_model} vs {sae_b.d_model}"
    )
    W_a = F.normalize(sae_a.W_dec.data.float(), dim=1)
    W_b = F.normalize(sae_b.W_dec.data.float(), dim=1)
    return W_a @ W_b.T


# ── Reciprocal best matches ────────────────────────────────────────────────────

@torch.no_grad()
def reciprocal_best_matches(
    sim: torch.Tensor,
    min_sim: float = 0.0,
) -> list[tuple[int, int, float]]:
    best_b_for_a = sim.argmax(dim=1)
    best_a_for_b = sim.argmax(dim=0)
    pairs = []
    for a in range(sim.shape[0]):
        b = int(best_b_for_a[a].item())
        if int(best_a_for_b[b].item()) == a:
            s = float(sim[a, b].item())
            if s >= min_sim:
                pairs.append((a, b, s))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


# ── Monosemanticity ────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_monosemanticity(sae: BatchTopKSAE) -> torch.Tensor:
    """1 - mean pairwise cosine similarity with other features. Higher = better."""
    W = F.normalize(sae.W_dec.data.float(), dim=1)
    sim = W @ W.T
    sim.fill_diagonal_(0.0)
    mean_sim = sim.sum(dim=1) / (sim.shape[0] - 1)
    return (1.0 - mean_sim.clamp(min=0.0)).cpu()


# ── CSV helpers ────────────────────────────────────────────────────────────────

def write_similarity_csv(
    sim: torch.Tensor,
    path: Path,
    label_a: str,
    label_b: str,
    threshold: float,
) -> list[tuple[int, int, float]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = sim >= threshold
    idxs_a, idxs_b = torch.where(mask)
    sims = sim[mask]
    order = sims.argsort(descending=True)
    idxs_a = idxs_a[order].cpu().numpy()
    idxs_b = idxs_b[order].cpu().numpy()
    sims   = sims[order].cpu().numpy()

    rows = []
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([f"{label_a}_feature_idx", f"{label_b}_feature_idx", "cosine_similarity"])
        for a, b, s in zip(idxs_a, idxs_b, sims):
            w.writerow([int(a), int(b), f"{s:.6f}"])
            rows.append((int(a), int(b), float(s)))
    print(f"  Wrote {len(rows)} pairs (>={threshold:.2f}) → {path}")
    return rows


def write_reciprocal_csv(
    pairs: list[tuple[int, int, float]],
    path: Path,
    label_a: str,
    label_b: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([f"{label_a}_feature_idx", f"{label_b}_feature_idx",
                    "cosine_similarity", "match_type"])
        for a, b, s in pairs:
            w.writerow([a, b, f"{s:.6f}", "reciprocal_best"])
    print(f"  Wrote {len(pairs)} reciprocal matches → {path}")


def write_monosemanticity_csv(scores: torch.Tensor, label: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["feature_idx", "dataset", "monosemanticity"])
        for i, v in enumerate(scores.tolist()):
            w.writerow([i, label, f"{v:.6f}"])
    n_above = (scores > 0.7).sum().item()
    print(f"  Monosemanticity ({label}): mean={scores.mean():.4f}  "
          f"{n_above}/{len(scores)} above 0.7 → {path}")


# ── UMAP ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def render_umap_all(
    cong_sae: BatchTopKSAE,
    individual_saes: list[tuple[str, BatchTopKSAE]],
    save_path: Path,
) -> None:
    try:
        import umap
    except ImportError:
        print("  [skip] umap-learn not installed — skipping UMAP plot.")
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)

    W_cong = F.normalize(cong_sae.W_dec.data.float(), dim=1).cpu().numpy()
    all_W  = [W_cong]
    labels = ["conglomerate"] * W_cong.shape[0]
    colors = ["#1A237E"]  # dark blue for conglomerate

    palette = ["#F44336", "#4CAF50", "#FF9800", "#9C27B0"]
    for (name, sae), color in zip(individual_saes, palette):
        W = F.normalize(sae.W_dec.data.float(), dim=1).cpu().numpy()
        all_W.append(W)
        labels += [name] * W.shape[0]
        colors.append(color)

    combined = np.vstack(all_W)
    print(f"  Fitting UMAP on {combined.shape[0]} decoder directions …")

    reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1, metric="cosine",
        n_components=2, random_state=42,
    )
    emb = reducer.fit_transform(combined)

    fig, ax = plt.subplots(figsize=(12, 9))
    ptr = 0
    for i, (name, W) in enumerate([("conglomerate", W_cong)] +
                                   [(n, None) for n, _ in individual_saes]):
        n = all_W[i].shape[0]
        ax.scatter(
            emb[ptr:ptr+n, 0], emb[ptr:ptr+n, 1],
            s=2.0, c=colors[i], alpha=0.4,
            label=name if i == 0 else name, rasterized=True,
        )
        ptr += n

    ax.legend(fontsize=11, markerscale=6)
    ax.set_title("UMAP — Conglomerate SAE vs Individual SAEs", fontsize=13)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  UMAP saved → {save_path}")


# ── Histogram of max-similarity distribution ──────────────────────────────────

def render_sim_histogram(
    sim: torch.Tensor,
    label_a: str,
    label_b: str,
    save_path: Path,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    max_sims = sim.max(dim=1).values.cpu().numpy()  # best match per conglomerate feature

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(max_sims, bins=50, color="#2196F3", edgecolor="white", linewidth=0.4)
    ax.axvline(0.7, color="red", linestyle="--", linewidth=1, label="threshold=0.7")
    n_above = (max_sims >= 0.7).sum()
    ax.set_title(
        f"{label_a} vs {label_b}  |  "
        f"{n_above}/{len(max_sims)} features with best-match >= 0.7",
        fontsize=11,
    )
    ax.set_xlabel("Max cosine similarity to individual SAE feature")
    ax.set_ylabel("Number of conglomerate features")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Histogram saved → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Compare conglomerate SAE vs individual per-dataset SAEs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--conglomerate_checkpoint", required=True,
                    help="Path to conglomerate SAE checkpoint directory.")
    ap.add_argument("--individual", action="append", metavar="PATH:LABEL", required=True,
                    help="Individual SAE checkpoint and label, e.g. "
                         "data/colon/checkpoints/step_0200000:colon.  "
                         "Repeat for each individual SAE.")
    ap.add_argument("--output_dir",  required=True)
    ap.add_argument("--threshold",   type=float, default=0.7,
                    help="Cosine similarity threshold for the CSV export.")
    ap.add_argument("--skip_umap",   action="store_true")
    ap.add_argument("--device",      type=str, default="cpu")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Parse individual SAEs
    individuals: list[tuple[str, str]] = []
    for entry in args.individual:
        if ":" not in entry:
            ap.error(f"--individual must be PATH:LABEL, got: {entry!r}")
        path, label = entry.rsplit(":", 1)
        individuals.append((path, label))

    print("=" * 60)
    print("Conglomerate vs Individual SAE Comparison")
    print("=" * 60)
    print(f"  Conglomerate : {args.conglomerate_checkpoint}")
    print(f"  Individuals  : {[l for _, l in individuals]}")
    print(f"  Output       : {out}")
    print(f"  Threshold    : {args.threshold}")
    print()

    # ── Load SAEs ─────────────────────────────────────────────────────────
    print(f"Loading conglomerate SAE …")
    cong_sae = load_sae(args.conglomerate_checkpoint, device=args.device)
    print(f"  d_model={cong_sae.d_model}  d_sae={cong_sae.d_sae}  k={cong_sae.k}")

    individual_saes: list[tuple[str, BatchTopKSAE]] = []
    for ckpt_path, label in individuals:
        print(f"Loading individual SAE ({label}) …")
        sae = load_sae(ckpt_path, device=args.device)
        print(f"  d_model={sae.d_model}  d_sae={sae.d_sae}  k={sae.k}")
        individual_saes.append((label, sae))

    # ── Monosemanticity of conglomerate ───────────────────────────────────
    print("\n" + "="*60)
    print("STEP 1: Conglomerate monosemanticity")
    print("="*60)
    mono_cong = compute_monosemanticity(cong_sae)
    write_monosemanticity_csv(mono_cong, "conglomerate",
                               out / "monosemanticity_conglomerate.csv")

    # ── Pairwise comparison ───────────────────────────────────────────────
    summary: list[dict] = []

    for label, ind_sae in individual_saes:
        print(f"\n{'='*60}")
        print(f"STEP 2+: Conglomerate vs {label}")
        print("="*60)

        sim = decoder_cosine_sim(cong_sae, ind_sae)
        print(f"  Sim matrix shape : {tuple(sim.shape)}")
        print(f"  Max similarity   : {sim.max().item():.4f}")
        print(f"  Mean max-sim     : {sim.max(dim=1).values.mean().item():.4f}")
        n_above = (sim >= args.threshold).sum().item()
        print(f"  Pairs >= {args.threshold}     : {n_above}")

        # Threshold-based CSV
        matched = write_similarity_csv(
            sim, out / f"similarity_conglomerate_vs_{label}.csv",
            "conglomerate", label, args.threshold,
        )

        # Reciprocal best matches
        recip = reciprocal_best_matches(sim, min_sim=0.3)
        write_reciprocal_csv(
            recip, out / f"reciprocal_conglomerate_vs_{label}.csv",
            "conglomerate", label,
        )
        strong_recip = [p for p in recip if p[2] >= args.threshold]
        print(f"  Strong reciprocals (>={args.threshold}): {len(strong_recip)}")

        # Histogram
        render_sim_histogram(
            sim, "conglomerate", label,
            out / f"histogram_conglomerate_vs_{label}.png",
        )

        summary.append({
            "individual_dataset": label,
            "conglomerate_d_sae": cong_sae.d_sae,
            "individual_d_sae":   ind_sae.d_sae,
            "max_similarity":     float(sim.max().item()),
            "mean_max_sim":       float(sim.max(dim=1).values.mean().item()),
            "n_pairs_above_threshold": n_above,
            "threshold": args.threshold,
            "n_reciprocal_matches":   len(recip),
            "n_strong_reciprocals":   len(strong_recip),
        })

    # ── UMAP of all decoder weights ───────────────────────────────────────
    if not args.skip_umap:
        print("\n" + "="*60)
        print("STEP 3: UMAP of all decoder weights")
        print("="*60)
        render_umap_all(
            cong_sae, individual_saes,
            save_path=out / "umap_conglomerate_vs_all.png",
        )

    # ── Save summary JSON ─────────────────────────────────────────────────
    summary_path = out / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "conglomerate_checkpoint": str(args.conglomerate_checkpoint),
            "threshold": args.threshold,
            "conglomerate_d_sae": cong_sae.d_sae,
            "conglomerate_d_model": cong_sae.d_model,
            "per_dataset": summary,
        }, f, indent=2)
    print(f"\n  Summary → {summary_path}")

    # ── Final report ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("DONE — Outputs:")
    print("="*60)
    print(f"  {out}/")
    print(f"  ├── monosemanticity_conglomerate.csv")
    for label, _ in individual_saes:
        print(f"  ├── similarity_conglomerate_vs_{label}.csv")
        print(f"  ├── reciprocal_conglomerate_vs_{label}.csv")
        print(f"  ├── histogram_conglomerate_vs_{label}.png")
    if not args.skip_umap:
        print(f"  ├── umap_conglomerate_vs_all.png")
    print(f"  └── summary.json")


if __name__ == "__main__":
    main()
