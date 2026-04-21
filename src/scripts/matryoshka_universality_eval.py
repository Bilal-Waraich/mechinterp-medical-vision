"""Matryoshka SAE universality evaluation.

Implements the five-stage filtering pipeline used in the thesis to identify
universal latents in a Matryoshka sparse autoencoder trained on a conglomerate
of chest X-ray datasets.

Pipeline
--------
  Stage 0 — Scanner-Feature Audit
      Fit a per-latent logistic classifier that predicts *dataset-of-origin*
      from the latent's activation; exclude latents with AUROC > tau_scan.

  Stage 1 — Activation Frequency
      A latent is a Level-1 candidate iff its non-zero activation frequency
      exceeds freq_thr on at least min_ds datasets.

  Stage 2 — Cross-Dataset Semantic Consistency
      For each L1 candidate, compute a per-dataset centroid c_f^(d) by taking
      the mean activation pattern over the T images with the highest activation
      of f in dataset d. The feature passes Level 2 iff the minimum pairwise
      cosine similarity across datasets is >= cos_thr.

  Stage 3 — Split Feature Audit (largest scale + ancestor trace)
      For every pair (f_a, f_b) of L2-confirmed features at k_max, compute the
      Jaccard overlap of their top-T activating image sets. If J >= split_thr,
      walk down the k_schedule to find the smallest k' at which a single
      ancestor latent g satisfies J(T_g^(k'), T_f^(k_max)) >= anc_thr for BOTH
      f = f_a and f = f_b. Retain only the ancestor; drop the split children.

  Stage 4 — Prefix Stability
      For every surviving latent, compute J(T_f^(k_i), T_f^(k_{i+1})) between
      adjacent k scales. A latent is prefix-stable iff all adjacent Jaccard
      values are >= stability_thr.

Outputs
-------
  universality_summary.json     — counts at each stage
  stage0_scanner_audit.csv      — per-latent AUROC
  stage1_frequency.csv          — per-latent per-dataset activation frequency
  stage2_semantic.csv           — per-latent min pairwise cosine
  stage3_splits.csv             — detected split pairs with ancestors
  stage4_prefix_stability.csv   — per-latent per-scale Jaccard values
  confirmed_universal_latents.json — final latent indices passing all stages
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from safetensors.torch import load_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from matryoshka_sae import MatryoshkaSAE


# ── Default thresholds (match the methodology section of the thesis) ─────────
T_TOP            = 50      # size of the top activating image set per dataset
FREQ_THR         = 0.01    # 1% activation frequency
MIN_DS           = 2       # must fire on at least this many datasets
COS_THR          = 0.70    # cross-dataset cosine similarity
SPLIT_THR        = 0.50    # Jaccard for split detection at k_max
ANC_THR          = 0.40    # Jaccard for ancestor trace at smaller k
STABILITY_THR    = 0.40    # adjacent-scale Jaccard for prefix stability
AUROC_THR        = 0.80    # scanner-audit exclusion threshold
BATCH_TOKENS     = 4096    # tokens per SAE forward batch
PATCHES_PER_IMG  = 49      # 7 x 7 grid at pm_index = 2
SCANNER_SAMPLES  = 20000   # per-class cap for scanner-audit training
RNG_SEED         = 0


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_tagged(values: List[str]) -> List[Tuple[str, str]]:
    """Parse `PATH:LABEL` entries into (label, path) tuples."""
    out = []
    for v in values:
        if ":" not in v:
            raise ValueError(f"Expected PATH:LABEL, got {v!r}")
        path, label = v.rsplit(":", 1)
        out.append((label, path))
    return out


def load_dataset_activations(path: str) -> torch.Tensor:
    d = load_file(path)
    if "activations" not in d:
        raise KeyError(f"{path}: no 'activations' key in safetensors file")
    a = d["activations"]
    if a.dim() != 2:
        raise ValueError(f"{path}: expected 2-D activations, got {tuple(a.shape)}")
    return a


# ─────────────────────────────────────────────────────────────────────────────
# Encoding
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_all_scales(
    sae: MatryoshkaSAE,
    acts: torch.Tensor,
    device: torch.device,
    batch: int = BATCH_TOKENS,
) -> Dict[int, torch.Tensor]:
    """Return sparse codes for each k in sae.k_schedule.

    Output: {k_i: Tensor [N, d_sae] on CPU}.
    """
    n = acts.shape[0]
    out: Dict[int, torch.Tensor] = {
        k: torch.empty(n, sae.d_sae, dtype=torch.float32) for k in sae.k_schedule
    }
    for i in range(0, n, batch):
        end = min(i + batch, n)
        chunk = acts[i:end].to(device)
        for k in sae.k_schedule:
            out[k][i:end] = sae.encode(chunk, k=k).cpu()
    return out


def codes_to_per_image(z: torch.Tensor, patches: int = PATCHES_PER_IMG) -> torch.Tensor:
    """Reduce patch-level codes [N_tokens, d_sae] to image-level scores.

    Returns a tensor [N_images, d_sae] giving the *maximum* activation of each
    latent over the `patches` tokens belonging to the image. This is the
    standard reduction for deriving top-image sets from patch-wise SAEs.
    """
    n_tokens, d = z.shape
    if n_tokens % patches != 0:
        raise ValueError(
            f"Token count {n_tokens} not divisible by {patches} patches/image"
        )
    n_img = n_tokens // patches
    return z.view(n_img, patches, d).amax(dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = set(a.tolist()), set(b.tolist())
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def top_image_set(img_scores: np.ndarray, t: int = T_TOP) -> np.ndarray:
    """Indices of the top-t images by activation for a single latent."""
    if img_scores.size == 0:
        return np.array([], dtype=np.int64)
    t = min(t, img_scores.size)
    return np.argpartition(-img_scores, t - 1)[:t]


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


# ─────────────────────────────────────────────────────────────────────────────
# Stage 0 — Scanner-Feature Audit
# ─────────────────────────────────────────────────────────────────────────────

def scanner_audit(
    z_max: torch.Tensor,
    dataset_ids: np.ndarray,
    n_datasets: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return per-latent max one-vs-rest AUROC for predicting dataset identity."""
    d_sae = z_max.shape[1]
    aurocs = np.zeros(d_sae, dtype=np.float32)

    # Downsample each class to SCANNER_SAMPLES to keep fitting tractable.
    idxs_per_class = [np.where(dataset_ids == c)[0] for c in range(n_datasets)]
    sampled = []
    for idxs in idxs_per_class:
        if idxs.size > SCANNER_SAMPLES:
            idxs = rng.choice(idxs, size=SCANNER_SAMPLES, replace=False)
        sampled.append(idxs)
    keep = np.concatenate(sampled)
    rng.shuffle(keep)

    X = z_max[keep].numpy().astype(np.float32)
    y = dataset_ids[keep]

    # One-vs-rest AUROC per latent using the raw activation as a score.
    # (Fitting a LogReg per latent per class is overkill; the sign of a single
    # feature is already a ranker — AUROC of the raw score is equivalent to
    # the Mann-Whitney test and is what the audit needs.)
    for c in range(n_datasets):
        y_c = (y == c).astype(np.int32)
        for f in range(d_sae):
            s = X[:, f]
            if np.all(s == s[0]):
                continue
            try:
                a = roc_auc_score(y_c, s)
            except ValueError:
                continue
            a = max(a, 1.0 - a)
            if a > aurocs[f]:
                aurocs[f] = a
    return aurocs


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Activation Frequency
# ─────────────────────────────────────────────────────────────────────────────

def activation_frequency(
    z_max_per_ds: Dict[str, torch.Tensor],
) -> Dict[str, np.ndarray]:
    """Per-dataset fraction of tokens with non-zero activation per latent."""
    return {
        name: (z > 0).float().mean(dim=0).numpy()
        for name, z in z_max_per_ds.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Cross-Dataset Semantic Consistency
# ─────────────────────────────────────────────────────────────────────────────

def semantic_consistency(
    img_scores_per_ds: Dict[str, np.ndarray],
    raw_acts_per_ds: Dict[str, torch.Tensor],
    candidate_latents: List[int],
    t: int = T_TOP,
) -> Dict[int, float]:
    """Minimum pairwise cosine of per-dataset centroids, for each candidate.

    The centroid c_f^(d) is the mean *raw input activation* vector (d_model
    dim) over the top-t images of latent f in dataset d. Using raw model
    activations gives a dataset-independent semantic embedding whose cosine
    measures whether the images that drive f across datasets share content.
    """
    datasets = list(img_scores_per_ds.keys())
    out: Dict[int, float] = {}
    for f in candidate_latents:
        centroids: List[np.ndarray] = []
        for d in datasets:
            scores = img_scores_per_ds[d][:, f]
            if (scores > 0).sum() == 0:
                centroids.append(None)  # type: ignore
                continue
            top = top_image_set(scores, t)
            img_acts = raw_acts_per_ds[d].view(
                raw_acts_per_ds[d].shape[0] // PATCHES_PER_IMG,
                PATCHES_PER_IMG,
                -1,
            ).mean(dim=1).numpy()
            centroids.append(img_acts[top].mean(axis=0))
        present = [c for c in centroids if c is not None]
        if len(present) < MIN_DS:
            out[f] = 0.0
            continue
        cos_vals = [
            cosine(present[i], present[j])
            for i in range(len(present))
            for j in range(i + 1, len(present))
        ]
        out[f] = float(min(cos_vals)) if cos_vals else 0.0
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Split Feature Audit with Ancestor Trace
# ─────────────────────────────────────────────────────────────────────────────

def top_sets_per_latent(
    img_scores_per_ds: Dict[str, np.ndarray],
    offsets: Dict[str, int],
    latents: List[int],
    t: int = T_TOP,
) -> Dict[int, np.ndarray]:
    """Union across datasets of top-t images (global indices) for each latent."""
    out: Dict[int, np.ndarray] = {}
    for f in latents:
        parts = []
        for name, scores in img_scores_per_ds.items():
            s = scores[:, f]
            if (s > 0).sum() == 0:
                continue
            idx = top_image_set(s, t) + offsets[name]
            parts.append(idx)
        out[f] = np.concatenate(parts) if parts else np.array([], dtype=np.int64)
    return out


def split_audit(
    confirmed: List[int],
    img_scores_max: Dict[str, np.ndarray],
    img_scores_per_scale: Dict[int, Dict[str, np.ndarray]],
    k_schedule: List[int],
    offsets: Dict[str, int],
    split_thr: float = SPLIT_THR,
    anc_thr: float = ANC_THR,
) -> Tuple[List[Tuple[int, int, int, int]], set]:
    """Identify split pairs and trace their ancestor latent.

    Returns
    -------
    splits : list of (f_a, f_b, k_ancestor, g_ancestor)  — g == -1 if no ancestor
    to_drop : set of latent indices dropped in favour of an ancestor
    """
    k_max = k_schedule[-1]
    top_at_kmax = top_sets_per_latent(img_scores_max, offsets, confirmed)
    splits: List[Tuple[int, int, int, int]] = []
    to_drop: set = set()

    # Precompute top sets at every scale (for every confirmed latent).
    top_per_scale: Dict[int, Dict[int, np.ndarray]] = {
        k: top_sets_per_latent(img_scores_per_scale[k], offsets, confirmed)
        for k in k_schedule
    }

    n = len(confirmed)
    for i in range(n):
        f_a = confirmed[i]
        ta = top_at_kmax[f_a]
        for j in range(i + 1, n):
            f_b = confirmed[j]
            tb = top_at_kmax[f_b]
            if jaccard(ta, tb) < split_thr:
                continue
            # Ancestor trace — smallest k' with a single g covering both.
            k_anc, g_anc = -1, -1
            for k_prime in k_schedule[:-1]:  # smaller scales only
                top_prime = top_per_scale[k_prime]
                for g, tg in top_prime.items():
                    if tg.size == 0:
                        continue
                    if jaccard(tg, ta) >= anc_thr and jaccard(tg, tb) >= anc_thr:
                        k_anc, g_anc = k_prime, g
                        break
                if k_anc != -1:
                    break
            splits.append((f_a, f_b, k_anc, g_anc))
            if g_anc != -1:
                to_drop.update({f_a, f_b})
    return splits, to_drop


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — Prefix Stability
# ─────────────────────────────────────────────────────────────────────────────

def prefix_stability(
    latents: List[int],
    img_scores_per_scale: Dict[int, Dict[str, np.ndarray]],
    k_schedule: List[int],
    offsets: Dict[str, int],
    thr: float = STABILITY_THR,
) -> Dict[int, List[float]]:
    per_scale_sets = {
        k: top_sets_per_latent(img_scores_per_scale[k], offsets, latents)
        for k in k_schedule
    }
    out: Dict[int, List[float]] = {}
    for f in latents:
        vals = []
        for a, b in zip(k_schedule[:-1], k_schedule[1:]):
            vals.append(jaccard(per_scale_sets[a][f], per_scale_sets[b][f]))
        out[f] = vals
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CSV writers
# ─────────────────────────────────────────────────────────────────────────────

def write_csv(path: Path, header: List[str], rows: List[List]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--checkpoint", required=True,
                    help="Path to Matryoshka SAE .pt checkpoint")
    ap.add_argument("--activations", required=True, action="append",
                    metavar="PATH:LABEL",
                    help="Per-dataset activation safetensors, repeatable")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--t_top",         type=int,   default=T_TOP)
    ap.add_argument("--freq_thr",      type=float, default=FREQ_THR)
    ap.add_argument("--min_ds",        type=int,   default=MIN_DS)
    ap.add_argument("--cos_thr",       type=float, default=COS_THR)
    ap.add_argument("--split_thr",     type=float, default=SPLIT_THR)
    ap.add_argument("--anc_thr",       type=float, default=ANC_THR)
    ap.add_argument("--stability_thr", type=float, default=STABILITY_THR)
    ap.add_argument("--auroc_thr",     type=float, default=AUROC_THR)
    ap.add_argument("--batch_size",    type=int,   default=BATCH_TOKENS)
    args = ap.parse_args()

    rng = np.random.default_rng(RNG_SEED)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # ── Load SAE ─────────────────────────────────────────────────────────────
    print(f"[load] {args.checkpoint}")
    sae = MatryoshkaSAE.load_from_checkpoint(args.checkpoint).to(device).eval()
    k_schedule: List[int] = list(sae.k_schedule)
    k_max = k_schedule[-1]
    d_sae = sae.d_sae
    print(f"[sae] d_sae={d_sae}, k_schedule={k_schedule}")

    # ── Load activations per dataset ─────────────────────────────────────────
    tagged = parse_tagged(args.activations)
    raw_acts: Dict[str, torch.Tensor] = {}
    for label, path in tagged:
        print(f"[load] {label}: {path}")
        raw_acts[label] = load_dataset_activations(path)
    dataset_names = list(raw_acts.keys())
    print(f"[datasets] {dataset_names}")

    # Image-index offsets for global top-set bookkeeping.
    offsets: Dict[str, int] = {}
    cursor = 0
    for name in dataset_names:
        n_img = raw_acts[name].shape[0] // PATCHES_PER_IMG
        offsets[name] = cursor
        cursor += n_img

    # ── Encode at every scale ────────────────────────────────────────────────
    # Only keep token-level codes for k_max (needed for scanner audit + frequency).
    # Lower-k token codes are discarded immediately after img_per_scale is computed.
    z_max_per_ds: Dict[str, torch.Tensor] = {}
    img_per_scale: Dict[int, Dict[str, np.ndarray]] = {k: {} for k in k_schedule}
    for name in dataset_names:
        print(f"[encode] {name}")
        per_k = encode_all_scales(sae, raw_acts[name], device, batch=args.batch_size)
        for k in k_schedule:
            img_per_scale[k][name] = codes_to_per_image(per_k[k]).numpy()
        z_max_per_ds[name] = per_k[k_max]
    img_max_per_ds = img_per_scale[k_max]

    # ── Stage 0 — Scanner audit ──────────────────────────────────────────────
    print("[stage 0] scanner audit")
    dataset_ids = np.concatenate([
        np.full(z_max_per_ds[name].shape[0], i, dtype=np.int32)
        for i, name in enumerate(dataset_names)
    ])
    z_max_all = torch.cat([z_max_per_ds[n] for n in dataset_names], dim=0)
    aurocs = scanner_audit(z_max_all, dataset_ids, len(dataset_names), rng)
    del z_max_all
    scanner_flagged = set(int(i) for i in np.where(aurocs > args.auroc_thr)[0])
    print(f"  flagged {len(scanner_flagged)} / {d_sae} latents (AUROC > {args.auroc_thr})")

    write_csv(out / "stage0_scanner_audit.csv",
              ["latent", "max_auroc", "flagged"],
              [[i, float(aurocs[i]), int(i in scanner_flagged)]
               for i in range(d_sae)])

    # ── Stage 1 — Activation frequency ───────────────────────────────────────
    print("[stage 1] activation frequency")
    freq = activation_frequency(z_max_per_ds)
    del z_max_per_ds  # token-level codes no longer needed after frequency
    level1: List[int] = []
    for f in range(d_sae):
        if f in scanner_flagged:
            continue
        hits = sum(freq[name][f] >= args.freq_thr for name in dataset_names)
        if hits >= args.min_ds:
            level1.append(f)
    print(f"  L1 candidates: {len(level1)}")

    write_csv(out / "stage1_frequency.csv",
              ["latent"] + [f"freq_{n}" for n in dataset_names] + ["is_l1"],
              [[f] + [float(freq[n][f]) for n in dataset_names] +
               [int(f in set(level1))]
               for f in range(d_sae)])

    # ── Stage 2 — Cross-dataset semantic consistency ─────────────────────────
    print("[stage 2] semantic consistency")
    min_cos = semantic_consistency(
        img_max_per_ds, raw_acts, level1, t=args.t_top,
    )
    level2 = [f for f, c in min_cos.items() if c >= args.cos_thr]
    del raw_acts  # no longer needed after stage 2
    print(f"  L2 confirmed: {len(level2)}")

    write_csv(out / "stage2_semantic.csv",
              ["latent", "min_pairwise_cosine", "is_l2"],
              [[f, float(min_cos[f]), int(f in set(level2))] for f in level1])

    # ── Stage 3 — Split audit with ancestor trace ────────────────────────────
    print("[stage 3] split audit")
    splits, to_drop = split_audit(
        level2, img_max_per_ds, img_per_scale, k_schedule, offsets,
        split_thr=args.split_thr, anc_thr=args.anc_thr,
    )
    after_split = [f for f in level2 if f not in to_drop]
    ancestors = sorted({g for _, _, _, g in splits if g != -1})
    after_split = sorted(set(after_split) | set(ancestors))
    print(f"  split pairs: {len(splits)}; "
          f"retained after ancestor trace: {len(after_split)}")

    write_csv(out / "stage3_splits.csv",
              ["f_a", "f_b", "jaccard_at_kmax",
               "k_ancestor", "g_ancestor"],
              [[a, b,
                float(jaccard(
                    top_sets_per_latent(img_max_per_ds, offsets, [a])[a],
                    top_sets_per_latent(img_max_per_ds, offsets, [b])[b],
                )),
                ka, ga] for a, b, ka, ga in splits])

    # ── Stage 4 — Prefix stability ───────────────────────────────────────────
    print("[stage 4] prefix stability")
    stab = prefix_stability(
        after_split, img_per_scale, k_schedule, offsets, thr=args.stability_thr,
    )
    universal: List[int] = [
        f for f in after_split if stab[f] and min(stab[f]) >= args.stability_thr
    ]
    print(f"  prefix-stable universal latents: {len(universal)}")

    pair_headers = [f"J_{a}_{b}" for a, b in zip(k_schedule[:-1], k_schedule[1:])]
    write_csv(out / "stage4_prefix_stability.csv",
              ["latent"] + pair_headers + ["min_jaccard", "is_stable"],
              [[f] + [float(v) for v in stab[f]] +
               [float(min(stab[f])) if stab[f] else 0.0,
                int(f in set(universal))]
               for f in after_split])

    # ── Summary ──────────────────────────────────────────────────────────────
    summary = {
        "d_sae": d_sae,
        "k_schedule": k_schedule,
        "datasets": dataset_names,
        "n_scanner_flagged":      len(scanner_flagged),
        "n_level1_candidates":    len(level1),
        "n_level2_confirmed":     len(level2),
        "n_split_pairs":          len(splits),
        "n_after_split":          len(after_split),
        "n_universal":            len(universal),
        "thresholds": {
            "t_top":         args.t_top,
            "freq_thr":      args.freq_thr,
            "min_ds":        args.min_ds,
            "cos_thr":       args.cos_thr,
            "split_thr":     args.split_thr,
            "anc_thr":       args.anc_thr,
            "stability_thr": args.stability_thr,
            "auroc_thr":     args.auroc_thr,
        },
    }
    with open(out / "universality_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(out / "confirmed_universal_latents.json", "w") as f:
        json.dump({"latents": sorted(universal)}, f, indent=2)

    print(f"[done] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
