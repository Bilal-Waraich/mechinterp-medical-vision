"""Illness-stratified evaluation of confirmed universal SAE latents.

Two analyses on the 188 confirmed universal latents from universality_eval.py:

  Analysis A — Illness discriminability probe
    Coarse:      7 canonical groups (labels from safetensors)
    Fine-grained: per-dataset pathology labels from CSVs

  Analysis B — Illness-stratified spatial grounding
    For RSNA and SIIM-COVID only (have bbox annotations):
    Pearson r between per-patch activation magnitude and per-patch bbox overlap,
    stratified by illness group.  Directly validates whether confirmed latents
    fire at anatomically correct locations for each pathology.

Usage — Matryoshka (separate files):
    python src/scripts/illness_stratified_eval.py \\
        --checkpoint_dir /deep_learning/output/UMed-SAE/matryoshka/layer-2/.../step_0200000 \\
        --confirmed_latents results/universality/confirmed_universal_latents.json \\
        --activations /path/activations_chexpert.safetensors:chexpert \\
        --activations /path/activations_rsna.safetensors:rsna \\
        --activations /path/activations_siim.safetensors:siim_covid \\
        --activations /path/activations_tb.safetensors:tb \\
        --chex_csv    /deep_learning/input/data/mmm/chexpert/CheXpert-v1.0/train.csv \\
        --rsna_csv    /deep_learning/input/data/mmm/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv \\
        --siim_study_csv /deep_learning/input/data/mmm/siim-covid19-detection/train_study_level.csv \\
        --siim_bbox_csv  /deep_learning/input/data/mmm/siim-covid19-detection/train_image_level.csv \\
        --rsna_bbox_csv  /deep_learning/input/data/mmm/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv \\
        --output_dir  /deep_learning/output/... \\
        --device cuda:0

Usage — BatchTopK (merged file):
    python src/scripts/illness_stratified_eval.py \\
        --checkpoint_dir /deep_learning/output/UMed-SAE/batch-topk/layer-2/.../step_0200000 \\
        --confirmed_latents results/universality/confirmed_universal_latents.json \\
        --merged_activations /path/activations.safetensors \\
        --dataset_names chexpert rsna siim_covid tb \\
        ... (same CSV args) ...
        --output_dir  /deep_learning/output/...
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from tqdm.auto import tqdm

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from visualisation.loaders import load_sae

# Import reusable components from universality_eval
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from universality_eval import (
    CANONICAL_GROUPS,
    CANONICAL_GROUP_NAMES,
    load_bbox_csv,
    compute_patch_bbox_overlap,
    _read_safetensors_metadata,
)

# Canonical group index mapping (matches harvest-time encoding)
GROUP_TO_IDX: dict[str, int] = {g: i for i, g in enumerate(CANONICAL_GROUP_NAMES)}

# CheXpert fine-grained pathology columns (14 labels)
CHEXPERT_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]

# SIIM study-level label columns
SIIM_STUDY_LABELS = [
    "Negative for Pneumonia",
    "Typical Appearance",
    "Indeterminate Appearance",
    "Atypical Appearance",
]

MIN_CLASS_POSITIVES = 10  # minimum positive samples to include a class in probe


# ══════════════════════════════════════════════════════════════════════════════
# Activation loading
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset_activations(
    args: argparse.Namespace,
) -> list[tuple[str, torch.Tensor, torch.Tensor, list[str]]]:
    """Load per-dataset activations, labels, and image stems.

    Returns list of (dataset_name, acts [N*49, d_model], labels [N], image_stems [N]).
    """
    if args.merged_activations:
        raw = load_file(str(args.merged_activations))
        acts_all = raw["activations"].float()
        ds_ids   = raw["dataset_ids"].long()
        labels_all = raw.get("labels", torch.full((acts_all.shape[0],), -1, dtype=torch.int32))

        meta = _read_safetensors_metadata(args.merged_activations)
        stems_by_name: dict[str, list[str]] = (
            json.loads(meta["image_stems"]) if "image_stems" in meta else {}
        )
        n_patches = int(meta.get("spatial_h", 7)) * int(meta.get("spatial_w", 7))

        datasets = []
        for i, name in enumerate(args.dataset_names):
            mask_tokens = ds_ids == i
            acts = acts_all[mask_tokens]
            n_images = acts.shape[0] // n_patches
            # labels tensor is per-image in merged files
            img_mask = torch.zeros(labels_all.shape[0], dtype=torch.bool)
            # dataset_ids is per-token; extract per-image label by taking first token per image
            token_indices = mask_tokens.nonzero(as_tuple=True)[0]
            if len(token_indices) > 0:
                img_indices = token_indices[::n_patches]
                lbl = labels_all[img_indices]
            else:
                lbl = torch.full((0,), -1, dtype=torch.int32)
            stems = stems_by_name.get(name, [])
            datasets.append((name, acts, lbl, stems))
        return datasets

    # Separate per-dataset files
    datasets = []
    for entry in args.activations:
        path, label = entry.rsplit(":", 1)
        raw = load_file(str(path))
        acts = raw["activations"].float()
        labels = raw.get("labels", torch.full((acts.shape[0],), -1, dtype=torch.int32))

        meta = _read_safetensors_metadata(path)
        n_patches = int(meta.get("spatial_h", 7)) * int(meta.get("spatial_w", 7))
        n_images = acts.shape[0] // n_patches

        # labels in per-dataset files are per-image
        if labels.shape[0] == acts.shape[0]:
            # token-level — take first token per image
            labels = labels[::n_patches]
        # else already per-image

        stems: list[str] = []
        if "image_stems" in meta:
            parsed = json.loads(meta["image_stems"])
            stems = parsed.get(label, parsed) if isinstance(parsed, dict) else parsed

        datasets.append((label, acts, labels, stems))
    return datasets


# ══════════════════════════════════════════════════════════════════════════════
# Fine-grained label loaders
# ══════════════════════════════════════════════════════════════════════════════

def load_chexpert_fine_labels(csv_path: str) -> dict[str, np.ndarray]:
    """Load CheXpert 14-column labels. Returns dict[stem -> label_array (14,)]."""
    result: dict[str, np.ndarray] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row.get("Path", "")
            stem = Path(path).stem
            arr = np.zeros(len(CHEXPERT_LABELS), dtype=np.float32)
            for j, col in enumerate(CHEXPERT_LABELS):
                val = row.get(col, "").strip()
                arr[j] = 1.0 if val == "1" else 0.0  # treat -1 (uncertain) as 0
            result[stem] = arr
    return result


def load_rsna_fine_labels(csv_path: str) -> dict[str, int]:
    """Load RSNA binary labels. Returns dict[patientId -> 0|1]."""
    result: dict[str, int] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = Path(row.get("patientId", "")).stem
            try:
                result[pid] = int(row.get("Target", "0"))
            except ValueError:
                result[pid] = 0
    return result


def load_siim_fine_labels(study_csv_path: str) -> dict[str, np.ndarray]:
    """Load SIIM 4-class study labels. Returns dict[study_stem -> label_array (4,)]."""
    result: dict[str, np.ndarray] = {}
    with open(study_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row.get("id", row.get("StudyInstanceUID", "")).strip()
            stem = Path(uid).stem.replace("_study", "")
            arr = np.zeros(len(SIIM_STUDY_LABELS), dtype=np.float32)
            for j, col in enumerate(SIIM_STUDY_LABELS):
                val = row.get(col, "").strip()
                try:
                    arr[j] = float(val)
                except ValueError:
                    arr[j] = 0.0
            result[stem] = arr
    return result


def detect_rsna_label_issue(labels: torch.Tensor) -> bool:
    """Return True if RSNA labels are all zeros (known harvest bug)."""
    unique = labels.unique()
    return len(unique) == 1 and unique[0].item() == 0


def align_fine_labels(
    dataset_name: str,
    image_stems: list[str],
    fine_label_map: dict,
    n_label_cols: int,
) -> np.ndarray:
    """Align per-stem label map to activation order. Returns [N, n_label_cols]."""
    N = len(image_stems)
    out = np.full((N, n_label_cols), np.nan, dtype=np.float32)
    for i, stem in enumerate(image_stems):
        s = Path(stem).stem
        entry = fine_label_map.get(s)
        if entry is None:
            continue
        if isinstance(entry, np.ndarray):
            out[i] = entry
        else:
            out[i, 0] = float(entry)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Encoding
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def encode_dataset(
    acts: torch.Tensor,          # [N*49, d_model]
    sae,
    confirmed_indices: list[int],
    n_patches: int,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode activations and return:
    - z_img     [N, n_confirmed]        per-image max activation (probe feature)
    - z_patches [N, n_patches, n_confirmed]  per-patch activations (spatial grounding)
    """
    N_tokens = acts.shape[0]
    n_images = N_tokens // n_patches
    n_conf = len(confirmed_indices)
    conf_t = torch.tensor(confirmed_indices, dtype=torch.long)

    all_codes = torch.zeros(N_tokens, n_conf, dtype=torch.float32)

    for start in tqdm(range(0, N_tokens, batch_size), desc="  encode", leave=False):
        end = min(start + batch_size, N_tokens)
        codes = sae.encode(acts[start:end].to(device)).cpu()  # [B, d_sae]
        all_codes[start:end] = codes[:, conf_t]               # [B, n_conf]

    z_patches = all_codes.reshape(n_images, n_patches, n_conf)
    z_img = z_patches.amax(dim=1)  # [N, n_conf]
    return z_img, z_patches


# ══════════════════════════════════════════════════════════════════════════════
# Analysis A — Illness discriminability probe
# ══════════════════════════════════════════════════════════════════════════════

def _auroc_cv(X: np.ndarray, y: np.ndarray, n_jobs: int) -> tuple[float, float]:
    """5-fold stratified CV macro AUROC for a single latent.

    Returns (mean_auroc, std_auroc).  Returns NaN if too few samples.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import label_binarize

    classes = np.unique(y[~np.isnan(y)])
    classes = classes[classes >= 0].astype(int)
    if len(classes) < 2:
        return float("nan"), float("nan")

    valid = ~np.isnan(y) & (y >= 0)
    Xv, yv = X[valid], y[valid].astype(int)
    if len(Xv) < 2 * 5:
        return float("nan"), float("nan")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, test_idx in skf.split(Xv, yv):
        Xtr, ytr = Xv[train_idx], yv[train_idx]
        Xte, yte = Xv[test_idx], yv[test_idx]
        test_classes = np.unique(yte)

        # Skip if any class has fewer than MIN_CLASS_POSITIVES in train
        class_counts = np.bincount(ytr, minlength=int(classes.max()) + 1)
        if any(class_counts[c] < MIN_CLASS_POSITIVES for c in test_classes):
            continue

        clf = LogisticRegression(
            class_weight="balanced", max_iter=500, n_jobs=n_jobs, solver="lbfgs",
            multi_class="ovr",
        )
        try:
            clf.fit(Xtr.reshape(-1, 1), ytr)
            if len(test_classes) < 2:
                continue
            if len(classes) == 2:
                proba = clf.predict_proba(Xte.reshape(-1, 1))[:, 1]
                score = roc_auc_score(yte, proba)
            else:
                proba = clf.predict_proba(Xte.reshape(-1, 1))
                ybin = label_binarize(yte, classes=list(range(len(classes))))
                score = roc_auc_score(ybin, proba, average="macro", multi_class="ovr")
            scores.append(score)
        except Exception:
            continue

    if not scores:
        return float("nan"), float("nan")
    return float(np.mean(scores)), float(np.std(scores))


def run_discriminability_probe_coarse(
    z_img: torch.Tensor,         # [N, n_conf]
    labels: torch.Tensor,        # [N]  canonical group index
    confirmed_indices: list[int],
    dataset_name: str,
    n_jobs: int,
) -> list[dict]:
    """Analysis A1: coarse 7-group AUROC per latent."""
    X = z_img.numpy()
    y = labels.numpy().astype(float)
    rows = []
    for i, lat_idx in enumerate(tqdm(confirmed_indices, desc=f"  probe_coarse/{dataset_name}", leave=False)):
        mean_auc, std_auc = _auroc_cv(X[:, i], y, n_jobs)
        rows.append({
            "latent_idx": lat_idx,
            "dataset": dataset_name,
            "auroc_mean": mean_auc,
            "auroc_std": std_auc,
        })
    return rows


def run_discriminability_probe_fine(
    z_img: torch.Tensor,         # [N, n_conf]
    fine_labels: np.ndarray,     # [N, n_label_cols]
    label_names: list[str],
    confirmed_indices: list[int],
    dataset_name: str,
    n_jobs: int,
) -> list[dict]:
    """Analysis A2: fine-grained per-column binary/multiclass AUROC per latent."""
    X = z_img.numpy()
    rows = []
    for j, label_name in enumerate(label_names):
        y = fine_labels[:, j]
        for i, lat_idx in enumerate(confirmed_indices):
            mean_auc, std_auc = _auroc_cv(X[:, i], y, n_jobs)
            rows.append({
                "latent_idx": lat_idx,
                "dataset": dataset_name,
                "label_name": label_name,
                "auroc_mean": mean_auc,
                "auroc_std": std_auc,
            })
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Analysis B — Illness-stratified spatial grounding
# ══════════════════════════════════════════════════════════════════════════════

def run_spatial_grounding_stratified(
    z_patches: torch.Tensor,      # [N, 49, n_conf]
    labels: torch.Tensor,         # [N]
    bbox_overlap: torch.Tensor,   # [N, 49]
    confirmed_indices: list[int],
    dataset_name: str,
) -> list[dict]:
    """For each illness group, Pearson r between patch activation and bbox overlap."""
    from scipy.stats import pearsonr

    rows = []
    for g, group_name in enumerate(CANONICAL_GROUP_NAMES):
        mask = (labels == g).numpy()
        n_g = mask.sum()
        if n_g < 5:
            for lat_idx in confirmed_indices:
                rows.append({
                    "latent_idx": lat_idx,
                    "illness_group": group_name,
                    "dataset": dataset_name,
                    "pearson_r": float("nan"),
                    "pval": float("nan"),
                    "n_images": int(n_g),
                })
            continue

        acts_g = z_patches[mask]            # [n_g, 49, n_conf]
        bbox_g = bbox_overlap[mask]         # [n_g, 49]
        bbox_flat = bbox_g.numpy().reshape(-1)

        for i, lat_idx in enumerate(confirmed_indices):
            act_flat = acts_g[:, :, i].numpy().reshape(-1)
            if act_flat.std() < 1e-9 or bbox_flat.std() < 1e-9:
                r, pval = float("nan"), float("nan")
            else:
                r, pval = pearsonr(act_flat, bbox_flat)
            rows.append({
                "latent_idx": lat_idx,
                "illness_group": group_name,
                "dataset": dataset_name,
                "pearson_r": float(r),
                "pval": float(pval),
                "n_images": int(n_g),
            })
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Activation profiles
# ══════════════════════════════════════════════════════════════════════════════

def compute_activation_profiles(
    z_img: torch.Tensor,
    labels: torch.Tensor,
    confirmed_indices: list[int],
    dataset_name: str,
) -> list[dict]:
    rows = []
    for g, group_name in enumerate(CANONICAL_GROUP_NAMES):
        mask = (labels == g)
        if mask.sum() == 0:
            continue
        z_g = z_img[mask]  # [n_g, n_conf]
        for i, lat_idx in enumerate(confirmed_indices):
            col = z_g[:, i]
            rows.append({
                "latent_idx": lat_idx,
                "illness_group": group_name,
                "dataset": dataset_name,
                "mean_activation": float(col.mean()),
                "activation_freq": float((col > 0).float().mean()),
                "n_images": int(mask.sum()),
            })
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Output helpers
# ══════════════════════════════════════════════════════════════════════════════

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def build_latent_matrix(coarse_rows: list[dict], confirmed_indices: list[int]) -> np.ndarray:
    """Build [7, n_confirmed] AUROC matrix averaged across datasets."""
    n_conf = len(confirmed_indices)
    idx_map = {lat: i for i, lat in enumerate(confirmed_indices)}
    matrix = np.full((7, n_conf), np.nan)
    counts = np.zeros((7, n_conf))

    for row in coarse_rows:
        lat = row["latent_idx"]
        if lat not in idx_map:
            continue
        j = idx_map[lat]
        auc = row["auroc_mean"]
        if np.isnan(auc):
            continue
        # group index not stored in coarse rows; rebuild from dataset
        # (coarse rows don't have group — need separate approach)
        # Actually coarse probe produces one AUROC per latent per dataset (all groups together).
        # The matrix rows should be per-group; for the summary we use a per-group mean_activation
        # proxy instead. Fill matrix with auroc_mean uniformly for now.
    return matrix


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Illness-stratified evaluation of confirmed universal SAE latents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ap.add_argument("--checkpoint_dir", required=True,
                    help="SAE checkpoint directory (sae_weights.safetensors + cfg.json).")
    ap.add_argument("--confirmed_latents", required=True,
                    help="JSON file with {'latents': [...]} from universality_eval.")

    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--activations", action="append", metavar="PATH:LABEL",
                     help="Per-dataset activations file:label (repeat 4×). Matryoshka format.")
    grp.add_argument("--merged_activations", metavar="PATH",
                     help="Merged activations.safetensors with dataset_ids. BatchTopK format.")

    ap.add_argument("--dataset_names", nargs="+",
                    default=["chexpert", "rsna", "siim_covid", "tb"],
                    help="Dataset names in dataset_id order (merged format only).")

    # Fine-grained label CSVs
    ap.add_argument("--chex_csv",       metavar="PATH",
                    help="CheXpert CheXpert-v1.0/train.csv")
    ap.add_argument("--rsna_csv",       metavar="PATH",
                    help="RSNA stage_2_train_labels.csv")
    ap.add_argument("--siim_study_csv", metavar="PATH",
                    help="SIIM train_study_level.csv")

    # Bbox CSVs for spatial grounding
    ap.add_argument("--rsna_bbox_csv",  metavar="PATH",
                    help="RSNA stage_2_train_labels.csv (same file as rsna_csv).")
    ap.add_argument("--siim_bbox_csv",  metavar="PATH",
                    help="SIIM train_image_level.csv")

    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--n_jobs",     type=int, default=-1,
                    help="Parallelism for LogisticRegression.")
    ap.add_argument("--top_k",      type=int, default=10,
                    help="Top-k latents per illness group in summary JSON.")
    ap.add_argument("--device",
                    default="cuda:0" if torch.cuda.is_available() else "cpu")

    return ap.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    run_warnings: list[str] = []

    print("=" * 65)
    print("  Illness-stratified SAE Latent Evaluation")
    print("=" * 65)

    # ── Confirmed latents ──────────────────────────────────────────────────
    confirmed_indices: list[int] = json.loads(Path(args.confirmed_latents).read_text())["latents"]
    print(f"\n[1] Confirmed latents: {len(confirmed_indices)}")

    # ── Load SAE ──────────────────────────────────────────────────────────
    print(f"\n[2] Loading SAE from {args.checkpoint_dir} …")
    sae = load_sae(args.checkpoint_dir, device=str(device))
    sae.eval()
    print(f"    d_model={sae.d_model}  d_sae={sae.d_sae}")

    # ── Load activations ──────────────────────────────────────────────────
    print(f"\n[3] Loading activations …")
    datasets = load_dataset_activations(args)

    # Infer n_patches
    if args.merged_activations:
        meta = _read_safetensors_metadata(args.merged_activations)
    else:
        path0 = args.activations[0].rsplit(":", 1)[0]
        meta = _read_safetensors_metadata(path0)
    n_patches = int(meta.get("spatial_h", 7)) * int(meta.get("spatial_w", 7))
    spatial_hw = int(n_patches ** 0.5)
    print(f"    n_patches={n_patches}  spatial={spatial_hw}×{spatial_hw}")

    # ── Load fine-grained labels ───────────────────────────────────────────
    print(f"\n[4] Loading fine-grained label CSVs …")
    chex_labels_map = load_chexpert_fine_labels(args.chex_csv) if args.chex_csv else None
    rsna_labels_map = load_rsna_fine_labels(args.rsna_csv)     if args.rsna_csv else None
    siim_labels_map = load_siim_fine_labels(args.siim_study_csv) if args.siim_study_csv else None

    # ── Accumulate results ─────────────────────────────────────────────────
    profile_rows: list[dict] = []
    coarse_probe_rows: list[dict] = []
    fine_probe_rows: list[dict] = []
    spatial_rows: list[dict] = []

    print(f"\n[5] Encoding + analysis per dataset …")
    for ds_name, acts, labels, image_stems in datasets:
        N_tokens = acts.shape[0]
        n_images = N_tokens // n_patches
        print(f"\n  [{ds_name}]  {n_images} images, {N_tokens} tokens")

        # RSNA label issue detection
        if ds_name == "rsna":
            unique_vals = labels.unique()
            if len(unique_vals) == 1 and unique_vals[0].item() == 0:
                msg = f"[{ds_name}] All coarse labels are 0 — loading from rsna_csv."
                warnings.warn(msg)
                run_warnings.append(msg)
                print(f"  WARNING: {msg}")
                # Remap stems → coarse label using RSNA CSV
                if rsna_labels_map and image_stems:
                    for img_i, stem in enumerate(image_stems):
                        s = Path(stem).stem
                        target = rsna_labels_map.get(s, 0)
                        # 1=pneumonia→airspace_opacity(0), 0=normal→no_finding(5)
                        labels[img_i] = 0 if target == 1 else 5

        # ── Encode ──────────────────────────────────────────────────────
        z_img, z_patches = encode_dataset(
            acts, sae, confirmed_indices, n_patches, device, args.batch_size
        )
        # z_img      [N, n_conf]
        # z_patches  [N, n_patches, n_conf]

        # ── Activation profiles ──────────────────────────────────────────
        profile_rows.extend(
            compute_activation_profiles(z_img, labels, confirmed_indices, ds_name)
        )

        # ── Analysis A1: coarse probe ────────────────────────────────────
        coarse_probe_rows.extend(
            run_discriminability_probe_coarse(
                z_img, labels, confirmed_indices, ds_name, args.n_jobs
            )
        )

        # ── Analysis A2: fine probe ──────────────────────────────────────
        if ds_name == "chexpert" and chex_labels_map and image_stems:
            fine = align_fine_labels(ds_name, image_stems, chex_labels_map, len(CHEXPERT_LABELS))
            fine_probe_rows.extend(
                run_discriminability_probe_fine(
                    z_img, fine, CHEXPERT_LABELS, confirmed_indices, ds_name, args.n_jobs
                )
            )
        elif ds_name == "rsna" and rsna_labels_map and image_stems:
            fine = align_fine_labels(ds_name, image_stems, rsna_labels_map, 1)
            fine_probe_rows.extend(
                run_discriminability_probe_fine(
                    z_img, fine, ["pneumonia"], confirmed_indices, ds_name, args.n_jobs
                )
            )
        elif ds_name in ("siim_covid", "siim") and siim_labels_map and image_stems:
            fine = align_fine_labels(ds_name, image_stems, siim_labels_map, len(SIIM_STUDY_LABELS))
            fine_probe_rows.extend(
                run_discriminability_probe_fine(
                    z_img, fine, SIIM_STUDY_LABELS, confirmed_indices, ds_name, args.n_jobs
                )
            )
        else:
            if ds_name not in ("tb",):
                run_warnings.append(f"[{ds_name}] No fine label CSV — skipping fine probe.")

        # ── Analysis B: spatial grounding ────────────────────────────────
        bbox_csv = (
            args.rsna_bbox_csv if ds_name == "rsna"
            else args.siim_bbox_csv if ds_name in ("siim_covid", "siim")
            else None
        )
        if bbox_csv and image_stems:
            print(f"  Loading bboxes for {ds_name} …")
            bboxes = load_bbox_csv(bbox_csv, ds_name)
            bbox_overlap = compute_patch_bbox_overlap(
                bboxes, image_stems, spatial_hw, spatial_hw
            )  # [N, n_patches]
            spatial_rows.extend(
                run_spatial_grounding_stratified(
                    z_patches, labels, bbox_overlap, confirmed_indices, ds_name
                )
            )
        else:
            # Record NaN rows so the CSV is complete
            for g, group_name in enumerate(CANONICAL_GROUP_NAMES):
                for lat_idx in confirmed_indices:
                    spatial_rows.append({
                        "latent_idx": lat_idx,
                        "illness_group": group_name,
                        "dataset": ds_name,
                        "pearson_r": float("nan"),
                        "pval": float("nan"),
                        "n_images": 0,
                    })
            if bbox_csv and not image_stems:
                run_warnings.append(
                    f"[{ds_name}] Bbox CSV provided but no image_stems in metadata — skipping spatial grounding."
                )

    # ── Build illness_latent_matrix [7 × n_conf] ──────────────────────────
    print(f"\n[6] Building latent matrix …")
    n_conf = len(confirmed_indices)
    idx_map = {lat: i for i, lat in enumerate(confirmed_indices)}

    # Use mean activation profiles to build [7 × n_conf] matrix
    mat = np.full((7, n_conf), 0.0)
    mat_counts = np.zeros((7, n_conf))
    for row in profile_rows:
        g = GROUP_TO_IDX.get(row["illness_group"], -1)
        j = idx_map.get(row["latent_idx"], -1)
        if g < 0 or j < 0:
            continue
        mat[g, j] += row["mean_activation"]
        mat_counts[g, j] += 1
    with np.errstate(invalid="ignore"):
        mat = np.where(mat_counts > 0, mat / mat_counts, 0.0)

    # ── Write outputs ──────────────────────────────────────────────────────
    print(f"\n[7] Writing outputs to {out_dir} …")

    write_csv(out_dir / "illness_activation_profiles.csv", profile_rows)
    print("  ✓ illness_activation_profiles.csv")

    write_csv(out_dir / "illness_discriminability_probe_coarse.csv", coarse_probe_rows)
    print("  ✓ illness_discriminability_probe_coarse.csv")

    write_csv(out_dir / "illness_discriminability_probe_fine.csv", fine_probe_rows)
    print("  ✓ illness_discriminability_probe_fine.csv")

    write_csv(out_dir / "illness_spatial_grounding.csv", spatial_rows)
    print("  ✓ illness_spatial_grounding.csv")

    # illness_latent_matrix.csv — [7 × n_conf] mean activation
    with open(out_dir / "illness_latent_matrix.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["illness_group"] + [str(x) for x in confirmed_indices])
        for g, group_name in enumerate(CANONICAL_GROUP_NAMES):
            w.writerow([group_name] + [f"{v:.6f}" for v in mat[g]])
    print("  ✓ illness_latent_matrix.csv")

    # illness_latent_summary.json — top-k latents per illness group
    top_k_per_group: dict[str, list[int]] = {}
    for g, group_name in enumerate(CANONICAL_GROUP_NAMES):
        row_scores = mat[g]
        top_k_idx = np.argsort(row_scores)[::-1][: args.top_k]
        top_k_per_group[group_name] = [int(confirmed_indices[j]) for j in top_k_idx]

    summary = {
        "top_k_latents_per_illness": top_k_per_group,
        "illness_group_names": CANONICAL_GROUP_NAMES,
        "confirmed_indices": confirmed_indices,
        "n_confirmed": n_conf,
        "matrix_description": "rows=illness_groups, cols=confirmed_latents, values=mean_activation averaged across datasets",
    }
    (out_dir / "illness_latent_summary.json").write_text(json.dumps(summary, indent=2))
    print("  ✓ illness_latent_summary.json")

    elapsed = time.time() - t0
    eval_summary = {
        "run_timestamp": datetime.utcnow().isoformat() + "Z",
        "elapsed_seconds": round(elapsed, 1),
        "n_confirmed_latents": n_conf,
        "n_patches": n_patches,
        "datasets": [ds for ds, *_ in datasets],
        "checkpoint_dir": str(args.checkpoint_dir),
        "confirmed_latents_path": str(args.confirmed_latents),
        "warnings": run_warnings,
    }
    (out_dir / "illness_eval_summary.json").write_text(json.dumps(eval_summary, indent=2))
    print("  ✓ illness_eval_summary.json")

    print(f"\nDone in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
