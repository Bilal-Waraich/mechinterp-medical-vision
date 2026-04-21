"""TD-3 Two-Level Universality Metric for X-ray SAE.

Implements the full universality evaluation pipeline described in
docs/methodology.md §1.2–1.5:

  Step 1  Load SAE + per-dataset activations (merged or separate file formats)
  Step 2  Apply scanner-audit exclusion list (optional)
  Step 3  Level 1  — per-token activation frequency ≥1% on ≥2 datasets
                      + canonical pathology class match (if label CSVs provided)
  Step 4  Level 2  — per-dataset d_model centroid cosine similarity ≥0.70
  Step 5  Cross-dataset probe — leave-one-dataset-out logistic AUROC (optional)
  Step 6  Spatial grounding — Pearson r vs bbox patch overlap (optional)
  Step 7  Save outputs

Input formats
-------------
Merged (BatchTopK jobs):
  Single activations.safetensors containing:
    - "activations"  [N_total, d_model]
    - "dataset_ids"  [N_total]  (int: 0=chexpert, 1=rsna, 2=siim_covid, 3=tb)

Separate (Matryoshka jobs):
  Four files, each with "activations" [N_i, d_model]; no dataset_ids tensor.
  Pass via --activations PATH:LABEL (repeat 4×).

Usage — Matryoshka
------------------
python src/scripts/universality_eval.py \\
    --checkpoint_dir /deep_learning/output/UMed-SAE/matryoshka/layer-2/xray-conglomerate/xray_v1/checkpoints/step_0200000 \\
    --activations /deep_learning/output/UMed-SAE/matryoshka/layer-2/xray-conglomerate/xray_v1/activations_chexpert.safetensors:chexpert \\
    --activations /deep_learning/output/UMed-SAE/matryoshka/layer-2/xray-conglomerate/xray_v1/activations_rsna.safetensors:rsna \\
    --activations /deep_learning/output/UMed-SAE/matryoshka/layer-2/xray-conglomerate/xray_v1/activations_siim.safetensors:siim_covid \\
    --activations /deep_learning/output/UMed-SAE/matryoshka/layer-2/xray-conglomerate/xray_v1/activations_tb.safetensors:tb \\
    --scanner_audit /deep_learning/output/.../audit/scanner_latent_audit.json \\
    --label_csv chexpert:/deep_learning/input/.../CheXpert-v1.0/train.csv \\
    --label_csv rsna:/deep_learning/input/.../stage_2_train_labels.csv \\
    --rsna_bbox_csv /deep_learning/input/.../stage_2_train_labels.csv \\
    --siim_bbox_csv /deep_learning/input/.../siim-covid19-detection/train_image_level.csv \\
    --output_dir /deep_learning/output/.../universality \\
    --device cuda:0

Usage — BatchTopK (merged file)
--------------------------------
python src/scripts/universality_eval.py \\
    --checkpoint_dir /deep_learning/output/UMed-SAE/batch-topk/layer-2/.../checkpoints/step_0200000 \\
    --merged_activations /deep_learning/output/UMed-SAE/batch-topk/layer-2/.../activations.safetensors \\
    --dataset_names chexpert rsna siim_covid tb \\
    --output_dir /deep_learning/output/.../universality \\
    --device cuda:0

See also
--------
docs/methodology.md §1 — full universality definition and rationale
src/scripts/scanner_latent_audit.py — produces scanner_latent_audit.json
src/scripts/null_baseline_eval.py — repeats this analysis on random/PCA null models
src/scripts/cross_dataset_eval.py — simplified top-K fraction proxy (exploratory)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tqdm.auto import tqdm

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from visualisation.loaders import load_sae


# ── Canonical pathology group mapping ─────────────────────────────────────────
# Fleischner Society taxonomy (Hansell et al. 2008).
# Maps raw dataset label strings → 7 canonical groups.
# CheXpert uncertain labels (-1) are treated as negative before this mapping.

CANONICAL_GROUPS: dict[str, str] = {
    # CheXpert
    "Consolidation":               "airspace_opacity",
    "Lung Opacity":                "airspace_opacity",
    "Pleural Effusion":            "pleural_abnormality",
    "Pleural Other":               "pleural_abnormality",
    "Cardiomegaly":                "cardiac_mediastinal",
    "Enlarged Cardiomediastinum":  "cardiac_mediastinal",
    "Atelectasis":                 "atelectasis",
    "Edema":                       "vascular_edema",
    "No Finding":                  "no_finding",
    "Support Devices":             "other",
    "Fracture":                    "other",
    # RSNA
    "Normal":                      "no_finding",
    "Pneumonia":                   "airspace_opacity",
    # SIIM-COVID19
    "Opacity":                     "airspace_opacity",
    "No finding":                  "no_finding",
    "Pneumothorax":                "other",
    # TB Shenzhen
    "Positive":                    "airspace_opacity",
    "Negative":                    "no_finding",
    # Normalised lower-case fallbacks
    "positive":                    "airspace_opacity",
    "negative":                    "no_finding",
    "normal":                      "no_finding",
    "pneumonia":                   "airspace_opacity",
}

CANONICAL_GROUP_NAMES = [
    "airspace_opacity",
    "pleural_abnormality",
    "cardiac_mediastinal",
    "atelectasis",
    "vascular_edema",
    "no_finding",
    "other",
]

# ── Token-level frequency threshold ───────────────────────────────────────────
FREQUENCY_THRESHOLD = 0.01   # 1% of tokens in a dataset


# ══════════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset_activations(
    args: argparse.Namespace,
) -> list[tuple[str, torch.Tensor]]:
    """Load per-dataset activation tensors.

    Returns list of (dataset_name, activations [N_i, d_model]).
    Supports both merged and separate file formats.
    """
    if args.merged_activations:
        raw = load_file(str(args.merged_activations))
        acts_all = raw["activations"].float()
        ds_ids   = raw["dataset_ids"].long()
        if ds_ids.shape[0] != acts_all.shape[0]:
            raise ValueError(
                f"dataset_ids length {ds_ids.shape[0]} != activations length "
                f"{acts_all.shape[0]}.  This merged file was produced by a "
                f"pre-Bug-A harvest_all.py (dataset_ids stored per-image).  "
                f"Re-harvest with the current harvest_all.py."
            )
        names    = args.dataset_names
        datasets = []
        for i, name in enumerate(names):
            mask = ds_ids == i
            datasets.append((name, acts_all[mask]))
        return datasets

    # Separate per-dataset files
    datasets = []
    for entry in args.activations:
        path, label = entry.rsplit(":", 1)
        raw = load_file(str(path))
        datasets.append((label, raw["activations"].float()))
    return datasets


def load_label_csvs(label_csv_args: list[str] | None) -> dict[str, dict[str, str]]:
    """Parse --label_csv DATASET:PATH pairs.

    Returns dict[dataset_name -> dict[image_id -> canonical_group]].
    Only the dominant (most-frequent positive) canonical group per image is kept.
    """
    if not label_csv_args:
        return {}

    result: dict[str, dict[str, str]] = {}
    for entry in label_csv_args:
        ds_name, csv_path = entry.split(":", 1)
        result[ds_name] = _parse_label_csv(ds_name, csv_path)
    return result


def _parse_label_csv(ds_name: str, csv_path: str) -> dict[str, str]:
    """Parse a single label CSV → {image_id: canonical_group}.

    Handles CheXpert, RSNA, SIIM, and TB formats automatically.
    """
    labels: dict[str, str] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        for row in reader:
            image_id = (
                row.get("Path") or row.get("patientId") or
                row.get("StudyInstanceUID") or row.get("image_id") or
                row.get("filename") or ""
            ).strip()
            if not image_id:
                continue

            # CheXpert: multi-label columns, uncertain (-1) treated as negative
            if "No Finding" in fieldnames:
                positives = []
                for col in fieldnames:
                    val = row.get(col, "").strip()
                    if val == "1" and col in CANONICAL_GROUPS:
                        positives.append(CANONICAL_GROUPS[col])
                group = positives[0] if positives else "no_finding"

            # RSNA: single "Target" column (1=pneumonia, 0=normal)
            elif "Target" in fieldnames:
                target = row.get("Target", "0").strip()
                group = "airspace_opacity" if target == "1" else "no_finding"

            # SIIM-COVID19: "label" column
            elif "label" in fieldnames:
                raw_label = row.get("label", "").strip()
                group = CANONICAL_GROUPS.get(raw_label, "other")

            # TB Shenzhen: filename encodes positive/negative
            else:
                fname = Path(image_id).stem.lower()
                if "positive" in fname or "_p_" in fname:
                    group = "airspace_opacity"
                else:
                    group = "no_finding"

            labels[Path(image_id).stem] = group

    return labels


# ══════════════════════════════════════════════════════════════════════════════
# Level 1 — Activation frequency
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_frequency_per_dataset(
    datasets: list[tuple[str, torch.Tensor]],
    sae,
    device: torch.device,
    batch_size: int = 4096,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Compute per-latent activation frequency for each dataset.

    Args:
        datasets: list of (name, activations [N_i, d_model])
        sae: trained SAE in eval mode
        device: torch device
        batch_size: token-level batch size for SAE encode

    Returns:
        freq_per_ds: dict[name -> Tensor [d_sae]]  (fraction of tokens where latent fires)
        n_tokens:    dict[name -> int]
    """
    freq_per_ds: dict[str, torch.Tensor] = {}
    n_tokens: dict[str, int] = {}

    for ds_name, acts in datasets:
        N = acts.shape[0]
        d_sae = sae.d_sae
        fire_count = torch.zeros(d_sae, dtype=torch.long)

        ts = datetime.now().strftime("%H:%M:%S")
        print(f"  [{ts}] {ds_name}: {N:,} tokens → computing activation frequency …")

        for start in tqdm(range(0, N, batch_size), desc=f"    {ds_name}", leave=False):
            end = min(start + batch_size, N)
            x = acts[start:end].to(device)
            codes = sae.encode(x).cpu()          # [B, d_sae]  sparse top-k
            fire_count += (codes > 0).sum(dim=0).long()

        freq_per_ds[ds_name] = fire_count.float() / N
        n_tokens[ds_name] = N

    return freq_per_ds, n_tokens


def apply_level1_filter(
    freq_per_ds: dict[str, torch.Tensor],
    n_tokens: dict[str, int],
    flagged_indices: set[int],
    threshold: float = FREQUENCY_THRESHOLD,
    min_datasets: int = 2,
) -> tuple[list[int], dict[int, list[str]]]:
    """Apply TD-3 Level 1: frequency threshold on ≥ min_datasets.

    Returns:
        candidates:      list of latent indices passing Level 1
        active_datasets: dict[latent_idx -> list of dataset names where active]
    """
    ds_names = list(freq_per_ds.keys())
    d_sae = next(iter(freq_per_ds.values())).shape[0]

    active_datasets: dict[int, list[str]] = {}
    candidates: list[int] = []

    for i in range(d_sae):
        if i in flagged_indices:
            continue
        active = [ds for ds in ds_names if freq_per_ds[ds][i].item() >= threshold]
        if len(active) >= min_datasets:
            candidates.append(i)
            active_datasets[i] = active

    return candidates, active_datasets


# ══════════════════════════════════════════════════════════════════════════════
# Level 2 — Representational direction alignment
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_centroids_per_dataset(
    datasets: list[tuple[str, torch.Tensor]],
    candidates: list[int],
    sae,
    device: torch.device,
    batch_size: int = 4096,
) -> dict[str, torch.Tensor]:
    """Compute per-latent d_model activation centroid for each dataset.

    For candidate latent i and dataset D:
      centroid[i] = mean of d_model activation vectors where latent i fires.

    Args:
        datasets:   list of (name, activations [N_i, d_model])
        candidates: list of latent indices from Level 1
        sae:        trained SAE in eval mode
        device:     torch device
        batch_size: token-level batch size

    Returns:
        dict[dataset_name -> Tensor [len(candidates), d_model]]
        Rows with zero norm indicate the latent never fired on that dataset.
    """
    cand_set = set(candidates)
    cand_list = sorted(cand_set)
    cand_pos = {v: i for i, v in enumerate(cand_list)}
    n_cands = len(cand_list)
    d_model = sae.d_model

    centroids: dict[str, torch.Tensor] = {}

    for ds_name, acts in datasets:
        N = acts.shape[0]
        # Accumulate sum of d_model vectors and count per candidate latent
        centroid_sum   = torch.zeros(n_cands, d_model, dtype=torch.float32)
        centroid_count = torch.zeros(n_cands, dtype=torch.long)

        ts = datetime.now().strftime("%H:%M:%S")
        print(f"  [{ts}] {ds_name}: computing d_model centroids for {n_cands} candidates …")

        for start in tqdm(range(0, N, batch_size), desc=f"    {ds_name}", leave=False):
            end = min(start + batch_size, N)
            chunk = acts[start:end]                     # [B, d_model] on CPU
            x = chunk.to(device)
            codes = sae.encode(x).cpu()                 # [B, d_sae]

            # For each candidate latent, accumulate d_model vectors where it fired
            for ci, lat_idx in enumerate(cand_list):
                fired_mask = codes[:, lat_idx] > 0      # [B] bool
                if fired_mask.any():
                    centroid_sum[ci]   += chunk[fired_mask].sum(dim=0)
                    centroid_count[ci] += fired_mask.sum().item()

        # Divide to get means; latents that never fired stay zero
        nz = centroid_count > 0
        centroid_sum[nz] = centroid_sum[nz] / centroid_count[nz].float().unsqueeze(1)
        centroids[ds_name] = centroid_sum     # [n_cands, d_model]

    return centroids


def apply_level2_filter(
    candidates: list[int],
    active_datasets: dict[int, list[str]],
    centroids: dict[str, torch.Tensor],
    cosine_threshold: float = 0.70,
) -> tuple[list[int], dict[int, dict]]:
    """Apply TD-3 Level 2: pairwise centroid cosine similarity ≥ threshold.

    For each candidate, checks ALL pairs of active-dataset centroids.
    All pairs must meet the threshold for the latent to be confirmed.

    Returns:
        confirmed:  list of confirmed universal latent indices
        cosine_info: dict[latent_idx -> {pair: cosine_sim, min_sim: float}]
    """
    cand_list = sorted(set(candidates))
    cand_pos  = {v: i for i, v in enumerate(cand_list)}

    confirmed: list[int] = []
    cosine_info: dict[int, dict] = {}

    for lat_idx in candidates:
        active_ds = active_datasets[lat_idx]
        ci = cand_pos[lat_idx]

        pair_sims: dict[str, float] = {}
        all_pass = True

        for a_idx in range(len(active_ds)):
            for b_idx in range(a_idx + 1, len(active_ds)):
                ds_a, ds_b = active_ds[a_idx], active_ds[b_idx]
                vec_a = centroids[ds_a][ci]
                vec_b = centroids[ds_b][ci]
                # Skip if either centroid is zero (latent never fired)
                if vec_a.norm() < 1e-8 or vec_b.norm() < 1e-8:
                    all_pass = False
                    break
                sim = float(F.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0)).item())
                pair_sims[f"{ds_a}↔{ds_b}"] = round(sim, 4)
                if sim < cosine_threshold:
                    all_pass = False
            if not all_pass:
                break

        min_sim = min(pair_sims.values()) if pair_sims else 0.0
        cosine_info[lat_idx] = {"pairs": pair_sims, "min_sim": round(min_sim, 4)}
        if all_pass and pair_sims:
            confirmed.append(lat_idx)

    return confirmed, cosine_info


# ══════════════════════════════════════════════════════════════════════════════
# Cross-dataset probe (supplementary)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_cross_dataset_probe(
    confirmed: list[int],
    datasets: list[tuple[str, torch.Tensor]],
    image_labels: dict[str, dict[str, str]],
    sae,
    device: torch.device,
    n_patches: int,
    batch_size: int = 4096,
) -> dict[int, dict[str, float]]:
    """Leave-one-dataset-out logistic probe on canonical-group labels.

    For each confirmed latent, trains a logistic regression on one dataset's
    per-image max-activation and tests zero-shot on the others.

    Args:
        confirmed:     confirmed universal latent indices
        datasets:      list of (name, activations [N_i, d_model])
        image_labels:  dict[dataset_name -> dict[image_id -> canonical_group]]
        sae:           trained SAE in eval mode
        device:        torch device
        n_patches:     tokens per image (e.g. 49 for 7×7)
        batch_size:    token-level batch size

    Returns:
        dict[latent_idx -> dict["train_ds→test_ds" -> auroc]]
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import LabelEncoder
    except ImportError:
        print("  [probe] scikit-learn not available — skipping cross-dataset probe.")
        return {}

    if not image_labels:
        print("  [probe] No label CSVs provided — skipping cross-dataset probe.")
        return {}

    # Build per-image max activations [n_images, d_sae] per dataset
    print("  Computing per-image feature_max for cross-dataset probe …")
    per_ds_max: dict[str, torch.Tensor] = {}
    per_ds_labels: dict[str, list[str]] = {}

    for ds_name, acts in datasets:
        if ds_name not in image_labels:
            continue
        N = acts.shape[0]
        n_images = N // n_patches
        d_sae = sae.d_sae

        feature_max = torch.zeros(n_images, d_sae)
        for start in tqdm(range(0, N, batch_size), desc=f"    {ds_name}", leave=False):
            end   = min(start + batch_size, N)
            codes = sae.encode(acts[start:end].to(device)).cpu()
            # Map token indices back to image indices
            img_start = start // n_patches
            img_end   = (end - 1) // n_patches + 1
            codes_img = codes.reshape(img_end - img_start, -1, d_sae)
            img_max   = codes_img.max(dim=1).values
            feature_max[img_start:img_end] = img_max

        # Align with label CSV — use sequential order
        labels_dict = image_labels[ds_name]
        label_list  = list(labels_dict.values())[:n_images]
        if len(label_list) < n_images:
            label_list += ["no_finding"] * (n_images - len(label_list))

        per_ds_max[ds_name]    = feature_max          # [n_images, d_sae]
        per_ds_labels[ds_name] = label_list

    ds_names = list(per_ds_max.keys())
    if len(ds_names) < 2:
        print("  [probe] Need ≥2 datasets with labels — skipping.")
        return {}

    probe_results: dict[int, dict[str, float]] = {}
    le = LabelEncoder()
    all_labels = [g for labels in per_ds_labels.values() for g in labels]
    le.fit(all_labels)

    for lat_idx in tqdm(confirmed, desc="  Cross-dataset probe"):
        auroc_map: dict[str, float] = {}

        for train_ds in ds_names:
            X_train = per_ds_max[train_ds][:, lat_idx].numpy().reshape(-1, 1)
            y_train = le.transform(per_ds_labels[train_ds])

            if len(np.unique(y_train)) < 2:
                continue
            clf = LogisticRegression(max_iter=200, random_state=42)
            try:
                clf.fit(X_train, y_train)
            except Exception:
                continue

            for test_ds in ds_names:
                if test_ds == train_ds:
                    continue
                X_test = per_ds_max[test_ds][:, lat_idx].numpy().reshape(-1, 1)
                y_test = le.transform(per_ds_labels[test_ds])
                if len(np.unique(y_test)) < 2:
                    continue
                try:
                    proba = clf.predict_proba(X_test)
                    # Macro-averaged one-vs-rest AUROC
                    auroc = roc_auc_score(y_test, proba, multi_class="ovr",
                                         average="macro", labels=clf.classes_)
                    auroc_map[f"{train_ds}→{test_ds}"] = round(float(auroc), 4)
                except Exception:
                    pass

        probe_results[lat_idx] = auroc_map

    return probe_results


# ══════════════════════════════════════════════════════════════════════════════
# Spatial grounding (bbox IoU)
# ══════════════════════════════════════════════════════════════════════════════

def load_bbox_csv(csv_path: str, dataset: str) -> dict[str, list[dict]]:
    """Parse RSNA or SIIM bounding box CSV.

    Returns dict[image_id -> list of {x1, y1, x2, y2, label}].
    """
    bboxes: dict[str, list[dict]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if dataset == "rsna":
                img_id = row.get("patientId", "").strip()
                target = row.get("Target", "0").strip()
                if target != "1":
                    continue
                try:
                    box = {
                        "x1": float(row["x"]),
                        "y1": float(row["y"]),
                        "x2": float(row["x"]) + float(row["width"]),
                        "y2": float(row["y"]) + float(row["height"]),
                    }
                except (KeyError, ValueError):
                    continue
            elif dataset == "siim_covid":
                img_id = row.get("id", row.get("StudyInstanceUID", "")).strip()
                raw = row.get("boxes", row.get("label", "")).strip()
                if not raw or raw.lower() in ("none", "no finding", ""):
                    continue
                # SIIM stores boxes as "x y width height" space-separated
                try:
                    parts = raw.split()
                    if len(parts) < 4:
                        continue
                    box = {
                        "x1": float(parts[0]),
                        "y1": float(parts[1]),
                        "x2": float(parts[0]) + float(parts[2]),
                        "y2": float(parts[1]) + float(parts[3]),
                    }
                except (ValueError, IndexError):
                    continue
            else:
                continue

            img_id = Path(img_id).stem
            bboxes.setdefault(img_id, []).append(box)

    return bboxes


def compute_patch_bbox_overlap(
    bboxes: dict[str, list[dict]],
    image_ids: list[str],
    spatial_h: int,
    spatial_w: int,
    image_size: int = 224,
) -> torch.Tensor:
    """Compute fractional bbox coverage per patch for each image.

    Args:
        bboxes:    dict[image_id -> list of bbox dicts]
        image_ids: ordered list of image IDs matching activation rows
        spatial_h: patch grid height (e.g. 7)
        spatial_w: patch grid width  (e.g. 7)
        image_size: original image size in pixels (224)

    Returns:
        Tensor [n_images, n_patches]  values in [0, 1]
    """
    n_images  = len(image_ids)
    n_patches = spatial_h * spatial_w
    patch_h   = image_size / spatial_h
    patch_w   = image_size / spatial_w
    overlap   = torch.zeros(n_images, n_patches)

    for img_i, img_id in enumerate(image_ids):
        img_boxes = bboxes.get(Path(img_id).stem, [])
        if not img_boxes:
            continue
        for pi in range(n_patches):
            row = pi // spatial_w
            col = pi  % spatial_w
            px1 = col * patch_w
            py1 = row * patch_h
            px2 = px1 + patch_w
            py2 = py1 + patch_h
            patch_area = patch_w * patch_h

            total_overlap = 0.0
            for box in img_boxes:
                ix1 = max(px1, box["x1"])
                iy1 = max(py1, box["y1"])
                ix2 = min(px2, box["x2"])
                iy2 = min(py2, box["y2"])
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                total_overlap += inter
            overlap[img_i, pi] = min(total_overlap / patch_area, 1.0)

    return overlap


@torch.no_grad()
def run_spatial_grounding(
    confirmed: list[int],
    datasets: list[tuple[str, torch.Tensor]],
    sae,
    device: torch.device,
    n_patches: int,
    rsna_bbox_csv:  Optional[str],
    siim_bbox_csv:  Optional[str],
    batch_size: int = 4096,
) -> dict[int, dict[str, float]]:
    """Compute Pearson r between per-patch activation magnitude and bbox overlap.

    Args:
        confirmed:      confirmed universal latent indices
        datasets:       list of (name, activations [N_i, d_model])
        sae:            trained SAE in eval mode
        device:         torch device
        n_patches:      tokens per image
        rsna_bbox_csv:  path to RSNA stage_2_train_labels.csv (or None)
        siim_bbox_csv:  path to SIIM train_image_level.csv    (or None)
        batch_size:     token-level batch size

    Returns:
        dict[latent_idx -> {"rsna_r": float, "siim_r": float, "pooled_r": float}]
    """
    spatial_h = spatial_w = int(n_patches ** 0.5)

    bbox_datasets: dict[str, tuple[dict, torch.Tensor]] = {}

    for ds_name, acts in datasets:
        csv_path = rsna_bbox_csv if ds_name == "rsna" else (
                   siim_bbox_csv if ds_name in ("siim_covid", "siim") else None)
        if csv_path is None:
            continue

        print(f"  Loading bboxes for {ds_name} …")
        bboxes = load_bbox_csv(csv_path, ds_name)
        n_images = acts.shape[0] // n_patches
        image_ids = [str(i) for i in range(n_images)]   # sequential ids
        overlap = compute_patch_bbox_overlap(
            bboxes, image_ids, spatial_h, spatial_w
        )  # [n_images, n_patches]

        # Encode all patches to get per-patch activations
        N = acts.shape[0]
        patch_acts = torch.zeros(N, sae.d_sae)
        for start in tqdm(range(0, N, batch_size), desc=f"    {ds_name} spatial", leave=False):
            end = min(start + batch_size, N)
            codes = sae.encode(acts[start:end].to(device)).cpu()
            patch_acts[start:end] = codes

        # Reshape to [n_images, n_patches, d_sae]
        patch_acts = patch_acts.reshape(n_images, n_patches, sae.d_sae)
        bbox_datasets[ds_name] = (overlap, patch_acts)

    if not bbox_datasets:
        print("  [spatial] No bbox CSVs provided — skipping spatial grounding.")
        return {}

    spatial_results: dict[int, dict[str, float]] = {}

    for lat_idx in tqdm(confirmed, desc="  Spatial grounding"):
        entry: dict[str, float] = {}
        all_act, all_overlap = [], []

        for ds_name, (overlap, patch_acts) in bbox_datasets.items():
            act_flat     = patch_acts[:, :, lat_idx].flatten().numpy()
            overlap_flat = overlap.flatten().numpy()
            # Only include images that have at least one bbox
            has_bbox = overlap.sum(dim=1) > 0
            if has_bbox.sum() < 10:
                continue
            act_sel     = patch_acts[has_bbox, :, lat_idx].flatten().numpy()
            overlap_sel = overlap[has_bbox].flatten().numpy()
            if act_sel.std() < 1e-8 or overlap_sel.std() < 1e-8:
                continue
            r = float(np.corrcoef(act_sel, overlap_sel)[0, 1])
            entry[f"{ds_name}_r"] = round(r, 4)
            all_act.extend(act_sel.tolist())
            all_overlap.extend(overlap_sel.tolist())

        if all_act and len(all_act) > 10:
            a, b = np.array(all_act), np.array(all_overlap)
            if a.std() > 1e-8 and b.std() > 1e-8:
                entry["pooled_r"] = round(float(np.corrcoef(a, b)[0, 1]), 4)

        spatial_results[lat_idx] = entry

    return spatial_results


# ══════════════════════════════════════════════════════════════════════════════
# Output
# ══════════════════════════════════════════════════════════════════════════════

def save_results(
    out_dir: Path,
    freq_per_ds:      dict[str, torch.Tensor],
    n_tokens:         dict[str, int],
    candidates:       list[int],
    active_datasets:  dict[int, list[str]],
    cosine_info:      dict[int, dict],
    confirmed:        list[int],
    probe_results:    dict[int, dict[str, float]],
    spatial_results:  dict[int, dict[str, float]],
    flagged_indices:  set[int],
    args:             argparse.Namespace,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "threshold_frequency":   FREQUENCY_THRESHOLD,
        "threshold_cosine":      args.cosine_threshold,
        "min_active_datasets":   args.min_datasets,
        "n_scanner_flagged":     len(flagged_indices),
        "n_level1_candidates":   len(candidates),
        "n_level2_confirmed":    len(confirmed),
        "dataset_token_counts":  {k: v for k, v in n_tokens.items()},
        "confirmed_indices":     confirmed,
    }
    (out_dir / "universality_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"  ✓ universality_summary.json")

    # Per-latent detail CSV
    csv_path = out_dir / "universal_candidates.csv"
    with open(csv_path, "w", newline="") as f:
        ds_names = list(freq_per_ds.keys())
        probe_cols = sorted({k for r in probe_results.values() for k in r})
        spatial_cols = sorted({k for r in spatial_results.values() for k in r})
        cols = (
            ["latent_idx", "level1_pass", "level2_pass", "active_datasets",
             "min_cosine_sim"]
            + [f"freq_{d}" for d in ds_names]
            + probe_cols
            + spatial_cols
        )
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()

        confirmed_set = set(confirmed)
        for lat_idx in candidates:
            row: dict = {
                "latent_idx":      lat_idx,
                "level1_pass":     1,
                "level2_pass":     int(lat_idx in confirmed_set),
                "active_datasets": "|".join(active_datasets.get(lat_idx, [])),
                "min_cosine_sim":  cosine_info.get(lat_idx, {}).get("min_sim", ""),
            }
            for d in ds_names:
                row[f"freq_{d}"] = f"{freq_per_ds[d][lat_idx].item():.5f}"
            for k in probe_cols:
                row[k] = probe_results.get(lat_idx, {}).get(k, "")
            for k in spatial_cols:
                row[k] = spatial_results.get(lat_idx, {}).get(k, "")
            writer.writerow(row)

    print(f"  ✓ universal_candidates.csv  ({len(candidates)} Level-1, {len(confirmed)} confirmed)")

    # Full per-latent activation frequency JSON for downstream use
    freq_json: dict[str, dict[str, float]] = {}
    for lat_idx in confirmed:
        freq_json[str(lat_idx)] = {
            ds: round(freq_per_ds[ds][lat_idx].item(), 6)
            for ds in freq_per_ds
        }
    (out_dir / "confirmed_latent_frequencies.json").write_text(
        json.dumps(freq_json, indent=2))
    print(f"  ✓ confirmed_latent_frequencies.json")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="TD-3 universality evaluation (Level 1 + Level 2 + probe + spatial).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # SAE checkpoint
    ap.add_argument("--checkpoint_dir", required=True,
                    help="SAE checkpoint directory (sae_weights.safetensors + cfg.json).")

    # Activation inputs — mutually exclusive formats
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--activations", action="append", metavar="PATH:LABEL",
                     help="Per-dataset activations file and label (repeat 4×). "
                          "Matryoshka format.")
    grp.add_argument("--merged_activations", metavar="PATH",
                     help="Single merged activations file with dataset_ids tensor. "
                          "BatchTopK format.")

    ap.add_argument("--dataset_names", nargs="+",
                    default=["chexpert", "rsna", "siim_covid", "tb"],
                    help="Dataset names in dataset_id order (merged format only).")

    # Scanner audit
    ap.add_argument("--scanner_audit", metavar="PATH",
                    help="scanner_latent_audit.json from scanner_latent_audit.py. "
                         "Flagged latents are excluded from all analysis.")

    # Label CSVs (optional — enables canonical class match + cross-dataset probe)
    ap.add_argument("--label_csv", action="append", metavar="DATASET:PATH",
                    help="Label CSV for canonical class matching and cross-dataset probe. "
                         "Repeat per dataset. E.g. --label_csv chexpert:/path/train.csv")

    # Bbox CSVs (optional — enables spatial grounding)
    ap.add_argument("--rsna_bbox_csv",  metavar="PATH",
                    help="RSNA stage_2_train_labels.csv for spatial grounding.")
    ap.add_argument("--siim_bbox_csv",  metavar="PATH",
                    help="SIIM train_image_level.csv for spatial grounding.")

    # Thresholds
    ap.add_argument("--freq_threshold",  type=float, default=FREQUENCY_THRESHOLD,
                    help="Level 1: minimum per-dataset token activation frequency.")
    ap.add_argument("--min_datasets",    type=int,   default=2,
                    help="Level 1: minimum number of datasets a latent must be active on.")
    ap.add_argument("--cosine_threshold", type=float, default=0.70,
                    help="Level 2: minimum pairwise d_model centroid cosine similarity.")

    # Matryoshka shell (optional)
    ap.add_argument("--matryoshka_shell", type=int, default=None, metavar="K",
                    help="If set, restrict Matryoshka SAE encode() to the top-K shell "
                         "(e.g. --matryoshka_shell 24). Ignored for BatchTopK checkpoints.")

    # Runtime
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)

    print("=" * 65)
    print("  TD-3 Universality Evaluation")
    print("=" * 65)

    # ── Load SAE ──────────────────────────────────────────────────────────
    print(f"\n[1/7] Loading SAE from {args.checkpoint_dir} …")
    sae = load_sae(args.checkpoint_dir, device=str(device))
    sae.eval()
    print(f"      d_model={sae.d_model}  d_sae={sae.d_sae}  k={sae.k}")

    # Matryoshka shell wrapper — restrict encode() to top-K shell if requested.
    # MatryoshkaSAE.encode() already accepts a k kwarg; we monkey-patch the
    # default so all downstream functions transparently use the shell budget.
    if args.matryoshka_shell is not None:
        shell_k = args.matryoshka_shell
        _orig_encode = sae.encode
        sae.encode = lambda x, k=shell_k: _orig_encode(x, k=k)
        print(f"      Matryoshka shell: restricting encode() to top-{shell_k} features")

    # ── Load activations ──────────────────────────────────────────────────
    print(f"\n[2/7] Loading activations …")
    datasets = load_dataset_activations(args)
    for name, acts in datasets:
        print(f"      {name}: {acts.shape[0]:,} tokens  ({acts.shape[0] // 49} images est.)")

    # Infer n_patches from metadata or default to 49 (7×7 at pm_index=2)
    n_patches = 49

    # ── Scanner audit ─────────────────────────────────────────────────────
    flagged_indices: set[int] = set()
    if args.scanner_audit:
        with open(args.scanner_audit) as f:
            audit = json.load(f)
        flagged_indices = set(audit.get("flagged_indices", []))
        print(f"\n[3/7] Scanner audit: excluding {len(flagged_indices)} flagged latents.")
    else:
        print(f"\n[3/7] No scanner audit provided — all latents eligible.")

    # ── Level 1 ───────────────────────────────────────────────────────────
    print(f"\n[4/7] Level 1: activation frequency (threshold={args.freq_threshold:.1%}) …")
    freq_per_ds, n_tokens = compute_frequency_per_dataset(
        datasets, sae, device, batch_size=args.batch_size)
    candidates, active_datasets = apply_level1_filter(
        freq_per_ds, n_tokens, flagged_indices,
        threshold=args.freq_threshold, min_datasets=args.min_datasets)
    print(f"      Level 1 candidates: {len(candidates)} / {sae.d_sae - len(flagged_indices)}")

    if not candidates:
        print("  No Level 1 candidates — exiting.")
        save_results(out_dir, freq_per_ds, n_tokens, [], {}, {}, [], {}, {}, flagged_indices, args)
        return

    # ── Level 2 ───────────────────────────────────────────────────────────
    print(f"\n[5/7] Level 2: d_model centroid cosine similarity (threshold={args.cosine_threshold}) …")
    centroids = compute_centroids_per_dataset(
        datasets, candidates, sae, device, batch_size=args.batch_size)
    confirmed, cosine_info = apply_level2_filter(
        candidates, active_datasets, centroids, cosine_threshold=args.cosine_threshold)
    print(f"      Level 2 confirmed:  {len(confirmed)} / {len(candidates)}")

    # ── Cross-dataset probe ───────────────────────────────────────────────
    print(f"\n[6/7] Cross-dataset probe …")
    image_labels = load_label_csvs(args.label_csv)
    probe_results = run_cross_dataset_probe(
        confirmed, datasets, image_labels, sae, device,
        n_patches=n_patches, batch_size=args.batch_size)

    # ── Spatial grounding ─────────────────────────────────────────────────
    print(f"\n[7/7] Spatial grounding …")
    spatial_results = run_spatial_grounding(
        confirmed, datasets, sae, device, n_patches,
        rsna_bbox_csv=args.rsna_bbox_csv,
        siim_bbox_csv=args.siim_bbox_csv,
        batch_size=args.batch_size)

    # ── Save ──────────────────────────────────────────────────────────────
    print(f"\nSaving results to {out_dir} …")
    save_results(out_dir, freq_per_ds, n_tokens, candidates, active_datasets,
                 cosine_info, confirmed, probe_results, spatial_results,
                 flagged_indices, args)

    print(f"\n{'=' * 65}")
    print(f"  Done.")
    print(f"  Level 1 candidates : {len(candidates)}")
    print(f"  Level 2 confirmed  : {len(confirmed)}")
    if probe_results:
        mean_auroc = np.mean([v for r in probe_results.values() for v in r.values()])
        print(f"  Mean probe AUROC   : {mean_auroc:.3f}")
    if spatial_results:
        pooled_rs = [r["pooled_r"] for r in spatial_results.values() if "pooled_r" in r]
        if pooled_rs:
            print(f"  Mean pooled r      : {np.mean(pooled_rs):.3f}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
