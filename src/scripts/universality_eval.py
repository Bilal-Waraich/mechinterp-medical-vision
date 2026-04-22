"""TD-3 Two-Level Universality Metric for X-ray SAE.

Implements the full universality evaluation pipeline described in
docs/methodology.md §1.2–1.5:

  Step 1  Load SAE + per-dataset activations (merged or separate file formats)
  Step 2  Apply scanner-audit exclusion list (optional)
  Step 3  Level 1  — per-token activation frequency ≥1% on ≥2 datasets
                      + canonical pathology class match (if label CSVs provided)
  Step 4  Level 2  — per-dataset d_model centroid cosine similarity ≥0.70
  Step 5  Cross-dataset probe  — leave-one-dataset-out logistic AUROC (optional)
  Step 6  Spatial grounding    — Pearson r vs bbox patch overlap with
                                 Bonferroni-corrected p-values (optional)
  Step 7  Split detection      — Jaccard co-activation over top-K images
                                 (methodology §1.2 Property 4)
  Step 8  Save outputs

Input formats
-------------
Merged (BatchTopK jobs):
  Single activations.safetensors containing:
    - "activations"  [N_total, d_model]
    - "dataset_ids"  [N_total]  (int: 0=chexpert, 1=rsna, 2=siim_covid, 3=tb)

Separate (Matryoshka jobs):
  Four files, each with "activations" [N_i, d_model]; no dataset_ids tensor.
  Pass via --activations PATH:LABEL (repeat 4×).

Usage — BatchTopK (merged file)
--------------------------------
python src/scripts/universality_eval.py \\
    --checkpoint_dir /deep_learning/output/UMed-SAE/batch-topk/layer-2/xray-conglomerate/xray_v1/checkpoints/step_0200000 \\
    --merged_activations /deep_learning/output/UMed-SAE/batch-topk/layer-2/xray-conglomerate/xray_v1/activations.safetensors \\
    --dataset_names chexpert rsna siim_covid tb \\
    --scanner_audit /deep_learning/output/.../audit/scanner_latent_audit.json \\
    --label_csv chexpert:/deep_learning/input/.../CheXpert-v1.0/train.csv \\
    --label_csv rsna:/deep_learning/input/.../stage_2_train_labels.csv \\
    --rsna_bbox_csv /deep_learning/input/.../stage_2_train_labels.csv \\
    --siim_bbox_csv /deep_learning/input/.../siim-covid19-detection/train_image_level.csv \\
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
from safetensors import safe_open
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

def _read_safetensors_metadata(path: str) -> dict[str, str]:
    with safe_open(str(path), framework="pt") as f:
        return dict(f.metadata() or {})


def load_dataset_activations(
    args: argparse.Namespace,
) -> list[tuple[str, torch.Tensor, list[str]]]:
    """Load per-dataset activation tensors and their per-image filename stems.

    Returns list of (dataset_name, activations [N_i, d_model], image_stems).
    ``image_stems`` has length ``N_i // n_patches`` and is used downstream to
    align each image with its label CSV / bbox CSV entry by filename.

    Supports both merged (BatchTopK) and separate per-dataset file formats.
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

        meta = _read_safetensors_metadata(args.merged_activations)
        stems_by_name: dict[str, list[str]] = (
            json.loads(meta["image_stems"]) if "image_stems" in meta else {}
        )

        names    = args.dataset_names
        datasets = []
        for i, name in enumerate(names):
            mask = ds_ids == i
            acts = acts_all[mask]
            stems = stems_by_name.get(name, [])
            datasets.append((name, acts, stems))
        return datasets

    # Separate per-dataset files
    datasets = []
    for entry in args.activations:
        path, label = entry.rsplit(":", 1)
        raw = load_file(str(path))
        acts = raw["activations"].float()
        meta = _read_safetensors_metadata(path)
        # Per-dataset file may store either `image_stems` as a flat JSON list
        # (single-dataset harvest) or as a dict keyed by label (rare).
        stems: list[str] = []
        if "image_stems" in meta:
            parsed = json.loads(meta["image_stems"])
            if isinstance(parsed, dict):
                stems = parsed.get(label, [])
            elif isinstance(parsed, list):
                stems = parsed
        datasets.append((label, acts, stems))
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
    datasets: list[tuple[str, torch.Tensor, list[str]]],
    sae,
    device: torch.device,
    batch_size: int = 4096,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Compute per-latent activation frequency for each dataset.

    Args:
        datasets: list of (name, activations [N_i, d_model], image_stems)
        sae: trained SAE in eval mode
        device: torch device
        batch_size: token-level batch size for SAE encode

    Returns:
        freq_per_ds: dict[name -> Tensor [d_sae]]  (fraction of tokens where latent fires)
        n_tokens:    dict[name -> int]
    """
    freq_per_ds: dict[str, torch.Tensor] = {}
    n_tokens: dict[str, int] = {}

    for ds_name, acts, _ in datasets:
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


@torch.no_grad()
def compute_per_image_max(
    datasets: list[tuple[str, torch.Tensor, list[str]]],
    sae,
    device: torch.device,
    n_patches: int,
    batch_size: int = 4096,
) -> dict[str, torch.Tensor]:
    """Encode every patch and collapse to per-image max activations.

    For each dataset produces a ``[n_images, d_sae]`` tensor of max activations
    over the n_patches tokens of every image.  This per-image view is shared
    between the canonical class-match gate (Level 1) and the cross-dataset
    logistic probe (Step 5), so both agree on which images each latent "fires
    on" most strongly.

    Returns
    -------
    dict[dataset_name -> Tensor [n_images, d_sae]]
    """
    result: dict[str, torch.Tensor] = {}
    imgs_per_batch = max(1, batch_size // n_patches)
    d_sae = sae.d_sae

    for ds_name, acts, _ in datasets:
        N = acts.shape[0]
        n_images = N // n_patches
        feat_max = torch.zeros(n_images, d_sae)

        ts = datetime.now().strftime("%H:%M:%S")
        print(f"  [{ts}] {ds_name}: {n_images:,} images → per-image feature_max …")

        for img_start in tqdm(range(0, n_images, imgs_per_batch), desc=f"    {ds_name}", leave=False):
            img_end = min(img_start + imgs_per_batch, n_images)
            tok_start = img_start * n_patches
            tok_end   = img_end   * n_patches
            codes = sae.encode(acts[tok_start:tok_end].to(device)).cpu()
            codes_img = codes.reshape(img_end - img_start, n_patches, d_sae)
            feat_max[img_start:img_end] = codes_img.max(dim=1).values

        result[ds_name] = feat_max

    return result


def compute_dominant_canonical_class(
    per_image_max:  dict[str, torch.Tensor],
    image_labels:   dict[str, dict[str, str]],
    stems_by_ds:    dict[str, list[str]],
    top_k:          int = 50,
) -> dict[str, list[Optional[str]]]:
    """For each (dataset, latent) pick the modal canonical class in the top-K.

    A latent's "dominant canonical class" on dataset D is the mode of the
    canonical-group labels of the top-K images in D where the latent fires
    most strongly (``feature_max`` rank).  Images whose stem is absent from
    the label CSV are ignored (not silently counted as ``no_finding``, which
    would bias the mode).  If no labeled image makes the top-K, the dominant
    class is ``None`` and any latent relying on this dataset will fail class
    match.

    Returns
    -------
    dict[dataset_name -> list[Optional[str]]]  (length d_sae)
    """
    dominant: dict[str, list[Optional[str]]] = {}

    for ds_name, feat_max in per_image_max.items():
        stems   = stems_by_ds.get(ds_name, [])
        labels  = image_labels.get(ds_name, {})
        n_images, d_sae = feat_max.shape

        # Pre-map stems → canonical group (None when missing from CSV)
        stem_to_group: list[Optional[str]] = [labels.get(s) for s in stems]
        dom: list[Optional[str]] = [None] * d_sae

        if n_images == 0 or not labels or len(stems) != n_images:
            dominant[ds_name] = dom
            continue

        # For each latent: get top-K activating images (only those actually firing)
        k = min(top_k, n_images)
        top_vals, top_idx = torch.topk(feat_max, k, dim=0)  # [k, d_sae]

        for lat in range(d_sae):
            votes: dict[str, int] = {}
            for rank in range(k):
                if top_vals[rank, lat].item() <= 0:
                    break  # topk is sorted descending → remaining are all zero
                grp = stem_to_group[int(top_idx[rank, lat].item())]
                if grp is None:
                    continue
                votes[grp] = votes.get(grp, 0) + 1
            if votes:
                dom[lat] = max(votes.items(), key=lambda kv: kv[1])[0]

        dominant[ds_name] = dom

    return dominant


def apply_level1_filter(
    freq_per_ds:      dict[str, torch.Tensor],
    n_tokens:         dict[str, int],
    flagged_indices:  set[int],
    threshold:        float = FREQUENCY_THRESHOLD,
    min_datasets:     int = 2,
    dominant_class:   Optional[dict[str, list[Optional[str]]]] = None,
) -> tuple[list[int], dict[int, list[str]], dict[int, Optional[str]]]:
    """Apply TD-3 Level 1: frequency threshold on ≥ min_datasets.

    When ``dominant_class`` is supplied (derived from ``--label_csv`` inputs),
    candidates must also satisfy the canonical-class-match gate from
    methodology §1.2 Property 1: the dominant canonical class must be the
    same on every dataset where the latent is active.  If a latent's active
    datasets disagree on their dominant class — or if any active dataset has
    no labeled image in its top-K — the latent is rejected.

    Returns
    -------
    candidates:       list of latent indices passing Level 1
    active_datasets:  dict[latent_idx -> list of dataset names where active]
    matched_class:    dict[latent_idx -> matched canonical class] (None when
                      class match was not enforced)
    """
    ds_names = list(freq_per_ds.keys())
    d_sae = next(iter(freq_per_ds.values())).shape[0]

    active_datasets: dict[int, list[str]] = {}
    matched_class:   dict[int, Optional[str]] = {}
    candidates: list[int] = []

    for i in range(d_sae):
        if i in flagged_indices:
            continue
        active = [ds for ds in ds_names if freq_per_ds[ds][i].item() >= threshold]
        if len(active) < min_datasets:
            continue

        # Canonical-class-match gate (only when labels were provided).
        if dominant_class is not None:
            classes = [dominant_class[ds][i] for ds in active]
            if any(c is None for c in classes):
                continue
            if len(set(classes)) > 1:
                continue
            matched_class[i] = classes[0]
        else:
            matched_class[i] = None

        candidates.append(i)
        active_datasets[i] = active

    return candidates, active_datasets, matched_class


# ══════════════════════════════════════════════════════════════════════════════
# Level 2 — Representational direction alignment
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_centroids_per_dataset(
    datasets: list[tuple[str, torch.Tensor, list[str]]],
    candidates: list[int],
    sae,
    device: torch.device,
    batch_size: int = 4096,
) -> dict[str, torch.Tensor]:
    """Compute per-latent d_model activation centroid for each dataset.

    For candidate latent i and dataset D:
      centroid[i] = mean of d_model activation vectors where latent i fires.

    Args:
        datasets:   list of (name, activations [N_i, d_model], image_stems)
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
    n_cands = len(cand_list)
    d_model = sae.d_model

    centroids: dict[str, torch.Tensor] = {}

    for ds_name, acts, _ in datasets:
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
    confirmed:     list[int],
    per_image_max: dict[str, torch.Tensor],
    image_labels:  dict[str, dict[str, str]],
    stems_by_ds:   dict[str, list[str]],
) -> dict[int, dict[str, float]]:
    """Leave-one-dataset-out logistic probe on canonical-group labels.

    For each confirmed latent, trains a logistic regression on one dataset's
    per-image max-activation and tests zero-shot on the others.  Images are
    joined to labels by the filename stems captured at harvest time, so
    activation ↔ label alignment is exact regardless of CSV row order.

    Args:
        confirmed:     confirmed universal latent indices
        per_image_max: dict[ds_name -> Tensor [n_images, d_sae]] from
                       ``compute_per_image_max`` (shared with Level 1 gate).
        image_labels:  dict[dataset_name -> dict[image_stem -> canonical_group]]
        stems_by_ds:   dict[dataset_name -> list of per-image stems]

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

    per_ds_max:    dict[str, torch.Tensor] = {}
    per_ds_labels: dict[str, list[str]] = {}

    for ds_name, feature_max in per_image_max.items():
        if ds_name not in image_labels:
            continue
        stems = stems_by_ds.get(ds_name, [])
        n_images = feature_max.shape[0]
        if len(stems) != n_images:
            print(f"  [probe] {ds_name}: {len(stems)} stems vs {n_images} images — "
                  f"activations were harvested without stems metadata; skipping.")
            continue

        # Align labels to activations via the harvest-time image stems.  Each
        # stem is looked up in the dataset's label CSV; images whose stem is
        # absent from the CSV default to "no_finding".
        labels_dict = image_labels[ds_name]
        label_list  = [labels_dict.get(stem, "no_finding") for stem in stems]

        per_ds_max[ds_name]    = feature_max
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
    datasets: list[tuple[str, torch.Tensor, list[str]]],
    sae,
    device: torch.device,
    n_patches: int,
    rsna_bbox_csv:  Optional[str],
    siim_bbox_csv:  Optional[str],
    batch_size: int = 4096,
    alpha: float = 0.05,
) -> tuple[dict[int, dict[str, float]], dict[str, float | int]]:
    """Compute Pearson r (+ p-value) between per-patch activation and bbox overlap.

    Implements methodology §1.2 Property 3.  For every confirmed latent and
    every bbox-annotated dataset we compute the per-image Pearson correlation
    between patch activation magnitude and bbox-overlap fraction, then a
    pooled correlation over images from all annotated datasets combined.
    Each correlation carries a two-sided p-value from
    ``scipy.stats.pearsonr``; these are then corrected for multiple testing
    with a Bonferroni adjustment so the family-wise α matches
    ``--bonferroni_alpha``:

        α_pooled  = α / n_confirmed
        α_per_ds  = α / (n_confirmed × n_bbox_datasets)

    A latent "passes spatial grounding" only when its pooled p-value clears
    the corrected pooled threshold.

    Returns
    -------
    spatial_results   : dict[lat_idx -> dict with keys
                        {ds}_r, {ds}_p, {ds}_pass,
                        pooled_r, pooled_p, pooled_pass]
    bonferroni_summary: dict with alpha, alpha_pooled, alpha_per_ds,
                        n_pooled_tested, n_pooled_pass, n_per_ds_tested,
                        n_per_ds_pass, n_bbox_datasets
    """
    from scipy.stats import pearsonr

    spatial_h = spatial_w = int(n_patches ** 0.5)

    bbox_datasets: dict[str, tuple[dict, torch.Tensor]] = {}

    for ds_name, acts, image_stems in datasets:
        csv_path = rsna_bbox_csv if ds_name == "rsna" else (
                   siim_bbox_csv if ds_name in ("siim_covid", "siim") else None)
        if csv_path is None:
            continue

        print(f"  Loading bboxes for {ds_name} …")
        bboxes = load_bbox_csv(csv_path, ds_name)
        n_images = acts.shape[0] // n_patches

        if len(image_stems) != n_images:
            print(f"  [spatial] {ds_name}: {len(image_stems)} stems vs {n_images} images — "
                  f"activations were harvested without stems metadata; skipping.")
            continue

        overlap = compute_patch_bbox_overlap(
            bboxes, image_stems, spatial_h, spatial_w
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
        return {}, {"n_pooled_tested": 0, "n_pooled_pass": 0,
                    "n_per_ds_tested": 0, "n_per_ds_pass": 0,
                    "n_bbox_datasets": 0, "alpha": alpha,
                    "alpha_pooled": alpha, "alpha_per_ds": alpha}

    n_bbox_datasets = len(bbox_datasets)
    n_conf = max(1, len(confirmed))
    alpha_pooled = alpha / n_conf
    alpha_per_ds = alpha / (n_conf * n_bbox_datasets)

    spatial_results: dict[int, dict[str, float]] = {}
    n_pooled_tested = n_pooled_pass = 0
    n_per_ds_tested = n_per_ds_pass = 0

    for lat_idx in tqdm(confirmed, desc="  Spatial grounding"):
        entry: dict[str, float] = {}
        all_act, all_overlap = [], []

        for ds_name, (overlap, patch_acts) in bbox_datasets.items():
            has_bbox = overlap.sum(dim=1) > 0
            if has_bbox.sum() < 10:
                continue
            act_sel     = patch_acts[has_bbox, :, lat_idx].flatten().numpy()
            overlap_sel = overlap[has_bbox].flatten().numpy()
            if act_sel.std() < 1e-8 or overlap_sel.std() < 1e-8:
                continue
            try:
                r, p = pearsonr(act_sel, overlap_sel)
            except ValueError:
                continue
            entry[f"{ds_name}_r"]    = round(float(r), 4)
            entry[f"{ds_name}_p"]    = float(p)
            entry[f"{ds_name}_pass"] = int(p < alpha_per_ds and r > 0)
            n_per_ds_tested += 1
            n_per_ds_pass   += entry[f"{ds_name}_pass"]

            all_act.extend(act_sel.tolist())
            all_overlap.extend(overlap_sel.tolist())

        if all_act and len(all_act) > 10:
            a = np.asarray(all_act)
            b = np.asarray(all_overlap)
            if a.std() > 1e-8 and b.std() > 1e-8:
                try:
                    r_p, p_p = pearsonr(a, b)
                    entry["pooled_r"]    = round(float(r_p), 4)
                    entry["pooled_p"]    = float(p_p)
                    entry["pooled_pass"] = int(p_p < alpha_pooled and r_p > 0)
                    n_pooled_tested += 1
                    n_pooled_pass   += entry["pooled_pass"]
                except ValueError:
                    pass

        spatial_results[lat_idx] = entry

    bonferroni_summary = {
        "alpha":             alpha,
        "alpha_pooled":      alpha_pooled,
        "alpha_per_ds":      alpha_per_ds,
        "n_bbox_datasets":   n_bbox_datasets,
        "n_confirmed":       n_conf,
        "n_pooled_tested":   n_pooled_tested,
        "n_pooled_pass":     n_pooled_pass,
        "n_per_ds_tested":   n_per_ds_tested,
        "n_per_ds_pass":     n_per_ds_pass,
    }
    return spatial_results, bonferroni_summary


# ══════════════════════════════════════════════════════════════════════════════
# Property 4 — Co-activation / split detection
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_split_detection(
    confirmed:           list[int],
    per_image_max:       Optional[dict[str, torch.Tensor]],
    datasets:            list[tuple[str, torch.Tensor, list[str]]],
    sae,
    device:              torch.device,
    n_patches:           int,
    batch_size:          int = 4096,
    top_k:               int = 50,
    jaccard_threshold:   float = 0.50,
    out_dir:             Optional[Path] = None,
) -> list[dict]:
    """Detect pairs of confirmed latents that encode the same concept (split).

    Implements methodology §1.2 Property 4.  For every confirmed latent we
    take its top-``top_k`` activating images across the combined dataset
    and compute pairwise Jaccard similarity.  Pairs above
    ``jaccard_threshold`` are reported as potential feature splits — the
    SAE has allocated two dictionary atoms to what looks like a single
    concept.  The combined feature_max matrix is also saved as a heatmap
    (``co_activation_heatmap.png``) when ``out_dir`` is provided.

    Returns
    -------
    list of dicts with keys {"i", "j", "jaccard"} sorted by Jaccard desc.
    """
    from visualisation.coactivation import (
        compute_coactivation_matrix,
        find_coactivating_pairs,
        render_coactivation_heatmap,
    )

    if not confirmed or len(confirmed) < 2:
        print("  [split] fewer than 2 confirmed latents — skipping.")
        return []

    # Combine per-image feature_max across datasets.  Re-compute if the caller
    # did not supply it (e.g. the user passed no --label_csv flags).
    if per_image_max is None:
        print("  [split] Computing per-image feature_max …")
        per_image_max = compute_per_image_max(
            datasets, sae, device, n_patches=n_patches, batch_size=batch_size)

    combined = torch.cat(list(per_image_max.values()), dim=0)  # [n_total_images, d_sae]
    stats = {"feature_max": combined}

    coact = compute_coactivation_matrix(
        stats, top_k=top_k, feature_indices=confirmed)
    pairs_raw = find_coactivating_pairs(
        coact, feature_indices=confirmed, threshold=jaccard_threshold)
    pairs = [{"i": int(i), "j": int(j), "jaccard": round(float(s), 4)}
             for (i, j, s) in pairs_raw]

    if out_dir is not None:
        try:
            heatmap_path = out_dir / "co_activation_heatmap.png"
            fig = render_coactivation_heatmap(
                coact, feature_indices=confirmed, save_path=str(heatmap_path))
            import matplotlib.pyplot as _plt
            _plt.close(fig)
            print(f"  [split] heatmap → {heatmap_path}")
        except Exception as e:
            print(f"  [split] heatmap failed: {e}")

    print(f"  [split] {len(pairs)} pair(s) with Jaccard ≥ {jaccard_threshold}")
    return pairs


# ══════════════════════════════════════════════════════════════════════════════
# Output
# ══════════════════════════════════════════════════════════════════════════════

def save_results(
    out_dir: Path,
    freq_per_ds:          dict[str, torch.Tensor],
    n_tokens:             dict[str, int],
    candidates:           list[int],
    active_datasets:      dict[int, list[str]],
    cosine_info:          dict[int, dict],
    confirmed:            list[int],
    probe_results:        dict[int, dict[str, float]],
    spatial_results:      dict[int, dict[str, float]],
    spatial_bonferroni:   dict,
    split_pairs:          list[dict],
    flagged_indices:      set[int],
    args:                 argparse.Namespace,
    matched_class:        Optional[dict[int, Optional[str]]] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    matched_class = matched_class or {}
    class_match_enforced = any(v is not None for v in matched_class.values())

    summary = {
        "threshold_frequency":     FREQUENCY_THRESHOLD,
        "threshold_cosine":        args.cosine_threshold,
        "min_active_datasets":     args.min_datasets,
        "class_match_enforced":    class_match_enforced,
        "class_match_top_k":       args.class_match_top_k,
        "bonferroni_alpha":        args.bonferroni_alpha,
        "split_jaccard_threshold": args.split_jaccard_threshold,
        "n_scanner_flagged":       len(flagged_indices),
        "n_level1_candidates":     len(candidates),
        "n_level2_confirmed":      len(confirmed),
        "n_spatial_bonferroni_pass": spatial_bonferroni.get("n_pooled_pass", 0),
        "n_spatial_bonferroni_tested": spatial_bonferroni.get("n_pooled_tested", 0),
        "n_split_pairs":           len(split_pairs),
        "spatial_bonferroni":      spatial_bonferroni,
        "dataset_token_counts":    {k: v for k, v in n_tokens.items()},
        "confirmed_indices":       confirmed,
    }
    (out_dir / "universality_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"  ✓ universality_summary.json")

    # Split pairs (feature splitting, methodology §1.2 Property 4)
    (out_dir / "split_pairs.json").write_text(json.dumps(split_pairs, indent=2))
    print(f"  ✓ split_pairs.json  ({len(split_pairs)} pairs)")

    # Per-latent detail CSV
    csv_path = out_dir / "universal_candidates.csv"
    with open(csv_path, "w", newline="") as f:
        ds_names = list(freq_per_ds.keys())
        probe_cols = sorted({k for r in probe_results.values() for k in r})
        spatial_cols = sorted({k for r in spatial_results.values() for k in r})
        base_cols = [
            "latent_idx", "level1_pass", "level2_pass", "active_datasets",
            "min_cosine_sim",
        ]
        if class_match_enforced:
            base_cols.append("matched_canonical_class")
        cols = (
            base_cols
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
            if class_match_enforced:
                row["matched_canonical_class"] = matched_class.get(lat_idx) or ""
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
        description=(
            "TD-3 universality evaluation: Level 1 (freq + class match) + "
            "Level 2 (centroid cosine) + cross-dataset probe + spatial grounding "
            "(Bonferroni-corrected) + split detection."
        ),
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
    ap.add_argument("--class_match_top_k", type=int, default=50,
                    help="Level 1 class match: the dominant canonical class on each "
                         "active dataset is the mode over this many top-activating images.")
    ap.add_argument("--bonferroni_alpha", type=float, default=0.05,
                    help="Family-wise α for Bonferroni correction on spatial grounding r.")
    ap.add_argument("--split_jaccard_threshold", type=float, default=0.50,
                    help="Property 4: Jaccard threshold above which a pair of confirmed "
                         "latents is flagged as a potential feature split.")
    ap.add_argument("--split_top_k", type=int, default=50,
                    help="Property 4: size of the per-latent top-K image set used for "
                         "the Jaccard split-detection heuristic.")

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
    print(f"\n[1/8] Loading SAE from {args.checkpoint_dir} …")
    sae = load_sae(args.checkpoint_dir, device=str(device))
    sae.eval()
    print(f"      d_model={sae.d_model}  d_sae={sae.d_sae}  k={sae.k}")

    # ── Load activations ──────────────────────────────────────────────────
    print(f"\n[2/8] Loading activations …")
    datasets = load_dataset_activations(args)

    # Infer n_patches from file metadata; fall back to 49 (7×7, pm_index=2/3).
    # Only swallow the narrow, expected failure modes — anything else (file
    # missing, corrupt header) should surface rather than silently masquerade
    # as a pm_index=2/3 harvest.
    n_patches = 49
    _acts_path = args.merged_activations or (args.activations[0].rsplit(":", 1)[0] if args.activations else None)
    if _acts_path:
        try:
            _meta = _read_safetensors_metadata(_acts_path)
            n_patches = int(_meta["spatial_h"]) * int(_meta["spatial_w"])
        except (KeyError, ValueError, TypeError):
            pass  # metadata missing/non-numeric — keep default 49

    for name, acts, _ in datasets:
        print(f"      {name}: {acts.shape[0]:,} tokens  ({acts.shape[0] // n_patches} images est.)")

    # ── Scanner audit ─────────────────────────────────────────────────────
    flagged_indices: set[int] = set()
    if args.scanner_audit:
        with open(args.scanner_audit) as f:
            audit = json.load(f)
        flagged_indices = set(audit.get("flagged_indices", []))
        print(f"\n[3/8] Scanner audit: excluding {len(flagged_indices)} flagged latents.")
    else:
        print(f"\n[3/8] No scanner audit provided — all latents eligible.")

    # ── Label CSVs + shared per-image feature_max ─────────────────────────
    # If labels are available, compute per-image feature_max once up-front;
    # it feeds both the Level 1 class-match gate (methodology §1.2 P1) and
    # the cross-dataset probe (§1.4), so the two agree on what "top-K
    # activating images" means per latent/dataset.
    image_labels = load_label_csvs(args.label_csv)
    stems_by_ds: dict[str, list[str]] = {name: stems for name, _, stems in datasets}

    per_image_max: dict[str, torch.Tensor] | None = None
    dominant_class: dict[str, list[str | None]] | None = None
    if image_labels:
        print(f"\n[Pre] Computing per-image feature_max for {len(image_labels)} labelled dataset(s) …")
        per_image_max = compute_per_image_max(
            datasets, sae, device, n_patches=n_patches, batch_size=args.batch_size)
        dominant_class = compute_dominant_canonical_class(
            per_image_max, image_labels, stems_by_ds,
            top_k=args.class_match_top_k,
        )

    # ── Level 1 ───────────────────────────────────────────────────────────
    gate_desc = "frequency + class match" if dominant_class is not None else "frequency"
    print(f"\n[4/8] Level 1 ({gate_desc}, threshold={args.freq_threshold:.1%}) …")
    freq_per_ds, n_tokens = compute_frequency_per_dataset(
        datasets, sae, device, batch_size=args.batch_size)
    candidates, active_datasets, matched_class = apply_level1_filter(
        freq_per_ds, n_tokens, flagged_indices,
        threshold=args.freq_threshold, min_datasets=args.min_datasets,
        dominant_class=dominant_class)
    print(f"      Level 1 candidates: {len(candidates)} / {sae.d_sae - len(flagged_indices)}")

    if not candidates:
        print("  No Level 1 candidates — exiting.")
        save_results(out_dir, freq_per_ds, n_tokens, [], {}, {}, [], {}, {}, {},
                     [], flagged_indices, args, matched_class={})
        return

    # ── Level 2 ───────────────────────────────────────────────────────────
    print(f"\n[5/8] Level 2: d_model centroid cosine similarity (threshold={args.cosine_threshold}) …")
    centroids = compute_centroids_per_dataset(
        datasets, candidates, sae, device, batch_size=args.batch_size)
    confirmed, cosine_info = apply_level2_filter(
        candidates, active_datasets, centroids, cosine_threshold=args.cosine_threshold)
    print(f"      Level 2 confirmed:  {len(confirmed)} / {len(candidates)}")

    # ── Cross-dataset probe ───────────────────────────────────────────────
    print(f"\n[6/8] Cross-dataset probe …")
    if per_image_max is None and image_labels:
        per_image_max = compute_per_image_max(
            datasets, sae, device, n_patches=n_patches, batch_size=args.batch_size)
    probe_results = run_cross_dataset_probe(
        confirmed,
        per_image_max=per_image_max or {},
        image_labels=image_labels,
        stems_by_ds=stems_by_ds,
    )

    # ── Spatial grounding ─────────────────────────────────────────────────
    print(f"\n[7/8] Spatial grounding …")
    spatial_results, spatial_bonferroni = run_spatial_grounding(
        confirmed, datasets, sae, device, n_patches,
        rsna_bbox_csv=args.rsna_bbox_csv,
        siim_bbox_csv=args.siim_bbox_csv,
        batch_size=args.batch_size,
        alpha=args.bonferroni_alpha)

    # ── Co-activation / split detection (Property 4) ──────────────────────
    print(f"\n[8/8] Co-activation split detection (Jaccard ≥ {args.split_jaccard_threshold}) …")
    split_pairs = run_split_detection(
        confirmed,
        per_image_max=per_image_max,
        datasets=datasets,
        sae=sae,
        device=device,
        n_patches=n_patches,
        batch_size=args.batch_size,
        top_k=args.split_top_k,
        jaccard_threshold=args.split_jaccard_threshold,
        out_dir=out_dir,
    )

    # ── Save ──────────────────────────────────────────────────────────────
    print(f"\nSaving results to {out_dir} …")
    save_results(out_dir, freq_per_ds, n_tokens, candidates, active_datasets,
                 cosine_info, confirmed, probe_results, spatial_results,
                 spatial_bonferroni, split_pairs, flagged_indices, args,
                 matched_class=matched_class)

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
        n_pass = spatial_bonferroni.get("n_pooled_pass", 0)
        n_tested = spatial_bonferroni.get("n_pooled_tested", 0)
        print(f"  Pooled Bonferroni  : {n_pass} / {n_tested} pass at α={args.bonferroni_alpha}")
    print(f"  Split pairs        : {len(split_pairs)}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
