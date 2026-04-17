r"""
Central path configuration for all analysis scripts.

Edit the paths in each section below, then all scripts will pick them up
automatically.

Storage root on the file server:
    \\fileserver\share\deep_learning\output\UMed-SAE\

X-ray experiment layout
-----------------------
  batch-topk/layer-2/xray-conglomerate/xray_v1/
    activations.safetensors            ← merged (all 4 datasets + dataset_ids)
    checkpoints/step_0200000/          ← final BatchTopK checkpoint

  matryoshka/layer-2/xray-conglomerate/xray_v1/
    activations_chexpert.safetensors   ─┐
    activations_rsna.safetensors        ├ per-dataset files (no dataset_ids)
    activations_siim.safetensors        │
    activations_tb.safetensors         ─┘
    checkpoints/step_0200000/          ← final Matryoshka checkpoint
"""

import os
from pathlib import Path

# Auto-detect environment: Docker (Linux) mounts the volume at /deep_learning/output,
# local Windows uses the network share directly.
if os.name == "nt":
    _ROOT = Path(r"\\fileserver\share\deep_learning\output\UMed-SAE")
else:
    _ROOT = Path("/deep_learning/output/UMed-SAE")

# ── Legacy (individual-dataset SAEs, Layer 2) ─────────────────────────────
_L2 = _ROOT / "batch-topk" / "layer-2"

DATASETS = {
    "colon": {
        "label":       "colon",
        "checkpoint":  _L2 / "SemiCOL/SemiCOL_v1/track_a/checkpoints/step_0050000",
        "activations": _L2 / "SemiCOL/SemiCOL_v1/activations_track_a.safetensors",
        "stats":       _L2 / "SemiCOL/SemiCOL_v1/track_a/vis_outputs/stats.pt",
        "patches_dir": _L2 / "SemiCOL/SemiCOL_v1/patches_track_a",
    },
    "breast": {
        "label":       "breast",
        "checkpoint":  _L2 / "tcga-breast/tcga-breast_v1/checkpoints/step_0050000",
        "activations": _L2 / "tcga-breast/tcga-breast_v1/activations.safetensors",
        "stats":       _L2 / "tcga-breast/tcga-breast_v1/vis_outputs/stats.pt",
        "patches_dir": _L2 / "tcga-breast/tcga-breast_v1/patches",
    },
    "brain": {
        "label":       "brain",
        "checkpoint":  _L2 / "brain-tumour/brain-tumour_v1/checkpoints/step_0050000",
        "activations": _L2 / "brain-tumour/brain-tumour_v1/activations.safetensors",
        "stats":       _L2 / "brain-tumour/brain-tumour_v1/vis_outputs/stats.pt",
        "patches_dir": _L2 / "brain-tumour/brain-tumour_v1/patches",
    },
    "pneumonia": {
        "label":       "pneumonia",
        "checkpoint":  _L2 / "pneumonia/pneumonia_v1/checkpoints/step_0050000",
        "activations": _L2 / "pneumonia/pneumonia_v1/activations.safetensors",
        "stats":       _L2 / "pneumonia/pneumonia_v1/vis_outputs/stats.pt",
        "patches_dir": _L2 / "pneumonia/pneumonia_v1/patches",
    },
}

# ── X-ray conglomerate — BatchTopK (pm_index=2, Layer 2) ──────────────────
_XRAY_BTK_L2 = _ROOT / "batch-topk" / "layer-2" / "xray-conglomerate" / "xray_v1"

XRAY_BATCHTOPK_L2 = {
    "checkpoint":  _XRAY_BTK_L2 / "checkpoints" / "step_0200000",
    # Single merged file; contains both 'activations' and 'dataset_ids' tensors.
    "activations": _XRAY_BTK_L2 / "activations.safetensors",
    "output_dir":  _XRAY_BTK_L2 / "universality",
    "d_model": 1024,
    "d_sae":   4096,
    "k":       48,
    "spatial_h": 7,
    "spatial_w": 7,
}

# ── X-ray conglomerate — Matryoshka (pm_index=2, Layer 2) ─────────────────
_XRAY_MAT_L2 = _ROOT / "matryoshka" / "layer-2" / "xray-conglomerate" / "xray_v1"

XRAY_MATRYOSHKA_L2 = {
    "checkpoint":  _XRAY_MAT_L2 / "checkpoints" / "step_0200000",
    # Four separate per-dataset files (no dataset_ids tensor).
    "activations": {
        "chexpert":  _XRAY_MAT_L2 / "activations_chexpert.safetensors",
        "rsna":      _XRAY_MAT_L2 / "activations_rsna.safetensors",
        "siim_covid":_XRAY_MAT_L2 / "activations_siim.safetensors",
        "tb":        _XRAY_MAT_L2 / "activations_tb.safetensors",
    },
    "output_dir":  _XRAY_MAT_L2 / "universality",
    "d_model": 1024,
    "d_sae":   4096,
    "k_schedule": [6, 12, 24, 48],
    "spatial_h": 7,
    "spatial_w": 7,
}

# ── X-ray conglomerate — BatchTopK (pm_index=3, Layer 3) ──────────────────
_XRAY_BTK_L3 = _ROOT / "batch-topk" / "layer-3" / "xray-conglomerate" / "xray_v1"

XRAY_BATCHTOPK_L3 = {
    "checkpoint":  _XRAY_BTK_L3 / "checkpoints" / "step_0200000",
    "activations": _XRAY_BTK_L3 / "activations.safetensors",
    "output_dir":  _XRAY_BTK_L3 / "universality",
    "d_model": 1024,
    "d_sae":   4096,
    "k":       48,
    "spatial_h": 7,
    "spatial_w": 7,
}

# ── X-ray conglomerate — Matryoshka (pm_index=3, Layer 3) ─────────────────
_XRAY_MAT_L3 = _ROOT / "matryoshka" / "layer-3" / "xray-conglomerate" / "xray_v1"

XRAY_MATRYOSHKA_L3 = {
    "checkpoint":  _XRAY_MAT_L3 / "checkpoints" / "step_0200000",
    "activations": {
        "chexpert":  _XRAY_MAT_L3 / "activations_chexpert.safetensors",
        "rsna":      _XRAY_MAT_L3 / "activations_rsna.safetensors",
        "siim_covid":_XRAY_MAT_L3 / "activations_siim.safetensors",
        "tb":        _XRAY_MAT_L3 / "activations_tb.safetensors",
    },
    "output_dir":  _XRAY_MAT_L3 / "universality",
    "d_model": 1024,
    "d_sae":   4096,
    "k_schedule": [6, 12, 24, 48],
    "spatial_h": 7,
    "spatial_w": 7,
}

# ── Label CSVs (canonical class matching in universality_eval.py) ─────────
# Pass these as: --label_csv chexpert:PATH rsna:PATH siim_covid:PATH tb:PATH
if os.name == "nt":
    _INPUT = Path(r"\\fileserver\share\deep_learning\input\data\mmm")
else:
    _INPUT = Path("/deep_learning/input/data/mmm")

XRAY_LABEL_CSVS = {
    "chexpert":   _INPUT / "chexpert" / "train.csv",
    "rsna":       _INPUT / "rsna-pneumonia-detection-challenge" / "stage_2_train_labels",
    "siim_covid": _INPUT / "siim-covid19-detection" / "train_study_level",
    "tb":         _INPUT / "shenzhen_metadata",
}

# ── Matryoshka shell to analyse (k=24 = second-tightest budget) ───────────
# Pass as --matryoshka_shell 24 to universality_eval.py.
# Rationale: k=6 is very sparse (may miss rare pathologies like TB),
# k=24 retains enough features to cover multi-finding images while still
# being well below k=48 (full), giving a meaningful universality contrast.
MATRYOSHKA_SHELL_K = 24

# ── Convenience: all four experiment configs ───────────────────────────────
XRAY_EXPERIMENTS = {
    "batchtopk_l2":   XRAY_BATCHTOPK_L2,
    "matryoshka_l2":  XRAY_MATRYOSHKA_L2,
    "batchtopk_l3":   XRAY_BATCHTOPK_L3,
    "matryoshka_l3":  XRAY_MATRYOSHKA_L3,
}

# ── Legacy: matches CSVs and feature chains (individual-dataset experiments) ─
MATCHES = {
    "colon_breast":    _L2 / "cosine-similarity-colon-breast-v2/reciprocal_best_matches.csv",
    "breast_brain":    _L2 / "cosine-similarity-breast-brain-v2/reciprocal_best_matches.csv",
    "brain_pneumonia": _L2 / "cosine-similarity-brain-pneumonia-v2/reciprocal_best_matches.csv",
}

CHAIN_1 = {"colon": 155,  "breast": 2662, "brain": 1590, "pneumonia": 688}
CHAIN_2 = {"colon": 3153, "breast": 1487, "brain": 3494, "pneumonia": 1596}

# ── Output root (legacy) ───────────────────────────────────────────────────
OUTPUT_ROOT = _L2 / "universality_outputs"
OUTPUTS = {
    "chain_visualisation": OUTPUT_ROOT / "chain_patch_visualisation",
    "feature_ablation":    OUTPUT_ROOT / "feature_ablation",
    "null_baseline":       OUTPUT_ROOT / "null_baseline",
}

# ── Spatial config (Layer 2 + 3 share same spatial dims at pm_index=2/3) ──
SPATIAL_H = 7
SPATIAL_W = 7
N_PATCHES = SPATIAL_H * SPATIAL_W  # 49
D_MODEL   = 1024
D_SAE     = 4096


def derive_chain(
    starting_dataset: str,
    starting_feature: int,
    ordered_datasets: list[str] | None = None,
) -> dict[str, int]:
    """Walk reciprocal best match CSVs to build a cross-dataset feature chain.

    Parameters
    ----------
    starting_dataset:
        Key from DATASETS for the first dataset (e.g. "colon").
    starting_feature:
        Feature index in the starting SAE.
    ordered_datasets:
        Sequence of dataset keys to follow. Defaults to
        ["colon", "breast", "brain", "pneumonia"].

    Returns
    -------
    dict mapping dataset label → feature index for the full chain.
    Stops early if a match CSV for the next pair is not found.
    """
    import csv as _csv

    if ordered_datasets is None:
        ordered_datasets = ["colon", "breast", "brain", "pneumonia"]

    pair_keys = {
        ("colon",  "breast"):    MATCHES["colon_breast"],
        ("breast", "brain"):     MATCHES["breast_brain"],
        ("brain",  "pneumonia"): MATCHES["brain_pneumonia"],
    }

    chain: dict[str, int] = {starting_dataset: starting_feature}
    start_idx = ordered_datasets.index(starting_dataset)
    current_feature = starting_feature

    for i in range(start_idx, len(ordered_datasets) - 1):
        ds_a = ordered_datasets[i]
        ds_b = ordered_datasets[i + 1]
        csv_path = pair_keys.get((ds_a, ds_b))
        if csv_path is None or not Path(csv_path).exists():
            break

        label_a = DATASETS[ds_a]["label"]
        label_b = DATASETS[ds_b]["label"]
        matched = None

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            cols = reader.fieldnames or []
            col_a = next((c for c in cols if label_a in c.lower() and "feature" in c.lower()), None)
            col_b = next((c for c in cols if label_b in c.lower() and "feature" in c.lower()), None)
            if not (col_a and col_b):
                break
            for row in reader:
                if int(row[col_a]) == current_feature:
                    matched = int(row[col_b])
                    break

        if matched is None:
            break

        chain[ds_b] = matched
        current_feature = matched

    return chain
