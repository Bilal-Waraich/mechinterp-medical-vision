r"""
Central path configuration for all analysis scripts.

Edit the paths in each dataset section below, then all scripts will
pick them up automatically.

Storage root on the file server:
    \\fileserver\share\deep_learning\output\UMed-SAE\batch-topk\layer-2
"""

import os
from pathlib import Path

# Auto-detect environment: Docker (Linux) mounts the volume at /deep_learning/output,
# local Windows uses the network share directly.
if os.name == "nt":
    _L2 = Path(r"\\fileserver\share\deep_learning\output\UMed-SAE\batch-topk\layer-2")
else:
    _L2 = Path("/deep_learning/output/UMed-SAE/batch-topk/layer-2")

# ── Dataset configs ────────────────────────────────────────────────────────
# breast stats is None — generate with:
#   python src/visualise.py collect \
#       --activations \\fileserver\...\tcga-breast_v1\activations.safetensors \
#       --checkpoint  \\fileserver\...\tcga-breast_v1\checkpoints\step_0050000 \
#       --output      \\fileserver\...\tcga-breast_v1\vis_outputs\stats.pt

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

# ── Matches CSVs (one per pair) ────────────────────────────────────────────
MATCHES = {
    "colon_breast":    _L2 / "cosine-similarity-colon-breast-v2/reciprocal_best_matches.csv",
    "breast_brain":    _L2 / "cosine-similarity-breast-brain-v2/reciprocal_best_matches.csv",
    "brain_pneumonia": _L2 / "cosine-similarity-brain-pneumonia-v2/reciprocal_best_matches.csv",
}

# ── Chain 1 definition ─────────────────────────────────────────────────────
# Colon F155 → Breast F2662 → Brain F1590 → Pneumonia F688
CHAIN_1 = {
    "colon":     155,
    "breast":    2662,
    "brain":     1590,
    "pneumonia": 688,
}

# ── Chain 2 definition ─────────────────────────────────────────────────────
# Colon F3153 → Breast F1487 → Brain F3494 → Pneumonia F1596
CHAIN_2 = {
    "colon":     3153,
    "breast":    1487,
    "brain":     3494,
    "pneumonia": 1596,
}

# ── Output directories ─────────────────────────────────────────────────────
OUTPUT_ROOT = _L2 / "universality_outputs"

OUTPUTS = {
    "chain_visualisation": OUTPUT_ROOT / "chain_patch_visualisation",
    "feature_ablation":    OUTPUT_ROOT / "feature_ablation",
    "null_baseline":       OUTPUT_ROOT / "null_baseline",
}

# ── Spatial config (Layer 2, pm_index=2) ──────────────────────────────────
SPATIAL_H = 7
SPATIAL_W = 7
N_PATCHES = SPATIAL_H * SPATIAL_W  # 49
D_MODEL   = 1024
D_SAE     = 4096


# ── Chain derivation ───────────────────────────────────────────────────────
# Use the hardcoded CHAIN_1 / CHAIN_2 dicts above, OR call derive_chain()
# to trace a feature programmatically through the match CSVs at runtime.
#
# Example — derive Chain 1 starting from Colon F155:
#   chain = derive_chain(starting_dataset="colon", starting_feature=155)
#   # returns {"colon": 155, "breast": 2662, "brain": 1590, "pneumonia": 688}

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
