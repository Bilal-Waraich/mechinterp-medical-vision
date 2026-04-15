"""CheXpert chest X-ray dataset for SAE activation harvesting.

Directory layout expected:
    <data_dir>/
        train.csv          ← per-image labels + view metadata
        train/
            patient*/
                study*/
                    view*.jpg   ← JPEG images

Labels (14 pathology classes from CheXpert, Irvin et al. 2019):
    Positive  →  1
    Negative  →  0
    Uncertain → -1  (treated as 0, i.e. negative — conservative choice)
    Missing   →  0

View filter: PA (Frontal, PA) only. Lateral and AP views are excluded to avoid
the AP/PA geometric confound (concern #5 in single-modality-sae-concerns.md).

Canonical group mapping (TD-3, Fleischner Society glossary):
    Airspace opacity  ← Consolidation, Lung Opacity
    Pleural           ← Pleural Effusion, Pleural Other
    Cardiac/mediastinal ← Cardiomegaly, Enlarged Cardiomediastinum
    Atelectasis       ← Atelectasis
    Vascular/edema    ← Edema
    No finding        ← No Finding
    Other             ← Support Devices, Fracture

Usage in harvest_all.py:
    from datasets.chexpert_dataset import CheXpertDataset
    dataset = CheXpertDataset(data_dir, n_samples=10_000, seed=42)
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

PATCH_SIZE = 224

# CheXpert pathology columns in train.csv (in order)
PATHOLOGY_COLS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

# Canonical group index (TD-3)
CANONICAL_GROUPS = [
    "airspace_opacity",      # 0
    "pleural",               # 1
    "cardiac_mediastinal",   # 2
    "atelectasis",           # 3
    "vascular_edema",        # 4
    "no_finding",            # 5
    "other",                 # 6
]

# Map each pathology column → canonical group index
PATHOLOGY_TO_CANONICAL: dict[str, int] = {
    "No Finding":                   5,
    "Enlarged Cardiomediastinum":   2,
    "Cardiomegaly":                 2,
    "Lung Opacity":                 0,
    "Lung Lesion":                  0,
    "Edema":                        4,
    "Consolidation":                0,
    "Pneumonia":                    0,
    "Atelectasis":                  3,
    "Pneumothorax":                 6,
    "Pleural Effusion":             1,
    "Pleural Other":                1,
    "Fracture":                     6,
    "Support Devices":              6,
}


class CheXpertDataset(Dataset):
    """PA-only CheXpert chest X-rays with pathology labels.

    Parameters
    ----------
    data_dir : str or Path
        Root of the CheXpert directory, containing ``train.csv`` and
        the ``train/`` image tree.
    n_samples : int or None
        If given, randomly subsample this many images (deterministic via seed).
    seed : int
        Random seed for subsampling. Default 42.

    Attributes
    ----------
    image_paths    : list[Path]
    labels         : list[int]   — dominant canonical group index per image
    canonical_vecs : list[list[int]]  — full 14-dim binary label vector per image
    class_names    : list[str]   — canonical group names
    """

    class_names: list[str] = CANONICAL_GROUPS

    def __init__(
        self,
        data_dir: str | Path,
        n_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        data_dir = Path(data_dir)
        csv_path = data_dir / "train.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"train.csv not found at {csv_path}")

        rows = self._parse_csv(csv_path, data_dir)

        if n_samples is not None and n_samples < len(rows):
            rng = random.Random(seed)
            rows = rng.sample(rows, n_samples)

        self.image_paths: list[Path] = [r["path"] for r in rows]
        self.canonical_vecs: list[list[int]] = [r["label_vec"] for r in rows]
        self.labels: list[int] = [r["canonical"] for r in rows]

        print(
            f"[CheXpertDataset] {len(self.image_paths):,} PA images "
            f"(n_samples={n_samples}) from {data_dir}"
        )

    # ── Dataset protocol ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        tensor = _load_image(self.image_paths[idx])
        return tensor, self.labels[idx]

    # ── Private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _parse_csv(csv_path: Path, data_dir: Path) -> list[dict]:
        """Parse train.csv, keep PA rows, resolve label vectors."""
        import csv

        rows = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # View filter: keep Frontal/PA only
                view = row.get("Frontal/Lateral", "").strip()
                ap_pa = row.get("AP/PA", "").strip()
                if view != "Frontal" or ap_pa != "PA":
                    continue

                # Resolve image path
                # train.csv Path column is relative to data_dir parent
                rel = row.get("Path", "").strip()
                # CheXpert stores paths like "CheXpert-v1.0/train/..."
                # or just "train/..." depending on version
                img_path = data_dir / rel
                if not img_path.exists():
                    # Try stripping leading component
                    parts = Path(rel).parts
                    for i in range(len(parts)):
                        candidate = data_dir / Path(*parts[i:])
                        if candidate.exists():
                            img_path = candidate
                            break
                    else:
                        continue  # skip missing files

                # Build 14-dim label vector (uncertain → 0)
                label_vec = []
                dominant_score = -1
                dominant_canonical = 5  # default: no finding
                for col in PATHOLOGY_COLS:
                    raw = row.get(col, "").strip()
                    if raw == "1.0" or raw == "1":
                        val = 1
                    else:
                        val = 0  # -1 uncertain and 0 absent both → 0
                    label_vec.append(val)
                    if val == 1:
                        canonical = PATHOLOGY_TO_CANONICAL[col]
                        # Prefer airspace_opacity > others for dominant class
                        if dominant_score < 1 or canonical == 0:
                            dominant_canonical = canonical
                            dominant_score = 1

                rows.append({
                    "path": img_path,
                    "label_vec": label_vec,
                    "canonical": dominant_canonical,
                })

        return rows


def _load_image(path: Path) -> torch.Tensor:
    """Load a JPEG/PNG chest X-ray as a normalised [3, 224, 224] float32 tensor."""
    with Image.open(path) as img:
        img = img.convert("L")  # ensure greyscale
        if img.size != (PATCH_SIZE, PATCH_SIZE):
            img = img.resize((PATCH_SIZE, PATCH_SIZE), Image.LANCZOS)
        img_rgb = img.convert("RGB")
        arr = np.array(img_rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]
