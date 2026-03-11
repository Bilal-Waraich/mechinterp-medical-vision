"""Brain Tumour MRI dataset (Kaggle: brain-tumor-classification-mri).

Four tumour classes stored as JPEG/PNG images in per-class subdirectories:

    <data_dir>/
        glioma_tumor/      gg (i).jpg
        meningioma_tumor/  m (i).jpg
        no_tumor/          <various>.jpg / image (i).jpg
        pituitary_tumor/   p (i).jpg

Normalisation: divide by 255 only → [0, 1] float32.
UMedPT was trained on [0, 1] intensities, NOT ImageNet-normalised.

Images are resized to 224×224 with LANCZOS resampling if they differ from
PATCH_SIZE × PATCH_SIZE.  No background filter is applied — MRI images are
always content-bearing.

Usage in harvest_activations.py
--------------------------------
    from datasets.brain_cancer_dataset import BrainTumorDataset
    dataset = BrainTumorDataset(data_dir="/path/to/Training")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# ── Constants ─────────────────────────────────────────────────────────────────

PATCH_SIZE = 224

# Canonical folder-name → (display name, 0-based label index)
# Matching is done by checking whether the folder name *contains* the key,
# so minor casing differences or extra suffixes are handled gracefully.
_FOLDER_MAP: dict[str, tuple[str, int]] = {
    "glioma_tumor":     ("glioma",     0),
    "meningioma_tumor": ("meningioma", 1),
    "no_tumor":         ("no_tumor",   2),
    "pituitary_tumor":  ("pituitary",  3),
}

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ── Dataset ───────────────────────────────────────────────────────────────────

class BrainTumorDataset(Dataset):
    """All MRI images from a single split (Training or Testing) directory.

    Parameters
    ----------
    data_dir : str or Path
        Path to the split root, e.g.
        ``/deep_learning/input/data/mmm/brain_tumor_kagglesets/
          brain-tumor-classification-mri/Training``
        Must contain the four class subdirectories listed above.

    Attributes
    ----------
    labels : list[int]
        0-based class index per image (0=glioma, 1=meningioma,
        2=no_tumor, 3=pituitary).
    class_names : list[str]
        ["glioma", "meningioma", "no_tumor", "pituitary"]
    image_paths : list[Path]
        Absolute path to each image file, in the same order as labels.
    """

    class_names: list[str] = ["glioma", "meningioma", "no_tumor", "pituitary"]

    def __init__(self, data_dir: str | Path) -> None:
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

        self.image_paths: list[Path] = []
        self.labels: list[int] = []

        class_counts: dict[str, int] = {}

        # Discover subdirectories and map them to class labels
        subdirs = sorted(p for p in data_dir.iterdir() if p.is_dir())
        if not subdirs:
            raise RuntimeError(f"No subdirectories found in {data_dir}")

        for subdir in subdirs:
            class_name, label_idx = self._resolve_class(subdir.name)
            if class_name is None:
                print(f"  [WARN] Unrecognised folder '{subdir.name}' — skipping")
                continue

            images = self._collect_images(subdir)
            if not images:
                print(f"  [WARN] No images found in {subdir}")
                continue

            self.image_paths.extend(images)
            self.labels.extend([label_idx] * len(images))
            class_counts[class_name] = len(images)

        total = len(self.image_paths)
        print(f"[BrainTumorDataset] {total:,} images from {data_dir}")
        print(f"  Class breakdown:")
        for name in self.class_names:
            n = class_counts.get(name, 0)
            print(f"    {name:<14}: {n:>6,}")

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _resolve_class(folder_name: str) -> tuple[str | None, int]:
        """Map a folder name to (class_name, label_idx), or (None, -1)."""
        lower = folder_name.lower()
        for key, (name, idx) in _FOLDER_MAP.items():
            if key in lower:
                return name, idx
        return None, -1

    @staticmethod
    def _collect_images(directory: Path) -> list[Path]:
        """Return all image files in *directory* (non-recursive)."""
        files = [
            p for p in sorted(directory.iterdir())
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
        ]
        return files

    # ── PyTorch Dataset interface ─────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path = self.image_paths[idx]
        label = self.labels[idx]

        with Image.open(path) as img:
            img = img.convert("RGB")
            if img.size != (PATCH_SIZE, PATCH_SIZE):
                img = img.resize((PATCH_SIZE, PATCH_SIZE), Image.LANCZOS)
            patch_np = np.array(img, dtype=np.float32) / 255.0

        tensor = torch.from_numpy(patch_np).permute(2, 0, 1)  # [3, H, W]
        return tensor, label
