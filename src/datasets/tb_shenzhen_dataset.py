"""TB Shenzhen chest X-ray dataset for SAE activation harvesting.

Directory layout expected (two accepted conventions):
    Convention A — filename-encoded labels:
        <data_dir>/
            CHNCXR_XXXX_0.png    ← normal (suffix _0)
            CHNCXR_XXXX_1.png    ← TB positive (suffix _1)

    Convention B — companion CSV:
        <data_dir>/
            *.png / *.jpg
            labels.csv           ← columns: filename, label (0=normal, 1=TB)

The dataset has ~700 images, heavily imbalanced (~340 TB, ~360 normal).
Because of the small size, ALL images are used (no n_samples cap applied
unless explicitly overridden).

Augmentation is applied during __getitem__ to reduce memorisation risk
from the 14× oversampling needed to match larger datasets in stratified
batching. Augmentations (training mode only):
    - Random horizontal flip (p=0.5)
    - ±10% brightness jitter
    - Random crop (scale 0.85–1.0) + resize back to 224×224

Canonical group mapping (TD-3):
    Airspace opacity ← TB positive (infiltrates, cavities)
    No finding       ← Normal / TB negative

Usage in harvest_all.py:
    from datasets.tb_shenzhen_dataset import TBShenzhenDataset
    dataset = TBShenzhenDataset(data_dir, augment=True)
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

PATCH_SIZE = 224

CANONICAL_GROUPS = [
    "airspace_opacity",      # 0 — TB infiltrate / positive
    "pleural",               # 1
    "cardiac_mediastinal",   # 2
    "atelectasis",           # 3
    "vascular_edema",        # 4
    "no_finding",            # 5 — TB negative / normal
    "other",                 # 6
]

# TB label → canonical group index
_TB_TO_CANONICAL = {0: 5, 1: 0}  # normal → no_finding, TB → airspace_opacity


class TBShenzhenDataset(Dataset):
    """All TB Shenzhen chest X-rays with optional augmentation.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing PNG/JPEG images.
    augment : bool
        If True, apply random flip, brightness jitter, and random crop
        during __getitem__. Use True during harvest to reduce memorisation.
    n_samples : int or None
        If given, subsample (deterministic via seed). Usually None for TB
        since all 700 images are always used.
    seed : int
        Random seed for subsampling.

    Attributes
    ----------
    image_paths : list[Path]
    labels      : list[int]  — 0 = normal, 1 = TB positive
    class_names : list[str]
    """

    class_names: list[str] = CANONICAL_GROUPS

    def __init__(
        self,
        data_dir: str | Path,
        augment: bool = True,
        n_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

        self.augment = augment
        pairs = self._discover_images(data_dir)

        if not pairs:
            raise RuntimeError(
                f"No images found in {data_dir}. Expected PNG/JPEG files with "
                "filename suffix _0 (normal) or _1 (TB), or a labels.csv."
            )

        if n_samples is not None and n_samples < len(pairs):
            rng = random.Random(seed)
            pairs = rng.sample(pairs, n_samples)

        self.image_paths: list[Path] = [p for p, _ in pairs]
        self._raw_labels: list[int] = [l for _, l in pairs]
        self.labels: list[int] = [_TB_TO_CANONICAL[l] for l in self._raw_labels]

        n_pos = sum(self._raw_labels)
        n_neg = len(self._raw_labels) - n_pos
        print(
            f"[TBShenzhenDataset] {len(self.image_paths):,} images "
            f"(TB={n_pos}, normal={n_neg}, augment={augment})  from {data_dir}"
        )

    # ── Dataset protocol ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        tensor = _load_image(self.image_paths[idx], augment=self.augment)
        return tensor, self.labels[idx]

    # ── Private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _discover_images(data_dir: Path) -> list[tuple[Path, int]]:
        """Return (path, raw_label) pairs from Convention A or B."""
        # Convention B: labels.csv takes priority
        csv_path = data_dir / "labels.csv"
        if csv_path.exists():
            return _load_from_csv(csv_path, data_dir)

        # Convention A: filename suffix _0 / _1
        pairs: list[tuple[Path, int]] = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"):
            for p in data_dir.glob(ext):
                stem = p.stem
                if stem.endswith("_1"):
                    pairs.append((p, 1))
                elif stem.endswith("_0"):
                    pairs.append((p, 0))
                # Files without suffix are skipped (metadata, etc.)

        return sorted(pairs, key=lambda x: x[0].name)


def _load_from_csv(csv_path: Path, data_dir: Path) -> list[tuple[Path, int]]:
    """Load image paths and labels from a companion CSV."""
    import csv as _csv

    pairs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            filename = row.get("filename", "").strip()
            try:
                label = int(row.get("label", "0").strip())
            except ValueError:
                label = 0
            p = data_dir / filename
            if p.exists():
                pairs.append((p, label))
    return pairs


def _load_image(path: Path, augment: bool = False) -> torch.Tensor:
    """Load and optionally augment a TB Shenzhen image.

    Augmentations (applied only when augment=True):
      - Random horizontal flip (p=0.5)
      - ±10% brightness jitter
      - Random crop (scale 0.85–1.0) then resize to 224×224
    """
    with Image.open(path) as img:
        img = img.convert("L")  # ensure greyscale

    if augment:
        # Random horizontal flip
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Brightness jitter ±10%
        factor = 1.0 + random.uniform(-0.10, 0.10)
        img = ImageEnhance.Brightness(img).enhance(factor)

        # Random crop: sample a scale in [0.85, 1.0], then crop and resize
        scale = random.uniform(0.85, 1.0)
        w, h = img.size
        crop_w = int(w * scale)
        crop_h = int(h * scale)
        left = random.randint(0, w - crop_w)
        top = random.randint(0, h - crop_h)
        img = img.crop((left, top, left + crop_w, top + crop_h))

    if img.size != (PATCH_SIZE, PATCH_SIZE):
        img = img.resize((PATCH_SIZE, PATCH_SIZE), Image.LANCZOS)

    img_rgb = img.convert("RGB")
    arr = np.array(img_rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)
