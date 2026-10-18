"""RSNA Pneumonia Detection Challenge dataset for SAE activation harvesting.

Directory layout expected under <data_dir>:
    <data_dir>/          ← pass this as data_dir (stage_2_train_images/)
        *.dcm            ← DICOM chest X-ray files (~26 000 images)

Labels:
    Each DICOM filename encodes a patient UID only; the binary
    pneumonia/normal label is stored separately in a CSV.  Because
    we harvest ALL images without per-image label lookup (to keep the
    loader self-contained), every image receives label 0 (unlabelled).
    Downstream visualisation uses patch-level activation values only,
    so the label is not used during SAE training or visualisation.

Normalisation: pixel array scaled to [0, 1] float32 then replicated to
    RGB (chest X-rays are greyscale; the encoder expects 3-channel input).
    UMedPT was trained on [0, 1] intensities, NOT ImageNet-normalised.

Images are resized to 224×224 with LANCZOS resampling.
No background filter — chest X-rays always contain content.

Usage in harvest_all.py:
    from datasets.pneumonia_dataset import PneumoniaDataset
    dataset = PneumoniaDataset(data_dir)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

PATCH_SIZE = 224


class PneumoniaDataset(Dataset):
    """All DICOM chest X-rays in the given directory.

    Parameters
    ----------
    data_dir : str or Path
        Path to the directory containing *.dcm files, e.g.
        ``$SAE_INPUT_ROOT/rsna-pneumonia-detection-challenge/
          stage_2_train_images``

    Attributes
    ----------
    image_paths : list[Path]
    labels      : list[int]  — all zeros (unlabelled; not used by SAE)
    class_names : list[str]
    """

    class_names: list[str] = ["xray"]

    def __init__(
        self,
        data_dir: str | Path,
        n_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

        # Collect paths lazily — stop early once we have enough to avoid
        # statting all ~26k DICOM files on slow NFS mounts.
        limit = n_samples if n_samples is not None else 10_000_000
        paths: list[Path] = []
        for ext in ("*.dcm", "*.png", "*.jpg", "*.jpeg"):
            for p in data_dir.glob(ext):
                paths.append(p)
                if len(paths) >= limit * 4:  # oversample then subsample for randomness
                    break
            if paths:
                break

        if not paths:
            raise RuntimeError(
                f"No DICOM or image files found in {data_dir}. "
                f"Expected *.dcm files from the RSNA Pneumonia Detection Challenge."
            )

        if n_samples is not None and len(paths) > n_samples:
            import random as _random
            rng = _random.Random(seed)
            paths = rng.sample(paths, n_samples)

        self.image_paths = paths
        self.labels: list[int] = [0] * len(self.image_paths)

        print(f"[PneumoniaDataset] {len(self.image_paths):,} images from {data_dir}", flush=True)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path = self.image_paths[idx]
        label = self.labels[idx]

        if path.suffix.lower() == ".dcm":
            tensor = self._load_dicom(path)
        else:
            tensor = self._load_standard(path)

        return tensor, label

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _load_dicom(path: Path) -> torch.Tensor:
        try:
            import pydicom
        except ImportError:
            raise ImportError(
                "pydicom is required to load DICOM files. "
                "Install it with: pip install pydicom"
            )
        dcm = pydicom.dcmread(str(path))
        arr = dcm.pixel_array.astype(np.float32)

        # Normalise to [0, 1]
        lo, hi = arr.min(), arr.max()
        if hi > lo:
            arr = (arr - lo) / (hi - lo)
        else:
            arr = np.zeros_like(arr)

        # Resize to 224×224
        img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
        if img.size != (PATCH_SIZE, PATCH_SIZE):
            img = img.resize((PATCH_SIZE, PATCH_SIZE), Image.LANCZOS)

        # Grey → RGB by replicating channel
        img_rgb = img.convert("RGB")
        tensor = torch.from_numpy(
            np.array(img_rgb, dtype=np.float32) / 255.0
        ).permute(2, 0, 1)  # [3, H, W]
        return tensor

    @staticmethod
    def _load_standard(path: Path) -> torch.Tensor:
        with Image.open(path) as img:
            img = img.convert("RGB")
            if img.size != (PATCH_SIZE, PATCH_SIZE):
                img = img.resize((PATCH_SIZE, PATCH_SIZE), Image.LANCZOS)
            arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)
