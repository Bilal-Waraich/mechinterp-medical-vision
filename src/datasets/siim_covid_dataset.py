"""SIIM-COVID19 chest X-ray dataset for SAE activation harvesting.

Directory layout expected:
    <data_dir>/
        train_study_level.csv      ← study-level labels (Negative/Typical/Indeterminate/Atypical)
        train_image_level.csv      ← image-level labels + bounding boxes
        train/
            <study_id>/
                <image_id>.dcm     ← DICOM chest X-rays

Bounding boxes are stored as a space-separated string in the 'boxes' column:
    "x_min y_min x_max y_max label x_min y_min x_max y_max label ..."
Coordinates are in the ORIGINAL image pixel space (~1024×1024 DICOM resolution).
Scale factors are computed per-image to map to 224×224 at patch validation time.

View filter: PA only (where AP/PA metadata is available). SIIM-COVID19 is
predominantly PA, but the image_level CSV includes a 'view' column in some
versions; we filter where possible.

Canonical group mapping (TD-3):
    Airspace opacity ← opacity, consolidation (Typical/Indeterminate presentations)
    No finding       ← Negative for Pneumonia

Usage in harvest_all.py:
    from datasets.siim_covid_dataset import SIIMCovidDataset
    dataset = SIIMCovidDataset(data_dir, n_samples=10_000, seed=42)

Bounding-box access (Phase 3 patch validation):
    dataset.bbox_records  — list of dicts: {image_id, boxes: [{x_min,y_min,x_max,y_max,label}], orig_w, orig_h}
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

PATCH_SIZE = 224

# Canonical group index (TD-3, matches CheXpert canonical groups)
CANONICAL_GROUPS = [
    "airspace_opacity",      # 0
    "pleural",               # 1
    "cardiac_mediastinal",   # 2
    "atelectasis",           # 3
    "vascular_edema",        # 4
    "no_finding",            # 5
    "other",                 # 6
]

# SIIM-COVID19 study labels → canonical group
_STUDY_LABEL_TO_CANONICAL = {
    "Negative for Pneumonia":  5,   # no_finding
    "Typical Appearance":      0,   # airspace_opacity (COVID opacity)
    "Indeterminate Appearance": 0,  # airspace_opacity (ambiguous opacity)
    "Atypical Appearance":     0,   # airspace_opacity (atypical)
}


class SIIMCovidDataset(Dataset):
    """PA-filtered SIIM-COVID19 chest X-rays with opacity labels and bboxes.

    Parameters
    ----------
    data_dir : str or Path
        Root of the SIIM-COVID19 dataset containing CSVs and train/ folder.
    n_samples : int or None
        Subsample to this many images (deterministic via seed).
    seed : int
        Random seed for subsampling.

    Attributes
    ----------
    image_paths   : list[Path]
    labels        : list[int]   — canonical group index per image
    class_names   : list[str]
    bbox_records  : list[dict]  — bbox metadata for Phase 3 patch validation
                                  {image_id, boxes: [{x_min,y_min,x_max,y_max,label}], orig_w, orig_h}
    """

    class_names: list[str] = CANONICAL_GROUPS

    def __init__(
        self,
        data_dir: str | Path,
        n_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        data_dir = Path(data_dir)

        records = self._parse_csvs(data_dir)

        if n_samples is not None and n_samples < len(records):
            rng = random.Random(seed)
            records = rng.sample(records, n_samples)

        self.image_paths: list[Path] = [r["path"] for r in records]
        self.labels: list[int] = [r["canonical"] for r in records]
        self.bbox_records: list[dict] = [r["bbox_record"] for r in records]

        n_with_boxes = sum(1 for r in self.bbox_records if r["boxes"])
        print(
            f"[SIIMCovidDataset] {len(self.image_paths):,} images "
            f"({n_with_boxes:,} with bboxes)  from {data_dir}",
            flush=True,
        )

    # ── Dataset protocol ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        tensor = _load_image(self.image_paths[idx])
        return tensor, self.labels[idx]

    # ── Private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _parse_csvs(data_dir: Path) -> list[dict]:
        """Build per-image records from study-level and image-level CSVs."""
        import csv

        # 1. Study-level labels: study_id → canonical group
        study_labels: dict[str, int] = {}
        study_csv = data_dir / "train_study_level.csv"
        if study_csv.exists():
            with open(study_csv, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sid = row.get("id", "").replace("_study", "").strip()
                    label_str = _pick_study_label(row)
                    study_labels[sid] = _STUDY_LABEL_TO_CANONICAL.get(label_str, 0)

        # 2. Image-level: image_id, bboxes, optional view
        img_csv = data_dir / "train_image_level.csv"
        if not img_csv.exists():
            raise FileNotFoundError(f"train_image_level.csv not found in {data_dir}")

        image_records: dict[str, dict] = {}  # image_id → record
        with open(img_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_id = row.get("id", "").replace("_image", "").strip()

                # View filter: keep PA where metadata exists
                view = row.get("view", "").strip().upper()
                if view and view not in ("PA", "FRONTAL", ""):
                    continue  # skip non-PA when view info available

                # Parse bboxes from 'boxes' column (space-separated)
                boxes = _parse_boxes(row.get("boxes", "").strip())

                # Study id: image_id encodes study path
                study_id = _extract_study_id(img_id, row)
                canonical = study_labels.get(study_id, 0)

                image_records[img_id] = {
                    "image_id": img_id,
                    "study_id": study_id,
                    "canonical": canonical,
                    "boxes_raw": boxes,
                }

        # 3. Resolve file paths under train/
        # Build a stem→path index once so we never rglob per-image on NFS.
        train_dir = data_dir / "train"
        print(f"[SIIMCovidDataset] Indexing files under {train_dir} ...", flush=True)
        path_index: dict[str, Path] = {}
        for p in train_dir.rglob("*"):
            if p.suffix.lower() in (".dcm", ".jpg", ".jpeg", ".png"):
                path_index[p.stem] = p
        print(f"[SIIMCovidDataset] Indexed {len(path_index):,} files.", flush=True)

        records = []
        for img_id, rec in image_records.items():
            path = path_index.get(img_id)
            if path is None:
                continue

            # Estimate original image dimensions from max bbox coordinate
            orig_w, orig_h = _estimate_orig_dims(rec["boxes_raw"])

            bbox_record = {
                "image_id": img_id,
                "boxes": rec["boxes_raw"],
                "orig_w": orig_w,
                "orig_h": orig_h,
            }

            records.append({
                "path": path,
                "canonical": rec["canonical"],
                "bbox_record": bbox_record,
            })

        if not records:
            raise RuntimeError(
                f"No usable images found in {data_dir}. "
                "Check that train/ folder and CSVs are present."
            )

        return records


# ── Module-level helpers ───────────────────────────────────────────────────────

def _pick_study_label(row: dict) -> str:
    """Return the positive label for a study-level CSV row."""
    for col in ("Negative for Pneumonia", "Typical Appearance",
                "Indeterminate Appearance", "Atypical Appearance"):
        if row.get(col, "").strip() == "1":
            return col
    return "Negative for Pneumonia"


def _parse_boxes(boxes_str: str) -> list[dict]:
    """Parse SIIM bbox string into list of box dicts.

    Format: "x_min y_min x_max y_max label [x_min y_min x_max y_max label ...]"
    Returns list of {x_min, y_min, x_max, y_max, label} or [] if no boxes / None.
    """
    if not boxes_str or boxes_str.lower() == "none":
        return []
    tokens = boxes_str.split()
    boxes = []
    i = 0
    while i + 4 < len(tokens):
        try:
            x_min = float(tokens[i])
            y_min = float(tokens[i + 1])
            x_max = float(tokens[i + 2])
            y_max = float(tokens[i + 3])
            label = tokens[i + 4] if i + 4 < len(tokens) else "opacity"
            boxes.append({"x_min": x_min, "y_min": y_min,
                           "x_max": x_max, "y_max": y_max,
                           "label": label})
            i += 5
        except (ValueError, IndexError):
            break
    return boxes


def _extract_study_id(img_id: str, row: dict) -> str:
    """Extract study id from image id or row metadata."""
    # SIIM image ids often encode study: <study_id>/<image_id>
    if "/" in img_id:
        return img_id.split("/")[0]
    # Fall back to StudyInstanceUID column if present
    study = row.get("StudyInstanceUID", "").strip()
    return study if study else img_id


def _find_image_path(train_dir: Path, img_id: str) -> Path | None:
    """Search for DICOM or converted image under train_dir."""
    # Try direct: train_dir/<img_id>.dcm
    for ext in (".dcm", ".jpg", ".jpeg", ".png"):
        p = train_dir / (img_id + ext)
        if p.exists():
            return p
    # Try nested: train_dir/*/<img_id>.*
    for ext in (".dcm", ".jpg", ".jpeg", ".png"):
        matches = list(train_dir.rglob(img_id + ext))
        if matches:
            return matches[0]
    return None


def _estimate_orig_dims(boxes: list[dict]) -> tuple[int, int]:
    """Estimate original image dimensions from max bbox coordinates."""
    if not boxes:
        return 1024, 1024  # SIIM default DICOM size
    max_x = max(b["x_max"] for b in boxes)
    max_y = max(b["y_max"] for b in boxes)
    # Add 5% margin as estimate
    return int(max_x * 1.05), int(max_y * 1.05)


def _load_image(path: Path) -> torch.Tensor:
    """Load DICOM or standard image as normalised [3, 224, 224] float32 tensor."""
    if path.suffix.lower() == ".dcm":
        try:
            import pydicom
        except ImportError:
            raise ImportError("pydicom required for DICOM loading: pip install pydicom")
        dcm = pydicom.dcmread(str(path))
        arr = dcm.pixel_array.astype(np.float32)
        lo, hi = arr.min(), arr.max()
        arr = (arr - lo) / (hi - lo + 1e-8)
        img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    else:
        with Image.open(path) as img:
            img = img.convert("L")

    if img.size != (PATCH_SIZE, PATCH_SIZE):
        img = img.resize((PATCH_SIZE, PATCH_SIZE), Image.LANCZOS)
    img_rgb = img.convert("RGB")
    arr = np.array(img_rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)
