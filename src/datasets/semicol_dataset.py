"""SemiCOL colorectal cancer dataset classes for SAE activation harvesting.

Two tracks:
  Track B (SemiCOLWeakDataset)   — WSI weak labels, used for SAE training
  Track A (SemiCOLManualDataset) — pixel-level masks, used for feature validation

Normalisation: divide by 255 only → [0, 1] float32.
UMedPT was trained on [0,1] intensities, NOT ImageNet-normalised.

Background filter: convert patch to HSV, skip if mean saturation < 0.05.

Directory layout expected under <data_dir>/DATASET_TRAIN/:
  02_WEAK/02_WEAK/DS_W_*/  <institution>_ome_tiff_{Tumor,NoTumor}/*.tiff
  01_MANUAL/DS_M_*/        <institution>_case_*/image/*.tiff
                                                mask/*.tiff
"""

from __future__ import annotations

import collections
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# ── Constants ─────────────────────────────────────────────────────────────────

PATCH_SIZE = 224
SAT_THRESHOLD = 0.05   # mean HSV-saturation below this → background

# Track A tissue class mapping (mask pixel value → class name).
# Value 9 (BACK) is skipped entirely.
# Verify with inspect_mask() before running at scale.
TISSUE_CLASSES = {
    1: "TUM",
    2: "STR",
    3: "MUS",
    4: "NORM",
    5: "LYM",
    6: "DEB",
    7: "MUC",
    8: "ADI",
    # 9: "BACK" — excluded
}
_BACK_LABEL = 9


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_background(patch_np: np.ndarray) -> bool:
    """Return True if the patch looks like background (low saturation).

    Parameters
    ----------
    patch_np : ndarray [H, W, 3] uint8
    """
    hsv = Image.fromarray(patch_np).convert("HSV")
    sat = np.array(hsv)[:, :, 1].mean() / 255.0
    return sat < SAT_THRESHOLD


def inspect_mask(path: str | Path) -> None:
    """Print unique pixel values and their counts from a mask TIFF.

    Use this to verify the tissue-class-to-pixel-value mapping (TISSUE_CLASSES
    dict above) before running the full dataset at scale.  Actual mask values
    must be confirmed against the dataset before trusting the mapping.

    Parameters
    ----------
    path : str or Path
        Path to a single mask TIFF file.

    Example
    -------
    >>> from datasets.semicol_dataset import inspect_mask
    >>> inspect_mask("/data/semicol/DATASET_TRAIN/01_MANUAL/DS_M_1/ukk_case_01/mask/tile_01.tiff")
    """
    import tiffslide

    path = Path(path)
    print(f"Inspecting mask: {path}")
    ts = tiffslide.TiffSlide(str(path))
    arr = np.array(ts.read_region((0, 0), 0, ts.level_dimensions[0]))
    ts.close()
    # Mask TIFFs may be RGBA or RGB; take first channel for label values
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    unique, counts = np.unique(arr, return_counts=True)
    total = arr.size
    print(f"  Shape : {arr.shape}")
    print(f"  dtype : {arr.dtype}")
    print(f"  {'Value':>8}  {'Count':>10}  {'%':>7}  Class")
    for v, c in zip(unique, counts):
        if int(v) == _BACK_LABEL:
            name = "BACK"
        else:
            name = TISSUE_CLASSES.get(int(v), "UNKNOWN")
        print(f"  {int(v):>8}  {c:>10}  {100 * c / total:>6.2f}%  {name}")


# ── Track B — Weak WSI dataset ────────────────────────────────────────────────

class SemiCOLWeakDataset(Dataset):
    """Non-overlapping 224×224 patches from SemiCOL Track B (OME-TIFF WSIs).

    Parameters
    ----------
    data_dir : str or Path
        Root of the SemiCOL data, e.g. $SAE_INPUT_ROOT/semicol
    max_per_slide : int
        Maximum patches to sample per WSI (stratified random cap).  Default 500.

    Attributes
    ----------
    labels : list[int]
        Patch-level label (0=NoTumor, 1=Tumor) — one per patch.
    institutions : list[str]
        Institution identifier per patch.
    slide_ids : list[str]
        Slide filename stem per patch.
    class_names : list[str]
        ["NoTumor", "Tumor"]
    """

    class_names = ["NoTumor", "Tumor"]

    _LABEL_MAP = {"Tumor": 1, "NoTumor": 0}

    _INSTITUTION_MAP = {
        "DS_W_1":  "UKK",
        "DS_W_2A": "WNS_HAMA",
        "DS_W_2B": "WNS_LEICA",
        "DS_W_3":  "LMU",
        "DS_W_4":  "UBERN",
    }

    def __init__(self, data_dir: str | Path, max_per_slide: int = 500):
        import tiffslide
        self._tiffslide = tiffslide

        data_dir = Path(data_dir)
        weak_root = data_dir / "DATASET_TRAIN" / "02_WEAK" / "02_WEAK"

        # (slide_path_str, row, col, label)
        self._patches: list[tuple[str, int, int, int]] = []
        self.labels: list[int] = []
        self.institutions: list[str] = []
        self.slide_ids: list[str] = []

        inst_counts: dict[str, int] = collections.defaultdict(int)
        class_counts: dict[str, int] = collections.defaultdict(int)

        for ds_dir in sorted(weak_root.iterdir()):
            if not ds_dir.is_dir():
                continue
            institution = self._INSTITUTION_MAP.get(ds_dir.name, ds_dir.name)

            for label_dir in sorted(ds_dir.iterdir()):
                if not label_dir.is_dir():
                    continue
                label = None
                for suffix, lbl in self._LABEL_MAP.items():
                    if suffix in label_dir.name:
                        label = lbl
                        break
                if label is None:
                    continue
                label_name = self.class_names[label]

                tiff_files = sorted(label_dir.glob("*.tiff")) + sorted(label_dir.glob("*.tif"))
                for slide_path in tiff_files:
                    n = self._index_slide(
                        slide_path, label, institution, label_name, max_per_slide,
                        inst_counts, class_counts,
                    )
                    _ = n

        total = sum(inst_counts.values())
        print(f"[SemiCOLWeakDataset] {total:,} patches  ({len(self._patches)} after background filter)")
        print(f"  Institution breakdown:")
        for inst, n in sorted(inst_counts.items()):
            print(f"    {inst:<12}: {n:>8,}")
        print(f"  Class breakdown:")
        for cls, n in sorted(class_counts.items()):
            print(f"    {cls:<8}: {n:>8,}")

    def _index_slide(
        self,
        slide_path: Path,
        label: int,
        institution: str,
        label_name: str,
        max_per_slide: int,
        inst_counts: dict,
        class_counts: dict,
    ) -> int:
        try:
            ts = self._tiffslide.TiffSlide(str(slide_path))
            W, H = ts.level_dimensions[0]
            ts.close()
        except Exception as exc:
            print(f"  [WARN] Cannot open {slide_path.name}: {exc}")
            return 0

        # Build full grid of candidate (row, col) offsets
        candidates: list[tuple[int, int]] = [
            (r, c)
            for r in range(0, H - PATCH_SIZE + 1, PATCH_SIZE)
            for c in range(0, W - PATCH_SIZE + 1, PATCH_SIZE)
        ]

        if len(candidates) > max_per_slide:
            rng = np.random.default_rng(seed=abs(hash(slide_path.name)) & 0xFFFF_FFFF)
            chosen = rng.choice(len(candidates), size=max_per_slide, replace=False)
            candidates = [candidates[i] for i in sorted(chosen)]

        n_added = 0
        try:
            ts = self._tiffslide.TiffSlide(str(slide_path))
            for r, c in candidates:
                region = ts.read_region((c, r), 0, (PATCH_SIZE, PATCH_SIZE))
                patch_np = np.array(region.convert("RGB"))
                if _is_background(patch_np):
                    continue
                self._patches.append((str(slide_path), r, c, label))
                self.labels.append(label)
                self.institutions.append(institution)
                self.slide_ids.append(slide_path.stem)
                inst_counts[institution] += 1
                class_counts[label_name] += 1
                n_added += 1
            ts.close()
        except Exception as exc:
            print(f"  [WARN] Error reading {slide_path.name}: {exc}")

        return n_added

    def __len__(self) -> int:
        return len(self._patches)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        slide_path, row, col, label = self._patches[idx]
        ts = self._tiffslide.TiffSlide(slide_path)
        region = ts.read_region((col, row), 0, (PATCH_SIZE, PATCH_SIZE))
        ts.close()
        patch_np = np.array(region.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(patch_np).permute(2, 0, 1)  # [3, H, W]
        return tensor, label


# ── Track A — Manual segmentation dataset ─────────────────────────────────────

class SemiCOLManualDataset(Dataset):
    """Non-overlapping 224×224 patches from SemiCOL Track A (annotated tiles).

    Patch label = mode of non-zero mask pixel values in the corresponding
    224×224 region.  Patches where >50% of mask pixels are zero are skipped.
    Background class (pixel value 9) is also skipped.

    Parameters
    ----------
    data_dir : str or Path
        Root of the SemiCOL data.
    max_per_case : int
        Maximum patches per case (stratified random cap).  Default 2000.

    Attributes
    ----------
    labels : list[int]
        0-based tissue class index per patch (0=TUM … 7=ADI).
    case_ids : list[str]
        Case folder name per patch.
    institutions : list[str]
        Institution identifier per patch.
    class_names : list[str]
        ["TUM", "STR", "MUS", "NORM", "LYM", "DEB", "MUC", "ADI"]
    """

    # pixel value (1-8) → 0-based label index
    _PIXEL_TO_IDX: dict[int, int] = {v: v - 1 for v in TISSUE_CLASSES}
    class_names: list[str] = [TISSUE_CLASSES[v] for v in sorted(TISSUE_CLASSES)]

    _INSTITUTION_MAP = {
        "DS_M_1": "UKK",
        "DS_M_2": "LMU",
    }

    def __init__(self, data_dir: str | Path, max_per_case: int = 2000):
        import tiffslide
        self._tiffslide = tiffslide

        data_dir = Path(data_dir)
        manual_root = data_dir / "DATASET_TRAIN" / "01_MANUAL"

        # (image_path_str, mask_path_str, row, col, label_idx)
        self._patches: list[tuple[str, str, int, int, int]] = []
        self.labels: list[int] = []
        self.case_ids: list[str] = []
        self.institutions: list[str] = []

        inst_counts: dict[str, int] = collections.defaultdict(int)
        class_counts: dict[str, int] = collections.defaultdict(int)

        if not manual_root.exists():
            print(f"  [ERROR] manual_root does not exist: {manual_root}")
            if data_dir.exists():
                print(f"  [DEBUG] Contents of data_dir ({data_dir}):")
                for p in sorted(data_dir.iterdir()):
                    print(f"    {p.name}  ({'dir' if p.is_dir() else 'file'})")
            else:
                print(f"  [ERROR] data_dir does not exist either: {data_dir}")
        else:
            ds_dirs = [p for p in manual_root.iterdir() if p.is_dir()]
            print(f"  [DEBUG] manual_root={manual_root}  ({len(ds_dirs)} dataset dirs: {[d.name for d in sorted(ds_dirs)]})")

        for ds_dir in sorted(manual_root.iterdir()) if manual_root.exists() else []:
            if not ds_dir.is_dir():
                continue
            institution = self._INSTITUTION_MAP.get(ds_dir.name, ds_dir.name)

            for case_dir in sorted(ds_dir.iterdir()):
                if not case_dir.is_dir():
                    continue
                img_dir = case_dir / "image"
                msk_dir = case_dir / "mask"
                if not img_dir.exists() or not msk_dir.exists():
                    print(f"  [DEBUG] Skipping {case_dir.name}: missing image/={img_dir.exists()} mask/={msk_dir.exists()}")
                    continue

                img_files = (
                    sorted(img_dir.glob("*.tiff"))
                    + sorted(img_dir.glob("*.tif"))
                    + sorted(img_dir.glob("*.png"))
                )
                if not img_files:
                    print(f"  [DEBUG] No image files in {img_dir}")
                    continue

                msk_files = (
                    sorted(msk_dir.glob("*.tiff"))
                    + sorted(msk_dir.glob("*.tif"))
                    + sorted(msk_dir.glob("*.png"))
                )
                msk_by_stem = {f.stem: f for f in msk_files}

                if not msk_files:
                    print(f"  [DEBUG] No mask files in {msk_dir}")
                    continue

                case_patches_before = len(self._patches)
                for img_path in img_files:
                    added = len(self._patches) - case_patches_before
                    if added >= max_per_case:
                        break
                    mask_path = msk_by_stem.get(img_path.stem + "-labelled") or msk_by_stem.get(img_path.stem)
                    if mask_path is None:
                        continue

                    self._index_tile(
                        img_path, mask_path, case_dir.name, institution,
                        max_per_case - added, inst_counts, class_counts,
                    )

        total = sum(inst_counts.values())
        print(f"[SemiCOLManualDataset] {total:,} patches  ({len(self._patches)} after filters)")
        print(f"  Institution breakdown:")
        for inst, n in sorted(inst_counts.items()):
            print(f"    {inst:<8}: {n:>8,}")
        print(f"  Class breakdown:")
        for cls in self.class_names:
            n = class_counts.get(cls, 0)
            print(f"    {cls:<6}: {n:>8,}")

    @staticmethod
    def _read_region_pil(path: Path, row: int, col: int) -> Image.Image:
        with Image.open(path) as img:
            return img.crop((col, row, col + PATCH_SIZE, row + PATCH_SIZE))

    def _index_tile(
        self,
        img_path: Path,
        mask_path: Path,
        case_id: str,
        institution: str,
        max_per_case: int,
        inst_counts: dict,
        class_counts: dict,
    ) -> None:
        use_pil = img_path.suffix.lower() == ".png"
        try:
            if use_pil:
                _img = Image.open(img_path)
                W, H = _img.size
                _img.close()
            else:
                ts_img = self._tiffslide.TiffSlide(str(img_path))
                W, H = ts_img.level_dimensions[0]
                ts_img.close()
        except Exception as exc:
            print(f"  [WARN] Cannot open image {img_path.name}: {exc}")
            return

        candidates: list[tuple[int, int]] = [
            (r, c)
            for r in range(0, H - PATCH_SIZE + 1, PATCH_SIZE)
            for c in range(0, W - PATCH_SIZE + 1, PATCH_SIZE)
        ]

        if len(candidates) > max_per_case:
            rng = np.random.default_rng(seed=abs(hash(img_path.name)) & 0xFFFF_FFFF)
            chosen = rng.choice(len(candidates), size=max_per_case, replace=False)
            candidates = [candidates[i] for i in sorted(chosen)]

        try:
            if use_pil:
                pil_img = Image.open(img_path)
                pil_msk = Image.open(mask_path)
            else:
                ts_img = self._tiffslide.TiffSlide(str(img_path))
                ts_msk = self._tiffslide.TiffSlide(str(mask_path))

            for r, c in candidates:
                if use_pil:
                    img_region = pil_img.crop((c, r, c + PATCH_SIZE, r + PATCH_SIZE))
                    msk_region = pil_msk.crop((c, r, c + PATCH_SIZE, r + PATCH_SIZE))
                else:
                    img_region = ts_img.read_region((c, r), 0, (PATCH_SIZE, PATCH_SIZE))
                    msk_region = ts_msk.read_region((c, r), 0, (PATCH_SIZE, PATCH_SIZE))

                img_np = np.array(img_region.convert("RGB"))

                if _is_background(img_np):
                    continue

                msk_np = np.array(msk_region)
                if msk_np.ndim == 3:
                    msk_np = msk_np[:, :, 0]

                nonzero = msk_np != 0
                if nonzero.sum() < (PATCH_SIZE * PATCH_SIZE * 0.5):
                    continue

                vals, cnts = np.unique(msk_np[nonzero], return_counts=True)
                dominant = int(vals[cnts.argmax()])

                if dominant == _BACK_LABEL or dominant not in self._PIXEL_TO_IDX:
                    continue

                label_idx = self._PIXEL_TO_IDX[dominant]
                label_name = TISSUE_CLASSES[dominant]

                self._patches.append((str(img_path), str(mask_path), r, c, label_idx))
                self.labels.append(label_idx)
                self.case_ids.append(case_id)
                self.institutions.append(institution)
                inst_counts[institution] += 1
                class_counts[label_name] += 1

            if use_pil:
                pil_img.close()
                pil_msk.close()
            else:
                ts_img.close()
                ts_msk.close()
        except Exception as exc:
            print(f"  [WARN] Error reading tile {img_path.name}: {exc}")

    def __len__(self) -> int:
        return len(self._patches)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, _mask_path, row, col, label_idx = self._patches[idx]
        if img_path.endswith(".png"):
            region = self._read_region_pil(Path(img_path), row, col)
        else:
            ts = self._tiffslide.TiffSlide(img_path)
            region = ts.read_region((col, row), 0, (PATCH_SIZE, PATCH_SIZE))
            ts.close()
        patch_np = np.array(region.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(patch_np).permute(2, 0, 1)
        return tensor, label_idx
