"""TCGA-BRCA breast cancer WSI dataset for SAE activation harvesting.

Directory layout under <data_dir>:
    <uuid>/
        <TCGA-barcode>.<file-UUID>.svs        # Aperio whole-slide image
        logs/
    breast_wsi_foreground/
        <TCGA-barcode>.<file-UUID>.geojson    # tissue foreground polygons
    tcga_brca_all_clean.csv                   # clinical metadata

Labels (OncoTree histological subtypes — from tcga_brca_all_clean.csv):
    0 = IDC    (Invasive Ductal Carcinoma,  ~80% of slides)
    1 = ILC    (Invasive Lobular Carcinoma, ~10%)
    2 = MDLC   (Mixed Ductal/Lobular,        ~5%)
    3 = IBC    (Inflammatory Breast Cancer, <1%)
    4 = Other  (rare subtypes / generic BRCA code)
   -1 = unknown (slide not in clinical CSV or code unrecognised)

Foreground masking:
    GeoJSON polygons in breast_wsi_foreground/ are rasterised once per slide at
    init time to a coarse boolean grid (1 cell = 224 px), so each candidate
    patch position is filtered in O(1) without re-reading the slide.
    Falls back to HSV saturation filter if shapely is not installed or the
    GeoJSON file for a slide is missing.

Usage in harvest_activations.py:
    from datasets.breast_cancer.tcga_breast_dataset import TCGABreastDataset
    dataset = TCGABreastDataset(data_dir, max_per_slide=1000)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

try:
    import tiffslide as _tiffslide_mod
    _TIFFSLIDE_AVAILABLE = True
except ImportError:
    _TIFFSLIDE_AVAILABLE = False

try:
    from shapely.geometry import shape as _shape
    from shapely.ops import unary_union as _unary_union
    _SHAPELY_AVAILABLE = True
except ImportError:
    _SHAPELY_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────

PATCH_SIZE    = 224
SAT_THRESHOLD = 0.05

# OncoTree histological subtypes present in TCGA-BRCA
CLASS_NAMES = ["IDC", "ILC", "MDLC", "IBC", "Other"]

# OncoTree code (lower-case) → numeric label
_LABEL_MAP: dict[str, int] = {
    "idc":  0,   # Invasive Ductal Carcinoma  (~80 %)
    "ilc":  1,   # Invasive Lobular Carcinoma (~10 %)
    "mdlc": 2,   # Mixed Ductal/Lobular        (~5 %)
    "ibc":  3,   # Inflammatory Breast Cancer  (<1 %)
    # Everything else maps to "Other"
    "brca": 4,   # Generic breast cancer code
    "mpt":  4,   # Metaplastic Breast Cancer
    "clis": 4,   # Carcinoma in situ
    "immc": 4,   # Invasive Mixed Mucinous
    "pure": 4,   # Pure Mucinous
}

# Column name candidates tried in order — first match wins
_PATIENT_COL_CANDIDATES = [
    "case_id", "submitter_id", "PATIENT_ID", "patient_id",
    "bcr_patient_barcode", "PATIENT",
]
_SUBTYPE_COL_CANDIDATES = [
    "oncotree_code",          # tcga_brca_all_clean.csv — histological subtype
    "ONCOTREE_CODE",
    "SUBTYPE", "subtype",     # other common naming conventions
    "PAM50", "pam50",
    "BRCA_Subtype_PAM50", "brca_subtype_pam50",
    "paper_BRCA_Subtype_PAM50",
]

# Dirs to skip when scanning for SVS files
_SKIP_DIRS = {"breast_wsi_foreground", "biospecimen.cart.2024-01-12", "logs"}


def _is_background(patch_np: np.ndarray) -> bool:
    hsv = Image.fromarray(patch_np).convert("HSV")
    sat = np.array(hsv)[:, :, 1].mean() / 255.0
    return sat < SAT_THRESHOLD


# ── Dataset ───────────────────────────────────────────────────────────────────

class TCGABreastDataset(Dataset):
    """Samples 224×224 patches from TCGA-BRCA Aperio SVS whole-slide images.

    Parameters
    ----------
    data_dir : str or Path
        Root of the tcga_breast download directory.
    max_per_slide : int
        Maximum patches sampled per SVS slide.
    seed : int
        RNG seed for reproducible patch position sampling.

    Attributes
    ----------
    labels : list[int]
        OncoTree label index per patch (0-4, or -1 for unknown).
    class_names : list[str]
        ["IDC", "ILC", "MDLC", "IBC", "Other"]
    """

    class_names = CLASS_NAMES

    def __init__(
        self,
        data_dir: str | Path,
        max_per_slide: int = 1000,
        seed: int = 42,
    ) -> None:
        if not _TIFFSLIDE_AVAILABLE:
            raise ImportError("tiffslide is required: pip install tiffslide")

        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"data_dir not found: {data_dir}")

        self._data_dir      = data_dir
        self._max_per_slide = max_per_slide

        if not _SHAPELY_AVAILABLE:
            print("[TCGABreastDataset] shapely not installed — using saturation fallback")

        fg_dir = data_dir / "breast_wsi_foreground"
        self._fg_dir = fg_dir if fg_dir.exists() else None
        if self._fg_dir is None:
            print("[TCGABreastDataset] breast_wsi_foreground/ not found — using saturation fallback")

        self._label_map = _load_label_map(data_dir)

        # (svs_path_str, row, col, label)
        self._patches: list[tuple[str, int, int, int]] = []
        self.labels:   list[int] = []

        label_counts: dict[str, int] = {n: 0 for n in CLASS_NAMES}
        label_counts["unknown"] = 0
        n_slides = 0

        svs_files = sorted(
            p
            for d in data_dir.iterdir()
            if d.is_dir() and d.name not in _SKIP_DIRS
            for p in d.glob("*.svs")
        )

        if not svs_files:
            print(f"[TCGABreastDataset] WARNING: no .svs files found under {data_dir}")

        rng = np.random.default_rng(seed)

        for svs_path in svs_files:
            label    = self._resolve_label(svs_path)
            fg_grid  = self._load_foreground_grid(svs_path)
            n_before = len(self._patches)
            self._index_slide(svs_path, label, fg_grid, rng)
            added = len(self._patches) - n_before
            if added:
                n_slides += 1
                lname = CLASS_NAMES[label] if 0 <= label < len(CLASS_NAMES) else "unknown"
                label_counts[lname] += added

        total = len(self._patches)
        print(f"[TCGABreastDataset] {total:,} patches from {n_slides} / {len(svs_files)} slides")
        print(f"  Class breakdown:")
        for name in CLASS_NAMES + ["unknown"]:
            n = label_counts[name]
            if n:
                print(f"    {name:<10}: {n:>8,}")

    # ── Label resolution ──────────────────────────────────────────────────────

    def _resolve_label(self, svs_path: Path) -> int:
        # e.g. TCGA-A2-A0ET-01Z-00-DX1.<UUID>.svs → patient TCGA-A2-A0ET
        slide_barcode = svs_path.stem.split(".")[0]
        patient_id    = "-".join(slide_barcode.split("-")[:3])
        return self._label_map.get(patient_id, -1)

    # ── Foreground masking ────────────────────────────────────────────────────

    def _load_foreground_grid(self, svs_path: Path) -> np.ndarray | None:
        """Rasterise the GeoJSON foreground polygon to a coarse boolean grid.

        Returns a (grid_h, grid_w) bool array where True means the
        PATCH_SIZE×PATCH_SIZE cell overlaps with tissue foreground.
        Returns None if GeoJSON is unavailable (falls back to saturation).
        """
        if not _SHAPELY_AVAILABLE or self._fg_dir is None:
            return None

        geojson_path = self._fg_dir / (svs_path.stem + ".geojson")
        if not geojson_path.exists():
            return None

        try:
            with open(geojson_path) as fh:
                gj = json.load(fh)

            if gj.get("type") == "FeatureCollection":
                geoms = [f["geometry"] for f in gj.get("features", [])]
            elif gj.get("type") == "Feature":
                geoms = [gj["geometry"]]
            else:
                geoms = [gj]

            tissue = _unary_union([_shape(g) for g in geoms if g])
        except Exception as exc:
            print(f"  [WARN] GeoJSON parse failed for {svs_path.name}: {exc}")
            return None

        try:
            ts = _tiffslide_mod.TiffSlide(str(svs_path))
            W, H = ts.dimensions
            ts.close()
        except Exception:
            return None

        grid_w = max(1, (W + PATCH_SIZE - 1) // PATCH_SIZE)
        grid_h = max(1, (H + PATCH_SIZE - 1) // PATCH_SIZE)

        from shapely.affinity import scale as _scale
        tissue_grid = _scale(tissue, xfact=1 / PATCH_SIZE, yfact=1 / PATCH_SIZE, origin=(0, 0))

        mask_img = Image.new("L", (grid_w, grid_h), 0)
        draw = ImageDraw.Draw(mask_img)
        self._draw_shapely(draw, tissue_grid)
        return np.array(mask_img, dtype=bool)

    @staticmethod
    def _draw_shapely(draw: ImageDraw.ImageDraw, geom) -> None:
        """Draw a shapely geometry onto a PIL ImageDraw, handling holes."""
        gtype = geom.geom_type
        if gtype == "Polygon":
            polys = [geom]
        elif gtype == "MultiPolygon":
            polys = list(geom.geoms)
        else:
            return

        for poly in polys:
            if poly.is_empty:
                continue
            ext = list(poly.exterior.coords)
            if len(ext) >= 3:
                draw.polygon(ext, fill=1)
            for interior in poly.interiors:
                holes = list(interior.coords)
                if len(holes) >= 3:
                    draw.polygon(holes, fill=0)

    # ── Slide indexing ────────────────────────────────────────────────────────

    def _index_slide(
        self,
        svs_path: Path,
        label: int,
        fg_grid: np.ndarray | None,
        rng: np.random.Generator,
    ) -> None:
        try:
            ts = _tiffslide_mod.TiffSlide(str(svs_path))
            W, H = ts.dimensions
            ts.close()
        except Exception as exc:
            print(f"  [WARN] Cannot open {svs_path.name}: {exc}")
            return

        rows = range(0, H - PATCH_SIZE + 1, PATCH_SIZE)
        cols = range(0, W - PATCH_SIZE + 1, PATCH_SIZE)
        candidates: list[tuple[int, int]] = [(r, c) for r in rows for c in cols]

        if len(candidates) > self._max_per_slide:
            idx = rng.choice(len(candidates), size=self._max_per_slide, replace=False)
            candidates = [candidates[i] for i in sorted(idx)]

        if fg_grid is not None:
            candidates = [
                (r, c) for r, c in candidates
                if fg_grid[r // PATCH_SIZE, c // PATCH_SIZE]
            ]
            for r, c in candidates:
                self._patches.append((str(svs_path), r, c, label))
                self.labels.append(label)
        else:
            try:
                ts = _tiffslide_mod.TiffSlide(str(svs_path))
                for r, c in candidates:
                    region   = ts.read_region((c, r), 0, (PATCH_SIZE, PATCH_SIZE))
                    patch_np = np.array(region.convert("RGB"))
                    if _is_background(patch_np):
                        continue
                    self._patches.append((str(svs_path), r, c, label))
                    self.labels.append(label)
                ts.close()
            except Exception as exc:
                print(f"  [WARN] Error reading {svs_path.name}: {exc}")

    # ── PyTorch Dataset interface ─────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._patches)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        svs_path, row, col, label = self._patches[idx]
        try:
            ts     = _tiffslide_mod.TiffSlide(svs_path)
            region = ts.read_region((col, row), 0, (PATCH_SIZE, PATCH_SIZE))
            ts.close()
            patch_np = np.array(region.convert("RGB"), dtype=np.float32) / 255.0
        except Exception:
            patch_np = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.float32)

        tensor = torch.from_numpy(patch_np).permute(2, 0, 1)  # [3, H, W]
        return tensor, max(label, 0)  # clamp unknown (-1) to 0


# ── CSV label loader ──────────────────────────────────────────────────────────

def _load_label_map(data_dir: Path) -> dict[str, int]:
    """Load patient_id (TCGA-XX-XXXX) → label index from tcga_brca_all_clean.csv.

    Uses the oncotree_code column (histological subtype, already present in the
    dataset) and maps to CLASS_NAMES indices via _LABEL_MAP.
    """
    csv_path = data_dir / "tcga_brca_all_clean.csv"
    if not csv_path.exists():
        print(f"[TCGABreastDataset] tcga_brca_all_clean.csv not found — all labels = -1")
        return {}

    try:
        with open(csv_path, newline="", encoding="utf-8-sig") as fh:
            reader  = csv.DictReader(fh)
            headers = list(reader.fieldnames or [])

            patient_col = next((c for c in _PATIENT_COL_CANDIDATES if c in headers), None)
            subtype_col = next((c for c in _SUBTYPE_COL_CANDIDATES if c in headers), None)

            if patient_col is None or subtype_col is None:
                print(f"[TCGABreastDataset] Cannot find patient/subtype columns.")
                print(f"  Tried patient cols : {_PATIENT_COL_CANDIDATES}")
                print(f"  Tried subtype cols : {_SUBTYPE_COL_CANDIDATES}")
                print(f"  First 10 actual cols: {headers[:10]}")
                return {}

            print(f"[TCGABreastDataset] Labels from '{subtype_col}' column (patient key: '{patient_col}')")

            label_map: dict[str, int] = {}
            for row in reader:
                pid  = row[patient_col].strip()[:12]   # TCGA-XX-XXXX (12 chars)
                code = row[subtype_col].strip().lower()
                label_map[pid] = _LABEL_MAP.get(code, -1)

        n_labelled = sum(1 for v in label_map.values() if v >= 0)
        print(f"[TCGABreastDataset] {len(label_map):,} patients loaded, {n_labelled:,} with known subtype")
        return label_map

    except Exception as exc:
        print(f"[TCGABreastDataset] Failed to parse tcga_brca_all_clean.csv: {exc}")
        return {}