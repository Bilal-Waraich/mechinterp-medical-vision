"""
Patch coverage analysis for chest X-ray datasets.
Determines what fraction of spatial annotations are large enough for reliable
patch-level validation in a 7×7 grid over 224×224 images (patch size = 32×32 px).

Datasets analyzed:
  1. SIIM ACR Pneumothorax  – RLE pixel masks (1024×1024)
  2. SIIM-COVID19           – bounding boxes from DICOM (variable size, ~2000–4000px)
  3. VinBigData             – bounding boxes (variable size per image)
"""

import ast
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path

# ── constants ──────────────────────────────────────────────────────────────────
TARGET = 224
GRID   = 7
PATCH  = TARGET // GRID          # 32 px
PATCH2 = PATCH ** 2              # 1024 px² – 1 full patch area
PATCH_2x2 = (2 * PATCH) ** 2    # 4096 px² – 2×2 patches area (for box metric: min_dim >= 64)

SUB_PATCH_THR  = PATCH           # < 32 → sub-patch
TWO_PATCH_THR  = 2 * PATCH       # >= 64 → spans ≥ 2 patches in each dim

RELIABLE_THR   = 0.20            # <20% sub-patch → reliable
USABLE_THR     = 0.40            # 20-40% sub-patch → usable with soft labels

BASE = Path("//fileserver/share/deep_learning/input/data/mmm")


# ── helpers ────────────────────────────────────────────────────────────────────

def classify(pct_sub: float) -> str:
    if pct_sub < RELIABLE_THR:
        return "RELIABLE"
    elif pct_sub < USABLE_THR:
        return "USABLE (soft labels)"
    else:
        return "UNRELIABLE (image-level only)"


def box_stats_scaled(widths_scaled, heights_scaled, label=""):
    """
    Given arrays of scaled (224-space) widths and heights, compute patch coverage stats.
    """
    n = len(widths_scaled)
    if n == 0:
        return None
    min_dims = np.minimum(widths_scaled, heights_scaled)
    sub      = (min_dims < SUB_PATCH_THR).sum()
    two_plus = (min_dims >= TWO_PATCH_THR).sum()
    pct_sub  = sub / n
    pct_2p   = two_plus / n
    return dict(
        label=label,
        n_boxes=n,
        pct_sub_patch=round(pct_sub * 100, 1),
        pct_2x2_plus=round(pct_2p * 100, 1),
        decision=classify(pct_sub),
        median_min_dim_scaled=round(float(np.median(min_dims)), 1),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. SIIM ACR PNEUMOTHORAX  – RLE masks (1024×1024)
# ══════════════════════════════════════════════════════════════════════════════

def decode_rle(rle_str: str, shape=(1024, 1024)):
    """Decode run-length encoded string to binary mask."""
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    tokens = rle_str.strip().split()
    for i in range(0, len(tokens), 2):
        start  = int(tokens[i])
        length = int(tokens[i + 1])
        mask[start:start + length] = 1
    return mask.reshape(shape)


def analyze_siim_pneumothorax():
    print("\n" + "=" * 70)
    print("SIIM ACR PNEUMOTHORAX  (RLE masks, 1024×1024)")
    print("=" * 70)

    csv_path = BASE / "siim-acr-pneumothorax-segmentation/stage_2_train.csv"
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    # keep only positive masks
    df_pos = df[df["EncodedPixels"] != "-1"].copy()

    orig_size = 1024
    scale = TARGET / orig_size   # 224/1024

    print(f"Total rows: {len(df)}")
    print(f"Positive masks (pneumothorax present): {len(df_pos)}")
    print(f"Scale factor: {scale:.4f}  →  1024px maps to {orig_size * scale:.0f}px in 224-space\n")

    # For each positive mask: compute area in 224-space
    # Area_224 = area_1024 * scale²
    # We also want bounding-box min dimension in 224-space

    areas_224 = []
    min_dims_224 = []

    for _, row in df_pos.iterrows():
        rle = row["EncodedPixels"]
        mask = decode_rle(rle, shape=(1024, 1024))
        area = mask.sum()
        areas_224.append(area * (scale ** 2))

        # bounding box from mask
        rows_with_px = np.any(mask, axis=1)
        cols_with_px = np.any(mask, axis=0)
        if rows_with_px.any():
            r0, r1 = np.where(rows_with_px)[0][[0, -1]]
            c0, c1 = np.where(cols_with_px)[0][[0, -1]]
            h = (r1 - r0 + 1) * scale
            w = (c1 - c0 + 1) * scale
            min_dims_224.append(min(w, h))
        else:
            min_dims_224.append(0)

    areas_224   = np.array(areas_224)
    min_dims_224 = np.array(min_dims_224)

    # Mask-area metric: fraction covering ≥ 1 full patch (area ≥ 32²=1024 px²)
    n = len(areas_224)
    cover_1patch = (areas_224 >= PATCH2).sum()
    cover_2x2    = (areas_224 >= PATCH_2x2).sum()
    sub_patch_bb = (min_dims_224 < SUB_PATCH_THR).sum()

    print("--- Mask area (scaled to 224-space) ---")
    print(f"  n masks                     : {n}")
    print(f"  area >= 32² (1 patch)       : {cover_1patch} ({100*cover_1patch/n:.1f}%)")
    print(f"  area >= 64² (2×2 patches)   : {cover_2x2} ({100*cover_2x2/n:.1f}%)")
    print(f"  median area (224-space px²) : {np.median(areas_224):.0f}")
    print()
    print("--- Bounding-box min dimension (scaled to 224-space) ---")
    print(f"  sub-patch (min_dim < 32)    : {sub_patch_bb} ({100*sub_patch_bb/n:.1f}%)")
    print(f"  2×2+ (min_dim >= 64)        : {(min_dims_224 >= TWO_PATCH_THR).sum()} ({100*(min_dims_224>=TWO_PATCH_THR).sum()/n:.1f}%)")
    print(f"  median min_dim (224-space)  : {np.median(min_dims_224):.1f} px")
    pct_sub = sub_patch_bb / n
    print(f"\n  Decision: {classify(pct_sub)}")
    return pct_sub


# ══════════════════════════════════════════════════════════════════════════════
# 2. SIIM-COVID19  – bounding boxes (variable size DICOM images)
# ══════════════════════════════════════════════════════════════════════════════

def get_covid_image_sizes_from_dcm(n_sample=500):
    """
    Sample DICOM files from SIIM-COVID19 to estimate image resolution distribution.
    Returns dict {image_id: (w, h)}.
    """
    import pydicom
    train_base = BASE / "siim-covid19-detection/train"
    sizes = {}
    count = 0
    for study_dir in train_base.iterdir():
        if count >= n_sample:
            break
        for series_dir in study_dir.iterdir():
            for dcm_file in series_dir.glob("*.dcm"):
                img_id = dcm_file.stem
                try:
                    ds = pydicom.dcmread(str(dcm_file), stop_before_pixels=True)
                    sizes[img_id] = (int(ds.Columns), int(ds.Rows))
                    count += 1
                except Exception:
                    pass
                if count >= n_sample:
                    break
            if count >= n_sample:
                break
    return sizes


def analyze_siim_covid19():
    print("\n" + "=" * 70)
    print("SIIM-COVID19 DETECTION  (bounding boxes, variable resolution)")
    print("=" * 70)

    csv_path = BASE / "siim-covid19-detection/train_image_level.csv"
    df = pd.read_csv(csv_path)

    # Parse boxes: format is JSON-like list of dicts with x, y, width, height
    # Only keep rows that have actual box annotations (not 'none')
    df_box = df[df["boxes"].notna() & (df["boxes"] != "")].copy()
    # Remove 'none' label rows
    df_box = df_box[~df_box["label"].str.startswith("none")].copy()

    print(f"Total image rows: {len(df)}")
    print(f"Rows with box annotations: {len(df_box)}")

    # Get image sizes from a sample of DICOM files
    print("\nSampling DICOM headers for image sizes (up to 500 images)…")
    try:
        size_map = get_covid_image_sizes_from_dcm(n_sample=500)
        print(f"  Sampled {len(size_map)} images")
        widths  = [s[0] for s in size_map.values()]
        heights = [s[1] for s in size_map.values()]
        print(f"  Width  range: {min(widths)}–{max(widths)}, median {np.median(widths):.0f}")
        print(f"  Height range: {min(heights)}–{max(heights)}, median {np.median(heights):.0f}")
        median_w = np.median(widths)
        median_h = np.median(heights)
    except Exception as e:
        print(f"  DICOM sampling failed ({e}), using fallback size 2320×2320 (typical)")
        size_map = {}
        median_w = median_h = 2320

    # Parse all boxes, scale each to 224-space using the image-level size
    # For images not in size_map, use the median
    all_w_scaled = []
    all_h_scaled = []
    label_boxes  = {}  # class → list of (w_scaled, h_scaled)

    for _, row in df_box.iterrows():
        img_id = row["id"].replace("_image", "")
        if img_id in size_map:
            orig_w, orig_h = size_map[img_id]
        else:
            orig_w, orig_h = median_w, median_h

        scale_x = TARGET / orig_w
        scale_y = TARGET / orig_h

        try:
            boxes = ast.literal_eval(row["boxes"])
        except Exception:
            try:
                boxes = json.loads(row["boxes"])
            except Exception:
                continue

        # Extract class label (e.g. "opacity")
        lbl = row["label"].split()[0] if isinstance(row["label"], str) else "unknown"

        for b in boxes:
            bw = float(b.get("width", 0))  * scale_x
            bh = float(b.get("height", 0)) * scale_y
            all_w_scaled.append(bw)
            all_h_scaled.append(bh)
            label_boxes.setdefault(lbl, [[], []])[0].append(bw)
            label_boxes[lbl][1].append(bh)

    all_w_scaled = np.array(all_w_scaled)
    all_h_scaled = np.array(all_h_scaled)

    print(f"\nTotal box annotations: {len(all_w_scaled)}")

    # Overall stats
    stats_all = box_stats_scaled(all_w_scaled, all_h_scaled, label="ALL")
    print("\n--- Overall (all classes) ---")
    _print_box_stats(stats_all)

    # Per-class
    print("\n--- Per class ---")
    for lbl, (ws, hs) in sorted(label_boxes.items()):
        s = box_stats_scaled(np.array(ws), np.array(hs), label=lbl)
        _print_box_stats(s)

    return stats_all["pct_sub_patch"] / 100


def _print_box_stats(s):
    if s is None:
        return
    print(f"  {s['label']:25s}  n={s['n_boxes']:5d}  "
          f"sub-patch={s['pct_sub_patch']:5.1f}%  "
          f"2×2+={s['pct_2x2_plus']:5.1f}%  "
          f"median_min_dim={s['median_min_dim_scaled']:5.1f}px  "
          f"→ {s['decision']}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. VinBigData  – bounding boxes (variable size, estimate from box coords)
# ══════════════════════════════════════════════════════════════════════════════

def analyze_vinbigdata():
    print("\n" + "=" * 70)
    print("VinBigData  (bounding boxes, variable resolution estimated from coords)")
    print("=" * 70)

    csv_path = BASE / "vinbigdata-chest-xray-abnormalities-detection/train.csv"
    df = pd.read_csv(csv_path)

    # Remove No-finding rows (no box coords)
    df_box = df[df["class_name"] != "No finding"].dropna(subset=["x_min","y_min","x_max","y_max"]).copy()
    print(f"Total rows: {len(df)}")
    print(f"Rows with boxes (non-'No finding'): {len(df_box)}")

    # Estimate original image size per image as max(x_max, y_max) * 1.05
    df_box["est_img_size"] = np.maximum(df_box["x_max"], df_box["y_max"]) * 1.05
    # Use per-image max estimate
    img_size = df_box.groupby("image_id")["est_img_size"].max()
    df_box = df_box.join(img_size.rename("img_size_est"), on="image_id")
    df_box["scale"] = TARGET / df_box["img_size_est"]

    df_box["w_scaled"] = (df_box["x_max"] - df_box["x_min"]) * df_box["scale"]
    df_box["h_scaled"] = (df_box["y_max"] - df_box["y_min"]) * df_box["scale"]

    print(f"\nEstimated orig size — median: {df_box['img_size_est'].median():.0f}  "
          f"range: {df_box['img_size_est'].min():.0f}–{df_box['img_size_est'].max():.0f}")

    # Overall
    stats_all = box_stats_scaled(df_box["w_scaled"].values, df_box["h_scaled"].values, label="ALL")
    print("\n--- Overall (all classes) ---")
    _print_box_stats(stats_all)

    # Per class
    print("\n--- Per class ---")
    for cls, grp in df_box.groupby("class_name"):
        s = box_stats_scaled(grp["w_scaled"].values, grp["h_scaled"].values, label=cls)
        _print_box_stats(s)

    return stats_all["pct_sub_patch"] / 100


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Patch-coverage analysis  |  target={TARGET}px  grid={GRID}×{GRID}  patch={PATCH}px")
    print(f"Thresholds: sub-patch < {SUB_PATCH_THR}px  |  2×2+ >= {TWO_PATCH_THR}px")
    print(f"Decision:   <{RELIABLE_THR*100:.0f}% sub → RELIABLE  |  "
          f"{RELIABLE_THR*100:.0f}–{USABLE_THR*100:.0f}% → USABLE  |  "
          f">{USABLE_THR*100:.0f}% → UNRELIABLE")

    pct_ptx   = analyze_siim_pneumothorax()
    pct_covid = analyze_siim_covid19()
    pct_vin   = analyze_vinbigdata()

    print("\n" + "=" * 70)
    print("SUMMARY – pm_index=2 patch-validation dataset selection")
    print("=" * 70)
    rows = [
        ("SIIM Pneumothorax",  pct_ptx,   "RLE masks"),
        ("SIIM-COVID19",       pct_covid,  "bounding boxes"),
        ("VinBigData",         pct_vin,    "bounding boxes (current pm_index=2)"),
    ]
    for name, pct, kind in rows:
        print(f"  {name:25s}  sub-patch={pct*100:.1f}%  ({kind})  →  {classify(pct)}")

    print()
    best = min(rows, key=lambda r: r[1])
    if best[0] != "VinBigData":
        print(f"  Recommendation: REPLACE VinBigData with {best[0]} at pm_index=2")
        print(f"  (lower sub-patch rate: {best[1]*100:.1f}% vs {pct_vin*100:.1f}%)")
    else:
        print("  Recommendation: KEEP VinBigData at pm_index=2 — it has the best patch coverage.")
