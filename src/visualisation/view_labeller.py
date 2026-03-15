import re
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Literal

ViewType = Literal["axial", "sagittal", "coronal", "unknown"]

def label_from_path(image_path: Path) -> ViewType:
    text = (str(image_path)).lower()
    if any(k in text for k in ["axial", "_ax_", "_ax.", "tra", "transverse"]):
        return "axial"
    if any(k in text for k in ["sagittal", "_sag_", "_sag."]):
        return "sagittal"
    if any(k in text for k in ["coronal", "_cor_", "_cor."]):
        return "coronal"
    return "unknown"

def label_from_image_geometry(image_path: Path) -> ViewType:
    try:
        img = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
        h, w = img.shape
        aspect = h / w
        mid = w // 2
        left  = img[:, :mid]
        right = np.fliplr(img[:, mid:mid + mid])
        min_w = min(left.shape[1], right.shape[1])
        corr = np.corrcoef(left[:, :min_w].ravel(), right[:, :min_w].ravel())[0, 1]
        symmetry = float(corr) if not np.isnan(corr) else 0.0
        if aspect > 1.2 and symmetry > 0.85:
            return "coronal"
        elif aspect < 0.85:
            return "axial"
        elif 0.85 <= aspect <= 1.2 and symmetry < 0.7:
            return "sagittal"
        else:
            return "unknown"
    except Exception:
        return "unknown"

def label_view(image_path: Path) -> ViewType:
    result = label_from_path(image_path)
    if result != "unknown":
        return result
    return label_from_image_geometry(image_path)

def label_dataset_views(images_dir: Path, save_path: Path | None = None) -> dict[str, ViewType]:
    import json
    results = {}
    image_paths = sorted(images_dir.glob("*"))
    image_paths = [p for p in image_paths if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tiff"}]
    for p in image_paths:
        results[p.name] = label_view(p)
    counts = {"axial": 0, "sagittal": 0, "coronal": 0, "unknown": 0}
    for v in results.values():
        counts[v] = counts.get(v, 0) + 1
    print("View labelling summary:")
    for view, count in counts.items():
        print(f"  {view}: {count} images")
    print(f"  unknown rate: {counts['unknown']/max(1,len(results)):.1%}")
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(results, indent=2))
        print(f"Labels saved -> {save_path}")
    return results
