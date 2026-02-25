'''
Please not that this is identical to the ipynb file and is meant to be used 
for NOMAD
'''



import sys
import os
import math
import time
from pathlib import Path
from typing import Optional

# ========================== EDIT THESE ==========================
DATA_ROOT = None                  # Path to SemiCOL DATASET_TRAIN, or None for synthetic
CHECKPOINT = "encoder-1.0.4.pt/encoder-1.0.4.pt/encoder.pt"
PYRAMID_LEVEL = 2                 # Feature pyramid level (2 = Stage 2→3 transition)
N_SAMPLES = 50_000               # Num synthetic images, or max_patches cap for SemiCOL
BATCH_SIZE = 32
IMAGE_SIZE = 224
OUTPUT_PATH = "data/dummy_activations.safetensors"
DEVICE = "cuda:0"                 # "cuda:0" or "cpu"
# ================================================================

# Path setup
project_root = Path("/app")
for subdir in ["UmedPT", "UmedPT-sdk"]:
    path = str(project_root / subdir)
    if path not in sys.path:
        sys.path.insert(0, path)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from safetensors.torch import save_file, load_file
from safetensors import safe_open

EPS = 1e-8
os.environ["MMM_LICENSE_ACCEPTED"] = "i accept"

print(f"PyTorch {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


SEMICOL_CLASSES = {
    0: "BACK", 1: "TUMOR", 2: "STROMA", 3: "MUSCLE", 4: "NECROSIS",
    5: "MUCUS", 6: "NORMAL", 7: "SUBMUCOSA", 8: "ADIPOSE", 9: "LYMPHOCYTES",
}


class SemiCOLPatchDataset(Dataset):
    def __init__(self, root: str, patch_size: int = 224, max_patches: Optional[int] = None, bg_threshold: float = 0.90):
        self.patch_size = patch_size
        self.bg_threshold = bg_threshold
        self.root = Path(root)
        self.patches = []
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_paths = self._discover_images()
        print(f"[SemiCOL] Found {len(image_paths)} image files")
        self._build_patch_index(image_paths, max_patches)
        print(f"[SemiCOL] Total patches: {len(self.patches)}")

    def _discover_images(self) -> list:
        paths = []
        for ext in {".tiff", ".tif", ".png", ".jpg", ".jpeg"}:
            paths.extend(self.root.rglob(f"*{ext}"))
            paths.extend(self.root.rglob(f"*{ext.upper()}"))
        paths = [p for p in paths if "mask" not in p.parent.name.lower()
                 and "mask" not in p.name.lower() and "label" not in p.name.lower()]
        return sorted(set(paths))

    def _build_patch_index(self, image_paths: list, max_patches: Optional[int]):
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        for img_path in image_paths:
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
            except Exception as e:
                print(f"  [WARN] Skipping {img_path.name}: {e}")
                continue
            ps = self.patch_size
            for row in range(0, h - ps + 1, ps):
                for col in range(0, w - ps + 1, ps):
                    self.patches.append((str(img_path), row, col))
                    if max_patches and len(self.patches) >= max_patches:
                        return

    def __len__(self): return len(self.patches)

    def __getitem__(self, idx: int):
        from PIL import Image
        import numpy as np
        img_path, row, col = self.patches[idx]
        ps = self.patch_size
        with Image.open(img_path) as img:
            patch = img.crop((col, row, col + ps, row + ps)).convert("RGB")
        arr = np.array(patch, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        brightness = tensor.mean()
        is_background = brightness > self.bg_threshold
        tensor = (tensor - self.mean) / self.std
        return tensor.float(), int(is_background)


class SyntheticShapeDataset(Dataset):
    SHAPES = ["circle", "square", "triangle", "cross"]

    def __init__(self, n_samples: int = 50_000, image_size: int = 224):
        self.n_samples = n_samples
        self.image_size = image_size

    def __len__(self): return self.n_samples

    def __getitem__(self, idx: int):
        H = W = self.image_size
        shape_idx = idx % len(self.SHAPES)
        shape_name = self.SHAPES[shape_idx]
        g = torch.Generator().manual_seed(idx)
        bg_r = torch.rand(1, generator=g).item() * 0.3 + 0.6
        bg_g = torch.rand(1, generator=g).item() * 0.3 + 0.5
        bg_b = torch.rand(1, generator=g).item() * 0.3 + 0.6
        img = torch.full((3, H, W), 0.0)
        img[0], img[1], img[2] = bg_r, bg_g, bg_b
        cx = int(torch.rand(1, generator=g).item() * 0.4 * W + 0.3 * W)
        cy = int(torch.rand(1, generator=g).item() * 0.4 * H + 0.3 * H)
        radius = int(torch.rand(1, generator=g).item() * 0.15 * H + 0.1 * H)
        fg_r = torch.rand(1, generator=g).item() * 0.4
        fg_g = torch.rand(1, generator=g).item() * 0.3
        fg_b = torch.rand(1, generator=g).item() * 0.5
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        if shape_name == "circle":
            mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius ** 2
        elif shape_name == "square":
            mask = ((xx - cx).abs() <= radius) & ((yy - cy).abs() <= radius)
        elif shape_name == "triangle":
            in_base = (yy >= cy) & (yy <= cy + radius)
            frac = (yy - cy).float() / max(radius, 1)
            half_w = (frac * radius).long()
            mask = in_base & ((xx - cx).abs() <= half_w)
        elif shape_name == "cross":
            arm = max(radius // 3, 2)
            vert = ((xx - cx).abs() <= arm) & ((yy - cy).abs() <= radius)
            horiz = ((yy - cy).abs() <= arm) & ((xx - cx).abs() <= radius)
            mask = vert | horiz
        else:
            mask = torch.zeros(H, W, dtype=torch.bool)
        img[0, mask], img[1, mask], img[2, mask] = fg_r, fg_g, fg_b
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (img - mean) / std
        return img.float(), shape_idx


def load_synthetic_dataset(n_samples: int, image_size: int) -> Dataset:
    try:
        from mmm.data_loading.synthetic.shape_dataset import ShapeDataset
        ds = ShapeDataset(n_samples=n_samples, image_size=image_size)
        print(f"[Dataset] Loaded MMM ShapeDataset ({n_samples} samples)")
        return ds
    except Exception as e:
        print(f"[Dataset] MMM ShapeDataset not available: {e}")
    print("[Dataset] Using fallback SyntheticShapeDataset")
    return SyntheticShapeDataset(n_samples=n_samples, image_size=image_size)


def load_umedpt(device: str = "cuda:0", checkpoint: Optional[str] = None):
    from mmm.api.M3Model import M3Model, M3_MODELS, DEFAULT_MODEL
    if checkpoint:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        print(f"[Model] Loading UMedPT from checkpoint: {checkpoint}")
        model = M3Model(M3_MODELS[DEFAULT_MODEL], device_identifier=device)
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        elif isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        try:
            model.load_state_dict(state, strict=False)
            print("[Model] Loaded checkpoint into full model")
        except Exception:
            try:
                model["encoder"].load_state_dict(state, strict=False)
                print("[Model] Loaded checkpoint into encoder only")
            except Exception as e:
                print(f"[Model] WARNING: partial load — {e}")
    else:
        print("[Model] Loading UMedPT (auto-download default weights)")
        model = M3Model(M3_MODELS[DEFAULT_MODEL], device_identifier=device)
    encoder = model["encoder"]
    encoder.eval()
    print(f"[Model] Encoder ready on {device}")
    return model, encoder


def normalize_activations(x: torch.Tensor, d_model: int) -> torch.Tensor:
    rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt() + EPS
    return x / rms * math.sqrt(d_model)


def harvest_activations(encoder, dataloader, pyramid_level, device):
    encoder.eval()
    all_acts = []
    print(f"Collecting activations from pyramid level {pyramid_level}...")
    t0 = time.time()
    with torch.inference_mode():
        for batch_idx, (images, _labels) in enumerate(dataloader):
            images = images.to(device)
            pyramid = encoder(images)
            feat = pyramid[pyramid_level]
            B, C, H, W = feat.shape
            flat = feat.permute(0, 2, 3, 1).reshape(B * H * W, C)
            all_acts.append(flat.cpu())
            if (batch_idx + 1) % 50 == 0:
                n = sum(a.shape[0] for a in all_acts)
                print(f"  Batch {batch_idx + 1:>5} | {n:>12,} vectors | {time.time() - t0:.1f}s")
    activations = torch.cat(all_acts, dim=0).float()
    del all_acts
    elapsed = time.time() - t0
    d = activations.shape[1]
    n = activations.shape[0]
    print(f"\nHarvested {n:,} vectors (d={d}) in {elapsed:.1f}s")
    avg_sq_before = activations.pow(2).sum(dim=-1).mean().item()
    activations = normalize_activations(activations, d)
    avg_sq_after = activations.pow(2).sum(dim=-1).mean().item()
    print(f"Normalization: avg ‖x‖² = {avg_sq_before:.1f} → {avg_sq_after:.1f} (target: {d})")
    return activations


if __name__ == "__main__":
    model, encoder = load_umedpt(DEVICE, checkpoint=CHECKPOINT)

    # Probe pyramid
    with torch.inference_mode():
        dummy = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
        pyramid = encoder(dummy)

    target = pyramid[PYRAMID_LEVEL]
    d_model = target.shape[1]
    spatial_h, spatial_w = target.shape[2], target.shape[3]
    print(f"\nSelected level {PYRAMID_LEVEL}: d_model={d_model}, spatial={spatial_h}×{spatial_w}")

    # Dataset
    if DATA_ROOT:
        dataset = SemiCOLPatchDataset(root=DATA_ROOT, patch_size=IMAGE_SIZE, max_patches=N_SAMPLES)
        data_source = "semicol"
    else:
        print("No DATA_ROOT set → using synthetic data")
        dataset = load_synthetic_dataset(N_SAMPLES, IMAGE_SIZE)
        data_source = "synthetic"

    n_images = len(dataset)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                            pin_memory=(device.type == "cuda"))

    # Harvest
    activations = harvest_activations(encoder, dataloader, PYRAMID_LEVEL, device)
    print(f"\nFinal shape: {activations.shape}")

    # Save
    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "data_source": data_source,
        "pyramid_level": str(PYRAMID_LEVEL),
        "d_model": str(d_model),
        "spatial_h": str(spatial_h),
        "spatial_w": str(spatial_w),
        "n_images": str(n_images),
        "n_effective": str(activations.shape[0]),
        "image_size": str(IMAGE_SIZE),
        "checkpoint": str(CHECKPOINT or "default"),
    }
    save_file({"activations": activations}, str(out_path), metadata=metadata)
    print(f"Saved to: {out_path} — shape: {tuple(activations.shape)}")