"""
Harvest activations from the UmedPT Swin Transformer for TopK SAE training.

Uses register_forward_hook on the PatchMerging layer at the Stage 2→3 boundary
of the Swin Transformer (features[4] in the torchvision SwinV2 backbone).

Pipeline
--------
  1. Load 50,000 synthetic shape images (mmm ShapeDataset, fallback to built-in)
  2. Load the encoder directly from encoder.pt
  3. Register a forward hook on PatchMergingV2 (Stage 2→3 boundary)
  4. Pass all batches through; hook captures (B, H', W', C') NHWC tensors
  5. Reshape to (B*H'*W', C') — the (B, C, H, W) → (B*H*W, C) flattening
  6. Anthropic-style normalization: x / (rms(x) + ε) * sqrt(d_model)
  7. Print N_effective and save to --output path

Usage (from repo root)
----------------------
  # Default: synthetic data, Stage 2→3 hook
  python src/scripts/harvest_activations.py

  # With a specific checkpoint
  python src/scripts/harvest_activations.py \
      --checkpoint encoder-1.0.4.pt/encoder.pt

  # All options
  python src/scripts/harvest_activations.py \
      --checkpoint encoder-1.0.4.pt/encoder.pt \
      --data_dir /path/to/semicol --track b \
      --n_samples 50000 --batch_size 256 \
      --pm_index 1 \
      --output data/activations.safetensors \
      --device cuda:0

PatchMerging index (--pm_index)
--------------------------------
  0 → Stage 1→2  (d_model=256,  spatial=28×28)
  1 → Stage 2→3  (d_model=512,  spatial=14×14)  ← default
  2 → Stage 3→4  (d_model=1024, spatial=7×7)
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from safetensors.torch import save_file
except ImportError:
    sys.exit("safetensors not installed.  Run: pip install safetensors")

# ─── Configuration ────────────────────────────────────────────────────────────

DEFAULT_CHECKPOINT  = "encoder-1.0.4.pt/encoder.pt"
DEFAULT_N_SAMPLES   = 50_000
DEFAULT_BATCH_SIZE  = 32
DEFAULT_IMAGE_SIZE  = 224
DEFAULT_OUTPUT_PATH = "data/dummy_activations.safetensors"
DEFAULT_PM_INDEX    = 1    # 0=Stage1→2  1=Stage2→3  2=Stage3→4
EPS                 = 1e-8

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ─── sys.path / environment ───────────────────────────────────────────────────

def _setup_mmm():
    """
    Ensure mmm (UmedPT) and m3_sdk (UmedPT-sdk) are importable.
    Searches common locations: /app (Docker/Nomad) and the repo root.
    """
    candidates = [
        Path("/app"),                                    # Docker / Nomad
        Path(__file__).resolve().parent.parent.parent,  # scripts/../../.. → repo root
        Path.cwd(),
    ]
    for root in candidates:
        for pkg in ("UmedPT", "UmedPT-sdk"):
            p = str(root / pkg)
            if Path(p).exists() and p not in sys.path:
                sys.path.insert(0, p)
    os.environ.setdefault("MMM_LICENSE_ACCEPTED", "i accept")


# ─── Datasets ────────────────────────────────────────────────────────────────

class _MMMShapeWrapper(Dataset):
    """
    Wraps mmm.data_loading.synthetic.shape_dataset.ShapeDataset.
    Extracts the image tensor and applies ImageNet normalisation so it is
    compatible with the (image, label) DataLoader interface.
    """

    def __init__(self, base_dataset: Dataset):
        self.base = base_dataset

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        sample = self.base[idx]
        # ShapeDataset returns a dict with key "image" (C,H,W) float32 in [0,1]
        img = sample["image"] if isinstance(sample, dict) else sample[0]
        img = (img.float() - _IMAGENET_MEAN) / _IMAGENET_STD
        return img, 0


class SyntheticShapeDataset(Dataset):
    """
    Built-in fallback: deterministic, procedurally generated geometric shapes
    (circle / square / triangle / cross) on coloured backgrounds.

    Each image is normalised to ImageNet statistics.
    """

    SHAPES = ["circle", "square", "triangle", "cross"]

    def __init__(self, n_samples: int = 50_000, image_size: int = 224):
        self.n  = n_samples
        self.sz = image_size

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        H = W = self.sz
        shape = self.SHAPES[idx % len(self.SHAPES)]
        g     = torch.Generator().manual_seed(idx)

        def rnd():
            return torch.rand(1, generator=g).item()

        img    = torch.zeros(3, H, W)
        img[0].fill_(rnd() * 0.3 + 0.6)
        img[1].fill_(rnd() * 0.3 + 0.5)
        img[2].fill_(rnd() * 0.3 + 0.6)

        cx = int(rnd() * 0.4 * W + 0.3 * W)
        cy = int(rnd() * 0.4 * H + 0.3 * H)
        r  = int(rnd() * 0.15 * H + 0.1 * H)
        fg = (rnd() * 0.4, rnd() * 0.3, rnd() * 0.5)

        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")

        if shape == "circle":
            mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r ** 2
        elif shape == "square":
            mask = ((xx - cx).abs() <= r) & ((yy - cy).abs() <= r)
        elif shape == "triangle":
            in_base = (yy >= cy) & (yy <= cy + r)
            half_w  = ((yy - cy).float() / max(r, 1) * r).long()
            mask    = in_base & ((xx - cx).abs() <= half_w)
        else:  # cross
            arm  = max(r // 3, 2)
            vert = ((xx - cx).abs() <= arm) & ((yy - cy).abs() <= r)
            horiz= ((yy - cy).abs() <= arm) & ((xx - cx).abs() <= r)
            mask = vert | horiz

        for c, v in enumerate(fg):
            img[c, mask] = v

        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
        return img.float(), idx % len(self.SHAPES)


def load_dataset(n_samples: int, image_size: int) -> Dataset:
    """
    Try mmm ShapeDataset first; fall back to built-in SyntheticShapeDataset.
    """
    _setup_mmm()
    try:
        from mmm.data_loading.synthetic.shape_dataset import ShapeDataset

        ds = ShapeDataset(n_samples=n_samples, image_size=image_size)
        print(f"[Dataset] mmm ShapeDataset  — {n_samples:,} samples  (image_size={image_size})")
        return _MMMShapeWrapper(ds)
    except Exception as exc:
        print(f"[Dataset] mmm ShapeDataset unavailable ({exc})")

    try:
        from mmm.data_loading.synthetic.artifact import ArtifactDataset

        ds = ArtifactDataset(n_samples=n_samples, image_size=image_size)
        print(f"[Dataset] mmm ArtifactDataset — {n_samples:,} samples")
        return _MMMShapeWrapper(ds)
    except Exception as exc:
        print(f"[Dataset] mmm ArtifactDataset unavailable ({exc})")

    print(f"[Dataset] Using built-in SyntheticShapeDataset — {n_samples:,} samples")
    return SyntheticShapeDataset(n_samples=n_samples, image_size=image_size)


# ─── Model loading ────────────────────────────────────────────────────────────

def load_encoder(checkpoint: str, device: str) -> nn.Module:
    """
    Load the PyramidEncoder module directly from a .pt file.

    The file contains a fully serialised PyramidEncoder (torch.save of the
    module object), so mmm must be importable before calling torch.load.
    """
    _setup_mmm()
    ckpt = Path(checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Encoder checkpoint not found: {ckpt.resolve()}\n"
            f"Hint: run from the repo root, or pass --checkpoint with an absolute path."
        )
    print(f"[Model] Loading from: {ckpt}")
    encoder = torch.load(str(ckpt), map_location=device, weights_only=False)
    encoder.eval()
    print(f"[Model] Loaded  ({type(encoder).__name__})  →  device={device}")
    return encoder


# ─── Hook ────────────────────────────────────────────────────────────────────

def _find_patch_merging_layers(encoder: nn.Module):
    """
    Navigate:
        encoder (PyramidEncoder)
            .model (TorchVisionSwinformer)
                .wrapped_model (torchvision SwinTransformer)
                    .features  (nn.Sequential)

    Return list of (feature_index, PatchMergingV2) tuples in order:
        index 0  →  Stage 1→2
        index 1  →  Stage 2→3   (default hook target)
        index 2  →  Stage 3→4
    """
    from torchvision.models.swin_transformer import PatchMergingV2

    features  = encoder.model.wrapped_model.features
    pm_layers = [
        (i, m) for i, m in enumerate(features) if isinstance(m, PatchMergingV2)
    ]
    labels = {0: "Stage 1→2", 1: "Stage 2→3", 2: "Stage 3→4"}
    for n, (fi, _) in enumerate(pm_layers):
        print(f"[Hook]  PM-{n}  {labels.get(n, '')}  features[{fi}]")
    return pm_layers


def register_hook(encoder: nn.Module, pm_index: int = DEFAULT_PM_INDEX):
    """
    Attach a forward hook to the PatchMergingV2 layer at pm_index.

    The hook stores the raw NHWC output tensor in `captured["feat"]` after
    every forward pass.  The caller is responsible for calling handle.remove().

    Returns
    -------
    handle  : torch.utils.hooks.RemovableHook
    captured: dict  — populated with key "feat" after each forward pass
    """
    pm_layers = _find_patch_merging_layers(encoder)
    if pm_index >= len(pm_layers):
        raise IndexError(
            f"pm_index={pm_index} requested but only {len(pm_layers)} "
            f"PatchMerging layers found."
        )

    feat_idx, pm_module = pm_layers[pm_index]
    stage = {0: "Stage 1→2", 1: "Stage 2→3", 2: "Stage 3→4"}.get(pm_index, f"PM-{pm_index}")
    print(f"[Hook]  Registered on  {stage}  (features[{feat_idx}])")

    captured: dict = {}

    def _hook(module, input, output):
        # output: (B, H', W', C')  —  Swin NHWC internal format
        captured["feat"] = output.detach().cpu()

    handle = pm_module.register_forward_hook(_hook)
    return handle, captured


# ─── Normalization ────────────────────────────────────────────────────────────

def normalize_activations(x: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Anthropic-style scalar normalization from the TopK SAE paper.

        x_norm = x / (sqrt(mean(x²)) + ε) * sqrt(d_model)

    Property: avg( ‖x_norm‖² ) ≈ d_model

    Args:
        x       : (N, D) float tensor  where D == d_model
        d_model : target squared-norm scale
    """
    rms    = x.pow(2).mean(dim=-1, keepdim=True).sqrt() + EPS
    x_norm = x / rms * math.sqrt(d_model)
    return x_norm


# ─── Main harvest function ────────────────────────────────────────────────────

def harvest_activations(
    encoder: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    pm_index: int = DEFAULT_PM_INDEX,
    patches_dir: Optional[str] = None,
) -> tuple[torch.Tensor, dict]:
    """
    Collect, flatten, and normalize activations from the hooked PM layer.

    Parameters
    ----------
    patches_dir : str or None
        If given, each 224×224 input patch is saved as a PNG named
        ``{patch_idx:07d}.png`` in this directory.  These files are
        required by ``visualise.py`` for spatial overlay rendering.
        Pixel values are clamped to [0, 1] before saving (no de-normalisation).

    Returns
    -------
    activations : (N_effective, d_model) float32 tensor
    info        : dict with d_model, spatial_h, spatial_w, n_images, n_effective
    """

    # ── attach hook ────────────────────────────────────────────────────────
    hook_handle, captured = register_hook(encoder, pm_index)

    # ── probe one dummy image to learn output shape ─────────────────────
    with torch.inference_mode():
        encoder(torch.rand(1, 3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, device=device))

    probe = captured["feat"]                    # (1, H', W', C')
    _, H_feat, W_feat, d_model = probe.shape
    n_spatial = H_feat * W_feat
    est_gb    = len(dataloader.dataset) * n_spatial * d_model * 4 / 1e9

    print(f"\n[Harvest] Hook output shape : (B, {H_feat}, {W_feat}, {d_model})")
    print(f"[Harvest] d_model           : {d_model}")
    print(f"[Harvest] Tokens / image    : {n_spatial}  ({H_feat}×{W_feat})")
    print(f"[Harvest] Expected N_eff    : {len(dataloader.dataset):,} × {n_spatial} "
          f"= {len(dataloader.dataset) * n_spatial:,}")
    print(f"[Harvest] Estimated size    : {est_gb:.1f} GB (float32)")
    print(f"[Harvest] Batches to run    : {len(dataloader)}\n")

    # ── patches_dir setup ───────────────────────────────────────────────
    _patches_out: Optional[Path] = None
    if patches_dir:
        from PIL import Image as _PILImage
        _patches_out = Path(patches_dir)
        _patches_out.mkdir(parents=True, exist_ok=True)
        print(f"[Harvest] Saving patch PNGs to {_patches_out}")

    # ── pre-allocate output tensor ───────────────────────────────────────
    # Size is known exactly: every image yields n_spatial tokens regardless
    # of batch boundaries, so no cat or grow-and-copy is needed.
    n_effective = len(dataloader.dataset) * n_spatial
    activations = torch.empty(n_effective, d_model, dtype=torch.float32)
    t0 = time.time()
    ptr = 0
    patch_ptr = 0  # tracks which patch (image) we are on, for PNG filenames

    encoder.eval()
    with torch.inference_mode():
        for batch_idx, (images, _) in enumerate(dataloader):
            encoder(images.to(device))           # hook fires here

            feat = captured["feat"]              # (B, H', W', C')  NHWC
            B, H, W, C = feat.shape

            # NHWC → (B*H*W, C): the two permutes in the original cancelled out
            flat = feat.reshape(B * H * W, C).float()

            # Normalize per-batch — normalize_activations is per-token (dim=-1)
            # so applying it here gives the same result as applying it globally.
            flat = normalize_activations(flat, d_model)

            n = flat.shape[0]
            activations[ptr : ptr + n] = flat
            ptr += n

            # ── Save patch PNGs (one per input image in the batch) ───────
            if _patches_out is not None:
                # images: [B, 3, H, W] float32 in [0,1] (SemiCOL) or
                # ImageNet-normalised (synthetic) — clamp to [0,1] either way
                imgs_cpu = images.cpu().clamp(0.0, 1.0)
                for i in range(B):
                    arr = (imgs_cpu[i].permute(1, 2, 0).numpy() * 255).astype("uint8")
                    _PILImage.fromarray(arr).save(
                        _patches_out / f"{patch_ptr:07d}.png"
                    )
                    patch_ptr += 1

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
                elapsed = time.time() - t0
                done = batch_idx + 1
                total = len(dataloader)
                eta = elapsed / done * (total - done) if done < total else 0
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"  [{ts}]  batch {done:>5} / {total}  |  "
                      f"{ptr:>12,} vectors  |  elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    hook_handle.remove()
    activations = activations[:ptr]  # guard: trim if dataset len was off
    elapsed = time.time() - t0

    n_eff = activations.shape[0]
    print(f"\n[Harvest] Done in {elapsed:.1f}s")
    print(f"[Harvest] N_effective = {n_eff:,}")

    # ── verify normalization (read-only, no extra allocation) ─────────────
    avg_sq_after = activations.pow(2).sum(dim=-1).mean().item()
    print(f"[Normalize] avg ‖x‖² = {avg_sq_after:.2f}  (target ≈ {d_model})")

    info = {
        "d_model"    : d_model,
        "spatial_h"  : H_feat,
        "spatial_w"  : W_feat,
        "n_images"   : len(dataloader.dataset),
        "n_effective": n_eff,
    }
    return activations, info


# ─── CLI entry-point ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Harvest activations from UmedPT Swin Transformer for TopK SAE training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--checkpoint",  default=DEFAULT_CHECKPOINT,
                    help="Path to encoder.pt  (PyramidEncoder module)")
    ap.add_argument("--n_samples",   type=int, default=DEFAULT_N_SAMPLES,
                    help="Number of synthetic images to generate")
    ap.add_argument("--batch_size",  type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--image_size",  type=int, default=DEFAULT_IMAGE_SIZE)
    ap.add_argument("--pm_index",    type=int, default=DEFAULT_PM_INDEX,
                    help="PatchMerging layer to hook: 0=Stage1→2  1=Stage2→3  2=Stage3→4")
    ap.add_argument("--output",      default=DEFAULT_OUTPUT_PATH)
    ap.add_argument("--device",
                    default="cuda:0" if torch.cuda.is_available() else "cpu")
    # ── Real-data options (optional; omit to keep synthetic fallback) ────────
    ap.add_argument("--data_dir",      type=str, default=None,
                    help="Path to dataset root. If omitted, synthetic data is used.")
    ap.add_argument("--dataset",       type=str, default="semicol",
                    choices=["semicol", "brain_tumor"],
                    help="Which dataset to load from --data_dir: "
                         "'semicol' (SemiCOL colorectal) or "
                         "'brain_tumor' (Kaggle brain-tumor-classification-mri Training/).")
    # SemiCOL-specific options (ignored when --dataset brain_tumor)
    ap.add_argument("--track",         type=str, default="b", choices=["a", "b"],
                    help="SemiCOL track: 'b'=weak WSI (SAE training), 'a'=manual masks (validation).")
    ap.add_argument("--max_per_slide", type=int, default=500,
                    help="SemiCOL Track B: max patches sampled per WSI slide.")
    ap.add_argument("--max_per_case",  type=int, default=2000,
                    help="SemiCOL Track A: max patches sampled per annotated case.")
    ap.add_argument("--patches_dir",   type=str, default=None,
                    help="If set, saves each input patch as {idx:07d}.png here. "
                         "Required for visualise.py spatial overlays.")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"{'='*60}")
    print(f"TopK SAE Activation Harvester")
    print(f"{'='*60}")
    print(f"Device     : {device}")
    if device.type == "cuda":
        print(f"GPU        : {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint : {args.checkpoint}")
    if args.data_dir:
        print(f"Dataset    : {args.dataset}")
        print(f"Data dir   : {args.data_dir}")
        if args.dataset == "semicol":
            print(f"Track      : {args.track}  ({'weak WSI' if args.track == 'b' else 'manual masks'})")
            if args.track == "b":
                print(f"Max/slide  : {args.max_per_slide}")
            else:
                print(f"Max/case   : {args.max_per_case}")
    else:
        print(f"N samples  : {args.n_samples:,}  (synthetic fallback)")
    print(f"Batch size : {args.batch_size}")
    print(f"PM index   : {args.pm_index}  (0=Stage1→2, 1=Stage2→3, 2=Stage3→4)")
    print(f"Output     : {args.output}")
    print()

    # 1. Dataset ──────────────────────────────────────────────────────────────
    if args.data_dir:
        if args.dataset == "brain_tumor":
            from datasets.brain_cancer_dataset import BrainTumorDataset
            dataset = BrainTumorDataset(args.data_dir)
        elif args.track == "b":
            from datasets.semicol_dataset import SemiCOLWeakDataset
            dataset = SemiCOLWeakDataset(args.data_dir, args.max_per_slide)
        else:
            from datasets.semicol_dataset import SemiCOLManualDataset
            dataset = SemiCOLManualDataset(args.data_dir, args.max_per_case)
    else:
        dataset = load_dataset(args.n_samples, args.image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=(device.type == "cuda"),
    )

    # 2. Model ────────────────────────────────────────────────────────────────
    encoder = load_encoder(args.checkpoint, args.device)

    # 3. Harvest ──────────────────────────────────────────────────────────────
    activations, info = harvest_activations(
        encoder, dataloader, device, args.pm_index,
        patches_dir=args.patches_dir,
    )

    # 4. Summary ──────────────────────────────────────────────────────────────
    n_eff   = info["n_effective"]
    d_model = info["d_model"]
    print(f"\n{'='*60}")
    print(f"N_effective = {n_eff:,}")
    print(f"d_model     = {d_model}")
    print(f"Tensor      : {tuple(activations.shape)}  dtype={activations.dtype}")
    print(f"{'='*60}\n")

    # 5. Save ─────────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import json as _json

    if not args.data_dir:
        track = "synthetic"
    elif args.dataset == "brain_tumor":
        track = "brain_tumor"
    else:
        track = args.track
    class_names = (
        dataset.class_names if hasattr(dataset, "class_names") else []
    )

    metadata = {
        "checkpoint"  : str(args.checkpoint),
        "d_model"     : str(d_model),
        "spatial_h"   : str(info["spatial_h"]),
        "spatial_w"   : str(info["spatial_w"]),
        "n_images"    : str(info["n_images"]),
        "n_effective" : str(n_eff),
        "pm_index"    : str(args.pm_index),
        "image_size"  : str(args.image_size),
        "track"       : track,
        "class_names" : _json.dumps(class_names),
    }

    # Labels and patch IDs are stored at patch granularity (one entry per
    # original 224×224 patch), NOT per activation row.  Each patch produces
    # spatial_h × spatial_w = 196 activation rows (for pm_index=1).
    # Downstream code must tile labels by n_patches if row-level labels are
    # needed (e.g. np.repeat(labels, spatial_h * spatial_w)).
    n_patches = len(dataset)  # patches after background filtering, not raw image count
    patch_labels = (
        torch.tensor(dataset.labels, dtype=torch.int32)
        if hasattr(dataset, "labels") and dataset.labels
        else torch.full((n_patches,), -1, dtype=torch.int32)
    )
    patch_ids = torch.arange(n_patches, dtype=torch.int32)

    save_file(
        {
            "activations": activations,
            "labels"     : patch_labels,   # [N_patches] int32, patch granularity
            "patch_ids"  : patch_ids,      # [N_patches] int32, sequential index
        },
        str(out_path),
        metadata=metadata,
    )

    size_mb = out_path.stat().st_size / 1e6
    print(f"Saved → {out_path}  ({size_mb:.1f} MB)")
    print(f"\n→ Bilal: {out_path} is ready for SAE training.")


if __name__ == "__main__":
    main()
