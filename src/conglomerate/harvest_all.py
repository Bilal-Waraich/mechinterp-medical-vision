"""
Harvest activations from ALL four medical-imaging datasets into one combined file.

Loads the UMedPT encoder once, registers a single PatchMerging hook, then runs
each dataset through in sequence.  All activation tensors are concatenated and
saved as a single safetensors file suitable for training a conglomerate SAE.

Datasets supported
------------------
  semicol      SemiCOL colorectal WSI patches  (Track B: weak labels)
  brain_tumor  Kaggle Brain Tumour MRI
  tcga_breast  TCGA-BRCA whole-slide images
  pneumonia    RSNA Pneumonia Detection DICOM

Any dataset whose ``--*_dir`` flag is omitted is silently skipped, so you can
harvest whatever subset is locally available.

Output tensors
--------------
  activations  [N_total, d_model]  float32  normalised activations
  labels       [N_patches]         int32    class label per patch (-1 if unknown)
  patch_ids    [N_patches]         int32    index within each dataset (resets per ds)
  dataset_ids  [N_patches]         int32    integer dataset index (0–3)

Metadata keys
-------------
  d_model, spatial_h, spatial_w, pm_index, n_images_total,
  n_effective_total, dataset_names (JSON list), dataset_counts (JSON list)

Pipeline
--------
  1. Load encoder from --checkpoint
  2. Register hook on PatchMergingV2 at --pm_index  (default 1 = Stage 2→3)
  3. For each dataset provided, run DataLoader → collect activations
  4. Concatenate all tensors
  5. Save combined file to --output

Usage
-----
  python src/conglomerate/harvest_all.py \\
      --checkpoint encoder-1.0.4.pt/encoder.pt \\
      --semicol_dir  /data/semicol  \\
      --brain_dir    /data/brain_tumor  \\
      --breast_dir   /data/tcga_breast  \\
      --pneumonia_dir /data/pneumonia  \\
      --output  data/conglomerate_activations.safetensors \\
      --pm_index 1 \\
      --device cuda:0
"""
from __future__ import annotations

import argparse
import json
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

# ── path setup ────────────────────────────────────────────────────────────────

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ── constants ─────────────────────────────────────────────────────────────────

DEFAULT_CHECKPOINT  = "encoder-1.0.4.pt/encoder.pt"
DEFAULT_OUTPUT      = "data/conglomerate_activations.safetensors"
DEFAULT_PM_INDEX    = 1
DEFAULT_BATCH_SIZE  = 32
DEFAULT_IMAGE_SIZE  = 224
DEFAULT_MAX_PER_SLIDE = 500
EPS = 1e-8

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

D_MODEL_MAP   = {0: 256, 1: 512, 2: 1024, 3: 1024}
N_PATCHES_MAP = {0: 784, 1: 196, 2: 49,   3: 49}

DATASET_INDEX = {
    "semicol":      0,
    "brain_tumor":  1,
    "tcga_breast":  2,
    "pneumonia":    3,
}

# ── mmm / UmedPT path bootstrap ───────────────────────────────────────────────

def _setup_mmm():
    candidates = [
        Path("/app"),
        Path(__file__).resolve().parent.parent.parent,
        Path.cwd(),
    ]
    for root in candidates:
        for pkg in ("UmedPT", "UmedPT-sdk"):
            p = str(root / pkg)
            if Path(p).exists() and p not in sys.path:
                sys.path.insert(0, p)
    os.environ.setdefault("MMM_LICENSE_ACCEPTED", "i accept")


# ── model loading ──────────────────────────────────────────────────────────────

def load_encoder(checkpoint: str, device: str) -> nn.Module:
    _setup_mmm()
    ckpt = Path(checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Encoder checkpoint not found: {ckpt.resolve()}\n"
            f"Run from the repo root or pass --checkpoint with an absolute path."
        )
    print(f"[Model] Loading from: {ckpt}")
    encoder = torch.load(str(ckpt), map_location=device, weights_only=False)
    encoder.eval()
    print(f"[Model] Loaded ({type(encoder).__name__})  device={device}")
    return encoder


# ── hook registration ──────────────────────────────────────────────────────────

def register_hook(encoder: nn.Module, pm_index: int):
    from torchvision.models.swin_transformer import PatchMergingV2

    features = encoder.model.wrapped_model.features
    pm_layers = [
        (i, m) for i, m in enumerate(features) if isinstance(m, PatchMergingV2)
    ]
    pm_layers.append((len(features) - 1, features[-1]))  # Stage 4

    if pm_index >= len(pm_layers):
        raise IndexError(
            f"pm_index={pm_index} but only {len(pm_layers)} PatchMerging layers exist."
        )

    feat_idx, pm_module = pm_layers[pm_index]
    stage_name = {0: "Stage 1→2", 1: "Stage 2→3", 2: "Stage 3→4", 3: "Stage 4"}.get(
        pm_index, f"PM-{pm_index}"
    )
    print(f"[Hook]  Registered on {stage_name}  (features[{feat_idx}])")

    captured: dict = {}

    def _hook(module, input, output):
        captured["feat"] = output.detach().cpu()

    handle = pm_module.register_forward_hook(_hook)
    return handle, captured


# ── normalisation ──────────────────────────────────────────────────────────────

def normalize_activations(x: torch.Tensor, d_model: int) -> torch.Tensor:
    rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt() + EPS
    return x / rms * math.sqrt(d_model)


# ── per-dataset harvesting ─────────────────────────────────────────────────────

def harvest_one_dataset(
    encoder: nn.Module,
    dataset: Dataset,
    captured: dict,
    device: torch.device,
    d_model: int,
    batch_size: int,
    dataset_name: str,
    dataset_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Harvest activations for a single dataset.

    Returns
    -------
    activations  [N_eff, d_model]
    labels       [N_patches]         int32
    patch_ids    [N_patches]         int32
    dataset_ids  [N_patches]         int32  (all == dataset_idx)
    info         dict
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # Probe to get spatial shape
    with torch.inference_mode():
        encoder(torch.rand(1, 3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, device=device))
    probe = captured["feat"]
    _, H_feat, W_feat, C = probe.shape
    assert C == d_model, f"Hook returned C={C}, expected d_model={d_model}"
    n_spatial = H_feat * W_feat
    n_eff = len(dataset) * n_spatial

    print(f"\n[{dataset_name}]  images={len(dataset):,}  "
          f"spatial={H_feat}×{W_feat}  tokens={n_eff:,}")

    activations = torch.empty(n_eff, d_model, dtype=torch.float32)
    ptr = 0
    t0 = time.time()

    encoder.eval()
    with torch.inference_mode():
        for i, (images, _) in enumerate(loader):
            encoder(images.to(device))
            feat = captured["feat"]          # (B, H, W, C)
            B, H, W, C_ = feat.shape
            flat = feat.reshape(B * H * W, C_).float()
            flat = normalize_activations(flat, d_model)
            activations[ptr : ptr + flat.shape[0]] = flat
            ptr += flat.shape[0]

            if (i + 1) % 20 == 0 or (i + 1) == len(loader):
                elapsed = time.time() - t0
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"  [{ts}]  {i+1:>4}/{len(loader)}  "
                      f"{ptr:>10,} tokens  {elapsed:.0f}s")

    activations = activations[:ptr]
    n_patches = len(dataset)

    labels = (
        torch.tensor(dataset.labels, dtype=torch.int32)
        if hasattr(dataset, "labels") and dataset.labels
        else torch.full((n_patches,), -1, dtype=torch.int32)
    )
    patch_ids  = torch.arange(n_patches, dtype=torch.int32)
    dataset_ids = torch.full((n_patches,), dataset_idx, dtype=torch.int32)

    info = {
        "name":       dataset_name,
        "n_images":   n_patches,
        "n_effective": activations.shape[0],
        "spatial_h":  H_feat,
        "spatial_w":  W_feat,
    }
    print(f"[{dataset_name}]  done — {activations.shape[0]:,} activation vectors")
    return activations, labels, patch_ids, dataset_ids, info


# ── dataset loaders ────────────────────────────────────────────────────────────

def build_datasets(args) -> list[tuple[str, Dataset]]:
    """Return list of (name, dataset) for every dataset directory provided."""
    _setup_mmm()
    result = []

    if args.semicol_dir:
        print(f"[Dataset] SemiCOL  → {args.semicol_dir}")
        from datasets.semicol_dataset import SemiCOLWeakDataset
        result.append(("semicol", SemiCOLWeakDataset(
            args.semicol_dir, max_per_slide=args.max_per_slide
        )))

    if args.brain_dir:
        print(f"[Dataset] Brain Tumor → {args.brain_dir}")
        from datasets.brain_cancer_dataset import BrainTumorDataset
        result.append(("brain_tumor", BrainTumorDataset(args.brain_dir)))

    if args.breast_dir:
        print(f"[Dataset] TCGA Breast → {args.breast_dir}")
        from datasets.breast_cancer.tcga_breast_dataset import TCGABreastDataset
        result.append(("tcga_breast", TCGABreastDataset(
            args.breast_dir, args.max_per_slide
        )))

    if args.pneumonia_dir:
        print(f"[Dataset] Pneumonia  → {args.pneumonia_dir}")
        from datasets.pneumonia_dataset import PneumoniaDataset
        result.append(("pneumonia", PneumoniaDataset(args.pneumonia_dir)))

    if not result:
        raise ValueError(
            "No dataset directories provided.  Pass at least one of: "
            "--semicol_dir, --brain_dir, --breast_dir, --pneumonia_dir"
        )

    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Harvest activations from all medical-imaging datasets for "
                    "conglomerate SAE training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--checkpoint",     default=DEFAULT_CHECKPOINT)
    ap.add_argument("--pm_index",       type=int, default=DEFAULT_PM_INDEX,
                    help="PatchMerging hook: 0=Stage1→2, 1=Stage2→3, 2=Stage3→4")
    ap.add_argument("--batch_size",     type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--output",         default=DEFAULT_OUTPUT)
    ap.add_argument("--device",
                    default="cuda:0" if torch.cuda.is_available() else "cpu")

    # Dataset directories (all optional — skip missing ones)
    ap.add_argument("--semicol_dir",    type=str, default=None,
                    help="SemiCOL dataset root")
    ap.add_argument("--brain_dir",      type=str, default=None,
                    help="Brain Tumor dataset root")
    ap.add_argument("--breast_dir",     type=str, default=None,
                    help="TCGA Breast dataset root")
    ap.add_argument("--pneumonia_dir",  type=str, default=None,
                    help="Pneumonia dataset root")

    # SemiCOL / TCGA-BRCA shared option
    ap.add_argument("--max_per_slide",  type=int, default=DEFAULT_MAX_PER_SLIDE,
                    help="Max patches per WSI slide (SemiCOL Track B, TCGA Breast)")

    args = ap.parse_args()

    d_model = D_MODEL_MAP[args.pm_index]
    device  = torch.device(args.device)

    print("=" * 60)
    print("Conglomerate Activation Harvester")
    print("=" * 60)
    print(f"  Checkpoint  : {args.checkpoint}")
    print(f"  PM index    : {args.pm_index}  (d_model={d_model})")
    print(f"  Device      : {device}")
    print(f"  Output      : {args.output}")
    print()

    # 1. Datasets ──────────────────────────────────────────────────────────────
    datasets = build_datasets(args)
    print(f"\n[Datasets] {len(datasets)} dataset(s) to process: "
          f"{[n for n, _ in datasets]}\n")

    # 2. Encoder + hook ────────────────────────────────────────────────────────
    encoder = load_encoder(args.checkpoint, args.device)
    hook_handle, captured = register_hook(encoder, args.pm_index)

    # 3. Harvest each dataset ──────────────────────────────────────────────────
    all_acts     : list[torch.Tensor] = []
    all_labels   : list[torch.Tensor] = []
    all_patch_ids: list[torch.Tensor] = []
    all_ds_ids   : list[torch.Tensor] = []
    all_infos    : list[dict]         = []

    t_total = time.time()
    for ds_name, dataset in datasets:
        ds_idx = DATASET_INDEX.get(ds_name, len(all_acts))
        acts, lbls, pids, dsids, info = harvest_one_dataset(
            encoder, dataset, captured, device, d_model,
            args.batch_size, ds_name, ds_idx,
        )
        all_acts.append(acts)
        all_labels.append(lbls)
        all_patch_ids.append(pids)
        all_ds_ids.append(dsids)
        all_infos.append(info)

    hook_handle.remove()

    # 4. Concatenate ───────────────────────────────────────────────────────────
    activations = torch.cat(all_acts,      dim=0)
    labels      = torch.cat(all_labels,    dim=0)
    patch_ids   = torch.cat(all_patch_ids, dim=0)
    dataset_ids = torch.cat(all_ds_ids,    dim=0)

    n_total = activations.shape[0]
    elapsed = time.time() - t_total

    print(f"\n{'='*60}")
    print(f"Combined activations : {tuple(activations.shape)}  dtype={activations.dtype}")
    print(f"Total tokens         : {n_total:,}")
    print(f"Total time           : {elapsed:.0f}s")
    for info in all_infos:
        print(f"  {info['name']:15s}  images={info['n_images']:>7,}  "
              f"tokens={info['n_effective']:>10,}")
    print(f"{'='*60}\n")

    # 5. Save ──────────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    spatial_h = all_infos[0]["spatial_h"]
    spatial_w = all_infos[0]["spatial_w"]
    n_images_total = sum(i["n_images"] for i in all_infos)

    metadata = {
        "pm_index":         str(args.pm_index),
        "d_model":          str(d_model),
        "spatial_h":        str(spatial_h),
        "spatial_w":        str(spatial_w),
        "n_images_total":   str(n_images_total),
        "n_effective_total": str(n_total),
        "dataset_names":    json.dumps([i["name"]    for i in all_infos]),
        "dataset_counts":   json.dumps([i["n_images"] for i in all_infos]),
        "checkpoint":       str(args.checkpoint),
    }

    save_file(
        {
            "activations": activations,
            "labels":      labels,
            "patch_ids":   patch_ids,
            "dataset_ids": dataset_ids,
        },
        str(out_path),
        metadata=metadata,
    )

    # Also write a companion JSON for easy inspection
    meta_path = out_path.parent / (out_path.stem + "_metadata.json")
    with open(meta_path, "w") as f:
        json.dump({
            "pm_index":     args.pm_index,
            "d_model":      d_model,
            "spatial_h":    spatial_h,
            "spatial_w":    spatial_w,
            "n_patches":    spatial_h * spatial_w,
            "n_images_total": n_images_total,
            "n_effective_total": n_total,
            "datasets": [
                {
                    "name":    info["name"],
                    "index":   DATASET_INDEX.get(info["name"], -1),
                    "n_images": info["n_images"],
                    "n_effective": info["n_effective"],
                }
                for info in all_infos
            ],
        }, f, indent=2)

    size_mb = out_path.stat().st_size / 1e6
    print(f"Saved → {out_path}  ({size_mb:.1f} MB)")
    print(f"Meta  → {meta_path}")
    print(f"\nReady for conglomerate SAE training:")
    print(f"  python src/conglomerate/train_conglomerate.py \\")
    print(f"      --activations_path {out_path} \\")
    print(f"      --d_model {d_model}")


if __name__ == "__main__":
    main()
