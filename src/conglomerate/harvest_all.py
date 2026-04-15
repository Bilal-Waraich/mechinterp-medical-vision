"""
Harvest activations from all four X-ray datasets into one combined file.

Loads the UMedPT encoder once, registers a single PatchMerging hook, then runs
each dataset through in sequence.  Per-dataset centering is applied before
concatenation.  All tensors are saved as a single safetensors file suitable for
stratified conglomerate SAE training.

Datasets
--------
  chexpert    CheXpert (PA only, 10k cap)
  rsna        RSNA Pneumonia Detection (PA DICOM, 10k cap)
  siim_covid  SIIM-COVID19 (PA, bbox metadata stored, 10k cap)
  tb          TB Shenzhen (all 700, augmented)

Any dataset whose ``--*_dir`` flag is omitted is silently skipped.

Output tensors
--------------
  activations  [N_total, d_model]  float32  RMS-normalised, per-dataset-centred
  labels       [N_patches]         int32    canonical group index (-1 if unknown)
  patch_ids    [N_patches]         int32    image index within each dataset
  dataset_ids  [N_patches]         int32    0=chexpert, 1=rsna, 2=siim_covid, 3=tb

Metadata keys
-------------
  d_model, spatial_h, spatial_w, pm_index, n_images_total,
  n_effective_total, dataset_names (JSON), dataset_counts (JSON),
  dataset_means (JSON list of lists — per-dataset mean vectors for reproducibility)

Usage
-----
  python src/conglomerate/harvest_all.py \\
      --checkpoint encoder-1.0.4.pt/encoder.pt \\
      --chexpert_dir   /data/chexpert \\
      --rsna_dir       /data/rsna-pneumonia \\
      --siim_dir       /data/siim-covid19 \\
      --tb_dir         /data/tb-shenzhen \\
      --output  data/xray_activations.safetensors \\
      --pm_index 2 \\
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
DEFAULT_OUTPUT      = "data/xray_activations.safetensors"
DEFAULT_PM_INDEX    = 2
DEFAULT_BATCH_SIZE  = 32
DEFAULT_IMAGE_SIZE  = 224
DEFAULT_N_SAMPLES   = 10_000   # per large dataset; TB always uses all images
EPS = 1e-8

D_MODEL_MAP   = {0: 256, 1: 512, 2: 1024, 3: 1024}
N_PATCHES_MAP = {0: 784, 1: 196, 2: 49,   3: 49}

DATASET_INDEX = {
    "chexpert":   0,
    "rsna":       1,
    "siim_covid": 2,
    "tb":         3,
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
            f"Run from repo root or pass --checkpoint with an absolute path."
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
    stage_name = {0: "Stage 1->2", 1: "Stage 2->3", 2: "Stage 3->4", 3: "Stage 4"}.get(
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
    """Anthropic-style RMS normalisation per token."""
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
    """Harvest RMS-normalised activations for one dataset.

    Returns
    -------
    activations  [N_eff, d_model]  — RMS-normalised, NOT yet centred
    labels       [N_images]        int32
    patch_ids    [N_images]        int32
    dataset_ids  [N_images]        int32
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

    # Probe spatial shape
    with torch.inference_mode():
        encoder(torch.rand(1, 3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, device=device))
    probe = captured["feat"]
    _, H_feat, W_feat, C = probe.shape
    assert C == d_model, f"Hook returned C={C}, expected d_model={d_model}"
    n_spatial = H_feat * W_feat
    n_eff = len(dataset) * n_spatial

    print(f"\n[{dataset_name}]  images={len(dataset):,}  "
          f"spatial={H_feat}x{W_feat}  tokens={n_eff:,}")

    activations = torch.empty(n_eff, d_model, dtype=torch.float32)
    ptr = 0
    t0 = time.time()

    encoder.eval()
    with torch.inference_mode():
        for i, (images, _) in enumerate(loader):
            encoder(images.to(device))
            feat = captured["feat"]           # (B, H, W, C)
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
    patch_ids   = torch.arange(n_patches, dtype=torch.int32)
    dataset_ids = torch.full((n_patches,), dataset_idx, dtype=torch.int32)

    info = {
        "name":        dataset_name,
        "n_images":    n_patches,
        "n_effective": activations.shape[0],
        "spatial_h":   H_feat,
        "spatial_w":   W_feat,
    }
    print(f"[{dataset_name}]  done — {activations.shape[0]:,} activation vectors")
    return activations, labels, patch_ids, dataset_ids, info


# ── per-dataset centring ───────────────────────────────────────────────────────

def center_activations(
    all_acts: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[list[float]]]:
    """Subtract each dataset's mean activation vector.

    This corrects for cross-dataset b_dec drift (concern #8): different
    hospitals/scanners have systematic mean activation shifts that bias the
    global mean and cause reconstruction error in every dataset.

    Returns the centred tensors and the mean vectors (for metadata storage).
    """
    centred = []
    means = []
    for acts in all_acts:
        mean = acts.mean(dim=0)           # [d_model]
        centred.append(acts - mean)
        means.append(mean.tolist())
    return centred, means


# ── dataset builder ────────────────────────────────────────────────────────────

def build_datasets(args) -> list[tuple[str, Dataset]]:
    """Return (name, dataset) for every dataset directory provided."""
    _setup_mmm()
    result = []

    if args.chexpert_dir:
        print(f"[Dataset] CheXpert    -> {args.chexpert_dir}")
        from datasets.chexpert_dataset import CheXpertDataset
        result.append(("chexpert", CheXpertDataset(
            args.chexpert_dir,
            n_samples=args.n_samples,
            seed=args.seed,
        )))

    if args.rsna_dir:
        print(f"[Dataset] RSNA        -> {args.rsna_dir}")
        from datasets.pneumonia_dataset import PneumoniaDataset
        # PneumoniaDataset does not yet support n_samples; wrap with SubsetDataset
        ds = PneumoniaDataset(args.rsna_dir)
        result.append(("rsna", _maybe_subsample(ds, args.n_samples, args.seed)))

    if args.siim_dir:
        print(f"[Dataset] SIIM-COVID  -> {args.siim_dir}")
        from datasets.siim_covid_dataset import SIIMCovidDataset
        result.append(("siim_covid", SIIMCovidDataset(
            args.siim_dir,
            n_samples=args.n_samples,
            seed=args.seed,
        )))

    if args.tb_dir:
        print(f"[Dataset] TB Shenzhen -> {args.tb_dir}")
        from datasets.tb_shenzhen_dataset import TBShenzhenDataset
        # TB: always use all images (no n_samples cap)
        result.append(("tb", TBShenzhenDataset(args.tb_dir, augment=True)))

    if not result:
        raise ValueError(
            "No dataset directories provided. Pass at least one of: "
            "--chexpert_dir, --rsna_dir, --siim_dir, --tb_dir"
        )

    return result


def _maybe_subsample(dataset: Dataset, n_samples: int | None, seed: int) -> Dataset:
    """Wrap dataset in a random subset if n_samples < len(dataset)."""
    if n_samples is None or n_samples >= len(dataset):
        return dataset
    import random as _random
    rng = _random.Random(seed)
    indices = rng.sample(range(len(dataset)), n_samples)
    from torch.utils.data import Subset
    return Subset(dataset, indices)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Harvest X-ray activations for conglomerate SAE training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--checkpoint",   default=DEFAULT_CHECKPOINT)
    ap.add_argument("--pm_index",     type=int, default=DEFAULT_PM_INDEX,
                    help="PatchMerging hook: 0=Stage1->2, 1=Stage2->3, 2=Stage3->4")
    ap.add_argument("--batch_size",   type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--output",       default=DEFAULT_OUTPUT)
    ap.add_argument("--n_samples",    type=int, default=DEFAULT_N_SAMPLES,
                    help="Max images per large dataset (CheXpert, RSNA, SIIM). "
                         "TB always uses all images.")
    ap.add_argument("--seed",         type=int, default=42,
                    help="Random seed for deterministic subsampling.")
    ap.add_argument("--no_centering", action="store_true",
                    help="Skip per-dataset activation centering (not recommended).")
    ap.add_argument("--device",
                    default="cuda:0" if torch.cuda.is_available() else "cpu")

    # Dataset directories (all optional)
    ap.add_argument("--chexpert_dir", type=str, default=None)
    ap.add_argument("--rsna_dir",     type=str, default=None)
    ap.add_argument("--siim_dir",     type=str, default=None)
    ap.add_argument("--tb_dir",       type=str, default=None)

    args = ap.parse_args()

    d_model = D_MODEL_MAP[args.pm_index]
    device  = torch.device(args.device)

    print("=" * 60)
    print("X-ray Conglomerate Activation Harvester")
    print("=" * 60)
    print(f"  Checkpoint  : {args.checkpoint}")
    print(f"  PM index    : {args.pm_index}  (d_model={d_model})")
    print(f"  n_samples   : {args.n_samples} per large dataset")
    print(f"  Centering   : {not args.no_centering}")
    print(f"  Device      : {device}")
    print(f"  Output      : {args.output}")
    print()

    # 1. Datasets
    datasets = build_datasets(args)
    print(f"\n[Datasets] {len(datasets)} dataset(s): {[n for n, _ in datasets]}\n")

    # 2. Encoder + hook
    encoder = load_encoder(args.checkpoint, args.device)
    hook_handle, captured = register_hook(encoder, args.pm_index)

    # 3. Harvest each dataset
    all_acts:      list[torch.Tensor] = []
    all_labels:    list[torch.Tensor] = []
    all_patch_ids: list[torch.Tensor] = []
    all_ds_ids:    list[torch.Tensor] = []
    all_infos:     list[dict]         = []

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

    # 4. Per-dataset centering (concern #8: cross-dataset b_dec drift)
    dataset_means: list[list[float]] = []
    if not args.no_centering:
        print("\n[Centering] Applying per-dataset activation centering...")
        all_acts, dataset_means = center_activations(all_acts)
        for info, mean_vec in zip(all_infos, dataset_means):
            mean_norm = (sum(v * v for v in mean_vec) ** 0.5)
            print(f"  {info['name']:15s}  mean_norm={mean_norm:.4f}")
    else:
        print("\n[Centering] Skipped (--no_centering flag set).")
        dataset_means = [[] for _ in all_infos]

    # 5. Concatenate
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

    # 6. Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    spatial_h = all_infos[0]["spatial_h"]
    spatial_w = all_infos[0]["spatial_w"]
    n_images_total = sum(i["n_images"] for i in all_infos)

    metadata = {
        "pm_index":          str(args.pm_index),
        "d_model":           str(d_model),
        "spatial_h":         str(spatial_h),
        "spatial_w":         str(spatial_w),
        "n_images_total":    str(n_images_total),
        "n_effective_total": str(n_total),
        "dataset_names":     json.dumps([i["name"]    for i in all_infos]),
        "dataset_counts":    json.dumps([i["n_images"] for i in all_infos]),
        "dataset_means":     json.dumps(dataset_means),
        "checkpoint":        str(args.checkpoint),
        "centred":           str(not args.no_centering),
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

    # Companion JSON
    meta_path = out_path.parent / (out_path.stem + "_metadata.json")
    with open(meta_path, "w") as f:
        json.dump({
            "pm_index":          args.pm_index,
            "d_model":           d_model,
            "spatial_h":         spatial_h,
            "spatial_w":         spatial_w,
            "n_patches":         spatial_h * spatial_w,
            "n_images_total":    n_images_total,
            "n_effective_total": n_total,
            "centred":           not args.no_centering,
            "datasets": [
                {
                    "name":        info["name"],
                    "index":       DATASET_INDEX.get(info["name"], -1),
                    "n_images":    info["n_images"],
                    "n_effective": info["n_effective"],
                }
                for info in all_infos
            ],
        }, f, indent=2)

    size_mb = out_path.stat().st_size / 1e6
    print(f"Saved -> {out_path}  ({size_mb:.1f} MB)")
    print(f"Meta  -> {meta_path}")
    print(f"\nReady for stratified SAE training:")
    print(f"  python src/train_sae.py \\")
    print(f"      --activations_path {out_path} \\")
    print(f"      --d_model {d_model}")


if __name__ == "__main__":
    main()
