"""Patch missing image_stems into existing Matryoshka safetensors metadata.

The Matryoshka L3 harvest did not embed image_stems in the safetensors metadata,
blocking spatial grounding analysis.  This script reconstructs the stems by
instantiating the same dataset classes with the same n_samples/seed that the
original harvest used, then rewrites each safetensors file with the stems added.

No GPU or encoder is needed — only the dataset file lists are read.

Usage:
    python src/scripts/patch_stems_metadata.py \\
        --chexpert_dir  /deep_learning/input/data/mmm/chexpert/CheXpert-v1.0 \\
        --rsna_dir      /deep_learning/input/data/mmm/rsna-pneumonia-detection-challenge \\
        --siim_dir      /deep_learning/input/data/mmm/siim-covid19-detection \\
        --tb_dir        /deep_learning/input/data/mmm/tb-shenzhen \\
        --safetensors_dir /deep_learning/output/UMed-SAE/matryoshka/layer-3/.../xray_v1 \\
        --n_samples     10000 \\
        --seed          42
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch


def _setup_path():
    repo_root = Path(__file__).resolve().parents[2]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def get_stems(dataset) -> list[str]:
    return [Path(p).stem for p in dataset.image_paths]


def patch_file(path: Path, dataset_label: str, stems: list[str], dry_run: bool):
    from safetensors import safe_open
    from safetensors.torch import save_file

    print(f"\n  Patching {path.name} ...")

    tensors = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        meta = dict(f.metadata() or {})
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    n_images = tensors["labels"].shape[0]
    if len(stems) != n_images:
        raise ValueError(
            f"{path.name}: dataset has {len(stems)} stems but safetensors has "
            f"{n_images} images — n_samples/seed mismatch, aborting"
        )

    if "image_stems" in meta:
        print(f"    Already has image_stems — skipping")
        return

    # Store stems as JSON, keyed by dataset label (matches load_dataset_activations)
    meta["image_stems"] = json.dumps({dataset_label: stems})
    print(f"    Adding image_stems: {len(stems)} stems  (first 3: {stems[:3]})")

    if dry_run:
        print(f"    [dry-run] would write {path}")
        return

    tmp = path.with_suffix(".tmp.safetensors")
    save_file(tensors, str(tmp), metadata=meta)
    shutil.move(str(tmp), str(path))
    print(f"    Written.")


def main():
    _setup_path()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chexpert_dir",    type=Path)
    ap.add_argument("--rsna_dir",        type=Path)
    ap.add_argument("--siim_dir",        type=Path)
    ap.add_argument("--tb_dir",          type=Path)
    ap.add_argument("--safetensors_dir", type=Path, required=True,
                    help="Directory containing activations_*.safetensors files")
    ap.add_argument("--n_samples",       type=int, default=10_000)
    ap.add_argument("--seed",            type=int, default=42)
    ap.add_argument("--dry_run",         action="store_true",
                    help="Show what would be done without writing")
    args = ap.parse_args()

    _setup_mmm()

    datasets_to_patch: list[tuple[str, Path, object]] = []

    if args.chexpert_dir:
        from datasets.chexpert_dataset import CheXpertDataset
        ds = CheXpertDataset(args.chexpert_dir, n_samples=args.n_samples, seed=args.seed)
        sf = args.safetensors_dir / "activations_chexpert.safetensors"
        datasets_to_patch.append(("chexpert", sf, ds))

    if args.rsna_dir:
        from datasets.pneumonia_dataset import PneumoniaDataset
        ds = PneumoniaDataset(args.rsna_dir, n_samples=args.n_samples, seed=args.seed)
        sf = args.safetensors_dir / "activations_rsna.safetensors"
        datasets_to_patch.append(("rsna", sf, ds))

    if args.siim_dir:
        from datasets.siim_covid_dataset import SIIMCovidDataset
        ds = SIIMCovidDataset(args.siim_dir, n_samples=args.n_samples, seed=args.seed)
        sf = args.safetensors_dir / "activations_siim.safetensors"
        datasets_to_patch.append(("siim_covid", sf, ds))

    if args.tb_dir:
        from datasets.tb_shenzhen_dataset import TBShenzhenDataset
        ds = TBShenzhenDataset(args.tb_dir, augment=True)
        sf = args.safetensors_dir / "activations_tb.safetensors"
        datasets_to_patch.append(("tb", sf, ds))

    if not datasets_to_patch:
        ap.error("Pass at least one dataset directory.")

    for label, sf_path, dataset in datasets_to_patch:
        if not sf_path.exists():
            print(f"  SKIP {sf_path.name} — file not found")
            continue
        stems = get_stems(dataset)
        patch_file(sf_path, label, stems, dry_run=args.dry_run)

    print("\nDone.")


def _setup_mmm():
    """Register mmm as a package so dataset imports work."""
    mmm_root = Path("/deep_learning/input/data/MechanisticInterpretability_BWA_NLA")
    if mmm_root.exists() and str(mmm_root) not in sys.path:
        sys.path.insert(0, str(mmm_root))


if __name__ == "__main__":
    main()
