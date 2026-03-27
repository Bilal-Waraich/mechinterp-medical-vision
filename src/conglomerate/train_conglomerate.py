"""Train a BatchTopK SAE on the conglomerate (all-datasets) activations.

This is a thin wrapper around the core training logic in src/train_sae.py.
It sets conglomerate-appropriate defaults and adds a --run_name flag for
organising multi-run experiments.

Pipeline
--------
  1. Harvest activations from all datasets:
       python src/conglomerate/harvest_all.py  [dataset dirs]
  2. Train the conglomerate SAE (this script):
       python src/conglomerate/train_conglomerate.py \\
           --activations_path data/conglomerate_activations.safetensors
  3. Compare against individual SAEs:
       python src/conglomerate/compare_conglomerate.py  [args]

Differences from individual-dataset training
---------------------------------------------
  - Default activations_path points to the conglomerate file
  - Default output_dir uses a 'conglomerate/' subdirectory
  - Default steps are higher (200 k) to account for the larger dataset
  - Wandb run name is tagged 'conglomerate' automatically

Usage
-----
  # Minimal (uses all defaults)
  python src/conglomerate/train_conglomerate.py \\
      --activations_path data/conglomerate_activations.safetensors

  # Full control
  python src/conglomerate/train_conglomerate.py \\
      --activations_path data/conglomerate_activations.safetensors \\
      --d_model 512 --k 32 --expansion_factor 4 \\
      --steps 200000 --batch_size 2048 --lr 2e-4 \\
      --output_dir data/conglomerate/checkpoints \\
      --device cuda:0 --use_amp
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import SAEConfig  # noqa: E402

# Conglomerate-specific defaults
CONG_ACTIVATIONS = "data/conglomerate_activations.safetensors"
CONG_OUTPUT_DIR  = "data/conglomerate/checkpoints"
CONG_STEPS       = 200_000
CONG_WANDB_PROJECT = "sae-umedpt-conglomerate"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train BatchTopK SAE on conglomerate (all-datasets) activations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",            type=str, default=None,
                        help="Path to JSON config file.")
    parser.add_argument("--activations_path",  type=str, default=CONG_ACTIVATIONS)
    parser.add_argument("--output_dir",        type=str, default=CONG_OUTPUT_DIR)
    parser.add_argument("--d_model",           type=int, default=None,
                        help="Activation dimension. Auto-detected from file if omitted.")
    parser.add_argument("--expansion_factor",  type=int, default=4)
    parser.add_argument("--k",                 type=int, default=32)
    parser.add_argument("--n_batches_to_dead", type=int, default=10)
    parser.add_argument("--top_k_aux",         type=int, default=512)
    parser.add_argument("--aux_penalty",       type=float, default=1/32)
    parser.add_argument("--l1_coeff",          type=float, default=2e-4)
    parser.add_argument("--lr",                type=float, default=2e-4)
    parser.add_argument("--lr_warmup_steps",   type=int, default=1000)
    parser.add_argument("--lr_schedule",       type=str, default="cosine",
                        choices=["constant", "cosine"])
    parser.add_argument("--beta1",             type=float, default=0.9)
    parser.add_argument("--beta2",             type=float, default=0.999)
    parser.add_argument("--max_grad_norm",     type=float, default=1.0)
    parser.add_argument("--batch_size",        type=int, default=2048)
    parser.add_argument("--steps",             type=int, default=CONG_STEPS)
    parser.add_argument("--checkpoint_every",  type=int, default=10_000)
    parser.add_argument("--wandb_project",     type=str, default=CONG_WANDB_PROJECT)
    parser.add_argument("--device",            type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_amp",           action="store_true", default=False)
    return parser


def auto_detect_d_model(activations_path: str) -> int | None:
    """Try to read d_model from a companion *_metadata.json if present."""
    meta_path = Path(activations_path).parent / (
        Path(activations_path).stem + "_metadata.json"
    )
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        d = meta.get("d_model")
        if d:
            print(f"[Config] d_model={d} auto-detected from {meta_path}")
            return int(d)

    # Fallback: read the safetensors header
    try:
        from safetensors.torch import load_file
        raw = load_file(str(activations_path))
        d = raw["activations"].shape[1]
        print(f"[Config] d_model={d} auto-detected from activations tensor shape")
        return d
    except Exception:
        return None


def main():
    parser = make_parser()
    args = parser.parse_args()

    # ── Build SAEConfig ────────────────────────────────────────────────────
    cfg = SAEConfig()

    # Start from JSON config if provided
    if args.config:
        with open(args.config) as f:
            for k, v in json.load(f).items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)

    # Apply CLI overrides
    for field, value in vars(args).items():
        if field == "config":
            continue
        if value is not None and hasattr(cfg, field):
            setattr(cfg, field, value)

    # Auto-detect d_model if not set explicitly
    if args.d_model is None:
        detected = auto_detect_d_model(cfg.activations_path)
        if detected:
            cfg.d_model = detected
        else:
            print(
                "[Warning] Could not auto-detect d_model.  "
                "Defaulting to 512 (Stage 2→3).  "
                "Pass --d_model explicitly if this is wrong."
            )
            cfg.d_model = 512

    print("=" * 60)
    print("Conglomerate SAE Trainer")
    print("=" * 60)
    print(f"  activations_path  : {cfg.activations_path}")
    print(f"  d_model           : {cfg.d_model}")
    print(f"  d_sae             : {cfg.d_sae}  (× {cfg.expansion_factor})")
    print(f"  k                 : {cfg.k}")
    print(f"  steps             : {cfg.steps:,}")
    print(f"  batch_size        : {cfg.batch_size}")
    print(f"  lr                : {cfg.lr}  schedule={cfg.lr_schedule}")
    print(f"  output_dir        : {cfg.output_dir}")
    print(f"  device            : {cfg.device}")
    print(f"  use_amp           : {cfg.use_amp}")
    print()

    # ── Delegate to the shared training loop ──────────────────────────────
    # Import here to avoid polluting the top-level namespace
    from train_sae import train  # noqa: PLC0415
    train(cfg)


if __name__ == "__main__":
    main()
