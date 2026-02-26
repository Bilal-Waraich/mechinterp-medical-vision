"""
Train a BatchTopK Sparse Autoencoder on pre-harvested UmedPT activations.

Uses the SAELens library (BatchTopKTrainingSAE + SAETrainer) with activations
produced by harvest_activations.py.

BatchTopK vs TopK
-----------------
  TopK    : each sample keeps exactly k features active.
  BatchTopK: k * batch_size features are active across the whole batch — the
             mean L0 is fixed at k, but individual samples can have more or fewer
             active features.  At inference the SAE is converted to JumpReLU.

Pipeline
--------
  1. Load pre-normalised activations from data/batchtopk_activations.safetensors
  2. Build TensorDataProvider (infinite, shuffling iterator over the tensor)
  3. Configure BatchTopKTrainingSAEConfig (d_in, d_sae, k, ...)
  4. Configure SAETrainerConfig (lr, schedule, checkpoints, logging, ...)
  5. Run SAETrainer.fit() -- returns the trained SAE
  6. Save trained SAE (BatchTopK -> JumpReLU inference format)

Usage
-----
  # Minimal: reads data/batchtopk_activations.safetensors, saves to sae_output/
  python phase-1-topk/scripts/train_sae.py

  # Full options
  python phase-1-topk/scripts/train_sae.py \\
      --activations  data/batchtopk_activations.safetensors \\
      --output_dir   sae_output/ \\
      --d_sae        8192 \\
      --k            50 \\
      --lr           5e-5 \\
      --n_epochs     10 \\
      --batch_size   4096 \\
      --device       cuda:0 \\
      --wandb

Key hyperparameters
-------------------
  --d_sae          Hidden dimension of the SAE.  Default: 16 x d_in.
  --k              Mean active features per batch step (BatchTopK sparsity).
                   Typical: k ~ 0.5% - 2% of d_sae.  Default: 50.
  --expansion      Alternative to --d_sae: expansion factor x d_in.  Default: 16.
  --lr             Peak learning rate.  Default: 5e-5.
  --warmup_steps   LR linear warmup steps.  Default: 500.
  --n_epochs       Full passes over the activation dataset.  Default: 10.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator

import torch

try:
    from safetensors.torch import load_file as st_load
    from safetensors import safe_open
except ImportError:
    sys.exit("safetensors not installed.  Run: pip install safetensors")

try:
    from sae_lens import BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig, SAETrainer
    from sae_lens.config import SAETrainerConfig, LoggingConfig
except ImportError:
    sys.exit(
        "sae-lens not installed.  Run: pip install sae-lens\n"
        "See: https://github.com/decoderesearch/SAELens"
    )


# ---- Defaults ----------------------------------------------------------------

DEFAULT_ACTIVATIONS   = "data/batchtopk_activations.safetensors"
DEFAULT_OUTPUT_DIR    = "sae_output"
DEFAULT_EXPANSION     = 16        # d_sae = expansion * d_in
DEFAULT_K             = 50        # BatchTopK active features per batch step
DEFAULT_LR            = 5e-5
DEFAULT_LR_END        = 1e-6
DEFAULT_BATCH_SIZE    = 4096
DEFAULT_N_EPOCHS      = 10
DEFAULT_WARMUP_STEPS  = 500
DEFAULT_N_CHECKPOINTS = 5
DEFAULT_DEAD_WINDOW   = 1_000


# ---- Data provider -----------------------------------------------------------

class TensorDataProvider:
    """
    Infinite, shuffling iterator over a pre-loaded activation tensor.

    Each call to __next__ returns the next batch.  When an epoch ends the
    tensor is re-shuffled automatically.

    SAETrainer expects a DataProvider = Iterator[torch.Tensor] that yields
    batches of shape (batch_size, d_in).
    """

    def __init__(self, activations: torch.Tensor, batch_size: int) -> None:
        self.activations = activations          # (N, d_in) float32
        self.batch_size  = batch_size
        self.n           = len(activations)
        self._indices    = torch.randperm(self.n)
        self._pos        = 0

    def __iter__(self) -> Iterator[torch.Tensor]:
        return self

    def __next__(self) -> torch.Tensor:
        if self._pos + self.batch_size > self.n:
            self._indices = torch.randperm(self.n)
            self._pos     = 0
        idx        = self._indices[self._pos : self._pos + self.batch_size]
        self._pos += self.batch_size
        return self.activations[idx]     # (batch_size, d_in)


# ---- Helpers -----------------------------------------------------------------

def _load_activations(path: str) -> tuple[torch.Tensor, dict[str, str]]:
    """Load activations and metadata from a safetensors file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Activations file not found: {p.resolve()}\n"
            f"Hint: run harvest_activations.py first."
        )
    metadata: dict[str, str] = {}
    with safe_open(str(p), framework="pt") as f:
        metadata = dict(f.metadata() or {})
    activations = st_load(str(p))["activations"]
    return activations, metadata


def _resolve_d_sae(d_in: int, d_sae_arg: int | None, expansion_arg: int) -> int:
    return d_sae_arg if d_sae_arg is not None else d_in * expansion_arg


# ---- Main --------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train BatchTopK SAE on UmedPT activations using SAELens",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--activations",   default=DEFAULT_ACTIVATIONS,
                    help="Path to .safetensors file from harvest_activations.py")
    ap.add_argument("--output_dir",    default=DEFAULT_OUTPUT_DIR,
                    help="Directory to save the trained SAE")
    ap.add_argument("--d_sae",         type=int, default=None,
                    help="SAE hidden dimension.  Overrides --expansion if set.")
    ap.add_argument("--expansion",     type=int, default=DEFAULT_EXPANSION,
                    help="d_sae = expansion * d_in  (used when --d_sae is not set)")
    ap.add_argument("--k",             type=int, default=DEFAULT_K,
                    help="Mean active features per batch step (BatchTopK sparsity)")
    ap.add_argument("--lr",            type=float, default=DEFAULT_LR)
    ap.add_argument("--lr_end",        type=float, default=DEFAULT_LR_END,
                    help="Final LR (cosine decay target)")
    ap.add_argument("--batch_size",    type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--n_epochs",      type=int, default=DEFAULT_N_EPOCHS,
                    help="Number of passes over the full activation dataset")
    ap.add_argument("--warmup_steps",  type=int, default=DEFAULT_WARMUP_STEPS,
                    help="Linear LR warm-up steps")
    ap.add_argument("--n_checkpoints", type=int, default=DEFAULT_N_CHECKPOINTS)
    ap.add_argument("--dead_window",   type=int, default=DEFAULT_DEAD_WINDOW,
                    help="Steps before a feature is considered dead")
    ap.add_argument("--device",
                    default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--wandb",         action="store_true",
                    help="Log to Weights & Biases")
    ap.add_argument("--wandb_project", default="umedpt-batchtopk-sae")
    ap.add_argument("--run_name",      default=None,
                    help="W&B run name (auto-generated if not set)")
    args = ap.parse_args()

    device  = torch.device(args.device)
    use_amp = device.type == "cuda"

    # 1. Load activations
    print(f"{'='*64}")
    print(f"BatchTopK SAE Trainer  --  UmedPT Swin Transformer")
    print(f"{'='*64}\n")
    print(f"[Data]  Loading: {args.activations}")

    activations, meta = _load_activations(args.activations)
    n_total, d_in     = activations.shape
    act_scale         = float(meta.get("activation_scale", 1.0))

    print(f"[Data]  Shape            : {tuple(activations.shape)}")
    print(f"[Data]  d_in             : {d_in}")
    print(f"[Data]  N activations    : {n_total:,}")
    print(f"[Data]  activation_scale : {act_scale:.6f}  (from harvest)")
    print(f"[Data]  Checkpoint used  : {meta.get('checkpoint', 'unknown')}")
    print(f"[Data]  PM index         : {meta.get('pm_index', 'unknown')}\n")

    # 2. SAE configuration
    d_sae = _resolve_d_sae(d_in, args.d_sae, args.expansion)

    total_training_samples = n_total * args.n_epochs
    total_training_steps   = total_training_samples // args.batch_size
    decay_steps            = max(total_training_steps - args.warmup_steps, 1)

    print(f"[SAE]   Architecture     : BatchTopK -> JumpReLU (inference)")
    print(f"[SAE]   d_in             : {d_in}")
    print(f"[SAE]   d_sae            : {d_sae}  (expansion={d_sae // d_in}x)")
    print(f"[SAE]   k                : {args.k}  (mean active features/step)")
    print(f"[Train] Total samples    : {total_training_samples:,}  ({args.n_epochs} epochs)")
    print(f"[Train] Total steps      : {total_training_steps:,}")
    print(f"[Train] Batch size       : {args.batch_size}")
    print(f"[Train] LR               : {args.lr} -> {args.lr_end}  (cosineannealing)")
    print(f"[Train] Warmup steps     : {args.warmup_steps}")
    print(f"[Train] Device           : {device}  AMP={use_amp}")
    print(f"[Train] W&B              : {args.wandb}\n")

    # 3. Instantiate BatchTopK SAE
    # normalize_activations="none" because harvest_activations.py already
    # applied the global BatchTopK scale.
    sae_cfg = BatchTopKTrainingSAEConfig(
        d_in                  = d_in,
        d_sae                 = d_sae,
        k                     = args.k,
        normalize_activations = "none",
        dtype                 = "float32",
    )
    sae = BatchTopKTrainingSAE(sae_cfg)

    # 4. Trainer configuration
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = str(out_dir / "checkpoints")
    run_name = args.run_name or f"batchtopk_d{d_in}_sae{d_sae}_k{args.k}"

    trainer_cfg = SAETrainerConfig(
        device                   = str(device),
        autocast                 = use_amp,
        lr                       = args.lr,
        lr_end                   = args.lr_end,
        lr_scheduler_name        = "cosineannealing",
        lr_warm_up_steps         = args.warmup_steps,
        lr_decay_steps           = decay_steps,
        n_restart_cycles         = 1,
        adam_beta1               = 0.9,
        adam_beta2               = 0.999,
        train_batch_size_samples = args.batch_size,
        total_training_samples   = total_training_samples,
        dead_feature_window      = args.dead_window,
        feature_sampling_window  = max(args.dead_window // 10, 100),
        n_checkpoints            = args.n_checkpoints,
        checkpoint_path          = ckpt_dir,
        save_final_checkpoint    = True,
        logger = LoggingConfig(
            log_to_wandb          = args.wandb,
            wandb_project         = args.wandb_project,
            run_name              = run_name,
            wandb_log_frequency   = 50,
            eval_every_n_wandb_logs = 20,
        ),
    )

    # 5. Data provider
    activations   = activations.to(device)
    data_provider = TensorDataProvider(activations, args.batch_size)

    # 6. Train
    print("[Train] Starting SAETrainer.fit() ...\n")
    trainer     = SAETrainer(trainer_cfg, sae, data_provider)
    trained_sae = trainer.fit()

    # 7. Save
    # Fold the harvest activation_scale into SAE weights so the inference SAE
    # operates on raw (un-scaled) activations from the encoder.
    print(f"\n[Save]  Folding activation_scale={act_scale:.6f} into SAE weights ...")
    if hasattr(trained_sae, "fold_activation_norm_scaling_factor"):
        trained_sae.fold_activation_norm_scaling_factor(act_scale)

    inference_path = str(out_dir / "inference")
    print(f"[Save]  Writing inference model -> {inference_path}")
    trained_sae.save_model(inference_path)

    print(f"\n{'='*64}")
    print(f"Training complete.")
    print(f"  Inference SAE : {inference_path}")
    print(f"  Checkpoints   : {ckpt_dir}")
    print(f"{'='*64}")
    print(f"\n->trained BatchTopK SAE saved to {out_dir}")


if __name__ == "__main__":
    main()
