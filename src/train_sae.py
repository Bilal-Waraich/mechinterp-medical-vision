"""Train a BatchTopK SAE on pre-harvested UMedPT activations.

Training follows the bartbussmann/BatchTopK reference:
  - Adam optimiser with configurable betas
  - Gradient clipping before the decoder constraint step
  - make_decoder_weights_and_grad_unit_norm() called AFTER backward and BEFORE
    optimizer.step() — this is the correct order for the unit-norm constraint
  - Checkpoints saved as sae_weights.safetensors + cfg.json (SAELens-compatible)
  - Wandb logging with SAELens-compatible metric names

Pipeline
--------
  1. harvest_activations.py  →  data/dummy_activations.safetensors
  2. python src/train_sae.py   (this script)
  3. Checkpoints written to data/checkpoints/step_XXXXXXX/

Usage
-----
  # Default config (d_model=384, k=32, 50k steps)
  python src/train_sae.py

  # Custom settings
  python src/train_sae.py --d_model 192 --k 16 --steps 100000

  # Different activation file
  python src/train_sae.py --activations_path data/my_acts.safetensors

Reference
---------
  https://github.com/bartbussmann/BatchTopK
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, TensorDataset

from config import SAEConfig
from sae_models import BatchTopKSAE

# ── Optional wandb ────────────────────────────────────────────────────────────

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ── Metric helpers ────────────────────────────────────────────────────────────

def explained_variance(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """Fraction of input variance explained by the reconstruction.

    ev = 1 - Var(x - x_hat) / Var(x)
    """
    residual_var = (x - x_hat).var().item()
    input_var = x.var().item()
    if input_var < 1e-8:
        return 0.0
    return 1.0 - residual_var / input_var


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(model: BatchTopKSAE, cfg: SAEConfig, step: int) -> Path:
    """Save weights + config in SAELens-compatible format.

    Layout::

        <output_dir>/step_XXXXXXX/
            sae_weights.safetensors   ← model state dict
            cfg.json                  ← architecture + training config
    """
    ckpt_dir = Path(cfg.output_dir) / f"step_{step:07d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Weights — contiguous tensors required by safetensors
    save_file(
        {k: v.contiguous().cpu() for k, v in model.state_dict().items()},
        str(ckpt_dir / "sae_weights.safetensors"),
    )

    # Config (SAELens-compatible fields)
    cfg_dict = {
        "architecture": "batch_topk",
        "d_model": cfg.d_model,
        "d_sae": cfg.d_sae,
        "k": cfg.k,
        "expansion_factor": cfg.expansion_factor,
        "n_batches_to_dead": cfg.n_batches_to_dead,
        "top_k_aux": cfg.top_k_aux,
        "aux_penalty": cfg.aux_penalty,
        "l1_coeff": cfg.l1_coeff,
        "lr": cfg.lr,
        "beta1": cfg.beta1,
        "beta2": cfg.beta2,
        "max_grad_norm": cfg.max_grad_norm,
        "batch_size": cfg.batch_size,
        "steps": cfg.steps,
        "activations_path": cfg.activations_path,
        "step": step,
    }
    (ckpt_dir / "cfg.json").write_text(json.dumps(cfg_dict, indent=2))

    return ckpt_dir


# ── Training loop ─────────────────────────────────────────────────────────────

def train(cfg: SAEConfig) -> BatchTopKSAE:
    """Run the full SAE training loop.

    Training order (matches bartbussmann/BatchTopK reference):
      1. forward pass
      2. loss.backward()
      3. clip_grad_norm_()
      4. make_decoder_weights_and_grad_unit_norm()  ← project + normalize
      5. optimizer.step()
      6. optimizer.zero_grad()

    Parameters
    ----------
    cfg : SAEConfig

    Returns
    -------
    Trained BatchTopKSAE on CPU.
    """
    device = torch.device(
        cfg.device if (cfg.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    )
    print(f"[Train] device = {device}")

    # ── Load pre-harvested activations ────────────────────────────────────
    acts_path = Path(cfg.activations_path)
    if not acts_path.exists():
        raise FileNotFoundError(
            f"Activations file not found: {acts_path.resolve()}\n"
            f"Run src/scripts/harvest_activations.py first."
        )

    print(f"[Data] Loading {acts_path} ...")
    raw = load_file(str(acts_path))
    acts: torch.Tensor = raw["activations"].to(torch.float32)
    print(f"[Data] Shape: {tuple(acts.shape)}  dtype={acts.dtype}")

    if acts.shape[1] != cfg.d_model:
        raise ValueError(
            f"Activation d_model={acts.shape[1]} does not match cfg.d_model={cfg.d_model}. "
            f"Pass --d_model {acts.shape[1]} or re-harvest with the correct hook."
        )

    dataset = TensorDataset(acts)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
        num_workers=0,
    )

    def _infinite():
        while True:
            yield from loader

    # ── Model ─────────────────────────────────────────────────────────────
    model = BatchTopKSAE(
        d_model=cfg.d_model,
        d_sae=cfg.d_sae,
        k=cfg.k,
        n_batches_to_dead=cfg.n_batches_to_dead,
        top_k_aux=cfg.top_k_aux,
        aux_penalty=cfg.aux_penalty,
        l1_coeff=cfg.l1_coeff,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] BatchTopKSAE  d_model={cfg.d_model}  d_sae={cfg.d_sae}  k={cfg.k}")
    print(f"[Model] Parameters: {n_params:,}")

    # ── Optimiser (Adam with configurable betas, matching reference) ──────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
    )

    # ── Output dir ────────────────────────────────────────────────────────
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # ── Wandb ─────────────────────────────────────────────────────────────
    use_wandb = _WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            config={
                "d_model": cfg.d_model,
                "d_sae": cfg.d_sae,
                "expansion_factor": cfg.expansion_factor,
                "k": cfg.k,
                "n_batches_to_dead": cfg.n_batches_to_dead,
                "l1_coeff": cfg.l1_coeff,
                "aux_penalty": cfg.aux_penalty,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "beta1": cfg.beta1,
                "beta2": cfg.beta2,
                "max_grad_norm": cfg.max_grad_norm,
                "steps": cfg.steps,
                "activations_path": str(acts_path),
                "n_activations": acts.shape[0],
            },
        )
    else:
        print("[Wandb] wandb not available — console logging only")

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"\n[Train] Starting — {cfg.steps:,} steps  "
          f"(checkpoint every {cfg.checkpoint_every:,})\n")

    model.train()
    data_iter = _infinite()

    for step in range(1, cfg.steps + 1):
        (x,) = next(data_iter)
        x = x.to(device)

        # -- Forward --
        output = model(x)
        loss = output["loss"]

        # -- Backward --
        loss.backward()

        # -- Gradient clipping (before decoder constraint) --
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

        # -- Project decoder gradients + normalize weights --
        # Must come AFTER backward and BEFORE optimizer.step()
        model.make_decoder_weights_and_grad_unit_norm()

        # -- Parameter update --
        optimizer.step()
        optimizer.zero_grad()

        # ── Metrics ───────────────────────────────────────────────────────
        with torch.no_grad():
            ev = explained_variance(x, output["sae_out"])
            dead_frac = (
                output["num_dead_features"].float() / cfg.d_sae
            ).item()

        metrics = {
            "loss/total": output["loss"].item(),
            "loss/reconstruction": output["l2_loss"].item(),
            "loss/l1": output["l1_loss"].item(),
            "loss/auxiliary": output["aux_loss"].item(),
            "sparsity/l0": output["l0_norm"].item(),
            "sparsity/l1_norm": output["l1_norm"].item(),
            "sparsity/dead_latents": dead_frac,
            "sparsity/explained_variance": ev,
        }

        if use_wandb:
            wandb.log(metrics, step=step)

        if step % 100 == 0 or step == 1:
            print(
                f"step {step:>6} / {cfg.steps}  |  "
                f"loss={metrics['loss/total']:.4f}  "
                f"recon={metrics['loss/reconstruction']:.4f}  "
                f"L0={metrics['sparsity/l0']:.1f}/{cfg.k}  "
                f"dead={metrics['sparsity/dead_latents']:.3f}  "
                f"EV={metrics['sparsity/explained_variance']:.3f}"
            )

        # ── Checkpoint ────────────────────────────────────────────────────
        if step % cfg.checkpoint_every == 0:
            ckpt = save_checkpoint(model, cfg, step)
            print(f"[Checkpoint] step {step} → {ckpt}")

    # Final checkpoint
    ckpt = save_checkpoint(model, cfg, cfg.steps)
    print(f"\n[Done] Final checkpoint → {ckpt}")

    if use_wandb:
        wandb.finish()

    return model.cpu()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train BatchTopK SAE on pre-harvested UMedPT activations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--expansion_factor", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--n_batches_to_dead", type=int, default=None)
    parser.add_argument("--top_k_aux", type=int, default=None)
    parser.add_argument("--aux_penalty", type=float, default=None)
    parser.add_argument("--l1_coeff", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--beta1", type=float, default=None)
    parser.add_argument("--beta2", type=float, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--checkpoint_every", type=int, default=None)
    parser.add_argument("--activations_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = SAEConfig()
    for field, value in vars(args).items():
        if value is not None:
            setattr(cfg, field, value)

    print("=" * 60)
    print("BatchTopK SAE Trainer  (bartbussmann/BatchTopK reference)")
    print("=" * 60)
    print(f"  d_model          : {cfg.d_model}")
    print(f"  d_sae            : {cfg.d_sae}  (expansion × {cfg.expansion_factor})")
    print(f"  k                : {cfg.k}")
    print(f"  l1_coeff         : {cfg.l1_coeff}")
    print(f"  n_batches_to_dead: {cfg.n_batches_to_dead}")
    print(f"  aux_penalty      : {cfg.aux_penalty:.4f}")
    print(f"  lr               : {cfg.lr}  betas=({cfg.beta1},{cfg.beta2})")
    print(f"  max_grad_norm    : {cfg.max_grad_norm}")
    print(f"  batch_size       : {cfg.batch_size}")
    print(f"  steps            : {cfg.steps:,}")
    print(f"  checkpoint_every : {cfg.checkpoint_every:,}")
    print(f"  activations_path : {cfg.activations_path}")
    print(f"  output_dir       : {cfg.output_dir}")
    print(f"  wandb_project    : {cfg.wandb_project}")
    print(f"  device           : {cfg.device}")
    print()

    train(cfg)


if __name__ == "__main__":
    main()
