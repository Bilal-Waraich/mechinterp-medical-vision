"""Train a Matryoshka SAE on pre-harvested UMedPT activations.

Supports four datasets simultaneously via round-robin batch sampling:
  - SemiCOL colorectal histology
  - Brain tumor MRI
  - TCGA breast cancer histology
  - Pneumonia chest X-ray

The Matryoshka SAE learns nested representations by computing reconstruction
loss at multiple sparsity levels (k_schedule). Features are automatically
ordered by importance: those active at the smallest k are the most universal.

Usage
-----
  # Four datasets, default config
  python src/train_matryoshka.py \\
    --activation_paths path/to/semicol.safetensors path/to/brain.safetensors \\
                       path/to/breast.safetensors  path/to/pneumonia.safetensors \\
    --dataset_names semicol brain breast pneumonia

  # Custom k_schedule
  python src/train_matryoshka.py --k_schedule 4 8 16 32 --d_model 1024

  # Load from JSON config
  python src/train_matryoshka.py --config configs/matryoshka_pm2.json

Pipeline
--------
  1. harvest_activations.py  →  data/<dataset>_activations.safetensors
  2. python src/train_matryoshka.py  (this script)
  3. Checkpoints in data/matryoshka_checkpoints/step_XXXXXXX/
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import List

import torch
from safetensors.torch import load_file, save_file
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

from matryoshka_config import MatryoshkaConfig
from matryoshka_sae import MatryoshkaSAE

# ── Optional wandb ────────────────────────────────────────────────────────────

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ── LR schedule ───────────────────────────────────────────────────────────────

def make_lr_lambda(warmup_steps: int, total_steps: int, schedule: str):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        if schedule == "constant":
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return 0.1 + 0.9 * cosine
    return lr_lambda


# ── Metric helpers ────────────────────────────────────────────────────────────

def explained_variance(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    residual_var = (x - x_hat).var().item()
    input_var = x.var().item()
    if input_var < 1e-8:
        return 0.0
    return 1.0 - residual_var / input_var


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(model: MatryoshkaSAE, cfg: MatryoshkaConfig, step: int) -> Path:
    ckpt_dir = Path(cfg.output_dir) / f"step_{step:07d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    save_file(
        {k: v.contiguous().cpu() for k, v in model.state_dict().items()},
        str(ckpt_dir / "sae_weights.safetensors"),
    )

    cfg_dict = {
        "architecture": "matryoshka_topk",
        "d_model": cfg.d_model,
        "d_sae": cfg.d_sae,
        "k_schedule": cfg.k_schedule,
        "k_max": cfg.k_max,
        "scale_weights": model.scale_weights,
        "expansion_factor": cfg.expansion_factor,
        "n_batches_to_dead": cfg.n_batches_to_dead,
        "aux_penalty": cfg.aux_penalty,
        "l1_coeff": cfg.l1_coeff,
        "lr": cfg.lr,
        "beta1": cfg.beta1,
        "beta2": cfg.beta2,
        "max_grad_norm": cfg.max_grad_norm,
        "batch_size": cfg.batch_size,
        "steps": cfg.steps,
        "activation_paths": cfg.activation_paths,
        "dataset_names": cfg.dataset_names,
        "lr_warmup_steps": cfg.lr_warmup_steps,
        "lr_schedule": cfg.lr_schedule,
        "step": step,
    }
    (ckpt_dir / "cfg.json").write_text(json.dumps(cfg_dict, indent=2))
    return ckpt_dir


# ── Multi-dataset loader ───────────────────────────────────────────────────────

def load_datasets(
    activation_paths: List[str],
    dataset_names: List[str],
    d_model: int,
) -> List[torch.Tensor]:
    """Load activations from each dataset, validate d_model, return list."""
    datasets = []
    for path, name in zip(activation_paths, dataset_names):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"Activations not found for '{name}': {p.resolve()}\n"
                f"Run src/scripts/harvest_activations.py first."
            )
        print(f"[Data] Loading '{name}' from {p} ...")
        raw = load_file(str(p))
        acts = raw["activations"].to(torch.float32)
        if acts.shape[1] != d_model:
            raise ValueError(
                f"Dataset '{name}': d_model={acts.shape[1]} != cfg.d_model={d_model}. "
                f"Re-harvest or adjust --d_model."
            )
        print(f"       Shape: {tuple(acts.shape)}")
        datasets.append(acts)
    return datasets


# ── Training loop ─────────────────────────────────────────────────────────────

def train(cfg: MatryoshkaConfig) -> MatryoshkaSAE:
    """Run the Matryoshka SAE training loop.

    Batches are sampled round-robin across all datasets so the model sees a
    balanced mix from every dataset at every epoch.
    """
    device = torch.device(
        cfg.device if (cfg.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    )
    print(f"[Train] device = {device}")

    # ── Load datasets ─────────────────────────────────────────────────────
    all_acts = load_datasets(cfg.activation_paths, cfg.dataset_names, cfg.d_model)

    # Build per-dataset infinite iterators
    loaders = []
    for acts in all_acts:
        ds = TensorDataset(acts)
        loader = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=(device.type == "cuda"),
            num_workers=0,
        )
        loaders.append(loader)

    def _make_infinite(loader):
        while True:
            yield from loader

    iters = [_make_infinite(l) for l in loaders]
    n_datasets = len(iters)

    # ── Model ─────────────────────────────────────────────────────────────
    model = MatryoshkaSAE(
        d_model=cfg.d_model,
        d_sae=cfg.d_sae,
        k_schedule=cfg.k_schedule,
        scale_weights=cfg.scale_weights,
        n_batches_to_dead=cfg.n_batches_to_dead,
        aux_penalty=cfg.aux_penalty,
        l1_coeff=cfg.l1_coeff,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] MatryoshkaSAE  d_model={cfg.d_model}  d_sae={cfg.d_sae}")
    print(f"[Model] k_schedule={cfg.k_schedule}  scale_weights={[f'{w:.3f}' for w in model.scale_weights]}")
    print(f"[Model] Parameters: {n_params:,}")

    # ── Init b_dec to mean across all datasets ─────────────────────────
    with torch.no_grad():
        means = []
        for acts in all_acts:
            n_sample = min(10_000, len(acts))
            means.append(acts[:n_sample].mean(dim=0))
        model.b_dec.data = torch.stack(means).mean(0).to(model.b_dec.device)
    print(f"[Init] b_dec → mean of all dataset means  (norm={model.b_dec.data.norm():.3f})")

    # ── Optimiser ─────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
    )

    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=make_lr_lambda(cfg.lr_warmup_steps, cfg.steps, cfg.lr_schedule),
    )
    print(f"[LR]   schedule={cfg.lr_schedule}  warmup={cfg.lr_warmup_steps}")

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # ── Wandb ─────────────────────────────────────────────────────────────
    use_wandb = _WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            config={
                "architecture": "matryoshka_topk",
                "d_model": cfg.d_model,
                "d_sae": cfg.d_sae,
                "k_schedule": cfg.k_schedule,
                "scale_weights": model.scale_weights,
                "expansion_factor": cfg.expansion_factor,
                "n_batches_to_dead": cfg.n_batches_to_dead,
                "l1_coeff": cfg.l1_coeff,
                "aux_penalty": cfg.aux_penalty,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "steps": cfg.steps,
                "datasets": cfg.dataset_names,
                "n_datasets": n_datasets,
            },
        )
    else:
        print("[Wandb] wandb not available — console logging only")

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"\n[Train] Starting — {cfg.steps:,} steps  "
          f"({n_datasets} datasets, round-robin)  "
          f"checkpoint every {cfg.checkpoint_every:,}\n")

    scaler = GradScaler(enabled=cfg.use_amp)
    model.train()
    t0 = datetime.now()

    for step in range(1, cfg.steps + 1):
        # Round-robin dataset selection
        dataset_idx = (step - 1) % n_datasets
        (x,) = next(iters[dataset_idx])
        x = x.to(device)

        with autocast(enabled=cfg.use_amp):
            output = model(x)
            loss = output["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        model.make_decoder_weights_and_grad_unit_norm()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        # ── Metrics ───────────────────────────────────────────────────────
        with torch.no_grad():
            ev = explained_variance(x, output["sae_out"])
            dead_frac = (output["num_dead_features"].float() / cfg.d_sae).item()

        current_lr = scheduler.get_last_lr()[0]
        ds_name = cfg.dataset_names[dataset_idx] if dataset_idx < len(cfg.dataset_names) else str(dataset_idx)

        metrics: dict = {
            "loss/total": output["loss"].item(),
            "loss/reconstruction": output["l2_loss"].item(),
            "loss/l1": output["l1_loss"].item(),
            "loss/auxiliary": output["aux_loss"].item(),
            "sparsity/l0": output["l0_norm"].item(),
            "sparsity/l1_norm": output["l1_norm"].item(),
            "sparsity/dead_latents": dead_frac,
            "sparsity/explained_variance": ev,
            "lr": current_lr,
            "dataset_idx": dataset_idx,
        }
        # Per-scale reconstruction losses
        for i, (k_i, l2_i) in enumerate(zip(cfg.k_schedule, output["l2_loss_per_scale"])):
            metrics[f"loss/recon_k{k_i}"] = l2_i

        if use_wandb:
            wandb.log(metrics, step=step)

        if step % 100 == 0 or step == 1:
            elapsed = (datetime.now() - t0).total_seconds()
            eta = elapsed / step * (cfg.steps - step) if step < cfg.steps else 0
            scale_str = " ".join(
                f"k{k_i}={l2_i:.4f}"
                for k_i, l2_i in zip(cfg.k_schedule, output["l2_loss_per_scale"])
            )
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"step {step:>6}/{cfg.steps}  ds={ds_name}  |  "
                f"loss={metrics['loss/total']:.4f}  "
                f"{scale_str}  "
                f"L0={metrics['sparsity/l0']:.1f}/{cfg.k_max}  "
                f"dead={dead_frac:.3f}  EV={ev:.3f}  "
                f"lr={current_lr:.2e}  eta={eta/3600:.1f}h"
            )

        if step % cfg.checkpoint_every == 0 or step == cfg.steps:
            ckpt = save_checkpoint(model, cfg, step)
            print(f"[Checkpoint] step {step} → {ckpt}")

    print("\n[Done] Training complete.")
    if use_wandb:
        wandb.finish()

    return model.cpu()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Matryoshka SAE on pre-harvested UMedPT activations (multi-dataset)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config. CLI args override JSON values.")
    parser.add_argument("--pm_index", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--expansion_factor", type=int, default=None)
    parser.add_argument("--k_schedule", type=int, nargs="+", default=None,
                        help="Ascending list of top-k values. E.g. --k_schedule 4 8 16 32")
    parser.add_argument("--scale_weights", type=float, nargs="+", default=None,
                        help="Per-scale loss weights (same length as k_schedule).")
    parser.add_argument("--n_batches_to_dead", type=int, default=None)
    parser.add_argument("--aux_penalty", type=float, default=None)
    parser.add_argument("--l1_coeff", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr_warmup_steps", type=int, default=None)
    parser.add_argument("--lr_schedule", type=str, default=None,
                        choices=["constant", "cosine"])
    parser.add_argument("--beta1", type=float, default=None)
    parser.add_argument("--beta2", type=float, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--checkpoint_every", type=int, default=None)
    parser.add_argument("--activation_paths", type=str, nargs="+", default=None,
                        help="One safetensors path per dataset.")
    parser.add_argument("--dataset_names", type=str, nargs="+", default=None,
                        help="Human-readable name per dataset (same order as --activation_paths).")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use_amp", action="store_true", default=None)
    args = parser.parse_args()

    cfg = MatryoshkaConfig()

    if args.config:
        with open(args.config) as f:
            for k, v in json.load(f).items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)

    for field_name, value in vars(args).items():
        if field_name != "config" and value is not None:
            setattr(cfg, field_name, value)

    # Auto-set d_model from pm_index if pm_index given without explicit d_model
    if args.pm_index is not None and args.d_model is None:
        cfg2 = MatryoshkaConfig.for_pm_index(cfg.pm_index)
        cfg.d_model = cfg2.d_model
        cfg.n_patches = cfg2.n_patches

    print("=" * 70)
    print("Matryoshka SAE Trainer")
    print("=" * 70)
    print(f"  d_model          : {cfg.d_model}")
    print(f"  d_sae            : {cfg.d_sae}  (×{cfg.expansion_factor})")
    print(f"  k_schedule       : {cfg.k_schedule}")
    print(f"  scale_weights    : {cfg.scale_weights} (None = 1/k_i normalised)")
    print(f"  l1_coeff         : {cfg.l1_coeff}")
    print(f"  aux_penalty      : {cfg.aux_penalty:.4f}")
    print(f"  lr               : {cfg.lr}  betas=({cfg.beta1},{cfg.beta2})")
    print(f"  lr_schedule      : {cfg.lr_schedule}  warmup={cfg.lr_warmup_steps}")
    print(f"  batch_size       : {cfg.batch_size}")
    print(f"  steps            : {cfg.steps:,}")
    print(f"  datasets         : {cfg.dataset_names}")
    print(f"  activation_paths : {cfg.activation_paths}")
    print(f"  output_dir       : {cfg.output_dir}")
    print(f"  device           : {cfg.device}")
    print()

    train(cfg)


if __name__ == "__main__":
    main()
