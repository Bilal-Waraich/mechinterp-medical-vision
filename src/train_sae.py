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
  1. src/conglomerate/harvest_all.py  →  data/xray_activations.safetensors
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
import math
from datetime import datetime
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset, Dataset, Sampler

from config import SAEConfig
from sae_models import BatchTopKSAE


# ── Stratified sampling ───────────────────────────────────────────────────────

class StratifiedActivationDataset(Dataset):
    """Wraps a flat activation tensor with dataset_ids for stratified sampling.

    Parameters
    ----------
    activations  : [N, d_model] float32
    dataset_ids  : [N] int32 — integer dataset index per token
    """

    def __init__(self, activations: torch.Tensor, dataset_ids: torch.Tensor) -> None:
        self.activations = activations
        self.dataset_ids = dataset_ids

    def __len__(self) -> int:
        return len(self.activations)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.activations[idx], self.dataset_ids[idx]


class StratifiedBatchSampler(Sampler):
    """Yields batches with equal representation from each dataset.

    Each batch contains ``batch_size // n_datasets`` tokens from every dataset.
    TB (dataset_id=3) is sampled with replacement so it can keep pace with
    larger datasets. All others are sampled without replacement per epoch.

    Parameters
    ----------
    dataset_ids   : [N] int32 tensor
    batch_size    : total tokens per batch (must be divisible by n_datasets)
    n_steps       : number of batches to yield (training steps)
    tb_dataset_id : dataset index that is oversampled (default 3)
    seed          : random seed
    """

    def __init__(
        self,
        dataset_ids: torch.Tensor,
        batch_size: int,
        n_steps: int,
        tb_dataset_id: int = 3,
        seed: int = 0,
    ) -> None:
        unique_ids = dataset_ids.unique().tolist()
        n_datasets = len(unique_ids)
        assert batch_size % n_datasets == 0, (
            f"batch_size={batch_size} must be divisible by n_datasets={n_datasets}"
        )
        self.per_ds = batch_size // n_datasets
        self.n_steps = n_steps
        self.tb_id = tb_dataset_id
        self.seed = seed

        # Build index lists per dataset
        self.indices_per_ds: dict[int, list[int]] = {}
        for uid in unique_ids:
            self.indices_per_ds[int(uid)] = (dataset_ids == uid).nonzero(as_tuple=True)[0].tolist()

    def __len__(self) -> int:
        return self.n_steps

    def __iter__(self):
        import random as _random
        rng = _random.Random(self.seed)

        # Per-epoch shuffled pools for non-TB datasets
        pools: dict[int, list[int]] = {}
        for uid, idxs in self.indices_per_ds.items():
            pools[uid] = list(idxs)
            rng.shuffle(pools[uid])
        pointers: dict[int, int] = {uid: 0 for uid in pools}

        for _ in range(self.n_steps):
            batch: list[int] = []
            for uid, idxs in self.indices_per_ds.items():
                if uid == self.tb_id:
                    # TB: sample with replacement
                    batch.extend(rng.choices(idxs, k=self.per_ds))
                else:
                    ptr = pointers[uid]
                    pool = pools[uid]
                    # Refill pool when exhausted (new epoch for this dataset)
                    if ptr + self.per_ds > len(pool):
                        rng.shuffle(pool)
                        ptr = 0
                    batch.extend(pool[ptr : ptr + self.per_ds])
                    pointers[uid] = ptr + self.per_ds
            rng.shuffle(batch)
            yield batch

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
            return float(step) / max(1, warmup_steps)  # linear 0 → 1
        if schedule == "constant":
            return 1.0
        # Cosine decay: 1 → 0.1 over remaining steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return 0.1 + 0.9 * cosine
    return lr_lambda


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
        "lr_warmup_steps": cfg.lr_warmup_steps,
        "lr_schedule": cfg.lr_schedule,
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
            f"Run src/conglomerate/harvest_all.py first."
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

    # ── Stratified or plain DataLoader ────────────────────────────────────
    dataset_ids: torch.Tensor | None = raw.get("dataset_ids", None)

    if cfg.stratified and dataset_ids is not None:
        n_datasets = int(dataset_ids.max().item()) + 1
        print(f"[Data] Stratified sampling: {n_datasets} datasets, "
              f"{cfg.batch_size // n_datasets} tokens/dataset/batch")
        train_dataset = StratifiedActivationDataset(acts, dataset_ids)
        sampler = StratifiedBatchSampler(
            dataset_ids=dataset_ids,
            batch_size=cfg.batch_size,
            n_steps=cfg.steps,
            tb_dataset_id=3,
            seed=0,
        )
        loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            pin_memory=(device.type == "cuda"),
            num_workers=0,
        )
        _stratified = True
    else:
        if cfg.stratified and dataset_ids is None:
            print("[Data] WARNING: stratified=True but no dataset_ids in file. "
                  "Falling back to shuffled DataLoader.")
        plain_dataset = TensorDataset(acts)
        loader = DataLoader(
            plain_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=(device.type == "cuda"),
            num_workers=0,
        )
        dataset_ids = None
        _stratified = False

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

    # ── Initialise b_dec ──────────────────────────────────────────────────
    # If activations were per-dataset centred (harvest_all.py default), the
    # global mean is near zero and we initialise b_dec to zero accordingly.
    # If not centred, fall back to global data mean.
    with torch.no_grad():
        n_sample = min(10_000, len(acts))
        sample = acts[:n_sample].to(model.b_dec.device)
        global_mean = sample.mean(dim=0)
        global_mean_norm = global_mean.norm().item()
        if global_mean_norm < 0.1 * (cfg.d_model ** 0.5):
            # Data already centred — initialise to zero
            model.b_dec.data.zero_()
            print(f"[Init] b_dec initialised to zero (data is centred, "
                  f"global_mean_norm={global_mean_norm:.3f})")
        else:
            model.b_dec.data = global_mean
            print(f"[Init] b_dec initialised to data mean "
                  f"(norm={global_mean_norm:.3f})")

    # ── Optimiser (Adam with configurable betas, matching reference) ──────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
    )

    # ── LR scheduler: warmup + optional cosine decay (CHANGE 4) ──────────
    from torch.optim.lr_scheduler import LambdaLR  # noqa: PLC0415

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=make_lr_lambda(cfg.lr_warmup_steps, cfg.steps, cfg.lr_schedule),
    )
    print(f"[LR]   schedule={cfg.lr_schedule}  warmup={cfg.lr_warmup_steps} steps")

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
    amp_label = "AMP/FP16" if cfg.use_amp else "FP32"
    print(f"\n[Train] Starting — {cfg.steps:,} steps  "
          f"(checkpoint every {cfg.checkpoint_every:,})  precision={amp_label}\n")

    scaler = GradScaler(enabled=cfg.use_amp)
    model.train()
    data_iter = _infinite()
    t0_train = datetime.now()

    for step in range(1, cfg.steps + 1):
        batch = next(data_iter)
        if _stratified:
            x, _batch_ds_ids = batch
            _batch_ds_ids = _batch_ds_ids.to(device)
        else:
            (x,) = batch
            _batch_ds_ids = None
        x = x.to(device)

        # -- Forward (FP16 inside autocast when use_amp=True) --
        with autocast(enabled=cfg.use_amp):
            output = model(x)
            loss = output["loss"]

        # -- Backward --
        scaler.scale(loss).backward()

        # -- Unscale before grad ops so clip + unit-norm work in FP32 --
        scaler.unscale_(optimizer)

        # -- Gradient clipping (before decoder constraint) --
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

        # -- Project decoder gradients + normalize weights --
        # Must come AFTER backward and BEFORE optimizer.step()
        model.make_decoder_weights_and_grad_unit_norm()

        # -- Parameter update --
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        # ── Metrics ───────────────────────────────────────────────────────
        with torch.no_grad():
            ev = explained_variance(x, output["sae_out"])
            dead_frac = (
                output["num_dead_features"].float() / cfg.d_sae
            ).item()

        current_lr = scheduler.get_last_lr()[0]

        metrics = {
            "loss/total": output["loss"].item(),
            "loss/reconstruction": output["l2_loss"].item(),
            "loss/l1": output["l1_loss"].item(),
            "loss/auxiliary": output["aux_loss"].item(),
            "sparsity/l0": output["l0_norm"].item(),
            "sparsity/l1_norm": output["l1_norm"].item(),
            "sparsity/dead_latents": dead_frac,
            "sparsity/explained_variance": ev,
            "lr": current_lr,
        }

        if use_wandb:
            wandb.log(metrics, step=step)

        # ── Per-dataset batch fraction (verify stratification) ────────────
        if _stratified and _batch_ds_ids is not None and step % 500 == 0:
            total = _batch_ds_ids.shape[0]
            fracs = {
                int(uid): (_batch_ds_ids == uid).sum().item() / total
                for uid in _batch_ds_ids.unique().tolist()
            }
            ds_names = {0: "chexpert", 1: "rsna", 2: "siim", 3: "tb"}
            frac_str = "  ".join(
                f"{ds_names.get(k, str(k))}={v:.2f}" for k, v in sorted(fracs.items())
            )
            print(f"[Stratification] step {step}  batch fracs: {frac_str}")
            if use_wandb:
                wandb.log(
                    {f"stratification/{ds_names.get(k, str(k))}": v
                     for k, v in fracs.items()},
                    step=step,
                )

        if step % 100 == 0 or step == 1:
            elapsed = (datetime.now() - t0_train).total_seconds()
            eta = elapsed / step * (cfg.steps - step) if step < cfg.steps else 0
            ts = datetime.now().strftime("%H:%M:%S")
            print(
                f"[{ts}] step {step:>6} / {cfg.steps}  |  "
                f"loss={metrics['loss/total']:.4f}  "
                f"recon={metrics['loss/reconstruction']:.4f}  "
                f"L0={metrics['sparsity/l0']:.1f}/{cfg.k}  "
                f"dead={metrics['sparsity/dead_latents']:.3f}  "
                f"EV={metrics['sparsity/explained_variance']:.3f}  "
                f"lr={current_lr:.2e}  "
                f"eta={eta/3600:.1f}h"
            )

        # ── Checkpoint ────────────────────────────────────────────────────
        if step % cfg.checkpoint_every == 0 or step == cfg.steps:
            ckpt = save_checkpoint(model, cfg, step)
            print(f"[Checkpoint] step {step} → {ckpt}")

    print(f"\n[Done] Training complete.")

    if use_wandb:
        wandb.finish()

    return model.cpu()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train BatchTopK SAE on pre-harvested UMedPT activations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config file. Overrides defaults.")
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--expansion_factor", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--n_batches_to_dead", type=int, default=None)
    parser.add_argument("--top_k_aux", type=int, default=None)
    parser.add_argument("--aux_penalty", type=float, default=None)
    parser.add_argument("--l1_coeff", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr_warmup_steps", type=int, default=None)
    parser.add_argument("--lr_schedule", type=str, default=None,
                        choices=["constant", "cosine"],
                        help="LR schedule: 'constant' or 'cosine' (with warmup).")
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
    parser.add_argument("--use_amp", action="store_true", default=None,
                        help="Enable mixed-precision (FP16) training via torch.cuda.amp (~2x speedup on Tensor Core GPUs).")
    parser.add_argument("--no_stratified", action="store_true", default=False,
                        help="Disable stratified sampling (use plain shuffled DataLoader).")
    args = parser.parse_args()

    cfg = SAEConfig()
    
    if args.config:
        import json
        with open(args.config, "r") as f:
            j = json.load(f)
            for k, v in j.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
                    
    for field, value in vars(args).items():
        if field not in ("config", "no_stratified") and value is not None:
            setattr(cfg, field, value)

    if args.no_stratified:
        cfg.stratified = False

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
    print(f"  lr_schedule      : {cfg.lr_schedule}  warmup={cfg.lr_warmup_steps}")
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
