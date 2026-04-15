"""Configuration for the Matryoshka SAE trainer.

Extends the base SAEConfig with Matryoshka-specific fields:
  - k_schedule        : list of ascending top-k values (one per nested scale)
  - scale_weights     : optional per-scale loss weights (None = 1/k_i normalised)
  - activation_paths  : per-dataset activation files (list of str)
  - dataset_names     : matching human-readable labels for logging
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MatryoshkaConfig:
    # ── Model dimensions ──────────────────────────────────────────────────
    pm_index: int = 2
    n_patches: int = 49
    d_model: int = 1024
    expansion_factor: int = 4

    # ── Matryoshka sparsity schedule ──────────────────────────────────────
    # Ascending list of top-k values; the SAE computes reconstruction loss
    # at each level. Default: four octave-spaced levels up to k=32.
    k_schedule: List[int] = field(default_factory=lambda: [4, 8, 16, 32])

    # Per-scale loss weights. None → use 1/k_i (normalised). If provided,
    # must have the same length as k_schedule.
    scale_weights: Optional[List[float]] = None

    # ── Dead feature handling ─────────────────────────────────────────────
    n_batches_to_dead: int = 10
    aux_penalty: float = 1 / 32
    l1_coeff: float = 2e-4

    # ── Optimiser ─────────────────────────────────────────────────────────
    lr: float = 2e-4
    lr_warmup_steps: int = 1000
    lr_schedule: str = "cosine"      # "constant" | "cosine"
    beta1: float = 0.9
    beta2: float = 0.999
    max_grad_norm: float = 1.0

    # ── Training ──────────────────────────────────────────────────────────
    batch_size: int = 2048
    steps: int = 50_000
    checkpoint_every: int = 5_000

    # ── Multi-dataset activation paths ────────────────────────────────────
    # Provide one path per dataset. Batches are sampled round-robin across
    # all datasets so the SAE sees a balanced mix during training.
    activation_paths: List[str] = field(default_factory=lambda: [
        "data/dummy_activations.safetensors",
    ])
    dataset_names: List[str] = field(default_factory=lambda: [
        "dataset_0",
    ])

    # ── Output / logging ──────────────────────────────────────────────────
    output_dir: str = "data/matryoshka_checkpoints"
    wandb_project: str = "matryoshka-sae-umedpt"

    # ── Device ────────────────────────────────────────────────────────────
    device: str = "cuda:0"
    use_amp: bool = False

    @property
    def d_sae(self) -> int:
        return self.d_model * self.expansion_factor

    @property
    def k_max(self) -> int:
        return self.k_schedule[-1]

    @classmethod
    def for_pm_index(cls, pm_index: int, **kwargs) -> "MatryoshkaConfig":
        d_model_map   = {0: 256, 1: 512,  2: 1024, 3: 1024}
        n_patches_map = {0: 784, 1: 196,  2: 49,   3: 49}
        return cls(
            pm_index=pm_index,
            d_model=d_model_map[pm_index],
            n_patches=n_patches_map[pm_index],
            **kwargs,
        )
