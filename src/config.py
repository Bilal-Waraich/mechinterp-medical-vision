"""SAE training configuration."""

from dataclasses import dataclass


@dataclass
class SAEConfig:
    # ── Model dimensions ──────────────────────────────────────────────────
    d_model: int = 384
    expansion_factor: int = 4

    # ── Sparsity ──────────────────────────────────────────────────────────
    k: int = 32                     # target active features per token

    # ── Dead feature handling (auxiliary loss) ────────────────────────────
    n_batches_to_dead: int = 10     # consecutive inactive batches → dead
    top_k_aux: int = 512            # dead features used in aux reconstruction
    aux_penalty: float = 1 / 32    # weight of the auxiliary loss term
    l1_coeff: float = 2e-4         # L1 sparsity penalty coefficient

    # ── Optimiser ─────────────────────────────────────────────────────────
    lr: float = 2e-4
    beta1: float = 0.9
    beta2: float = 0.999
    max_grad_norm: float = 1.0     # gradient clipping threshold

    # ── Training ──────────────────────────────────────────────────────────
    batch_size: int = 2048
    steps: int = 50_000
    checkpoint_every: int = 5_000

    # ── Paths ─────────────────────────────────────────────────────────────
    activations_path: str = "data/dummy_activations.safetensors"
    output_dir: str = "data/checkpoints"

    # ── Logging ───────────────────────────────────────────────────────────
    wandb_project: str = "sae-umedpt"

    # ── Device ────────────────────────────────────────────────────────────
    device: str = "cuda:0"

    @property
    def d_sae(self) -> int:
        """Dictionary size = d_model * expansion_factor."""
        return self.d_model * self.expansion_factor
