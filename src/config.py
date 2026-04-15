"""SAE training configuration.

# pm_index=0: d_model=256,   spatial=28x28, n_patches=784
# pm_index=1: d_model=512,   spatial=14x14, n_patches=196
# pm_index=2: d_model=1024,  spatial=7x7,   n_patches=49   <- default (universality experiments)
# pm_index=3: d_model=1024,  spatial=7x7,   n_patches=49
"""

from dataclasses import dataclass


@dataclass
class SAEConfig:
    pm_index: int = 2
    n_patches: int = 49

    # ── Model dimensions ──────────────────────────────────────────────────
    d_model: int = 1024
    expansion_factor: int = 4

    # ── Sparsity ──────────────────────────────────────────────────────────
    k: int = 32                     # target active features per token

    # ── Dead feature handling (auxiliary loss) ────────────────────────────
    n_batches_to_dead: int = 25     # consecutive inactive batches → dead
                                    # (increased from 10: rare diseases in mixed
                                    # datasets may not appear in 10 consecutive
                                    # batches, causing premature feature recycling)
    top_k_aux: int = 512            # dead features used in aux reconstruction
    aux_penalty: float = 1 / 32    # weight of the auxiliary loss term
    l1_coeff: float = 2e-4         # L1 sparsity penalty coefficient

    # ── Optimiser ─────────────────────────────────────────────────────────
    lr: float = 2e-4
    lr_warmup_steps: int = 1000     # linear warmup steps (0 to disable)
    lr_schedule: str = "cosine"     # "constant" | "cosine"
    beta1: float = 0.9
    beta2: float = 0.999
    max_grad_norm: float = 1.0     # gradient clipping threshold

    # ── Training ──────────────────────────────────────────────────────────
    batch_size: int = 2048
    steps: int = 50_000
    checkpoint_every: int = 5_000

    # ── Stratified sampling ───────────────────────────────────────────────
    stratified: bool = True         # enforce equal batch composition across datasets
                                    # requires dataset_ids tensor in activations file

    # ── Paths ─────────────────────────────────────────────────────────────
    activations_path: str = "data/dummy_activations.safetensors"
    output_dir: str = "data/checkpoints"

    # ── Logging ───────────────────────────────────────────────────────────
    wandb_project: str = "sae-umedpt"

    # ── Device ────────────────────────────────────────────────────────────
    device: str = "cuda:0"
    use_amp: bool = False       # mixed-precision (FP16) training via torch.cuda.amp

    @property
    def d_sae(self) -> int:
        """Dictionary size = d_model * expansion_factor."""
        return self.d_model * self.expansion_factor

    @classmethod
    def for_pm_index(cls, pm_index: int, **kwargs) -> "SAEConfig":
        """
        Returns SAEConfig with correct d_model and n_patches for the given pm_index.
        Values confirmed from UMedPT PyramidEncoder source in Task 1.
        """
        d_model_map  = {0: 256, 1: 512,  2: 1024, 3: 1024}
        n_patches_map = {0: 784, 1: 196,  2: 49, 3: 49}
        
        # Note on expansion_factor: if d_model >= 1024, keep expansion_factor=4 
        # giving d_sae=4096. This keeps the overcomplete ratio consistent across layers.
        
        return cls(
            pm_index=pm_index,
            d_model=d_model_map[pm_index],
            n_patches=n_patches_map[pm_index],
            **kwargs,
        )
