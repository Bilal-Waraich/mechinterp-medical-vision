"""Matryoshka Sparse Autoencoder.

A Matryoshka SAE trains nested representations by computing reconstruction loss
at multiple sparsity levels simultaneously. This forces features to be ordered
by importance: the top-k features at any scale are a subset of the top-k features
at the next larger scale, so the *first* features learned are universally the most
useful ones.

Architecture
------------
  Same encoder/decoder as BatchTopKSAE, but forward() computes:

    acts = ReLU((x - b_dec) @ W_enc + b_enc)   [B, d_sae]

  Then, for each k_i in k_schedule (ascending, e.g. [4, 8, 16, 32]):
    acts_i = BatchTopK at level k_i              [B, d_sae]
    x_hat_i = acts_i @ W_dec + b_dec            [B, d_model]
    l2_i   = MSE(x_hat_i, x)

  Total loss = sum_i (scale_weights[i] * l2_i) + l1 + aux

Matryoshka Nesting
------------------
  The key insight: since k_schedule is ascending and we use the SAME encoder,
  the sparse codes at level k_i are the global top-(k_i * B) entries. Any
  feature that fires at scale k_i also fires at scale k_j > k_i because the
  larger set is a superset. This nests the representations.

  The multi-scale loss creates gradient pressure to make the highest-magnitude
  features (those selected at ALL scales) reconstruct well, naturally ordering
  features by global importance.

Scale Weights
-------------
  By default, larger scales get smaller weights (the smallest-k scale dominates):
    weight[i] = 1 / k_schedule[i]   (normalised to sum to 1)
  This prevents the large-k terms from drowning out the fine-grained signal.
  Custom weights can be passed via cfg.scale_weights.

Reference
---------
  Matryoshka Representation Learning — Kusupati et al. (NeurIPS 2022)
  BatchTopK SAE — Bussmann et al. (2024)  github.com/bartbussmann/BatchTopK
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class MatryoshkaSAE(nn.Module):
    """Sparse Autoencoder with Matryoshka multi-scale sparsity.

    Parameters
    ----------
    d_model : int
        Input activation dimension.
    d_sae : int
        Dictionary size (number of learned features).
    k_schedule : list[int]
        Ascending list of top-k values for each Matryoshka scale.
        E.g. [4, 8, 16, 32] — four nested levels.
    scale_weights : list[float] | None
        Loss weight for each scale. If None, uses 1/k_i normalised to sum=1.
    n_batches_to_dead : int
        Consecutive batches of inactivity before a feature is declared dead.
    aux_penalty : float
        Coefficient for the auxiliary dead-feature revival loss.
    l1_coeff : float
        Coefficient for the L1 sparsity penalty (applied at the largest scale).
    """

    def __init__(
        self,
        d_model: int,
        d_sae: int,
        k_schedule: List[int],
        scale_weights: Optional[List[float]] = None,
        n_batches_to_dead: int = 10,
        aux_penalty: float = 1 / 32,
        l1_coeff: float = 2e-4,
    ):
        super().__init__()
        assert len(k_schedule) >= 1, "k_schedule must have at least one entry."
        assert k_schedule == sorted(k_schedule), "k_schedule must be ascending."
        assert all(k > 0 for k in k_schedule), "All k values must be positive."
        assert k_schedule[-1] <= d_sae, "Largest k cannot exceed d_sae."

        self.d_model = d_model
        self.d_sae = d_sae
        self.k_schedule = k_schedule
        self.k = k_schedule[-1]  # maximum k (mirrors BatchTopKSAE.k for compat)
        self.n_batches_to_dead = n_batches_to_dead
        self.aux_penalty = aux_penalty
        self.l1_coeff = l1_coeff

        # Normalised scale weights: smaller k → larger weight by default
        if scale_weights is not None:
            assert len(scale_weights) == len(k_schedule)
            total = sum(scale_weights)
            self.scale_weights = [w / total for w in scale_weights]
        else:
            raw = [1.0 / k for k in k_schedule]
            total = sum(raw)
            self.scale_weights = [w / total for w in raw]

        # ── Encoder parameters ────────────────────────────────────────────
        self.W_enc = nn.Parameter(torch.empty(d_model, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        # ── Decoder parameters ────────────────────────────────────────────
        # Shape (d_sae, d_model): rows constrained to unit norm externally.
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))

        # ── Dead feature counter ──────────────────────────────────────────
        self.register_buffer("num_batches_not_active", torch.zeros(d_sae))

        self._init_weights()

    # ── Initialisation ─────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.W_enc)
        with torch.no_grad():
            self.W_dec.data = self.W_enc.data.T.clone()
            self.W_dec.data /= (
                self.W_dec.data.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            )
            self.W_dec.data *= 0.1
            self.W_enc.data = self.W_dec.data.T.clone().contiguous()
        nn.init.zeros_(self.b_enc)

    # ── Checkpoint ─────────────────────────────────────────────────────────

    @classmethod
    def load_from_checkpoint(cls, path: str) -> "MatryoshkaSAE":
        """Load from a checkpoint directory (sae_weights.safetensors + cfg.json) or a .pt file."""
        from pathlib import Path
        p = Path(path)
        if p.is_dir():
            import json
            from safetensors.torch import load_file
            cfg = json.loads((p / "cfg.json").read_text())
            state_dict = load_file(str(p / "sae_weights.safetensors"), device="cpu")
            k_schedule = cfg.get("k_schedule", [cfg.get("k", 32)])
            d_model, d_sae = cfg["d_model"], cfg["d_sae"]
        else:
            state_dict = torch.load(path, map_location="cpu")
            k_schedule = state_dict.get("k_schedule", [32])
            if isinstance(k_schedule, torch.Tensor):
                k_schedule = k_schedule.tolist()
            d_model, d_sae = state_dict["W_enc"].shape[0], state_dict["W_enc"].shape[1]
        sae = cls(d_model=d_model, d_sae=d_sae, k_schedule=k_schedule)
        sae.load_state_dict({k: v for k, v in state_dict.items()
                             if k not in ("k_schedule",)}, strict=False)
        sae.eval()
        return sae

    # ── Decoder constraint ─────────────────────────────────────────────────

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self) -> None:
        """Project W_dec gradients tangent to unit sphere, then renormalise.

        Call AFTER loss.backward() and BEFORE optimizer.step().
        """
        if self.W_dec.grad is None:
            return
        W_dec_normed = F.normalize(self.W_dec.data, dim=-1)
        W_dec_grad_proj = (
            (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        )
        self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed

    # ── Dead feature tracking ──────────────────────────────────────────────

    @torch.no_grad()
    def update_inactive_features(self, acts: torch.Tensor) -> None:
        """Track consecutively inactive features using the largest-scale acts."""
        self.num_batches_not_active += (acts.sum(0) == 0).float()
        self.num_batches_not_active[acts.sum(0) > 0] = 0

    # ── Auxiliary loss ─────────────────────────────────────────────────────

    def get_auxiliary_loss(
        self,
        x: torch.Tensor,
        x_reconstruct: torch.Tensor,
        acts: torch.Tensor,
    ) -> torch.Tensor:
        """Revive dead features by reconstructing the residual with them."""
        dead = self.num_batches_not_active >= self.n_batches_to_dead
        num_dead = int(dead.sum())
        if num_dead == 0:
            return torch.tensor(0.0, dtype=x.dtype, device=x.device)

        residual = (x.float() - x_reconstruct.float()).detach()
        k_aux = self.d_model // 2
        scale = min(num_dead / max(k_aux, 1), 1.0)
        k_aux = min(k_aux, num_dead)

        acts_dead = acts[:, dead]
        acts_topk_aux = torch.topk(acts_dead, k_aux, dim=-1)
        acts_aux = torch.zeros_like(acts_dead).scatter(
            -1, acts_topk_aux.indices, acts_topk_aux.values
        )
        x_reconstruct_aux = acts_aux @ self.W_dec[dead]
        aux_loss = (x_reconstruct_aux.float() - residual).pow(2).mean()
        return self.aux_penalty * scale * aux_loss

    # ── Encode / Decode helpers ────────────────────────────────────────────

    def encode(self, x: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
        """Return sparse codes using per-sample top-k.

        Parameters
        ----------
        x : Tensor [B, d_model]
        k : int | None
            Number of active features. Defaults to the largest scale k.
        """
        if k is None:
            k = self.k
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        topk_vals = torch.topk(acts, k, dim=-1)
        return torch.zeros_like(acts).scatter(-1, topk_vals.indices, topk_vals.values)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct input from sparse latent codes."""
        return z @ self.W_dec + self.b_dec

    # ── BatchTopK at a given level ─────────────────────────────────────────

    def _batch_topk(self, acts: torch.Tensor, k: int) -> torch.Tensor:
        """Apply BatchTopK (global top k*B selection) at sparsity level k."""
        B = acts.shape[0]
        n_keep = min(k * B, acts.numel())
        result = torch.topk(acts.flatten(), n_keep, dim=-1)
        return (
            torch.zeros_like(acts.flatten())
            .scatter(-1, result.indices, result.values)
            .reshape(acts.shape)
        )

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> dict:
        """Matryoshka forward pass — computes reconstruction at all scales.

        Parameters
        ----------
        x : Tensor [B, d_model]

        Returns
        -------
        dict with keys:
            sae_out          — reconstruction at the largest scale [B, d_model]
            feature_acts     — sparse codes at the largest scale [B, d_sae]
            num_dead_features — count of dead features (int tensor)
            loss             — total weighted loss
            l2_loss          — MSE at largest scale
            l2_loss_per_scale — list of MSE floats, one per scale
            l1_loss          — L1 penalty (largest scale)
            l0_norm          — mean active features per token (largest scale)
            l1_norm          — mean L1 norm (largest scale)
            aux_loss         — auxiliary dead-feature revival loss
        """
        # 1. Encode (shared across all scales)
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)  # [B, d_sae]

        # 2. Multi-scale Matryoshka reconstruction loss
        total_l2 = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        l2_per_scale: list[float] = []
        acts_topk_max = None

        for k_i, w_i in zip(self.k_schedule, self.scale_weights):
            acts_i = self._batch_topk(acts, k_i)
            x_hat_i = acts_i @ self.W_dec + self.b_dec
            l2_i = (x_hat_i.float() - x.float()).pow(2).mean()
            total_l2 = total_l2 + w_i * l2_i
            l2_per_scale.append(l2_i.item())
            if k_i == self.k_schedule[-1]:
                acts_topk_max = acts_i
                x_hat_max = x_hat_i

        # 3. Dead feature tracking (at the largest scale)
        if self.training:
            self.update_inactive_features(acts_topk_max)

        # 4. L1 penalty (largest scale — already the most expressive)
        l1_norm = acts_topk_max.float().abs().sum(-1).mean()
        l1_loss = self.l1_coeff * l1_norm
        l0_norm = (acts_topk_max > 0).float().sum(-1).mean()

        # 5. Auxiliary loss (dead feature revival using largest-scale acts)
        aux_loss = self.get_auxiliary_loss(x, x_hat_max, acts)

        loss = total_l2 + l1_loss + aux_loss

        num_dead = (self.num_batches_not_active >= self.n_batches_to_dead).sum()

        return {
            "sae_out": x_hat_max,
            "feature_acts": acts_topk_max,
            "num_dead_features": num_dead,
            "loss": loss,
            "l2_loss": total_l2,          # weighted multi-scale L2
            "l2_loss_per_scale": l2_per_scale,
            "l1_loss": l1_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss,
        }
