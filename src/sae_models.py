"""BatchTopK Sparse Autoencoder for UMedPT activations.

Architecture
------------
  Encoder : acts = F.relu((x - b_dec) @ W_enc)
  Sparsity: BatchTopK — select top k*B values globally across the flattened batch
  Decoder : x_hat = acts_topk @ W_dec + b_dec

Losses
------
  loss = l2_loss + l1_loss + aux_loss
    l2_loss  : MSE reconstruction loss
    l1_loss  : L1 sparsity penalty (l1_coeff * mean absolute activations)
    aux_loss : Dead-feature revival — reconstructs residual using dead features

Dead Feature Tracking
---------------------
  num_batches_not_active[i] counts consecutive batches where feature i was
  inactive.  Resets to 0 when the feature fires.  Features with count >=
  n_batches_to_dead are considered dead and targeted by the auxiliary loss.

Decoder Constraint
------------------
  Rows of W_dec are kept at unit norm.  During training, call
  make_decoder_weights_and_grad_unit_norm() AFTER loss.backward() and BEFORE
  optimizer.step() — this projects gradients to be tangent to the unit sphere,
  then renormalizes the weights.

Reference: https://github.com/bartbussmann/BatchTopK
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchTopKSAE(nn.Module):
    """Sparse Autoencoder with BatchTopK sparsity and unit-norm decoder.

    Faithfully implements the architecture from bartbussmann/BatchTopK.

    Parameters
    ----------
    d_model : int
        Input activation dimension.
    d_sae : int
        Dictionary size (number of learned features).
    k : int
        Target number of active features per token.
    n_batches_to_dead : int
        Consecutive batches of inactivity before a feature is dead.
    top_k_aux : int
        Number of dead features used in the auxiliary reconstruction.
    aux_penalty : float
        Coefficient for the auxiliary dead-feature revival loss.
    l1_coeff : float
        Coefficient for the L1 sparsity penalty.
    """

    def __init__(
        self,
        d_model: int,
        d_sae: int,
        k: int,
        n_batches_to_dead: int = 10,
        top_k_aux: int = 512,
        aux_penalty: float = 1 / 32,
        l1_coeff: float = 2e-4,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k
        self.n_batches_to_dead = n_batches_to_dead
        self.top_k_aux = top_k_aux
        self.aux_penalty = aux_penalty
        self.l1_coeff = l1_coeff

        # ── Encoder parameters ────────────────────────────────────────────
        self.W_enc = nn.Parameter(torch.empty(d_model, d_sae))

        # b_dec is subtracted before encoding and added after decoding.
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        # ── Decoder parameters ────────────────────────────────────────────
        # Shape (d_sae, d_model): each ROW is one dictionary atom.
        # Rows are constrained to unit norm; enforced externally via
        # make_decoder_weights_and_grad_unit_norm().
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))

        # ── Dead feature counter ──────────────────────────────────────────
        # Counts consecutive batches where each feature was inactive.
        # Stored as a buffer so it moves with the model's device.
        self.register_buffer("num_batches_not_active", torch.zeros(d_sae))

        self._init_weights()

    # ── Initialisation ─────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.W_enc)
        # Initialise W_dec as the transpose of W_enc (as in reference), then normalize
        with torch.no_grad():
            self.W_dec.data = self.W_enc.data.T.clone()
            self.W_dec.data = F.normalize(self.W_dec.data, dim=-1)

    # ── Decoder constraint ─────────────────────────────────────────────────

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self) -> None:
        """Project W_dec gradients tangent to the unit-sphere, then normalize weights.

        Call AFTER loss.backward() and BEFORE optimizer.step().

        Steps:
          1. Compute the component of the gradient along the weight direction.
          2. Subtract it so the optimizer only moves the decoder perpendicular
             to the current weight vector (i.e., along the sphere surface).
          3. Renormalize the weight data to unit norm.
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
        """Increment inactivity counter for features that did not fire.

        Parameters
        ----------
        acts : Tensor [B, d_sae]
            Sparse activations for the current batch.
        """
        self.num_batches_not_active += (acts.sum(0) == 0).float()
        self.num_batches_not_active[acts.sum(0) > 0] = 0

    # ── Auxiliary loss ─────────────────────────────────────────────────────

    def get_auxiliary_loss(
        self,
        x: torch.Tensor,
        x_reconstruct: torch.Tensor,
        acts: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct the residual using dead features to revive them.

        Takes the top top_k_aux dead features and trains them to explain
        the portion of the input that the main features failed to capture.
        """
        dead = self.num_batches_not_active >= self.n_batches_to_dead
        if dead.sum() == 0:
            return torch.tensor(0.0, dtype=x.dtype, device=x.device)

        residual = x.float() - x_reconstruct.float()
        n_aux = min(int(self.top_k_aux), int(dead.sum()))

        acts_dead = acts[:, dead]  # [B, n_dead]
        acts_topk_aux = torch.topk(acts_dead, n_aux, dim=-1)
        acts_aux = torch.zeros_like(acts_dead).scatter(
            -1, acts_topk_aux.indices, acts_topk_aux.values
        )
        x_reconstruct_aux = acts_aux @ self.W_dec[dead]
        return self.aux_penalty * (x_reconstruct_aux.float() - residual).pow(2).mean()

    # ── Encode / Decode helpers ────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return sparse latent codes using per-sample top-k.

        Uses per-sample (not batch-level) top-k so it works for any batch
        size including single samples at inference time.
        """
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)
        topk_vals = torch.topk(acts, self.k, dim=-1)
        return torch.zeros_like(acts).scatter(-1, topk_vals.indices, topk_vals.values)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct input from sparse latent codes."""
        return z @ self.W_dec + self.b_dec

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> dict:
        """Full encoder → ReLU → BatchTopK → decoder pass.

        Parameters
        ----------
        x : Tensor [B, d_model]

        Returns
        -------
        dict
            sae_out         — reconstruction [B, d_model]
            feature_acts    — sparse codes [B, d_sae]
            num_dead_features — count of dead features (scalar int)
            loss            — total loss (l2 + l1 + aux)
            l2_loss         — MSE reconstruction loss
            l1_loss         — L1 sparsity penalty
            l0_norm         — mean active features per token
            l1_norm         — mean L1 norm of activations
            aux_loss        — auxiliary dead-feature revival loss
        """
        # 1. Encode with ReLU (following reference: acts = relu(x_cent @ W_enc))
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)  # [B, d_sae]

        # 2. BatchTopK: global top-(k*B) selection across the flattened batch
        B = x.shape[0]
        n_keep = min(self.k * B, acts.numel())
        acts_topk_result = torch.topk(acts.flatten(), n_keep, dim=-1)
        acts_topk = (
            torch.zeros_like(acts.flatten())
            .scatter(-1, acts_topk_result.indices, acts_topk_result.values)
            .reshape(acts.shape)
        )  # [B, d_sae]

        # 3. Decode
        x_reconstruct = acts_topk @ self.W_dec + self.b_dec  # [B, d_model]

        # 4. Update dead feature tracking (training only)
        if self.training:
            self.update_inactive_features(acts_topk)

        # 5. Losses
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts_topk.float().abs().sum(-1).mean()
        l1_loss = self.l1_coeff * l1_norm
        l0_norm = (acts_topk > 0).float().sum(-1).mean()
        aux_loss = self.get_auxiliary_loss(x, x_reconstruct, acts)
        loss = l2_loss + l1_loss + aux_loss

        num_dead = (self.num_batches_not_active >= self.n_batches_to_dead).sum()

        return {
            "sae_out": x_reconstruct,
            "feature_acts": acts_topk,
            "num_dead_features": num_dead,
            "loss": loss,
            "l2_loss": l2_loss,
            "l1_loss": l1_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss,
        }
