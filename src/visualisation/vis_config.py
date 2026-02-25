"""Visualisation-phase configuration (Phase 3).

Separate from SAEConfig — this dataclass holds everything needed to
load a trained SAE checkpoint plus its associated activations / images
and render visual dashboards.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass
class VisConfig:
    # ── SAE checkpoint ────────────────────────────────────────────────────
    sae_checkpoint_dir: str = ""
    """Path to a checkpoint dir, e.g. "data/checkpoints/step_050000".
    Must contain ``sae_weights.safetensors`` and ``cfg.json``."""

    # ── Activations / images ──────────────────────────────────────────────
    activations_path: str = ""
    """Path to ``.safetensors`` file used during training.
    Shape ``[N_eff, d_model]``, key ``"activations"``.
    Metadata keys: ``spatial_h``, ``spatial_w``, ``n_images``, ``d_model``."""

    images_dir: str = ""
    """Path to folder of original images (PNG/JPG).
    Used for semantic-dict crops and spatial overlays.
    Images are matched to activation rows by sort order."""

    # ── Swin spatial layout ───────────────────────────────────────────────
    # Must match the pm_index used during harvest.
    # pm_index=1 (default): d_model=512, spatial_h=14, spatial_w=14
    spatial_h: int = 14
    spatial_w: int = 14

    # ── Visualisation settings ────────────────────────────────────────────
    top_k_images: int = 9
    """Canonical examples per feature (3×3 grid)."""

    top_n_features: int = 20
    """Features to render in batch-dashboard mode."""

    top_n_edges: int = 50
    """Edges to keep after graph pruning."""

    activation_threshold: float = 0.0

    # ── Output ────────────────────────────────────────────────────────────
    output_dir: str = "data/vis_outputs"
    device: str = "cuda:0"

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    @property
    def n_patches(self) -> int:
        """Total number of spatial patches per image."""
        return self.spatial_h * self.spatial_w

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str, **kwargs) -> "VisConfig":
        """Build a VisConfig by reading ``cfg.json`` from *checkpoint_dir*.

        ``cfg.json`` is the training-time config snapshot written by the
        training loop.  We pull ``d_model`` (and anything else useful)
        from it so the user only needs to point at a checkpoint.

        Any extra *kwargs* are forwarded to the dataclass constructor and
        override values derived from the checkpoint.
        """
        cfg_path = Path(checkpoint_dir) / "cfg.json"
        with open(cfg_path, "r", encoding="utf-8") as f:
            train_cfg: dict = json.load(f)

        # Auto-fill fields from the training config
        defaults: dict = {
            "sae_checkpoint_dir": str(checkpoint_dir),
        }

        # Propagate spatial layout if present in the training config
        for key in ("spatial_h", "spatial_w", "d_model"):
            if key in train_cfg:
                defaults[key] = int(train_cfg[key])

        # User-supplied kwargs take priority
        defaults.update(kwargs)

        # Only pass keys that are actual VisConfig fields
        valid_keys = {f.name for f in fields(cls)}
        defaults = {k: v for k, v in defaults.items() if k in valid_keys}

        return cls(**defaults)


# ── Smoke test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile, os

    # 1. Basic construction
    vc = VisConfig(
        sae_checkpoint_dir="data/checkpoints/step_050000",
        activations_path="data/acts.safetensors",
        images_dir="data/images",
    )
    assert vc.n_patches == 14 * 14
    assert vc.spatial_h == 14
    print(f"[OK] VisConfig basic: n_patches={vc.n_patches}")

    # 2. from_checkpoint with a temp cfg.json
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = {"d_model": 512, "expansion_factor": 4, "k": 32, "spatial_h": 7, "spatial_w": 7}
        with open(os.path.join(tmpdir, "cfg.json"), "w") as f:
            json.dump(cfg, f)

        vc2 = VisConfig.from_checkpoint(
            tmpdir,
            activations_path="acts.safetensors",
            images_dir="imgs",
        )
        assert vc2.spatial_h == 7
        assert vc2.spatial_w == 7
        assert vc2.n_patches == 49
        print(f"[OK] from_checkpoint: spatial_h={vc2.spatial_h}, n_patches={vc2.n_patches}")

    # 3. kwargs override
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = {"d_model": 512}
        with open(os.path.join(tmpdir, "cfg.json"), "w") as f:
            json.dump(cfg, f)

        vc3 = VisConfig.from_checkpoint(tmpdir, spatial_h=28, spatial_w=28)
        assert vc3.n_patches == 28 * 28
        print(f"[OK] kwargs override: n_patches={vc3.n_patches}")

    print("\nAll VisConfig smoke tests passed ✓")
