"""Shared pytest fixtures for visualisation tests.

All fixtures run on CPU with tiny dimensions to keep tests fast.
"""

import sys
from pathlib import Path

import pytest
import torch

# Ensure src/ is importable
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from sae_models import BatchTopKSAE
from visualisation.vis_config import VisConfig
from visualisation.collector import compute_feature_stats


# ── Dimensions ────────────────────────────────────────────────────────────
D_MODEL = 8
D_SAE = 32
K = 4
SPATIAL_H = 2
SPATIAL_W = 2
N_PATCHES = SPATIAL_H * SPATIAL_W  # 4
N_IMAGES = 10
N_EFF = N_IMAGES * N_PATCHES       # 40


@pytest.fixture(scope="session")
def tiny_sae() -> BatchTopKSAE:
    """Small BatchTopKSAE with random weights, eval mode, on CPU."""
    sae = BatchTopKSAE(d_model=D_MODEL, d_sae=D_SAE, k=K)
    sae.eval()
    return sae


@pytest.fixture(scope="session")
def tiny_activations() -> torch.Tensor:
    """Random activations [N_eff, d_model] on CPU."""
    torch.manual_seed(0)
    return torch.randn(N_EFF, D_MODEL)


@pytest.fixture(scope="session")
def tiny_config() -> VisConfig:
    """VisConfig matching the tiny test dimensions."""
    return VisConfig(
        spatial_h=SPATIAL_H,
        spatial_w=SPATIAL_W,
        top_k_images=3,
        top_n_edges=10,
        activation_threshold=0.0,
        device="cpu",
    )


@pytest.fixture(scope="session")
def tiny_stats(tiny_activations, tiny_sae, tiny_config) -> dict:
    """Pre-computed feature stats for the tiny setup."""
    return compute_feature_stats(
        tiny_activations, tiny_sae, tiny_config, batch_size=16,
    )
