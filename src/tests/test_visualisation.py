"""Pytest tests for the visualisation package.

All tests run on CPU with the tiny fixtures defined in conftest.py.
No disk I/O or GPU required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from PIL import Image

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from tests.conftest import D_MODEL, D_SAE, K, SPATIAL_H, SPATIAL_W, N_PATCHES, N_IMAGES

from visualisation.collector import (
    get_dead_features,
    get_most_active_features,
    get_top_images_per_feature,
)
from visualisation.spatial_maps import (
    get_image_activations,
    get_spatial_activation_map,
    upsample_and_normalise,
)
from visualisation.graph_utils import (
    get_active_features,
    compute_interaction_matrix,
    compute_prompt_attribution,
    prune_graph,
    build_graph_data,
)
from visualisation.semantic_dict import get_patch_crop


# ═══════════════════════════════════════════════════════════════════════════
# collector.py
# ═══════════════════════════════════════════════════════════════════════════

class TestCollector:
    def test_stats_shapes(self, tiny_stats):
        assert tiny_stats["feature_max"].shape == (N_IMAGES, D_SAE)
        assert tiny_stats["feature_argmax"].shape == (N_IMAGES, D_SAE)
        assert tiny_stats["feature_mean"].shape == (N_IMAGES, D_SAE)
        assert tiny_stats["global_max"].shape == (D_SAE,)
        assert tiny_stats["n_images"] == N_IMAGES

    def test_patch_argmax_valid(self, tiny_stats):
        argmax = tiny_stats["feature_argmax"]
        assert (argmax >= 0).all()
        assert (argmax < N_PATCHES).all()

    def test_dead_features_returns_tensor(self, tiny_stats):
        dead = get_dead_features(tiny_stats, threshold=0.0)
        assert isinstance(dead, torch.Tensor)
        assert dead.ndim == 1


# ═══════════════════════════════════════════════════════════════════════════
# spatial_maps.py
# ═══════════════════════════════════════════════════════════════════════════

class TestSpatialMaps:
    def test_get_image_activations_shape(self, tiny_activations, tiny_sae, tiny_config):
        f = get_image_activations(0, tiny_activations, tiny_sae, tiny_config)
        assert f.shape == (N_PATCHES, D_SAE)

    def test_spatial_map_shape(self, tiny_activations, tiny_sae, tiny_config):
        f = get_image_activations(0, tiny_activations, tiny_sae, tiny_config)
        hm = get_spatial_activation_map(f, feature_idx=0,
                                        spatial_h=SPATIAL_H, spatial_w=SPATIAL_W)
        assert hm.shape == (SPATIAL_H, SPATIAL_W)

    def test_upsample_range(self):
        hm = torch.tensor([[0.0, 1.0], [0.5, 0.3]])
        up = upsample_and_normalise(hm, target_h=8, target_w=8)
        assert up.shape == (8, 8)
        assert up.min() >= 0.0
        assert up.max() <= 1.0 + 1e-6


# ═══════════════════════════════════════════════════════════════════════════
# graph_utils.py
# ═══════════════════════════════════════════════════════════════════════════

class TestGraphUtils:
    def test_interaction_diagonal_zero(self, tiny_activations, tiny_sae, tiny_config):
        f = get_image_activations(0, tiny_activations, tiny_sae, tiny_config)
        active = get_active_features(f, threshold=0.0)
        if active.numel() == 0:
            pytest.skip("No active features in this random seed")
        inter = compute_interaction_matrix(tiny_sae, active)
        assert inter.diagonal().abs().sum().item() == 0.0

    def test_attribution_shape(self, tiny_activations, tiny_sae, tiny_config):
        f = get_image_activations(0, tiny_activations, tiny_sae, tiny_config)
        active = get_active_features(f, threshold=0.0)
        if active.numel() == 0:
            pytest.skip("No active features")
        inter = compute_interaction_matrix(tiny_sae, active)
        attr = compute_prompt_attribution(f, active, inter)
        n = active.numel()
        assert attr.shape == (n, n)

    def test_prune_sparsity(self, tiny_activations, tiny_sae, tiny_config):
        f = get_image_activations(0, tiny_activations, tiny_sae, tiny_config)
        active = get_active_features(f, threshold=0.0)
        if active.numel() == 0:
            pytest.skip("No active features")
        inter = compute_interaction_matrix(tiny_sae, active)
        attr = compute_prompt_attribution(f, active, inter)
        top_n = 5
        pruned = prune_graph(attr, top_n=top_n)
        n_nonzero = (pruned != 0).sum().item()
        expected = min(top_n, active.numel() ** 2)
        assert n_nonzero <= expected

    def test_build_graph_data_keys(self, tiny_activations, tiny_sae, tiny_config):
        gd = build_graph_data(0, tiny_activations, tiny_sae, tiny_config)
        expected_keys = {
            "active_indices", "attribution", "edge_attributions", "logit_attribution",
            "feature_strengths", "n_active", "image_idx",
        }
        assert set(gd.keys()) == expected_keys
        assert gd["image_idx"] == 0
        assert gd["logit_attribution"] is None  # no classifier passed


# ═══════════════════════════════════════════════════════════════════════════
# semantic_dict.py
# ═══════════════════════════════════════════════════════════════════════════

class TestSemanticDict:
    @pytest.fixture
    def dummy_img(self):
        return Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )

    def test_patch_crop_size(self, dummy_img):
        """Centre patch with context=1 should be 3×3 cells."""
        crop = get_patch_crop(dummy_img, patch_idx=7 * 14 + 7,
                              spatial_h=14, spatial_w=14, context_patches=1)
        cell_h = 224 / 14  # 16
        cell_w = 224 / 14  # 16
        # 3 cells wide/tall
        assert crop.size[0] == int(3 * cell_w)
        assert crop.size[1] == int(3 * cell_h)

    def test_patch_crop_bounds_top_left(self, dummy_img):
        """Top-left corner patch with context=2: clamped to image edge."""
        crop = get_patch_crop(dummy_img, patch_idx=0,
                              spatial_h=14, spatial_w=14, context_patches=2)
        # row 0, col 0 → rows [0, 3), cols [0, 3) (clamped from -2..1 to 0..3)
        cell = 224 / 14
        assert crop.size[0] == int(3 * cell)
        assert crop.size[1] == int(3 * cell)

    def test_patch_crop_bounds_bottom_right(self, dummy_img):
        """Bottom-right corner patch: clamped so crop doesn't exceed image."""
        last_patch = 14 * 14 - 1  # row=13, col=13
        crop = get_patch_crop(dummy_img, patch_idx=last_patch,
                              spatial_h=14, spatial_w=14, context_patches=2)
        # Should not be empty
        assert crop.size[0] > 0
        assert crop.size[1] > 0
        # Should not exceed image bounds
        assert crop.size[0] <= 224
        assert crop.size[1] <= 224
