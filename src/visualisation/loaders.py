"""Loaders for SAE checkpoints, activation tensors, and image file lists.

These are the entry-points for Phase 3 visualisation code to load the two
key artefacts produced by training (checkpoint + harvested activations)
plus the original image paths for overlay rendering.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

# Allow running from the repo root (`python -m src.visualisation.loaders`)
# or from within `src/` — either way, make sure `src` is importable.
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from sae_models import BatchTopKSAE  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────
# SAE loader
# ──────────────────────────────────────────────────────────────────────────

def load_sae(checkpoint_dir: str, device: str = "cpu") -> BatchTopKSAE:
    """Load a trained BatchTopKSAE from a checkpoint directory.

    The directory must contain:
      * ``cfg.json`` — training config with at least ``d_model``,
        ``expansion_factor``, and ``k``.
      * ``sae_weights.safetensors`` — state dict with keys
        ``W_enc``, ``W_dec``, ``b_dec``, ``num_batches_not_active``.

    Parameters
    ----------
    checkpoint_dir : str
        Path to the checkpoint directory.
    device : str
        Target device, e.g. ``"cpu"`` or ``"cuda:0"``.

    Returns
    -------
    BatchTopKSAE
        Model loaded in **eval** mode on *device*.
    """
    ckpt = Path(checkpoint_dir)

    # 1. Read training config
    with open(ckpt / "cfg.json", "r", encoding="utf-8") as f:
        cfg: dict = json.load(f)

    d_model: int = int(cfg["d_model"])
    expansion_factor: int = int(cfg["expansion_factor"])
    k: int = int(cfg["k"])
    d_sae: int = d_model * expansion_factor

    # 2. Instantiate model
    model = BatchTopKSAE(d_model=d_model, d_sae=d_sae, k=k)

    # 3. Load weights
    state = load_file(str(ckpt / "sae_weights.safetensors"), device=device)
    model.load_state_dict(state, strict=False)

    model.to(device)
    model.eval()
    return model


# ──────────────────────────────────────────────────────────────────────────
# Activation loader
# ──────────────────────────────────────────────────────────────────────────

def load_activations(
    activations_path: str,
    device: str = "cpu",
) -> tuple[torch.Tensor, dict]:
    """Load harvested activations from a ``.safetensors`` file.

    Parameters
    ----------
    activations_path : str
        Path to the ``.safetensors`` file.  Must contain a tensor stored
        under the key ``"activations"`` with shape ``[N_eff, d_model]``.
        String metadata keys (``spatial_h``, ``spatial_w``, ``n_images``,
        ``d_model``, …) are also extracted.
    device : str
        Target device.

    Returns
    -------
    activations : Tensor [N_eff, d_model]
        Float32 activations on *device*.
    metadata : dict
        All metadata from the safetensors header, with numeric-looking
        string values cast to ``int``.
    """
    from safetensors import safe_open

    tensors = load_file(activations_path, device=device)
    activations = tensors["activations"].to(dtype=torch.float32)

    # Extract metadata via safe_open (safetensors stores metadata as str→str)
    metadata: dict = {}
    with safe_open(activations_path, framework="pt") as f:
        raw_meta = f.metadata() or {}
        for k, v in raw_meta.items():
            # Cast to int if the value looks like a decimal integer
            try:
                metadata[k] = int(v)
            except (ValueError, TypeError):
                metadata[k] = v

    return activations, metadata


# ──────────────────────────────────────────────────────────────────────────
# Image-list loader
# ──────────────────────────────────────────────────────────────────────────

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
_DCM_EXTENSIONS = {".dcm"}


def load_images(images_dir: str) -> list[Path]:
    """Return a sorted list of image paths found recursively under *images_dir*.

    Searches for ``.png``, ``.jpg``, ``.jpeg`` at any depth.  Falls back to
    ``.dcm`` (DICOM) if no standard image files are found.
    Sort order is alphabetical, matching the harvest order.

    Parameters
    ----------
    images_dir : str
        Directory containing the original images.

    Returns
    -------
    list[Path]
        Sorted image paths.
    """
    root = Path(images_dir)
    paths = sorted(
        p for p in root.rglob("*")
        if p.suffix.lower() in _IMAGE_EXTENSIONS
    )
    if not paths:
        # Fall back to DICOM
        paths = sorted(
            p for p in root.rglob("*")
            if p.suffix.lower() in _DCM_EXTENSIONS
        )
    return paths


def load_images_multi(
    images_dirs: list[str],
    image_stems_by_ds: dict[str, list[str]],
    dataset_names: list[str],
) -> list[Path]:
    """Build a flat, activation-aligned list of image paths from multiple dirs.

    Uses the ``image_stems`` metadata written by ``harvest_all.py`` to match
    each harvested image back to its file path in the appropriate directory.

    Parameters
    ----------
    images_dirs : list[str]
        Each entry has the format ``"path:label"`` where *label* matches a
        dataset name from the merged activations.
    image_stems_by_ds : dict[str, list[str]]
        ``{dataset_name: [stem, ...]}`` — one stem per image in harvest order.
        Obtained from the merged safetensors metadata key ``image_stems``.
    dataset_names : list[str]
        Ordered list of dataset names matching the ``dataset_ids`` tensor.

    Returns
    -------
    list[Path]
        Flat list of image paths, ordered to match the activation tensor.
        Entries whose stem could not be resolved are set to ``None``.
    """
    # Build {label: dir_path}
    dir_by_label: dict[str, Path] = {}
    for entry in images_dirs:
        path, label = entry.rsplit(":", 1)
        dir_by_label[label] = Path(path)

    # Build {stem: path} lookup per dataset (recursive scan)
    stem_to_path: dict[str, dict[str, Path]] = {}
    for label, root in dir_by_label.items():
        mapping: dict[str, Path] = {}
        all_files = list(root.rglob("*"))
        for p in all_files:
            if p.suffix.lower() in (_IMAGE_EXTENSIONS | _DCM_EXTENSIONS):
                mapping[p.stem] = p
        stem_to_path[label] = mapping
        print(f"  {label}: {len(mapping)} images indexed from {root}")

    # Build flat image_paths in activation order
    image_paths: list[Path | None] = []
    n_resolved = 0
    n_missing = 0
    for ds_name in dataset_names:
        stems = image_stems_by_ds.get(ds_name, [])
        lookup = stem_to_path.get(ds_name, {})
        for stem in stems:
            p = lookup.get(stem)
            image_paths.append(p)
            if p is not None:
                n_resolved += 1
            else:
                n_missing += 1

    print(f"  Image stem matching: {n_resolved} resolved, {n_missing} missing")
    return image_paths


# ── Smoke tests ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile
    import os

    print("=" * 60)
    print("  loaders.py — smoke tests")
    print("=" * 60)

    # ── 1. load_images ────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        for name in ["c_img.png", "a_img.jpg", "b_img.PNG", "readme.txt"]:
            Path(tmpdir, name).touch()

        imgs = load_images(tmpdir)
        names = [p.name for p in imgs]
        # .PNG is also valid, .txt should be excluded
        assert "readme.txt" not in names
        assert names == sorted(names), f"Not sorted: {names}"
        assert len(imgs) == 3
        print(f"[OK] load_images: {names}")

    # ── 2. load_activations ──────────────────────────────────────────
    from safetensors.torch import save_file

    with tempfile.TemporaryDirectory() as tmpdir:
        act_path = os.path.join(tmpdir, "acts.safetensors")
        dummy = torch.randn(196, 512)
        save_file(
            {"activations": dummy},
            act_path,
            metadata={"spatial_h": "14", "spatial_w": "14", "n_images": "1", "d_model": "512"},
        )

        acts, meta = load_activations(act_path, device="cpu")
        assert acts.shape == (196, 512)
        assert meta["spatial_h"] == 14  # int, not str
        assert meta["d_model"] == 512
        print(f"[OK] load_activations: shape={acts.shape}, meta={meta}")

    # ── 3. load_sae ──────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write a fake cfg.json
        cfg = {"d_model": 64, "expansion_factor": 2, "k": 8}
        with open(os.path.join(tmpdir, "cfg.json"), "w") as f:
            json.dump(cfg, f)

        # Create a tiny model, save its state dict
        tiny = BatchTopKSAE(d_model=64, d_sae=128, k=8)
        state = {k: v.contiguous() for k, v in tiny.state_dict().items()}
        save_file(state, os.path.join(tmpdir, "sae_weights.safetensors"))

        loaded = load_sae(tmpdir, device="cpu")
        assert loaded.d_model == 64
        assert loaded.d_sae == 128
        assert loaded.k == 8
        assert not loaded.training  # must be in eval mode
        print(f"[OK] load_sae: d_model={loaded.d_model}, d_sae={loaded.d_sae}, eval={not loaded.training}")

    print("\nAll loaders smoke tests passed ✓")
