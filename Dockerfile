# ─────────────────────────────────────────────────────────────────────────────
# SAE Pipeline — unified image for harvest + train
#
# Encoder checkpoint is NOT baked in — it is mounted at runtime from the
# input CSI volume at:
#   /deep_learning/input/data/MechanisticInterpretability_BWA_NLA/encoder-1.0.4.pt/encoder.pt
#
# Build (from repo root):
#   docker build -t registry.example.com/<your-project>/sae-pipeline:latest .
#   docker push registry.example.com/<your-project>/sae-pipeline:latest
#
# Run locally (harvest only, GPU):
#   docker run --gpus all \
#       -v /path/to/encoder/dir:/deep_learning/input:ro \
#       -v $(pwd)/data:/alloc/data \
#       registry.example.com/<your-project>/sae-pipeline:latest \
#       python src/scripts/harvest_activations.py \
#           --checkpoint /deep_learning/input/data/MechanisticInterpretability_BWA_NLA/encoder-1.0.4.pt/encoder.pt \
#           --n_samples 10000 --batch_size 64 \
#           --output /alloc/data/activations.safetensors
#
# In production: tasks and args are defined in nomad/sae-pipeline.nomad
# ─────────────────────────────────────────────────────────────────────────────

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        unzip \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── UmedPT packages (cached layer — only re-runs if UmedPT/ changes) ──────────
COPY UmedPT/     ./UmedPT/
COPY UmedPT-sdk/ ./UmedPT-sdk/
RUN pip install --no-cache-dir -e ./UmedPT && \
    pip install --no-cache-dir -e ./UmedPT-sdk

# ── Python dependencies ───────────────────────────────────────────────────────
# torch/cuda already in base image; torchvision added for PatchMergingV2
# wandb intentionally omitted — train_sae.py gracefully falls back to
# console-only logging when wandb is not installed (try/except ImportError)
RUN pip install --no-cache-dir \
        torchvision \
        "safetensors>=0.4.0" \
        "numpy>=1.24.0" \
        pillow \
        "scikit-image>=0.25.2,<0.26" \
        opencv-python-headless \
        "tqdm>=4.66.1,<5" \
        "pandas>=2.1.1,<3" \
        "matplotlib>=3.7.0" \
        "networkx>=3.0" \
        "packaging>=21.0" \
        logfire \
        segmentation-models-pytorch \
        monai \
        torchinfo \
        shapely \
        rasterio \
        tiffslide \
        universal-pathlib \
        pydantic-settings \
        albumentations

# ── SAE source ────────────────────────────────────────────────────────────────
COPY src/ ./src/

# ── Runtime directories ───────────────────────────────────────────────────────
# /deep_learning/input  → encoder checkpoint (CSI volume, read-only, mounted by Nomad)
# /deep_learning/output → SAE checkpoints   (CSI volume, read-write, mounted by Nomad)
# NOMAD_ALLOC_DIR       → intermediate activations (ephemeral, shared between tasks)
RUN mkdir -p /deep_learning/input /deep_learning/output

# ── Environment ───────────────────────────────────────────────────────────────
ENV MMM_LICENSE_ACCEPTED="i accept"
# src/ on PYTHONPATH so train_sae.py can import config and sae_models
ENV PYTHONPATH="/app/src"
