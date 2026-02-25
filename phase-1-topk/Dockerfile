# ─────────────────────────────────────────────────────────────────────────────
# TopK SAE — Activation Harvester
#
# Runs scripts/harvest_activations.py inside a Nomad job.
#
# Build (from repo root):
#   docker build -f phase-1-topk/Dockerfile -t sae-harvest:latest .
#
# Run locally (GPU):
#   docker run --gpus all \
#       -v $(pwd)/data:/app/data \
#       sae-harvest:latest
#
# Override options:
#   docker run --gpus all \
#       -v $(pwd)/data:/app/data \
#       -e N_SAMPLES=10000 \
#       -e PM_INDEX=1 \
#       sae-harvest:latest
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

# ── Copy UmedPT packages first (cache-friendly layer) ─────────────────────────
COPY UmedPT/     ./UmedPT/
COPY UmedPT-sdk/ ./UmedPT-sdk/

# ── Install packages ──────────────────────────────────────────────────────────
# UmedPT / mmm
RUN pip install --no-cache-dir -e ./UmedPT

# UmedPT-sdk / m3_sdk
RUN pip install --no-cache-dir -e ./UmedPT-sdk

# Extra runtime dependencies
RUN pip install --no-cache-dir \
        safetensors \
        pillow \
        tqdm \
        numpy \
        pandas \
        matplotlib \
        scikit-image \
        opencv-python-headless

# ── Copy encoder weights ──────────────────────────────────────────────────────
# encoder-1.0.4.pt/ contains: encoder.pt, decoder.pt, fcosfpn.pt, etc.
COPY encoder-1.0.4.pt/ ./encoder-1.0.4.pt/

# ── Copy harvest script ───────────────────────────────────────────────────────
COPY phase-1-topk/scripts/harvest_activations.py ./scripts/harvest_activations.py

# ── Output directory (mount a volume here in production) ─────────────────────
RUN mkdir -p /app/data

# ── Environment ───────────────────────────────────────────────────────────────
ENV MMM_LICENSE_ACCEPTED="i accept"

# Configurable via -e flags or Nomad task env block
ENV CHECKPOINT="encoder-1.0.4.pt/encoder.pt"
ENV N_SAMPLES="50000"
ENV BATCH_SIZE="32"
ENV IMAGE_SIZE="224"
ENV PM_INDEX="1"
ENV OUTPUT_PATH="data/dummy_activations.safetensors"
ENV DEVICE="cuda:0"

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD python scripts/harvest_activations.py \
        --checkpoint  "${CHECKPOINT}"  \
        --n_samples   "${N_SAMPLES}"   \
        --batch_size  "${BATCH_SIZE}"  \
        --image_size  "${IMAGE_SIZE}"  \
        --pm_index    "${PM_INDEX}"    \
        --output      "${OUTPUT_PATH}" \
        --device      "${DEVICE}"
