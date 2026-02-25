###
### BatchTopK SAE — Activation Harvester + Trainer
### Runs on Nomad (GPU).  Select job via JOB env var (default: harvest).
###
### Build (from repo root):
###   docker build -t sae-batchtopk:latest .
###
### Run harvest:
###   docker run --gpus all -v $(pwd)/data:/app/data sae-batchtopk:latest
###
### Run training (after harvest):
###   docker run --gpus all \
###       -v $(pwd)/data:/app/data \
###       -v $(pwd)/sae_output:/app/sae_output \
###       -e JOB=train \
###       sae-batchtopk:latest
###

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy local UmedPT packages
COPY UmedPT/     ./UmedPT/
COPY UmedPT-sdk/ ./UmedPT-sdk/

# Install all Python deps in one layer (pandas/matplotlib omitted — not used)
RUN pip install --no-cache-dir -e ./UmedPT -e ./UmedPT-sdk \
 && pip install --no-cache-dir \
    numpy \
    tqdm \
    safetensors \
    sae-lens \
    pillow \
    scikit-image \
    opencv-python-headless

# Copy checkpoint and scripts
COPY encoder-1.0.4.pt/                          ./encoder-1.0.4.pt/
COPY phase-1-topk/scripts/harvest_activations.py ./scripts/harvest_activations.py
COPY phase-1-topk/scripts/train_sae.py           ./scripts/train_sae.py

# Output directories
RUN mkdir -p /app/data /app/sae_output

ENV MMM_LICENSE_ACCEPTED="i accept"

# ── Harvest defaults ──────────────────────────────────────────────────────────
ENV CHECKPOINT="encoder-1.0.4.pt/encoder.pt"
ENV N_SAMPLES="50000"
ENV BATCH_SIZE="32"
ENV IMAGE_SIZE="224"
ENV PM_INDEX="1"
ENV OUTPUT_PATH="data/batchtopk_activations.safetensors"
ENV DEVICE="cuda:0"

# ── Training defaults ─────────────────────────────────────────────────────────
ENV ACT_PATH="data/batchtopk_activations.safetensors"
ENV OUTPUT_DIR="sae_output"
ENV D_SAE=""
ENV K="50"
ENV LR="5e-5"
ENV N_EPOCHS="10"
ENV BATCH_SIZE_TRAIN="4096"

# ── Job selector (harvest | train) ───────────────────────────────────────────
ENV JOB="harvest"

CMD if [ "${JOB}" = "train" ]; then \
        python scripts/train_sae.py \
            --activations "${ACT_PATH}" \
            --output_dir  "${OUTPUT_DIR}" \
            ${D_SAE:+--d_sae "${D_SAE}"} \
            --k           "${K}" \
            --lr          "${LR}" \
            --n_epochs    "${N_EPOCHS}" \
            --batch_size  "${BATCH_SIZE_TRAIN}" \
            --device      "${DEVICE}"; \
    else \
        python scripts/harvest_activations.py \
            --checkpoint  "${CHECKPOINT}" \
            --n_samples   "${N_SAMPLES}" \
            --batch_size  "${BATCH_SIZE}" \
            --image_size  "${IMAGE_SIZE}" \
            --pm_index    "${PM_INDEX}" \
            --output      "${OUTPUT_PATH}" \
            --device      "${DEVICE}"; \
    fi
