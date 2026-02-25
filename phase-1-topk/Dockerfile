# ─────────────────────────────────────────────────────────────────────────────
# BatchTopK SAE — Activation Harvester + Trainer
#
# Two jobs in one image; select via CMD override or Nomad task env:
#
#   harvest (default):
#     docker run --gpus all -v $(pwd)/data:/app/data sae-batchtopk:latest
#
#   train:
#     docker run --gpus all \
#         -v $(pwd)/data:/app/data \
#         -v $(pwd)/sae_output:/app/sae_output \
#         -e JOB=train \
#         sae-batchtopk:latest
#
# Build (from repo root):
#   docker build -f phase-1-topk/Dockerfile -t sae-batchtopk:latest .
#
# Override env vars per-job:
#   Harvest  : N_SAMPLES, BATCH_SIZE, IMAGE_SIZE, PM_INDEX, CHECKPOINT, OUTPUT_PATH, DEVICE
#   Training : ACT_PATH, OUTPUT_DIR, D_SAE, K, LR, N_EPOCHS, BATCH_SIZE_TRAIN, DEVICE
# ─────────────────────────────────────────────────────────────────────────────

# pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime is the smallest official image
# that ships both CUDA 12.1 + cuDNN 8 + PyTorch 2.1.  The "devel" variant is
# ~3 GB larger and not needed at runtime.
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app


RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        unzip \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Copy UmedPT packages first (cache-friendly layer) ─────────────────────────
COPY UmedPT/     ./UmedPT/
COPY UmedPT-sdk/ ./UmedPT-sdk/


RUN pip install --no-cache-dir -e ./UmedPT -e ./UmedPT-sdk \
 && pip install --no-cache-dir \
        safetensors \
        sae-lens \
        pillow \
        tqdm \
        numpy \
        scikit-image \
        opencv-python-headless

# ── Copy encoder weights ──────────────────────────────────────────────────────
COPY encoder-1.0.4.pt/ ./encoder-1.0.4.pt/

# ── Copy scripts ──────────────────────────────────────────────────────────────
COPY phase-1-topk/scripts/harvest_activations.py ./scripts/harvest_activations.py
COPY phase-1-topk/scripts/train_sae.py           ./scripts/train_sae.py

# ── Output directories (mount volumes here in production) ─────────────────────
RUN mkdir -p /app/data /app/sae_output

# ── Environment ───────────────────────────────────────────────────────────────
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

# ── Job selector ──────────────────────────────────────────────────────────────
# JOB=harvest  →  run harvest_activations.py  (default)
# JOB=train    →  run train_sae.py
ENV JOB="harvest"

# ── Entrypoint ────────────────────────────────────────────────────────────────
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
