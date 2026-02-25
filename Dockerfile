###
### This is used to run the harvester on NOMAD
###


FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy local UmedPT packages
COPY UmedPT/ ./UmedPT/
COPY UmedPT-sdk/ ./UmedPT-sdk/

# Install UmedPT packages
RUN pip install --no-cache-dir -e ./UmedPT
RUN pip install --no-cache-dir -e ./UmedPT-sdk

# Install remaining dependencies
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    matplotlib \
    tqdm \
    safetensors \
    pillow \
    opencv-python-headless

# Copy checkpoint and script
COPY encoder-1.0.4.pt/ ./encoder-1.0.4.pt/
COPY phase-1/harvest_activations.py .

# Output directory
RUN mkdir -p /app/data

ENV MMM_LICENSE_ACCEPTED="i accept"

CMD ["python", "harvest_activations.py"]