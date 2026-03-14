#!/bin/bash
set -e

echo "Running PM2 pipeline..."

python src/scripts/harvest_activations.py \
  --pm_index 2 \
  --output data/activations_pm2.safetensors

python src/train_sae.py \
  --config configs/sae_pm2.json \
  --activations_path data/activations_pm2.safetensors \
  --output_dir data/checkpoints_pm2
