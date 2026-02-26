# BatchTopK Sparse Auto-Encoder for UMedPT

> Mechanistic interpretability through sparse dictionary learning on Swin Transformer activations.

This repo implements the full **Harvest → Train → Visualise** pipeline for a [BatchTopK Sparse Autoencoder](https://github.com/bartbussmann/BatchTopK) applied to the [UMedPT](https://github.com/FraunhoferMEVIS/MedicalMultitaskModeling) multi-task Swin Transformer.

---

## Repository Structure

```
sparse-auto-encoder/
├── pyproject.toml                      # Dependencies & project metadata
├── README.md
│
├── encoder-1.0.4.pt/                   # UMedPT encoder checkpoint (not tracked)
│   └── encoder.pt
│
├── UmedPT/                             # UMedPT source (submodule / symlink)
├── UmedPT-sdk/                         # UMedPT SDK
│
├── src/
│   ├── config.py                       # SAEConfig (training hyperparams)
│   ├── sae_models.py                   # BatchTopKSAE nn.Module
│   ├── train_sae.py                    # Phase 2: training loop + CLI
│   ├── visualise.py                    # Phase 3: top-level CLI entry-point
│   │
│   ├── scripts/
│   │   └── harvest_activations.py      # Phase 1: activation harvesting CLI
│   │
│   ├── visualisation/                  # Phase 3: visualisation modules
│   │   ├── __init__.py
│   │   ├── vis_config.py               #   VisConfig dataclass
│   │   ├── loaders.py                  #   Load SAE, activations, images
│   │   ├── collector.py                #   Feature statistics (max, argmax, mean)
│   │   ├── semantic_dict.py            #   Layer 1: patch crop grids
│   │   ├── spatial_maps.py             #   Layer 2: per-feature heatmap overlays
│   │   ├── graph_utils.py              #   Layer 3: W_dec·W_enc interaction matrix
│   │   └── attribution_graph.py        #   Layer 3: networkx graph renderer
│   │
│   └── tests/
│       ├── conftest.py                 #   Shared pytest fixtures
│       └── test_visualisation.py       #   13 CPU-only tests
│
└── data/                               # All generated artefacts (gitignored)
    ├── dummy_activations.safetensors
    ├── checkpoints/
    │   └── step_050000/
    └── vis_outputs/
```

---

## Installation

```bash
cd sparse-auto-encoder

# Core dependencies
pip install -e ".[dev]"

# UMedPT dependencies (required for Phase 1 — activation harvesting)
pip install -e UmedPT/
```

---

## End-to-End Pipeline

### Phase 1 — Harvest Activations

Extract intermediate representations from the frozen Swin Transformer backbone.

```bash
python src/scripts/harvest_activations.py \
    --checkpoint encoder-1.0.4.pt/encoder-1.0.4.pt/encoder.pt \
    --n_samples 50000 \
    --batch_size 32 \
    --image_size 224 \
    --pm_index 1 \
    --output data/dummy_activations.safetensors \
    --device cuda:0
```

| Parameter | Description |
|---|---|
| `--checkpoint` | Path to the serialised UMedPT `encoder.pt` |
| `--n_samples` | Number of images to process |
| `--pm_index` | PatchMerging layer index (see table below) |
| `--output` | Output `.safetensors` file |

**PatchMerging index guide:**

| `--pm_index` | Boundary | `d_model` | Spatial |
|:---:|---|:---:|:---:|
| 0 | Stage 1→2 | 256 | 28×28 |
| **1** | **Stage 2→3** | **512** | **14×14** |
| 2 | Stage 3→4 | 1024 | 7×7 |

**What it does:**
1. Loads images (mmm ShapeDataset or built-in synthetic fallback)
2. Hooks the Swin PatchMerging layer at the chosen stage boundary
3. Captures activations `(B, H', W', C')` → flattens to `(N_eff, d_model)`
4. Applies Anthropic-style RMS normalisation
5. Saves to `.safetensors` with metadata (`spatial_h`, `spatial_w`, `d_model`, `n_images`)

**Output:** `data/dummy_activations.safetensors`

---

### Phase 2 — Train SAE

Train a BatchTopK Sparse Autoencoder on the harvested activations.

```bash
python src/train_sae.py \
    --d_model 512 \
    --expansion_factor 4 \
    --k 32 \
    --batch_size 2048 \
    --steps 50000 \
    --checkpoint_every 5000 \
    --activations_path data/dummy_activations.safetensors \
    --output_dir data/checkpoints \
    --device cuda:0
```

| Parameter | Default | Description |
|---|:---:|---|
| `--d_model` | 384 | Input activation dimension |
| `--expansion_factor` | 4 | Dictionary size = `d_model × expansion_factor` |
| `--k` | 32 | Target active features per token |
| `--batch_size` | 2048 | Training batch size |
| `--steps` | 50,000 | Total training steps |
| `--checkpoint_every` | 5,000 | Steps between checkpoints |
| `--wandb_project` | `sae-umedpt` | W&B project (optional) |

**Output:** `data/checkpoints/step_050000/` containing:
- `sae_weights.safetensors` — model state dict
- `cfg.json` — full training config snapshot

---

### Phase 3 — Visualise

Generate interpretability dashboards from a trained SAE checkpoint.

> **Note:** `--images_dir` must contain the same images used during harvest, sorted alphabetically to match the activation order.

#### Option A: Run the full pipeline at once

```bash
python src/visualise.py all \
    --checkpoint_dir data/checkpoints/step_050000 \
    --activations_path data/dummy_activations.safetensors \
    --images_dir data/images \
    --output_dir data/vis_outputs \
    --device cuda:0
```

This runs **collect → Layer 1 → Layer 3** automatically:
1. Computes and saves feature statistics (`stats.pt`)
2. Renders semantic dictionaries for the top-20 most active features
3. Renders attribution graphs with spatial companion panels for the first 5 images

#### Option B: Run individual steps

<details>
<summary><b>Step 3a — Collect feature statistics</b></summary>

```bash
python src/visualise.py collect \
    --checkpoint_dir data/checkpoints/step_050000 \
    --activations_path data/dummy_activations.safetensors \
    --stats_output data/vis_outputs/stats.pt \
    --device cuda:0
```

Encodes all activation rows through the SAE, reshapes to `[n_images, n_patches, d_sae]`, and computes per-feature/per-image statistics (`feature_max`, `feature_argmax`, `feature_mean`, `global_max`).
</details>

<details>
<summary><b>Step 3b — Layer 1: Semantic dictionaries</b></summary>

```bash
python src/visualise.py layer1 \
    --checkpoint_dir data/checkpoints/step_050000 \
    --activations_path data/dummy_activations.safetensors \
    --images_dir data/images \
    --stats_path data/vis_outputs/stats.pt \
    --top_n 20 \
    --output_dir data/vis_outputs/semantic_dicts
```

For each feature, shows a 3×3 grid of the top-9 images that activated it most, cropped to the most-activating patch with spatial context. Use `--feature_idx 42` for a single feature instead of `--top_n`.
</details>

<details>
<summary><b>Step 3c — Layer 2: Spatial activation maps</b></summary>

```bash
python src/visualise.py layer2 \
    --checkpoint_dir data/checkpoints/step_050000 \
    --activations_path data/dummy_activations.safetensors \
    --images_dir data/images \
    --image_idx 7 \
    --feature_indices 0 14 203 \
    --output_dir data/vis_outputs/spatial_maps
```

For a single image, renders per-feature heatmaps showing *where* each feature fires, with three panels: original image | heatmap | overlay.
</details>

<details>
<summary><b>Step 3d — Layer 3: Attribution graph</b></summary>

```bash
python src/visualise.py layer3 \
    --checkpoint_dir data/checkpoints/step_050000 \
    --activations_path data/dummy_activations.safetensors \
    --images_dir data/images \
    --image_idx 7 \
    --output_dir data/vis_outputs/attribution_graphs
```

Computes pairwise feature interactions via the linear approximation `interaction(i→j) = W_dec[i] · W_enc[:, j]`, weighted by mean activation. Renders a directed networkx graph (blue = excitatory, red = inhibitory) plus a companion spatial-overlay panel.
</details>

---

### Output Structure

```
data/vis_outputs/
├── stats.pt                                    # Feature statistics
├── semantic_dicts/                             # Layer 1
│   ├── feature_00042.png
│   ├── feature_00107.png
│   └── ...
├── spatial_maps/                               # Layer 2
│   └── spatial_image_00007.png
└── attribution_graphs/                         # Layer 3
    ├── graph_image_00007.png
    └── graph_spatial_image_00007.png
```

---

## Tests

All tests run on CPU with tiny synthetic data (no GPU or disk I/O required):

```bash
python -m pytest src/tests/test_visualisation.py -v
```

---

## Architecture Overview

```
Phase 1: Harvest                Phase 2: Train               Phase 3: Visualise
─────────────────              ──────────────────           ────────────────────
                                                           
Swin Transformer               Flat activations             Trained SAE
    │                          [N_eff, d_model]             + activations
    │ forward hook                  │                       + images
    ▼                               ▼                           │
PatchMerging(pm_idx)       ┌────────────────┐          ┌───────┴────────┐
    │                      │  BatchTopKSAE  │          │                │
    │ (B,H,W,C) → flat     │  d_sae = d×4   │          │   visualise.py │
    ▼                      │  k = 32        │          │   CLI router   │
[N_eff, d_model]           └────────┬───────┘          │                │
    │                               │                  └───────┬────────┘
    │ RMS normalise                 │ checkpoint                │
    ▼                               ▼                  ┌───────┼────────┐
.safetensors               step_XXXXX/                 │       │        │
                           ├─ sae_weights.st           ▼       ▼        ▼
                           └─ cfg.json              Layer1  Layer2   Layer3
                                                    crops   spatial  graph
                                                    3×3     heatmap  W_dec·W_enc
```

---

## Authors

- **Nik** — 81419030+Nikolay-Ts@users.noreply.github.com
- **Bilal** — 157110822+Bilal-Waraich@users.noreply.github.com

Fraunhofer MEVIS — Mechanistic Interpretability Group
