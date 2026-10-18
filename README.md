# BatchTopK Sparse Auto-Encoder for UMedPT

> Mechanistic interpretability through sparse dictionary learning on Swin Transformer activations.

This repo implements the full **Harvest → Train → Visualise** pipeline for a [BatchTopK Sparse Autoencoder](https://github.com/bartbussmann/BatchTopK) applied to the [UMedPT](https://github.com/FraunhoferMEVIS/MedicalMultitaskModeling) multi-task Swin Transformer.

---

## Authors

- **Nikolay Tsonev** — [@Nikolay-Ts](https://github.com/Nikolay-Ts)
- **Bilal Waraich** — [@Bilal-Waraich](https://github.com/Bilal-Waraich)

Fraunhofer MEVIS — Mechanistic Interpretability Group

## Branches

- **`batch-topk`** (default) — the BatchTopK SAE pipeline: harvest → train → visualise, plus the thesis analyses.
- **`Matryoshka`** — the Matryoshka SAE implementation (`src/matryoshka_sae.py`) and its analyses. This work lives only on that branch.
