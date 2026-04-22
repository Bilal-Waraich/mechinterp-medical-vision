# Experiment Plan: Single-Modality SAE for Universality

X-ray SAE trained across multiple disease datasets to test whether a foundation
model (UMedPT Swin Transformer) learns universal semantic representations.

---

## Thesis Decisions

| ID | Decision | Chosen | Rationale |
|---|---|---|---|
| TD-1 | Encoder checkpoint | **Base pretrained (pending path confirmation with UMedPT team)** | Base → clean universality claim; fine-tuned confounds by task-adaptation magnitude differences |
| TD-2 | Hook stage | **pm_index=2** (Stage 3→4, 7×7, d_model=1024) | Semantic disease features over local texture; SIIM-COVID19 bbox coverage confirms patch validation is feasible |
| TD-3 | Universality metric | **Two-stage: latent-level (≥1% frequency on ≥2 datasets + canonical class match) + direction-level (cosine sim ≥0.7 between per-dataset activation centroids)** | See TD-3 section below |
| TD-4 | Dictionary size | **expansion_factor=4** (d_sae=4096) | Forces merging over splitting; reserve 8 for cross-modality extension |
| TD-5 | Dataset caps | **CheXpert 10k PA, RSNA 10k, SIIM-COVID19 10k, TB 700 (all)** | SIIM replaces VinBigData: only 7.2% sub-patch bboxes vs 59.1% for VinBig at pm_index=2 |

---

## TD-3: Universality Metric (Locked)

Universality is evaluated at two levels. Both must pass for a latent to be
reported as a universal feature.

### Level 1 — Latent-level (activation frequency)

**Threshold basis:** Anthropic's SAEs target ~1% activation sparsity per token
(Gao et al., "Scaling and evaluating sparse autoencoders", 2024). Features below
0.01% are in the dead/noise band; features above 10% encode generic anatomy.
1% sits in the interpretable medium-density band.

**Rule:** A latent is *active on dataset D* if it fires on **≥1% of that
dataset's tokens**.

At pm_index=2 (49 tokens/image):
- Large datasets (10k images → 490k tokens): threshold = 4,900 tokens
- TB (700 images → 34,300 tokens): threshold = 343 tokens

A latent is a **universal feature candidate** if:
1. It is active (≥1% frequency) on ≥2 of the 4 datasets
2. Its dominant canonical pathology class matches across those datasets

### Level 2 — Feature direction (cosine similarity)

**Threshold basis:** SVCCA (Raghu et al. 2017) uses ρ̄ ≥ 0.5 for subspace
similarity. We compare individual directions, which is stricter, so threshold = 0.7.

**Rule:** For each candidate latent, compute the centroid of all activating
tokens per dataset in d_model space. Cosine similarity between any two
per-dataset centroids must be **≥0.70** for the latent to be confirmed universal.

Report the full distribution of cosine similarities for candidate latents.

### Pathology canonical groups

Labels across datasets are mapped to 7 canonical groups before analysis, grounded
in the Fleischner Society glossary (same taxonomy used by CheXpert and SIIM).
Inter-rater kappa for chest X-ray labels is 0.53–0.68 (RadGraph, Peng et al. 2021),
so matching at the raw-label level would be too strict.

| Canonical group | CheXpert | RSNA | SIIM-COVID19 | TB Shenzhen |
|---|---|---|---|---|
| Airspace opacity | Consolidation, Lung Opacity | Pneumonia positive | Opacity / consolidation | TB infiltrate (positive) |
| Pleural abnormality | Pleural Effusion, Pleural Other | — | — | — |
| Cardiac / mediastinal | Cardiomegaly, Enlarged Cardiomediastinum | — | — | — |
| Atelectasis | Atelectasis | — | — | — |
| Vascular / edema | Edema | — | — | — |
| No finding | No Finding | Normal | No finding | Negative |
| Other | Support Devices, Fracture | — | Pneumothorax | — |

### Null baseline

Both Level 1 and Level 2 tests are repeated on:
- A random Gaussian SAE (same architecture, untrained)
- A PCA baseline (top-4096 principal components)

If the null baseline passes the frequency threshold at a similar rate, the
threshold is too lenient and must be tightened.

### Cross-dataset probe (supplementary)

For candidate latents, train a logistic probe on one dataset's canonical-group
labels and test zero-shot on another. High AUROC = encoder places same pathology
in the same direction across datasets. This is the "leave-one-dataset-out" probe
and serves as independent confirmation of direction-level universality.

---

## Datasets

| Dataset | Images (cap) | View filter | Labels | Patch coverage at pm_index=2 |
|---|---|---|---|---|
| CheXpert | 10k | PA only | 14 pathology classes (uncertain = negative) | Image-level only |
| RSNA Pneumonia | 10k | PA only | Pneumonia/normal + bboxes | 17.2% sub-patch — usable |
| SIIM-COVID19 | 10k | PA only | Opacity/consolidation + bboxes | **7.2% sub-patch, 44.9% span 2×2+ patches — primary patch validation** |
| TB Shenzhen | 700 (all) | N/A | TB positive / negative | Image-level only |

TB oversampled with replacement during stratified batching. Augmentation
(random crop, brightness jitter, horizontal flip) applied to TB loader only.

CheXpert uncertain labels (-1): treated as **negative** (conservative).

---

## Concerns and Mitigations

| # | Concern | Severity | Mitigation | Status |
|---|---|---|---|---|
| 2 | Dominant dataset imbalance | High | Stratified DataLoader (25% per dataset per batch) | ✅ `StratifiedBatchSampler` in `src/train_sae.py` |
| 5 | Geometric/view bias | High | PA-only filter at harvest time | ✅ Applied in CheXpert + SIIM dataset loaders |
| 7 | Dead features with heterogeneous data | High | n_batches_to_dead → 25 | ✅ Default raised to 25 in `src/config.py` |
| 8 | Cross-dataset b_dec drift | High | Per-dataset centering before concatenation | ✅ `center_activations()` in `src/conglomerate/harvest_all.py`; `b_dec` initialised to zero on centred data |
| 9 | Encoder fine-tuning confound | High | Use base encoder (TD-1, pending) | ⚠️ Pending TD-1 resolution |
| 1 | Scanner latent trap | Moderate | Multinomial LR probe per latent → dataset_id (AUROC > 0.80 → flag) | ✅ `src/scripts/scanner_latent_audit.py` with balanced subsample |
| 6 | Hook stage | — | pm_index=2 (TD-2) | ✅ Locked |
| 3 | Magnitude mismatch | Moderate | RMS normalisation + per-dataset centering | ✅ Both applied in `src/conglomerate/harvest_all.py` |

---

## Phase 0 — Infrastructure

**Goal:** pipeline correct before any experiments run.

**Exit criterion:** harvest runs on all 4 datasets; batch composition verified
empirically (25% each); scanner-latent audit script runs on a toy checkpoint.

### Tasks

**0.1 — `src/datasets/chexpert_dataset.py`**
- Parse `train.csv` for PA-view filter and pathology labels
- Uncertain labels (-1) treated as negative (0)
- Resize to 224×224, greyscale → 3-channel
- `--n_samples` cap with deterministic seed

**0.2 — `src/datasets/siim_covid_dataset.py`**
- PA-view filter via CSV metadata
- Store bounding box metadata (image_id, x_min, y_min, x_max, y_max, label) for patch validation
- `--n_samples` cap

**0.3 — `src/datasets/tb_shenzhen_dataset.py`**
- PNG/JPEG loader (no DICOM)
- TB positive/negative from filename or companion CSV
- Augmentation: random horizontal flip, ±10% brightness, random crop + resize

**0.4 — `src/conglomerate/harvest_all.py`**
- Wire CheXpert, SIIM-COVID19, TB loaders
- `--n_samples` cap per dataset (deterministic seed)
- Per-dataset centering: subtract each dataset's mean activation vector before concatenation
- Store per-dataset mean vectors in output metadata
- Store `dataset_ids` tensor alongside activations

**0.5 — `src/train_sae.py`**
- `StratifiedActivationDataset`: enforces 25% per dataset per batch
- TB sampled with replacement; others without replacement per epoch
- Load `dataset_ids` tensor from `.safetensors`
- Log per-dataset batch fraction every 500 steps

**0.6 — `src/config.py`**
- `n_batches_to_dead`: 10 → 25
- `pm_index` default: 1 → 2
- Add `stratified: bool = True`

**0.7 — `src/scripts/scanner_latent_audit.py`**
- Load SAE checkpoint + activations with dataset_ids
- Per-latent logistic regression → dataset_id
- Output: per-latent dataset-predictive AUROC, sorted
- Flag latents with AUROC > 0.80 as scanner latents
- Save flagged indices to `scanner_latents.json`

---

## Phase 1 — Pilot Training

**Exit criterion:** dead features <5% at step 10k; per-dataset reconstruction
loss within 2× of each other; no instability.

| Task | Detail |
|---|---|
| 1.1 Small harvest | 2k images per dataset (8k total), pm_index=2, base encoder |
| 1.2 Short run | 10k steps; log dead-feature fraction and per-dataset reconstruction loss separately |
| 1.3 Sanity checks | Dead features, loss balance, TB memorisation (TB-only latents emerging?) |
| 1.4 Tune if needed | Bad TB loss → increase n_batches_to_dead; obvious splitting → tighten l1_coeff |

---

## Phase 2 — Full Training

| Task | Detail |
|---|---|
| 2.1 Full harvest | 10k CheXpert PA, 10k RSNA, 10k SIIM-COVID19 PA, 700 TB at pm_index=2 |
| 2.2 Full training | 200k steps, batch_size=2048, l1_coeff=2e-4, expansion_factor=4 |
| 2.3 Scanner-latent audit | Run scanner_latent_audit.py; flag AUROC >0.80; record count and identities |
| 2.4 Checkpoint selection | Where dead features stabilise and per-dataset EV plateaus |

---

## Phase 3 — Universality Analysis

| Task | Detail |
|---|---|
| 3.1 Co-activation matrix | For each non-scanner latent: activation frequency per (dataset × canonical class). Build latent × (dataset, class) matrix. |
| 3.2 Universal feature identification (Level 1) | Apply TD-3 Level 1: ≥1% frequency on ≥2 datasets + canonical class match. Report count, fraction, which classes covered. |
| 3.3 Direction confirmation (Level 2) | For Level 1 candidates: compute per-dataset activation centroids, cosine similarity ≥0.70. Report full distribution. |
| 3.4 Patch-level validation | SIIM-COVID19 + RSNA bboxes → soft patch-label assignment (continuous IoU overlap). Pearson r between latent activation magnitude and patch overlap. |
| 3.5 Cross-dataset probe | Leave-one-dataset-out logistic probe on canonical classes. AUROC as supplementary universality evidence. |
| 3.6 Null baseline | Repeat 3.2 + 3.3 on random SAE and PCA baseline. |
| 3.7 Qualitative inspection | Scanner-clean + AUROC >0.80 patch correlation latents only. Show top-N activating patches. Aim ≤15 latents. |

---

## Phase 4 — Cross-Modality Transfer (Extension)

**Contingent on Phase 3 finding genuine universal features.**

| Task | Detail |
|---|---|
| 4.1 MRI harvest | BraTS or MSD at pm_index=2, same base encoder, no SAE retraining |
| 4.2 Frozen encoder inference | Pass MRI activations through X-ray SAE encoder half only |
| 4.3 Transfer analysis | Do universal X-ray latents activate on semantically related MRI regions? |
| 4.4 Negative control | Scanner latents from Phase 2.3 must not transfer |

---

## Open Questions

- **TD-1:** Confirm base encoder checkpoint path with UMedPT team before any harvest.
- **CheXpert TD-1 dependency:** If only fine-tuned checkpoints are available, must decide whether to treat fine-tuned-per-dataset as the new TD-1 scope.
- **Clinical access:** Radiologist available for Phase 3.7? If not, bbox correlation (3.4) is primary quantitative ground truth.
