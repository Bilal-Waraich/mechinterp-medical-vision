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

## Phase 0 — Infrastructure (Complete)

**Goal:** pipeline correct before any experiments run.
**Exit criterion (met):** harvest runs on all 4 datasets; batch composition verified empirically (25% each); scanner-latent audit runs on every checkpoint.

### Delivered

| Task | Artifact | Status |
|---|---|---|
| 0.1 CheXpert loader | `src/datasets/chexpert_dataset.py` — PA filter, 14-label vector, uncertain → 0, `--n_samples` cap | ✅ |
| 0.2 SIIM-COVID loader | `src/datasets/siim_covid_dataset.py` — PA filter, bbox records, study-level canonical groups | ✅ |
| 0.3 TB Shenzhen loader | `src/datasets/tb_shenzhen_dataset.py` — filename-encoded labels, augmentation | ✅ |
| 0.4 Conglomerate harvester | `src/conglomerate/harvest_all.py` — all four datasets, per-dataset centering, `dataset_ids` + `image_stems` metadata | ✅ |
| 0.5 Stratified trainer | `src/train_sae.py` — `StratifiedBatchSampler`, TB with replacement, per-dataset fraction logged every 500 steps | ✅ |
| 0.6 Config defaults | `src/config.py` — `n_batches_to_dead=25`, `pm_index=2`, `stratified=True` | ✅ |
| 0.7 Scanner audit | `src/scripts/scanner_latent_audit.py` — multinomial LR per latent, balanced subsample, AUROC > 0.80 → flag | ✅ |

### Additions beyond the original Phase-0 scope

- **Image-stems metadata (`docs/methodology.md` §1.6):** `harvest_all.py` records harvest-order filename stems so downstream scripts can align activations to label and bbox CSVs by stem rather than row order.
- **Bonferroni-corrected spatial grounding:** `universality_eval.run_spatial_grounding` uses `scipy.stats.pearsonr` and applies a family-wise α correction across confirmed latents (see `--bonferroni_alpha`).
- **Automated split detection:** `universality_eval.run_split_detection` runs the Jaccard co-activation check on confirmed latents and writes `split_pairs.json` + `co_activation_heatmap.png`.

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

Run end-to-end by chaining the four scripts below (the original runs used a cluster job scheduler for this). Every task corresponds to a function inside `src/scripts/universality_eval.py` unless noted.

| Task | Script / function | Output |
|---|---|---|
| 3.0 Scanner audit | `scripts/scanner_latent_audit.py` — multinomial LR probe per latent (balanced subsample) | `scanner_latent_audit.json` + `.csv` |
| 3.1 Level 1 frequency + canonical class match | `compute_frequency_per_dataset` + `compute_per_image_max` + `compute_dominant_canonical_class` + `apply_level1_filter` | `universal_candidates.csv` (`level1_pass`, `matched_canonical_class`) |
| 3.2 Level 2 centroid direction | `compute_centroids_per_dataset` + `apply_level2_filter` (cosine ≥ 0.70 all pairs) | `level2_pass`, `min_cosine_sim` columns; `confirmed_latent_frequencies.json` |
| 3.3 Cross-dataset probe | `run_cross_dataset_probe` — leave-one-out logistic regression, shares the per-image `feature_max` with Step 3.1 for identity consistency | Per-latent `{train}→{test}` AUROC columns |
| 3.4 Spatial grounding (Bonferroni) | `run_spatial_grounding` — `scipy.stats.pearsonr` + family-wise α correction | `{ds}_r` / `{ds}_p` / `{ds}_pass` + `pooled_r` / `pooled_p` / `pooled_pass` columns; `spatial_bonferroni` block in summary JSON |
| 3.5 Split detection (Property 4) | `run_split_detection` — Jaccard of top-K activating images across confirmed latents | `split_pairs.json`, `co_activation_heatmap.png`, `n_split_pairs` in summary |
| 3.6 Null baseline | `scripts/null_baseline_eval.py` — random SAE + PCA null, same L1/L2 gates | `null_baseline_results.json` |
| 3.7 Qualitative inspection | `scripts/inspect_universal_latents.py` — aim ≤15 confirmed latents | `latent_XXXXX/{semantic_grid, frequency_bar, patch_ablation}.png` + `index.md` |

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
