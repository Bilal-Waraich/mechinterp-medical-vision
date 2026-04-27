# Methodology: Illness-Stratified Evaluation of Universal SAE Latents

This document describes the complete methodology used to evaluate confirmed universal latents from the Matryoshka SAE against clinical illness categories. It covers the upstream universality confirmation pipeline, the illness subgrouping taxonomy, the discriminability probes, the spatial grounding procedure, and all relevant implementation decisions.

---

## 1. Prerequisites — Upstream Universality Confirmation

The illness-stratified evaluation operates on a set of **confirmed universal latents** produced by a prior two-level universality filter (`universality_eval.py`). Understanding that filter is necessary context because the confirmed set defines which latents enter the illness evaluation at all.

### 1.1 SAE Architecture and Encoding

The model is a **Matryoshka Sparse Autoencoder** trained on frozen patch-level activations extracted from a vision encoder (SwinTransformer) at layer 3. The encoder processes chest X-ray images resized to 224 × 224 pixels and produces a 7 × 7 = **49 patch tokens** per image, each of dimension `d_model`. The SAE maps each token into a sparse latent code of dimension `d_sae` using a learned encoder; only the top-*k* latents fire per token (BatchTopK variant) or via threshold-gated activation (Matryoshka variant).

Pre-harvested activations are stored as safetensors files — either one merged file per architecture (BatchTopK: `activations.safetensors` with a `dataset_ids` tensor distinguishing datasets) or four separate per-dataset files (Matryoshka). This run uses the **Matryoshka separate-file format**: `activations_{chexpert,rsna,siim_covid,tb}.safetensors`.

### 1.2 Level 1 — Activation Frequency Filter

For each latent `i` in the SAE's `d_sae`-dimensional latent space:

1. The SAE encoder is applied to all tokens in each dataset in batches of 4096.
2. The **token-level activation frequency** `freq_i^D` is computed as the fraction of tokens in dataset `D` for which latent `i` fires (has activation > 0).
3. A latent passes Level 1 if `freq_i^D ≥ 0.01` (1% threshold) on **at least 2 of the 4 datasets**.

This filter removes dead latents (which never fire) and dataset-specific latents (which fire only on one dataset). A 1% token-level frequency corresponds to a latent that activates on roughly 1 in 100 patches across an entire dataset, which is a conservative threshold designed to include any latent that carries even moderate signal.

Latents flagged by a prior scanner-artifact audit (which identifies latents that fire preferentially on specific scanner hardware rather than pathological content) are excluded from Level 1 candidates regardless of frequency.

**Output:** a list of *candidate* latent indices passing Level 1.

### 1.3 Level 2 — Representational Direction Alignment

Level 2 tests whether each candidate latent encodes the **same representational direction** across datasets, not just a similar frequency profile.

For each candidate latent `i` and each dataset `D` where it is active:

1. All tokens in dataset `D` where latent `i` fires are identified.
2. Their `d_model`-dimensional activation vectors (the input to the SAE, not the SAE codes) are averaged to produce a **d_model centroid** `c_i^D`.
3. For every pair of active datasets `(D_a, D_b)`, the **cosine similarity** `cos(c_i^{D_a}, c_i^{D_b})` is computed.
4. A latent is **confirmed universal** if and only if *all* pairwise cosine similarities exceed **0.70**.

The intuition is: if latent `i` reliably fires on conceptually similar image content across datasets (e.g. pleural effusion in CheXpert and in SIIM-COVID), the input space vectors that trigger it should point in roughly the same direction, even though the absolute scale may differ. The 0.70 cosine threshold means the centroids must agree in direction within ~45.6°.

**Output:** `confirmed_universal_latents.json` — a list of latent indices passing both levels. For this run: **252 confirmed latents** from a `d_sae`-dimensional latent space evaluated on 4 datasets.

---

## 2. Illness Taxonomy — Canonical Group Mapping

The four datasets use heterogeneous labelling conventions. To enable cross-dataset comparison, all raw labels are mapped to **7 canonical illness groups** based on the Fleischner Society radiological taxonomy (Hansell et al., 2008). This mapping is applied at harvest time so that each image receives a single canonical group index stored alongside its activations.

### 2.1 Canonical Groups

| Group index | Canonical name | Radiological meaning |
|-------------|---------------|---------------------|
| 0 | `airspace_opacity` | Consolidation, pneumonia, ground-glass opacity, lung opacity |
| 1 | `pleural_abnormality` | Pleural effusion, pleural thickening, pneumothorax (pleural origin) |
| 2 | `cardiac_mediastinal` | Cardiomegaly, enlarged cardiomediastinum |
| 3 | `atelectasis` | Partial or complete lung collapse |
| 4 | `vascular_edema` | Pulmonary edema of vascular/cardiac origin |
| 5 | `no_finding` | Radiologically normal study |
| 6 | `other` | Support devices, fractures, incidental findings |

### 2.2 Per-Dataset Label Mapping

**CheXpert** provides 14 pathology columns per image with ternary values: 1 (positive), 0 (negative), −1 (uncertain). The uncertain label is treated as negative (0) throughout. Because CheXpert is multi-label, a single dominant canonical group is assigned per image: the first positive column that maps to a canonical group is used, processing columns in the order they appear in the CSV. If no column is positive, the image is assigned `no_finding`.

Specific mappings:

| CheXpert label | Canonical group |
|---------------|----------------|
| Consolidation | airspace_opacity |
| Lung Opacity | airspace_opacity |
| Pleural Effusion | pleural_abnormality |
| Pleural Other | pleural_abnormality |
| Cardiomegaly | cardiac_mediastinal |
| Enlarged Cardiomediastinum | cardiac_mediastinal |
| Atelectasis | atelectasis |
| Edema | vascular_edema |
| No Finding | no_finding |
| Support Devices | other |
| Fracture | other |

**RSNA** is a binary dataset with a single `Target` column (1 = pneumonia, 0 = normal). Pneumonia maps to `airspace_opacity`; normal maps to `no_finding`. RSNA provides no multi-class labels, so the coarse 7-group discriminability probe cannot be computed for this dataset (all AUROC values are NaN).

**SIIM-COVID19** provides study-level labels with four mutually exclusive categories used as fine-grained labels:
- `Negative for Pneumonia` → `no_finding`
- `Typical Appearance` → `airspace_opacity` (typical COVID bilateral GGO)
- `Indeterminate Appearance` → `other`
- `Atypical Appearance` → `other`

At the coarse level, SIIM images labelled `Opacity` map to `airspace_opacity` and `No finding` to `no_finding`.

**TB Shenzhen** encodes class information in the filename: filenames containing `_p_` or `positive` are assigned `airspace_opacity` (TB-positive); all others are assigned `no_finding`. No fine-grained labels exist for this dataset.

---

## 3. Illness-Stratified Evaluation Pipeline

The evaluation is implemented in `src/scripts/illness_stratified_eval.py` and orchestrated as a Nomad batch job (`05-illness-stratified-eval-l3.nomad`) running in a Docker container with 1 GPU, 8 vCPUs, and 64 GB RAM.

The pipeline has four analyses: activation profiling, coarse discriminability probe, fine-grained discriminability probe, and spatial grounding.

### 3.1 Encoding

For each dataset, the pre-harvested activation tensor `[N_images × 49, d_model]` is passed through the SAE encoder in token-level batches of 4096:

```
z_patches[n, p, :] = SAE.encode(acts[n*49 + p])    # [N, 49, d_sae]
```

Only the 252 confirmed latent indices are retained from the full `d_sae`-dimensional code to reduce memory footprint. Two representations are derived:

- **`z_img`** `[N, 252]` — per-image feature vector computed as the **maximum activation across the 49 patches** for each confirmed latent. This max-pooling operation captures whether a latent fires anywhere in the image, which is appropriate for pathology features that may be localised to a subregion.
- **`z_patches`** `[N, 49, 252]` — full per-patch activations, retained for spatial grounding analysis.

All encoding is performed under `torch.no_grad()`.

### 3.2 Activation Profiles

For each confirmed latent `i`, each illness group `g`, and each dataset `D`, two statistics are computed over all images in group `g` within dataset `D`:

1. **Mean activation:** `mean(z_img[:, i])` over images in group `g` — the average peak activation of the latent within the illness group.
2. **Activation frequency:** `mean(z_img[:, i] > 0)` — the fraction of images in group `g` on which the latent fires at least once across any of its 49 patches.

These profiles are stored in `illness_activation_profiles.csv` with columns `[latent_idx, illness_group, dataset, mean_activation, activation_freq, n_images]`.

The illness latent matrix `[7 × 252]` stored in `illness_latent_matrix.csv` is derived from the activation profiles: each cell `(g, i)` contains the mean activation of latent `i` for illness group `g`, averaged across all datasets where the group has at least one image. The `top_k_latents_per_illness` in `illness_latent_summary.json` are then the top-10 latents per group ranked by this mean activation value.

**Important note:** The top-k ranking by raw mean activation is dominated by high-magnitude backbone latents that fire universally across all groups. It is not a discriminability ranking. The actual discriminability is measured by the probe below.

### 3.3 Analysis A1 — Coarse Discriminability Probe

The coarse probe measures how well a single confirmed latent's activation can discriminate between the 7 canonical illness groups.

**Setup.** For each confirmed latent `i` and each dataset `D` (excluding RSNA), the per-image max activations `z_img[:, i]` form a one-dimensional feature vector `X ∈ ℝ^N`. The corresponding canonical group index vector `y ∈ {0,...,6}^N` forms the label.

**Model.** A logistic regression classifier with balanced class weights (`class_weight='balanced'`), L-BFGS solver, one-vs-rest multi-class strategy (`multi_class='ovr'`), and maximum 500 iterations is fitted on `X.reshape(-1, 1)` (i.e. the single latent activation as the sole feature).

**Evaluation protocol.** Performance is estimated via **5-fold stratified cross-validation** with a fixed random seed (42). In each fold:

1. Only folds where all classes present in the test set have at least `MIN_CLASS_POSITIVES = 10` training samples are included; folds failing this check are skipped.
2. For binary problems (2 classes), AUROC is computed directly on the positive-class probability.
3. For multi-class problems (>2 classes), labels are binarised and **macro-averaged one-vs-rest AUROC** is computed using `roc_auc_score(..., average='macro', multi_class='ovr')`.

The reported `auroc_mean` and `auroc_std` are the mean and standard deviation of valid fold scores. If fewer than 2 valid folds exist or the dataset has fewer than 10 samples per class, the result is reported as NaN.

This yields one `(latent_idx, dataset, auroc_mean, auroc_std)` row per latent per dataset, stored in `illness_discriminability_probe_coarse.csv`.

**Interpretation.** An AUROC of 0.5 means the latent carries no illness-discriminative information. An AUROC of 1.0 means it perfectly separates illness groups. Because the feature is a single scalar (one latent's max activation), high AUROC values directly indicate that the latent's magnitude is linearly separable between groups.

### 3.4 Analysis A2 — Fine-Grained Discriminability Probe

The fine-grained probe repeats the same logistic regression procedure but with **per-dataset pathology labels** rather than the 7 coarse canonical groups. This allows testing whether individual latents discriminate specific clinical labels within each dataset.

**CheXpert (14 labels):** Each of the 14 pathology columns is treated as an independent binary classification problem. Uncertain values (−1) are excluded from training and evaluation (treated as NaN in the alignment matrix). The probe is run for each `(latent, label)` pair.

**SIIM-COVID19 (4 labels):** The four study-level categories (`Negative for Pneumonia`, `Typical Appearance`, `Indeterminate Appearance`, `Atypical Appearance`) are each treated as binary one-vs-rest targets.

**RSNA (1 label):** The binary `pneumonia` column is treated as a binary target.

**TB Shenzhen:** No fine-grained label CSV is provided; the fine probe is skipped.

Label alignment is performed by matching image file stems between the activation order and the label CSV. Images whose stem cannot be matched to the CSV are assigned NaN and excluded from training/evaluation.

Results are stored in `illness_discriminability_probe_fine.csv` with columns `[latent_idx, dataset, label_name, auroc_mean, auroc_std]`.

**Known data gap:** In this run, the fine probe produced NaN for all CheXpert and TB entries. This occurs when `image_stems` are not embedded in the safetensors metadata, preventing stem-to-label alignment. Only RSNA's binary `pneumonia` label has valid values, but these are near chance (AUROC ≈ 0.49–0.51), consistent with the coarse probe finding that RSNA's NaN coarse labels prevented meaningful training signal.

### 3.5 Analysis B — Illness-Stratified Spatial Grounding

Spatial grounding tests whether a latent's **within-image activation pattern** correlates with the anatomical location of the pathology, as annotated by bounding boxes.

**Datasets with bounding boxes:** RSNA (`stage_2_train_labels.csv`, format: `x, y, width, height`) and SIIM-COVID19 (`train_image_level.csv`, format: `x y width height` space-separated in a `boxes` column). Only images where `Target=1` (RSNA) or a non-empty, non-"none" bounding box string (SIIM) are included.

**Patch-to-bbox overlap computation.** For each image and each of the 49 patches (arranged in a 7 × 7 spatial grid over the 224 × 224 pixel image, so each patch covers a 32 × 32 pixel region):

1. The patch's pixel coordinates `(px1, py1, px2, py2)` are computed from its grid position.
2. The intersection area with each annotated bounding box is computed as `max(0, min(px2, box.x2) - max(px1, box.x1)) × max(0, min(py2, box.y2) - max(py1, box.y1))`.
3. The fractional patch coverage = total intersection area / patch area, clamped to [0, 1].
4. Multiple bounding boxes per image have their overlaps summed before clamping.

This produces a tensor `bbox_overlap[N, 49]` with values in [0, 1] indicating how much of each patch is covered by an annotated disease region.

**Stratified Pearson correlation.** For each confirmed latent `i`, illness group `g`, and dataset `D`:

1. Images in group `g` within dataset `D` are selected (group determined by the canonical label stored in the activations).
2. The per-patch activation vector `z_patches[g_images, :, i]` is flattened to shape `[n_g × 49]`.
3. The corresponding `bbox_overlap[g_images, :]` is flattened to the same shape.
4. If either vector has near-zero standard deviation (< 1e-9), the correlation is NaN.
5. Otherwise, the **Pearson correlation coefficient** `r` and its p-value are computed using `scipy.stats.pearsonr`.
6. Groups with fewer than 5 images are skipped (result is NaN).

Results are stored in `illness_spatial_grounding.csv` with columns `[latent_idx, illness_group, dataset, pearson_r, pval, n_images]`.

A positive Pearson r indicates the latent fires more strongly at patches that overlap with annotated disease regions — direct evidence of spatial pathology sensitivity.

**Known data gap:** In this run, all `pearson_r` values are NaN despite 504 entries having `n_images > 0`. The root cause is that `image_stems` were not embedded in the safetensors metadata for the Matryoshka activation files, so the spatial grounding code fell back to generating sequential placeholder stems (`["0", "1", "2", ...]`) that do not match any keys in the bounding box dictionary. This means `bbox_overlap` is all zeros for every image, and the standard deviation of `bbox_overlap` is below the 1e-9 threshold, producing NaN.

---

## 4. Output Files

| File | Contents |
|------|----------|
| `illness_eval_summary.json` | Run metadata: timestamp, elapsed time, n confirmed latents, n patches, datasets, warnings |
| `illness_latent_matrix.csv` | `[7 × 252]` matrix of mean activation per illness group per confirmed latent, averaged across datasets |
| `illness_latent_summary.json` | Top-10 latents per illness group by mean activation; list of all confirmed indices |
| `illness_activation_profiles.csv` | Per-(latent, illness_group, dataset): mean activation, activation frequency, n images |
| `illness_discriminability_probe_coarse.csv` | Per-(latent, dataset): 5-fold CV macro AUROC for 7-group illness classification |
| `illness_discriminability_probe_fine.csv` | Per-(latent, dataset, label): 5-fold CV binary AUROC for fine-grained pathology labels |
| `illness_spatial_grounding.csv` | Per-(latent, illness_group, dataset): Pearson r between patch activation and bbox overlap |

---

## 5. Compute Infrastructure

The job runs as a Nomad batch task (`xray-analysis-05-illness-stratified-eval-l3`) within a Docker container on the Fraunhofer MeVis cluster:

| Resource | Allocation |
|----------|-----------|
| GPU | 1 × NVIDIA GPU (cuda:0) |
| CPU | 8000 MHz (8 vCPUs effective) |
| RAM | 64 GB |
| Encoding batch size | 4096 tokens |
| Logistic regression parallelism | `n_jobs=-1` (all available CPUs) |
| Total elapsed time | 346 seconds (~5.8 minutes) |

Both architectures (BatchTopK and Matryoshka) are evaluated in the same job sequentially. This run analyzes the Matryoshka L3 checkpoint at step 200,000.

---

## 6. Design Decisions and Limitations

### 6.1 Max-pooling as image-level feature

Per-image max activation across 49 patches is used as the discriminability feature rather than mean activation. Max-pooling is preferred because pathological findings are often localised — a consolidation may appear in only 2–3 of 49 patches. Mean activation would dilute the signal. The max operation ensures that a latent firing strongly in even one patch is captured at the image level.

### 6.2 Single-feature logistic regression

Each latent is evaluated individually as a single scalar feature. This makes AUROC directly interpretable as the discriminative information carried by one latent in isolation, without confounding from other latents. It also means the AUROC lower-bounds what a multi-latent classifier could achieve.

### 6.3 Balanced class weights

The `class_weight='balanced'` setting in the logistic regression compensates for the large class imbalance in CheXpert and SIIM (e.g. `no_finding` and `airspace_opacity` are much more frequent than `cardiac_mediastinal`). Without balancing, the classifier would trivially predict the majority class and achieve artificially high accuracy.

### 6.4 Uncertain CheXpert labels treated as negative

CheXpert's uncertain (−1) labels are mapped to 0 (negative) for the coarse group assignment and excluded from fine-probe evaluation. This is conservative: treating uncertain labels as negative potentially misclassifies some true positives, which would suppress the measured AUROC. The alternative (treating uncertain as positive) would inflate it.

### 6.5 Dominant-label strategy for multi-label CheXpert

CheXpert is inherently multi-label, but the coarse probe requires a single group per image. Taking only the first positive label introduces ambiguity for images with multiple conditions. A more principled approach would be to assign each image to all applicable groups and run per-group binary probes (similar to what the fine probe does), but this is deferred given the scope of the analysis.

### 6.6 Missing image stems in Matryoshka activation files

The Matryoshka activation harvesting pipeline does not embed `image_stems` metadata into the safetensors files, unlike the BatchTopK pipeline. This affects two analyses:
- **Fine-grained probe:** Label CSV alignment fails silently; all rows become NaN.
- **Spatial grounding:** Bounding box lookup fails; all `pearson_r` become NaN.

This is a known limitation of this run. Both analyses require re-harvesting with `image_stems` embedded in metadata, or post-hoc stem reconstruction from the dataset order.

---

## 7. Relationship to Matryoshka SAE Properties

The illness evaluation was designed to test a specific hypothesis about Matryoshka SAEs: that the nested training objective, which encourages meaningful feature hierarchies at multiple sparsity levels, produces latents that are both universal across datasets *and* semantically aligned with clinical categories. The two-level universality filter (Section 1) establishes cross-dataset universality as a prerequisite; the illness evaluation then probes whether these universal latents carry clinical discriminative information.

The finding that a subset of universal latents achieves AUROC > 0.80 on SIIM-COVID (a dataset the SAE was not explicitly trained to separate) provides evidence that the learned representations are not just dataset-level features but encode genuine pathological signal. The bipolar structure observed — some universal latents suppressed by pathology, others activated by it — mirrors the feature hierarchy expected from the Matryoshka nested objective, where coarser features (global healthy/sick state) are expected to dominate the lower-sparsity regime and finer disease-specific features emerge at higher sparsity.
