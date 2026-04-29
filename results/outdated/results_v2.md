# Illness-Stratified Evaluation — Layer 3 BatchTopK (xray_v2)

**Run date:** 2026-04-29  
**Elapsed:** 84.3 s  
**Checkpoint:** `batch-topk/layer-3/xray-conglomerate/xray_v2/checkpoints/step_0200000`  
**Confirmed universal latents:** 65 (out of d_sae = 4096)  
**Datasets:** CheXpert (10 000 img), RSNA (10 000 img), SIIM-COVID (6 334 img), TB (662 img)  
**Script version:** `illness_stratified_eval.py` with stem-alias fix (image_stems now resolved for all datasets)

---

## 1. Activation Profiles — Pathology-Preference Ratio (rho)

rho = mean activation on sick images / mean activation on no-finding images, averaged across datasets.

| Tier | Criterion | Count | Examples |
|------|-----------|-------|---------|
| T1 — strong pathology preference | rho >= 3.0 | **10** | 1088, 3797, 3488, 3868, 589, 694, 2974, 2587, 4013, 1777 |
| T2 — moderate preference | 2.0 <= rho < 3.0 | **12** | 57, 264, 1288, 1411, 1433, … |
| T3 — weak preference | 1.5 <= rho < 2.0 | **8** | 326, 827, 1219, 1734, 1914, … |
| T4 — non-specific | rho < 1.5 | **35** | 8, 103, 130, 211, 380, … |

Top-10 latents by rho:

| Latent | rho |
|--------|-----|
| 1088 | 5.18 |
| 3797 | 4.80 |
| 3488 | 4.08 |
| 3868 | 3.94 |
| 589  | 3.79 |
| 694  | 3.54 |
| 2974 | 3.39 |
| 2587 | 3.38 |
| 4013 | 3.14 |
| 1777 | 3.10 |

**Interpretation:** 22 latents (T1+T2, 34%) show meaningful pathology preference. 35 latents (54%) are non-specific across illness groups.

### Per-group top latents (mean activation)

| Illness group | Top-1 | Top-2 | Top-3 |
|---------------|-------|-------|-------|
| airspace_opacity | lat 2784 (22.1) | lat 57 (22.0) | lat 2257 (20.0) |
| pleural_abnormality | lat 57 (47.1) | lat 3399 (42.4) | lat 3488 (41.9) |
| cardiac_mediastinal | lat 3399 (49.8) | lat 3590 (36.3) | lat 2784 (32.8) |
| atelectasis | lat 3399 (48.0) | lat 57 (41.2) | lat 2784 (40.1) |
| vascular_edema | lat 3399 (44.9) | lat 2784 (41.9) | lat 1288 (36.8) |
| no_finding | lat 3590 (22.0) | lat 3538 (20.5) | lat 3399 (20.3) |

**Latent 3399** dominates pleural, cardiac, atelectasis, and vascular groups — likely a general pathology detector rather than a specific one. **Latent 57** leads airspace_opacity and is strong in atelectasis/pleural. **Latent 3590** is the strongest no-finding latent.

---

## 2. Analysis A1 — Coarse Illness Discriminability Probe (7-group AUROC)

5-fold stratified logistic regression (1 feature = per-latent max-pooled activation), macro OvR AUROC.

| Dataset | Mean AUROC | Max AUROC | Latents > 0.60 | Latents > 0.65 |
|---------|-----------|----------|----------------|----------------|
| CheXpert | 0.538 | 0.592 (lat 1288) | 0 | 0 |
| RSNA | 0.563 | 0.666 (lat 4087) | 4 | 1 |
| SIIM-COVID | 0.546 | 0.621 (lat 981) | 2 | 0 |
| TB | 0.535 | 0.608 (lat 57) | 1 | 0 |
| **Overall** | **0.578** | **0.666** | **16** | **1** |

Top-5 per dataset:

- **CheXpert:** lat 1288 (0.592), lat 2587 (0.583), lat 3488 (0.580), lat 589 (0.578), lat 2989 (0.577)
- **RSNA:** lat 4087 (0.666), lat 3538 (0.632), lat 2257 (0.630), lat 2325 (0.627), lat 1576 (0.621)
- **SIIM-COVID:** lat 981 (0.621), lat 1576 (0.611), lat 4087 (0.608), lat 130 (0.601), lat 3952 (0.597)
- **TB:** lat 57 (0.608), lat 2784 (0.592), lat 4087 (0.585), lat 827 (0.583), lat 2257 (0.579)

**Interpretation:** Discriminability is modest. Only latent 4087 exceeds AUROC 0.65, and only on RSNA. The near-chance scores on CheXpert and TB suggest the coarse 7-group label is too coarse for single-latent linear probing, or the 65 confirmed latents primarily encode image-level texture/quality rather than explicit pathology categories.

**Latent 4087** appears in the top-3 for RSNA, SIIM-COVID, and TB — the most cross-dataset discriminative single latent.

---

## 3. Analysis A2 — Fine-Grained Pathology Probe

### RSNA (binary pneumonia label)

RSNA fine results are identical to coarse AUROC (binary task, same data). Best: lat 4087 (0.666), 14 latents > 0.60, 1 > 0.65.

### CheXpert (14 pathology labels) — DATA QUALITY ISSUE

All CheXpert fine probe entries are NaN. Root cause: at harvest time, all 10 000 CheXpert image stems were stored as `"view1_frontal"` (non-unique — the harvest script used only the filename, not the patient/study path). The `align_fine_labels` function cannot distinguish between images. **This is a harvest-level bug.**

Fix required: re-harvest CheXpert activations storing full relative paths as stems (e.g. `patient00001/study1/view1_frontal`) and rerun.

### SIIM-COVID (4-class study labels) — DATA QUALITY ISSUE

All SIIM-COVID fine probe entries are NaN. Stems in the safetensors are 12-character image-level hex IDs (e.g. `000a312787f2`), but the study-level CSV uses study instance UIDs. Image-to-study mapping is missing.

Fix required: map image-level stems to study IDs at harvest time (or pass a side-file mapping image stem → study UID during eval).

### TB

No fine-grained labels available (binary healthy/sick only, which is the coarse label). Skipped by design.

---

## 4. Analysis B — Spatial Grounding (Pearson r, patch activation vs. bbox overlap)

Only RSNA and SIIM-COVID have bounding box annotations. SIIM spatial grounding was skipped because image_stems were present but bbox matching requires per-image IDs.

For RSNA, only the **airspace_opacity** illness group had sufficient images for computation:

| Dataset | Group | n latents | Mean Pearson r | Max Pearson r |
|---------|-------|-----------|---------------|--------------|
| RSNA | airspace_opacity | 65 | -0.004 | 0.036 |

All other RSNA illness groups produced NaN (too few images with bbox annotations in those groups).

**Interpretation:** Near-zero Pearson r for RSNA/airspace_opacity indicates the confirmed universal latents do **not** preferentially activate at pneumonia bbox locations. The max-pooled activation is driven by global image features rather than spatially localised pathology patches. This is expected for latents that capture image-level patterns, and does not contradict universality — but it means these latents are not spatially grounded.

---

## 5. Summary and Outstanding Issues

### What works
- Coarse illness discriminability probe: all 4 datasets fully evaluated, 65 latents
- RSNA fine probe: binary pneumonia label, full results
- Activation profiles: all 4 datasets, rho-based tier classification complete
- Stem resolution fix: all 4 datasets now correctly resolve image stems from safetensors metadata (prior run had 0 stems for CheXpert/SIIM)

### Key findings
- **34% of confirmed latents (22/65)** show meaningful pathology preference (rho >= 2.0)
- **Coarse discriminability is modest** — peak AUROC 0.666, mean 0.578. Single-latent linear probing barely separates the 7 canonical illness groups.
- **Latent 3399** is the dominant pathology-active latent across pleural, cardiac, atelectasis, and vascular groups
- **Latent 4087** is the most cross-dataset discriminative latent (top-3 on 3/4 datasets)
- **No spatial grounding** detected via Pearson r — latents are not patch-localised to pathology bounding boxes

### Remaining data quality issues

| Issue | Impact | Fix |
|-------|--------|-----|
| CheXpert harvest stores non-unique stems (`view1_frontal`) | CheXpert fine probe entirely broken (0 valid AUROCs) | Re-harvest storing `patient/study/filename` as stem |
| SIIM image stems vs. study-level CSV IDs mismatch | SIIM fine probe entirely broken | Map image stem → study UID at harvest or provide side-file |
| SIIM spatial grounding skipped | No bbox spatial analysis for COVID | Requires image-level stem-to-bbox alignment |

### Top confirmed latents for thesis

| Rank | Latent | Notable property |
|------|--------|-----------------|
| 1 | 3399 | Highest mean activation across pleural/cardiac/atelectasis/vascular (general pathology hub) |
| 2 | 57 | Strongest airspace_opacity latent; top-1 TB coarse |
| 3 | 4087 | Highest AUROC (0.666 RSNA), cross-dataset discriminative |
| 4 | 1088 | Highest rho=5.18 (strongest sick vs. healthy preference) |
| 5 | 3590 | Strongest no-finding latent (baseline/healthy activation) |
