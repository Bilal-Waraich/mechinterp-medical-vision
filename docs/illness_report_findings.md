# Illness Report — Analysis Findings
**Checkpoint:** `matryoshka/layer-3/xray-conglomerate/xray_v1/step_0200000`  
**Confirmed universal latents:** 252 across 4 datasets (CheXpert, RSNA, SIIM-COVID, TB)  
**Analysis date:** 2026-04-27

---

## Overview

The illness evaluation probes 252 confirmed universal latents from the Matryoshka SAE against 7 coarse illness groups and up to 19 fine-grained diagnostic labels across four chest X-ray datasets. The latents fall into four functional tiers based on their activation profiles and discriminability (AUROC).

**Key result:** The SAE has learned a clean functional hierarchy — backbone/domain features, healthy-lung detectors, pathology-presence detectors, and disease-specific features — without any explicit supervision for these categories.

---

## Tier 1 — Backbone / Always-On Latents (non-discriminative)

Latents **3512** and **2640** are the most activated features in the entire confirmed set, yet they carry **zero diagnostic information**. Both fire at 100% activation frequency across every illness group and every dataset, with nearly flat activation values (~700 and ~650 respectively).

Despite appearing as `#1` and `#2` in the `top_k_latents_per_illness` for all seven illness groups, their coarse AUROC is ~0.56 — barely above chance. This is because the top-k ranking uses raw mean activation, which is dominated by these globally active features.

Four additional latents — **3321, 1969, 1084, 3832** — appear in the top-10 by activation for all 7 illness groups and are equally non-discriminative. These six latents likely encode domain-level features such as "this is a chest X-ray image," acting as a bias or intercept term in the latent space.

> **Implication for the thesis:** The `top_k_latents_per_illness` metric in the summary JSON is misleading when used to identify clinically meaningful features. Discriminability-based ranking (AUROC) reveals a completely different and more interpretable set of latents.

---

## Tier 2 — Healthy-Lung Detectors

These latents fire most strongly for healthy (`no_finding`) images and are suppressed when pathology is present. They encode a **"normal lung" signal** rather than any specific disease.

### Latent 818 — Universal Healthy-Lung Feature

| Dataset | Coarse AUROC |
|---------|-------------|
| CheXpert | 0.631 |
| SIIM-COVID | 0.830 |
| TB | 0.664 |

**Activation profile:**
- `no_finding`: activation ~450–554, frequency **98%** across all three datasets
- Sick groups: activation ~180–290, frequency drops to 70–88%

Latent 818 reliably fires on nearly every healthy image at high intensity, then drops significantly when any pathology is present. Its consistent performance across CheXpert, SIIM-COVID, and TB establishes it as a **genuinely universal healthy-lung feature**. It likely encodes normal lung parenchymal texture or aeration pattern.

### Latent 1384 — Strongest Universal Discriminator

| Dataset | Coarse AUROC |
|---------|-------------|
| CheXpert | 0.622 |
| SIIM-COVID | **0.843** (highest of all latents) |
| TB | 0.634 |

**Activation profile:**
- `no_finding`: activation ~101–225, frequency 85–96%
- Sick groups: activation ~63–120, frequency drops to 52–74%

Latent 1384 achieves the single highest AUROC in the entire analysis (0.843 on SIIM-COVID), and ranks #1 on both SIIM-COVID and TB. The SIIM-COVID result is particularly striking — this latent cleanly separates COVID-positive from COVID-negative chest X-rays using only its activation magnitude. Like 818, it fires preferentially for healthy lungs and is suppressed by pathology.

### Latent 884 — Secondary Healthy-Lung Feature

| Dataset | Coarse AUROC |
|---------|-------------|
| CheXpert | 0.596 |
| SIIM-COVID | 0.740 |
| TB | 0.529 |

Similar profile to 818 (97% frequency for `no_finding`, drops for sick groups) but weaker on TB, suggesting partial dataset specificity.

---

## Tier 3 — Pathology-Presence Detectors

These latents have the **opposite polarity** to Tier 2 — they fire strongly when pathology is present and are suppressed for healthy images. They encode "something is wrong" rather than identifying which condition specifically.

### Latent 2682 — Airspace/Pleural Pathology Detector

| Dataset | Coarse AUROC |
|---------|-------------|
| CheXpert | 0.579 |
| SIIM-COVID | 0.779 |
| TB | **0.668** (top TB latent) |

**Activation profile:**
- `no_finding`: activation 8.8–42.7, frequency only **18–32%** (barely fires on healthy images)
- Airspace opacity: activation 111–211, frequency 47–70%
- Pleural/vascular also elevated

This latent is the inverse of 818. It fires on fewer than one-third of healthy images, making it a sensitive indicator of pathological changes. The top z-score standout for `airspace_opacity` (Δ=81.1 above other illness groups). Ranking #3 on SIIM-COVID and #1 on TB confirms cross-dataset universality.

### Latent 3348 — Pleural/Fluid-Related Pathology

| Dataset | Coarse AUROC |
|---------|-------------|
| CheXpert | **0.642** (top CheXpert latent) |
| SIIM-COVID | 0.683 |
| TB | 0.613 |

Z-score standout for:
- `pleural_abnormality` (Δ=219.8 — the single largest standout in the dataset)
- `atelectasis` (Δ=106.6)
- `vascular_edema` (Δ=124.5)

Latent 3348 likely encodes fluid-related or density-related lung changes. The shared activation across pleural, atelectasis, and vascular conditions is clinically coherent — all three involve increased radiological density.

### Latent 2619 — Cardiac/Vascular Pathology

| Dataset | Coarse AUROC |
|---------|-------------|
| CheXpert | 0.614 |
| SIIM-COVID | 0.708 |
| TB | 0.522 |

Z-score standout for:
- `cardiac_mediastinal` (Δ=185.3)
- `vascular_edema` (Δ=147.6)

Peaks for conditions involving the heart and major vessels. The low TB AUROC (0.522) makes sense — TB primarily affects lung parenchyma, not cardiac structures. This latent likely encodes cardiomegaly or mediastinal widening features.

---

## Tier 4 — Illness-Specific Standout Latents

The following latents show the highest z-score differential for a specific illness group relative to all others. These are the most interpretable disease-specific features.

| Latent | Peak illness | Δ vs. others | Secondary illness | Interpretation |
|--------|-------------|-------------|-----------------|----------------|
| 818 | `no_finding` | +254.5 | — | Healthy lung texture |
| 3348 | `pleural_abnormality` | +219.8 | atelectasis (+106.6) | Pleural/fluid density |
| 2619 | `cardiac_mediastinal` | +185.3 | vascular_edema (+147.6) | Cardiomegaly/vasculature |
| 884 | `no_finding` | +125.4 | — | Secondary healthy feature |
| 3348 | `vascular_edema` | +124.5 | — | Fluid-related changes |
| 2682 | `airspace_opacity` | +81.1 | — | Consolidation/airspace |
| 1781 | `no_finding` | +111.5 | airspace_opacity | Borderline aeration |
| 2185 | `pleural_abnormality` | +80.8 | — | Pleural-specific |
| 3569 | `pleural_abnormality` | +90.7 | atelectasis (+42.3) | Pleural feature |
| 1237 | `pleural_abnormality` | +76.6 | atelectasis (+58.2) | Shared pleural-atelectasis |

---

## Cross-Illness Structural Patterns

### Pleural Abnormality ↔ Atelectasis Overlap

Latents **3348** and **1237** are top standouts for *both* pleural_abnormality and atelectasis. This is clinically expected — atelectasis (lung collapse) causes increased density that overlaps with pleural thickening and effusion, and the two conditions frequently co-occur. The SAE has recovered this pathophysiological relationship in its latent structure without any explicit multi-label training.

### Cardiac Mediastinal ↔ Vascular Edema Overlap

Latents **2619** and **215** are top standouts for *both* cardiac_mediastinal and vascular_edema. This reflects a well-established clinical relationship: left-sided cardiac failure simultaneously causes cardiomegaly (enlarging the mediastinum) and pulmonary edema (visible as vascular congestion). The SAE independently learned this co-occurrence pattern.

---

## Dataset Behavior

| Dataset | Best latent | Best AUROC | Notes |
|---------|------------|-----------|-------|
| CheXpert | 3348 | 0.642 | Multi-label; all 252 latents have valid AUROC |
| SIIM-COVID | 1384 | 0.843 | Binary task; sharpest discrimination overall |
| TB | 2682 | 0.668 | Smaller dataset; higher AUROC variance |
| RSNA | — | NaN | Binary task; coarse probe N/A (requires multi-class labels) |

SIIM-COVID consistently produces the highest AUROC values, likely because COVID pneumonia has distinctive bilateral ground-glass opacity patterns that are less ambiguous than the overlapping multi-label presentations in CheXpert.

---

## Data Gaps and Pipeline Issues

### Spatial Grounding — All NaN

The `illness_spatial_grounding.csv` file contains 504 entries with `n_images > 0` (images were processed, up to 4,598 for SIIM-COVID), but **all `pearson_r` values are NaN**. The correlation between patch-level latent activations and anatomical disease regions was never successfully computed.

This is likely a bug in the spatial grounding pipeline — possibly a failure to load segmentation masks, a shape mismatch between activation maps and mask tensors, or a silent exception during the correlation computation.

**Impact:** No spatial localization claims can be made. This is a notable limitation for the interpretability argument.

### Fine-Grained Label Probe — All NaN (except RSNA)

The `illness_discriminability_probe_fine.csv` contains per-label AUROC for 19 diagnostic labels (Atelectasis, Cardiomegaly, Edema, etc.) but **all CheXpert and TB values are NaN**. Only RSNA's binary `pneumonia` label has values, and these are uniformly ~0.49–0.51 (chance level), confirming the fine probe did not produce usable output.

**Impact:** Cannot map latents to specific CheXpert label categories (e.g. distinguish "Edema" from "Pleural Effusion" at the fine level).

### RSNA Coarse Probe — All NaN

RSNA is a binary dataset (pneumonia / not-pneumonia) with no multi-class illness group labels, so the coarse illness-group probe produces no valid AUROC for this dataset.

---

## Summary

The Matryoshka SAE at layer 3, step 200k, has learned a structured and clinically interpretable decomposition of chest X-ray pathology:

1. **6 backbone latents** (led by 3512, 2640) fire universally and encode domain identity rather than disease.
2. **3 healthy-lung latents** (818, 1384, 884) cleanly detect normal lung state, with 1384 reaching AUROC=0.843 on SIIM-COVID.
3. **3 pathology-presence latents** (2682, 3348, 2619) detect the presence of abnormality with some condition specificity.
4. **Cross-condition sharing** of latents between pleural/atelectasis and cardiac/vascular groups mirrors known clinical co-occurrence patterns.
5. **Spatial grounding and fine-label probing failed** (pipeline NaN), leaving spatial interpretability and fine-grained label mapping as open questions.

The clearest result for the thesis is that the SAE recovered a **bipolar healthy↔pathological encoding** across datasets without supervision, with the healthy-lung and pathology-presence latents forming complementary pairs. This supports the argument that Matryoshka SAEs can learn hierarchically meaningful and clinically coherent features from medical image encoders.
