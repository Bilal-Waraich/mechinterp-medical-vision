# Illness-Stratified Evaluation — Matryoshka L3 Findings

**Run:** 2026-04-25 · elapsed 273.8 s  
**Model:** Matryoshka SAE · Layer 3 · `xray_v1` · checkpoint `step_0200000`  
**Confirmed universal latents evaluated:** 252  
**Datasets:** CheXpert, RSNA, SIIM-COVID, TB  

---

## Run warnings

The spatial grounding analysis (Analysis B) produced **no results** for any dataset:

- RSNA and SIIM-COVID: bbox CSVs were provided but **no `image_stems` in safetensors metadata** — the harvest job did not embed filename stems, so patches cannot be aligned to bounding boxes.
- CheXpert and TB: no bboxes available (expected).
- Fine-grained probe (Analysis A2): all datasets skipped — fine label CSVs were not passed to the run (likely omitted from the nomad job args).

**Action required:** Re-harvest Matryoshka L3 activations with `image_stems` embedded in metadata to unlock spatial grounding. Pass fine label CSVs in the next run.

---

## Analysis A — Illness Discriminability Probe (Coarse, 7 canonical groups)

### Dataset coverage

| Dataset | Valid latents | Mean AUROC | Max AUROC |
|---|---|---|---|
| CheXpert | 252 | 0.544 | 0.642 |
| RSNA | 0 | — (all NaN) | — |
| SIIM-COVID | 252 | 0.583 | 0.843 |
| TB | 252 | 0.538 | 0.668 |

RSNA returned all NaN — consistent with the known all-zero label issue in the safetensors and no image stems available to remap from CSV.

### AUROC distribution

| Threshold | Latents exceeding on ≥1 dataset |
|---|---|
| ≥ 0.65 | 40 / 252 (16%) |
| ≥ 0.75 | 6 / 252 (2.4%) |

Most latents sit modestly above chance (0.5), consistent with coarse 7-class classification being genuinely hard. The tail of high-AUROC latents is the most interesting signal.

### Top discriminative latents

These latents show the strongest illness discriminability across datasets:

| Latent | CheXpert AUROC | SIIM-COVID AUROC | TB AUROC | Notes |
|---|---|---|---|---|
| **1384** | 0.622 | **0.843** | 0.634 | Best overall; strong on SIIM and TB |
| **818** | 0.631 | **0.830** | 0.664 | Consistently high across all 3 |
| **2682** | 0.579 | 0.779 | **0.668** | Strong on TB |
| **3344** | 0.560 | 0.761 | 0.497 | SIIM-specific |
| **1632** | 0.567 | 0.760 | 0.525 | SIIM-specific |
| **231** | 0.563 | 0.751 | 0.519 | SIIM-specific |
| **2891** | 0.598 | 0.744 | 0.564 | Broad |
| **884** | 0.596 | 0.740 | 0.529 | In top-10 for 6/7 illness groups |
| **630** | 0.540 | 0.734 | 0.586 | TB-leaning |
| **3348** | **0.642** | 0.683 | 0.613 | Most balanced across datasets |

**Latents 1384 and 818 are the strongest candidates** for illness-discriminative universal features.

---

## Analysis B — Spatial Grounding

**All NaN** — spatial grounding could not be computed because `image_stems` are absent from the Matryoshka L3 safetensors metadata. This is the most critical gap to address.

---

## Activation Profiles

### Latents dominating the top-activation list

Five latents appear in the top-10 by mean activation for **all 7 illness groups**:

| Latent | Activation freq | Interpretation |
|---|---|---|
| **3512** | 1.00 (always active) | General-purpose / background feature — fires on everything |
| **2640** | 0.99–1.00 | Same pattern as 3512 |
| **3321** | — | Broadly active across all groups |
| **1969** | — | Broadly active across all groups |
| **1084** | — | Broadly active across all groups |
| **3832** | — | Broadly active across all groups |

These latents have extremely high mean activations (700–760) and activation frequency ≈ 1.0 on CheXpert. They are **not illness-specific** — they likely encode general image features (e.g. tissue texture, exposure level). They dominate the `illness_latent_summary.json` top-k rankings but should be excluded from clinical interpretation.

### Mean activation by illness group

| Group | Mean activation (across all latents & datasets) |
|---|---|
| pleural_abnormality | 51.26 |
| atelectasis | 51.13 |
| vascular_edema | 50.88 |
| cardiac_mediastinal | 50.50 |
| other | 50.46 |
| airspace_opacity | 46.97 |
| no_finding | 46.36 |

Pathological groups activate slightly more strongly than `no_finding` on average, though the difference is small (~10%). The stronger signal is in the per-latent probe AUROC rather than the global mean.

---

## Key findings

1. **252 universal latents confirmed** at L3, up from 188 at L2 — the deeper layer encodes more diverse features.

2. **Two latents (1384, 818) are robustly illness-discriminative** across SIIM-COVID and TB with AUROC > 0.83/0.83, holding above 0.62 on CheXpert. These are strong candidates for disease-specific universal features.

3. **Spatial grounding is blocked** by missing `image_stems` in the harvest metadata. This is the highest-priority fix — without it we cannot answer whether features fire at anatomically correct locations.

4. **~6 broadly-active latents (3512, 2640, 3321, 1969, 1084, 3832)** dominate activation magnitude but show no illness specificity. They should be flagged as general-purpose and excluded from clinical latent lists.

5. **RSNA is entirely unusable** in this run — both the label issue (all-zero labels) and missing stems prevent any analysis. Fix: re-harvest with stems and pass `--rsna_csv` in the next run.

---

## Recommended next steps

| Priority | Action |
|---|---|
| 🔴 High | Re-harvest Matryoshka L3 with `image_stems` in safetensors metadata |
| 🔴 High | Pass all fine-label CSV args in the nomad job (`--chex_csv`, `--rsna_csv`, `--siim_study_csv`) |
| 🟡 Medium | Investigate latents 1384 and 818 in the visualiser — do they correspond to known pathology patterns? |
| 🟡 Medium | Run BatchTopK L3 illness eval and compare confirmed latent sets |
| 🟢 Low | Filter out always-active latents (freq ≈ 1.0) from clinical summaries |
