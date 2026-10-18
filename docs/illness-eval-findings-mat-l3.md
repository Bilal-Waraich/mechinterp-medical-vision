# Illness-Stratified Evaluation — Matryoshka L3 Findings

> **Note:** the cluster job files (`nomad/...`) referenced in this document are not part of the public repository. The scripts they call are in `src/`; see the README for how to run them.

**Universality run:** 2026-04-28 · 5-stage Matryoshka pipeline  
**Illness eval run:** 2026-04-27 · elapsed 346 s  
**Model:** Matryoshka SAE · Layer 3 · `xray_v1` · checkpoint `step_0200000`  
**Confirmed universal latents:** 252 (validated against fresh 5-stage universality run — 252/252 match)  
**Datasets:** CheXpert, RSNA, SIIM-COVID, TB

---

## Universality funnel (5-stage pipeline)

| Stage | Description | Latents remaining |
|-------|-------------|------------------|
| Total SAE latents | d_sae | 4096 |
| Stage 0 — Scanner audit | Exclude latents predicting dataset-of-origin (AUROC > 0.80) | 4096 (0 flagged) |
| Stage 1 — Activation frequency | Fire ≥ 1% of tokens on ≥ 2 datasets | 863 candidates |
| Stage 2 — Semantic consistency | Pairwise centroid cosine ≥ 0.70 across active datasets | 279 confirmed |
| Stage 3 — Split deduplication | Remove Jaccard-redundant split pairs, keep ancestors | 279 (0 pairs found) |
| Stage 4 — Prefix stability | Adjacent-scale Jaccard ≥ 0.40 across k = 6→12→24→48 | **252 universal** |

---

## Known data gaps in this run

- **Spatial grounding (Analysis B): all NaN.** Bbox CSVs were provided but `image_stems` are absent from the Matryoshka L3 safetensors metadata — patch-to-bbox alignment is not possible. Fix: re-harvest with stems embedded.
- **Fine-grained probe (Analysis A2): all NaN** for CheXpert and TB. Label CSV paths were not passed to the nomad job; only RSNA's binary pneumonia label has values, all near chance (AUROC ≈ 0.50). Fix: pass `--chex_csv`, `--siim_study_csv` in the next run.
- **RSNA coarse probe: all NaN.** RSNA is a binary dataset with no multi-class illness group structure; the 7-group coarse probe cannot be applied.

---

## Latent taxonomy

### Tier 1 — Backbone / always-on (6 latents)

These appear in the top-10 by raw mean activation for **all 7 illness groups** and are therefore non-discriminative. They fire at ≥ 99% frequency with near-flat activation across every group and dataset. They encode domain-level signal ("this is a chest X-ray") rather than pathology.

| Latent | Activation freq | Mean AUROC | Interpretation |
|--------|----------------|-----------|----------------|
| 3512 | 1.00 on every image | 0.557 | Global X-ray presence feature |
| 2640 | 0.99–1.00 | 0.567 | Global X-ray presence feature |
| 3321 | High across all groups | 0.587 | Broadly active |
| 1969 | High across all groups | 0.567 | Broadly active |
| 1084 | High across all groups | 0.593 | Broadly active |
| 3832 | High across all groups | 0.538 | Broadly active |

These latents dominate `illness_latent_summary.json` top-k rankings (sorted by raw activation) and should be excluded from clinical interpretability claims. The top-k-by-activation metric is misleading for identifying diagnostically useful features.

---

### Tier 2 — Healthy-lung detectors (7 latents)

Fire most strongly and frequently for `no_finding` images; suppressed when any pathology is present. They encode a **healthy / normal lung** signal rather than a specific disease.

| Latent | CheXpert | SIIM | TB | Mean | `no_finding` freq | Sick freq (CheXpert) |
|--------|---------|------|-----|------|-------------------|-----------------------|
| **818** | 0.631 | 0.830 | 0.664 | **0.708** | 0.98 | 0.72–0.88 |
| **1384** | 0.622 | 0.843 | 0.634 | **0.700** | 0.95–0.96 | 0.52–0.74 |
| **884** | 0.596 | 0.740 | 0.529 | 0.622 | 0.97 | 0.66–0.89 |
| **2891** | 0.598 | 0.744 | 0.564 | 0.635 | 0.76 | 0.22–0.38 |
| **3757** | 0.618 | 0.680 | 0.564 | 0.621 | 0.92 | 0.43–0.63 |
| **2512** | 0.625 | 0.685 | 0.550 | 0.620 | 0.84 | 0.17–0.44 |
| **1632** | 0.567 | 0.760 | 0.525 | 0.617 | 0.86–0.90 | 0.28–0.67 |

**Latents 818 and 1384** are the most robustly discriminative features in the entire confirmed set and are the only latents achieving mean AUROC > 0.70 across all three datasets. They are strong candidates for universally learned "healthy lung" representations.

**Latent 2891** is notable for a different reason: its absolute activation is low (≤ 50), but its frequency ratio between healthy (76%) and sick (22–38%) is very sharp — it functions more like a sparse binary indicator of healthy state than a graded activation.

**Latent 3757** shows a TB-specific pattern: activation 302.1 with 83% frequency on TB `airspace_opacity` images, but 258.9/82% on TB `no_finding`. Its healthy-lung interpretation holds for CheXpert (92% freq for no_finding vs 43–63% sick) but is less clean on TB, suggesting partial dataset specificity.

---

### Tier 3 — Pathology-presence detectors (3 latents)

Opposite polarity to Tier 2: fire strongly for sick images, suppressed for healthy. These encode a general "pathology present" signal with some condition specificity.

| Latent | CheXpert | SIIM | TB | Mean | `no_finding` freq | Sick freq (CheXpert) | Peak illness |
|--------|---------|------|-----|------|-------------------|-----------------------|-------------|
| **2682** | 0.579 | 0.779 | 0.668 | 0.675 | 0.18–0.32 | 0.47–0.70 | airspace_opacity |
| **3348** | 0.642 | 0.683 | 0.613 | **0.646** | 0.74–0.93 | 0.86–0.95 | pleural_abnormality |
| **2619** | 0.614 | 0.708 | 0.522 | 0.615 | 0.61–0.71 | 0.69–0.94 | cardiac_mediastinal |

**Latent 2682** is the clearest pathology-presence detector: fires on only 18–32% of healthy images but 47–70% of sick images, and this inversion holds across CheXpert, SIIM-COVID, and TB. It is the top z-score standout for `airspace_opacity` (Δ = 81.1 above other illness groups).

**Latent 3348** has the highest CheXpert AUROC of any latent (0.642) and the most balanced cross-dataset profile (all three > 0.61). It peaks for `pleural_abnormality` (Δ = 219.8 — the single largest illness standout in the dataset) but also cross-activates for `atelectasis` and `vascular_edema`, consistent with all three involving increased lung density or fluid accumulation.

**Latent 2619** peaks for `cardiac_mediastinal` (Δ = 185.3) and `vascular_edema` (Δ = 147.6). Low TB AUROC (0.522) is expected: TB is parenchymal, not cardiac. The cardiac/vascular co-activation reflects the clinical reality that left-heart failure causes both cardiomegaly and pulmonary edema simultaneously.

---

### Tier 4 — Illness-specific standout latents

Latents with the highest z-score differential for one illness group relative to all others, beyond those covered in Tier 3.

| Latent | Peak illness | Δ | Secondary illness | AUROC (mean) | Interpretation |
|--------|-------------|---|-----------------|-------------|----------------|
| 818 | no_finding | +254.5 | — | 0.708 | *(see Tier 2)* |
| 3348 | pleural_abnormality | +219.8 | atelectasis (+106.6) | 0.646 | *(see Tier 3)* |
| 2619 | cardiac_mediastinal | +185.3 | vascular_edema (+147.6) | 0.615 | *(see Tier 3)* |
| 884 | no_finding | +125.4 | — | 0.622 | *(see Tier 2)* |
| 1781 | no_finding | +111.5 | airspace_opacity | 0.564 | Borderline healthy/opacity feature |
| 1084 | other | +107.3 | — | 0.593 | Incidental findings / support devices |
| 3757 | other | +94.6 | no_finding | 0.621 | Mixed healthy/other signal |
| 2682 | airspace_opacity | +81.1 | — | 0.675 | *(see Tier 3)* |
| 3785 | other | +75.6 | atelectasis | 0.564 | Atelectasis/incidental overlap |
| 215 | vascular_edema | +86.3 | cardiac_mediastinal (+65.4) | 0.621 | Cardiac/vascular pathology |

---

## Cross-illness structural patterns

### Pleural abnormality ↔ Atelectasis

Latents 3348 and 1237 are top standouts for both groups. Clinically expected: atelectasis (lung collapse) produces increased density that overlaps radiologically with pleural thickening or effusion, and the two conditions frequently co-occur.

### Cardiac mediastinal ↔ Vascular edema

Latents 2619 and 215 are top standouts for both groups. Clinically expected: left-sided cardiac failure simultaneously causes cardiomegaly and pulmonary vascular congestion — the SAE independently learned this co-occurrence.

---

## Discriminability summary

### Only 3 latents achieve AUROC > 0.60 on all three datasets

| Latent | CheXpert | SIIM | TB | Mean | Tier |
|--------|---------|------|-----|------|------|
| 818 | 0.631 | 0.830 | 0.664 | 0.708 | Healthy-lung |
| 1384 | 0.622 | 0.843 | 0.634 | 0.700 | Healthy-lung |
| 3348 | 0.642 | 0.683 | 0.613 | 0.646 | Pathology (pleural/fluid) |

All other high-AUROC latents score well on SIIM-COVID (sharpest COVID signal) but trail on CheXpert or TB.

### AUROC distribution across 252 latents

| Threshold | Count (≥1 dataset) | Count (all 3 datasets) |
|-----------|-------------------|----------------------|
| > 0.70 | 12 / 252 (4.8%) | 2 / 252 (0.8%) |
| > 0.65 | 40 / 252 (15.9%) | 3 / 252 (1.2%) |
| > 0.60 | 91 / 252 (36.1%) | 3 / 252 (1.2%) |

Most latents sit modestly above chance (mean AUROC ~0.54–0.58). The interpretable signal is concentrated in a small tail.

---

## Next steps

| Priority | Action | Rationale |
|----------|--------|-----------|
| 🔴 | Re-harvest Matryoshka L3 activations with `image_stems` embedded in safetensors metadata | Unlocks spatial grounding (Analysis B) — currently all NaN; needed to validate whether discriminative latents fire at anatomically correct locations |
| 🔴 | Pass `--chex_csv`, `--siim_study_csv`, `--rsna_csv` in the illness nomad job | Unlocks fine-grained per-label probe (Analysis A2); currently all NaN for CheXpert/TB |
| 🟡 | Run `matryoshka-inspect-l3.nomad` on the 252 confirmed latents | Renders `semantic_grid.png` (top-8 patch crops per dataset) and `patch_ablation.png` for visual interpretation; focus on Tiers 2–3 latents |
| 🟡 | Prioritise visual inspection on latents 818, 1384, 2682, 3348, 2619, 2891 | These are the most clinically interpretable — the semantic grids will confirm or refute the tier assignments |
| 🟡 | Investigate why only 3 latents are robust across all 3 datasets | SIIM-COVID produces consistently higher AUROC than CheXpert or TB — likely due to COVID's distinctive bilateral GGO pattern; TB's small dataset (700 images) causes high AUROC variance |
| 🟢 | Filter backbone latents (3512, 2640, 3321, 1969, 1084, 3832) from all clinical summaries | Raw top-k activation ranking is dominated by these; any report should rank by AUROC, not mean activation |
| 🟢 | Compare illness eval results with BatchTopK L3 confirmed latents | Tests whether the Matryoshka-specific structural tiers (healthy/pathology/backbone) replicate in the non-hierarchical baseline |
