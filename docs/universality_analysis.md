# Universality Analysis Pipeline

How the codebase determines whether a trained SAE has learned features that are universal — i.e. features that fire consistently across multiple datasets for the same disease concept, rather than learning dataset-specific shortcuts or scanner artefacts.

The pipeline is hierarchical: each step gates the next. A latent must pass every prior filter before the more expensive checks are run on it.

---

## Step 0 — Scanner Latent Audit (Pre-filter)

**File:** `src/scripts/scanner_latent_audit.py`

Before any universality check, latents that encode scanner/hospital identity rather than pathology are identified and removed from consideration.

**Method:** For each latent, fit a multinomial logistic regression predicting `dataset_id` (0=CheXpert, 1=RSNA, 2=SIIM, 3=TB) from the latent's per-token activation magnitude. One-vs-rest AUROC is read off from `clf.predict_proba` scores and macro-averaged across datasets.

**Balanced subsample.** CheXpert contributes roughly 15× as many tokens as TB after spatial expansion, so an unstratified fit would be dominated by the big datasets. The script therefore draws a *balanced* subsample — `probe_sample_size // n_datasets` tokens per dataset (default 25 000 per dataset, 100 000 total) — once up front and reuses it for every latent's fit. Subsample size and RNG seed are recorded in the output JSON.

**Metric:** Macro-averaged one-vs-rest AUROC per latent.

**Threshold:** AUROC > 0.80 → latent flagged as a scanner latent.

**Logic:** If a feature's activation magnitude alone can predict which hospital/dataset an image came from with AUROC > 0.80, it is encoding acquisition differences (scanner type, exposure, demographics) rather than disease. Such a feature cannot be universal in any meaningful clinical sense.

**Output:** `scanner_latent_audit.json` — threshold, per-latent AUROC, flagged indices, `probe_sample_size`, `probe_seed`. Flagged indices are excluded from all downstream universality steps.

---

## Step 1 — Level 1: Activation Frequency Filter

**File:** `src/scripts/universality_eval.py` → `compute_frequency_per_dataset()`, `apply_level1_filter()`

**Question:** Does this latent fire at all, on enough datasets?

**Method:** For each latent and each dataset, compute the fraction of that dataset's tokens where the latent is active (activation > 0).

```
frequency(latent_i, dataset_d) = count(tokens in dataset_d where latent_i > 0)
                                  ─────────────────────────────────────────────
                                       total tokens in dataset_d
```

**Threshold:** frequency ≥ 1% on at least 2 of the 4 datasets.

**Why 1%:** This is Anthropic's target sparsity level from Gao et al. (2024). Features below 0.01% are in the dead/noise band; features above 10% encode generic anatomy that is not disease-specific. 1% sits in the interpretable medium-density band.

**Why ≥ 2 datasets:** A feature firing on only one dataset could be dataset-specific. It must appear in at least two to be a candidate for universality.

**Canonical class match (enforced when `--label_csv` flags are supplied):** For each (latent, dataset) pair, the dominant canonical class is the mode of the canonical groups of the top `--class_match_top_k` images (default 50) ranked by the latent's per-image `feature_max`. Images with no entry in the dataset's label CSV are dropped before the mode is taken (rather than silently counted as `no_finding`, which would bias the result toward the majority class). A candidate passes the class-match gate only when every active dataset agrees on its dominant class; any disagreement — or a dataset with no labelled image in its top-K — rejects the candidate. A feature firing on "pleural effusion" in CheXpert but "no finding" in RSNA will not pass even if frequency thresholds are met.

When no `--label_csv` flag is supplied the class-match gate is skipped and Level 1 falls back to frequency-only. The rows of `universal_candidates.csv` then omit the `matched_canonical_class` column.

**Output:** Set of Level 1 candidate latent indices + `matched_canonical_class` per candidate (when class match was enforced) — typically a small fraction of the full d_sae=4096 dictionary.

---

## Step 2 — Level 2: Representational Direction Alignment

**File:** `src/scripts/universality_eval.py` → `compute_centroids_per_dataset()`, `apply_level2_filter()`

**Question:** When this latent fires on two different datasets, is it responding to the same direction in the encoder's representation space?

This is the key universality check. A latent could satisfy Level 1 (fires on multiple datasets) but fire on completely different encoder directions in each dataset — meaning it is polysemantic, not universal. Level 2 tests whether the underlying representation that triggers the latent is geometrically consistent across datasets.

**Method:**

For each Level 1 candidate latent `i` and each dataset `d`, compute the **activation centroid** — the mean of all encoder token vectors (in d_model space) where latent `i` fired:

```
centroid(latent_i, dataset_d) = mean({ x ∈ dataset_d : latent_i(x) > 0 })
```

This gives one `[1024]`-dim vector per (latent, dataset) pair. Then for every pair of datasets (D1, D2), compute cosine similarity between their centroids:

```
cosine_sim = (centroid_D1 · centroid_D2) / (||centroid_D1|| × ||centroid_D2||)
```

**Threshold:** All pairwise dataset combinations must achieve cosine similarity ≥ 0.70.

**Why 0.70:** SVCCA (Raghu et al. 2017) uses ρ̄ ≥ 0.50 for subspace similarity. Since we compare individual directions (stricter than subspace), the threshold is raised to 0.70.

**Why all pairs must pass:** If a latent fires on CheXpert and RSNA with similar direction but fires on SIIM with a completely different direction, it is not universally representing the same concept — it is polysemantic across datasets.

**Output:** Set of confirmed universal latents + the minimum pairwise cosine similarity for each (reported in results).

---

## Step 3 — Cross-Dataset Probe (Supplementary)

**File:** `src/scripts/universality_eval.py` → `run_cross_dataset_probe()`

**Question:** Do the confirmed universal latents encode pathology information that generalises zero-shot to unseen datasets?

**Method:** Leave-one-dataset-out logistic regression. For each confirmed latent:
- Train a logistic regression mapping latent activation magnitude → canonical pathology group, using tokens from 3 of the 4 datasets
- Evaluate on the held-out dataset (zero-shot)
- Report macro-averaged one-vs-rest AUROC

**This is not a pass/fail gate** — it is supplementary evidence. A universal feature by the centroid criterion should also generalise in this probe. If it does not, that is a flag worth investigating.

**Output:** Per-latent leave-one-out AUROC table in `universal_candidates.csv`.

---

## Step 4 — Spatial Grounding (Supplementary)

**File:** `src/scripts/universality_eval.py` → `run_spatial_grounding()`

**Question:** Does the latent fire at the correct anatomical location, or just correlate with image-level labels?

**Method:** Uses the 7×7 spatial grid recovered by reversing flattening. For each confirmed latent and each image in RSNA or SIIM-COVID19 (the two datasets with radiologist bounding boxes):

1. Compute the continuous IoU overlap between each of the 49 patches and the radiologist bounding boxes:
   ```
   overlap[img_i, patch_j] = fraction of patch_j's area covered by any bbox in img_i
   ```
2. Compute Pearson correlation between the latent's 49 activation magnitudes and the 49 overlap values across all images:
   ```
   r = Pearson(activation_magnitudes, bbox_overlap_values)
   ```

**Threshold (Bonferroni-corrected):** Pearson `r` and its two-sided `p`-value come from `scipy.stats.pearsonr`. With `n_confirmed` universal candidates and `n_bbox_datasets` ∈ {1, 2} datasets with bboxes:

```
α_pooled  = --bonferroni_alpha / n_confirmed                   # default 0.05 / n_confirmed
α_per_ds  = --bonferroni_alpha / (n_confirmed × n_bbox_datasets)
```

A latent passes spatial grounding when its pooled r > 0 **and** its pooled p < α_pooled. Per-dataset pass/fail flags use α_per_ds and are surfaced as `{ds}_pass` columns in `universal_candidates.csv`. The corrected α values, number of tests performed, and number of passes are all persisted under `spatial_bonferroni` inside `universality_summary.json`.

**Why this matters:** A latent could be labelled "airspace opacity" because it fires on opacity images at the image level, but actually respond to the cardiac silhouette or background. Spatial grounding checks whether the latent fires at the locations where radiologists annotated pathology. This is the strongest form of evidence — the feature is not just statistically correlated with the label, it is firing at the right place.

**Output:** Per-latent Pearson r, p, and Bonferroni pass flags (per dataset + pooled) in `universal_candidates.csv`; family-wise summary in `universality_summary.json`.

---

## Step 5 — Co-activation Analysis (Feature Splitting Detection)

**Invocation:** `src/scripts/universality_eval.py` runs this automatically as its final step via `run_split_detection()` — no separate script invocation is needed.

**Files:**
- `src/visualisation/coactivation.py` → `compute_coactivation_matrix()` (Jaccard of top-K image sets), `find_coactivating_pairs()`, `render_coactivation_heatmap()`
- `src/visualisation/universality.py` → `compute_co_activation_matrix()` (Pearson of per-image feature_max) — available but not currently invoked by the pipeline

**Question:** Are any two confirmed universal latents encoding the same concept (feature splitting)?

Feature splitting is a known SAE failure mode: instead of one latent for "pleural effusion", the SAE might learn two latents that each fire on a subset of pleural effusion cases, often split by dataset or severity. Both latents would pass Level 1 and Level 2 individually but together they represent one concept, not two.

### Jaccard Co-activation (the metric used for split flagging)

For each confirmed latent we take its top-`--split_top_k` images (default 50) across the combined 4-dataset population — reusing the per-image `feature_max` computed in Step 1 so image identity is consistent with the rest of the pipeline. For each pair of confirmed latents (i, j):

```
Jaccard(i, j) = |top_K(i) ∩ top_K(j)| / |top_K(i) ∪ top_K(j)|
```

**Threshold:** Jaccard ≥ `--split_jaccard_threshold` (default 0.50) → pair flagged as a potential feature split.

Jaccard is more robust than Pearson here because it focuses on *which* images a feature fires on most strongly, ignoring magnitude distribution differences that arise from dataset-specific activation scales.

**Output:**
- `split_pairs.json` — list of `{i, j, jaccard}` records, sorted descending
- `co_activation_heatmap.png` — Jaccard heatmap over the confirmed latents
- `universality_summary.json` records `n_split_pairs` and `split_jaccard_threshold`

---

## Null Baselines

**File:** `src/scripts/null_baseline_eval.py`

Every metric above is also computed on two baselines to establish what universality looks like by chance:

### Random SAE Baseline
A freshly initialised SAE with the same architecture (d_model=1024, d_sae=4096, k=32) but random weights. Run through Level 1 + Level 2. Expected result: near-zero confirmed latents. Any confirmed universals would indicate the thresholds are too loose.

### PCA Baseline
Top-d_sae PCA components fitted on a sample of 50,000 activation tokens. A `PCANullSAE` wrapper treats each component as a latent: it "fires" when the projection magnitude exceeds a top-k threshold. Run through Level 1 + Level 2.

The PCA baseline is more meaningful than random — it captures linear structure already present in the encoder representations without any SAE training. A trained SAE should substantially outperform PCA on universality, otherwise the SAE is not learning anything beyond what PCA would find.

**Output:** `null_baseline_results.json` — comparison table: trained SAE vs random vs PCA on Level 1 count, Level 2 confirmed count, and median cosine similarity.

---

## Summary of All Metrics

| Step | Metric | Threshold | Pass/Fail |
|---|---|---|---|
| 0 — Scanner audit | Multinomial LR macro AUROC (dataset_id prediction) | < 0.80 to keep | Pre-filter |
| 1 — Level 1 | Activation frequency per dataset | ≥ 1% on ≥ 2 datasets | Hard gate |
| 1 — Level 1 (when --label_csv supplied) | Canonical class match across active datasets | Must agree on dominant class | Hard gate |
| 2 — Level 2 | Cosine similarity of per-dataset centroids | ≥ 0.70 all pairs | Hard gate |
| 3 — Probe | Leave-one-out logistic AUROC | — | Supplementary |
| 4 — Spatial | Pearson r vs bbox overlap | p_pooled < α / n_confirmed (default α = 0.05), r > 0 | Supplementary, Bonferroni-corrected |
| 5 — Split detection | Jaccard of top-K image sets | < 0.50 to keep distinct | Diagnostic (flags splits) |
| Baseline | Level 1 + Level 2 on random SAE | — | Comparison |
| Baseline | Level 1 + Level 2 on PCA | — | Comparison |

---

## Canonical Pathology Groups

All label-based checks map raw dataset labels to 7 Fleischner Society groups:

| Group | CheXpert | RSNA | SIIM-COVID19 | TB Shenzhen |
|---|---|---|---|---|
| `airspace_opacity` | Consolidation, Lung Opacity, Lung Lesion, Pneumonia | Pneumonia | Typical / Indeterminate / Atypical Appearance | TB Positive |
| `pleural` | Pleural Effusion, Pleural Other | — | — | — |
| `cardiac_mediastinal` | Cardiomegaly, Enlarged Cardiomediastinum | — | — | — |
| `atelectasis` | Atelectasis | — | — | — |
| `vascular_edema` | Edema | — | — | — |
| `no_finding` | No Finding | Normal | Negative for Pneumonia | Normal |
| `other` | Pneumothorax, Fracture, Support Devices | — | — | — |

CheXpert uncertain labels (−1) are treated as negative throughout.

---

## Output Files

| File | Contents |
|---|---|
| `scanner_latent_audit.json` | Flagged scanner latent indices, per-latent macro AUROC, `probe_sample_size`, `probe_seed` |
| `scanner_latent_audit.csv` | Per-latent row: macro AUROC + per-dataset AUROCs |
| `universality_summary.json` | Counts at each level, thresholds used, confirmed indices, Bonferroni summary, split-pair count |
| `universal_candidates.csv` | Per-latent: frequency/dataset, min cosine sim, matched canonical class (when class match enforced), probe AUROC, spatial `r`/`p`/`pass` (per dataset + pooled) |
| `confirmed_latent_frequencies.json` | Per-latent frequency per dataset (for downstream use) |
| `split_pairs.json` | List of `{i, j, jaccard}` pairs with Jaccard ≥ split threshold |
| `co_activation_heatmap.png` | Jaccard co-activation heatmap over the confirmed latents |
| `null_baseline_results.json` | Trained SAE vs random SAE vs PCA comparison |
| `latent_XXXXX/semantic_grid.png` | Top-K activating patches per dataset per latent |
| `latent_XXXXX/frequency_bar.png` | Per-dataset activation frequency bar chart |
| `latent_XXXXX/patch_ablation.png` | Deletion-mode ablation heatmaps |

---

## References

Each technique and its source. Items marked **project-specific** have no external citation — they were designed for this codebase.

---

**SAE architecture · BatchTopK sparsity · RMS normalisation · 1% frequency threshold**
Gao, L., Dupré la Tour, T., Tillman, H., et al. (2024).
*Scaling and Evaluating Sparse Autoencoders.*
arXiv:2406.04093. https://arxiv.org/abs/2406.04093

---

**Universality hypothesis** — the core theoretical motivation: analogous features form across models trained on similar domains.
Olah, C., Cammarata, N., Schubert, L., et al. (2020).
*Zoom In: An Introduction to Circuits.*
Distill. https://distill.pub/2020/circuits/zoom-in/

---

**Superposition** — why neurons encode multiple features simultaneously, and why SAEs are needed to recover monosemantic features.
Elhage, N., Hume, T., Olsson, C., et al. (2022).
*Toy Models of Superposition.*
Transformer Circuits Thread. https://transformer-circuits.pub/2022/toy_model/index.html

---

**Cosine similarity threshold 0.70** — derived from SVCCA's subspace similarity criterion (ρ̄ ≥ 0.50), raised to 0.70 because we compare individual directions rather than subspaces.
Raghu, M., Gilmer, J., Yosinski, J., & Sohl-Dickstein, J. (2017).
*SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability.*
NeurIPS 2017. arXiv:1706.05806. https://arxiv.org/abs/1706.05806

---

**Canonical pathology groups (7-class Fleischner taxonomy)** — the label grouping used for canonical class matching across datasets.
Hansell, D. M., Bankier, A. A., MacMahon, H., McLoud, T. C., Müller, N. L., & Remy, J. (2008).
*Fleischner Society: Glossary of Terms for Thoracic Imaging.*
Radiology, 246(3), 697–722. https://pubmed.ncbi.nlm.nih.gov/18195376/

---

**Inter-rater label noise** — justification for why raw labels cannot be matched directly across datasets (inter-rater κ = 0.53–0.68 for chest X-ray labels).
Peng, Y., et al. (2021).
*RadGraph: Extracting Clinical Entities and Relations from Radiology Reports.*
NeurIPS 2021 Datasets and Benchmarks Track.

---

**Universality across architectures** (related work)
Klabunde, M., et al. (2024).
*Towards Universality: Studying Mechanistic Similarity Across Language Model Architectures.*
arXiv:2410.06672. https://arxiv.org/abs/2410.06672

---

**Universality in biological and artificial neural networks** (related work)
Tuckute, G., et al. (2024).
*Universality of Representation in Biological and Artificial Neural Networks.*
bioRxiv. https://www.biorxiv.org/content/10.1101/2024.12.26.629294v1

---

**Swin Transformer** — the backbone of UMedPT; source of the 7×7 spatial token grid at Stage 3→4.
Liu, Z., Lin, Y., Cao, Y., et al. (2021).
*Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.*
ICCV 2021. arXiv:2103.14030. https://arxiv.org/abs/2103.14030

---

### Project-Specific (no external citation)

The following were designed specifically for this project and do not derive from a published paper:

- **Scanner latent audit** (AUROC > 0.80 pre-filter) — logistic regression predicting dataset_id from latent activation magnitude
- **Per-dataset activation centering** — subtract each dataset's mean activation vector before concatenation to correct b_dec drift
- **Spatial grounding via Pearson r vs bbox overlap** — correlating per-patch activation magnitude against continuous IoU with radiologist bounding boxes
- **Jaccard co-activation for split detection** — Jaccard similarity of top-50 image sets to identify feature splitting across latents
