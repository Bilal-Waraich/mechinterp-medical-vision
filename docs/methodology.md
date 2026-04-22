# Methodology

Sparse Autoencoder trained on UMedPT Swin Transformer activations across four
chest X-ray datasets to test whether a medical foundation model learns universal
semantic representations.

---

## 1. Universality

### 1.1 General definition

A foundation model trained on diverse medical images may have learned
representations that capture genuine clinical concepts — things like "airspace
opacity" or "no pathological finding" — rather than memorising the quirks of
individual hospitals or scanners. Universality asks: **do these representations
encode the same biological concept regardless of where the image came from, who
the patient was, or what equipment was used?**

If a SAE latent fires consistently on consolidation in a Stanford X-ray
(CheXpert), a US community hospital X-ray (RSNA), a COVID chest film (SIIM),
and a Shenzhen TB film — and fires in the same direction in the model's
representation space — that is evidence the foundation model learned something
real about pathology, not something incidental about acquisition.

This definition is grounded in the **universality hypothesis** from mechanistic
interpretability (Olah et al., 2020), which proposes that analogous features
form across models and tasks when those models are trained on similar domains.
We operationalise the hypothesis within a single model and single modality:
universality across patient populations and disease contexts rather than across
model architectures.

---

### 1.2 Technical definition

A SAE latent is **universal** if it jointly satisfies four properties. All four
must hold for a latent to be reported as a confirmed universal feature.

#### Property 1 — Domain invariance (does the feature fire?)

**Definition:** A latent is *active on dataset D* if it fires on ≥1% of that
dataset's tokens. A latent is a **universal candidate** if it is active on ≥2
of the 4 datasets AND its dominant canonical pathology class matches across
those datasets.

**Threshold rationale:** Anthropic's k-sparse SAEs target roughly 1% activation
density per token (Gao et al., 2024). Features below 0.01% are in the
dead/noise band; features above 10% encode generic anatomy. The 1% threshold
sits in the interpretable medium-density band.

**Class-match operationalisation:** For each (latent, dataset) pair, the
*dominant canonical class* is the mode of the canonical labels among the top
``class_match_top_k`` images (default 50) ranked by the latent's per-image
`feature_max`. Images whose filename stem is missing from the dataset's label
CSV are excluded from the mode (rather than silently counted as `no_finding`,
which would bias the result toward the majority class). A latent clears the
class-match gate when every active dataset agrees on its dominant class; any
disagreement, or a dataset with no labelled image in its top-K, rejects the
latent. When no `--label_csv` flag is supplied the class-match gate is
skipped and Level 1 falls back to frequency-only.

**Token counts at pm_index=2 (7×7 = 49 tokens/image):**

| Dataset | Images | Tokens | 1% threshold |
|---|---|---|---|
| CheXpert | 10,000 | 490,000 | 4,900 tokens |
| RSNA | 10,000 | 490,000 | 4,900 tokens |
| SIIM-COVID19 | ~6,334 | ~310,000 | 3,100 tokens |
| TB Shenzhen | 700 | 34,300 | 343 tokens |

**Canonical pathology groups:** Labels across datasets are mapped to 7 canonical
groups before class-match checking, using the Fleischner Society taxonomy
(Hansell et al., 2008), which is also the labelling convention used by CheXpert:

| Canonical group | CheXpert | RSNA | SIIM-COVID19 | TB Shenzhen |
|---|---|---|---|---|
| Airspace opacity | Consolidation, Lung Opacity | Pneumonia positive | Opacity/consolidation | TB infiltrate (positive) |
| Pleural abnormality | Pleural Effusion, Pleural Other | — | — | — |
| Cardiac/mediastinal | Cardiomegaly, Enlarged Cardiomediastinum | — | — | — |
| Atelectasis | Atelectasis | — | — | — |
| Vascular/edema | Edema | — | — | — |
| No finding | No Finding | Normal | No finding | Negative |
| Other | Support Devices, Fracture | — | Pneumothorax | — |

CheXpert uncertain labels (−1) are treated as negative (conservative, following
the CheXpert paper convention).

---

#### Property 2 — Cross-population robustness / Representational direction alignment (does it fire in the same direction?)

**Definition:** For each Property 1 candidate, compute the centroid of all
activating token vectors in d_model space separately per dataset. The cosine
similarity between any two per-dataset centroids must be **≥0.70** for the
latent to be confirmed universal.

**Threshold rationale:** SVCCA (Raghu et al., 2017) uses ρ̄ ≥ 0.5 for subspace
similarity between layers. We compare individual directions, which is a stricter
test, so the threshold is raised to 0.70.

**Rationale:** A latent could clear the 1% frequency threshold on two datasets
while activating for entirely different visual reasons in each — e.g., firing on
consolidation in CheXpert but on cardiac borders in RSNA, simply because both
are common in their respective datasets. Direction alignment rules this out: if
the encoder places the same pathology concept in the same representational
direction across datasets, the per-dataset centroids will be similar. If they
are not, the latent is frequent but not universal.

**Cross-population robustness** is a consequence of jointly passing Properties 1
and 2 rather than a separate test. Because the four datasets span distinct
patient populations and acquisition contexts — Stanford academic hospital
(CheXpert), US general population (RSNA), international COVID cohort (SIIM), and
Shenzhen China (TB) — a latent that passes both properties is implicitly
robust to demographic, epidemiological, and institutional variation.

**Status:** ✅ Implemented in `src/scripts/universality_eval.py` —
`compute_centroids_per_dataset()` + `apply_level2_filter()`. The class-match
clause of Property 1 is enforced by `apply_level1_filter()` when the optional
`dominant_class` dictionary is passed (built from the `--label_csv` inputs via
`compute_dominant_canonical_class()`).

---

#### Property 3 — Spatial grounding (does it fire for the right reason?)

**Definition:** For each Property 1+2 candidate, compute the Pearson correlation
between per-patch activation magnitude and patch-level pathology coverage
(derived from RSNA bounding boxes and SIIM bounding boxes via continuous IoU
overlap). A confirmed universal latent must show statistically significant
positive r (p < 0.05 after Bonferroni correction for the number of candidates
tested).

**Bonferroni implementation:** Pearson r and its two-sided p-value come from
`scipy.stats.pearsonr`. Bonferroni thresholds are derived from the
`--bonferroni_alpha` flag (default α = 0.05):

```
α_pooled  = α / n_confirmed
α_per_ds  = α / (n_confirmed × n_bbox_datasets)
```

A latent passes spatial grounding when its pooled p-value is below
``α_pooled`` **and** its pooled r is positive. Per-dataset pass/fail flags use
``α_per_ds`` and appear as `{ds}_pass` columns in `universal_candidates.csv`.
The corrected thresholds, test counts, and number of passes are persisted under
`spatial_bonferroni` in `universality_summary.json` for reproducibility.

**Rationale:** Without spatial grounding, a latent that passes Properties 1+2
could be firing on a consistent anatomical or geometric artifact (rib grid, lung
border, cardiac silhouette) rather than on pathology. Spatial grounding
distinguishes a genuine pathology-encoding feature from a universal artifact.

**Status:** ✅ Implemented in `src/scripts/universality_eval.py` —
`load_bbox_csv()`, `compute_patch_bbox_overlap()`, `run_spatial_grounding()`
(returns a `(per_latent_results, bonferroni_summary)` tuple). Requires
`--rsna_bbox_csv` and/or `--siim_bbox_csv` at runtime.

---

#### Property 4 — Monosemanticity (is it one concept, not a superposition?)

**Definition:** A latent is monosemantic if it encodes a single concept across
its activating images, as opposed to being polysemantic (encoding multiple
unrelated concepts in superposition). Monosemanticity is a prerequisite for
meaningful universality: a latent that fires on consolidation in CheXpert and
on the cardiac border in RSNA has passed the frequency threshold but encodes
nothing universal.

**Measurement:** The co-activation matrix (Pearson correlation of feature_max
vectors across all images) is used to detect **feature splitting** — the failure
mode where a single concept is distributed across 2–3 latents, each firing on
a slightly different visual presentation across datasets. Latents with high
pairwise correlation that map to the same canonical class are flagged as split
features and reported separately.

**Relationship to superposition:** The SAE resolves superposition by construction
(Elhage et al., 2022) — sparsity forces the network to allocate separate latents
for separate concepts. Within a well-trained SAE, all confirmed universal
features should be monosemantic. What we are testing is whether they are
monosemantic *and* semantically coherent (spatially grounded, class-aligned) or
monosemantic but clinically uninterpretable (e.g., a consistent rib shadow
artifact). The scanner audit (Section 1.3) handles the latter case.

**Status:** ✅ Co-activation matrix implemented in
`src/visualisation/universality.py:compute_co_activation_matrix()` (Pearson)
and `src/visualisation/coactivation.py:compute_coactivation_matrix()`
(Jaccard of top-K image sets). Split-pair detection runs automatically at the
end of the pipeline via `universality_eval.run_split_detection()` — any pair
of confirmed latents with Jaccard ≥ `--split_jaccard_threshold` (default 0.50)
is reported in `split_pairs.json` and counted in `universality_summary.json`
as `n_split_pairs`. A companion `co_activation_heatmap.png` of the confirmed
latents is written to the output directory.

---

### 1.3 Scanner audit (pre-filter)

Before universality analysis, latents encoding scanner/acquisition artifacts
rather than pathology are identified and excluded. A multinomial logistic
regression probe is fitted per latent from the latent's per-token activation
magnitude to `dataset_id`; one-vs-rest AUROC is read off from the probe's
`predict_proba` scores and macro-averaged across datasets. Latents with macro
AUROC > 0.80 are flagged as **scanner latents** and excluded from Properties
1–4 testing.

**Rationale:** A latent that fires universally because every image from dataset A
looks the same (exposure setting, edge enhancement, DICOM post-processing)
would pass the frequency threshold but carry no clinical meaning. Excluding
scanner latents before universality analysis ensures that confirmed universal
features cannot be explained by acquisition confounds.

**Balanced subsample:** CheXpert contributes roughly 15× as many tokens as TB
after spatial expansion, so a random sample would effectively probe only the
large datasets. The implementation therefore draws a balanced subsample —
`--probe_sample_size // n_datasets` tokens per dataset (default 25 000 per
dataset, 100 000 total) — once up front and fits every latent on that same
sample. The probe's training/evaluation set, seed, and subsample size are
persisted in `scanner_latent_audit.json` (`probe_sample_size`, `probe_seed`)
so the audit is reproducible.

**Status:** ✅ Implemented in `src/scripts/scanner_latent_audit.py`. Requires
the merged `activations.safetensors` format produced by
`src/conglomerate/harvest_all.py` (must contain both `activations` and
`dataset_ids` tensors).

---

### 1.4 Cross-dataset probe (supplementary confirmation)

For each latent confirmed by Properties 1+2, a logistic regression probe is
trained on the canonical-group labels of **one dataset only**, then evaluated
zero-shot on the remaining datasets using the SAE latent activations as features.
High leave-one-dataset-out AUROC is independent confirmation that the encoder
places the same pathology concept in the same representational direction across
datasets — it validates the centroid cosine similarity result from a
discriminative rather than a geometric perspective.

This test is **supplementary**, not a pass/fail gate. It provides an
interpretable quantitative signal for the thesis: a universal latent with
AUROC ≫ 0.5 on unseen datasets is evidence of genuine cross-dataset
generalisation, not just co-occurrence.

**Status:** ✅ Implemented in `src/scripts/universality_eval.py` —
`run_cross_dataset_probe()`. Requires `--label_csv DATASET:PATH` per dataset.

---

### 1.5 Null baseline

Both Level 1 (Property 1) and Level 2 (Property 2) tests are repeated on:

1. **Random Gaussian SAE** — same architecture as the trained SAE, weights
   randomly initialised, never trained. Measures how many latents pass the
   frequency threshold by chance given the activation distribution.
2. **PCA baseline** — top-4096 principal components of the combined activation
   data. Measures universality of the linear structure already present in the
   encoder without any SAE training.

The primary thesis claim is that the trained SAE's confirmed universal latent
count is significantly above the null rate. If the null baseline passes the
frequency threshold at a similar rate, the 1% threshold is too lenient and must
be tightened before reporting results.

**Status:** ✅ Implemented in `src/scripts/null_baseline_eval.py`.
Supports `--null_mode random pca`. Outputs comparison table against
trained SAE results when `--trained_summary` is provided.

---

### 1.6 Activation–label alignment (image stems)

Downstream analyses that join activations to label CSVs (Property 1 class
match, cross-dataset probe) or to bounding-box CSVs (Property 3) need a key
that identifies *which image* each activation row came from. A positional
alignment (the i-th image in the activation file = the i-th row in the CSV)
would be fragile: CSV row order depends on the data provider's export, while
activation order depends on each `Dataset` class's iteration order.

Instead, `src/conglomerate/harvest_all.py` records each dataset's filename
stems in harvest order and saves them in the merged `.safetensors` metadata
header as a single JSON field:

```json
"image_stems": {
  "chexpert":   ["patient00001_study1_view1_frontal", ...],
  "rsna":       ["00aecb01-a116-45a2-956c-08d7fa55433c", ...],
  "siim_covid": ["ff0879eb20ed_study", ...],
  "tb":         ["CHNCXR_0001_0", ...]
}
```

Downstream scripts read the stems via `safetensors.safe_open(...).metadata()`
and look each stem up in the CSV keyed by `Path(image_id).stem`, so the
activation ↔ label join is exact even when the CSV has been reordered,
filtered, or had rows added. The activation file's own shape-mismatch guard
(`dataset_ids.shape[0] != activations.shape[0]`) additionally rejects files
produced by older harvesters that stored per-image rather than per-token
metadata.

---

### 1.7 Operational pipeline

Given a trained BatchTopK SAE checkpoint at pm_index=2 or pm_index=3, the
universality analysis proceeds as:

```
Step 1 — Scanner audit
  script : src/scripts/scanner_latent_audit.py
  input  : SAE checkpoint + activations.safetensors (with dataset_ids)
  flags  : --probe_sample_size N   (balanced per-dataset subsample)
           --threshold 0.80        (macro AUROC cut-off)
  output : scanner_latent_audit.json  (flagged indices excluded from all analysis)
  status : ✅ implemented

Step 2 — Level 1 frequency check + canonical class match
  script : src/scripts/universality_eval.py
  input  : SAE checkpoint + merged activations (or 4 per-dataset files)
           (optional) --label_csv DATASET:PATH per dataset
             ↳ when supplied, enforces the Property 1 class-match gate
           (optional) --scanner_audit to exclude flagged latents
  output : universal_candidates.csv, universality_summary.json
  status : ✅ implemented  (compute_frequency_per_dataset
                             + compute_per_image_max
                             + compute_dominant_canonical_class
                             + apply_level1_filter)

Step 3 — Level 2 direction check
  script : src/scripts/universality_eval.py  (same run as Step 2)
  input  : Level 1 candidates + activations
  output : confirmed_latent_frequencies.json + cosine sim columns in CSV
  status : ✅ implemented  (compute_centroids_per_dataset + apply_level2_filter)

Step 4 — Cross-dataset probe
  script : src/scripts/universality_eval.py  (same run)
  input  : --label_csv per dataset (shares the per-image feature_max
           computed in Step 2, so train/test image identity is consistent
           with the Level 1 gate)
  output : per-candidate leave-one-out AUROC columns in CSV
  status : ✅ implemented  (run_cross_dataset_probe)

Step 5 — Spatial grounding (Bonferroni-corrected)
  script : src/scripts/universality_eval.py  (same run)
  input  : --rsna_bbox_csv + --siim_bbox_csv
           --bonferroni_alpha  (family-wise α, default 0.05)
  output : per-candidate {ds}_r / {ds}_p / {ds}_pass and
           pooled_r / pooled_p / pooled_pass columns in CSV,
           Bonferroni summary (α_pooled, α_per_ds, pass counts)
           under `spatial_bonferroni` in universality_summary.json
  status : ✅ implemented  (load_bbox_csv + compute_patch_bbox_overlap
                             + run_spatial_grounding using
                             scipy.stats.pearsonr)

Step 6 — Split detection (Property 4)
  script : src/scripts/universality_eval.py  (same run)
  input  : confirmed latents + per-image feature_max
           --split_jaccard_threshold (default 0.50)
           --split_top_k             (default 50)
  output : split_pairs.json, co_activation_heatmap.png,
           n_split_pairs in universality_summary.json
  status : ✅ implemented  (run_split_detection via
                             visualisation/coactivation.py)

Step 7 — Null baseline
  script : src/scripts/null_baseline_eval.py
  input  : same activation files; --trained_summary from Step 2
  output : null_baseline_results.json  (random SAE + PCA comparison table)
  status : ✅ implemented  (random init SAE + PCA null via SVD)

Step 8 — Qualitative inspection
  script : src/scripts/inspect_universal_latents.py
  input  : confirmed_latent_frequencies.json (from Step 2/3) + activations
           + image directories (optional, for patch crops and ablation)
  output : per-latent/ frequency_bar.png, semantic_grid.png,
           patch_ablation.png + index.md
  status : ✅ implemented  (render_semantic_grid + render_ablation_panel
                             + render_frequency_bar)
```

---

### 1.9 References

- Olah, C., Cammarata, N., Schubert, L., Goh, G., Petrov, M., & Carter, S.
  (2020). **Zoom In: An Introduction to Circuits.** *Distill.*
  https://distill.pub/2020/circuits/zoom-in/

- Elhage, N., Hume, T., Olsson, C., et al. (2022). **Toy Models of
  Superposition.** *Transformer Circuits Thread.*
  https://transformer-circuits.pub/2022/toy_model/index.html

- Gao, L., Dupré la Tour, T., Tillman, H., et al. (2024). **Scaling and
  Evaluating Sparse Autoencoders.** *arXiv:2406.04093.*
  https://arxiv.org/abs/2406.04093

- Raghu, M., Gilmer, J., Yosinski, J., & Sohl-Dickstein, J. (2017). **SVCCA:
  Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and
  Interpretability.** *NeurIPS 2017.*
  https://arxiv.org/abs/1706.05806

- Hansell, D. M., Bankier, A. A., MacMahon, H., McLoud, T. C., Müller, N. L.,
  & Remy, J. (2008). **Fleischner Society: Glossary of Terms for Thoracic
  Imaging.** *Radiology, 246*(3), 697–722.
  https://pubmed.ncbi.nlm.nih.gov/18195376/

- Klabunde, M., et al. (2024). **Towards Universality: Studying Mechanistic
  Similarity Across Language Model Architectures.** *arXiv:2410.06672.*
  https://arxiv.org/abs/2410.06672

- Tuckute, G., et al. (2024). **Universality of Representation in Biological
  and Artificial Neural Networks.** *bioRxiv.*
  https://www.biorxiv.org/content/10.1101/2024.12.26.629294v1
