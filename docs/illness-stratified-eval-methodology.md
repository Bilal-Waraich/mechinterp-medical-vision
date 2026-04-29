# Illness-Stratified Evaluation — Methodology

**Script:** `src/scripts/illness_stratified_eval.py`  
**Pipeline step:** 05 — post-universality clinical evaluation  
**Input:** 252 confirmed universal latents (5-stage Matryoshka universality pipeline) + per-dataset activation safetensors  
**Output:** activation profiles, discriminability probes (coarse + fine), spatial grounding, illness latent matrix

---

## 0. Context and Purpose

The 5-stage universality pipeline (step 04) produces a set of **confirmed universal latents** — SAE features that are active across ≥ 2 datasets, directionally consistent in representation space, and stable under the Matryoshka nested-sparsity schedule. This confirms *that* the features are universal but does not say *what* they encode.

The illness-stratified evaluation answers two questions:

1. **Discriminability (Analysis A).** Can each confirmed latent distinguish between different illness groups? A latent that is universal but not discriminative encodes a generic anatomical signal rather than a clinically meaningful concept.
2. **Spatial grounding (Analysis B).** Does each confirmed latent fire at the anatomically correct location within the image? A latent that achieves high discriminability by firing broadly (e.g., at the image border) rather than over the diseased region is not truly encoding pathology.

---

## 1. Input: Confirmed Latent Set

Let the set of confirmed universal latents from step 04 be

$$\mathcal{C} = \{ c_1, c_2, \ldots, c_K \}, \quad K = 252$$

where each $c_k \in \{0, 1, \ldots, d_{\text{SAE}} - 1\}$ is an index into the SAE latent dimension ($d_{\text{SAE}} = 4096$).

The step-04 pipeline guarantees that every $c_k$ satisfies:
- **Frequency:** fires on ≥ 1% of tokens on ≥ 2 of 4 datasets
- **Directional consistency:** pairwise centroid cosine similarity ≥ 0.70 across active datasets
- **Prefix stability:** adjacent-scale Jaccard ≥ 0.40 across Matryoshka shells $k \in \{6, 12, 24, 48\}$
- **Scanner invariance:** macro AUROC for dataset-of-origin prediction ≤ 0.80

---

## 2. Illness Group Taxonomy

Clinical chest X-ray pathologies are mapped to 7 canonical groups following the **Fleischner Society Glossary** (Hansell et al., 2008):

| Index $g$ | Canonical Group | CheXpert labels | SIIM labels | RSNA | TB |
|------------|----------------|-----------------|-------------|------|----|
| 0 | `airspace_opacity` | Consolidation, Lung Opacity | Opacity/consolidation | Pneumonia positive | TB infiltrate |
| 1 | `pleural_abnormality` | Pleural Effusion, Pleural Other | — | — | — |
| 2 | `cardiac_mediastinal` | Cardiomegaly, Enlarged Cardiomediastinum | — | — | — |
| 3 | `atelectasis` | Atelectasis | — | — | — |
| 4 | `vascular_edema` | Edema | — | — | — |
| 5 | `no_finding` | No Finding | Negative for Pneumonia | Normal | Negative |
| 6 | `other` | Support Devices, Fracture, Pneumonia | Pneumothorax, Atypical | — | — |

Each image $n$ in dataset $d$ is assigned a canonical label $y_n^d \in \{0, 1, \ldots, 6\}$ at harvest time. CheXpert uncertain labels (−1) are treated conservatively as 0 (absent). Images that do not match any canonical group remain unlabelled ($y_n^d = -1$) and are excluded from all probes.

---

## 3. Patch-Level Activation Encoding

### 3.1 Notation

Each dataset $d \in \{\text{chexpert, rsna, siim\_covid, tb}\}$ contains $N_d$ images. The pre-harvest SwinTransformer layer-3 activations are stored in a safetensors file as a dense tensor of shape

$$\mathbf{A}^d \in \mathbb{R}^{(N_d \cdot P) \times d_{\text{model}}}$$

where $P = H_p \cdot W_p = 7 \times 7 = 49$ is the number of spatial tokens per image (each corresponding to a $32 \times 32$-pixel patch in the $224 \times 224$ input image) and $d_{\text{model}} = 1024$ is the SwinTransformer layer-3 hidden dimension.

### 3.2 SAE Encoding

The trained Matryoshka SAE encoder $f_{\text{enc}}: \mathbb{R}^{d_{\text{model}}} \rightarrow \mathbb{R}^{d_{\text{SAE}}}$ is applied patch-by-patch in batches:

$$\mathbf{Z}^d = f_{\text{enc}}(\mathbf{A}^d) \in \mathbb{R}^{(N_d \cdot P) \times d_{\text{SAE}}}$$

The activation values $Z^d_{t,i} \geq 0$ are non-negative due to the ReLU nonlinearity in the SAE encoder. Under the Matryoshka k-schedule (pm\_index = 3, $k \in \{6, 12, 24, 48\}$), at most 48 latents are non-zero per token at inference time (the full $k=48$ budget is used for evaluation).

### 3.3 Restricting to Confirmed Latents

Only the $K = 252$ confirmed latent indices $\mathcal{C}$ are retained:

$$\tilde{\mathbf{Z}}^d = \mathbf{Z}^d_{:, \mathcal{C}} \in \mathbb{R}^{(N_d \cdot P) \times K}$$

This is implemented as an index-select operation on the full SAE output.

### 3.4 Max-Pooling to Image Level

The patch-level tensor is reshaped to

$$\tilde{\mathbf{Z}}^d_{\text{patches}} = \tilde{\mathbf{Z}}^d.\text{reshape}(N_d, P, K) \in \mathbb{R}^{N_d \times P \times K}$$

and aggregated to image level by **spatial max-pooling**:

$$\mathbf{z}^{d,\text{img}}_{n,k} = \max_{p \in \{1,\ldots,P\}} \tilde{Z}^d_{\text{patches}}[n, p, k] \in \mathbb{R}^{N_d \times K}$$

**Rationale for max-pooling.** Pathological findings are spatially localised — consolidation, effusion, and atelectasis each occupy a fraction of the lung field. Mean-pooling would dilute the signal over the 40+ non-pathological patches; max-pooling captures the peak firing over the most pathologically active region and is directly interpretable as "does this latent fire at all on this image?" Max-pooled activations $z^{d,\text{img}}_{n,k}$ are the feature used in all discriminability probes (Analysis A).

---

## 4. Activation Profiles

Before fitting any probe, per-group activation statistics are computed as a descriptive summary.

### 4.1 Mean Activation

For illness group $g$, dataset $d$, and confirmed latent $k$:

$$\mu^d_{g,k} = \frac{1}{|\mathcal{N}^d_g|} \sum_{n \in \mathcal{N}^d_g} z^{d,\text{img}}_{n,k}$$

where $\mathcal{N}^d_g = \{ n : y^d_n = g \}$ is the set of images in group $g$ for dataset $d$.

### 4.2 Activation Frequency

$$\phi^d_{g,k} = \frac{1}{|\mathcal{N}^d_g|} \sum_{n \in \mathcal{N}^d_g} \mathbf{1}[z^{d,\text{img}}_{n,k} > 0]$$

### 4.3 Illness Latent Matrix

The 7 × K **illness latent matrix** $\mathbf{M} \in \mathbb{R}^{7 \times K}$ is the mean activation averaged across datasets:

$$M_{g,k} = \frac{1}{|\mathcal{D}_{g,k}|} \sum_{d \in \mathcal{D}_{g,k}} \mu^d_{g,k}$$

where $\mathcal{D}_{g,k}$ is the set of datasets for which group $g$ has at least one image with latent $k$ active.

**Limitation.** The illness latent matrix ranks latents by raw activation magnitude. Backbone latents (those encoding domain-level signal rather than pathology) have high activation on every image and therefore dominate top-$k$ rankings regardless of clinical discriminability. AUROC, not raw activation, is the correct primary metric for identifying clinically meaningful latents.

---

## 5. Analysis A1 — Coarse Discriminability Probe

### 5.1 Objective

For each confirmed latent $k$ and dataset $d$, estimate how well the scalar feature $z^{d,\text{img}}_{n,k}$ can distinguish between the 7 canonical illness groups as a **one-vs-rest multiclass classifier**.

### 5.2 Feature Construction

Each image is represented by a single scalar:

$$x_n = z^{d,\text{img}}_{n,k} \in \mathbb{R}$$

This is the one-dimensional feature input to the logistic regression classifier. The scalar design is intentional: if a single SAE latent is truly encoding a clinical concept, it should have discriminative power on its own without needing to combine with other latents.

### 5.3 Classifier

A **logistic regression** classifier with one-vs-rest (OvR) decomposition is fitted:

$$\hat{P}(y = g \mid x) = \sigma(w_g x + b_g), \quad g \in \{0, \ldots, 6\}$$

where $\sigma(v) = (1 + e^{-v})^{-1}$ is the sigmoid function. The parameters $(w_g, b_g)$ are estimated by L2-penalised maximum likelihood with **class-weight balancing**: each class $g$ is assigned weight $\frac{N_d}{7 \cdot |\mathcal{N}^d_g|}$, so rare pathology classes (TB has few pleural or cardiac cases) contribute proportionally rather than being dominated by the majority `no_finding` class.

### 5.4 Cross-Validation Protocol

To avoid overfitting on small datasets (TB: $N_{\text{TB}} \approx 700$), model performance is estimated with **5-fold stratified cross-validation**:

$$\mathcal{D}_{\text{train}}^{(f)}, \mathcal{D}_{\text{test}}^{(f)} = \text{StratifiedKFold}(K=5, \text{random\_state}=42)$$

Stratification ensures that the illness group distribution is approximately preserved in each fold. Each fold that has fewer than 10 training samples in any class represented in the test set is skipped.

For each fold $f$:
1. Fit $(w_g^{(f)}, b_g^{(f)})$ on $\mathcal{D}_{\text{train}}^{(f)}$
2. Compute predicted probabilities $\hat{P}^{(f)}(y=g \mid x_n)$ for all $n \in \mathcal{D}_{\text{test}}^{(f)}$
3. Compute macro-averaged OvR AUROC:

$$\text{AUROC}_{\text{macro}}^{(f)} = \frac{1}{7} \sum_{g=0}^{6} \text{AUROC}\!\left( \{ y_n = g \}_{n \in \mathcal{D}_{\text{test}}^{(f)}},\; \{ \hat{P}^{(f)}(y=g \mid x_n) \}_{n \in \mathcal{D}_{\text{test}}^{(f)}} \right)$$

### 5.5 Reported Metric

The final discriminability score for latent $k$ on dataset $d$ is:

$$\overline{\text{AUROC}}_{k,d} = \frac{1}{|F|} \sum_{f \in F} \text{AUROC}_{\text{macro}}^{(f)}, \quad \sigma_{k,d} = \text{std}\left( \{\text{AUROC}_{\text{macro}}^{(f)}\}_{f \in F} \right)$$

where $F$ is the set of valid (non-skipped) folds.

**Baseline.** A random classifier yields AUROC = 0.5 by definition. The macro OvR formulation means a latent that fires uniformly on all classes also achieves AUROC ≈ 0.5.

### 5.6 Cross-Dataset Mean AUROC

To assess robustness across patient populations, a **cross-dataset mean AUROC** is computed for each latent by averaging over the datasets for which the coarse probe returned a valid (non-NaN) score:

$$\overline{\text{AUROC}}_k = \frac{1}{|\mathcal{D}_k|} \sum_{d \in \mathcal{D}_k} \overline{\text{AUROC}}_{k,d}$$

where $\mathcal{D}_k$ excludes RSNA (binary dataset: no multi-class structure, probe returns NaN) and any dataset where all fold scores were NaN.

---

## 6. Analysis A2 — Fine-Grained Per-Label Probe

The coarse probe treats illness groups as the target. The fine-grained probe replaces the 7-group label with **per-dataset clinical labels** at full granularity:

| Dataset | Label type | Labels |
|---------|-----------|--------|
| CheXpert | 14 binary labels | No Finding, Cardiomegaly, Lung Opacity, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion, Pleural Other, Fracture, Support Devices, Lung Lesion, Enlarged Cardiomediastinum |
| RSNA | Binary | Pneumonia (0/1) |
| SIIM-COVID | 4-class | Negative, Typical, Indeterminate, Atypical |
| TB | — | No fine labels available |

For each label $\ell$ and latent $k$, the same 5-fold stratified CV AUROC is computed as in §5. For binary labels the AUROC is computed directly; for SIIM's 4-class label the macro OvR formulation from §5.4 is applied.

**Precondition:** Fine labels are aligned to activations via `image_stems` — filename identifiers embedded in the safetensors metadata that map each row to the original DICOM file. In the current Matryoshka L3 run, `image_stems` are absent from the metadata, so all fine probe values are NaN. Fix: re-harvest with `image_stems` embedded.

---

## 7. Analysis B — Illness-Stratified Spatial Grounding

### 7.1 Objective

For images in illness group $g$, test whether each confirmed latent $k$ fires preferentially over the pathological region. This is measured as the Pearson correlation between per-patch activation and per-patch pathology coverage.

### 7.2 Bounding Box Overlap

Ground-truth pathology locations are available for RSNA (pneumonia bounding boxes) and SIIM-COVID (opacity bounding boxes) as pixel-level coordinates $(x_{\min}, y_{\min}, x_{\max}, y_{\max})$ in the original image space.

For image $n$ with $B_n$ bounding boxes $\{b_1, \ldots, b_{B_n}\}$, the **fractional overlap** of patch $p$ (occupying pixels $[u_0, u_1) \times [v_0, v_1)$ in the $224 \times 224$ image) with the union of all bounding boxes is:

$$o_{n,p} = \frac{\text{Area}\!\left( [u_0, u_1) \times [v_0, v_1) \cap \bigcup_{b=1}^{B_n} b \right)}{\text{Area}\!\left( [u_0, u_1) \times [v_0, v_1) \right)}$$

Each patch covers $32 \times 32$ pixels ($224 / 7 = 32$). The overlap $o_{n,p} \in [0, 1]$ is 0 for patches with no pathology and 1 for patches fully inside the annotated region.

For images in group `no_finding` (no annotations), $o_{n,p} = 0$ for all patches.

### 7.3 Pearson Correlation

For illness group $g$ on dataset $d$, flatten the patch-level activation and overlap arrays across all images in the group:

$$\mathbf{a}_{k,g,d} = \left[ \tilde{Z}^d_{\text{patches}}[n, p, k] : n \in \mathcal{N}^d_g,\; p \in \{1,\ldots,P\} \right] \in \mathbb{R}^{|\mathcal{N}^d_g| \cdot P}$$

$$\mathbf{o}_{g,d} = \left[ o_{n,p} : n \in \mathcal{N}^d_g,\; p \in \{1,\ldots,P\} \right] \in \mathbb{R}^{|\mathcal{N}^d_g| \cdot P}$$

The Pearson correlation is:

$$r_{k,g,d} = \frac{\sum_i \left( a_i - \bar{a} \right)\left( o_i - \bar{o} \right)}{\sqrt{\sum_i (a_i - \bar{a})^2} \cdot \sqrt{\sum_i (o_i - \bar{o})^2}}$$

where the sums are over all $|\mathcal{N}^d_g| \cdot P$ patch-image pairs.

If $\text{std}(\mathbf{a}_{k,g,d}) < 10^{-9}$ (latent never fires on any patch in the group) or $\text{std}(\mathbf{o}_{g,d}) < 10^{-9}$ (no bounding boxes in the group, all patches have $o = 0$), the correlation is undefined and reported as NaN.

A two-sided p-value is computed under the null hypothesis $r = 0$, using the $t$-distribution with $|\mathcal{N}^d_g| \cdot P - 2$ degrees of freedom:

$$t = r_{k,g,d} \sqrt{\frac{|\mathcal{N}^d_g| \cdot P - 2}{1 - r_{k,g,d}^2}}$$

### 7.4 Interpretation

A confirmed latent with $r_{k,g,d} > 0$ and $p < 0.05$ for illness group $g$ fires preferentially over pathological regions in images of that group. This validates that the latent encodes the spatial location of the pathology, not just its presence.

**Current status:** All spatial grounding values are NaN for the Matryoshka L3 evaluation because `image_stems` are absent from the metadata — the bbox lookup receives sequential placeholder stems ("0", "1", ...) that do not match any dictionary key, producing $\mathbf{o}_{g,d} = \mathbf{0}$ and $\text{std} < 10^{-9}$.

---

## 8. Output Files

| File | Shape / Content |
|------|----------------|
| `illness_activation_profiles.csv` | Rows: (latent_idx, illness_group, dataset). Columns: mean_activation, activation_freq, n_images |
| `illness_latent_matrix.csv` | 7 × 252 mean activation, averaged across datasets. Rows = illness groups, columns = confirmed latent indices |
| `illness_latent_summary.json` | Top-10 latent indices per illness group (ranked by mean activation — caveat: dominated by backbone latents) |
| `illness_discriminability_probe_coarse.csv` | (latent_idx, dataset) → auroc_mean, auroc_std |
| `illness_discriminability_probe_fine.csv` | (latent_idx, dataset, label_name) → auroc_mean, auroc_std |
| `illness_spatial_grounding.csv` | (latent_idx, illness_group, dataset) → pearson_r, pval, n_images |
| `illness_eval_summary.json` | Run metadata: timestamp, n_confirmed_latents, checkpoint, warnings |

---

## 9. Discriminability Thresholds and Taxonomy

Based on the resulting AUROC distribution over the 252 confirmed latents, a four-tier taxonomy is proposed:

| Tier | Criterion | Interpretation |
|------|-----------|---------------|
| **1 — Backbone** | Activation freq ≥ 0.99 on every image; AUROC ≈ 0.54–0.59 | Domain signal ("this is a chest X-ray"); not clinically discriminative |
| **2 — Healthy-lung** | Higher activation freq and mean on `no_finding` than on sick groups; mean AUROC > 0.60 on ≥ 2 datasets | Encodes healthy / normal lung state |
| **3 — Pathology-presence** | Higher activation on sick groups than `no_finding`; mean AUROC > 0.60 on ≥ 2 datasets | Encodes presence of some pathology |
| **4 — Illness-specific** | High z-score differential: $\Delta_{g,k} = \mu_{g,k} - \frac{1}{6}\sum_{g' \neq g} \mu_{g',k}$ peaks for one illness group | Candidate for encoding a specific pathological pattern |

The z-score differential $\Delta_{g,k}$ measures, in absolute activation units, how much more strongly latent $k$ fires for illness group $g$ than for the average of all other groups across all datasets. Large $|\Delta_{g,k}|$ identifies standout group-latent pairs for qualitative visual inspection.

---

## 10. Design Decisions and Rationale

### Why max-pooling over mean-pooling?

Pathological findings are spatially localised. For a 224×224 image divided into 7×7 = 49 patches, a focal consolidation occupying, e.g., 10% of the lung field activates 3–5 patches. Mean-pooling across 49 patches dilutes this signal by a factor of roughly 10. Max-pooling preserves the peak signal and is equivalent to asking "does this latent fire anywhere on this image?" — the natural binary question for presence/absence detection.

### Why logistic regression over a nonlinear classifier?

A linear classifier (logistic regression) tests whether the scalar latent activation is *directly* informative, without any feature engineering. A nonlinear classifier could achieve higher AUROC by thresholding or binning the activation value, but this would not confirm that the latent encodes a semantically grounded clinical concept in its activation magnitude. The linear probe is standard in mechanistic interpretability (Alain & Bengio, 2017) as a measure of the information content of a representation.

### Why class-balanced weighting?

Chest X-ray datasets are heavily imbalanced — `no_finding` typically represents 40–60% of images while rare conditions like `pleural_abnormality` may represent < 5%. Without class balancing, a classifier that always predicts `no_finding` achieves high accuracy but fails the rare classes. Balancing ensures that every illness group contributes equally to the macro AUROC, making it a fair measure of multi-class discriminability.

### Why stratified cross-validation?

TB has only ~700 images. Without stratification, random splitting could put all TB `pleural_abnormality` images (< 10) in a single fold, causing fold-skipping and a high-variance AUROC estimate. Stratification guarantees that group proportions are approximately preserved in every fold, and the 5-fold scheme provides an estimate of generalisation error without withholding too much data from any training fold.

---

## 11. References

- Alain, G., & Bengio, Y. (2017). **Understanding Intermediate Layers Using Linear Classifier Probes.** *ICLR Workshop.* https://arxiv.org/abs/1610.01644
- Hansell, D. M., et al. (2008). **Fleischner Society: Glossary of Terms for Thoracic Imaging.** *Radiology, 246*(3), 697–722.
- DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988). **Comparing the Areas under Two or More Correlated Receiver Operating Characteristic Curves.** *Biometrics, 44*(3), 837–845.
- Irvin, J., et al. (2019). **CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison.** *AAAI 2019.* https://arxiv.org/abs/1901.07031
- Vokinger, K. N., et al. (2021). **SIIM-FISABIO-RSNA COVID-19 Detection Challenge.** *Kaggle.*
- Gao, L., et al. (2024). **Scaling and Evaluating Sparse Autoencoders.** *arXiv:2406.04093.* https://arxiv.org/abs/2406.04093
