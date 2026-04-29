# Illness-Stratified Evaluation — Methodology

**Script:** `src/scripts/illness_stratified_eval.py`  
**Pipeline step:** 05 — post-universality clinical evaluation  
**Input:** confirmed universal latents from the BatchTopK universality pipeline + merged `activations.safetensors`  
**Output:** activation profiles, discriminability probes (coarse + fine), spatial grounding, illness latent matrix

---

## 0. Context and Purpose

The universality pipeline (`src/scripts/universality_eval.py`) produces a set of **confirmed universal latents** — SAE features that are active across ≥ 2 datasets, directionally consistent in representation space, and not encoding scanner/acquisition identity. This confirms *that* the features are universal but does not say *what* they encode.

The illness-stratified evaluation answers two questions:

1. **Discriminability (Analysis A).** Can each confirmed latent distinguish between different illness groups? A latent that is universal but not discriminative encodes a generic anatomical signal rather than a clinically meaningful concept.
2. **Spatial grounding (Analysis B).** Does each confirmed latent fire at the anatomically correct location within the image? A latent that achieves high discriminability by firing broadly (e.g., at the image border) rather than over the diseased region is not truly encoding pathology.

---

## 1. Input: Confirmed Latent Set

Let the set of confirmed universal latents from the universality pipeline be

$$\mathcal{C} = \{ c_1, c_2, \ldots, c_K \}$$

where each $c_k \in \{0, 1, \ldots, d_{\text{SAE}} - 1\}$ is an index into the SAE latent dimension ($d_{\text{SAE}} = 4096$) and $K$ is the number of confirmed latents for the run being evaluated (e.g. 65 for layer-3 xray_v2).

The universality pipeline guarantees that every $c_k$ satisfies:
- **Frequency:** fires on ≥ 1% of tokens on ≥ 2 of 4 datasets
- **Directional consistency:** pairwise centroid cosine similarity ≥ 0.70 across active datasets
- **Scanner invariance:** macro AUROC for dataset-of-origin prediction ≤ 0.80 (scanner audit pre-filter)
- **Monosemanticity:** Jaccard similarity of top-50 image sets < 0.50 with every other confirmed latent (no feature splitting)

**Note on the canonical class-match gate.** The universality eval's modal-class gate resolves to `no_finding` for all confirmed latents in the current runs — a consequence of `no_finding` being the plurality class in CheXpert and RSNA. The illness-stratified eval bypasses this by computing per-group activation profiles directly, revealing whether latents activate differentially on pathological images even when the class-match gate labels them `no_finding`.

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

**RSNA label fix.** The BatchTopK harvest stores RSNA labels as all-zero (a known loader issue — `PneumoniaDataset` assigns `label=0` to every image regardless of pathology). The script detects this at runtime and remaps labels from `stage_2_train_labels.csv` via `image_stems`: RSNA Target=1 → `airspace_opacity` (index 0), Target=0 → `no_finding` (index 5).

---

## 3. Activation Loading — BatchTopK Merged Format

Unlike a per-dataset format, the BatchTopK pipeline stores a single **merged** `activations.safetensors` containing all four datasets concatenated. The script uses `--merged_activations` mode.

The merged file contains:
- `activations`: `[N_total × d_model]` — all tokens from all datasets, concatenated
- `dataset_ids`: `[N_total]` — integer 0–3 identifying which dataset each token belongs to
- `labels`: `[N_total]` — canonical group index per token (per-token repeat of the per-image label)
- metadata: `image_stems` JSON dict mapping dataset name → list of filename stems in harvest order

Per-dataset tensors are recovered by masking on `dataset_ids`. Image-level labels are extracted by taking the first token of each 49-token image block (`token_indices[::n_patches]`).

---

## 4. Patch-Level Activation Encoding

### 4.1 Notation

Each dataset $d \in \{\text{chexpert, rsna, siim\_covid, tb}\}$ contains $N_d$ images. The pre-harvest SwinTransformer layer-3 activations (at `pm_index=3`) are stored as a dense tensor of shape

$$\mathbf{A}^d \in \mathbb{R}^{(N_d \cdot P) \times d_{\text{model}}}$$

where $P = 7 \times 7 = 49$ spatial tokens per image and $d_{\text{model}} = 1024$.

### 4.2 SAE Encoding

The trained BatchTopK SAE encoder $f_{\text{enc}}: \mathbb{R}^{d_{\text{model}}} \rightarrow \mathbb{R}^{d_{\text{SAE}}}$ is applied patch-by-patch:

$$\mathbf{Z}^d = f_{\text{enc}}(\mathbf{A}^d) \in \mathbb{R}^{(N_d \cdot P) \times d_{\text{SAE}}}$$

The encoder uses `sae.encode()` (per-sample top-k), so each token independently has at most $k=32$ non-zero latents. This differs from `sae.forward()` which uses batch-level top-k and is not used at inference time here.

### 4.3 Restricting to Confirmed Latents

Only the $K$ confirmed latent indices $\mathcal{C}$ are retained:

$$\tilde{\mathbf{Z}}^d = \mathbf{Z}^d_{:, \mathcal{C}} \in \mathbb{R}^{(N_d \cdot P) \times K}$$

### 4.4 Max-Pooling to Image Level

$$\tilde{\mathbf{Z}}^d_{\text{patches}} = \tilde{\mathbf{Z}}^d.\text{reshape}(N_d, P, K) \in \mathbb{R}^{N_d \times P \times K}$$

$$\mathbf{z}^{d,\text{img}}_{n,k} = \max_{p \in \{1,\ldots,P\}} \tilde{Z}^d_{\text{patches}}[n, p, k] \in \mathbb{R}^{N_d \times K}$$

**Rationale for max-pooling.** Pathological findings are spatially localised. Mean-pooling across 49 patches dilutes the signal over non-pathological patches; max-pooling captures whether the latent fires anywhere on this image, which is the natural question for presence/absence detection.

---

## 5. Activation Profiles

Before fitting any probe, per-group activation statistics are computed as a descriptive summary.

### 5.1 Mean Activation

For illness group $g$, dataset $d$, and confirmed latent $k$:

$$\mu^d_{g,k} = \frac{1}{|\mathcal{N}^d_g|} \sum_{n \in \mathcal{N}^d_g} z^{d,\text{img}}_{n,k}$$

where $\mathcal{N}^d_g = \{ n : y^d_n = g \}$ is the set of images in group $g$ for dataset $d$.

### 5.2 Activation Frequency

$$\phi^d_{g,k} = \frac{1}{|\mathcal{N}^d_g|} \sum_{n \in \mathcal{N}^d_g} \mathbf{1}[z^{d,\text{img}}_{n,k} > 0]$$

### 5.3 Illness Latent Matrix

The 7 × K **illness latent matrix** $\mathbf{M} \in \mathbb{R}^{7 \times K}$ is the mean activation averaged across datasets:

$$M_{g,k} = \frac{1}{|\mathcal{D}_{g,k}|} \sum_{d \in \mathcal{D}_{g,k}} \mu^d_{g,k}$$

**Limitation.** Raw activation magnitude is biased toward latents that fire broadly. The correct primary discriminability metric is AUROC. The matrix is most useful for computing the pathology-preference ratio (§5.4).

### 5.4 Pathology-Preference Ratio

$$\rho_k = \frac{\frac{1}{6} \sum_{g \neq 5} M_{g,k}}{M_{5,k}}$$

where group 5 is `no_finding`. Latents with $\rho_k > 1$ fire more on sick images; latents with $\rho_k < 1$ prefer healthy images. This ratio is a practical ranking of latents independent of the canonical class-match gate.

---

## 6. Analysis A1 — Coarse Discriminability Probe

### 6.1 Objective

For each confirmed latent $k$ and dataset $d$, estimate how well the scalar feature $z^{d,\text{img}}_{n,k}$ can distinguish between the 7 canonical illness groups as a **one-vs-rest multiclass classifier**.

### 6.2 Feature Construction

Each image is represented by a single scalar:

$$x_n = z^{d,\text{img}}_{n,k} \in \mathbb{R}$$

The scalar design is intentional: if a single SAE latent truly encodes a clinical concept, it should have discriminative power on its own.

### 6.3 Classifier

A **logistic regression** classifier with one-vs-rest (OvR) decomposition:

$$\hat{P}(y = g \mid x) = \sigma(w_g x + b_g), \quad g \in \{0, \ldots, 6\}$$

with **class-weight balancing** so that rare pathology classes contribute proportionally rather than being dominated by `no_finding`.

### 6.4 Cross-Validation Protocol

**5-fold stratified cross-validation** with `random_state=42`. Folds with fewer than 10 training samples in any class represented in the test set are skipped.

Metric: macro-averaged OvR AUROC per fold, averaged across valid folds.

**Baseline:** AUROC = 0.5 for a random or uniformly-firing classifier.

---

## 7. Analysis A2 — Fine-Grained Per-Label Probe

The coarse probe uses 7-group canonical labels. The fine-grained probe uses **per-dataset clinical labels** at full granularity:

| Dataset | Label type | Labels |
|---------|-----------|--------|
| CheXpert | 14 binary labels | No Finding, Cardiomegaly, Lung Opacity, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion, Pleural Other, Fracture, Support Devices, Lung Lesion, Enlarged Cardiomediastinum |
| RSNA | Binary | Pneumonia (0/1) |
| SIIM-COVID | 4-class | Negative, Typical, Indeterminate, Atypical |
| TB | — | No fine labels available |

For each label $\ell$ and latent $k$, the same 5-fold stratified CV AUROC is computed. Labels are aligned to activations via `image_stems` from the safetensors metadata. The BatchTopK merged file embeds `image_stems`, so alignment works correctly for CheXpert and RSNA.

---

## 8. Analysis B — Illness-Stratified Spatial Grounding

### 8.1 Objective

For images in illness group $g$, test whether each confirmed latent $k$ fires preferentially over the pathological region. Computing this per illness group (rather than pooled across all images) avoids conflating the `no_finding` population — which has no bounding boxes — with the opacity population.

### 8.2 Bounding Box Overlap

Ground-truth pathology locations are available for RSNA (pneumonia bounding boxes) and SIIM-COVID (opacity bounding boxes).

For image $n$ with $B_n$ bounding boxes, the **fractional overlap** of patch $p$ with the union of all bounding boxes is:

$$o_{n,p} = \frac{\text{Area}\!\left( \text{patch}_p \cap \bigcup_{b=1}^{B_n} b \right)}{\text{Area}\!\left( \text{patch}_p \right)} \in [0, 1]$$

Each patch covers $32 \times 32$ pixels ($224 / 7 = 32$). Bounding box coordinates are in original DICOM pixel space and scaled per-image to the $224 \times 224$ input.

### 8.3 Pearson Correlation

For illness group $g$ on dataset $d$, flatten patch activations and overlaps across all images in the group and compute:

$$r_{k,g,d} = \text{Pearson}\!\left( \mathbf{a}_{k,g,d},\; \mathbf{o}_{g,d} \right)$$

A two-sided p-value is computed via `scipy.stats.pearsonr` with $|\mathcal{N}^d_g| \cdot P - 2$ degrees of freedom. Correlations are NaN when either std < 10⁻⁹ (latent silent or no bounding boxes in group).

### 8.4 Interpretation

$r_{k,g,d} > 0$ with $p < 0.05$ means the latent fires preferentially at annotated pathology locations within that illness group — the strongest form of evidence for genuine pathology encoding.

---

## 9. Output Files

| File | Shape / Content |
|------|----------------|
| `illness_activation_profiles.csv` | Rows: (latent_idx, illness_group, dataset). Columns: mean_activation, activation_freq, n_images |
| `illness_latent_matrix.csv` | 7 × K mean activation, averaged across datasets |
| `illness_latent_summary.json` | Top-10 latent indices per illness group (by mean activation) |
| `illness_discriminability_probe_coarse.csv` | (latent_idx, dataset) → auroc_mean, auroc_std |
| `illness_discriminability_probe_fine.csv` | (latent_idx, dataset, label_name) → auroc_mean, auroc_std |
| `illness_spatial_grounding.csv` | (latent_idx, illness_group, dataset) → pearson_r, pval, n_images |
| `illness_eval_summary.json` | Run metadata: timestamp, n_confirmed_latents, checkpoint, warnings |

---

## 10. Discriminability Taxonomy

Based on the pathology-preference ratio $\rho_k$ and AUROC:

| Tier | Criterion | Interpretation |
|------|-----------|---------------|
| **1 — Geometry/framing** | $\rho_k < 1.0$ | Fires more on healthy images; encodes image framing or border geometry |
| **2 — Anatomy/background** | $1.0 \leq \rho_k < 2.0$; AUROC ≈ 0.5 | Fires uniformly; domain signal without clinical specificity |
| **3 — Pathology-presence** | $2.0 \leq \rho_k < 3.5$; AUROC > 0.55 | Fires more on sick images globally; not spatially localised |
| **4 — Pathology-localised** | $\rho_k \geq 3.5$; spatial r > 0 on airspace_opacity | Fires on sick images AND at the annotated lesion location |

---

## 11. Design Decisions and Rationale

### Why max-pooling over mean-pooling?

A focal consolidation activates 3–5 of 49 patches. Mean-pooling dilutes this by ~10×; max-pooling preserves the peak and answers "does this latent fire anywhere on this image?"

### Why logistic regression?

A linear probe tests whether the scalar latent activation is *directly* informative — standard in mechanistic interpretability (Alain & Bengio, 2017). A nonlinear classifier could threshold the activation value but would not confirm semantically grounded magnitude encoding.

### Why class-balanced weighting?

`no_finding` represents ~75% of RSNA images. Without balancing, a classifier predicting `no_finding` everywhere achieves high accuracy but zero discriminability for pathological classes.

### Why stratified cross-validation?

TB has ~662 images total with few examples of some canonical groups. Stratification preserves group proportions per fold, preventing pathology classes from being absent from test folds.

---

## 12. References

- Alain, G., & Bengio, Y. (2017). **Understanding Intermediate Layers Using Linear Classifier Probes.** *ICLR Workshop.* https://arxiv.org/abs/1610.01644
- Hansell, D. M., et al. (2008). **Fleischner Society: Glossary of Terms for Thoracic Imaging.** *Radiology, 246*(3), 697–722.
- DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988). **Comparing the Areas under Two or More Correlated Receiver Operating Characteristic Curves.** *Biometrics, 44*(3), 837–845.
- Irvin, J., et al. (2019). **CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison.** *AAAI 2019.* https://arxiv.org/abs/1901.07031
- Gao, L., et al. (2024). **Scaling and Evaluating Sparse Autoencoders.** *arXiv:2406.04093.* https://arxiv.org/abs/2406.04093
