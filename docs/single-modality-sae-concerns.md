# Single-Modality SAE: Risks and Mitigations

Analysis of risks when running a SAE on a single modality (MRI or X-ray) across
multiple disease datasets. Each concern is rated by severity for each modality
and cross-referenced with the current codebase state.

---

## 1. The Scanner Latent Trap (Spurious Correlates)

**Risk:** The SAE dedicates dictionary entries to memorising the hospital/scanner
rather than the biology. A latent fires on every image from Dataset A regardless
of pathology.

| Modality | Severity | Reason |
|---|---|---|
| MRI | **High** | Field strength (1.5T vs 3T), manufacturer (Siemens/Philips/GE), pulse sequence parameters all produce systematic activation shifts. SPIDER dataset explicitly spans multiple manufacturers. |
| X-ray | **Moderate** | Less hardware variation than MRI, but institutional exposure settings, DICOM post-processing (edge enhancement, windowing), and patient population differ visibly between CheXpert (Stanford), ChestXRay2017 (NIH), and VinBigData (Vietnam). |

**Detection:** After training, fit a linear classifier from SAE latents → dataset ID.
Latents with high per-dataset predictive power are artifact features.

**Mitigation:** Tag each activation vector with its source dataset ID at harvest
time (store as a parallel metadata tensor). Audit post-hoc and lesion artifact
latents during clinical evaluation.

**Codebase status:** ⚠️ Not handled. The `.safetensors` file stores only the raw
activation tensor; source provenance is discarded at harvest time.

---

## 2. The Dominant Dataset Problem (Imbalance)

**Risk:** SAEs minimise reconstruction loss globally. A large dataset crowds out
smaller ones — the SAE optimises its dictionary for the dominant source and treats
minority datasets as noise.

| Modality | Severity | Reason |
|---|---|---|
| MRI | **Moderate** | BraTS (~369 cases × ~155 slices ≈ 57k slices) vs MSD Hippocampus (~260 cases × ~128 slices ≈ 33k slices). Imbalance is real but manageable. |
| X-ray | **High** | CheXpert has ~224k images; tuberculosis-shenzhen has ~700. A 300× imbalance means the SAE will be effectively a CheXpert SAE. Stratified sampling is non-negotiable. |

**Mitigation:** Stratified sampling at the DataLoader level — force each batch to
contain an equal number of activation vectors from each source dataset. Requires
storing dataset IDs alongside activations.

**Codebase status:** ⚠️ Not handled. `train_sae.py:179` uses a plain `TensorDataset`
with `shuffle=True`. No stratification.

---

## 3. Activation Magnitude Mismatch

**Risk:** Datasets with higher absolute activation variance dominate the
reconstruction loss, causing the SAE to ignore the structural features of
lower-variance datasets.

| Modality | Severity | Reason |
|---|---|---|
| MRI | **Moderate** | Different contrast mechanisms produce different activation scales, but RMS normalisation largely compensates. Per-dataset mean shifts remain. |
| X-ray | **High** | X-ray intensity is a direct function of exposure (kVp, mAs) and patient thickness — not tissue physics. Two hospitals imaging the same pathology will produce activations at different absolute scales. |

**Mitigation:** Per-dataset centering (subtract each dataset's mean activation
vector) before concatenation, in addition to the existing RMS normalisation.
Initialise `b_dec` to zero on already-centred data rather than the global mean.

**Codebase status:** ✅ Partially handled. `harvest_activations.py` applies
Anthropic-style RMS normalisation (`x / (rms(x) + ε) * sqrt(d_model)`) per token.
`b_dec` is initialised to the global data mean (`train_sae.py:211`). However,
systematic per-dataset mean shifts are not corrected.

---

## 4. Feature Splitting (The Opposite of Universality)

**Risk:** With a large dictionary and weak sparsity, the SAE learns separate
latents for the same pathology across datasets rather than merging them into a
single universal feature.

| Modality | Severity | Reason |
|---|---|---|
| MRI | **Moderate** | Multi-contrast MRI makes the same lesion look different across channels. Some splitting is expected; the question is whether the SAE merges or splits at the semantic level. |
| X-ray | **High** | The same pathology (e.g., pleural effusion) looks visibly different across institutions, views, and patient sizes. The SAE has more surface variation to exploit for splitting. |

**Detection:** After training, check co-activation — if Latent A fires for
pathology X in Dataset 1 and Latent B fires for pathology X in Dataset 2, and
they never co-activate, you have a split.

**Mitigation:** Tighten `l1_coeff` (try 5e-4 to 1e-3) and reduce `expansion_factor`
(try 8 → 4 or 2) to force the SAE to merge rather than split. The dead-feature
auxiliary loss (`aux_penalty`, `n_batches_to_dead`) already pushes against wasted
capacity but does not directly target splitting.

**Codebase status:** ✅ Tooling present. `l1_coeff`, `expansion_factor`, and
auxiliary loss are all configurable in `SAEConfig`. No automatic split-detection
metric is implemented.

---

## 5. Geometric and View Biases

**Risk:** The SAE learns latents encoding acquisition geometry rather than
pathology — mistaking view artefacts for disease features.

| Modality | Severity | Reason |
|---|---|---|
| MRI | **Low** | Brain MRI datasets (BraTS, MSD) use consistent axial acquisition. Spine MRI (SPIDER) mixes sagittal and axial, but geometry is less confounded with pathology than in X-ray. |
| X-ray | **High** | AP vs PA view is a strong signal the encoder encodes. Portable (AP, supine) exams make the heart appear 15-20% larger than departmental (PA, standing) — a feature labelled "enlarged heart" may actually be "AP view". Lateral views add another geometry axis. |

**Mitigation for X-ray:**
- Filter datasets to PA-only during harvest, or
- Store view metadata and run the same scanner-latent audit (#1) against view ID.
- Use view-geometry as a deliberate stress test: a truly disease-universal feature
  should fire regardless of view.

**Codebase status:** ⚠️ Not applicable for MRI brain. Not handled for X-ray.

---

## 6. Which Stage to Hook

**Risk:** Hooking too early captures local texture/geometry rather than semantic
disease representations. Hooking too late loses spatial resolution needed for
lesion localisation.

| Stage (pm_index) | d_model | Spatial | Encodes |
|---|---|---|---|
| 0 → Stage 1→2 | 256 | 28×28 | Edges, textures, local intensity |
| 1 → Stage 2→3 | 512 | 14×14 | Organ boundaries, tissue regions |
| **2 → Stage 3→4** | **1024** | **7×7** | **Semantic task features, global context** |

**Recommendation:** For disease-level universality, **pm_index=2** is the better
target. At Stage 2→3 (current default), representations still encode substantial
local texture — a glioma and an MS lesion look very different at this level.
At Stage 3→4, the encoder has made high-level "what is this region?" decisions.

**X-ray specific:** This is even more important for X-ray. Early stages strongly
encode projection geometry (rib shadows, diaphragm angle, heart border) — these
are view-dependent confounds. Stage 3→4 is more likely to represent "what disease
is present" rather than "what does this projection look like."

**Codebase status:** ✅ Configurable via `--pm_index`. Default is pm_index=1
(Stage 2→3). `SAEConfig.for_pm_index()` handles the d_model/n_patches mapping.
**Consider switching to pm_index=2 for universality experiments.**

---

## 7. Dead Features with Heterogeneous Data

**Risk:** With the current `n_batches_to_dead=10`, a disease-specific feature that
legitimately does not fire for 10 consecutive batches (because that disease is
underrepresented in those batches by chance) gets recycled prematurely.

| Modality | Severity | Reason |
|---|---|---|
| MRI | **Moderate** | Rare pathologies (hippocampal atrophy, early MS lesion) may not appear in 10 consecutive batches in a mixed dataset. |
| X-ray | **High** | Rare diseases (TB with 700 images, pneumothorax) in a mixed dataset with CheXpert will frequently go 10+ batches without a representative sample. |

**Mitigation:** Increase `n_batches_to_dead` to 25-50 when mixing datasets.
Stratified sampling (#2) also reduces this risk by ensuring rare-disease samples
appear in every batch.

**Codebase status:** ⚠️ Configurable but default (10) is too low for heterogeneous
multi-dataset training.

---

## 8. Cross-Dataset b_dec Drift

**Risk:** `b_dec` is initialised to the global activation mean. If datasets have
different mean activation directions, the global mean is a poor bias for all of
them, causing systematic reconstruction error in every dataset.

| Modality | Severity | Reason |
|---|---|---|
| MRI | **Moderate** | Healthy brain tissue vs tumour tissue have different mean activations; mixing datasets shifts the global mean away from all of them. |
| X-ray | **High** | Exposure and processing differences cause substantial mean activation direction shifts across institutions. |

**Mitigation:** Center each dataset separately before concatenation, then
initialise `b_dec` to zero (or the mean of already-centred data).

**Codebase status:** ⚠️ `b_dec` is initialised to global data mean (`train_sae.py:211`).
No per-dataset centering.

---

## 9. Encoder Fine-Tuning Confound

**Risk:** UMedPT was fine-tuned separately per downstream task. Activations
harvested from a BraTS-fine-tuned encoder will be systematically shifted relative
to activations from an MS-fine-tuned encoder. Apparent universality may reflect
the base encoder's representations, not a shared semantic structure.

| Modality | Severity | Same for both |
|---|---|---|
| MRI | **High** | Yes |
| X-ray | **High** | Yes |

**Decision required before any experiment:**
- **Harvest from the base (pretrained) encoder** — tests whether the foundation
  model learned universal representations before task-specific fine-tuning. This
  is the cleaner universality claim.
- **Harvest from each task's fine-tuned encoder** — tests whether task-specific
  adaptation preserves a shared core. Different, also valid, but confounded by
  fine-tuning magnitude differences across tasks.

**Codebase status:** ⚠️ Not addressed. The harvest script loads a single checkpoint
(`--checkpoint`). Multi-encoder harvesting is not implemented.

---

## 10. Patch-Level Label Assignment

**Risk:** SAE activations are at the patch level (14×14 or 7×7 over a 224×224
input). Without a strategy to assign ground-truth labels to patches, qualitative
validation ("does this feature fire on lesion patches?") is ambiguous.

| Modality | Severity | Reason |
|---|---|---|
| MRI | **Moderate** | Voxel-level segmentation masks are available for most datasets (BraTS, MSD, SPIDER). Downsampling the mask to the patch grid via max-pooling is straightforward. |
| X-ray | **High** | Most X-ray datasets have image-level labels (CheXpert, ChestXRay2017) or bounding boxes (VinBigData), not pixel masks. Patch label assignment requires noisy heuristics (bounding box overlap, uniform projection of image-level labels). |

**Mitigation for X-ray:** Use VinBigData bounding boxes with IoU-based patch
assignment (patch is positive if >50% overlap with pathology bounding box).
Treat image-level-only datasets as weakly-supervised validation only.

**Codebase status:** ⚠️ Not implemented. The visualisation pipeline operates on
full-image features; no patch-to-label mapping exists.

---

## Summary Table

| # | Concern | MRI severity | X-ray severity | Codebase status |
|---|---|---|---|---|
| 1 | Scanner latent trap | High | Moderate | ⚠️ Not handled |
| 2 | Dominant dataset imbalance | Moderate | **High** | ⚠️ Not handled |
| 3 | Magnitude mismatch | Moderate | **High** | ✅ Partially (RMS norm) |
| 4 | Feature splitting | Moderate | **High** | ✅ Knobs exist |
| 5 | Geometric/view bias | Low | **High** | ⚠️ Not handled (X-ray) |
| 6 | Which stage to hook | — | — | ✅ Configurable; switch to pm_index=2 |
| 7 | Dead features / heterogeneous data | Moderate | **High** | ⚠️ Default too low |
| 8 | Cross-dataset b_dec drift | Moderate | **High** | ⚠️ Not handled |
| 9 | Encoder fine-tuning confound | High | High | ⚠️ Decision required |
| 10 | Patch-level label assignment | Moderate | **High** | ⚠️ Not implemented |
