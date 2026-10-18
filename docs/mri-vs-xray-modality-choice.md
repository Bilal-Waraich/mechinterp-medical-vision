# MRI vs X-ray: Modality Choice for Single-Modality Universality SAE

The core experiment: train a SAE on activations from a single imaging modality
across multiple disease datasets, and ask whether shared latent features correspond
to universal semantic concepts rather than modality-specific artefacts.

---

## Decision Criteria

### 1. Semantic richness of encoder representations

**MRI wins.**

MRI forces the encoder to learn tissue-property distinctions:
- T2 hyperintensity (water content, oedema)
- Diffusion restriction (ADC maps — cell density, infarct)
- T1 contrast enhancement (blood-brain barrier breakdown)

These are biologically grounded signal differences, not just brightness variations.
SAE features that emerge at Stage 3→4 are more likely to correspond to real
biological concepts that a clinician would recognise.

X-ray intensity is a direct projection of attenuation — primarily density and
thickness. It encodes less tissue-specific information, and much of the image
structure reflects anatomy (ribs, clavicles, diaphragm) rather than pathology.

---

### 2. Encoder representation quality

**X-ray wins.**

UMedPT was pre-trained on a massive multi-modal corpus. X-ray is the dominant
modality in public medical imaging datasets. The encoder has seen orders of
magnitude more X-ray data than MRI during pre-training. X-ray representations
are likely better calibrated — more features, more reliable activations, higher
explained variance in SAE reconstructions.

A universality finding on X-ray is arguably stronger evidence precisely because
the encoder had more opportunity to specialise for X-ray during training. Shared
features that survive despite abundant modality-specific data are genuinely robust.

---

### 3. Dataset diversity and disease coverage

**Roughly equal; different flavours.**

| MRI datasets available | Disease |
|---|---|
| BraTS 2020 | Glioma, oedema |
| MSseg2 | Multiple sclerosis lesions |
| MSD Task04 Hippocampus | Structural atrophy |
| MSD Task05 Prostate | Prostate cancer zones |
| SPIDER | Lumbar disc/vertebra pathology |
| PI-CAI | Prostate cancer (T2/ADC/DWI) |
| MSD Task02 Heart | Left atrium |

| X-ray datasets available | Disease |
|---|---|
| ChestXRay2017 (NIH) | 14 pathology labels inc. effusion, pneumonia, nodules |
| CheXpert (Stanford) | 14 labels, uncertainty annotations |
| PLCO | Lung cancer screening |
| siim-covid19 | COVID pneumonia |
| rsna-pneumonia | Pneumonia |
| tuberculosis-shenzhen | TB |
| VinBigData | 14 pathology classes with bounding boxes |

MRI disease diversity is higher across body systems (brain, spine, prostate,
cardiac). X-ray disease diversity is higher within the chest, with more datasets
and more images per disease class.

---

### 4. Independent samples per compute budget

**X-ray wins.**

MRI volumes give many slices, but adjacent slices share 90%+ of their content —
they are not independent samples. At `batch_size=2048` with a Swin patch grid,
you are drawing largely autocorrelated samples from a small set of volumes.

Each chest X-ray is a genuinely independent image from an independent patient.
X-ray provides much higher effective sample diversity per training step.

---

### 5. Ground truth quality for validation

**MRI wins.**

| Annotation type | MRI | X-ray |
|---|---|---|
| Voxel/pixel masks | Most datasets (BraTS, MSD, SPIDER) | Rare (only ChestX-Det, CheXmask) |
| Bounding boxes | Some | VinBigData, RSNA-Pneumonia |
| Image-level labels | All | Most (CheXpert, ChestXRay2017) |

For patch-level validation of SAE features (does this latent fire on lesion
patches?), voxel masks are essential. MRI provides these; most X-ray datasets do
not. This makes quantitative SAE interpretability substantially harder with X-ray.

---

### 6. Confound severity

**MRI has fewer confounds overall; X-ray has more but they are testable.**

| Confound | MRI | X-ray |
|---|---|---|
| Scanner/institution artefacts (#1) | High | Moderate |
| Dataset imbalance (#2) | Moderate | High |
| Magnitude mismatch (#3) | Moderate | High |
| Feature splitting (#4) | Moderate | High |
| Geometric/view bias (#5) | Low | High |
| Patch-level label noise (#10) | Low | High |

X-ray's confounds are systematic and pervasive. However, view geometry (#5) can
be turned into a deliberate probe: a universally-robust feature should fire
regardless of AP vs PA view. This makes X-ray's confounds *more useful* for
stress-testing universality claims, not just more dangerous.

---

## Recommendation

### For a clean, interpretable universality story → MRI (brain)

Proposed dataset combination:
1. **BraTS 2020** — glioma/oedema (mass lesion, fast-growing)
2. **MSseg2** — MS lesions (diffuse white matter, slow-evolving)
3. **MSD Hippocampus** — structural atrophy (no lesion, shape-based)
4. **SPIDER** — lumbar spine (same MRI physics, completely different anatomy)

This 4-way split gives two clean axes:
- **Pathology type:** mass vs diffuse vs structural
- **Anatomy:** brain vs spine

Features shared across all four are hard to explain as anything other than
genuine semantic universality.

---

### For statistical power and a stronger generalisability claim → X-ray

Proposed dataset combination (PA-only filter applied):
1. **CheXpert** — effusion, atelectasis, cardiomegaly (large, well-labelled)
2. **ChestXRay2017** — pneumonia, nodules (independent institution)
3. **VinBigData** — 14 pathologies with bounding boxes (non-Western population)
4. **tuberculosis-shenzhen** — TB (rare disease stress test)

Stratified sampling (#2) is mandatory. PA-view filtering for CheXpert and NIH
is strongly recommended to remove the AP/PA geometric confound (#5).

---

### The strongest possible experiment → Cross-modality transfer

Train the SAE on MRI brain. After training, pass chest X-ray activations through
the frozen SAE encoder and check whether known-disease latents activate on their
X-ray equivalents (e.g., does the "consolidation" MRI feature activate on
CheXpert consolidation X-rays?).

This tests a strictly stronger claim: not just within-modality universality
(expected), but cross-modality feature transfer (novel). A positive result would
be a meaningful contribution to mechanistic interpretability of foundation models.

---

## Summary

| Criterion | MRI | X-ray |
|---|---|---|
| Semantic feature richness | ✅ Higher | ❌ Lower |
| Encoder representation quality | ⚠️ Moderate | ✅ Higher |
| Sample independence | ❌ Slice autocorrelation | ✅ Independent images |
| Disease diversity | ✅ Multi-system | ✅ Multi-pathology (chest) |
| Ground truth for validation | ✅ Voxel masks | ⚠️ Mostly image-level |
| Confound severity | ✅ Lower overall | ⚠️ Higher, but testable |
| Interpretability of findings | ✅ Cleaner story | ⚠️ More caveats required |
| Statistical power | ⚠️ Moderate | ✅ More data |
