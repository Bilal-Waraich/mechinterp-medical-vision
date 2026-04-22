# Chest X-Ray Datasets for Cross-Disease SAE

Four datasets used for SAE activation harvesting across diverse chest pathologies.

---

## X-Ray View Directions

**PA (Posteroanterior):** Beam travels back → front. Patient stands facing the detector with chest pressed against it, source ~1.8m behind. Minimises cardiac magnification and gives the most geometrically accurate view of heart size and lung fields. Standard view for outpatient chest X-rays.

**AP (Anteroposterior):** Beam travels front → back. Used for bedridden/ICU patients (portable X-ray machine). Heart appears larger (magnified) because it is farther from the detector. Generally lower image quality.

Mixing PA and AP introduces a systematic confound — the model can learn "AP = sicker patient" rather than actual pathology. CheXpert and SIIM-COVID therefore filter to **PA-only**.

---

## 1. CheXpert

| Property | Detail |
|---|---|
| **Path** | `//fileserver/share/deep_learning/input/data/mmm/chexpert/CheXpert-v1.0` |
| **Source** | Stanford Hospital (Irvin et al. 2019) |
| **Image volume** | ~223,414 training images |
| **Format** | JPEG |
| **Resolution** | Resized to 224×224 for training |
| **View direction** | **Frontal PA only** (lateral and AP excluded to avoid AP/PA geometric confound) |
| **Label type** | Multi-label, 14 pathology classes per image |
| **Label encoding** | Positive=1, Negative=0, Uncertain=−1 (treated as 0) |

**Pathology labels (14):**
- No Finding
- Enlarged Cardiomediastinum
- Cardiomegaly
- Lung Opacity
- Lung Lesion
- Edema
- Consolidation
- Pneumonia
- Atelectasis
- Pneumothorax
- Pleural Effusion
- Pleural Other
- Fracture
- Support Devices

**Canonical group mapping (TD-3):**
| Group | Pathologies |
|---|---|
| `airspace_opacity` | Consolidation, Lung Opacity, Lung Lesion, Pneumonia |
| `pleural` | Pleural Effusion, Pleural Other |
| `cardiac_mediastinal` | Cardiomegaly, Enlarged Cardiomediastinum |
| `atelectasis` | Atelectasis |
| `vascular_edema` | Edema |
| `no_finding` | No Finding |
| `other` | Pneumothorax, Fracture, Support Devices |

**Notes:** Largest and most label-rich dataset in the suite. Hierarchical directory structure: `patient*/study*/view*.jpg`. Also has a small version and a localization subset.

---

## 2. RSNA Pneumonia Detection Challenge

| Property | Detail |
|---|---|
| **Path** | `//fileserver/share/deep_learning/input/data/mmm/rsna-pneumonia-detection-challenge/stage_2_train_images` |
| **Source** | Radiological Society of North America (Kaggle challenge) |
| **Image volume** | ~26,000 images |
| **Format** | DICOM (`.dcm`) |
| **Resolution** | Native ~1024×1024 DICOM, resized to 224×224 |
| **View direction** | Frontal (predominantly PA) |
| **Label type** | Binary (pneumonia / normal) + **bounding boxes** for opacity regions |
| **Label encoding** | `PneumoniaDataset` loads every image with label=0 (placeholder). Downstream analysis reads the real canonical labels from `stage_2_train_labels.csv` via the `--label_csv rsna:...` flag to `universality_eval.py` and joins them to activations by filename stem (see `docs/methodology.md` §1.6). |

**Classes:**
- Normal
- Pneumonia (with bounding box annotations for opacity location)

**Notes:** Labels are stored in a companion CSV separate from the DICOM files; the current loader harvests all images as unlabelled (label=0). Bounding boxes provide spatial localization. Replaces the simpler ChestXRay2017 dataset which only has 2 classes and ~5k images.

---

## 3. SIIM-COVID19 Detection Challenge

| Property | Detail |
|---|---|
| **Path** | `//fileserver/share/deep_learning/input/data/mmm/siim-covid19-detection` |
| **Source** | Society for Imaging Informatics in Medicine (SIIM) / Kaggle challenge |
| **Image volume** | ~6,000 studies (~one image per study) |
| **Format** | DICOM (`.dcm`) |
| **Resolution** | Native ~1024×1024 DICOM, resized to 224×224 |
| **View direction** | **Frontal PA** (filtered where view metadata available) |
| **Label type** | Study-level (4 classes) + image-level **bounding boxes** for opacities |

**Study-level labels (4):**
- Negative for Pneumonia → canonical: `no_finding`
- Typical Appearance (classic COVID opacity) → canonical: `airspace_opacity`
- Indeterminate Appearance → canonical: `airspace_opacity`
- Atypical Appearance → canonical: `airspace_opacity`

**Bounding box annotations:**
- Image-level boxes marking opacity regions (space-separated `x_min y_min x_max y_max label` format)
- Coordinates in original DICOM pixel space (~1024×1024); scale factors computed per-image for 224×224 mapping
- Available via `dataset.bbox_records` for patch-level validation

**Notes:** COVID-19 opacities (ground-glass, consolidation) overlap heavily with other airspace opacity causes — useful for testing whether the SAE separates COVID-specific features from generic opacity features. Two-level annotation (study + image) allows both classification and localisation tasks.

---

## 4. Tuberculosis — Shenzhen

| Property | Detail |
|---|---|
| **Path** | `//fileserver/share/deep_learning/input/data/mmm/tuberculosis-chest-xrays-shenzhen` |
| **Source** | Shenzhen No.3 People's Hospital, China |
| **Image volume** | ~662–700 images |
| **Format** | PNG (filename-encoded labels: `CHNCXR_XXXX_0.png` = normal, `CHNCXR_XXXX_1.png` = TB) |
| **Resolution** | Resized to 224×224 |
| **View direction** | Frontal |
| **Label type** | Binary |
| **Class balance** | ~340 TB-positive, ~360 normal (near-balanced) |

**Classes:**
- Normal (suffix `_0`) → canonical: `no_finding`
- TB-positive (suffix `_1`) → canonical: `airspace_opacity` (infiltrates, cavities)

**Augmentation (used during harvest to compensate for small size):**
- Random horizontal flip (p=0.5)
- ±10% brightness jitter
- Random crop (scale 0.85–1.0) → resize to 224×224

**Notes:** Deliberately small and disease-specific — acts as an outlier probe. If the SAE learns modular features, TB-specific neurons should be separable from the shared pneumonia/effusion/atelectasis modules learned across the other three datasets. Requires ~14× oversampling to match larger datasets in stratified batching.

---

## Summary Comparison

| Dataset | Images | View | Label Type | Disease Focus | Format |
|---|---|---|---|---|---|
| CheXpert | ~223,414 | PA frontal only | Multi-label (14) | General chest pathology | JPEG |
| RSNA Pneumonia | ~26,000 | Frontal | Binary + bboxes | Pneumonia detection | DICOM |
| SIIM-COVID19 | ~6,000 | PA frontal | 4-class + bboxes | COVID-19 opacities | DICOM |
| TB Shenzhen | ~662 | Frontal | Binary | Tuberculosis | PNG |

**Total (deduplicated):** ~256,000 images across 4 datasets

---

## Canonical Group Alignment

All four datasets map to the same 7-group canonical label space (TD-3):

| Index | Group | Datasets using it |
|---|---|---|
| 0 | `airspace_opacity` | CheXpert, RSNA, SIIM-COVID19, TB (TB+) |
| 1 | `pleural` | CheXpert |
| 2 | `cardiac_mediastinal` | CheXpert |
| 3 | `atelectasis` | CheXpert |
| 4 | `vascular_edema` | CheXpert |
| 5 | `no_finding` | All four |
| 6 | `other` | CheXpert (Pneumothorax, Fracture, Support Devices) |

---

## Notes & Caveats

- **ChestXRay2017** (Kaggle pneumonia vs normal, ~5k images) was originally considered but **replaced** by RSNA Pneumonia, which has bounding box annotations and 5× more images.
- **VinBigData** was swapped out in favour of SIIM-COVID19 to get a temporally distinct pathogen (COVID-19) with bounding box annotations and cleaner label structure.
- All images are normalised to `[0, 1]` float32, NOT ImageNet-normalised. UMedPT encoder was trained on `[0, 1]` intensities.
- All images are grayscale X-rays replicated to 3-channel RGB as required by the encoder.
- CheXpert uncertain labels (`−1`) are treated as negative (conservative choice).
- SIIM-COVID19 opacities overlap with CheXpert/RSNA pneumonia opacities — a key test of whether the SAE disentangles COVID-specific from generic airspace opacity features.
