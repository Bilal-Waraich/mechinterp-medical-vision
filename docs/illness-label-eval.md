# Illness-Label Evaluation of Universal SAE Latents

## Motivation

The universality pipeline (`universality_eval.py`) confirms 188 latents that activate consistently across chest X-ray datasets.  
What it does **not** yet answer: *do these latents encode medically meaningful pathology, or just low-level visual structure?*

The supervisor asked to validate the confirmed universal latents against illness labels and spatial annotations. This document describes two complementary analyses added in `src/scripts/illness_stratified_eval.py`.

---

## Analysis A — Fine-Grained Classification Labels

### What

Run illness-discriminability probes at two label granularities:

| Granularity | Labels used | Datasets |
|---|---|---|
| Coarse (7 canonical groups) | Canonical group index already in safetensors `"labels"` tensor | All 4 X-ray datasets |
| Fine-grained | CheXpert: 14 pathology columns (multi-label); SIIM-COVID: 4 study labels; RSNA: binary; TB: binary | Loaded from label CSVs on Gaia |

For each granularity and each confirmed universal latent, train a one-vs-rest logistic regression using per-image max activation (single feature) to predict illness label. Report per-class AUROC.

### Why fine-grained matters

The canonical groups intentionally collapse related conditions (e.g. COVID-typical opacity, tuberculosis infiltrate, and bacterial pneumonia all map to `airspace_opacity`). The fine-grained labels let us check whether a latent that fires for `airspace_opacity` is tracking a specific sub-type (e.g. COVID opacity only) or genuinely represents the broader radiological pattern.

### Label sources

| Dataset | Fine-grained label | CSV path on Gaia |
|---|---|---|
| CheXpert | 14 pathology columns (binary, uncertain→0) | `$SAE_INPUT_ROOT/chexpert/train.csv` |
| SIIM-COVID | Negative / Typical / Indeterminate / Atypical | `$SAE_INPUT_ROOT/siim-covid19-detection/train_study_level.csv` |
| RSNA | 0=normal, 1=pneumonia | `$SAE_INPUT_ROOT/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv` |
| TB | 0=normal, 1=TB positive | Encoded in filename; no extra CSV needed |

### Output

```
illness_activation_profiles.csv          — mean act + freq per (latent, illness, dataset)
illness_discriminability_probe_coarse.csv  — AUROC with 7 canonical groups
illness_discriminability_probe_fine.csv    — AUROC with fine-grained labels
illness_latent_summary.json              — top-k latents per illness, matrix for heatmap
```

---

## Analysis B — Illness-Stratified Spatial Grounding

### What

The existing `run_spatial_grounding` step computes, for each confirmed latent, the Pearson *r* between per-patch activation magnitude and the fraction of a bounding box covered by that patch — pooled across all images.

Analysis B **stratifies this by illness group**: compute the spatial Pearson *r* separately for images belonging to each canonical illness group.

Datasets with bounding box annotations: RSNA (pneumonia bboxes) and SIIM-COVID (opacity bboxes).

### Why this matters

A latent with:
- High AUROC for `airspace_opacity` (from Analysis A) **and**
- High spatial Pearson *r* within `airspace_opacity` images (from Analysis B)

is strong evidence that the latent encodes opacity *spatially* — it fires on the pathology region, not just anywhere in an opacity image. This closes the loop between classification signal and spatial localisation.

Latents that score high on one but not the other reveal an interesting distinction:
- High AUROC + low spatial r → latent encodes a global image-level cue (e.g. overall lung texture)  
- Low AUROC + high spatial r → latent fires on a spatially consistent region that does not discriminate illness classes (e.g. cardiac silhouette)

### Spatial grounding method (unchanged from existing pipeline)

For each (image, patch):
```
overlap(patch, i) = area(patch ∩ union_of_bboxes) / area(patch)
```
Then compute Pearson *r* between `latent_activation[patch]` and `overlap[patch]` across all patches in all images of a given illness group.

### Output

```
illness_spatial_grounding.csv    — Pearson r per (latent, illness_group, dataset)
```

Columns: `latent_idx`, `illness_group`, `dataset`, `pearson_r`, `n_images`, `n_patches_with_bbox`.

---

## Implementation Notes

### New script

`src/scripts/illness_stratified_eval.py` — standalone post-hoc analysis. No SAE training; reads saved activations + confirmed latents JSON.

Mirrors the CLI of `universality_eval.py` (same `--activations PATH:LABEL` / `--merged_activations` pattern). Reuses:
- `load_sae()` from `src/visualisation/loaders.py`
- `CANONICAL_GROUPS`, `load_bbox_csv()`, `compute_patch_bbox_overlap()` from `universality_eval.py`
- Label CSV parser extended to return fine-grained labels

### Key data layout

```
safetensors:
  "activations"  [N_images × 49, d_model]   float32
  "labels"       [N_images]                  int32   (canonical group 0–6, -1=unknown)
```

Per-image max activation (used as probe feature):
```python
z = sae.encode(acts)                             # [N_images*49, d_sae]
z_img = z.reshape(N_images, 49, d_sae).amax(1)   # [N_images, d_sae]
```

### Edge cases

| Case | Handling |
|---|---|
| RSNA `"labels"` tensor all zeros (known issue) | Auto-detected; labels loaded from bbox CSV instead |
| Label `-1` (unknown) | Excluded from all stats via `valid_mask = labels != -1` |
| Illness group absent from dataset | NaN recorded, not propagated as zero |
| `class_weight='balanced'` | Used in all logistic regressions to handle imbalance |

### Output directory

```
<experiment_output_dir>/illness/
    illness_activation_profiles.csv
    illness_discriminability_probe_coarse.csv
    illness_discriminability_probe_fine.csv
    illness_spatial_grounding.csv
    illness_latent_summary.json
    illness_latent_matrix.csv
    illness_eval_summary.json
```
