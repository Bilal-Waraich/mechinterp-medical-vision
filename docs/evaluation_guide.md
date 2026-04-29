# Evaluation & Visualisation Guide

How to read and judge the output of the full BatchTopK SAE analysis pipeline and the Phase-3 visualisations. This doc assumes the pipeline has run end-to-end (scanner audit → universality eval → null baseline → inspect latents) and the visualisation jobs have produced `vis_outputs/`.

Organised in the order you should actually look at things, with concrete metric thresholds, red flags, and what to do when a step produces something that looks off.

---

## 0. Mental model before you start

The pipeline answers three separable questions:

1. **Are the universal latents real?** (Is the SAE learning something that isn't just "random projections activate together" or "the dataset-specific scanner signature"?)
2. **What do they *mean*?** (What anatomical or clinical concept does each universal latent represent?)
3. **Do they encode pathology, or just anatomy?** (A latent can be universal and visually coherent while firing identically on healthy and sick images — illness stratification is the only way to tell.)

Steps 1–3 of the pipeline answer the first question with numbers. Step 4 and the visualisation jobs answer the second with pictures. **Step 5 (illness-stratified eval) answers the third with probes and spatial grounding.** Never skip straight to the pictures — a latent can look beautiful and still be an artefact. The null baseline and scanner audit exist specifically to keep you honest about this.

### The canonical class-match gate caveat
The universality eval labels each confirmed latent with a `matched_canonical_class`. For the current BatchTopK layer-3 runs this resolves to `no_finding` for all confirmed latents — not because the latents are uninteresting, but because `no_finding` is the plurality class in both CheXpert (~40%) and RSNA (~75%), so the top-50 activating images for any broadly-firing latent are dominated by `no_finding` by majority. **Do not interpret `matched_canonical_class = no_finding` as "this latent encodes healthy anatomy."** Use the pathology-preference ratio ρ_k from the illness report instead.

### Layer semantics
- **Layer 2** (`pm_index=2`, Swin Stage 2→3): mid-level features. Expect texture-like and local structure latents (edges of ribs, lung-field boundaries, diaphragm line).
- **Layer 3** (`pm_index=3`, final Swin stage): high-level features. Expect more class-correlated, region-level latents (whole-lung presence, heart silhouette, consolidation regions).

If L3 shows *fewer* confirmed universal latents than L2, that is expected behaviour — late layers become more task-specific.

---

## 1. Scanner audit — `scanner_audit/scanner_latent_audit.json`

**What it does:** trains a logistic probe on each latent's activation → predicts *which dataset* the image came from. Latents that can predict the dataset too well (AUROC ≥ 0.80 by default) are flagged as "scanner-correlated" and excluded from universality analysis. This stops you counting "this latent fires for RSNA PNGs vs CheXpert JPEGs" as a universal concept.

### Key fields
```json
{
  "n_total_latents": 4096,
  "n_flagged":       ???,
  "flagged_fraction": 0.xx,
  "threshold": 0.80,
  "flagged_indices": [...]
}
```

### How to read it
| `flagged_fraction` | Meaning | Action |
|---|---|---|
| < 5% | Healthy. Most latents are dataset-agnostic. | Continue. |
| 5–20% | Normal range — some site-specific features leaked through. | Continue. Check that Step 2's `n_level2_confirmed` is not dominated by flagged indices (it shouldn't be — they're excluded). |
| 20–50% | SAE is partly modelling scanner/dataset identity. | Continue, but note in report. Consider harvest-time normalisation review. |
| > 50% | Serious — the SAE is largely a scanner detector. | Investigate: check Anthropic RMS normalisation, check dataset-balanced harvesting, maybe retrain. |

### What to look for
- **Compare L2 vs L3:** L3 typically flags fewer latents (late layers are more class-driven than site-driven). Opposite pattern is a red flag.
- **Cross-reference with Step 2 confirmed set:** `set(confirmed) & set(flagged_indices)` should be **empty** by construction. If it isn't, the code has a bug.

---

## 2. Universality eval — `eval/universality_summary.json`

The centrepiece. Two gates are applied in sequence on the unflagged latents.

### Key fields
```json
{
  "d_sae":                      4096,
  "n_scanner_flagged":          <from step 1>,
  "n_level1_candidates":        ???,   // passed frequency gate
  "n_level2_confirmed":         ???,   // + cosine gate → these are "universal"
  "n_spatial_bonferroni_pass":  ???,   // passed class-correlation test (bbox or label)
  "n_spatial_bonferroni_tested": ???,
  "n_split_pairs":              ???    // pairs of latents that merge/split same concept
}
```

### Gate 1 — Frequency (Level 1)
A latent must activate on **≥ 1% of tokens in ≥ 2 datasets**. Filters dead and ultra-rare latents.

| `n_level1_candidates / d_sae` | Interpretation |
|---|---|
| < 1% | SAE is very sparse / many dead latents. If `k=48`, `d_sae=4096`, expected ~1% is fine. |
| 1–10% | Typical. |
| > 30% | SAE is under-sparsifying — everything fires everywhere. Check `k` and activation norm. |

### Gate 2 — Geometry (Level 2)
For each Level-1 candidate, compute its **d_model centroid** (mean input direction that activates it) per dataset. The latent is "confirmed universal" if the centroids are cosine-similar (≥ 0.70) across at least `min_datasets=2`. This is the stronger test — it says the latent responds to the *same direction in feature space* across datasets, not just "fires often in both."

**Key metric:** `n_level2_confirmed`.

| Value | Interpretation |
|---|---|
| 0 | Either the SAE is broken or the datasets really are disjoint. Investigate before reporting. |
| 1–20 | Small but real universal set. Normal for late layers. |
| 20–100 | Strong universality signal. Most plausible for mid layers (L2). |
| > 200 | Check: is the cosine threshold too lenient? Try raising to 0.80 and rerun. |

### Bonferroni class-correlation
For datasets with class labels or bboxes, each confirmed latent is tested: does its activation correlate with the positive class, surviving a Bonferroni correction across all confirmed latents? 

- `n_spatial_bonferroni_pass` ≥ 1 → you have at least one latent that is both universal *and* clinically meaningful. These are the ones to showcase.
- Zero passes does **not** invalidate universality — it just means none of the universal latents align 1:1 with the coarse class labels you have (which is common; SAEs often find sub-class features).

### Artefacts to open
1. `universality_summary.json` — top-level numbers.
2. `universal_candidates.csv` — one row per Level-1 candidate with all scores. Sort by `cosine_max_pair` descending and scan the top.
3. `confirmed_latent_frequencies.json` — per-dataset activation frequencies for each confirmed latent. A healthy confirmed latent fires at roughly comparable rates across datasets; ratios above ~5× are suspicious (may be a dataset-specific acquisition artefact that slipped past the audit).
4. `split_pairs.json` — pairs of latents that appear to represent the same concept split across two features (high Jaccard overlap on top-activating images). Fine in small numbers; many split pairs suggests `d_sae` is too large.

---

## 3. Null baseline — `null_baseline/null_baseline_results.json`

The sanity check. Runs the same Level-1+Level-2 pipeline on (a) a **random SAE** and (b) a **PCA** of the activations. The trained SAE must beat both.

### Key fields
```json
{
  "trained_sae": "...universality_summary.json",
  "null_baselines": [
    {"null_model": "random", "n_level1_candidates": ???, "n_level2_confirmed": ???, ...},
    {"null_model": "pca",    "n_level1_candidates": ???, "n_level2_confirmed": ???, ...}
  ]
}
```

### How to read it
Compute these three ratios:

| Ratio | What it means | Healthy value |
|---|---|---|
| `trained.level2 / random.level2` | Specificity vs dense random features | ≥ 2× (ideally ≥ 10×) |
| `trained.level2 / pca.level2` | Specificity vs linear variance axes | ≥ 1.5× |
| `random.level1 / d_sae` | Random SAE passes freq trivially (no sparsity) | Usually huge (> 30%) |

**Key insight:** a random SAE with no top-k constraint will *trivially* pass Level 1 (every latent fires on everything), but will largely fail Level 2 (random directions are not cosine-aligned across datasets). So compare **Level 2** counts, not Level 1.

### Red flags
- Trained L2 ≤ random L2 → **the SAE is no better than random.** Likely causes: training collapsed, wrong layer, wrong activations. Do not publish.
- Trained L2 ≤ PCA L2 → SAE is not outperforming a linear baseline. The features may be real but aren't specifically the value-add of SAEs; re-examine k and expansion factor.
- PCA L2 ≫ random L2 but trained ≈ PCA → activations are highly low-rank; SAE isn't adding resolution.

---

## 4. Inspect latents — `inspect_latents/`

Per-latent spatial heatmaps: for each confirmed universal latent, pick the top-8 activating images per dataset, overlay the 7×7 activation map, save a PNG.

### Files
- `inspect_latents/latent_XXXX.png` — one per confirmed latent. Each row = one dataset, 8 columns of images with heatmap overlays.
- `inspect_latents/index.md` — clickable catalogue.
- `inspect_latents/index_complete` — sentinel file; absence means this step crashed or was skipped.

### What to check per latent
Open `index.md` and go through latents in this order:

1. **Bonferroni-passing latents first.** These are class-correlated, so you have a prior on what to look for (e.g. a pneumonia-correlated latent should highlight consolidated regions).
2. **Sort remaining latents by `cosine_max_pair` descending** (from `universal_candidates.csv`). The strongest geometric agreement usually means the cleanest semantics.

### Per-latent checklist
For each latent, ask:
- **Does the highlight sit on the same anatomy across datasets?** Rib edge in CheXpert *and* rib edge in TB = universal structural feature. Rib in CheXpert and lung field in RSNA = probably a coincidence that survived the gates.
- **Does it highlight anywhere *on* the patient, or on the margins / artefacts?** Margin-hugging (corners, text overlays, markers) = residual scanner signature that the audit didn't catch. Note it and exclude from narrative.
- **Is the pattern lateralised (only left, only right)?** Informative — many real clinical features are lateralised.
- **Does activation magnitude decay smoothly across the top-8?** If image 1 has 10× the activation of image 8, the latent is probably firing on an outlier. Look for monotone decay.

### Common shapes and what they mean
| Heatmap shape | Likely meaning |
|---|---|
| Localised blob on mid-chest | Cardiomediastinal feature. Good candidate for paper figure. |
| Symmetric bilateral activation on lung fields | Generic "lung presence" detector. Real but uninteresting. |
| Ring around image perimeter | Border/padding artefact. Audit failed. |
| Diagonal line | Often a marker or lead wire. Scanner-adjacent — suspect. |
| Diffuse whole-image activation | Either a global intensity feature or a failed feature. Check Layer-1 dashboard for semantic clarity. |

---

## 5. Illness-stratified evaluation — `illness/`

**Script:** `src/scripts/illness_stratified_eval.py`  
**Run after:** universality eval has produced a confirmed latent list and the merged `activations.safetensors` is available.  
**Reference:** `docs/illness-stratified-eval-methodology.md` for full mathematical detail.

This is the most diagnostically useful step for the thesis. It answers whether each confirmed latent is a generic anatomical feature or a pathology-sensitive feature, and if pathology-sensitive, whether it spatially localises to the diseased region.

### Files produced

| File | What you actually read |
|---|---|
| `illness_latent_summary.json` | Top-10 latent indices per illness group by mean activation. First thing to open. |
| `illness_activation_profiles.csv` | Per-latent, per-group, per-dataset: `mean_activation`, `activation_freq`, `n_images`. |
| `illness_latent_matrix.csv` | 7 × K matrix — rows are illness groups, columns are confirmed latents. Heatmap this. |
| `illness_discriminability_probe_coarse.csv` | Per-latent AUROC for 7-group canonical classification (5-fold CV logistic regression). |
| `illness_discriminability_probe_fine.csv` | Per-latent AUROC for fine clinical labels (CheXpert 14-label, RSNA binary, SIIM 4-class). |
| `illness_spatial_grounding.csv` | Per-latent Pearson r between patch activations and bbox overlap, per illness group. |

### Inspection order

**Step A — Rank by pathology-preference ratio ρ_k**

Compute from `illness_activation_profiles.csv`:

```
ρ_k = mean(mean_activation over groups 0,1,2,3,4,6) / mean_activation for group 5 (no_finding)
```

Sort all K confirmed latents by ρ_k. This immediately partitions the set:

| ρ_k | Tier | What to expect in patch ablation |
|---|---|---|
| < 1.0 | Geometry/framing | Fires at image border, apical corners, or lateral wall edge. Visual: symmetric ring or corner blobs. |
| 1.0–2.0 | Anatomy/background | Fires broadly across both healthy and sick images. Visual: diffuse whole-image activation. |
| 2.0–3.5 | Pathology-presence | Fires more on sick images but not localised. Visual: loose cloud over lung fields. |
| ≥ 3.5 | Pathology-localised | Fires on sick images AND at specific anatomical location. Visual: tight reproducible blob on one lung region. |

For the BatchTopK layer-3 xray_v2 run: ρ_k ranges 0.61–5.18 across 65 confirmed latents. The highest-ρ latents (e.g. latent 1088 at ρ=5.18) show tight spatial clusters in patch ablation; lowest-ρ (e.g. latent 981 at ρ=0.61) fire at the lateral image border.

**Step B — Check AUROC for your top-ρ latents**

Open `illness_discriminability_probe_coarse.csv`. Filter to latents with ρ_k ≥ 2.0. AUROC > 0.55 on at least one dataset with a matching pathology group confirms the latent is individually discriminative.

For the current run: mean AUROC is 0.54 across all datasets, best individual ≈ 0.67 on RSNA for airspace_opacity latents. These are modest but above the 0.5 random baseline, consistent with scalar single-feature probes.

For fine-grained labels: open `illness_discriminability_probe_fine.csv` and filter to `label_name = Consolidation` or `Lung Opacity` for airspace latents, `Pleural Effusion` for pleural latents.

**Step C — Spatial grounding for airspace_opacity latents**

Open `illness_spatial_grounding.csv`. Filter to `illness_group = airspace_opacity` and `dataset ∈ {rsna, siim_covid}` (only datasets with bounding boxes). Look for latents where `pearson_r > 0` and `pval < 0.05`. These latents fire at the annotated opacity location, not just broadly on sick images.

For the current run: spatial r values are near zero even for the best latents — consistent with Tier 3 (pathology-presence, not localised) for most of the confirmed set. Latent 1088 is the strongest candidate for Tier 4 based on patch ablation alone.

**Step D — Cross-reference with inspect_latents/**

For each latent you've identified as interesting from the numbers:
1. Open `inspect_latents/latent_XXXXX/` (or the corresponding PNG).
2. Confirm the heatmap cluster site matches the illness group: airspace_opacity → lower/mid lung; pleural → lateral margin; cardiac → mediastinum.
3. Check consistency: does the cluster sit in the same anatomical zone across all 4 dataset rows?

A latent that is Tier 3 by ρ_k but shows a tight consistent cluster in 3/4 dataset rows is worth reporting as a candidate Tier 4 even if the Pearson r is not significant (small n in the bbox-annotated group can suppress p-values).

### What the illness eval adds to the thesis argument

The universality eval establishes that the SAE latents are **cross-dataset and scanner-invariant**. The illness eval establishes that *within* the universal set, a gradient exists from geometry detectors to pathology detectors. This gradient is the main empirical result:

- Geometry/framing detectors (ρ < 1): SAE has learned X-ray composition conventions.
- Pathology-presence detectors (ρ 2–3.5, AUROC > 0.55): SAE has disentangled "sick patient globally" from anatomy.
- Pathology-localised detectors (ρ ≥ 3.5, spatial r > 0): SAE has learned a concept grounded in the physical location of disease.

The thesis claim is: an SAE trained on a multi-task foundation model's intermediate representations spontaneously disentangles these three levels without disease supervision. The illness eval is the evidence.

---

## 6. Phase-3 visualisation — `vis_outputs/`

Produced by `visualise.py all`. These are the *general* SAE visualisations (not specific to the universal set) and complement the pipeline.

### `vis_outputs/stats.pt`
Raw tensor of feature statistics (max activation per image, top-activating image indices, etc.). You usually don't read this directly — it feeds the layer-1/2/3 visualisations. Useful sanity check:

```python
import torch
s = torch.load("stats.pt")
n_dead = (s["global_max"] == 0).sum().item()
print(f"Dead: {n_dead} / {s['global_max'].numel()}")
```

Dead-latent fraction should roughly match `1 - (Level 1 count / d_sae)` from the universality summary. Large discrepancy = bug.

### `vis_outputs/semantic_dicts/` — Layer 1
Patch-crop collages for the top-20 most active features. Each feature gets a page showing the 8×8 grid of patch crops that most activate it. This is the **fastest way to form a hypothesis** about what a feature represents.

**How to read:**
- Consistent visual content across the 64 crops → clear feature (e.g. all rib edges, all diaphragm corners).
- Mixed content → poly-semantic feature. Common at low `d_sae`; not bad, just harder to label.
- Empty grid → feature was among the top-20 active globally but is dead per-patch; indicates a bug or an extreme outlier image.

### `vis_outputs/attribution_graphs/` — Layer 3
Graph + spatial panel for the first 5 images. Shows which SAE features contribute most to the model's output via the decoder paths.

**How to read:**
- Node size = feature importance for that prediction.
- Edge thickness = strength of contribution.
- Spatial panel (right side) = heatmap of each important feature on the image.

This answers "for *this image*, which latents drove the prediction, and where on the image did they fire?" Use it to:
- Sanity check that the universal latents you identified in Step 4 also show up as high-importance for relevant images.
- Find cases where a Bonferroni-passing latent is highlighted for an image of the matching class.

### `vis_outputs/dashboard.html`
Interactive browser for the top-50 features. Contains:
- Patch crops per feature.
- Co-activation heatmap (which features fire together).
- Per-feature activation histograms.

**Use case:** sharing results with collaborators who won't run Python. The co-activation heatmap is also where you catch *duplicated concepts* — pairs of features with co-activation ≈ 1.0 are redundant and suggest `d_sae` could be smaller.

---

## 6. Cross-layer comparison (L2 vs L3)

Once both pipelines + vis jobs are done:

1. **Count overlap:** intersect confirmed-indices from `eval/universality_summary.json` across L2 and L3. Low overlap is expected (different `d_sae` spaces, different layer depths).
2. **Concept stability:** for each Bonferroni-passing L2 latent, find its nearest L3 counterpart by activation correlation. If the same concept emerges at both depths, that's a strong universality signal (feature is preserved up the hierarchy).
3. **Anatomical granularity shift:** L2 should skew local (edges, textures), L3 global (regions, organs). If the reverse is seen, investigate layer ordering in the harvest config.

---

## 7. When to stop and iterate vs publish

### Stop and retrain if:
- `flagged_fraction > 0.5` (scanner audit).
- `trained_L2 ≤ random_L2` (null baseline).
- `n_level2_confirmed == 0` on both layers.
- `inspect_latents` shows only margin/artefact activations.

### Iterate hyperparameters if:
- Dead fraction > 50% → raise `k` or lower `expansion_factor`.
- Everything fires on everything (L1 > 30% of d_sae) → lower `k` or raise frequency threshold.
- Many `split_pairs` → lower `expansion_factor`.

### Ready to write up if:
- `flagged_fraction < 0.2`.
- `trained_L2 / random_L2 ≥ 2`.
- `n_level2_confirmed ≥ 10` with at least 1 Bonferroni pass on at least one layer.
- Spot-checking 10 random confirmed latents in `inspect_latents/` shows 7+ with coherent anatomical activation.
- Illness-stratified eval shows ρ_k > 1.0 for ≥ 50% of confirmed latents (SAE is preferring pathological images for most of its universal features).
- At least 1 confirmed latent achieves ρ_k ≥ 3.5 with a spatially consistent patch cluster (even if Pearson r p-value is not < 0.05).

---

## 8. Pitfalls and gotchas

- **`n_patches=49` vs `batch_size=4096`.** `4096 % 49 ≠ 0`, so any code that reshapes `(batch_size, n_patches, d_sae)` must snap to image boundaries. The fix is in `inspect_universal_latents.get_feature_max`; check it's still there if you see reshape errors.
- **DICOM images (RSNA).** Require `pydicom` in the Docker image. Without it, `load_image_paths` returns empty for RSNA and the spatial panels for RSNA will be blank — the numbers in `universality_summary.json` are still correct (they come from activations, not images), only the visuals are affected.
- **CheXpert nesting.** CheXpert JPGs are 3 levels deep (`train/patient.../study.../view.jpg`); `load_image_paths` must use `rglob` not `iterdir`.
- **Idempotency.** All four pipeline steps have sentinel files; re-submitting after a crash resumes from the failed step. If you want to force a rerun, delete the sentinel (e.g. `rm scanner_audit/scanner_latent_audit.json`).
- **Order of activation index vs image index.** `activations.safetensors` must be in the same order as `load_image_paths` returns. If `image_paths` and `n_images` disagree in `validate_setup`, the visualisation is pointing at the wrong images and all spatial interpretation is wrong.

---

## 9. One-screen report template

When writing up a layer for discussion, produce this table:

| Metric | L2 value | L3 value | Notes |
|---|---|---|---|
| `flagged_fraction` |  |  | scanner audit |
| `n_level1_candidates` |  |  | freq gate |
| `n_level2_confirmed` |  |  | **headline number** |
| `n_spatial_bonferroni_pass` |  |  | class-correlated |
| Trained / random ratio |  |  | null sanity |
| Trained / PCA ratio |  |  | null sanity |
| Dead fraction |  |  | from stats.pt |
| Latents with coherent visuals |  |  | manual, sampled 10 |
| Latents ρ_k ≥ 2.0 (pathology-presence) |  |  | illness eval |
| Latents ρ_k ≥ 3.5 (pathology-localised) |  |  | illness eval |
| Best single-latent AUROC (airspace) |  |  | illness probe |
| Latents with spatial r > 0, p < 0.05 |  |  | illness grounding |

This is the single artefact to share. Everything else is supporting evidence.
