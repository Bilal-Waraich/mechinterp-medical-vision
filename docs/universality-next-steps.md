# Universality Analysis — Next Steps

**Date:** 2026-04-09
**Based on:** `universality-analysis-L2.md`

---

## What Is Proven So Far

- Decoder directions of matched features are geometrically close across SAEs
- Modality governs the degree of sharing (same-modality >> cross-modality)
- 2 complete 4-dataset chains exist, with Chain 1 having high activations everywhere

## What Is Not Yet Proven

Cosine similarity shows the decoder *vectors* point in similar directions. It does **not** show the features encode the *same concept*. A positional detector and a texture detector could both have similar decoder directions by coincidence.

---

## Next Steps, in Priority Order

### 1. Max-Activating Patch Visualisation (highest priority)

For each feature in Chain 1 (`Colon F155 → Breast F2662 → Brain F1590 → Pneumonia F688`), retrieve the top-k patches that maximally activate it across each dataset's validation set. Then visually inspect whether the patches look like the same concept across datasets.

- Converts a geometric claim into a semantic one
- If patches look the same (e.g., all four show nuclei boundaries, or all show low-frequency gradients) — strong evidence of universality
- If they look different — the cosine similarity is spurious

**Infrastructure needed:** index into stored activations from `harvest_activations.py` and pull top patches per feature.

---

### 2. Permutation Null Baseline (statistical rigour)

Compute pairwise cosine similarities between two SAEs trained on the *same* dataset (or between random weight matrices of the same shape) to establish a null distribution. Currently there is no baseline — it is not possible to say whether 0.696 cross-modality is meaningful without knowing what "random" looks like for 4096-dim dictionaries.

**Key question:** does the cross-modality cosine of 0.696 sit above the 95th percentile of the null? If not, the result is not statistically meaningful.

---

### 3. Layer Sweep — L1 and L3

Run the identical pipeline on layers 1 and 3. Prior work on vision transformers suggests universality should increase with depth.

| Expected outcome | Interpretation |
|---|---|
| Universality increases with depth | Consistent with known ViT behaviour — deeper layers are more semantic |
| Peaks at L2 then drops | L2 is a bottleneck layer (PatchMerging) — may be a special case |
| Flat across layers | Universality is not depth-dependent in this model |

---

### 4. Cross-SAE Activation Patching (causal test)

For a matched feature pair (e.g., Breast F2662 ↔ Brain F1590):

1. Take a brain image, encode it through the brain SAE
2. Replace Brain F1590's activation with the breast SAE's encoding of a colon image's F155 activation
3. Decode and measure whether the downstream model output shifts in the same direction as it would for the colon image

This is the strongest available evidence — **functional equivalence under causal intervention** — and goes beyond geometric similarity entirely.

---

## Priority Summary

| Step | What it proves | Effort |
|---|---|---|
| Max-activating patches | Features encode the same *concept* | Low — infrastructure exists |
| Permutation null | Result is statistically above chance | Medium — need null SAE pairs |
| Layer sweep | Universality varies with depth as expected | Medium — re-run existing pipeline |
| Activation patching | Features are *functionally* equivalent | High — new intervention code |

**Recommended starting point:** max-activating patch extraction and visualisation for Chain 1 features, using the existing `harvest_activations.py` infrastructure.
