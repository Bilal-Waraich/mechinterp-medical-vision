# Activation Normalisation in the X-Ray SAE Pipeline

There are three distinct normalisation steps, applied at different stages. They are complementary — each removes a different type of unwanted variation.

---

## Step 1 — RMS Normalisation (per token, at harvest time)

**Where:** `src/conglomerate/harvest_all.py` → `normalize_activations()`

**When:** Applied to every token immediately after the UMedPT encoder produces it, before anything is saved to disk.

**Formula (Gao et al. 2024 — Anthropic TopK SAE paper):**

```
x_norm = x / (sqrt(mean(x²)) + ε) * sqrt(d_model)
```

In code:
```python
rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt() + EPS
return x / rms * math.sqrt(d_model)
```

**Applied to:** Each token vector independently (dim=-1), so a batch of shape `[B*H*W, d_model]` produces an output of the same shape with every token rescaled independently.

**What it removes:** Per-token magnitude differences caused by X-ray exposure settings (kVp, mAs) and patient body thickness. Two identical lungs imaged at different hospitals with different exposure settings would produce activation vectors pointing in similar directions but at very different scales. RMS norm sets every token to RMS = 1, then rescales to RMS = `sqrt(d_model)` to match the expected norm the SAE weight initialisation assumes.

**What it does NOT remove:** Systematic directional offsets between datasets (scanner/hospital bias). Two hospitals may have similar per-token magnitudes after RMS norm but their mean activation vectors still point in different directions. That is handled by Step 2.

---

## Step 2 — Per-Dataset Mean Centering (at harvest time, after RMS norm)

**Where:** `src/conglomerate/harvest_all.py` → `center_activations()`

**When:** Applied after all tokens for a dataset have been harvested and RMS-normalised, but before the four datasets are concatenated into the final `.safetensors` file.

**Formula:**

```
x_centred = x - mean(x)        # mean computed over all tokens in this dataset
```

In code:
```python
for acts in all_acts:           # acts: [N_dataset, d_model]
    mean = acts.mean(dim=0)     # [d_model]
    centred.append(acts - mean)
```

**Applied to:** All tokens from one dataset at once, subtracting the single mean vector of that dataset.

**What it removes:** Cross-dataset mean shifts. Different hospitals and scanners produce activation distributions centred at different points in the `d_model`-dimensional space. If not corrected, the SAE's decoder bias (`b_dec`) drifts toward whichever dataset has the largest mean norm, causing systematic reconstruction error on the other datasets. After centering, all four datasets are centred near the origin, so `b_dec` is initialised to zero.

**Can be skipped** with `--no_centering` (not recommended). The metadata field `"centred": true/false` records whether centering was applied.

**The mean vectors are saved** in the `.safetensors` metadata as `dataset_means` (a JSON list of lists) for reproducibility — you can reconstruct the original uncentred activations by adding them back.

---

## Step 3 — Decoder Unit-Norm Constraint (per training step)

**Where:** `src/sae_models.py` → `make_decoder_weights_and_grad_unit_norm()`

**When:** Called every training step, after `loss.backward()` and before `optimizer.step()`.

**Formula:**

```
W_dec_normed = W_dec / ||W_dec||₂        # per row (per feature direction)
grad_projected = (grad · W_dec_normed) * W_dec_normed
W_dec.grad -= grad_projected             # remove radial component from gradient
W_dec = W_dec_normed                     # renormalise weights
```

In code:
```python
W_dec_normed = F.normalize(self.W_dec.data, dim=-1)
W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
self.W_dec.grad -= W_dec_grad_proj
self.W_dec.data = W_dec_normed
```

**Applied to:** Each row of `W_dec` (shape `[d_sae, d_model]`), where each row is one dictionary atom / feature direction.

**What it removes:** Scale ambiguity between the encoder and decoder. Without this constraint, the SAE could learn large-magnitude encoder weights paired with tiny decoder weights (or vice versa), making feature activations uninterpretable — a feature with activation value 100 means something very different from a feature with activation value 0.01. By keeping every decoder direction as a unit vector, feature activation magnitudes are directly comparable across features.

**Gradient projection:** The gradient is projected to be tangent to the unit sphere before the optimiser step. This means the optimiser only moves each decoder vector along the sphere surface (changing its direction) and never radially (changing its norm). Without this projection, the optimiser would immediately push the vectors off unit norm even though the weights are renormalised afterwards.

---

## Summary: Order of Operations

```
Raw encoder output  [B*H*W, d_model]
        │
        ▼
① RMS normalisation (per token)
        │   removes: per-token magnitude differences (exposure, patient thickness)
        ▼
Harvest loop complete — per-dataset tensors accumulated
        │
        ▼
② Per-dataset centering (per dataset mean subtracted)
        │   removes: cross-dataset directional offsets (scanner/hospital bias)
        ▼
Concatenated, saved to .safetensors
        │
        ▼
SAE training — every step:
        │
        ▼
③ Decoder unit-norm constraint (per row of W_dec)
            removes: encoder/decoder scale ambiguity
                     makes feature activation magnitudes comparable
```

---

## What Is NOT Normalised

- **Spatial position** — no positional normalisation is applied. A patch at position (3, 3) is treated identically to one at (0, 0). The SAE learns features that are position-agnostic by design.
- **Per-feature activation scale** — feature activation values after the encoder are not normalised. The unit-norm decoder constraint ensures the *decoder* directions are comparable, but a feature that fires with activation 5.0 is genuinely more strongly activated than one at 0.5.
- **Image-level statistics** — no per-image normalisation (e.g. histogram equalisation) is applied before the encoder. The encoder itself was pretrained on raw `[0, 1]` pixel values and is expected to handle typical X-ray intensity variation internally.
