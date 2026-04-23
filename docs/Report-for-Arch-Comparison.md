# Sparse Autoencoder Pipeline: BatchTopK

Branch: `batch-topk`  
Docker image: `sae-batchtopk-2.4:latest`  
Target layers: `pm_index=2` (Swin Stage 2→3) and `pm_index=3` (final Swin stage)

---

## 1. Architecture & Forward Pass

**Source:** `src/sae_models.py` — class `BatchTopKSAE`

### Parameter layout

| Tensor | Shape | Notes |
|---|---|---|
| `W_enc` | `[d_model, d_sae]` | Encoder weight matrix |
| `b_enc` | `[d_sae]` | Encoder bias |
| `b_dec` | `[d_model]` | Pre/post centring bias |
| `W_dec` | `[d_sae, d_model]` | Decoder (atom matrix); each **row** = one dictionary atom |
| `num_batches_not_active` | `[d_sae]` | Non-trainable buffer; counts consecutive inactive batches per feature |

For `pm_index ∈ {2,3}`: `d_model=1024`, `expansion_factor=4` → `d_sae=4096`.

### Sparsity mechanism — BatchTopK

Sparsity is enforced by **global hard top-k selection across the entire flattened batch**, not per-sample.

**Encode path** (`encode()` — inference/evaluation only):
```
x_cent  = x − b_dec                               # [B, d_model]  centre
pre_acts = ReLU( x_cent @ W_enc + b_enc )          # [B, d_sae]   pre-sparse activations
topk     = top-k(pre_acts, k=k, dim=-1)            # per-sample, k active features
codes    = scatter(zeros, topk.indices, topk.values) # [B, d_sae]  sparse
```
`encode()` uses **per-sample top-k** so it works for any batch size including B=1.

**Forward path** (`forward()` — training only):
```
x_cent   = x − b_dec                              # [B, d_model]
acts     = ReLU( x_cent @ W_enc + b_enc )          # [B, d_sae]   dense pre-sparse
flat     = acts.flatten()                          # [B·d_sae]
n_keep   = min(k·B, B·d_sae)
topk     = top-k(flat, n_keep)                     # GLOBAL selection across batch
acts_topk = scatter(zeros_like(flat), topk.indices, topk.values)
           .reshape(B, d_sae)                      # [B, d_sae]  sparse
x_hat    = acts_topk @ W_dec + b_dec               # [B, d_model] reconstruction
```

The **critical distinction** from per-sample top-k: global top-k allows some samples to have more than `k` active features and some fewer, while guaranteeing exactly `k·B` total active features across the batch. This relaxes the strict per-token constraint during training, producing softer gradients, but keeps the batch-level sparsity budget fixed.

### Bottleneck tensor shapes (pm_index=2/3)

| Stage | Tensor | Shape |
|---|---|---|
| Input | `x` | `[B, 1024]` |
| After centring | `x_cent` | `[B, 1024]` |
| Pre-sparse activations | `acts` | `[B, 4096]` |
| **Bottleneck (sparse codes)** | `acts_topk` | `[B, 4096]` — at most `k·B` non-zero entries |
| Reconstruction | `x_hat` | `[B, 1024]` |

At inference with `n_patches=49` per image and `batch_size=4096`, an image-aligned batch covers exactly `4096 // 49 * 49 = 4067` tokens.

### Initialisation
1. `W_enc` initialised with `kaiming_uniform_`.
2. `W_dec = (W_enc.T).clone()`, then each row normalised to unit norm, then scaled by `0.1` (Anthropic heuristic: SAE grows into reconstruction rather than starting large).
3. `W_enc` is then re-assigned to `W_dec.T` for symmetric initialisation.
4. `b_enc = 0`, `b_dec = 0` (or global data mean if activations are not centred — checked at training start).

---

## 2. Training Loop & Loss Computation

**Source:** `src/train_sae.py` — function `train()`

### Loss function

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{L1}} + \mathcal{L}_{\text{aux}}$$

**Reconstruction loss** (Mean Squared Error):
$$\mathcal{L}_{\text{recon}} = \frac{1}{B \cdot d_{\text{model}}} \sum_{i,j} (\hat{x}_{ij} - x_{ij})^2$$
Computed in `float32` regardless of AMP setting to avoid NaN accumulation.

**L1 sparsity penalty:**
$$\mathcal{L}_{\text{L1}} = \lambda_1 \cdot \frac{1}{B} \sum_i \| \mathbf{z}_i \|_1$$
where $\lambda_1 = $ `l1_coeff = 2e-4` (default). Despite the hard top-k, L1 is retained to penalise large activation magnitudes on the selected features, discouraging a few latents from dominating.

**Auxiliary dead-feature revival loss:**

Latent $f$ is declared **dead** when `num_batches_not_active[f] >= n_batches_to_dead` (default 25 — raised from 10 because rare diseases may not appear in 10 consecutive batches in mixed-dataset training).

When dead features exist:
```
residual    = (x − x_hat).detach()                    # [B, d_model]; no main-branch grad
k_aux       = d_model // 2  = 512                     # paper heuristic
scale       = min(n_dead / k_aux, 1.0)                # reduces aux loss when few dead
acts_dead   = acts[:, dead_mask]                      # [B, n_dead]
acts_aux    = top-k(acts_dead, k_aux, per-sample)     # [B, n_dead] sparse
x_aux       = acts_aux @ W_dec[dead_mask]             # [B, d_model]
L_aux       = aux_penalty * scale * MSE(x_aux, residual)
```
`aux_penalty = 1/32` (default). The `.detach()` on `residual` means gradients flow **only** through dead features, not back into the main reconstruction path.

### Decoder unit-norm constraint

Enforced via `make_decoder_weights_and_grad_unit_norm()`, called in this exact order:

```
loss.backward()                                       # 1. compute gradients
clip_grad_norm_(model.parameters(), max_grad_norm=1.0) # 2. clip
make_decoder_weights_and_grad_unit_norm()             # 3. project + renorm
optimizer.step()                                      # 4. update
optimizer.zero_grad()                                 # 5. clear
scheduler.step()                                      # 6. advance LR
```

The projection step (`make_decoder_weights_and_grad_unit_norm`):
```
W_normed    = normalize(W_dec, dim=-1)                # unit-normalised rows
grad_proj   = (W_dec.grad · W_normed).sum(-1, keepdim=True) · W_normed
                                                      # radial component of grad
W_dec.grad -= grad_proj                               # project to tangent plane
W_dec.data  = W_normed                                # renormalise weight
```
This keeps each dictionary atom exactly on the unit sphere. Without this, the decoder would shrink atoms to approach trivially small reconstruction errors.

### LR schedule
- Linear warmup: step 0 → `lr_warmup_steps=1000`, LR scales from 0 to `lr=2e-4`.
- Cosine decay from step 1000 → 50000, LR decays from `2e-4` to `0.1 × 2e-4 = 2e-5` (floor at 10% of peak, not zero).

### Stratified sampling
If `dataset_ids` is present in the activations file, `StratifiedBatchSampler` enforces equal representation: `batch_size // n_datasets` tokens per dataset per batch. TB (dataset_id=3) is sampled **with replacement** (oversampled) because it is ~15× smaller than CheXpert after spatial flattening. Non-TB datasets cycle without replacement within each pseudo-epoch and reshuffle when exhausted.

### AMP / FP16
Optional (`use_amp=False` by default). When enabled, `torch.cuda.amp.autocast` wraps the forward pass; `GradScaler` handles the loss scaling. Unscaling is applied before `clip_grad_norm_` and `make_decoder_weights_and_grad_unit_norm` so those ops always work in FP32.

---

## 3. Configuration Parameters

**Source:** `src/config.py` — dataclass `SAEConfig`

### Layer-specific dimensions

| `pm_index` | Swin stage | `d_model` | `n_patches` (spatial) |
|---|---|---|---|
| 0 | Stage 0→1 | 256 | 784 (28×28) |
| 1 | Stage 1→2 | 512 | 196 (14×14) |
| **2** | **Stage 2→3** | **1024** | **49 (7×7)** ← universality experiments |
| **3** | **Final stage** | **1024** | **49 (7×7)** ← universality experiments |

`d_sae = d_model × expansion_factor = 1024 × 4 = 4096` for both pm_index 2 and 3.

### Default hyperparameters (as-shipped on this branch)

| Parameter | Default | Notes |
|---|---|---|
| `k` | 32 | Active features per token (per-sample at eval; globally k·B at training) |
| `expansion_factor` | 4 | d_sae = d_model × 4 |
| `l1_coeff` | 2e-4 | L1 sparsity penalty coefficient |
| `aux_penalty` | 1/32 ≈ 0.03125 | Auxiliary dead-feature loss weight |
| `n_batches_to_dead` | **25** | Consecutive inactive batches before dead; raised from reference value of 10 for mixed-dataset training |
| `top_k_aux` | 512 | Dead features used in aux reconstruction (also = d_model // 2 at runtime) |
| `lr` | 2e-4 | Adam learning rate (peak) |
| `lr_warmup_steps` | 1000 | Linear warmup from 0 to `lr` |
| `lr_schedule` | `"cosine"` | Cosine decay from step 1000 to 50000, floor=10% |
| `beta1`, `beta2` | 0.9, 0.999 | Adam moments |
| `max_grad_norm` | 1.0 | Gradient clipping |
| `batch_size` | 2048 | Tokens per step (stratified: 512/dataset) |
| `steps` | 50,000 | Total training steps |
| `checkpoint_every` | 5,000 | Checkpoint frequency |
| `stratified` | True | Equal dataset representation per batch |
| `use_amp` | False | FP32 by default |

---

## 4. Universality Evaluation Adaptations

**Sources:** `src/scripts/universality_eval.py`, `src/scripts/scanner_latent_audit.py`

### 4.1 Input format — merged vs separate activations

BatchTopK produces a **single merged** `activations.safetensors` containing:
- `"activations"` tensor: `[N_total, d_model]` where `N_total = sum(N_i)` across all datasets
- `"dataset_ids"` tensor: `[N_total]` integer labels (0=chexpert, 1=rsna, 2=siim_covid, 3=tb) at **token** granularity (one label per spatial patch, not per image)

The `universality_eval.py` script splits by `dataset_ids` to recover per-dataset slices. The `scanner_latent_audit.py` script requires this format directly.

**Important note in the audit docstring (lines 45–66):** Matryoshka jobs produce *four separate per-dataset files without `dataset_ids`*. To run the scanner audit on a Matryoshka checkpoint, you must concatenate manually first. BatchTopK does **not** require this — its merged format is the native input.

### 4.2 Scanner audit (Step 1) — no architecture-specific changes

The audit is architecture-agnostic: it operates on `encode(x)` outputs (`[N, d_sae]` sparse codes) regardless of how sparsity was produced.

For each latent $f$:
1. Extract per-token activation magnitude: `codes[:, f]` → `[N_sample, 1]`
2. Fit `LogisticRegression(max_iter=200)` (sklearn lbfgs, multinomial softmax for 4 classes) from the single-feature to `dataset_id`
3. Compute one-vs-rest AUROC per dataset → macro-average AUROC

**Subsampling:** balanced draw of `sample_size // n_datasets` tokens per dataset (default `sample_size=100,000`, so 25,000 tokens per dataset). This corrects for CheXpert being ~15× larger than TB in raw token count.

Dead latents (max activation < 1e-8 in the sample) are skipped and recorded as AUROC = 0 (not flagged).

**Flag threshold:** macro-AUROC ≥ 0.80. Flagged latents are excluded from all downstream universality analysis.

### 4.3 Level 1 — Frequency gate

For each unflagged latent $f$ and dataset $d$:
$$\text{freq}(f, d) = \frac{1}{N_d} \sum_{i=1}^{N_d} \mathbf{1}[\text{code}_{id} > 0]$$

**Gate:** latent $f$ passes Level 1 if $\text{freq}(f, d) \geq 0.01$ for at least `min_datasets=2` datasets.

**BatchTopK-specific consideration:** because sparsity is enforced by global batch-level top-k rather than per-token hard gating, a given token may have zero activations for some features at training time (features that didn't make the global top-k) but non-zero at `encode()` time (per-sample top-k). The evaluation uses `encode()`, so Level 1 measures **per-sample top-k sparsity**, not batch-level sparsity. This means frequency measurements are slightly higher than the effective sparsity seen during training — this is expected and intentional.

### 4.4 Level 2 — d_model centroid cosine similarity

For each Level-1 candidate $f$:
1. For each dataset $d$, compute the **d_model centroid**: the mean input activation vector over all tokens where $f$ is active
   $$\mathbf{c}_{f,d} = \frac{1}{|\mathcal{A}_{f,d}|} \sum_{i \in \mathcal{A}_{f,d}} x_i \quad \in \mathbb{R}^{d_{\text{model}}}$$
   where $\mathcal{A}_{f,d} = \{i : \text{code}_{f,d,i} > 0\}$
2. For all pairs $(d_1, d_2)$ from active datasets, compute cosine similarity between centroids
3. **Gate:** latent passes Level 2 if $\min_{d_1 \neq d_2} \cos(\mathbf{c}_{f,d_1}, \mathbf{c}_{f,d_2}) \geq 0.70$ across at least `min_datasets=2`

**BatchTopK-specific consideration on the strict-k cutoff and spatial grounding:** The k=32 hard cutoff means activations are truly binary-valued in structure (a feature either fires on a token at a specific magnitude or is exactly zero). For spatial bounding-box grounding (Pearson $r$ between per-patch activation and fractional bbox overlap), this is **beneficial** — zero is a genuine absence signal, not a small residual. However, it also means that for images where a feature's pre-activation would rank 33rd–50th, the contribution is silently dropped to zero even if it would have been meaningful. The spatial grounding test uses `encode()` activations (per-sample top-k) which slightly relaxes this concern. The Bonferroni correction is applied across all confirmed Level-2 latents (not all tested), reducing the multiple-testing burden to a realistic count.

### 4.5 Cross-dataset probe, split detection, and other steps

These steps are architecture-agnostic:
- **Cross-dataset probe** (Step 5): leave-one-dataset-out logistic regression AUROC on per-image feature_max values
- **Spatial grounding** (Step 6): Pearson $r$ between per-patch activation vector and fractional bbox overlap; Bonferroni correction at $\alpha=0.05$ across confirmed latents
- **Split detection** (Step 7): Jaccard overlap of top-`split_top_k=50` activating images between pairs of confirmed latents; threshold ≥ 0.50 flags as a split pair

None of these change in behaviour between BatchTopK and an alternative architecture — they consume `encode()` outputs.

---

## 5. File I/O & Output Artifacts

### 5.1 Model checkpoint — `step_XXXXXXX/`

Two files per checkpoint:

**`sae_weights.safetensors`** — contains the full `model.state_dict()`:
```
W_enc                     float32  [d_model, d_sae]   = [1024, 4096]
b_enc                     float32  [d_sae]             = [4096]
b_dec                     float32  [d_model]            = [1024]
W_dec                     float32  [d_sae, d_model]    = [4096, 1024]
num_batches_not_active    float32  [d_sae]              = [4096]  ← buffer, not a parameter
```
Total: `1024×4096 + 4096 + 1024 + 4096×1024 + 4096` ≈ **8.4M parameters** + 4k-element buffer.

**`cfg.json`** — SAELens-compatible config:
```json
{
  "architecture": "batch_topk",
  "d_model": 1024,
  "d_sae": 4096,
  "k": 32,
  "expansion_factor": 4,
  "n_batches_to_dead": 25,
  "top_k_aux": 512,
  "aux_penalty": 0.03125,
  "l1_coeff": 0.0002,
  "lr": 0.0002,
  "beta1": 0.9,
  "beta2": 0.999,
  "max_grad_norm": 1.0,
  "batch_size": 2048,
  "steps": 50000,
  "activations_path": "...",
  "lr_warmup_steps": 1000,
  "lr_schedule": "cosine",
  "step": <checkpoint_step>
}
```
Note: `"architecture": "batch_topk"` is the discriminator key. No Matryoshka-specific fields exist on this branch.

### 5.2 Activation file — `activations.safetensors`

Produced by `harvest_all.py`, consumed by all evaluation scripts.

```
"activations"    float32   [N_total, d_model]   flattened tokens, all datasets
"dataset_ids"    int32     [N_total]             per-token dataset index
```

**Safetensors metadata** (JSON in file header):
```json
{
  "image_stems": "{\"chexpert\": [...], \"rsna\": [...], \"siim_covid\": [...], \"tb\": [...]}"
}
```
`image_stems` maps dataset name → ordered list of filename stems, used to align activations to label CSVs and bbox CSVs during `universality_eval.py`. This is essential for spatial grounding.

### 5.3 Universality eval outputs — `universality/eval/`

| File | Contents |
|---|---|
| `universality_summary.json` | Top-level counts (see schema below) |
| `universal_candidates.csv` | One row per Level-1 candidate: `latent_idx, freq_chexpert, freq_rsna, ..., cosine_max_pair, confirmed` |
| `confirmed_latent_frequencies.json` | Per-confirmed-latent per-dataset activation frequencies |
| `split_pairs.json` | Pairs of latents with Jaccard ≥ 0.50 on top-50 activating images |

**`universality_summary.json` schema:**
```json
{
  "d_sae": 4096,
  "n_scanner_flagged": <int>,
  "n_level1_candidates": <int>,
  "n_level2_confirmed": <int>,
  "n_spatial_bonferroni_pass": <int>,
  "n_spatial_bonferroni_tested": <int>,
  "n_split_pairs": <int>
}
```
There are no BatchTopK-specific extra fields. A Matryoshka architecture would not include `n_scanner_flagged` in the same place because the scanner audit requires a pre-processing step to merge files first; the summary schema would be identical in shape.

### 5.4 Null baseline outputs — `universality/null_baseline/`

`null_baseline_results.json`:
```json
{
  "trained_sae": "<path to universality_summary.json>",
  "null_baselines": [
    {
      "null_model": "random",
      "d_sae": 4096,
      "n_level1_candidates": <int>,
      "n_level2_confirmed": <int>,
      "level1_rate": <float>,
      "level2_rate": <float>,
      "confirmed_indices": [...]
    },
    {
      "null_model": "pca",
      ...
    }
  ]
}
```

### 5.5 Inspect latents outputs — `universality/inspect_latents/`

| File | Contents |
|---|---|
| `latent_XXXX.png` | Per-confirmed-latent: `(n_datasets × top_k)` grid of heatmap-overlaid images + patch ablation panel; `top_k=8` by default |
| `index.md` | Catalogue linking all latent PNGs |
| `index_complete` | Sentinel file written after all PNGs are produced; absence = incomplete run |

### 5.6 Visualisation outputs — `vis_outputs/`

Produced by `visualise.py all`:

| File | Contents |
|---|---|
| `stats.pt` | `torch.save` dict: `global_max [d_sae]`, `top_image_indices [d_sae, top_k]`, `n_images` int |
| `semantic_dicts/` | Patch-crop collages for top-20 most active features |
| `attribution_graphs/` | Attribution graph + spatial companion panel PNGs (first 5 images) |
| `dashboard.html` | Interactive HTML for top-50 features including co-activation matrix |
