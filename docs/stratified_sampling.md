# Stratified Sampling in the X-Ray SAE Pipeline

## The Problem

The four datasets are massively imbalanced:

| Dataset | Images | Tokens (@ 49 patches/image) |
|---|---|---|
| CheXpert | ~10,000 (capped) | ~490,000 |
| RSNA Pneumonia | ~10,000 (capped) | ~490,000 |
| SIIM-COVID19 | ~6,000 | ~294,000 |
| TB Shenzhen | ~700 (all images) | ~34,300 |

Without intervention, TB Shenzhen represents only **~2.7%** of the combined token pool. A plain shuffled DataLoader would train on TB so rarely that the SAE would learn almost nothing TB-specific — TB neurons would either never fire or collapse into the generic "airspace opacity" direction learned from the much larger CheXpert/RSNA/SIIM data.

---

## How It Works: Two Stages

Stratification is enforced at two points in the pipeline: during **harvesting** (encoding the images) and during **training** (constructing each batch).

---

### Stage 1 — Harvesting (`harvest_all.py`)

When images are passed through the UMedPT encoder to extract activations, each activation vector is tagged with a `dataset_id` integer:

```
0 = chexpert
1 = rsna
2 = siim_covid
3 = tb
```

These IDs are stored alongside the activations in the `.safetensors` output file as a `[N_total]` int32 tensor called `dataset_ids`. The training script reads this tensor to know which dataset each token came from.

**Per-dataset centering** is also applied at this stage: each dataset's mean activation vector is subtracted from its own tokens before concatenation. This corrects for scanner/hospital-level mean shifts that would otherwise bias the global decoder bias (`b_dec`) toward whichever dataset has the largest mean activation norm. Without this, the SAE would waste capacity fitting systematic offset differences between hospitals rather than pathology-specific directions.

---

### Stage 2 — Training (`train_sae.py`)

Two custom PyTorch classes replace the default `DataLoader` logic:

#### `StratifiedActivationDataset`
A thin wrapper that keeps the activations tensor and `dataset_ids` tensor together so the sampler can access both.

#### `StratifiedBatchSampler`
The core logic. For each training step it constructs a batch index list that pulls **exactly equal numbers of tokens from every dataset**:

```
tokens_per_dataset = batch_size // n_datasets
```

So for `batch_size=2048` (current default in `src/config.py`) and 4 datasets, each batch contains **512 tokens from each of CheXpert, RSNA, SIIM-COVID19, and TB** — regardless of how many total tokens each dataset has.

**TB gets special treatment:** because TB has far fewer unique images than the others, it is sampled **with replacement** (i.e. the same TB activations can appear in multiple batches). All other datasets are sampled **without replacement per epoch** — their token pools are shuffled at the start of each epoch and drawn sequentially, refilling when exhausted.

```
if uid == tb_dataset_id:
    batch.extend(rng.choices(idxs, k=per_ds))   # with replacement
else:
    batch.extend(pool[ptr : ptr + per_ds])        # without replacement
```

After picking indices from each dataset, the full batch is shuffled before being returned, so the SAE never sees 1,024 consecutive TB tokens — the dataset origin is mixed within each batch.

---

## Why This Matters for the SAE

The SAE is a linear encoder + decoder. Its features (latent dimensions) are learned directions in the activation space. If TB barely appears in training:

1. **TB-specific directions never get reinforced** — the reconstruction loss gradient for TB is essentially zero most of the time.
2. **TB tokens get reconstructed by CheXpert/RSNA features** — the SAE repurposes nearby airspace-opacity features to also handle TB, meaning features are not cleanly interpretable.
3. **Feature monosemanticity is lost for TB** — a feature that fires on "TB cavity + COVID opacity + RSNA consolidation" is not useful for mechanistic interpretation.

With stratified sampling, TB tokens contribute **25% of every gradient update**, the same as each other dataset. This forces the SAE to maintain dedicated features for TB-specific patterns even though TB is ~14× smaller in raw data volume.

---

## Verification

Every 500 training steps, the trainer logs the actual per-dataset fraction observed in the batch to both the console and Weights & Biases:

```
[Stratification] step 500  batch fracs: chexpert=0.25  rsna=0.25  siim=0.25  tb=0.25
```

W&B metric keys: `stratification/chexpert`, `stratification/rsna`, `stratification/siim`, `stratification/tb`

These should all be ~0.25 throughout training. Significant deviation indicates a bug in the sampler or that the activations file is missing the `dataset_ids` tensor (in which case the trainer falls back to a plain shuffled DataLoader and logs a warning).

---

## Disabling Stratification

Pass `--no_stratified` to `train_sae.py` to fall back to a plain shuffled DataLoader. This is useful for ablation experiments to measure how much stratification contributes to TB feature quality.

```bash
python src/train_sae.py --no_stratified
```

---

## Summary

| Step | What happens | Why |
|---|---|---|
| Harvest | Each token tagged with `dataset_id` (0–3) | Enables sampler to track origin |
| Harvest | Per-dataset mean subtracted (centering) | Removes scanner/hospital offset from activations |
| Training | `StratifiedBatchSampler` enforces equal per-dataset quota per batch | Prevents large datasets from dominating gradient updates |
| Training (TB) | TB sampled with replacement | TB has too few unique images to fill a quota without repetition |
| Training | Batch indices shuffled before yielding | Prevents dataset-ordered gradient signal within a batch |
| Training | Fraction logged every 500 steps | Verifies stratification is working |
