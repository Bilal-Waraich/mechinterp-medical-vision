# Matryoshka SAE — Universality Analysis

How the codebase identifies *universal* latents in a **Matryoshka SAE** — i.e. features that fire for the same semantic concept across multiple chest X-ray datasets, at the same image regions, and remain stable across the Matryoshka nesting scales.

The analysis is run by two Nomad jobs in `nomad/xray/analysis/`:

1. **`matryoshka-universality.nomad`** — the single-script, Matryoshka-native 5-stage pipeline that exploits the model's `k_schedule` nesting.
2. **`pipeline-matryoshka-l{2,3}.nomad`** — the full per-layer analysis pipelines (pre-merge → scanner audit → universality eval → null baseline → inspect latents), reusing the generic universality evaluator but pinned to a fixed Matryoshka shell.

Both are backed by `src/scripts/matryoshka_universality_eval.py` (the 5-stage script) and `src/scripts/universality_eval.py` (the shell-based evaluator). They rely on the nested representation produced by [`src/matryoshka_sae.py`](../src/matryoshka_sae.py): because `k_schedule` is ascending and the encoder is shared, codes at `k_i` are a subset of codes at `k_j > k_i`, so the first features learned are the most globally important.

---

## What makes the Matryoshka case different

A standard BatchTopK SAE produces a single sparse code per token. A Matryoshka SAE produces **one sparse code per scale** `k_i ∈ k_schedule` (e.g. `[4, 8, 16, 32]`), all sharing the same encoder. This gives the universality analysis two extra signals that a standard SAE cannot produce:

- **Ancestor tracing.** If two "fine-scale" features look like a split of a single "coarse-scale" feature, the coarse feature can be recovered at a smaller `k` and used instead — resolving feature splitting without manual merging.
- **Prefix stability.** A genuinely universal concept should be selected at every scale. We can measure how much the top-activating images of a latent change as we walk down `k_schedule`.

Both are implemented in `matryoshka_universality_eval.py` and are the reason the Matryoshka analysis adds two stages (3 and 4) on top of the standard Level-1 / Level-2 filters.

---

## The five-stage pipeline (`matryoshka_universality_eval.py`)

Invoked by `nomad/xray/analysis/matryoshka-universality.nomad`. Each stage gates the next, so the expensive checks (cosine centroids, pairwise Jaccards) only run on latents that survive the cheaper ones.

Default thresholds (matching the thesis methodology) live at the top of the script:

```
T_TOP=50, FREQ_THR=0.01, MIN_DS=2, COS_THR=0.70,
SPLIT_THR=0.50, ANC_THR=0.40, STABILITY_THR=0.40, AUROC_THR=0.80
```

### Stage 0 — Scanner-Feature Audit

[`scanner_audit()`](../src/scripts/matryoshka_universality_eval.py:179)

For each latent, compute the max one-vs-rest AUROC for predicting *dataset identity* from that latent's activation alone (Mann–Whitney equivalent — no logistic fit needed). Any latent with AUROC > `auroc_thr` (default 0.80) is flagged as encoding scanner/site information and removed from all later stages.

Uses the **largest-scale code** `z(k_max)` only — scanner artefacts live in the most expressive representation. Classes are downsampled to `SCANNER_SAMPLES = 20 000` per dataset to keep AUROC computation tractable over a 4 096-wide dictionary.

### Stage 1 — Activation Frequency

[`activation_frequency()`](../src/scripts/matryoshka_universality_eval.py:226)

Per-dataset non-zero activation frequency at `k_max`. A latent becomes a Level-1 candidate iff its frequency exceeds `freq_thr` (default 1%) on at least `min_ds` (default 2) datasets. The 1% target matches the interpretable medium-density band used in Gao et al. (2024): below ≈0.01% is dead/noise, above ≈10% is generic anatomy.

### Stage 2 — Cross-Dataset Semantic Consistency

[`semantic_consistency()`](../src/scripts/matryoshka_universality_eval.py:240)

For every Level-1 candidate `f` and every dataset `d`:

1. Pick the top-`T_TOP = 50` images in `d` by the per-image max activation of `f` (patch-to-image reduction in [`codes_to_per_image()`](../src/scripts/matryoshka_universality_eval.py:132)).
2. Compute the **centroid** `c_f^(d)` as the mean raw encoder-input activation vector (d_model-dim) over those 50 images — *not* the SAE code, so the embedding is dataset-independent and directly comparable.
3. Measure the minimum pairwise cosine of these centroids across datasets.

Latents with `min_cos ≥ cos_thr` (default 0.70) are declared **Level-2 confirmed** — their top-activating images across datasets share a consistent direction in representation space, so the latent is firing on the same *concept* everywhere, not coincidentally on different ones.

### Stage 3 — Split Feature Audit with Ancestor Trace (Matryoshka-specific)

[`split_audit()`](../src/scripts/matryoshka_universality_eval.py:306)

Matryoshka SAEs are known to *split* a coarse concept into several siblings at large `k`. Stage 3 detects and repairs this:

1. At `k = k_max`, for every pair `(f_a, f_b)` of Level-2 confirmed latents, compute the Jaccard overlap of their top-image sets (unioned across datasets). If `J ≥ split_thr` (default 0.50), the pair is a split candidate.
2. **Ancestor trace** — walk down `k_schedule` (smaller `k` first) and look for a single coarse latent `g` whose top-image set overlaps *both* `f_a` and `f_b` by `J ≥ anc_thr` (default 0.40). The smallest `k'` where such a `g` exists is recorded.
3. If an ancestor is found, `f_a` and `f_b` are dropped and `g` is retained; if not, both children stay.

This is only possible *because* the model is Matryoshka: the per-scale top-image sets are computable from the same encoder without retraining.

### Stage 4 — Prefix Stability (Matryoshka-specific)

[`prefix_stability()`](../src/scripts/matryoshka_universality_eval.py:364)

For every surviving latent `f`, compute the Jaccard between its top-image sets at every adjacent `(k_i, k_{i+1})` pair. A latent is **prefix-stable** iff *every* adjacent Jaccard ≥ `stability_thr` (default 0.40).

Intuition: a truly universal concept should be selected by the encoder at every scale — a feature that only emerges at `k_max` is brittle. Prefix stability filters out features that exist only because the budget was large enough to afford them.

### Outputs

Written under `universality_matryoshka/`:

| File | Contents |
|---|---|
| `stage0_scanner_audit.csv` | per-latent max AUROC + flag |
| `stage1_frequency.csv` | per-latent per-dataset activation frequency |
| `stage2_semantic.csv` | per-latent minimum pairwise cosine |
| `stage3_splits.csv` | detected split pairs `(f_a, f_b, k_ancestor, g_ancestor)` |
| `stage4_prefix_stability.csv` | per-latent per-scale Jaccards + min |
| `universality_summary.json` | counts at every stage + thresholds used |
| `confirmed_universal_latents.json` | final indices passing all 5 stages |

---

## Nomad job 1 — `matryoshka-universality.nomad`

[`nomad/xray/analysis/matryoshka-universality.nomad`](../nomad/xray/analysis/matryoshka-universality.nomad)

A one-task job that runs the full 5-stage script in-process.

- **Checkpoint:** `…/matryoshka/layer-2/xray-conglomerate/xray_v1/checkpoints/step_0200000`
- **Per-dataset activations:** 4 `.safetensors` files (chexpert, rsna, siim_covid, tb), passed with `--activations PATH:LABEL` repeat flags. Parsed by [`parse_tagged()`](../src/scripts/matryoshka_universality_eval.py:84) into `(label, path)` tuples.
- **Output:** `$BASE/universality_matryoshka/`
- **Idempotent:** skipped if `universality_summary.json` already exists.
- **Resources:** 1 GPU, 4 CPUs, 64 GB RAM. A `poststop` sidecar emails `stdout/stderr` to the owners.

All thresholds are set explicitly in the `args` block, so the Nomad job is self-documenting: `t_top=50`, `freq_thr=0.01`, `min_ds=2`, `cos_thr=0.70`, `split_thr=0.50`, `anc_thr=0.40`, `stability_thr=0.40`, `auroc_thr=0.80`, `batch_size=4096`.

This is the canonical Matryoshka universality run — it is the only job that exercises Stages 3 and 4 (ancestor trace, prefix stability).

---

## Nomad jobs 2 & 3 — `pipeline-matryoshka-l{2,3}.nomad`

[`nomad/xray/analysis/pipeline-matryoshka-l2.nomad`](../nomad/xray/analysis/pipeline-matryoshka-l2.nomad), [`pipeline-matryoshka-l3.nomad`](../nomad/xray/analysis/pipeline-matryoshka-l3.nomad)

Identical structure, one for each Swin layer hooked (`pm_index = 2` and `pm_index = 3`). These run the **shell-fixed** universality analysis: pick a single Matryoshka shell (`--matryoshka_shell 24`) and treat the model as a conventional sparse code at that scale, then run the generic universality evaluator on top.

Four sequential, re-entrant steps (each skipped if its sentinel output already exists):

1. **Pre-step — merge per-dataset activations.** Concatenates the four `.safetensors` files into `activations_merged.safetensors` with an added `dataset_ids` tensor. This is what the scanner audit needs to fit its logistic classifier.
2. **Step 1 — Scanner audit.** `src/scripts/scanner_latent_audit.py` fits a per-latent `LogisticRegression` predicting `dataset_ids` from the activation. Threshold `0.80`. Produces `scanner_latent_audit.json`.
3. **Step 2 — Universality eval (`matryoshka_shell=24`).** `src/scripts/universality_eval.py` runs the Level-1 (frequency) and Level-2 (centroid cosine) filters on the shell-24 code, masked by the scanner-audit flags. Thresholds: `freq ≥ 0.01`, `cos ≥ 0.70`, `min_datasets ≥ 2`. Label CSVs are passed so the canonical-class agreement check can run.
4. **Step 3 — Null baseline.** `src/scripts/null_baseline_eval.py` repeats the same L1/L2 filters on a **random** and a **PCA** decoder of the same shape, producing the baseline expected under "no learned structure" — so the trained SAE's count can be compared against a chance level.
5. **Step 4 — Inspect latents.** For each confirmed latent, `src/scripts/inspect_universal_latents.py` exports the top-8 activating image patches per dataset to disk so they can be reviewed visually.

The `l2` and `l3` jobs differ only in the checkpoint path (`layer-2` vs `layer-3`) and the `pm_index` meta label — they let the analysis answer *"is the universal-feature count similar at the two layers we hook?"*

> **Scope note.** Because these pipelines fix `matryoshka_shell=24`, they do **not** exercise the Matryoshka-only stages (split audit, prefix stability). They treat the nested representation as a single ordinary top-k code. The headline Matryoshka-specific results come from `matryoshka-universality.nomad`; the per-layer pipelines exist to make fair comparisons against the BatchTopK baseline, which has no shells.

---

## How the two views combine

- **`matryoshka-universality.nomad`** answers *"does the Matryoshka nesting itself help us isolate universal features?"* (Stages 3 + 4 only make sense under the nested code.)
- **`pipeline-matryoshka-l{2,3}.nomad`** answer *"at a fixed sparsity shell, does a Matryoshka SAE find more universal features than the scanner-audited BatchTopK baseline, and how does that scale with depth?"*

Stages 0–2 are shared between the two views (with small implementation differences — the pipelines use a `LogisticRegression` fit, the single script uses raw-score AUROC). Stages 3–4 are Matryoshka-only and are the contribution this thesis makes on top of the standard universality filter stack.
