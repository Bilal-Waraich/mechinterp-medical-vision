"""Scanner latent audit for conglomerate SAE.

Detects SAE latents that encode scanner/hospital artefacts rather than
pathology (Concern #1 in single-modality-sae-concerns.md).

Method
------
For each latent f:
  1. Collect per-token activation magnitudes across all images.
  2. Fit a multinomial logistic regression from activation magnitude -> dataset_id.
  3. Compute one-vs-rest AUROC per dataset, then take the macro-average.

A latent with macro-AUROC > THRESHOLD is flagged as a "scanner latent" — it
carries significant dataset-predictive signal and should be excluded from
universality analysis.

Output
------
  <output_dir>/scanner_latent_audit.json
    {
      "threshold": 0.80,
      "n_flagged": <int>,
      "flagged_indices": [<int>, ...],
      "per_latent_auroc": [<float>, ...],   -- macro AUROC, one per latent
      "dataset_names": [...],
    }

  <output_dir>/scanner_latent_audit.csv
    latent_idx, macro_auroc, ds0_auroc, ds1_auroc, ..., flagged

Usage
-----
  python src/scripts/scanner_latent_audit.py \\
      --sae_checkpoint  data/checkpoints/step_0200000 \\
      --activations     data/xray_activations.safetensors \\
      --output_dir      data/audit \\
      --threshold       0.80 \\
      --device          cuda:0

Input format
------------
The ``--activations`` file must contain both ``activations`` and ``dataset_ids``
tensors.  BatchTopK jobs produce this format directly (single merged file).

Matryoshka jobs produce 4 separate per-dataset files WITHOUT a ``dataset_ids``
tensor.  To run the audit on a Matryoshka checkpoint, concatenate the 4 files
first:

  python -c "
  from safetensors.torch import load_file, save_file
  import torch
  files = [
      ('chexpert',  'activations_chexpert.safetensors'),
      ('rsna',      'activations_rsna.safetensors'),
      ('siim_covid','activations_siim.safetensors'),
      ('tb',        'activations_tb.safetensors'),
  ]
  acts, ids = [], []
  for ds_id, (name, path) in enumerate(files):
      d = load_file(path)
      acts.append(d['activations'])
      ids.append(torch.full((d['activations'].shape[0],), ds_id, dtype=torch.int32))
  save_file({'activations': torch.cat(acts), 'dataset_ids': torch.cat(ids)},
            'activations_merged.safetensors')
  "

See also
--------
docs/methodology.md §1.3 — scanner audit rationale and pipeline position
src/scripts/cross_dataset_eval.py — simplified universality proxy (run after audit)
src/scripts/universality_eval.py — TD-3 Level 1+2 metric (TO BE BUILT)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

DEFAULT_THRESHOLD = 0.80
DEFAULT_BATCH     = 4096   # tokens per forward pass through SAE encoder


def load_sae(checkpoint_dir: str | Path, device: str) -> tuple:
    """Load SAE weights and config from a checkpoint directory."""
    from safetensors.torch import load_file
    ckpt = Path(checkpoint_dir)

    weights = load_file(str(ckpt / "sae_weights.safetensors"), device=device)
    with open(ckpt / "cfg.json") as f:
        cfg = json.load(f)

    from sae_models import BatchTopKSAE
    model = BatchTopKSAE(
        d_model=cfg["d_model"],
        d_sae=cfg["d_sae"],
        k=cfg["k"],
        n_batches_to_dead=cfg.get("n_batches_to_dead", 25),
        top_k_aux=cfg.get("top_k_aux", 512),
        aux_penalty=cfg.get("aux_penalty", 1 / 32),
        l1_coeff=cfg.get("l1_coeff", 2e-4),
    )
    model.load_state_dict(weights)
    model.eval()
    model.to(device)
    return model, cfg


def encode_activations(
    model,
    acts: torch.Tensor,
    device: str,
    batch_size: int = DEFAULT_BATCH,
) -> torch.Tensor:
    """Run SAE encoder on all activations, return sparse codes [N, d_sae]."""
    d_sae = model.W_enc.shape[1]
    codes = torch.zeros(len(acts), d_sae, dtype=torch.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(acts), batch_size):
            chunk = acts[start : start + batch_size].to(device)
            out = model.encode(chunk)   # top-k sparse [B, d_sae]
            codes[start : start + out.shape[0]] = out.cpu()
    return codes


def compute_per_latent_auroc(
    codes: torch.Tensor,
    dataset_ids: torch.Tensor,
    n_datasets: int,
    sample_size: int = 100_000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a logistic regression probe per latent and return AUROCs.

    Implements the scanner audit described in ``docs/methodology.md`` §1.3:
    for each latent we fit a multinomial logistic regression from the latent's
    per-token activation magnitude to ``dataset_id``, then score the probe
    with one-vs-rest AUROC (macro-averaged across datasets).  Latents whose
    macro AUROC exceeds the caller's threshold are scanner latents and are
    excluded from the universality analysis.

    To prevent the fit from being dominated by the largest dataset (CheXpert
    contributes ~15× as many tokens as TB after post-spatial expansion), we
    draw a balanced subsample up front: ``sample_size // n_datasets`` tokens
    per dataset.  The same subsample is reused for every latent so results
    are comparable and the whole audit stays within minutes rather than
    hours.  Dead latents (no activation in the sample) are skipped and
    recorded as AUROC = 0.

    Parameters
    ----------
    codes         : [N, d_sae] float   sparse SAE codes
    dataset_ids   : [N] int            dataset index per token
    n_datasets    : int                number of datasets
    sample_size   : int                total tokens to use (0 → use all)
    seed          : int                RNG seed for deterministic subsampling

    Returns
    -------
    macro_aurocs  : [d_sae] float — macro-averaged one-vs-rest AUROC
    per_ds_aurocs : [d_sae, n_datasets] float
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import label_binarize

    codes_np = codes.numpy()
    ids_np   = dataset_ids.numpy().astype(int)
    d_sae    = codes_np.shape[1]

    # ── Balanced subsample ────────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    if sample_size and sample_size < codes_np.shape[0]:
        per_ds = max(1, sample_size // n_datasets)
        idx_list = []
        for ds in range(n_datasets):
            pool = np.where(ids_np == ds)[0]
            if pool.size == 0:
                continue
            n_take = min(per_ds, pool.size)
            idx_list.append(rng.choice(pool, size=n_take, replace=False))
        sample_idx = np.concatenate(idx_list)
    else:
        sample_idx = np.arange(codes_np.shape[0])

    codes_s = codes_np[sample_idx]
    ids_s   = ids_np[sample_idx]
    classes = list(range(n_datasets))
    ids_bin_s = label_binarize(ids_s, classes=classes)          # [N_s, n_datasets]
    # label_binarize collapses to a single column in the binary case.
    if ids_bin_s.shape[1] == 1:
        ids_bin_s = np.hstack([1 - ids_bin_s, ids_bin_s])

    print(f"  Probe fit set: {len(sample_idx):,} tokens  "
          f"({n_datasets} classes, balanced)")

    # ── Per-latent fit ────────────────────────────────────────────────────
    macro_aurocs  = np.zeros(d_sae, dtype=np.float32)
    per_ds_aurocs = np.zeros((d_sae, n_datasets), dtype=np.float32)

    for i in range(d_sae):
        feat = codes_s[:, i : i + 1]          # [N_s, 1]

        # Skip dead latents (no activation in the sample)
        if feat.max() < 1e-8:
            continue

        # Default sklearn LogisticRegression (lbfgs solver) fits a
        # multinomial softmax when len(classes) > 2.  Single-feature fits
        # converge in well under a second on ~100k samples.
        try:
            clf = LogisticRegression(max_iter=200, random_state=seed)
            clf.fit(feat, ids_s)
        except ValueError:
            continue

        proba = clf.predict_proba(feat)       # [N_s, len(clf.classes_)]
        cls_index = {int(c): j for j, c in enumerate(clf.classes_)}

        ds_aurocs = []
        for ds in range(n_datasets):
            y_true = ids_bin_s[:, ds]
            if ds not in cls_index or y_true.sum() == 0 or y_true.sum() == len(y_true):
                ds_aurocs.append(0.5)
                continue
            score = proba[:, cls_index[ds]]
            try:
                ds_aurocs.append(float(roc_auc_score(y_true, score)))
            except ValueError:
                ds_aurocs.append(0.5)

        per_ds_aurocs[i] = ds_aurocs
        macro_aurocs[i]  = float(np.mean(ds_aurocs))

        if (i + 1) % 500 == 0:
            print(f"  [{i+1:>6}/{d_sae}] current latent AUROC={macro_aurocs[i]:.3f}")

    return macro_aurocs, per_ds_aurocs


def main():
    ap = argparse.ArgumentParser(
        description="Scanner latent audit for conglomerate SAE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--sae_checkpoint",  required=True,
                    help="Path to SAE checkpoint directory (contains sae_weights.safetensors + cfg.json)")
    ap.add_argument("--activations",     required=True,
                    help="Path to harvested activations .safetensors (must contain dataset_ids)")
    ap.add_argument("--output_dir",      default="data/audit")
    ap.add_argument("--threshold",       type=float, default=DEFAULT_THRESHOLD,
                    help="AUROC threshold above which a latent is flagged as scanner latent")
    ap.add_argument("--batch_size",      type=int, default=DEFAULT_BATCH)
    ap.add_argument("--probe_sample_size", type=int, default=100_000,
                    help="Balanced token sample size for the per-latent LR probe. "
                         "0 to use all tokens (slow).")
    ap.add_argument("--seed",            type=int, default=42,
                    help="RNG seed for the balanced probe subsample.")
    ap.add_argument("--device",
                    default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────
    from safetensors.torch import load_file
    print(f"[Audit] Loading activations from {args.activations} ...")
    raw = load_file(args.activations)
    acts        = raw["activations"].to(torch.float32)
    dataset_ids = raw["dataset_ids"].to(torch.int32)
    print(f"[Audit] Activations: {tuple(acts.shape)}")

    if dataset_ids.shape[0] != acts.shape[0]:
        sys.exit(
            f"[Audit] ERROR: dataset_ids has {dataset_ids.shape[0]:,} entries but "
            f"activations has {acts.shape[0]:,} rows — lengths must match.\n"
            f"This file was likely produced by a pre-Bug-A harvest_all.py "
            f"(dataset_ids stored per-image, not per-token).\n"
            f"Re-harvest with the current harvest_all.py."
        )

    n_datasets   = int(dataset_ids.max().item()) + 1
    dataset_names = [f"dataset_{i}" for i in range(n_datasets)]

    # Try to read dataset names from metadata
    try:
        import json as _json
        meta_path = Path(args.activations).parent / (
            Path(args.activations).stem + "_metadata.json"
        )
        if meta_path.exists():
            with open(meta_path) as f:
                meta = _json.load(f)
            if "datasets" in meta:
                for ds in meta["datasets"]:
                    dataset_names[ds["index"]] = ds["name"]
    except Exception:
        pass

    print(f"[Audit] {n_datasets} datasets: {dataset_names}")

    # ── Load SAE ──────────────────────────────────────────────────────────
    print(f"[Audit] Loading SAE from {args.sae_checkpoint} ...")
    model, cfg = load_sae(args.sae_checkpoint, args.device)
    d_sae = cfg["d_sae"]
    print(f"[Audit] SAE d_model={cfg['d_model']}  d_sae={d_sae}  k={cfg['k']}")

    # ── Encode activations ────────────────────────────────────────────────
    print(f"[Audit] Encoding {len(acts):,} tokens through SAE encoder ...")
    codes = encode_activations(model, acts, args.device, args.batch_size)
    print(f"[Audit] Sparse codes: {tuple(codes.shape)}")

    # ── Compute per-latent AUROC ──────────────────────────────────────────
    print(f"[Audit] Fitting logistic probes for {d_sae} latents ...")
    try:
        import sklearn  # noqa: F401
    except ImportError:
        sys.exit("scikit-learn required: pip install scikit-learn")

    macro_aurocs, per_ds_aurocs = compute_per_latent_auroc(
        codes, dataset_ids, n_datasets,
        sample_size=args.probe_sample_size,
        seed=args.seed,
    )

    # ── Flag scanner latents ──────────────────────────────────────────────
    flagged = np.where(macro_aurocs > args.threshold)[0].tolist()
    print(f"\n[Audit] Flagged {len(flagged)} / {d_sae} latents "
          f"(AUROC > {args.threshold:.2f})")

    # ── Save outputs ──────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audit_json = {
        "threshold":          args.threshold,
        "probe_sample_size":  args.probe_sample_size,
        "probe_seed":         args.seed,
        "n_flagged":          len(flagged),
        "n_total_latents":    d_sae,
        "flagged_fraction":   len(flagged) / d_sae,
        "flagged_indices":    flagged,
        "per_latent_auroc":   macro_aurocs.tolist(),
        "dataset_names":      dataset_names,
    }
    json_path = out_dir / "scanner_latent_audit.json"
    with open(json_path, "w") as f:
        json.dump(audit_json, f, indent=2)
    print(f"[Audit] JSON  -> {json_path}")

    csv_path = out_dir / "scanner_latent_audit.csv"
    with open(csv_path, "w", newline="") as f:
        cols = ["latent_idx", "macro_auroc"] + [f"auroc_{n}" for n in dataset_names] + ["flagged"]
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for i in range(d_sae):
            row = {
                "latent_idx":  i,
                "macro_auroc": f"{macro_aurocs[i]:.4f}",
                "flagged":     int(i in flagged),
            }
            for j, name in enumerate(dataset_names):
                row[f"auroc_{name}"] = f"{per_ds_aurocs[i, j]:.4f}"
            writer.writerow(row)
    print(f"[Audit] CSV   -> {csv_path}")

    # Summary: top flagged latents
    if flagged:
        top = sorted(flagged, key=lambda i: -macro_aurocs[i])[:10]
        print("\n[Audit] Top flagged latents:")
        for i in top:
            ds_str = "  ".join(
                f"{dataset_names[j]}={per_ds_aurocs[i, j]:.2f}"
                for j in range(n_datasets)
            )
            print(f"  latent {i:>5}  macro={macro_aurocs[i]:.3f}  [{ds_str}]")

    print("\n[Audit] Done.")
    print(f"  Flagged latents saved to {json_path}")
    print(f"  Use 'flagged_indices' to exclude scanner latents from Phase 3 analysis.")


if __name__ == "__main__":
    main()
