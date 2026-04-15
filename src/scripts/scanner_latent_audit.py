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
) -> tuple[np.ndarray, np.ndarray]:
    """Fit logistic regression per latent, return macro AUROC and per-dataset AUROCs.

    To keep memory manageable, we fit one probe per latent (d_sae independent
    logistic regressions) using the latent's activation magnitude as the sole
    feature.  This is deliberately weak — a single-feature probe that achieves
    AUROC > 0.80 is a strong signal of scanner encoding.

    Returns
    -------
    macro_aurocs  : [d_sae] float — macro-averaged one-vs-rest AUROC
    per_ds_aurocs : [d_sae, n_datasets] float
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import label_binarize

    d_sae = codes.shape[1]
    codes_np = codes.numpy()
    ids_np   = dataset_ids.numpy().astype(int)
    classes  = list(range(n_datasets))
    ids_bin  = label_binarize(ids_np, classes=classes)  # [N, n_datasets]

    macro_aurocs  = np.zeros(d_sae, dtype=np.float32)
    per_ds_aurocs = np.zeros((d_sae, n_datasets), dtype=np.float32)

    for i in range(d_sae):
        feat = codes_np[:, i : i + 1]          # [N, 1]

        # Skip dead latents (no activation)
        if feat.max() < 1e-8:
            continue

        ds_aurocs = []
        for ds in range(n_datasets):
            y_true = ids_bin[:, ds]
            if y_true.sum() == 0 or y_true.sum() == len(y_true):
                ds_aurocs.append(0.5)
                continue
            try:
                auc = roc_auc_score(y_true, feat[:, 0])
            except Exception:
                auc = 0.5
            ds_aurocs.append(float(auc))

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
        codes, dataset_ids, n_datasets
    )

    # ── Flag scanner latents ──────────────────────────────────────────────
    flagged = np.where(macro_aurocs > args.threshold)[0].tolist()
    print(f"\n[Audit] Flagged {len(flagged)} / {d_sae} latents "
          f"(AUROC > {args.threshold:.2f})")

    # ── Save outputs ──────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audit_json = {
        "threshold":        args.threshold,
        "n_flagged":        len(flagged),
        "n_total_latents":  d_sae,
        "flagged_fraction": len(flagged) / d_sae,
        "flagged_indices":  flagged,
        "per_latent_auroc": macro_aurocs.tolist(),
        "dataset_names":    dataset_names,
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
