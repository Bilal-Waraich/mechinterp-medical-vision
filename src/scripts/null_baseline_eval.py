"""Null baseline for universality evaluation.

Repeats the TD-3 Level 1 + Level 2 universality checks on two null models:

  random  — BatchTopKSAE with the same architecture but randomly initialised
             weights. Measures how many latents pass by chance given the
             activation distribution.

  pca     — Top-d_sae PCA components of the activation data treated as
             "latents". A PCA component "fires" when its projection magnitude
             exceeds a threshold derived from the same 1% frequency criterion.
             Measures universality of linear structure already present in the
             encoder without any SAE training.

The primary thesis claim is that the trained SAE's confirmed universal latent
count is significantly above the null rate from both baselines.

Usage
-----
python src/scripts/null_baseline_eval.py \\
    --checkpoint_dir  /deep_learning/output/.../checkpoints/step_0200000 \\
    --activations     /deep_learning/output/.../activations_chexpert.safetensors:chexpert \\
    --activations     /deep_learning/output/.../activations_rsna.safetensors:rsna \\
    --activations     /deep_learning/output/.../activations_siim.safetensors:siim_covid \\
    --activations     /deep_learning/output/.../activations_tb.safetensors:tb \\
    --null_mode       random pca \\
    --output_dir      /deep_learning/output/.../universality/null_baseline \\
    --device          cuda:0

See also
--------
docs/methodology.md §1.5 — null baseline rationale
src/scripts/universality_eval.py — run this first to get the trained SAE results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from tqdm.auto import tqdm

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sae_models import BatchTopKSAE
from visualisation.loaders import load_sae
from scripts.universality_eval import (
    load_dataset_activations,
    compute_frequency_per_dataset,
    apply_level1_filter,
    compute_centroids_per_dataset,
    apply_level2_filter,
    FREQUENCY_THRESHOLD,
)


# ══════════════════════════════════════════════════════════════════════════════
# Null model builders
# ══════════════════════════════════════════════════════════════════════════════

def build_random_sae(reference_sae):
    """Return a randomly initialised SAE with the same architecture."""
    from matryoshka_sae import MatryoshkaSAE
    if isinstance(reference_sae, MatryoshkaSAE):
        model = MatryoshkaSAE(
            d_model=reference_sae.d_model,
            d_sae=reference_sae.d_sae,
            k_schedule=reference_sae.k_schedule,
            scale_weights=getattr(reference_sae, "scale_weights", None),
            n_batches_to_dead=reference_sae.n_batches_to_dead,
            aux_penalty=reference_sae.aux_penalty,
            l1_coeff=reference_sae.l1_coeff,
        )
    else:
        model = BatchTopKSAE(
            d_model=reference_sae.d_model,
            d_sae=reference_sae.d_sae,
            k=reference_sae.k,
            n_batches_to_dead=reference_sae.n_batches_to_dead,
            aux_penalty=reference_sae.aux_penalty,
            l1_coeff=reference_sae.l1_coeff,
        )
    model.eval()
    return model


class PCANullSAE:
    """Wraps a PCA projection to expose the same encode() interface as BatchTopKSAE.

    A PCA component is treated as "firing" on a token if its projection magnitude
    is above the top-k threshold — same k as the reference SAE — so the sparsity
    level is matched.

    Parameters
    ----------
    components : Tensor [d_sae, d_model]   (PCA eigenvectors, unit norm rows)
    k          : int                        (number of active components per token)
    """

    def __init__(self, components: torch.Tensor, k: int):
        self.components = components          # [d_sae, d_model]
        self.k = k
        self.d_sae   = components.shape[0]
        self.d_model = components.shape[1]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Project x onto PCA components, keep top-k magnitudes, zero the rest.

        Args:
            x : Tensor [B, d_model]

        Returns:
            Tensor [B, d_sae]  sparse (top-k non-zero)
        """
        proj = x @ self.components.T          # [B, d_sae]  signed projections
        magnitudes = proj.abs()
        k = min(self.k, self.d_sae)
        topk_vals, topk_idx = magnitudes.topk(k, dim=1)
        sparse = torch.zeros_like(magnitudes)
        sparse.scatter_(1, topk_idx, topk_vals)
        return sparse

    def to(self, device):
        self.components = self.components.to(device)
        return self

    def eval(self):
        return self


@torch.no_grad()
def build_pca_sae(
    datasets: list[tuple[str, torch.Tensor]],
    d_sae: int,
    k: int,
    device: torch.device,
    sample_cap: int = 50_000,
) -> PCANullSAE:
    """Fit PCA on a sample of activations from all datasets combined.

    Args:
        datasets:   list of (name, activations [N_i, d_model])
        d_sae:      number of PCA components to keep (matches SAE dictionary size)
        k:          top-k threshold matching reference SAE
        device:     torch device for projection
        sample_cap: max tokens to sample per dataset for PCA fitting

    Returns:
        PCANullSAE ready for encode()
    """
    print(f"  Fitting PCA ({d_sae} components) on activation sample …")
    chunks = []
    for _, acts in datasets:
        n = min(sample_cap, acts.shape[0])
        idx = torch.randperm(acts.shape[0])[:n]
        chunks.append(acts[idx].float())
    data = torch.cat(chunks, dim=0)            # [N_sample, d_model]

    # Centre
    mean = data.mean(dim=0)
    data_c = data - mean

    # SVD — use torch.linalg.svd for GPU compatibility
    try:
        _, _, Vt = torch.linalg.svd(data_c, full_matrices=False)
    except Exception:
        # Fallback: numpy SVD on CPU
        u, s, vt = np.linalg.svd(data_c.numpy(), full_matrices=False)
        Vt = torch.from_numpy(vt)

    components = Vt[:d_sae]                    # [d_sae, d_model]
    # Rows are already unit norm from SVD
    print(f"  PCA components shape: {list(components.shape)}")
    return PCANullSAE(components=components.to(device), k=k)


# ══════════════════════════════════════════════════════════════════════════════
# Run null evaluation
# ══════════════════════════════════════════════════════════════════════════════

def run_null_eval(
    null_model,
    null_name: str,
    datasets: list[tuple[str, torch.Tensor]],
    device: torch.device,
    freq_threshold: float,
    min_datasets: int,
    cosine_threshold: float,
    batch_size: int,
) -> dict:
    """Run Level 1 + Level 2 on a null model and return summary counts."""

    print(f"\n── Null model: {null_name} ──")

    # Level 1
    freq_per_ds, n_tokens = compute_frequency_per_dataset(
        datasets, null_model, device, batch_size=batch_size)
    candidates, active_datasets = apply_level1_filter(
        freq_per_ds, n_tokens, flagged_indices=set(),
        threshold=freq_threshold, min_datasets=min_datasets)
    print(f"  Level 1 candidates: {len(candidates)}")

    if not candidates:
        return {
            "null_model":          null_name,
            "n_level1_candidates": 0,
            "n_level2_confirmed":  0,
            "level1_rate":         0.0,
            "level2_rate":         0.0,
        }

    # Level 2
    centroids = compute_centroids_per_dataset(
        datasets, candidates, null_model, device, batch_size=batch_size)
    confirmed, _ = apply_level2_filter(
        candidates, active_datasets, centroids,
        cosine_threshold=cosine_threshold)
    print(f"  Level 2 confirmed:  {len(confirmed)}")

    d_sae = null_model.d_sae
    return {
        "null_model":          null_name,
        "d_sae":               d_sae,
        "n_level1_candidates": len(candidates),
        "n_level2_confirmed":  len(confirmed),
        "level1_rate":         round(len(candidates) / d_sae, 5),
        "level2_rate":         round(len(confirmed)  / d_sae, 5),
        "confirmed_indices":   confirmed,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Null baseline universality evaluation (random SAE + PCA).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--checkpoint_dir", required=True,
                    help="Reference SAE checkpoint (architecture is copied for random null).")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--activations", action="append", metavar="PATH:LABEL",
                     help="Per-dataset activation files (Matryoshka format).")
    grp.add_argument("--merged_activations", metavar="PATH",
                     help="Merged activation file with dataset_ids (BatchTopK format).")
    ap.add_argument("--dataset_names", nargs="+",
                    default=["chexpert", "rsna", "siim_covid", "tb"])
    ap.add_argument("--null_mode", nargs="+", choices=["random", "pca"],
                    default=["random", "pca"],
                    help="Which null models to evaluate.")
    ap.add_argument("--trained_summary",
                    help="universality_summary.json from universality_eval.py "
                         "(used to print comparison). Optional.")
    ap.add_argument("--freq_threshold",   type=float, default=FREQUENCY_THRESHOLD)
    ap.add_argument("--min_datasets",     type=int,   default=2)
    ap.add_argument("--cosine_threshold", type=float, default=0.70)
    ap.add_argument("--pca_sample_cap",   type=int,   default=50_000,
                    help="Max tokens per dataset to use for PCA fitting.")
    ap.add_argument("--matryoshka_shell", type=int, default=None, metavar="K",
                    help="If set, restrict Matryoshka SAE encode() to the top-K shell "
                         "(e.g. --matryoshka_shell 24). Ignored for BatchTopK checkpoints.")
    ap.add_argument("--output_dir",  required=True)
    ap.add_argument("--batch_size",  type=int, default=4096)
    ap.add_argument("--device",
                    default="cuda:0" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  Null Baseline Universality Evaluation")
    print("=" * 65)

    # Load reference SAE (for architecture)
    print(f"\nLoading reference SAE from {args.checkpoint_dir} …")
    ref_sae = load_sae(args.checkpoint_dir, device=str(device))
    ref_sae.eval()
    print(f"  d_model={ref_sae.d_model}  d_sae={ref_sae.d_sae}  k={ref_sae.k}")

    if args.matryoshka_shell is not None:
        shell_k = args.matryoshka_shell
        _orig = ref_sae.encode
        ref_sae.encode = lambda x, k=shell_k: _orig(x, k=k)
        print(f"  Matryoshka shell: restricting encode() to top-{shell_k} features")

    # Load activations
    print("\nLoading activations …")
    datasets = load_dataset_activations(args)
    for name, acts in datasets:
        print(f"  {name}: {acts.shape[0]:,} tokens")

    results = []

    # Random SAE null
    if "random" in args.null_mode:
        random_sae = build_random_sae(ref_sae)
        random_sae.to(device)
        result = run_null_eval(
            random_sae, "random_sae", datasets, device,
            args.freq_threshold, args.min_datasets, args.cosine_threshold,
            args.batch_size)
        results.append(result)

    # PCA null
    if "pca" in args.null_mode:
        pca_model = build_pca_sae(datasets, ref_sae.d_sae, ref_sae.k, device,
                                   sample_cap=args.pca_sample_cap)
        result = run_null_eval(
            pca_model, "pca", datasets, device,
            args.freq_threshold, args.min_datasets, args.cosine_threshold,
            args.batch_size)
        results.append(result)

    # Load trained SAE results for comparison
    trained_summary: dict = {}
    if args.trained_summary:
        with open(args.trained_summary) as f:
            trained_summary = json.load(f)

    # Save
    output = {
        "trained_sae": trained_summary or "(not provided)",
        "null_baselines": results,
    }
    out_path = out_dir / "null_baseline_results.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n  ✓ Saved → {out_path}")

    # Print comparison table
    print(f"\n{'=' * 65}")
    print(f"  {'Model':<20} {'L1 candidates':>15} {'L2 confirmed':>14}")
    print(f"  {'-'*49}")
    if trained_summary:
        print(f"  {'trained SAE':<20} "
              f"{trained_summary.get('n_level1_candidates', '?'):>15} "
              f"{trained_summary.get('n_level2_confirmed', '?'):>14}")
    for r in results:
        print(f"  {r['null_model']:<20} {r['n_level1_candidates']:>15} {r['n_level2_confirmed']:>14}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
