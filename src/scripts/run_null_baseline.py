"""Null baseline computation for cosine similarity.

Generates a null distribution of cosine similarities by comparing
randomly initialised weight matrices of the same shape as the SAEs,
or optionally using real SAEs if provided. This establishes statistical
significance thresholds for the measured cross-modality similarities.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from visualisation.loaders import load_sae

def compute_null_distribution(W_a: torch.Tensor, W_b: torch.Tensor):
    """
    Computes cosine similarity matrix and returns the maximum similarity
    for each feature simulating the best match distribution.
    """
    W_a = F.normalize(W_a.float(), dim=1)
    W_b = F.normalize(W_b.float(), dim=1)
    
    sim = W_a @ W_b.T
    
    # Distribution of max similarities
    best_matches = sim.max(dim=1).values.cpu().numpy()
    
    p95 = np.percentile(best_matches, 95)
    p99 = np.percentile(best_matches, 99)
    mean_match = np.mean(best_matches)
    
    return best_matches, float(mean_match), float(p95), float(p99)

def main():
    parser = argparse.ArgumentParser(description="Compute null baseline for cross-SAE cosine similarity")
    parser.add_argument("--d_model", type=int, default=1024, help="Dimension of the model")
    parser.add_argument("--d_sae", type=int, default=4096, help="Dictionary size")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the results")
    parser.add_argument("--checkpoint_a", type=str, default=None, help="Optional real SAE to use instead of random")
    parser.add_argument("--checkpoint_b", type=str, default=None, help="Optional real SAE to use instead of random")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Configurations:")
    print(f"  d_model: {args.d_model}")
    print(f"  d_sae:   {args.d_sae}")
    print(f"  device:  {args.device}")
    
    if args.checkpoint_a:
        print(f"Loading real SAE A from {args.checkpoint_a}")
        sae_a = load_sae(args.checkpoint_a, device=args.device)
        W_a = sae_a.W_dec.data
    else:
        print("Generating random normal matrix A")
        W_a = torch.randn(args.d_sae, args.d_model, device=args.device)
        
    if args.checkpoint_b:
        print(f"Loading real SAE B from {args.checkpoint_b}")
        sae_b = load_sae(args.checkpoint_b, device=args.device)
        W_b = sae_b.W_dec.data
    else:
        print("Generating random normal matrix B")
        W_b = torch.randn(args.d_sae, args.d_model, device=args.device)
        
    print("Computing null distribution...")
    best_matches, mean_match, p95, p99 = compute_null_distribution(W_a, W_b)
    
    results = {
        "d_model": args.d_model,
        "d_sae": args.d_sae,
        "used_real_sae_a": args.checkpoint_a is not None,
        "used_real_sae_b": args.checkpoint_b is not None,
        "mean_best_match": round(mean_match, 4),
        "p95_threshold": round(p95, 4),
        "p99_threshold": round(p99, 4),
    }
    
    # Save JSON summary
    summary_path = out_dir / "null_baseline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved summary to {summary_path}")
    
    # Plot Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(best_matches, bins=50, color="lightsteelblue", edgecolor="white")
    ax.axvline(p95, color="orange", linestyle="--", label=f"95th percentile ({p95:.3f})")
    ax.axvline(p99, color="red", linestyle="--", label=f"99th percentile ({p99:.3f})")
    ax.set_title("Null Baseline Cosine Similarity Distribution")
    ax.set_xlabel("Max Cosine Similarity")
    ax.set_ylabel("Frequency")
    ax.legend()
    
    plot_path = out_dir / "null_baseline_distribution.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {plot_path}")
    
    print("\n" + "="*50)
    print(f"  Mean best match: {mean_match:.4f}")
    print(f"  95th percentile: {p95:.4f}")
    print(f"  99th percentile: {p99:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
