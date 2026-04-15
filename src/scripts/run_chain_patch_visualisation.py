"""Visualise top patches for matched feature chains across multiple datasets.

Extracts top activating patches for specific features corresponding to
user-defined chains (either predefined like CHAIN_1/CHAIN_2 or dynamically
derived from the cosine similarity CSVs). Side-by-side plots are produced
to visually evaluate cross-modality semantic equivalence.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import numpy as np
from PIL import Image

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Ensure configs/ is importable
_CONFIGS_DIR = _SRC_DIR.parent
if str(_CONFIGS_DIR) not in sys.path:
    sys.path.insert(0, str(_CONFIGS_DIR))

from configs.paths import DATASETS, CHAIN_1, CHAIN_2, OUTPUTS, derive_chain, SPATIAL_H, SPATIAL_W
from visualisation.semantic_dict import get_patch_crop
from visualisation.loaders import load_images

def render_chain(
    chain: dict[str, int], 
    chain_name: str, 
    top_k: int, 
    output_dir: Path
):
    print(f"\nRendering {chain_name}:")
    for k, v in chain.items():
        print(f"  {k.capitalize()}: F{v}")
        
    num_datasets = len(chain)
    
    fig = plt.figure(figsize=(4 * num_datasets, top_k * 1.5))
    gs = gridspec.GridSpec(top_k, num_datasets, wspace=0.1, hspace=0.1)
    
    dataset_keys = list(chain.keys())
    
    for col_idx, ds_key in enumerate(dataset_keys):
        feature_idx = chain[ds_key]
        
        info = DATASETS[ds_key]
        patches_dir = info["patches_dir"]
        stats_path = info["stats"]
        
        if not Path(stats_path).exists():
            print(f"  [ERROR] Stats file missing for {ds_key}: {stats_path}")
            continue
            
        stats = torch.load(stats_path, map_location="cpu")
        images = load_images(patches_dir)
        
        feature_max = stats["feature_max"][:, feature_idx]       # [n_images]
        feature_argmax = stats["feature_argmax"][:, feature_idx] # [n_images]
        
        top_indices = torch.topk(feature_max, min(top_k, feature_max.shape[0])).indices.tolist()
        
        # Plot col header
        fig.text(
            (col_idx + 0.5) / num_datasets, 0.92,
            f"{info['label'].capitalize()} (F{feature_idx})",
            ha="center", va="bottom", fontsize=14, fontweight="bold",
        )
        
        for row_idx, img_idx in enumerate(top_indices):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.axis("off")
            
            patch_idx = int(feature_argmax[img_idx].item())
            act_val = feature_max[img_idx].item()
            
            try:
                img_path = images[img_idx]
                img = Image.open(img_path).convert("RGB")
                crop = get_patch_crop(
                    img, patch_idx,
                    SPATIAL_H, SPATIAL_W,
                    context_patches=1,
                )
                ax.imshow(np.array(crop))
                ax.set_title(f"Act: {act_val:.2f}", fontsize=8, pad=2)
            except Exception as e:
                ax.text(0.5, 0.5, f"Err: {e}", ha="center", va="center",
                        fontsize=6, transform=ax.transAxes)

    plt.suptitle(
        f"Chain Visualisation: {chain_name}", 
        fontsize=18, fontweight="bold", y=0.98
    )
    
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"{chain_name.lower().replace(' ', '_')}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {chain_name} visualisation to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Render max-activating patches for feature chains")
    parser.add_argument("--chain", type=str, choices=["CHAIN_1", "CHAIN_2", "derived"], default="CHAIN_1",
                        help="Which chain to visualise")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top patches to show per feature")
    parser.add_argument("--start_ds", type=str, default="colon", help="Starting dataset for derived chain")
    parser.add_argument("--start_feat", type=int, default=155, help="Starting feature for derived chain")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUTS["chain_visualisation"]),
                        help="Where to save the visuals")
    
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    
    if args.chain == "CHAIN_1":
        chain = CHAIN_1
        chain_name = "Chain 1"
    elif args.chain == "CHAIN_2":
        chain = CHAIN_2
        chain_name = "Chain 2"
    elif args.chain == "derived":
        chain = derive_chain(args.start_ds, args.start_feat)
        chain_name = f"Derived Chain (Start: {args.start_ds.capitalize()} F{args.start_feat})"
    else:
        raise ValueError(f"Unknown chain option: {args.chain}")
        
    if not chain:
        print("Error: Derived chain is empty.")
        return
        
    render_chain(chain, chain_name, args.top_k, out_dir)

if __name__ == "__main__":
    main()
