#!/usr/bin/env python
"""MoCoO Figure 2 — 5×6 UMAP embedding grid.

Generates a grid of UMAP projections (rows = datasets, columns = configs),
coloured by ground-truth cell type from the source h5ad files.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import umap

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from benchmarks.scripts.plotting.shared import setup_fonts
from mocoo.visualization.style import (
    FS_AXIS,
    FS_LEGEND,
    FS_TITLE,
    FS_TICK,
    apply_style,
    get_config_colors,
    get_config_order,
    get_legend_name,
    save_figure,
)

setup_fonts()
apply_style()

_CONFIGS = get_config_order()
_CONFIG_COLORS = get_config_colors()
_DATASET_ORDER = ["IRALL", "dentate", "endo", "paul", "spinoids"]
_DATASET_DISPLAY = {
    "IRALL": "IRALL",
    "dentate": "Dentate",
    "endo": "Endo",
    "paul": "Paul",
    "spinoids": "Spinoids",
}
_CONFIG_DIR = {
    "VAE": "VAE",
    "VAE+ODE": "VAE_ODE",
    "VAE+MoCo": "VAE_MoCo",
    "VAE+MoCo+Proto": "VAE_MoCo_Proto",
    "VAE+ODE+MoCo": "VAE_ODE_MoCo",
    "Full": "Full",
}
DATASET_SPECS = {
    "IRALL": {"path": "LAB/scRL/IRALL.h5ad", "max_cells": 3000, "type_col": "cell_type"},
    "dentate": {"path": "vGAE_LAB/data/dentate.h5ad", "max_cells": 3000, "type_col": "ClusterName"},
    "endo": {"path": "vGAE_LAB/data/endo.h5ad", "max_cells": 2500, "type_col": "clusters"},
    "paul": {"path": "LAB/data/paul.h5ad", "max_cells": 2700, "type_col": "paul15_clusters"},
    "spinoids": {"path": "LAB/data/spinoids.h5ad", "max_cells": 3000, "type_col": "annotation"},
}


def _load_cell_types(data_dir: str, dataset: str) -> np.ndarray:
    """Load ground-truth cell-type labels matching the subsampled latents."""
    spec = DATASET_SPECS[dataset]
    adata = sc.read_h5ad(os.path.join(data_dir, spec["path"]))
    if adata.n_obs > spec["max_cells"]:
        sc.pp.subsample(adata, n_obs=spec["max_cells"], random_state=42)
    return adata.obs[spec["type_col"]].astype(str).values


def _load_latent(results_dir: Path, dataset: str, config: str) -> np.ndarray:
    """Load whole_latent from the latents.npz file."""
    npz_path = results_dir / dataset / _CONFIG_DIR[config] / "latents.npz"
    return np.load(str(npz_path))["whole_latent"]


def _compute_umap(latent: np.ndarray, seed: int = 42) -> np.ndarray:
    """Compute 2-D UMAP embedding."""
    reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=15,
                        min_dist=0.3)
    return reducer.fit_transform(latent)


def main():
    parser = argparse.ArgumentParser(description="Generate Figure 2 UMAP grid")
    parser.add_argument("--data-dir", default=os.environ.get("MOCOO_DATA_DIR",
                        os.path.expanduser("~")),
                        help="Base data directory for h5ad files")
    parser.add_argument("--results-dir",
                        default=str(Path(__file__).resolve().parent.parent.parent / "results"),
                        help="Benchmark results directory")
    parser.add_argument("--out-dir",
                        default=str(Path(__file__).resolve().parent.parent.parent / "figures"),
                        help="Output directory for figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_rows = len(_DATASET_ORDER)
    n_cols = len(_CONFIGS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.8, n_rows * 2.6),
                             constrained_layout=True)

    for ri, ds in enumerate(_DATASET_ORDER):
        print(f"Processing {ds}...")
        cell_types = _load_cell_types(args.data_dir, ds)
        unique_types = sorted(set(cell_types))
        cmap = plt.cm.get_cmap("tab20", len(unique_types))
        type_to_color = {ct: cmap(i) for i, ct in enumerate(unique_types)}

        for ci, cfg in enumerate(_CONFIGS):
            ax = axes[ri, ci]
            latent = _load_latent(results_dir, ds, cfg)
            emb = _compute_umap(latent)
            colors = [type_to_color[ct] for ct in cell_types]

            ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=1.5, alpha=0.6,
                       rasterized=True, linewidths=0)
            ax.set_xticks([])
            ax.set_yticks([])

            if ri == 0:
                ax.set_title(get_legend_name(cfg), fontsize=FS_TITLE, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(_DATASET_DISPLAY[ds], fontsize=FS_AXIS, fontweight="bold")

        # Per-dataset legend on the right
        ax_last = axes[ri, -1]
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=type_to_color[ct],
                       markersize=4, label=ct, linewidth=0)
            for ct in unique_types
        ]
        ax_last.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
                       fontsize=max(FS_TICK - 2, 5), frameon=False, handletextpad=0.3,
                       labelspacing=0.2, borderpad=0.1, ncol=1)

    fig.suptitle("UMAP Embeddings — 5 Datasets × 6 Configurations",
                 fontsize=FS_TITLE + 2, fontweight="bold", y=1.01)

    out_path = out_dir / "fig2_umap_grid.png"
    issues = save_figure(fig, str(out_path), vcd_label="fig2_umap_grid")
    plt.close(fig)

    print(f"Saved: {out_path}")
    print(f"Saved: {out_path.with_suffix('.pdf')}")
    if issues:
        print(f"VCD issues: {len(issues)}")
        for iss in issues:
            print(f"  - {iss}")


if __name__ == "__main__":
    main()
