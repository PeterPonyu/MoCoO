#!/usr/bin/env python
"""MoCoO Supplementary Figure S4 — ODE-driven pseudotime & trajectory analysis.

Four-panel figure (IRALL, 3 representative configs: VAE, VAE+ODE, Full):
  (A) PCA of latent spaces coloured by cell type (top row) and pseudotime (bottom)
  (B) Per-cell-type pseudotime distribution (violin plots)
  (C) Per-dimension gradient magnitude (gene expression importance along ODE axis)
  (D) Trajectory smoothness: pairwise distance histograms per config
"""
from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from benchmarks.scripts.plotting.shared import setup_fonts
from mocoo.visualization.style import (
    FS_AXIS, FS_LEGEND, FS_SMALL, FS_TICK, FS_TITLE,
    FIG_WIDTH_IN, DPI,
    apply_style, save_figure, add_panel_label,
    get_config_colors, get_display_name,
)
from mocoo.visualization.direct_layout import bind_figure_region

setup_fonts()
apply_style()

_CONFIGS = ["VAE", "VAE+ODE", "Full"]
_DIR_MAP = {"VAE": "VAE", "VAE+ODE": "VAE_ODE", "Full": "Full"}


def _load_latents(results_dir: Path, config: str):
    cfg_dir = _DIR_MAP.get(config, config)
    fp = results_dir / "IRALL" / cfg_dir / "latents.npz"
    if not fp.exists():
        return None, None
    d = np.load(fp)
    return d["whole_latent"], d["whole_labels"]


def _load_gradients(results_dir: Path, config: str):
    cfg_dir = _DIR_MAP.get(config, config)
    fp = results_dir / "IRALL" / cfg_dir / "gradients.npy"
    if not fp.exists():
        return None
    return np.load(fp)


def _compute_pseudotime(latent: np.ndarray) -> np.ndarray:
    """Infer pseudotime as PC1 projection, normalised to [0, 1]."""
    from sklearn.decomposition import PCA
    pc1 = PCA(n_components=1).fit_transform(latent).ravel()
    mn, mx = pc1.min(), pc1.max()
    if mx - mn < 1e-12:
        return np.zeros_like(pc1)
    return (pc1 - mn) / (mx - mn)


def _pca2d(latent: np.ndarray) -> np.ndarray:
    from sklearn.decomposition import PCA
    return PCA(n_components=2).fit_transform(latent)


def _pairwise_dists(latent: np.ndarray, n_sample: int = 2000) -> np.ndarray:
    """Compute sequential pairwise distances after pseudotime ordering."""
    pt = _compute_pseudotime(latent)
    order = np.argsort(pt)
    z = latent[order]
    if len(z) > n_sample:
        idx = np.linspace(0, len(z) - 1, n_sample, dtype=int)
        z = z[idx]
    return np.linalg.norm(np.diff(z, axis=0), axis=1)


def make_figure(results_dir: Path, out_path: Path):
    config_colors = get_config_colors()

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * 1.0))
    root = bind_figure_region(fig, (0.10, 0.06, 0.88, 0.90))
    (r_top, r_bot) = root.split_rows([1, 1], gap=0.12)
    (r_a, r_b) = r_top.split_cols([3, 2], gap=0.14)
    (r_c, r_d) = r_bot.split_cols([1, 1], gap=0.14)

    # ── Panel (A): PCA coloured by cell type (3 configs side by side) ──
    pca_regions = r_a.split_rows([1, 1], gap=0.08)
    pca_type_regions = pca_regions[0].split_cols([1] * len(_CONFIGS), gap=0.06)
    pca_pt_regions = pca_regions[1].split_cols([1] * len(_CONFIGS), gap=0.06)

    all_labels = set()
    # Collect label set
    for cfg in _CONFIGS:
        lat, lbl = _load_latents(results_dir, cfg)
        if lbl is not None:
            all_labels.update(np.unique(lbl))
    label_list = sorted(all_labels)
    label_cmap = plt.cm.tab20(np.linspace(0, 1, max(len(label_list), 1)))

    for i, cfg in enumerate(_CONFIGS):
        lat, lbl = _load_latents(results_dir, cfg)
        if lat is None:
            continue
        pca = _pca2d(lat)
        pt = _compute_pseudotime(lat)

        # Cell type PCA
        ax_type = pca_type_regions[i].add_axes(fig)
        for j, lab in enumerate(label_list):
            mask = (lbl == lab) if isinstance(lab, str) else (lbl == lab)
            ax_type.scatter(pca[mask, 0], pca[mask, 1], s=1.5,
                            c=[label_cmap[j % len(label_cmap)]], alpha=0.5,
                            rasterized=True)
        ax_type.set_title(get_display_name(cfg), fontsize=FS_TICK)
        ax_type.set_xticks([])
        ax_type.set_yticks([])
        if i == 0:
            ax_type.set_ylabel("Cell type", fontsize=FS_SMALL)

        # Pseudotime PCA
        ax_pt = pca_pt_regions[i].add_axes(fig)
        sc = ax_pt.scatter(pca[:, 0], pca[:, 1], s=1.5, c=pt,
                           cmap="viridis", alpha=0.5, rasterized=True)
        ax_pt.set_xticks([])
        ax_pt.set_yticks([])
        if i == 0:
            ax_pt.set_ylabel("Pseudotime", fontsize=FS_SMALL)

    add_panel_label(pca_type_regions[0].add_axes(fig) if False else
                    fig.axes[0], "A", x=-0.30, y=1.16)

    # ── Panel (B): Pseudotime distributions per cell type ──
    ax_b = r_b.add_axes(fig)
    lat_full, lbl_full = _load_latents(results_dir, "Full")
    if lat_full is not None:
        pt_full = _compute_pseudotime(lat_full)
        unique_labels = sorted(set(lbl_full) if isinstance(lbl_full[0], str)
                               else np.unique(lbl_full).tolist())
        vdata = [pt_full[lbl_full == lab] for lab in unique_labels]
        parts = ax_b.violinplot(vdata, positions=range(len(unique_labels)),
                                showmeans=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_alpha(0.6)
        ax_b.set_xticks(range(len(unique_labels)))
        tick_labels = [str(l)[:8] for l in unique_labels]
        ax_b.set_xticklabels(tick_labels, fontsize=FS_SMALL - 1,
                             rotation=45, ha="right")
        ax_b.set_ylabel("Pseudotime", fontsize=FS_AXIS)
        ax_b.set_title("Per-type Distribution (Full)", fontsize=FS_TITLE)
    add_panel_label(ax_b, "B", x=-0.34, y=1.16)

    # ── Panel (C): Gradient magnitude per latent dimension ──
    ax_c = r_c.add_axes(fig)
    for cfg in _CONFIGS:
        grads = _load_gradients(results_dir, cfg)
        if grads is None:
            continue
        dim_importance = np.mean(np.abs(grads), axis=0)
        color = config_colors.get(cfg, "#888888")
        ax_c.bar(np.arange(len(dim_importance)) + _CONFIGS.index(cfg) * 0.25,
                 dim_importance, width=0.22, color=color, alpha=0.8,
                 label=get_display_name(cfg))
    ax_c.set_xlabel("Latent Dimension", fontsize=FS_AXIS)
    ax_c.set_ylabel("Mean |Gradient|", fontsize=FS_AXIS)
    ax_c.set_title("Gene Expression Importance", fontsize=FS_TITLE)
    ax_c.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_c.legend(fontsize=FS_SMALL, loc="upper right")
    add_panel_label(ax_c, "C", x=-0.26, y=1.20)

    # ── Panel (D): Trajectory smoothness ──
    ax_d = r_d.add_axes(fig)
    for cfg in _CONFIGS:
        lat, _ = _load_latents(results_dir, cfg)
        if lat is None:
            continue
        dists = _pairwise_dists(lat)
        color = config_colors.get(cfg, "#888888")
        ax_d.hist(dists, bins=50, alpha=0.5, color=color, density=True,
                  label=get_display_name(cfg))
    ax_d.set_xlabel("Sequential Pairwise Distance", fontsize=FS_AXIS)
    ax_d.set_ylabel("Density", fontsize=FS_AXIS)
    ax_d.set_title("Trajectory Smoothness", fontsize=FS_TITLE)
    ax_d.legend(fontsize=FS_SMALL, loc="upper right")
    add_panel_label(ax_d, "D", x=-0.18, y=1.14)

    save_figure(fig, str(out_path), vcd_label="figS4_ode_trajectory",
                vcd_verbose=True)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="MoCoO Supp Fig S4: ODE Trajectory")
    parser.add_argument("--resultsdir", type=Path,
                        default=Path(__file__).resolve().parent.parent.parent
                        / "results")
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=None)
    args = parser.parse_args()
    outdir = args.outdir or (args.resultsdir.parent / "figures")
    Path(outdir).mkdir(parents=True, exist_ok=True)
    make_figure(args.resultsdir, Path(outdir) / "figS4_ode_trajectory.png")


if __name__ == "__main__":
    main()
