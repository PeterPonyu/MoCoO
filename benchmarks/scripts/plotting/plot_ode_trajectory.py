#!/usr/bin/env python
"""
MoCoO Figure 6 — ODE-Driven Pseudotime & Cellular Trajectory Analysis
======================================================================
The key differentiator of MoCoO vs. plain VAE is the Neural ODE component.
This figure demonstrates that the ODE latent space captures *continuous*
cellular dynamics that can be interpreted as pseudotime.

Layout (17 × 21 cm):
  Row 0 (A): PCA of VAE+ODE latent space — 3 views:
             left=coloured by cell type, middle=coloured by pseudotime (PC1),
             right=coloured by pseudotime (PC2).
             VAE (no ODE) shown as comparison.
  Row 1 (B): Pseudotime distributions — violin plots of pseudotime per cell
             type, for VAE vs VAE+ODE. Shows that ODE latent space
             *stratifies* cell states more continuously.
  Row 2 (C): Gene expression along pseudotime — line plots of top 8
             marker genes (from RF importance) ranked by their correlation
             with pseudotime. Smoothed with rolling mean.
  Row 3 (D): Comparison of latent space trajectory smoothness:
             pairwise distance distribution (histogram) and nearest-neighbour
             graph entropy per config with ODE enrichment highlighted.

Usage:
    python benchmarks/plot_ode_trajectory.py
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
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from vcd import detect_all_conflicts
from benchmarks.scripts.plotting.shared import setup_fonts, load_benchmark_npz, export_subpanels, panel_label, add_config_legend_footnote
from mocoo.visualization.style import (
    FIG_WIDTH_IN, FIG_HEIGHT_IN, DPI,
    FS_LABEL, FS_TITLE, FS_AXIS, FS_TICK, FS_LEGEND, FS_SMALL,
    apply_style, get_config_order, get_config_colors, get_short_name, get_tick_name,
)

FS_LEG = FS_LEGEND

setup_fonts()
apply_style()

_CONFIGS = get_config_order()
_CONFIG_COLOR = get_config_colors()
_SCATTER = dict(s=1.2, alpha=0.55, linewidths=0, rasterized=True)


def _load_data(rdir: Path):
    data = load_benchmark_npz(rdir)
    return data["configs"], data["latents"], data["labels"]


def _pca2(latent):
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(latent), pca


def _pseudotime(latent):
    """PC1 as proxy pseudotime (rescaled to [0,1])."""
    pca  = PCA(n_components=1, random_state=42)
    pt   = pca.fit_transform(latent)[:, 0]
    return (pt - pt.min()) / (pt.max() - pt.min() + 1e-9)


def _nn_entropy(latent, k=10):
    """Mean entropy of label distribution in k-NN for each cell."""
    nn   = NearestNeighbors(n_neighbors=k + 1).fit(latent)
    _, idx = nn.kneighbors(latent)
    return float(np.mean([
        -np.sum([p * np.log(p + 1e-9)
                 for p in np.unique(row, return_counts=True)[1] / k])
        for row in idx[:, 1:]
    ]))


# ── Panel A: PCA comparison — VAE vs VAE+ODE ──────────────────────────────

def _draw_pca_comparison(axes, fig, configs, latents, labels):
    cm20  = plt.colormaps.get_cmap("tab20")
    virid = plt.colormaps.get_cmap("plasma")
    show  = ["VAE", "VAE+ODE", "Full"]
    ax_first = None
    for j, cfg in enumerate(show):
        if cfg not in configs:
            continue
        ci  = configs.index(cfg)
        emb, _   = _pca2(latents[ci])
        pt       = _pseudotime(latents[ci])
        uniq     = np.unique(labels[ci])

        # Left = celltype, middle = pseudotime PC1
        ax_ct = axes[0][j]
        if j == 0:
            ax_first = ax_ct

        for k, lb in enumerate(uniq):
            m = labels[ci] == lb
            ax_ct.scatter(emb[m, 0], emb[m, 1], color=cm20(k % 20), **_SCATTER)

        ax_ct.set_xticks([]); ax_ct.set_yticks([])
        ax_ct.set_title(f"{cfg} (Cell type)", fontsize=FS_TITLE, pad=2)
        # No xlabel on top row to avoid overlap with row-below titles
        if j == 0:
            ax_ct.set_ylabel("PC 2", fontsize=FS_AXIS)
        for spine in ax_ct.spines.values():
            spine.set_visible(False)

        ax_pt = axes[1][j]
        sc = ax_pt.scatter(emb[:, 0], emb[:, 1], c=pt,
                           cmap="plasma", vmin=0, vmax=1, **_SCATTER)
        ax_pt.set_xticks([]); ax_pt.set_yticks([])
        ax_pt.set_title(f"{cfg} (Pseudotime)", fontsize=FS_TITLE, pad=2)
        if j == 0:
            ax_pt.set_ylabel("PC 2", fontsize=FS_AXIS)
        for spine in ax_pt.spines.values():
            spine.set_visible(False)
        cax = ax_pt.inset_axes([0.90, 0.05, 0.03, 0.28])
        cb  = fig.colorbar(sc, cax=cax)
        cb.ax.tick_params(labelsize=6, length=1.2, pad=0.4)
        cb.set_label("Pseudotime", fontsize=6, labelpad=1)

    # Shared cell-type legend between Row A and Row B
    uniq = np.unique(labels[0])
    handles = [plt.Line2D([0],[0], marker="o", color="w",
                           markerfacecolor=cm20(k % 20), markersize=3.5)
               for k in range(len(uniq))]
    n_leg_cols = len(uniq)
    fig.legend(handles, [str(lb) for lb in uniq],
               fontsize=max(FS_SMALL - 2, 6), ncol=n_leg_cols, loc="lower center",
               bbox_to_anchor=(0.53, 0.690),
               frameon=False, handletextpad=0.1,
               borderpad=0.1, markerscale=0.8, columnspacing=0.4)
    return ax_first


# ── Panel B: Pseudotime violin per cell type ──────────────────────────────

def _draw_pseudotime_violins(axes, fig, configs, latents, labels):
    """VAE vs VAE+ODE pseudotime distributions per cell type."""
    compare = [("VAE", 0), ("VAE+ODE", 1)]
    ax_first = None
    for j, (cfg, col) in enumerate(compare):
        if cfg not in configs:
            continue
        ci = configs.index(cfg)
        pt  = _pseudotime(latents[ci])
        uniq = np.unique(labels[ci])

        ax = axes[j]
        if j == 0:
            ax_first = ax

        data_per_type = [pt[labels[ci] == lb] for lb in uniq]
        cm20 = plt.colormaps.get_cmap("tab20")

        parts = ax.violinplot(data_per_type, positions=range(len(uniq)),
                               showmedians=True, showextrema=False,
                               widths=0.55)
        for k, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(cm20(k % 20))
            pc.set_alpha(0.65)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(0.8)

        ax.set_xticks(range(len(uniq)))
        ax.set_xticklabels([str(lb) for lb in uniq],
                            fontsize=FS_SMALL, rotation=90, ha="center")
        ax.set_title(f"PT per Type ({cfg})",
                     fontsize=FS_TITLE, pad=1)
        if j == 0:
            ax.set_ylabel("Pseudotime", fontsize=FS_AXIS)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(alpha=0.22, linestyle="--", linewidth=0.4, axis="y")
        ax.set_ylim(0.0, 1.05)
        ax.yaxis.set_major_locator(plt.MaxNLocator(5, prune="both"))
    return ax_first


# ── Panel C: Gene expression along pseudotime ─────────────────────────────

def _draw_gene_pseudotime(ax, fig, configs, latents, labels, adata_path: str):
    """Top 8 marker genes (by ARI correlation with pseudotime)."""
    try:
        import scanpy as sc
        adata = sc.read_h5ad(adata_path)
        if adata.n_obs > 3000:
            sc.pp.subsample(adata, n_obs=3000, random_state=42)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        try:
            sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat_v3",
                                        layer="counts" if "counts" in adata.layers else None)
            adata = adata[:, adata.var.highly_variable]
        except Exception:
            pass
        X_raw = (adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X))
        X_raw = X_raw.astype(np.float32)
        gene_names = np.asarray(adata.var_names)
        has_expr = True
    except Exception as e:
        print(f"  Warning: could not load expression data ({e}), using synthetic proxy.")
        has_expr = False

    # Use Full or VAE+ODE config for pseudotime
    cfg = "Full" if "Full" in configs else "VAE+ODE"
    ci  = configs.index(cfg)
    pt  = _pseudotime(latents[ci])
    n   = len(pt)

    if has_expr:
        X_sub = X_raw[:n]
        # Pearson r of each gene with pseudotime
        pt_c  = pt - pt.mean()
        gene_cors = (X_sub - X_sub.mean(axis=0)) @ pt_c / (
            n * X_sub.std(axis=0, ddof=1).clip(1e-8) * pt_c.std(ddof=1).clip(1e-8)
        )
        # Top positive correlations
        top_idx = np.argsort(gene_cors)[-8:][::-1]
        top_names = gene_names[top_idx]

        order  = np.argsort(pt)
        pt_ord = pt[order]
        cm10   = plt.colormaps.get_cmap("tab10")

        for k, gi in enumerate(top_idx):
            expr   = X_sub[order, gi].astype(float)
            smooth = gaussian_filter1d(expr, sigma=max(1, n // 150))
            ax.plot(pt_ord, smooth, lw=0.8, alpha=0.80,
                    color=cm10(k % 10), label=gene_names[gi])

        ax.set_xlabel("Pseudotime (PC1 of ODE latent)", fontsize=FS_AXIS)
        ax.set_ylabel("Norm. Expr.", fontsize=FS_AXIS)
        ax.set_title("Gene Expr. Along Pseudotime",
                     fontsize=FS_AXIS, pad=2)
        handles, lbls = ax.get_legend_handles_labels()
        fig.legend(handles, lbls, fontsize=FS_SMALL, frameon=False,
                   ncol=len(lbls), loc="upper center",
                   bbox_to_anchor=(0.5, 0.24),
                   handlelength=1.0, labelspacing=0.15,
                   columnspacing=0.6)
    else:
        # Synthetic: show pseudotime distribution as histogram per config
        for i, cfg_name in enumerate(["VAE", "VAE+ODE", "Full"]):
            if cfg_name not in configs:
                continue
            ci2 = configs.index(cfg_name)
            pt2 = _pseudotime(latents[ci2])
            ax.hist(pt2, bins=50, density=True, alpha=0.55,
                    color=_CONFIG_COLOR[cfg_name], label=cfg_name)
        ax.set_xlabel("Pseudotime [0,1]", fontsize=FS_AXIS)
        ax.set_ylabel("Density", fontsize=FS_AXIS)
        ax.set_title("PT Distribution per Config",
                     fontsize=FS_AXIS, pad=2)
        ax.legend(fontsize=FS_LEG, frameon=False, loc="lower left",
                  bbox_to_anchor=(0.0, 1.10), ncol=3)

    ax.tick_params(labelsize=FS_TICK)
    ax.grid(alpha=0.22, linestyle="--", linewidth=0.4)
    ax.set_xlim(0.0, 1.0)  # pseudotime is [0,1], prevent tick overshoot
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(max(0, ylo), yhi)
    from matplotlib.ticker import FixedLocator
    ax.xaxis.set_major_locator(FixedLocator([0, 0.25, 0.5, 0.75, 1.0]))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4, prune="both"))
    return ax


# ── Panel D: Trajectory smoothness — pairwise distances ───────────────────

def _draw_trajectory_smoothness(axes, fig, configs, latents, labels):
    """Histogram of pairwise distances + NN entropy comparison.
    Smoother trajectories = more uniform pairwise distance distribution.
    """
    ax_hist = axes[0]
    ax_ent  = axes[1]

    for i, cfg in enumerate(configs):
        lat = latents[i]
        # Sample 300 cells for pairwise dist
        idx300 = np.random.default_rng(42).choice(len(lat), 300, replace=False)
        sub    = lat[idx300]
        diffs  = sub[:, None, :] - sub[None, :, :]
        dists  = np.linalg.norm(diffs, axis=-1).ravel()
        dists  = dists[dists > 0]

        ax_hist.hist(dists, bins=40, density=True,
                     alpha=0.55, color=_CONFIG_COLOR[cfg], label=cfg,
                     histtype="step", lw=1.1)

    ax_hist.set_xlabel("Pairwise L2 Distance", fontsize=FS_AXIS)
    ax_hist.set_ylabel("Density", fontsize=FS_AXIS)
    ax_hist.set_title("Pairwise Distance Dist.", fontsize=FS_TITLE, pad=2)
    ax_hist.tick_params(labelsize=FS_TICK)
    ax_hist.grid(alpha=0.22, linestyle="--", linewidth=0.4)
    # Enforce axis limits before locator so ticks stay inside borders
    xmin, xmax = ax_hist.get_xlim()
    margin = (xmax - xmin) * 0.04
    ax_hist.set_xlim(xmin - margin, xmax - margin * 3)
    from matplotlib.ticker import MaxNLocator as _MNL
    ax_hist.xaxis.set_major_locator(_MNL(4, prune="both"))

    # NN entropy per config
    entropies = []
    for i, cfg in enumerate(configs):
        ent = _nn_entropy(latents[i], k=10)
        entropies.append(ent)

    x = np.arange(len(configs))
    bars = ax_ent.bar(x, entropies, color=[_CONFIG_COLOR[c] for c in configs],
                      alpha=0.80, edgecolor="black", linewidth=0.4)
    # Highlight ODE configs
    for k, cfg in enumerate(configs):
        if "ODE" in cfg:
            bars[k].set_edgecolor("#DD8452")
            bars[k].set_linewidth(1.3)
    ax_ent.set_xticks(x)
    ax_ent.set_xticklabels([get_tick_name(c) for c in configs],
                            fontsize=FS_SMALL, rotation=0, ha="center")
    ax_ent.set_xlim(-0.5, len(configs) - 0.5)
    ax_ent.set_ylabel("kNN entropy", fontsize=FS_AXIS)
    ax_ent.set_title("kNN Entropy", fontsize=FS_TITLE, pad=1)
    ax_ent.tick_params(labelsize=FS_TICK)
    ax_ent.grid(alpha=0.22, linestyle="--", linewidth=0.4, axis="y")
    ax_ent.yaxis.set_major_locator(plt.MaxNLocator(4, prune="both"))

    return ax_hist


# ── Main ───────────────────────────────────────────────────────────────────

def build_figure(rdir: Path, outdir: Path, adata_path: str):
    configs, latents, labels = _load_data(rdir)

    fig = plt.figure(figsize=(FIG_WIDTH_IN * 1.10, FIG_HEIGHT_IN * 0.90), dpi=DPI)

    n_cfgs = 3  # VAE, VAE+ODE, Full

    # A: 2-row × n_cfgs-col PCA grid — shifted down to leave legend band at top
    _a_cw = (0.88 - 0.05 * (n_cfgs - 1)) / n_cfgs
    _a_rh = 0.10
    _a_top = 0.96
    grid_A = [
        [fig.add_axes([0.08 + c * (_a_cw + 0.05),
                       _a_top - (r + 1) * _a_rh - r * 0.03,
                       _a_cw, _a_rh])
         for c in range(n_cfgs)]
        for r in range(2)
    ]

    # B: 2 violin plots — wider gap from A
    _b_aw = (0.88 - 0.06) / 2
    axes_B = [
        fig.add_axes([0.08, 0.50, _b_aw, 0.14]),
        fig.add_axes([0.08 + _b_aw + 0.06, 0.50, _b_aw, 0.14]),
    ]

    # C: single wide gene expression panel
    ax_C_single = fig.add_axes([0.08, 0.28, 0.88, 0.14])

    # D: 2 panels — pairwise dist histogram + NN entropy bars
    _d_aw = (0.88 - 0.10) / 2
    axes_D = [
        fig.add_axes([0.08, 0.07, _d_aw, 0.12]),
        fig.add_axes([0.08 + _d_aw + 0.10, 0.07, _d_aw, 0.12]),
    ]

    print("  Drawing Panel A (PCA pseudotime)...")
    ax_A = _draw_pca_comparison(grid_A, fig, configs, latents, labels)

    print("  Drawing Panel B (Pseudotime violins)...")
    ax_B = _draw_pseudotime_violins(axes_B, fig, configs, latents, labels)

    print("  Drawing Panel C (Gene expression along pseudotime)...")
    ax_C = _draw_gene_pseudotime(ax_C_single, fig, configs, latents, labels, adata_path)

    print("  Drawing Panel D (Trajectory smoothness)...")
    ax_D = _draw_trajectory_smoothness(axes_D, fig, configs, latents, labels)
    panel_label(fig, ax_A, "A", x_off=-0.08, y_off=0.010)
    panel_label(fig, ax_B, "B", x_off=-0.08, y_off=0.030)
    panel_label(fig, ax_C, "C", x_off=-0.08, y_off=0.035)
    panel_label(fig, ax_D, "D", x_off=-0.08, y_off=0.010)

    fig.canvas.draw()
    print("\n── Conflict Detection ──")
    issues = detect_all_conflicts(fig, label="ode_trajectory", verbose=True)

    outpath = outdir / "supp_ode_trajectory.png"
    from mocoo.visualization.style import save_figure
    save_figure(fig, outpath)

    # Export individual panel sub-figures
    sub_dir = outdir / "supp_ode_trajectory"
    sub_dir.mkdir(parents=True, exist_ok=True)
    export_subpanels(fig, sub_dir, [(ax_A, "panelA_pca"),
                                     (ax_B, "panelB_pseudotime_violin"),
                                     (ax_C, "panelC_gene_expression"),
                                     (ax_D, "panelD_smoothness")])
    plt.close(fig)

    n_warn = sum(1 for x in issues if x.get("severity") == "warning")
    n_err  = sum(1 for x in issues if x.get("severity") == "error")
    print(f"\nSaved -> {outpath}")
    print(f"{n_warn} warnings | {n_err} errors")
    return issues


def main():
    _benchmarks = Path(__file__).resolve().parent.parent.parent  # benchmarks/
    _data_base = os.environ.get("MOCOO_DATA_DIR", "data")
    p = argparse.ArgumentParser()
    p.add_argument("--resultsdir",
                   default=str(_benchmarks / "results" / "single_dataset"))
    p.add_argument("--outdir",
                   default=str(_benchmarks / "figures"))
    p.add_argument("--data",
                   default=os.path.join(_data_base, "Desktop/datasets/IRALL.h5ad"))
    args = p.parse_args()
    rdir   = Path(args.resultsdir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    return build_figure(rdir, outdir, args.data)


if __name__ == "__main__":
    main()
