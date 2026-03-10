#!/usr/bin/env python
"""
Biological validation composed figure for MoCoO latents - v5
(ABCD panels, 17 cm x 21 cm, all 6 configs).

   (A) Perturbation strategy    [left column, row 0]
       A-top : Global perturbation robustness — all 6 configs, line + CI
       A-bot : Per-component sensitivity (Full) — Permutation importance bar chart
   (B) Latent UMAP projections  [right block, row 0]
       B0  : UMAP cell-type; B1-B4 : top-4 component-intensity UMAPs
   (C) Gene expression UMAPs    [full width, row 1]
       C0  : UMAP cell-type (ref.); C1-C4 : top-gene expression per comp (RF importance)
   (D) Per-config Pearson heatmaps  [full width, row 2, 2x3 grid]
       All 6 configs shown. Each config selects its OWN top components and genes.

Figure <= 17 cm x 21 cm; aspect >= 17:21.
Text sizes maximised under the 13-pass visual-conflict detector.

Usage:
    python -m benchmarks.plot_biological_validation
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from benchmarks.scripts.pipeline.visual_conflict_detector import detect_all_conflicts
from benchmarks.scripts.plotting.shared import setup_fonts, load_benchmark_npz, export_subpanels, panel_label
from mocoo.visualization.style import (
    FIG_WIDTH_IN, FIG_HEIGHT_IN, DPI,
    FS_LABEL, FS_TITLE, FS_AXIS, FS_TICK, FS_LEGEND, FS_SMALL,
    apply_style, get_config_colors,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

setup_fonts()
apply_style()

# ── Style constants (imported from mocoo.visualization.style) ─────────────────
FS_LEG = FS_LEGEND

# ── Scientific constants ──────────────────────────────────────────────────────
TOP_COMPS      = 6    # latent dims shown in heatmaps (per config)
UMAP_COMPS     = 4    # subset shown as intensity / gene-expr UMAPs
GENES_PER_COMP = 3    # top genes per component in heatmap
NOISE_SCALES   = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0], dtype=float)
HEATMAP_CONFIGS = [
    "VAE", "VAE+ODE", "VAE+MoCo",
    "VAE+MoCo+Proto", "VAE+ODE+MoCo", "Full",
]
_SCATTER_KW    = dict(s=0.8, alpha=0.45, linewidths=0, rasterized=True)


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def _load_benchmark(rdir: Path):
    return load_benchmark_npz(rdir)


def _load_expression(path: str, max_cells: int = 3000, hvg: int = 3000):
    adata = sc.read_h5ad(path)
    if adata.n_obs > max_cells:
        sc.pp.subsample(adata, n_obs=max_cells, random_state=42)
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=hvg,
                                    flavor="seurat_v3", layer="counts")
    except Exception:
        sc.pp.highly_variable_genes(adata, n_top_genes=hvg)
    adata = adata[:, adata.var["highly_variable"]].copy()
    return adata


# ══════════════════════════════════════════════════════════════════════════════
# Computation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _compute_umap(latent: np.ndarray, cache_path: Path | None = None) -> np.ndarray:
    if cache_path is not None and cache_path.exists():
        print(f"  Loading UMAP cache: {cache_path}")
        return np.load(cache_path)["emb"]
    import umap as _umap
    reducer = _umap.UMAP(n_components=2, random_state=42,
                         n_neighbors=30, min_dist=0.3, metric="euclidean")
    emb = reducer.fit_transform(latent).astype(np.float32)
    if cache_path is not None:
        np.savez_compressed(cache_path, emb=emb)
        print(f"  UMAP cached -> {cache_path}")
    return emb


def _numpy_gene_latent_corr(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Pure-NumPy Pearson r. X:(cells,genes), Z:(cells,latent) -> (genes,latent)."""
    X = np.asarray(X, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    X -= X.mean(axis=0)
    Z -= Z.mean(axis=0)
    X /= X.std(axis=0, ddof=1) + 1e-8
    Z /= Z.std(axis=0, ddof=1) + 1e-8
    return (X.T @ Z) / (X.shape[0] - 1)


def _select_top_components(corr: np.ndarray, n: int) -> np.ndarray:
    # Use max positive correlation instead of absolute mean
    scores = np.maximum(corr, 0).max(axis=0)
    return np.argsort(scores)[-n:][::-1]


def _select_genes_per_comp(corr, comp_indices, gene_names, k):
    gene_idx_table, gene_name_table = [], []
    for ci in comp_indices:
        # Use positive correlation only
        top = np.argsort(corr[:, ci])[-k:][::-1]
        gene_idx_table.append(top)
        gene_name_table.append(gene_names[top])
    return gene_idx_table, gene_name_table


def _global_perturbation(latent, labels, noise_scales=NOISE_SCALES,
                          repeats: int = 6, seed: int = 42):
    """Perturb ALL dims simultaneously — measures overall latent robustness."""
    rng = np.random.default_rng(seed)
    latent = np.asarray(latent, dtype=np.float32)
    labels = np.asarray(labels)
    n = latent.shape[0]
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(latent)
    feat_std = latent.std(axis=0, ddof=1) + 1e-8
    means, stds = [], []
    for s in noise_scales:
        vals = []
        for _ in range(repeats):
            noise = (rng.normal(0.0, s, latent.shape).astype(np.float32) * feat_std)
            z_p = latent + noise
            _, ind = nn.kneighbors(z_p)
            nbr = ind[:, 0]
            mask = nbr == np.arange(n)
            nbr[mask] = ind[mask, 1]
            vals.append((labels[nbr] == labels).mean())
        means.append(np.mean(vals))
        stds.append(np.std(vals))
    return np.asarray(means, dtype=np.float32), np.asarray(stds, dtype=np.float32)


def _permutation_importance(latent, labels, seed=42):
    """Permute ONE dimension at a time -> drop in kNN retention."""
    rng = np.random.default_rng(seed)
    latent = np.asarray(latent, dtype=np.float32)
    labels = np.asarray(labels)
    n = latent.shape[0]
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(latent)
    
    # Baseline
    _, ind = nn.kneighbors(latent)
    nbr = ind[:, 0]
    mask = nbr == np.arange(n)
    nbr[mask] = ind[mask, 1]
    baseline_acc = (labels[nbr] == labels).mean()
    
    n_comps = latent.shape[1]
    drops = np.zeros(n_comps, dtype=np.float32)
    
    for ci in range(n_comps):
        z_p = latent.copy()
        rng.shuffle(z_p[:, ci])
        _, ind = nn.kneighbors(z_p)
        nbr = ind[:, 0]
        mask = nbr == np.arange(n)
        nbr[mask] = ind[mask, 1]
        acc = (labels[nbr] == labels).mean()
        drops[ci] = baseline_acc - acc
        
    return drops


def _get_rf_top_genes(X, Z, comp_indices, gene_names, top_k=1):
    """Use Random Forest feature importance to find top genes for components."""
    gene_idx_table, gene_name_table = [], []
    for ci in comp_indices:
        rf = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=42, n_jobs=-1)
        rf.fit(X, Z[:, ci])
        top = np.argsort(rf.feature_importances_)[-top_k:][::-1]
        gene_idx_table.append(top)
        gene_name_table.append(gene_names[top])
    return gene_idx_table, gene_name_table


# ══════════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ══════════════════════════════════════════════════════════════════════════════

def _umap_lims(emb: np.ndarray, pad: float = 0.05):
    xlo, xhi = emb[:, 0].min(), emb[:, 0].max()
    ylo, yhi = emb[:, 1].min(), emb[:, 1].max()
    dx, dy = (xhi - xlo) * pad, (yhi - ylo) * pad
    return xlo - dx, xhi + dx, ylo - dy, yhi + dy


def _inset_cbar(fig, ax, mappable, label: str = ""):
    """Tiny colorbar inside bottom-right of axes (avoids cross-panel leakage)."""
    cax = ax.inset_axes([0.78, 0.04, 0.03, 0.26])
    cb = fig.colorbar(mappable, cax=cax)
    cb.ax.tick_params(labelsize=FS_SMALL, length=1.5, pad=0.5)
    if label:
        cb.ax.set_ylabel(label, fontsize=FS_SMALL, labelpad=1)
    return cb


def _umap_celltype(ax, emb, labels, title, show_ylabel=True):
    uniq = np.unique(labels)
    cm20 = plt.colormaps.get_cmap("tab20")
    for k, lb in enumerate(uniq):
        m = labels == lb
        ax.scatter(emb[m, 0], emb[m, 1],
                   color=cm20(k % 20), clip_on=False, **_SCATTER_KW)
    xl, xh, yl, yh = _umap_lims(emb)
    ax.set_xlim(xl, xh); ax.set_ylim(yl, yh)
    ax.set_title(title, fontsize=FS_TITLE, pad=3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP 1", fontsize=FS_AXIS)
    if show_ylabel:
        ax.set_ylabel("UMAP 2", fontsize=FS_AXIS)
    if len(uniq) <= 12:
        handles = [plt.Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=cm20(k % 20), markersize=3)
                   for k in range(len(uniq))]
        ax.legend(handles, [str(lb) for lb in uniq],
                  fontsize=FS_LEG, ncol=2, loc="upper right",
                  framealpha=0.65, handletextpad=0.15,
                  borderpad=0.2, markerscale=0.7,
                  columnspacing=0.5)


def _umap_scalar(ax, emb, values, title, cmap_name, cbar_label, fig, show_ylabel=True):
    v5, v95 = np.percentile(values, 5), np.percentile(values, 95)
    if v95 - v5 < 1e-6:
        v5, v95 = values.min(), values.max() + 1e-6
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=values,
                    cmap=cmap_name, vmin=v5, vmax=v95,
                    clip_on=False, **_SCATTER_KW)
    xl, xh, yl, yh = _umap_lims(emb)
    ax.set_xlim(xl, xh); ax.set_ylim(yl, yh)
    ax.set_title(title, fontsize=FS_TITLE, pad=3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP 1", fontsize=FS_AXIS)
    if show_ylabel:
        ax.set_ylabel("UMAP 2", fontsize=FS_AXIS)
    _inset_cbar(fig, ax, sc, label=cbar_label)



# ══════════════════════════════════════════════════════════════════════════════
# Panel drawing functions
# ══════════════════════════════════════════════════════════════════════════════

def _draw_panel_A(gs_A, fig, configs, latents, labels_all,
                  full_idx, perm_drops):
    """Panel A — two stacked plots: global robustness + per-comp sensitivity."""

    # A-top: global perturbation line chart
    ax_a1 = fig.add_subplot(gs_A[0])
    cm10 = plt.colormaps.get_cmap("tab10")
    for i, cfg in enumerate(configs):
        mu, sd = _global_perturbation(latents[i], labels_all[i],
                                       noise_scales=NOISE_SCALES,
                                       repeats=5, seed=42 + i)
        ax_a1.plot(NOISE_SCALES, mu, lw=1.2, color=cm10(i % 10), label=cfg)
        ax_a1.fill_between(NOISE_SCALES,
                           np.clip(mu - sd, 0, 1), np.clip(mu + sd, 0, 1),
                           color=cm10(i % 10), alpha=0.12)
    ax_a1.set_title("Robustness to Global Noise\n(Higher is better)",
                    fontsize=FS_TITLE, pad=3)
    ax_a1.set_xlabel("Noise Scale ($\\sigma$)", fontsize=FS_AXIS)
    ax_a1.set_ylabel("kNN Accuracy Retention", fontsize=FS_AXIS)
    ax_a1.set_ylim(0.0, 1.06)
    ax_a1.set_yticks(np.linspace(0.0, 1.0, 6))
    ax_a1.tick_params(labelsize=FS_TICK)
    ax_a1.grid(alpha=0.22, linestyle="--", linewidth=0.4)
    ax_a1.legend(fontsize=FS_LEG, frameon=False, ncol=2,
                 loc="upper right",
                 handlelength=1.0, labelspacing=0.2, columnspacing=0.6)

    # A-bot: per-component sensitivity bar chart (Full config)
    ax_a2 = fig.add_subplot(gs_A[1])
    
    # Plot top 10 components by permutation drop
    top_10_idx = np.argsort(perm_drops)[-10:][::-1]
    top_10_drops = perm_drops[top_10_idx]
    
    x_pos = np.arange(10)
    ax_a2.bar(x_pos, top_10_drops, color=get_config_colors()["Full"], edgecolor="black", linewidth=0.5)
    ax_a2.set_xticks(x_pos)
    ax_a2.set_xticklabels([f"Z{ci+1}" for ci in top_10_idx], fontsize=FS_TICK, rotation=45, ha="right")
    ax_a2.set_title("Latent Dimension Importance\n(Higher drop = more important)",
                    fontsize=FS_TITLE, pad=2)
    ax_a2.set_xlabel("Component", fontsize=FS_AXIS)
    ax_a2.set_ylabel("Accuracy Drop ($\\Delta$)", fontsize=FS_AXIS)
    ax_a2.tick_params(labelsize=FS_TICK)
    ax_a2.grid(alpha=0.22, linestyle="--", linewidth=0.4, axis="y")
    
    # Add explanatory text
    ax_a2.text(0.95, 0.95, "Drop in kNN accuracy\nwhen a single dimension\nis permuted",
               transform=ax_a2.transAxes, fontsize=FS_SMALL,
               va="top", ha="right", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8, lw=0.5))

    return ax_a1  # return topmost axes for panel-label placement


def _draw_panel_B(gs_B, fig, emb, Z_full, labels_f, n_cells,
                  umap_comp_indices):
    """Panel B — UMAP: cell-type + TOP_COMPS component-intensity UMAPs."""
    ax_b0 = fig.add_subplot(gs_B[0])
    _umap_celltype(ax_b0, emb, labels_f,
                   title="UMAP\nCell type (Full)", show_ylabel=True)
    for j, ci in enumerate(umap_comp_indices):
        ax_bj = fig.add_subplot(gs_B[j + 1])
        _umap_scalar(ax_bj, emb, Z_full[:n_cells, ci],
                     title=f"Z{ci+1} intensity",
                     cmap_name="plasma",
                     cbar_label=f"Z{ci+1}",
                     fig=fig, show_ylabel=False)
    return ax_b0


def _draw_panel_C(gs_C, fig, emb, X_raw, labels_f,
                  umap_comp_indices, rf_gene_idx_table, rf_gene_name_table):
    """Panel C — UMAP: cell-type reference + top-gene expression UMAPs (RF importance)."""
    ax_c0 = fig.add_subplot(gs_C[0])
    _umap_celltype(ax_c0, emb, labels_f,
                   title="UMAP\nCell type (ref.)", show_ylabel=True)
    for j, ci in enumerate(umap_comp_indices):
        ax_cj = fig.add_subplot(gs_C[j + 1])
        g_idx = rf_gene_idx_table[j][0]
        tg    = rf_gene_name_table[j][0]
        _umap_scalar(ax_cj, emb, X_raw[:, g_idx],
                     title=f"{tg}\n(Z{ci+1} RF top gene)",
                     cmap_name="YlOrRd",
                     cbar_label="log1p",
                     fig=fig, show_ylabel=False)
    return ax_c0


def _draw_panel_D(gs_D, fig, configs, latents, X_raw, n_cells, gene_names):
    """Panel D — per-config component-grouped Pearson heatmaps (2x3 grid)."""
    k = GENES_PER_COMP
    ax_d0 = None
    
    for j, cfg in enumerate(HEATMAP_CONFIGS):
        r, c = divmod(j, 3)
        ax_dj = fig.add_subplot(gs_D[r, c])
        if j == 0:
            ax_d0 = ax_dj
            
        # Compute correlation and select top components/genes FOR THIS CONFIG
        cfg_idx = configs.index(cfg)
        Z_cfg = latents[cfg_idx][:n_cells]
        corr_mat = _numpy_gene_latent_corr(X_raw, Z_cfg)
        
        comp_indices = _select_top_components(corr_mat, n=TOP_COMPS)
        gene_idx_table, gene_name_table = _select_genes_per_comp(
            corr_mat, comp_indices, gene_names, k=k)
            
        ordered_gene_idx = np.concatenate(gene_idx_table)
        mat = corr_mat[ordered_gene_idx, :][:, comp_indices]

        vmax = float(np.percentile(np.abs(mat), 99))
        vmax = max(vmax, 0.15)

        im = ax_dj.imshow(mat, aspect="auto", cmap="RdBu_r",
                          vmin=-vmax, vmax=vmax, interpolation="nearest")
        # X: component labels
        ax_dj.set_xticks(np.arange(TOP_COMPS))
        ax_dj.set_xticklabels([f"Z{ci+1}" for ci in comp_indices],
                               fontsize=FS_TICK, rotation=45, ha="right")

        # Y: one label per component group (top gene of each group)
        ytick_pos = np.arange(TOP_COMPS) * k
        ytick_lbl = [gnames[0] for gnames in gene_name_table]
        ax_dj.set_yticks(ytick_pos)
        ax_dj.set_yticklabels(ytick_lbl, fontsize=FS_TICK)
        if c == 0:
            ax_dj.set_ylabel("Top gene / group", fontsize=FS_AXIS)
        else:
            ax_dj.set_ylabel("")

        # White dividers between groups
        for gi in range(1, TOP_COMPS):
            ax_dj.axhline(gi * k - 0.5, color="white", lw=1.0)

        ax_dj.set_title(cfg, fontsize=FS_TITLE, pad=3)

        # Inset colorbar (right edge inside the axes)
        cax = ax_dj.inset_axes([1.03, 0.08, 0.035, 0.84])
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=FS_TICK, length=1.5)
        cb.set_label("r", fontsize=FS_AXIS, labelpad=2)

    return ax_d0


# ══════════════════════════════════════════════════════════════════════════════
# Main figure builder
# ══════════════════════════════════════════════════════════════════════════════

def build_figure(data, adata, outpath: Path):
    configs    = data["configs"]
    latents    = data["latents"]
    labels_all = data["labels"]

    # ── 1. Full config index ───────────────────────────────────────────────
    full_idx = configs.index("Full") if "Full" in configs else len(configs) - 1
    Z_full   = latents[full_idx]
    labels_f = labels_all[full_idx]

    # ── 2. UMAP (cached) ──────────────────────────────────────────────────
    # Cache stored alongside benchmark data in results/ (not figures/)
    results_dir = outpath.parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    cache = results_dir / "bv_umap_cache.npz"
    print("Computing / loading UMAP ...")
    emb     = _compute_umap(Z_full, cache_path=cache)
    n_cells = emb.shape[0]

    # ── 3. Expression matrix (aligned to latent cell count) ───────────────
    X_raw = (adata.X.toarray() if hasattr(adata.X, "toarray")
             else np.asarray(adata.X))
    X_raw      = X_raw[:n_cells].astype(np.float32)
    gene_names = np.asarray(adata.var_names)

    # ── 4. Permutation Importance (Panel A & B) ───────────────────────────
    print("Computing Permutation Importance for Full config ...")
    perm_drops = _permutation_importance(Z_full, labels_f, seed=42)
    umap_comp_indices = np.argsort(perm_drops)[-UMAP_COMPS:][::-1]
    print(f"  Top {UMAP_COMPS} components by permutation: {['Z'+str(c+1) for c in umap_comp_indices]}")

    # ── 5. RF Gene Importance (Panel C) ───────────────────────────────────
    print("Computing RF Gene Importance for top components ...")
    rf_gene_idx_table, rf_gene_name_table = _get_rf_top_genes(
        X_raw, Z_full[:n_cells], umap_comp_indices, gene_names, top_k=1)

    # ── 6. Figure skeleton ────────────────────────────────────────────────
    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=DPI)

    outer = gridspec.GridSpec(
        4, 1,
        height_ratios=[2.5, 2.5, 2.5, 5.5],
        hspace=0.28,
        figure=fig,
    )
    # Row 0: Panel A (2 columns)
    gs_A = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[0], wspace=0.20)

    # Row 1: Panel B (full width)
    gs_B = gridspec.GridSpecFromSubplotSpec(
        1, UMAP_COMPS + 1, subplot_spec=outer[1], wspace=0.12)

    # Row 2: Panel C (full width)
    gs_C = gridspec.GridSpecFromSubplotSpec(
        1, UMAP_COMPS + 1, subplot_spec=outer[2], wspace=0.12)

    # Row 3: Panel D — 2 rows x 3 cols (all 6 configs)
    gs_D = gridspec.GridSpecFromSubplotSpec(
        2, 3, subplot_spec=outer[3], wspace=0.58, hspace=0.34)

    # ── 7. Draw panels ────────────────────────────────────────────────────
    print("  Drawing Panel A ...")
    ax_A = _draw_panel_A(gs_A, fig, configs, latents, labels_all,
                         full_idx, perm_drops)
    print("  Drawing Panel B ...")
    ax_B = _draw_panel_B(gs_B, fig, emb, Z_full, labels_f, n_cells,
                         umap_comp_indices)
    print("  Drawing Panel C ...")
    ax_C = _draw_panel_C(gs_C, fig, emb, X_raw, labels_f,
                         umap_comp_indices, rf_gene_idx_table, rf_gene_name_table)
    print("  Drawing Panel D ...")
    ax_D = _draw_panel_D(gs_D, fig, configs, latents, X_raw, n_cells, gene_names)

    # ── 8. Global layout before placing panel letters ─────────────────────
    fig.subplots_adjust(left=0.10, right=0.96, top=0.97, bottom=0.03)

    # ── 9. Panel letters — placed AFTER subplots_adjust fixes positions ───
    panel_label(fig, ax_A, "A", x_off=-0.018)
    panel_label(fig, ax_B, "B", x_off=-0.018)
    panel_label(fig, ax_C, "C", x_off=-0.018)
    panel_label(fig, ax_D, "D", x_off=-0.018)

    # ── 10. Conflict detection (all 13 passes) ────────────────────────────
    print("\n── Conflict Detection ──")
    fig.canvas.draw()
    issues = detect_all_conflicts(fig, label="bio_validation_abcd", verbose=True)

    from mocoo.visualization.style import save_figure
    save_figure(fig, outpath)

    # Export individual panel sub-figures
    sub_dir = outpath.parent / "fig3_biological_validation"
    sub_dir.mkdir(parents=True, exist_ok=True)
    export_subpanels(fig, sub_dir, [(ax_A, "panelA_permutation_importance"),
                                     (ax_B, "panelB_umap_components"),
                                     (ax_C, "panelC_gene_importance"),
                                     (ax_D, "panelD_all_configs_umap")])
    plt.close(fig)

    n_warn = sum(1 for x in issues if x.get("severity") == "warning")
    n_err  = sum(1 for x in issues if x.get("severity") == "error")
    status = "✓ OK" if (n_warn + n_err) == 0 else "✗ ISSUES"
    print(f"\nSaved -> {outpath}")
    print(f"{n_warn} warnings | {n_err} errors | {status}")
    return issues


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    _benchmarks = Path(__file__).resolve().parent.parent.parent  # benchmarks/
    _data_base = os.environ.get("MOCOO_DATA_DIR", "data")
    parser = argparse.ArgumentParser()
    parser.add_argument("--resultsdir",
                        default=str(_benchmarks / "results" / "single_dataset"))
    parser.add_argument("--data",
                        default=os.path.join(_data_base, "LAB/scRL/IRALL.h5ad"))
    parser.add_argument("--outdir",
                        default=str(_benchmarks / "figures"))
    args = parser.parse_args()

    rdir   = Path(args.resultsdir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading benchmark data ...")
    data = _load_benchmark(rdir)
    print("Loading / preprocessing expression data ...")
    adata = _load_expression(args.data, max_cells=3000, hvg=3000)
    print("Building figure ...")
    build_figure(data, adata, outdir / "supp_biological_validation.png")


if __name__ == "__main__":
    main()
