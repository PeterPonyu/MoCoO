#!/usr/bin/env python
"""
MoCoO Figure 2 — Quantitative Latent Space Comparison
======================================================
Layout (17 × 21 cm):
  Row 0 (A): UMAP grid — all 6 configs side-by-side, 3 per row, coloured by
             cell type. Shows visual clustering quality improvement.
  Row 1 (B): Three grouped bar charts — ARI, NMI, ASW — across all 6 configs,
             with test-set bars overlaid as pattern-filled bars.
  Row 2 (C): Neighbourhood quality metrics (DREX trustworthiness / continuity /
             Spearman) and DRE overall quality per config.
  Row 3 (D): Latent-space structure: participation ratio, manifold
             dimensionality, anisotropy — shows ODE compresses the latent
             geometry without sacrificing diversity.

Usage:
    python benchmarks/plot_quant_comparison.py
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from umap import UMAP

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from vcd import detect_all_conflicts

# ── Import centralized style ────────────────────────────────────────────────
from mocoo.visualization.style import (
    FIG_WIDTH_IN as FIG_W, FIG_HEIGHT_IN as FIG_H, DPI,
    FS_LABEL, FS_TITLE, FS_AXIS, FS_TICK, FS_LEGEND as FS_LEG, FS_SMALL,
    get_config_colors, get_config_order, get_short_name, apply_style,
    get_tick_name, get_legend_name, metric_title, FMT_SCORE_SHORT,

)
from benchmarks.scripts.plotting.shared import (
    setup_fonts, unify_metric_keys, load_benchmark_npz,
    load_config_metrics, export_subpanels, panel_label,
    add_config_legend_footnote, add_metric_footnote, load_multiseed_stats,
)

apply_style()

# ── Fonts ──────────────────────────────────────────────────────────────────
setup_fonts()

# ── Style constants from centralized module ──────────────────────────────────
_CONFIGS = get_config_order()
_CONFIG_COLOR = get_config_colors()
_SCATTER = dict(s=1.2, alpha=0.60, linewidths=0, rasterized=True)
_XSHORT = {c: get_short_name(c) for c in _CONFIGS}

# ── Data loading ───────────────────────────────────────────────────────────
def _load_data(rdir: Path):
    data = load_benchmark_npz(rdir)
    configs, latents, labels = data["configs"], data["latents"], data["labels"]
    metrics = load_config_metrics(rdir, configs)
    return configs, latents, labels, metrics


def _compute_umap(latent, cache_dir: Path, tag: str):
    cache = cache_dir / f"qc_umap_{tag}.npz"
    if cache.exists():
        return np.load(cache)["emb"]
    emb = UMAP(n_components=2, random_state=42, min_dist=0.3,
               n_neighbors=30, verbose=False).fit_transform(latent)
    np.savez_compressed(cache, emb=emb)
    return emb



def _highlight_best(ax, bars, vals, higher_better=True):
    """Highlight the best-performing bar with a bold edge."""
    best_i = int(np.argmax(vals)) if higher_better else int(np.argmin(vals))
    bars[best_i].set_edgecolor("crimson")
    bars[best_i].set_linewidth(1.4)


# ── Panel drawing ──────────────────────────────────────────────────────────

def _draw_umap_grid(axes_grid, fig, configs, latents, labels, cache_dir):
    """2-row × 3-col UMAP grid coloured by cell type."""
    cm20 = plt.colormaps.get_cmap("tab20")
    ax_first = None
    for j, cfg in enumerate(configs):
        r, c = divmod(j, 3)
        ax = axes_grid[r][c]
        if j == 0:
            ax_first = ax
        emb    = _compute_umap(latents[j], cache_dir, cfg.replace("+", "_"))
        uniq   = np.unique(labels[j])
        for k, lb in enumerate(uniq):
            m = labels[j] == lb
            ax.scatter(emb[m, 0], emb[m, 1], color=cm20(k % 20), **_SCATTER)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(cfg, fontsize=FS_TITLE, pad=2)
        for spine in ax.spines.values():
            spine.set_visible(False)
    # Shared cell-type legend as a full-width strip between rows A and B
    uniq = np.unique(labels[0])
    handles = [plt.Line2D([0],[0], marker="o", color="w",
                           markerfacecolor=cm20(k % 20), markersize=3)
               for k in range(len(uniq))]
    n_cols = min(max(6, len(uniq) // 2), 10)
    fig.legend(handles, [str(lb) for lb in uniq],
               fontsize=max(FS_SMALL - 2, 4), ncol=len(uniq), loc="lower center",
               bbox_to_anchor=(0.50, 0.675),
               frameon=False, handletextpad=0.15,
               borderpad=0.15, markerscale=1.0, columnspacing=0.6)
    return ax_first


def _draw_clustering_bars(axes_list, fig, configs, metrics, multiseed_stats=None):
    """Grouped bar charts: ARI, NMI, ASW (val + test)."""
    metric_pairs = [
        ("ARI",      "test_ARI",  "ARI",  True),
        ("NMI",      "test_NMI",  "NMI",  True),
        ("ASW",      "test_ASW",  "ASW",  True),
    ]
    x = np.arange(len(configs))
    w = 0.38
    ax_first = None
    for j, (vkey, tkey, label, higher_better) in enumerate(metric_pairs):
        ax = axes_list[j]
        if j == 0:
            ax_first = ax
        vals  = [metrics[c].get(vkey, 0) for c in configs]
        tvals = [metrics[c].get(tkey, 0) for c in configs]
        colors = [_CONFIG_COLOR[c] for c in configs]
        bars1 = ax.bar(x - w/2, vals,  w, color=colors, alpha=0.85,
                       edgecolor="black", linewidth=0.4, label="Val")
        if multiseed_stats:
            yerr = [multiseed_stats.get(c, {}).get(vkey, (0, 0))[1] for c in configs]
            ax.errorbar(x - w/2, vals, yerr=yerr, fmt="none",
                        ecolor="black", capsize=2.5, capthick=0.7, elinewidth=0.7, zorder=5)
        bars2 = ax.bar(x + w/2, tvals, w, color=colors, alpha=0.4,
                       edgecolor="black", linewidth=0.4, hatch="//",
                       label="Test")
        # Highlight best config
        _highlight_best(ax, bars1, vals, higher_better)
        ax.set_xticks(x)
        short = [_XSHORT[c] for c in configs]
        ax.set_xticklabels(short, fontsize=max(FS_SMALL - 1, 6), rotation=45, ha="right")
        ax.set_xlim(-0.5, len(configs) - 0.5)
        ax.set_title(f"{label} {'↑' if higher_better else '↓'}",
                     fontsize=FS_TITLE, pad=1)
        if j == 0:
            ax.set_ylabel("Score", fontsize=FS_AXIS)
        ax.tick_params(axis="x", labelsize=max(FS_SMALL - 1, 6))
        ax.tick_params(axis="y", labelsize=FS_TICK)
        ax.grid(alpha=0.22, linestyle="--", linewidth=0.4, axis="y")
        ax.set_ylim(0, max(max(vals), max(tvals)) * 1.25)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4, prune="both"))
        # Val/Test indicator inside the first bar panel
        if j == 0:
            ax.text(0.98, 0.95, "\u25a0 Val  \u25cb Test",
                    transform=ax.transAxes,
                    fontsize=FS_SMALL, va="top", ha="right")
    return ax_first


def _draw_neighbourhood_quality(axes_list, fig, configs, metrics):
    """DREX: trustworthiness, continuity, distance Spearman + DRE overall."""
    items = [
        ("DREX_trustworthiness",       "Trustworthiness \u2191", True),
        ("DREX_neighborhood_symmetry", "Nbr. Symmetry \u2191",  True),
        ("DREX_distance_spearman",     "Dist. Spearman \u2191",  True),
        ("DRE_umap_overall_quality",   "DRE Quality \u2191",     True),
    ]
    x = np.arange(len(configs))
    colors = [_CONFIG_COLOR[c] for c in configs]
    short = [_XSHORT[c] for c in configs]
    ax_first = None
    for j, (key, label, higher_better) in enumerate(items):
        ax = axes_list[j]
        if j == 0:
            ax_first = ax
        vals = [metrics[c].get(key, 0) for c in configs]
        # Compute ylim BEFORE drawing bars to avoid patch_truncation
        valid_vals = [v for v in vals if v > 0 and np.isfinite(v)]
        ylo = 0.0
        if valid_vals:
            vmin, vmax = min(valid_vals), max(valid_vals)
            val_range = vmax - vmin
            if val_range < 0.15 * vmax and vmin > 0.3:
                ylo = max(0, vmin - val_range * 1.5)
                yhi = vmax + val_range * 1.5
            else:
                yhi = vmax * 1.18
        else:
            yhi = 1.0
        bars = ax.bar(x, [v - ylo for v in vals], bottom=ylo,
                      color=colors, alpha=0.80,
                      edgecolor="black", linewidth=0.4)
        # Highlight best
        _highlight_best(ax, bars, vals, higher_better)
        ax.set_xticks(x)
        ax.set_xticklabels(short, fontsize=max(FS_SMALL - 1, 5), rotation=90, ha="center")
        ax.set_xlim(-0.5, len(configs) - 0.5)
        ax.set_title(label, fontsize=FS_AXIS, pad=1)
        if j == 0:
            ax.set_ylabel("Score", fontsize=FS_AXIS)
        ax.tick_params(axis="x", labelsize=max(FS_SMALL - 1, 6))
        ax.tick_params(axis="y", labelsize=max(FS_SMALL - 1, 6))
        ax.grid(alpha=0.22, linestyle="--", linewidth=0.4, axis="y")
        ax.set_ylim(ylo, yhi)
        if ylo > 0:
            ax.yaxis.get_major_formatter().set_useOffset(False)
            ax.ticklabel_format(axis='y', useOffset=False, style='plain')
        ax.yaxis.set_major_locator(plt.MaxNLocator(4, prune="both"))
    return ax_first


def _draw_latent_structure(axes_list, fig, configs, metrics):
    """Latent structure: participation ratio, manifold dimensionality, anisotropy."""
    items = [
        ("LSE_participation_ratio",    "P.Ratio ↑",    True),
        ("LSE_manifold_dimensionality","Int.Dim ↓",    False),
        ("LSE_anisotropy_score",        "Anisotropy ↑", True),
        ("LSE_overall_quality",         "LSE Qual. ↑",  True),
    ]
    x = np.arange(len(configs))
    colors = [_CONFIG_COLOR[c] for c in configs]
    short = [_XSHORT[c] for c in configs]
    ax_first = None
    for j, (key, label, higher_better) in enumerate(items):
        ax = axes_list[j]
        if j == 0:
            ax_first = ax
        vals = [metrics[c].get(key, 0) for c in configs]
        bars = ax.bar(x, vals, color=colors, alpha=0.80,
                      edgecolor="black", linewidth=0.4)
        # Highlight best
        _highlight_best(ax, bars, vals, higher_better)
        ax.set_xticks(x)
        ax.set_xticklabels(short, fontsize=FS_SMALL, rotation=90, ha="center")
        ax.set_xlim(-0.5, len(configs) - 0.5)
        ax.set_title(label, fontsize=FS_AXIS, pad=1)
        if j == 0:
            ax.set_ylabel("Value", fontsize=FS_AXIS)
        ax.tick_params(axis="x", labelsize=max(FS_SMALL - 1, 6))
        ax.tick_params(axis="y", labelsize=max(FS_SMALL - 1, 6))
        ax.grid(alpha=0.22, linestyle="--", linewidth=0.4, axis="y")
        valid_vals = [v for v in vals if np.isfinite(v)]
        ymax = max(abs(v) for v in valid_vals) * 1.18 if valid_vals else 1.0
        ax.set_ylim(0, ymax)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4, prune="both"))
    return ax_first


# ── Main ───────────────────────────────────────────────────────────────────

def build_figure(rdir: Path, outdir: Path, multiseed_stats=None):
    configs, latents, labels, metrics = _load_data(rdir)
    cache_dir = rdir

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)

    # Row A: UMAP grid (2×3) — taller rows, more vertical room for legend
    axes_A = [
        [fig.add_axes([0.060, 0.855, 0.280, 0.120]),
         fig.add_axes([0.370, 0.855, 0.280, 0.120]),
         fig.add_axes([0.680, 0.855, 0.280, 0.120])],
        [fig.add_axes([0.060, 0.700, 0.280, 0.120]),
         fig.add_axes([0.370, 0.700, 0.280, 0.120]),
         fig.add_axes([0.680, 0.700, 0.280, 0.120])],
    ]

    # Row B: 3 bar charts — wider gap below for rotated xtick labels
    axes_B = [
        fig.add_axes([0.100, 0.510, 0.253, 0.145]),
        fig.add_axes([0.393, 0.510, 0.253, 0.145]),
        fig.add_axes([0.707, 0.510, 0.253, 0.145]),
    ]

    # Row C: 4 neighbourhood quality bars — explicit wider gutters for y-ticks
    axes_C = [
        fig.add_axes([0.090, 0.295, 0.162, 0.125]),
        fig.add_axes([0.310, 0.295, 0.162, 0.125]),
        fig.add_axes([0.530, 0.295, 0.162, 0.125]),
        fig.add_axes([0.750, 0.295, 0.162, 0.125]),
    ]

    # Row D: 4 latent structure bars — explicit wider gutters for y-ticks
    axes_D = [
        fig.add_axes([0.090, 0.085, 0.162, 0.120]),
        fig.add_axes([0.310, 0.085, 0.162, 0.120]),
        fig.add_axes([0.530, 0.085, 0.162, 0.120]),
        fig.add_axes([0.750, 0.085, 0.162, 0.120]),
    ]

    print("  Drawing Panel A (UMAP grid)...")
    ax_A = _draw_umap_grid(axes_A, fig, configs, latents, labels, cache_dir)

    print("  Drawing Panel B (Clustering metrics)...")
    ax_B = _draw_clustering_bars(axes_B, fig, configs, metrics, multiseed_stats=multiseed_stats)

    print("  Drawing Panel C (Neighbourhood quality)...")
    ax_C = _draw_neighbourhood_quality(axes_C, fig, configs, metrics)

    print("  Drawing Panel D (Latent structure)...")
    ax_D = _draw_latent_structure(axes_D, fig, configs, metrics)
    panel_label(fig, ax_A, "A", x_off=-0.026)
    panel_label(fig, ax_B, "B", x_off=-0.026)
    panel_label(fig, ax_C, "C", x_off=-0.026)
    panel_label(fig, ax_D, "D", x_off=-0.026)

    print("\n── Conflict Detection ──")
    issues = detect_all_conflicts(fig, label="quant_comparison", verbose=True)

    outpath = outdir / "fig2_quant_comparison.png"
    from mocoo.visualization.style import save_figure
    save_figure(fig, outpath)

    # Export individual panel sub-figures
    sub_dir = outdir / "fig2_quant_comparison"
    sub_dir.mkdir(parents=True, exist_ok=True)
    export_subpanels(fig, sub_dir, [(ax_A, "panelA_umap"),
                                     (ax_B, "panelB_core_metrics"),
                                     (ax_C, "panelC_dre_metrics"),
                                     (ax_D, "panelD_cal_dav")])
    plt.close(fig)

    n_warn = sum(1 for x in issues if x.get("severity") == "warning")
    n_err  = sum(1 for x in issues if x.get("severity") == "error")
    print(f"\nSaved -> {outpath}")
    print(f"{n_warn} warnings | {n_err} errors")
    return issues


def main():
    _benchmarks = Path(__file__).resolve().parent.parent.parent  # benchmarks/
    p = argparse.ArgumentParser()
    p.add_argument("--resultsdir",
                   default=str(_benchmarks / "results" / "single_dataset"))
    p.add_argument("--outdir",
                   default=str(_benchmarks / "figures"))
    p.add_argument("--multiseed-csv", default=None)
    args = p.parse_args()
    rdir   = Path(args.resultsdir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ms_stats = None
    if args.multiseed_csv:
        ms_stats = load_multiseed_stats(Path(args.multiseed_csv))
    return build_figure(rdir, outdir, multiseed_stats=ms_stats)


if __name__ == "__main__":
    main()
