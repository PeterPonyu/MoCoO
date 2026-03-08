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
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
import numpy as np
from sklearn.neighbors import NearestNeighbors
from umap import UMAP

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from benchmarks.scripts.pipeline.visual_conflict_detector import detect_all_conflicts

# ── Import centralized style ────────────────────────────────────────────────
from mocoo.visualization.style import (
    FIG_WIDTH_IN as FIG_W, FIG_HEIGHT_IN as FIG_H, DPI, SAVEFIG_KW,
    FS_LABEL, FS_TITLE, FS_AXIS, FS_TICK, FS_LEGEND as FS_LEG, FS_SMALL,
    get_config_colors, get_config_order, get_short_name, apply_style,
)

apply_style()

# ── Fonts ──────────────────────────────────────────────────────────────────
_FONT_DIR = Path(__file__).resolve().parent.parent.parent / "fonts"
for _fp in (_FONT_DIR / "Arial.ttf", _FONT_DIR / "Arial Bold.ttf"):
    if _fp.exists():
        fm.fontManager.addfont(str(_fp))
if (_FONT_DIR / "Arial.ttf").exists():
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["Arial"] + list(
        matplotlib.rcParams.get("font.sans-serif", []))

# ── Style constants from centralized module ──────────────────────────────────
_CONFIGS = get_config_order()
_CONFIG_COLOR = get_config_colors()
_SCATTER = dict(s=1.2, alpha=0.50, linewidths=0, rasterized=True)
_XSHORT = {c: get_short_name(c) for c in _CONFIGS}

# ── Data loading ───────────────────────────────────────────────────────────
def _unify_metric_keys(m: dict) -> dict:
    """Normalise JSON metric keys so downstream code uses short names."""
    _MAP = {
        "full_ARI": "ARI", "full_NMI": "NMI", "full_ASW": "ASW",
        "full_CH": "CAL", "full_DB": "DAV", "corr": "COR",
        "CH": "CAL", "DB": "DAV",
    }
    for src, dst in _MAP.items():
        if src in m and dst not in m:
            m[dst] = m[src]
    return m

def _load_data(rdir: Path):
    npz = np.load(rdir / "benchmark_data.npz", allow_pickle=True)
    configs  = [str(c) for c in npz["configs"]]
    latents  = [np.asarray(z, dtype=np.float32) for z in npz["latents"]]
    labels   = [np.asarray(lb) for lb in npz["labels"]]
    metrics  = {}
    for cfg in configs:
        key = cfg.replace("+", "_")
        jf  = rdir / f"{key}.json"
        if jf.exists():
            with open(jf) as f:
                metrics[cfg] = _unify_metric_keys(json.load(f))
    return configs, latents, labels, metrics


def _compute_umap(latent, cache_dir: Path, tag: str):
    cache = cache_dir / f"qc_umap_{tag}.npz"
    if cache.exists():
        return np.load(cache)["emb"]
    emb = UMAP(n_components=2, random_state=42, min_dist=0.3,
               n_neighbors=20, verbose=False).fit_transform(latent)
    np.savez_compressed(cache, emb=emb)
    return emb


def _export_subpanels(fig, sub_dir: Path, panels: list) -> None:
    """Save each panel (axes) as a standalone PNG cropped tightly."""
    renderer = fig.canvas.get_renderer()
    for ax, name in panels:
        if ax is None:
            continue
        try:
            bbox = ax.get_tightbbox(renderer)
            if bbox is None:
                continue
            extent = bbox.transformed(fig.dpi_scale_trans.inverted())
            sp = sub_dir / f"{name}.png"
            fig.savefig(sp, dpi=DPI, bbox_inches=extent)
        except Exception as exc:
            print(f"  sub-panel {name}: skipped ({exc})")


def _panel_label(fig, ax, letter):
    pos = ax.get_position()
    fig.text(pos.x0 - 0.026, pos.y1 + 0.006,
             f"({letter})", fontsize=FS_LABEL, fontweight="bold",
             va="bottom", ha="right", clip_on=False)


def _highlight_best(ax, bars, vals, higher_better=True):
    """Highlight the best-performing bar with a bold edge and value label."""
    best_i = int(np.argmax(vals)) if higher_better else int(np.argmin(vals))
    bars[best_i].set_edgecolor("crimson")
    bars[best_i].set_linewidth(1.4)
    # Add value label above the best bar
    yval = vals[best_i]
    fmt = f"{yval:.3f}" if yval < 1.0 else f"{yval:.1f}"
    ax.annotate(fmt, xy=(best_i, yval),
                xytext=(0, 3), textcoords="offset points",
                ha="center", fontsize=FS_SMALL, color="crimson",
                fontweight="bold")


# ── Panel drawing ──────────────────────────────────────────────────────────

def _draw_umap_grid(gs, fig, configs, latents, labels, cache_dir):
    """2-row × 3-col UMAP grid coloured by cell type."""
    cm20 = plt.colormaps.get_cmap("tab20")
    ax_first = None
    for j, cfg in enumerate(configs):
        r, c = divmod(j, 3)
        ax = fig.add_subplot(gs[r, c])
        if j == 0:
            ax_first = ax
        emb    = _compute_umap(latents[j], cache_dir, cfg.replace("+", "_"))
        uniq   = np.unique(labels[j])
        for k, lb in enumerate(uniq):
            m = labels[j] == lb
            ax.scatter(emb[m, 0], emb[m, 1], color=cm20(k % 20), **_SCATTER)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(cfg, fontsize=FS_TITLE, pad=2,
                     color=_CONFIG_COLOR[cfg], fontweight="bold")
        for spine in ax.spines.values():
            spine.set_edgecolor(_CONFIG_COLOR[cfg])
            spine.set_linewidth(1.0)
    # Shared legend in first panel
    uniq = np.unique(labels[0])
    handles = [plt.Line2D([0],[0], marker="o", color="w",
                           markerfacecolor=cm20(k % 20), markersize=3)
               for k in range(len(uniq))]
    ax_first.legend(handles, [str(lb) for lb in uniq],
                    fontsize=FS_LEG, ncol=2, loc="lower left",
                    framealpha=0.70, handletextpad=0.1,
                    borderpad=0.2, markerscale=0.9, columnspacing=0.4)
    return ax_first


def _draw_clustering_bars(gs, fig, configs, metrics):
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
        ax = fig.add_subplot(gs[j])
        if j == 0:
            ax_first = ax
        vals  = [metrics[c].get(vkey, 0) for c in configs]
        tvals = [metrics[c].get(tkey, 0) for c in configs]
        colors = [_CONFIG_COLOR[c] for c in configs]
        bars1 = ax.bar(x - w/2, vals,  w, color=colors, alpha=0.85,
                       edgecolor="black", linewidth=0.4, label="Val")
        bars2 = ax.bar(x + w/2, tvals, w, color=colors, alpha=0.4,
                       edgecolor="black", linewidth=0.4, hatch="//",
                       label="Test")
        # Highlight best config
        _highlight_best(ax, bars1, vals, higher_better)
        ax.set_xticks(x)
        short = [_XSHORT[c] for c in configs]
        ax.set_xticklabels(short, fontsize=FS_TICK, rotation=45, ha="right")
        ax.set_title(f"{label} {'↑' if higher_better else '↓'}",
                     fontsize=FS_TITLE, pad=2)
        if j == 0:
            ax.set_ylabel("Score", fontsize=FS_AXIS)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(alpha=0.22, linestyle="--", linewidth=0.4, axis="y")
        ax.set_ylim(0, max(max(vals), max(tvals)) * 1.18)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4, prune="upper"))
        if j == 0:
            ax.legend(fontsize=FS_LEG, frameon=True, loc="upper right",
                      framealpha=0.75, handlelength=0.8, labelspacing=0.15,
                      edgecolor="#cccccc")
    return ax_first


def _draw_neighbourhood_quality(gs, fig, configs, metrics):
    """DREX: trustworthiness, continuity, distance Spearman + DRE overall."""
    items = [
        ("DREX_trustworthiness",  "Trustworthiness ↑", True),
        ("DREX_continuity",       "Continuity ↑",      True),
        ("DREX_distance_spearman","Dist. Spearman ↑",  True),
        ("DRE_umap_overall_quality", "DRE Quality ↑",  True),
    ]
    x = np.arange(len(configs))
    colors = [_CONFIG_COLOR[c] for c in configs]
    short = [_XSHORT[c] for c in configs]
    ax_first = None
    for j, (key, label, higher_better) in enumerate(items):
        ax = fig.add_subplot(gs[j])
        if j == 0:
            ax_first = ax
        vals = [metrics[c].get(key, 0) for c in configs]
        bars = ax.bar(x, vals, color=colors, alpha=0.80,
                      edgecolor="black", linewidth=0.4)
        # Highlight best
        _highlight_best(ax, bars, vals, higher_better)
        ax.set_xticks(x)
        ax.set_xticklabels(short, fontsize=FS_TICK, rotation=45, ha="right")
        ax.set_title(label, fontsize=FS_TITLE, pad=2)
        if j == 0:
            ax.set_ylabel("Score", fontsize=FS_AXIS)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(alpha=0.22, linestyle="--", linewidth=0.4, axis="y")
        # Smart y-axis: zoom in to show differences if values are clustered
        valid_vals = [v for v in vals if v > 0 and np.isfinite(v)]
        if valid_vals:
            vmin, vmax = min(valid_vals), max(valid_vals)
            val_range = vmax - vmin
            if val_range < 0.15 * vmax and vmin > 0.3:
                # Values are tightly clustered — zoom in
                ylo = max(0, vmin - val_range * 1.5)
                yhi = vmax + val_range * 1.5
                ax.set_ylim(ylo, yhi)
                # Disable offset notation — show plain values
                ax.yaxis.get_major_formatter().set_useOffset(False)
                ax.ticklabel_format(axis='y', useOffset=False, style='plain')
            else:
                ax.set_ylim(0, vmax * 1.18)
        else:
            ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_locator(plt.MaxNLocator(5, prune="both"))
    return ax_first


def _draw_latent_structure(gs, fig, configs, metrics):
    """Latent structure: participation ratio, manifold dimensionality, anisotropy."""
    items = [
        ("LSE_participation_ratio",    "Participation\nRatio ↑",   True),
        ("LSE_manifold_dimensionality","Intrinsic\nDimension ↓",   False),
        ("LSE_anisotropy_score",        "Anisotropy\n(spread) ↑",  True),
        ("LSE_overall_quality",         "LSE Overall\nQuality ↑",  True),
    ]
    x = np.arange(len(configs))
    colors = [_CONFIG_COLOR[c] for c in configs]
    short = [_XSHORT[c] for c in configs]
    ax_first = None
    for j, (key, label, higher_better) in enumerate(items):
        ax = fig.add_subplot(gs[j])
        if j == 0:
            ax_first = ax
        vals = [metrics[c].get(key, 0) for c in configs]
        bars = ax.bar(x, vals, color=colors, alpha=0.80,
                      edgecolor="black", linewidth=0.4)
        # Highlight best
        _highlight_best(ax, bars, vals, higher_better)
        ax.set_xticks(x)
        ax.set_xticklabels(short, fontsize=FS_TICK, rotation=45, ha="right")
        ax.set_title(label, fontsize=FS_TITLE, pad=2)
        if j == 0:
            ax.set_ylabel("Value", fontsize=FS_AXIS)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(alpha=0.22, linestyle="--", linewidth=0.4, axis="y")
        valid_vals = [v for v in vals if np.isfinite(v)]
        ymax = max(abs(v) for v in valid_vals) * 1.18 if valid_vals else 1.0
        ax.set_ylim(0, ymax)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4, prune="upper"))
    return ax_first


# ── Main ───────────────────────────────────────────────────────────────────

def build_figure(rdir: Path, outdir: Path):
    configs, latents, labels, metrics = _load_data(rdir)
    cache_dir = rdir

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)

    outer = gridspec.GridSpec(
        4, 1,
        height_ratios=[3.8, 2.6, 2.6, 2.6],
        hspace=0.35,
        figure=fig,
    )

    # Row A: UMAP grid (2×3)
    gs_A = gridspec.GridSpecFromSubplotSpec(
        2, 3, subplot_spec=outer[0], wspace=0.12, hspace=0.28)

    # Row B: 3 bar charts
    gs_B = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[1], wspace=0.32)

    # Row C: 4 neighbourhood quality bars
    gs_C = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=outer[2], wspace=0.32)

    # Row D: 4 latent structure bars
    gs_D = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=outer[3], wspace=0.32)

    print("  Drawing Panel A (UMAP grid)...")
    ax_A = _draw_umap_grid(gs_A, fig, configs, latents, labels, cache_dir)

    print("  Drawing Panel B (Clustering metrics)...")
    ax_B = _draw_clustering_bars(gs_B, fig, configs, metrics)

    print("  Drawing Panel C (Neighbourhood quality)...")
    ax_C = _draw_neighbourhood_quality(gs_C, fig, configs, metrics)

    print("  Drawing Panel D (Latent structure)...")
    ax_D = _draw_latent_structure(gs_D, fig, configs, metrics)

    fig.subplots_adjust(left=0.10, right=0.97, top=0.96, bottom=0.08)

    _panel_label(fig, ax_A, "A")
    _panel_label(fig, ax_B, "B")
    _panel_label(fig, ax_C, "C")
    _panel_label(fig, ax_D, "D")

    print("\n── Conflict Detection ──")
    issues = detect_all_conflicts(fig, label="quant_comparison", verbose=True)

    outpath = outdir / "quant_comparison.png"
    fig.savefig(outpath, **SAVEFIG_KW)

    # Export individual panel sub-figures
    sub_dir = outdir / "fig2_quant_comparison"
    sub_dir.mkdir(parents=True, exist_ok=True)
    _export_subpanels(fig, sub_dir, [(ax_A, "panelA_umap"),
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
                   default=str(_benchmarks / "results" / "dataset_default"))
    p.add_argument("--outdir",
                   default=str(_benchmarks / "figures"))
    args = p.parse_args()
    rdir   = Path(args.resultsdir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    build_figure(rdir, outdir)


if __name__ == "__main__":
    main()
