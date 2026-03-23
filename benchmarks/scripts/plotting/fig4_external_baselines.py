#!/usr/bin/env python
"""MoCoO Figure — External baselines comparison (proposed metrics).

Three-panel figure:
  (a)  Clustering metrics bars: MoCoO vs external methods (ARI / NMI / ASW / DAV)
  (b)  Embedding quality bars: MoCoO-only (DRE / LSE / DREX / LSEX)
       with dashed "N/A" baseline for external methods
  (c)  Combined rank dot-plot across all available metrics
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

warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from benchmarks.scripts.plotting.shared import setup_fonts
from mocoo.visualization.style import (
    FS_AXIS, FS_LEGEND, FS_SMALL, FS_TICK, FS_TITLE,
    FIG_WIDTH_IN, DPI,
    PROPOSED_CLUSTERING, PROPOSED_QUALITY, PROPOSED_METRICS,
    PROPOSED_DIRECTION, PROPOSED_SHORT_LABELS,
    apply_style, save_figure, add_panel_label,
    get_config_colors, get_base_config_order, get_legend_name,
)
from mocoo.visualization.direct_layout import bind_figure_region

setup_fonts()
apply_style()

_EXTERNAL_COLORS = {
    "scVI": "#7570B3",
    "DPT": "#66A61E",
    "PCA+KMeans": "#A6761D",
    "Harmony": "#E7298A",
}

# Clustering-bar palette (one color per metric, not per method)
_CLUSTER_METRIC_COLORS = {
    "ARI": "#0072B2",
    "NMI": "#E69F00",
    "ASW": "#009E73",
    "DAV": "#CC79A7",
}

_QUALITY_METRIC_COLORS = {
    "DRE_umap_overall_quality": "#0072B2",
    "LSE_overall_quality": "#E69F00",
    "DREX_overall_quality": "#009E73",
    "LSEX_overall_quality": "#CC79A7",
}


def _load_baselines(results_dir: Path):
    """Load external_baselines.csv -> per-method mean across seeds."""
    fp = results_dir / "baselines" / "external_baselines.csv"
    if not fp.exists():
        return {}
    acc = {}
    with open(fp) as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["method"].strip()
            if method not in acc:
                acc[method] = {m: [] for m in PROPOSED_CLUSTERING}
            for m in PROPOSED_CLUSTERING:
                try:
                    if m == "DAV":
                        # External CSV has DB (Davies-Bouldin) — same metric
                        acc[method][m].append(float(row["DB"]))
                    else:
                        acc[method][m].append(float(row[m]))
                except (KeyError, ValueError):
                    pass
    return {m: {k: np.mean(v) if v else np.nan for k, v in vals.items()}
            for m, vals in acc.items()}


def _load_internal(results_dir: Path):
    """Load internal config means from summary_expanded.csv files."""
    ds_dirs = sorted(results_dir.iterdir())
    scores = {}
    for d in ds_dirs:
        fp = d / "summary_expanded.csv"
        if not fp.exists():
            continue
        with open(fp) as f:
            reader = csv.DictReader(f)
            for row in reader:
                cfg = row.get("config", "").strip()
                split = row.get("split", "").strip()
                if split != "whole":
                    continue
                if cfg not in scores:
                    scores[cfg] = {m: [] for m in PROPOSED_METRICS}
                for m in PROPOSED_METRICS:
                    try:
                        scores[cfg][m].append(float(row[m]))
                    except (KeyError, ValueError):
                        pass
    return {c: {k: np.mean(v) if v else np.nan for k, v in vals.items()}
            for c, vals in scores.items() if any(vals[m] for m in PROPOSED_CLUSTERING)}


def make_figure(results_dir: Path, out_path: Path):
    ext = _load_baselines(results_dir)
    internal = _load_internal(results_dir)
    if not ext:
        print("No baseline data found.")
        return

    config_colors = get_config_colors()

    # Methods to show: VAE, Full (best MoCoO), then external
    mocoo_show = []
    if "VAE" in internal:
        mocoo_show.append("VAE")
    if "Full" in internal:
        mocoo_show.append("Full")
    ext_methods = sorted(ext.keys())
    all_methods = mocoo_show + ext_methods

    # ── Figure layout: 3 panels stacked as 2 cols top + 1 col bottom ──
    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * 0.85))
    root = bind_figure_region(fig, (0.08, 0.08, 0.95, 0.95))
    (r_top, r_bot) = root.split_rows([1.2, 1], gap=0.10)
    (r_a, r_b) = r_top.split_cols([1, 1], gap=0.08)

    # ════════════════════════════════════════════════════════════════════
    # Panel (a): Clustering metrics — all methods
    # ════════════════════════════════════════════════════════════════════
    ax_a = r_a.add_axes(fig)
    n_methods = len(all_methods)
    n_metrics = len(PROPOSED_CLUSTERING)
    x = np.arange(n_methods)
    w = 0.8 / n_metrics

    for mi, metric in enumerate(PROPOSED_CLUSTERING):
        vals = []
        for method in all_methods:
            if method in internal:
                vals.append(internal[method].get(metric, 0))
            elif method in ext:
                vals.append(ext[method].get(metric, 0))
            else:
                vals.append(0)
        offset = (mi - n_metrics / 2 + 0.5) * w
        color = _CLUSTER_METRIC_COLORS[metric]
        ax_a.bar(x + offset, vals, w, label=PROPOSED_SHORT_LABELS[metric],
                 color=color, zorder=3, edgecolor="white", linewidth=0.3)

    ax_a.set_xticks(x)
    xlabels = [get_legend_name(m) if m in internal else m for m in all_methods]
    ax_a.set_xticklabels(xlabels, fontsize=FS_SMALL, rotation=30, ha="right")
    ax_a.set_ylabel("Score", fontsize=FS_AXIS)
    ax_a.set_title("Clustering Metrics", fontsize=FS_TITLE)
    ax_a.legend(fontsize=FS_LEGEND, loc="upper right", ncol=2)
    add_panel_label(ax_a, "a")

    # ════════════════════════════════════════════════════════════════════
    # Panel (b): Embedding quality — MoCoO only, external shown as N/A
    # ════════════════════════════════════════════════════════════════════
    ax_b = r_b.add_axes(fig)
    n_q = len(PROPOSED_QUALITY)
    x_b = np.arange(n_methods)
    w_b = 0.8 / n_q

    for mi, metric in enumerate(PROPOSED_QUALITY):
        vals = []
        hatch_list = []
        for method in all_methods:
            if method in internal:
                vals.append(internal[method].get(metric, 0))
                hatch_list.append(None)
            else:
                vals.append(0)
                hatch_list.append("///")
        offset = (mi - n_q / 2 + 0.5) * w_b
        color = _QUALITY_METRIC_COLORS[metric]
        bars = ax_b.bar(x_b + offset, vals, w_b,
                        label=PROPOSED_SHORT_LABELS[metric],
                        color=color, zorder=3, edgecolor="white", linewidth=0.3)
        # Apply hatch to external methods (N/A)
        for bar, h in zip(bars, hatch_list):
            if h:
                bar.set_hatch(h)
                bar.set_facecolor("#E0E0E0")
                bar.set_edgecolor("#999999")

    # Add "N/A" text annotation for external methods
    for xi, method in enumerate(all_methods):
        if method not in internal:
            ax_b.text(xi, 0.02, "N/A", ha="center", va="bottom",
                      fontsize=FS_SMALL - 1, color="#666666", fontstyle="italic")

    ax_b.set_xticks(x_b)
    xlabels_b = [get_legend_name(m) if m in internal else m for m in all_methods]
    ax_b.set_xticklabels(xlabels_b, fontsize=FS_SMALL, rotation=30, ha="right")
    ax_b.set_ylabel("Score", fontsize=FS_AXIS)
    ax_b.set_title("Embedding Quality (Proposed)", fontsize=FS_TITLE)
    ax_b.legend(fontsize=FS_LEGEND, loc="upper right", ncol=2)
    add_panel_label(ax_b, "b")

    # ════════════════════════════════════════════════════════════════════
    # Panel (c): Combined rank dot-plot
    # ════════════════════════════════════════════════════════════════════
    ax_c = r_bot.add_axes(fig)

    # Combine all scores; normalise per metric, then average rank
    all_scores = {}
    for method in all_methods:
        all_scores[method] = {}
        if method in internal:
            for m in PROPOSED_METRICS:
                all_scores[method][m] = internal[method].get(m, np.nan)
        elif method in ext:
            for m in PROPOSED_CLUSTERING:
                all_scores[method][m] = ext[method].get(m, np.nan)
            for m in PROPOSED_QUALITY:
                all_scores[method][m] = np.nan  # not available

    # Min-max normalise per metric (skip NaN; flip DAV since lower is better)
    normed = {m: {} for m in PROPOSED_METRICS}
    for metric in PROPOSED_METRICS:
        vals_valid = []
        for method in all_methods:
            v = all_scores[method].get(metric, np.nan)
            if not np.isnan(v):
                vals_valid.append(v)
        if not vals_valid:
            continue
        lo, hi = min(vals_valid), max(vals_valid)
        rng = hi - lo if hi > lo else 1.0
        for method in all_methods:
            v = all_scores[method].get(metric, np.nan)
            if np.isnan(v):
                normed[metric][method] = np.nan
            else:
                n_val = (v - lo) / rng
                # Flip if lower is better (DAV)
                if not PROPOSED_DIRECTION.get(metric, True):
                    n_val = 1.0 - n_val
                normed[metric][method] = n_val

    # Average normalised score per method (ignoring NaN)
    avg_normed = []
    for method in all_methods:
        vals = [normed[m].get(method, np.nan) for m in PROPOSED_METRICS]
        valid = [v for v in vals if not np.isnan(v)]
        avg = np.mean(valid) if valid else 0
        n_avail = len(valid)
        avg_normed.append((method, avg, n_avail))

    avg_normed.sort(key=lambda t: t[1], reverse=True)

    y_pos = np.arange(len(avg_normed))
    labels = [t[0] for t in avg_normed]
    vals = [t[1] for t in avg_normed]
    n_avail = [t[2] for t in avg_normed]
    colors = []
    for l in labels:
        if l in config_colors:
            colors.append(config_colors[l])
        elif l in _EXTERNAL_COLORS:
            colors.append(_EXTERNAL_COLORS[l])
        else:
            colors.append("#888888")

    bars_c = ax_c.barh(y_pos, vals, color=colors, height=0.55, zorder=3,
                       edgecolor="white", linewidth=0.3)
    ax_c.set_yticks(y_pos)
    display_labels = [get_legend_name(l) if l in internal else l for l in labels]
    ax_c.set_yticklabels(display_labels, fontsize=FS_TICK)
    ax_c.set_xlabel("Mean Normalised Score (across proposed metrics)", fontsize=FS_AXIS)
    ax_c.set_title("Combined Ranking", fontsize=FS_TITLE)
    ax_c.invert_yaxis()

    # Annotate with metric count
    for yi, (v, n) in enumerate(zip(vals, n_avail)):
        suffix = f" ({n}/{len(PROPOSED_METRICS)} metrics)"
        ax_c.text(v + 0.01, yi, f"{v:.2f}{suffix}",
                  va="center", fontsize=FS_SMALL, color="#333333")

    add_panel_label(ax_c, "c")

    save_figure(fig, str(out_path), vcd_label="fig4_external_baselines", vcd_verbose=True)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    benchmarks_dir = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(description="MoCoO: External Baselines (Proposed Metrics)")
    parser.add_argument("--resultsdir", "--results-dir", type=Path,
                        default=benchmarks_dir / "results")
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.out:
        out = args.out
    elif args.outdir:
        out = args.outdir / "fig4_external_baselines.png"
    else:
        out = benchmarks_dir / "figures" / "fig4_external_baselines.png"
    make_figure(args.resultsdir, out)


if __name__ == "__main__":
    main()
