#!/usr/bin/env python
"""MoCoO Supplementary Figure S7 — Multi-seed robustness analysis.

Two-panel figure using the direct_layout geometry engine:
  (a)  Box/violin per config showing ARI / NMI / ASW distributions (5 seeds)
  (b)  Mean ± 95 % CI bars with pairwise significance annotations
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
    apply_style, save_figure, add_panel_label,
    get_config_colors, get_base_config_order, get_tick_name, get_legend_name,
)
from mocoo.visualization.direct_layout import bind_figure_region

setup_fonts()
apply_style()


def _load_multiseed(results_dir: Path):
    """Load multiseed CSV into per-config arrays."""
    csv_path = results_dir / "multiseed_IRALL.csv"
    if not csv_path.exists():
        csv_path = results_dir / "multiseed" / "multiseed_IRALL.csv"
    if not csv_path.exists():
        return {}
    data = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cfg = row["config"].strip()
            if cfg not in data:
                data[cfg] = {"ARI": [], "NMI": [], "ASW": []}
            for m in ("ARI", "NMI", "ASW"):
                try:
                    data[cfg][m].append(float(row[m]))
                except (KeyError, ValueError):
                    pass
    return data


def _load_significance(results_dir: Path):
    """Load significance summary CSV."""
    fp = results_dir / "significance" / "significance_summary.csv"
    if not fp.exists():
        fp = results_dir / "multiseed" / "significance" / "significance_summary.csv"
    if not fp.exists():
        return {}
    rows = {}
    with open(fp) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cfg = row["config"].strip()
            rows[cfg] = row
    return rows


def make_figure(results_dir: Path, out_path: Path):
    ms_data = _load_multiseed(results_dir)
    sig_data = _load_significance(results_dir)
    if not ms_data:
        print("No multiseed data found.")
        return

    configs = get_base_config_order()
    config_colors = get_config_colors()
    configs = [c for c in configs if c in ms_data]

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * 0.75))
    root = bind_figure_region(fig, (0.16, 0.08, 0.88, 0.90))
    (r_a, r_b) = root.split_cols([3, 2], gap=0.10)

    # --- Panel (a): Horizontal boxplots per metric (stacked vertically) ---
    metrics = ["ARI", "NMI", "ASW"]
    regions_a = r_a.split_rows([1] * len(metrics), gap=0.08)

    for mi, metric in enumerate(metrics):
        ax = regions_a[mi].add_axes(fig)
        box_data = [ms_data[c][metric] for c in configs]
        colors = [config_colors.get(c, "#888888") for c in configs]
        bp = ax.boxplot(box_data, patch_artist=True, widths=0.6, vert=False,
                        medianprops=dict(color="black", linewidth=1.2),
                        capprops=dict(linewidth=0),
                        whiskerprops=dict(linewidth=0),
                        flierprops=dict(markersize=0))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        # Overlay individual seeds
        for j, vals in enumerate(box_data):
            jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
            ax.scatter(vals, [j + 1 + jit for jit in jitter],
                       color="black", s=12, zorder=5, alpha=0.7)

        ax.set_yticks(range(1, len(configs) + 1))
        ax.set_yticklabels([get_tick_name(c) for c in configs],
                           fontsize=FS_SMALL)
        ax.set_xlabel(metric, fontsize=FS_AXIS)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        if mi == 0:
            ax.set_title("Per-seed Distributions", fontsize=FS_TITLE)
            add_panel_label(ax, "a")

    # --- Panel (b): Mean ± CI bar chart ---
    ax_b = r_b.add_axes(fig)
    metric_show = "ARI"
    x = np.arange(len(configs))
    means = []
    lows = []
    highs = []
    for c in configs:
        if c in sig_data:
            m = float(sig_data[c].get(f"{metric_show}_mean", 0))
            lo = float(sig_data[c].get(f"{metric_show}_ci95_lo", m))
            hi = float(sig_data[c].get(f"{metric_show}_ci95_hi", m))
        else:
            vals = ms_data[c][metric_show]
            m = np.mean(vals)
            lo = m - 1.96 * np.std(vals) / max(np.sqrt(len(vals)), 1)
            hi = m + 1.96 * np.std(vals) / max(np.sqrt(len(vals)), 1)
        means.append(m)
        lows.append(lo)
        highs.append(hi)

    means_a = np.array(means)
    errs = np.array([means_a - lows, np.array(highs) - means_a])
    colors = [config_colors.get(c, "#888888") for c in configs]
    ax_b.barh(x, means_a, xerr=errs, color=colors, capsize=3, height=0.6,
              zorder=3, ecolor="black")
    ax_b.set_yticks(x)
    ax_b.set_yticklabels([get_tick_name(c) for c in configs], fontsize=FS_TICK)
    ax_b.set_xlabel(f"{metric_show} (mean ± 95% CI)", fontsize=FS_AXIS)
    ax_b.set_title("Robustness (5 seeds)", fontsize=FS_TITLE)
    add_panel_label(ax_b, "b")

    save_figure(fig, str(out_path), vcd_label="figS7_multiseed", vcd_verbose=True)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="MoCoO Fig S7: Multi-seed")
    parser.add_argument("--resultsdir", type=Path,
                        default=Path(__file__).resolve().parent.parent.parent
                        / "results")
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()
    outdir = args.outdir or (args.resultsdir.parent / "figures")
    Path(outdir).mkdir(parents=True, exist_ok=True)
    make_figure(args.resultsdir, Path(outdir) / "figS7_multiseed.png")


if __name__ == "__main__":
    main()
