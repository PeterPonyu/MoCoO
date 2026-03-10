#!/usr/bin/env python
"""
MoCoO Figure 7 — Generalization: Validation vs Test Metrics
=============================================================
Paired bar chart showing validation (training split) vs held-out test
(test_ext) metrics for each configuration, assessing generalization.

Row 1: Core clustering (ARI, NMI, ASW)
Row 2: Aggregate quality (DRE overall, DREX overall, LSE overall, LSEX overall)

Usage:
    python -m benchmarks.scripts.plotting.plot_generalization
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

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from benchmarks.scripts.plotting.shared import setup_fonts, panel_label, add_config_legend_footnote, load_multiseed_stats, add_metric_footnote
from benchmarks.scripts.pipeline.visual_conflict_detector import detect_all_conflicts
from mocoo.visualization.style import (
    FIG_WIDTH_IN, FIG_HEIGHT_IN, DPI,
    FS_LABEL, FS_TITLE, FS_AXIS, FS_TICK, FS_LEGEND, FS_SMALL,
    apply_style, get_config_order, get_config_colors, get_short_name, get_tick_name,
    row_of_axes,
)

setup_fonts()
apply_style()

_CONFIGS = get_config_order()
_COLORS = get_config_colors()

# Row 1: clustering metrics
_ROW1 = [
    ("ARI", "ARI"),
    ("NMI", "NMI"),
    ("ASW", "ASW"),
]

# Row 2: aggregate quality metrics
_ROW2 = [
    ("DRE_umap_overall_quality", "DRE (UMAP)"),
    ("DREX_overall_quality", "DREX"),
    ("LSE_overall_quality", "LSE"),
    ("LSEX_overall_quality", "LSEX"),
]


def _load_metrics(rdir: Path) -> dict:
    data = {}
    for cfg in _CONFIGS:
        key = cfg.replace("+", "_")
        jf = rdir / f"{key}.json"
        if jf.exists():
            with open(jf) as f:
                data[cfg] = json.load(f)
    return data


def _draw_paired_bars(ax, data, metric_key, metric_label, configs, multiseed_stats=None):
    """Draw paired val/test bars for one metric across all configs."""
    n = len(configs)
    x = np.arange(n)
    width = 0.35

    val_vals = []
    test_vals = []
    for cfg in configs:
        m = data.get(cfg, {})
        val_vals.append(m.get(metric_key, np.nan))
        test_vals.append(m.get(f"test_ext_{metric_key}", np.nan))

    bars_val = ax.bar(x - width / 2, val_vals, width,
                       color=[_COLORS[c] for c in configs],
                       edgecolor="white", linewidth=0.3,
                       alpha=0.85)
    bars_test = ax.bar(x + width / 2, test_vals, width,
                        color=[_COLORS[c] for c in configs],
                        edgecolor="white", linewidth=0.3,
                        hatch="//", alpha=0.55)

    if multiseed_stats:
        val_yerr = [multiseed_stats.get(c, {}).get(metric_key, (0, 0))[1]
                    for c in configs]
        test_key = f"test_{metric_key}"
        test_yerr = [multiseed_stats.get(c, {}).get(test_key, (0, 0))[1]
                     for c in configs]
        ax.errorbar(x - width / 2, val_vals, yerr=val_yerr, fmt="none",
                    ecolor="black", capsize=2, capthick=0.6, elinewidth=0.6, zorder=5)
        ax.errorbar(x + width / 2, test_vals, yerr=test_yerr, fmt="none",
                    ecolor="black", capsize=2, capthick=0.6, elinewidth=0.6, zorder=5)

    # Annotate val-test gap only if meaningful (>0.01), placed above bars
    for i in range(n):
        if np.isfinite(val_vals[i]) and np.isfinite(test_vals[i]):
            gap = test_vals[i] - val_vals[i]
            if abs(gap) >= 0.01:   # suppress tiny / noisy gaps
                sign = "+" if gap >= 0 else ""
                base_y = max(v for v in [val_vals[i], test_vals[i]]
                              if np.isfinite(v)) + 0.012
                ax.text(x[i], base_y, f"{sign}{gap:.2f}",
                        ha="center", va="bottom", fontsize=max(FS_SMALL - 1, 6),
                        color="#666666")

    # Zoom y-axis to data range for better readability
    all_vals = [v for v in val_vals + test_vals if np.isfinite(v)]
    if all_vals:
        ymin = min(all_vals)
        ymax = max(all_vals)
        margin = (ymax - ymin) * 0.30
        ax.set_ylim(max(0, ymin - margin * 0.5), ymax + margin * 2.8)

    ax.set_xticks(x)
    ax.set_xticklabels([get_tick_name(c) for c in configs],
                        fontsize=FS_TICK, rotation=30, ha="right")
    ax.set_title(metric_label, fontsize=FS_TITLE, pad=3)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.22, linestyle="--", linewidth=0.4, axis="y")


def build_figure(rdir: Path, outdir: Path, multiseed_stats=None):
    data = _load_metrics(rdir)
    if not data:
        print(f"No JSON files found in {rdir}")
        return []

    configs_present = [c for c in _CONFIGS if c in data]

    n_row1 = len(_ROW1)
    n_row2 = len(_ROW2)

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN * 0.65))

    # Row 1: 3 equally-spaced clustering panels
    ax_row1 = row_of_axes(fig, 3, [0.08, 0.56, 0.88, 0.38], gap=0.06)

    # Row 2: 4 equally-spaced quality panels
    ax_row2 = row_of_axes(fig, 4, [0.08, 0.08, 0.88, 0.38], gap=0.04)

    # Row 1: clustering (3 metrics, now spanning full width)
    for j, (mk, ml) in enumerate(_ROW1):
        _draw_paired_bars(ax_row1[j], data, mk, ml, configs_present, multiseed_stats=multiseed_stats)
        if j == 0:
            ax_row1[j].set_ylabel("Score", fontsize=FS_AXIS)

    # Row 2: quality aggregates (4 metrics)
    for j, (mk, ml) in enumerate(_ROW2):
        _draw_paired_bars(ax_row2[j], data, mk, ml, configs_present, multiseed_stats=multiseed_stats)
        if j == 0:
            ax_row2[j].set_ylabel("Score", fontsize=FS_AXIS)

    # Manual legend: solid = Val, hatched = Test
    import matplotlib.patches as mpatches
    val_patch = mpatches.Patch(facecolor="#888888", edgecolor="white", alpha=0.85, label="Val (train split)")
    test_patch = mpatches.Patch(facecolor="#888888", edgecolor="white", alpha=0.55, hatch="//", label="Test (held-out)")
    fig.legend(handles=[val_patch, test_patch], fontsize=FS_LEGEND, ncol=2,
               loc="upper right", bbox_to_anchor=(0.97, 0.99),
               frameon=True, framealpha=0.65)

    # Panel labels
    panel_label(fig, ax_row1[0], "A", x_off=-0.04, y_off=0.008)
    panel_label(fig, ax_row2[0], "B", x_off=-0.04, y_off=0.008)

    add_config_legend_footnote(fig, y_pos=0.005)
    add_metric_footnote(fig, ["ARI", "NMI", "ASW", "DRE", "DREX", "LSE"], y_pos=-0.005)

    outpath = outdir / "fig7_generalization.png"

    print("\n── Conflict Detection ──")
    issues = detect_all_conflicts(fig, label="generalization", verbose=True)
    n_warn = sum(1 for i in issues if i.get("severity") == "warning")
    n_err = sum(1 for i in issues if i.get("severity") == "error")

    from mocoo.visualization.style import save_figure
    save_figure(fig, outpath)
    plt.close(fig)
    print(f"Saved: {outpath}")
    print(f"{n_warn} warnings | {n_err} errors")
    return issues


def main():
    _benchmarks = Path(__file__).resolve().parent.parent.parent
    p = argparse.ArgumentParser()
    p.add_argument("--resultsdir",
                   default=str(_benchmarks / "results" / "beta_ablation" / "beta_0.1"))
    p.add_argument("--outdir",
                   default=str(_benchmarks / "figures"))
    p.add_argument("--multiseed-csv", default=None)
    args = p.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ms = load_multiseed_stats(Path(args.multiseed_csv)) if args.multiseed_csv else None
    build_figure(Path(args.resultsdir), outdir, multiseed_stats=ms)


if __name__ == "__main__":
    main()
