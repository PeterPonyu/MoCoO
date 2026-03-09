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

from benchmarks.scripts.plotting.shared import setup_fonts, panel_label
from mocoo.visualization.style import (
    FIG_WIDTH_IN, FIG_HEIGHT_IN, DPI, SAVEFIG_KW,
    FS_LABEL, FS_TITLE, FS_AXIS, FS_TICK, FS_LEGEND, FS_SMALL,
    apply_style, get_config_order, get_config_colors, get_short_name,
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


def _draw_paired_bars(ax, data, metric_key, metric_label, configs):
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

    bars_val = ax.bar(x - width / 2, val_vals, width, label="Val",
                       color="#4C72B0", edgecolor="white", linewidth=0.3,
                       alpha=0.85)
    bars_test = ax.bar(x + width / 2, test_vals, width, label="Test",
                        color="#DD8452", edgecolor="white", linewidth=0.3,
                        alpha=0.85)

    # Annotate val-test gap
    for i in range(n):
        if np.isfinite(val_vals[i]) and np.isfinite(test_vals[i]):
            gap = test_vals[i] - val_vals[i]
            sign = "+" if gap >= 0 else ""
            y_pos = max(val_vals[i], test_vals[i]) + 0.01
            ax.text(x[i], y_pos, f"{sign}{gap:.3f}",
                    ha="center", va="bottom", fontsize=FS_SMALL,
                    color="#666666")

    ax.set_xticks(x)
    ax.set_xticklabels([get_short_name(c) for c in configs],
                        fontsize=FS_TICK, rotation=30, ha="right")
    ax.set_title(metric_label, fontsize=FS_TITLE, pad=3)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.22, linestyle="--", linewidth=0.4, axis="y")


def build_figure(rdir: Path, outdir: Path):
    data = _load_metrics(rdir)
    if not data:
        print(f"No JSON files found in {rdir}")
        return []

    configs_present = [c for c in _CONFIGS if c in data]

    n_row1 = len(_ROW1)
    n_row2 = len(_ROW2)
    n_cols = max(n_row1, n_row2)

    fig, axes = plt.subplots(2, n_cols, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN * 0.65),
                              gridspec_kw={"hspace": 0.55, "wspace": 0.40})

    # Row 1: clustering
    for j, (mk, ml) in enumerate(_ROW1):
        _draw_paired_bars(axes[0, j], data, mk, ml, configs_present)
        if j == 0:
            axes[0, j].set_ylabel("Score", fontsize=FS_AXIS)

    # Hide unused axes in row 1
    for j in range(n_row1, n_cols):
        axes[0, j].set_visible(False)

    # Row 2: quality aggregates
    for j, (mk, ml) in enumerate(_ROW2):
        _draw_paired_bars(axes[1, j], data, mk, ml, configs_present)
        if j == 0:
            axes[1, j].set_ylabel("Score", fontsize=FS_AXIS)

    # Single legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=FS_LEGEND, ncol=2,
               loc="upper right", bbox_to_anchor=(0.98, 1.0),
               frameon=True, framealpha=0.65)

    fig.subplots_adjust(left=0.10, right=0.96, top=0.92, bottom=0.12)

    # Panel labels
    panel_label(fig, axes[0, 0], "A", x_off=-0.06, y_off=0.008)
    panel_label(fig, axes[1, 0], "B", x_off=-0.06, y_off=0.008)

    outpath = outdir / "fig7_generalization.png"
    fig.savefig(outpath, **SAVEFIG_KW)
    plt.close(fig)
    print(f"Saved: {outpath}")
    return []


def main():
    _benchmarks = Path(__file__).resolve().parent.parent.parent
    p = argparse.ArgumentParser()
    p.add_argument("--resultsdir",
                   default=str(_benchmarks / "results" / "beta_ablation" / "beta_0.1"))
    p.add_argument("--outdir",
                   default=str(_benchmarks / "figures"))
    args = p.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    build_figure(Path(args.resultsdir), outdir)


if __name__ == "__main__":
    main()
