#!/usr/bin/env python
"""MoCoO Figure 2 — Ablation boxplots on proposed metrics.

Single composed figure with two panels of 4 metrics each:
  (a) Clustering:  NMI, ARI, ASW, DAV
  (b) Quality:     DRE, LSE, DREX, LSEX

Each subplot shows per-config boxplot distributions across all available
datasets.  Four rows per panel correspond to train/val/test/whole splits.
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
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from benchmarks.scripts.plotting.shared import setup_fonts
from mocoo.visualization.style import (
    FS_AXIS,
    FS_LEGEND,
    FS_TITLE,
    FS_TICK,
    PROPOSED_CLUSTERING,
    PROPOSED_QUALITY,
    PROPOSED_DIRECTION,
    PROPOSED_SHORT_LABELS,
    HIGHLIGHT_CONFIGS,
    HIGHLIGHT_EDGE_WIDTH,
    apply_style,
    get_config_colors,
    get_base_config_order,
    get_legend_name,
    get_tick_name,
    save_figure,
    style_boxplot,
)

setup_fonts()
apply_style()

_CONFIGS = get_base_config_order()
_CONFIG_COLORS = get_config_colors()
_DATASET_ORDER = [
    "endo", "setty",                        # Stem cell / early development
    "paul", "IRALL", "hemato",              # Hematopoietic / immune
    "dentate", "spinoids", "astrocyte",     # Neural development
    "lung", "retina", "teeth", "spine",     # Organ-specific development
    "hepatoblastoma", "brainmet",           # Cancer
    "breast", "gastric", "livercancer",     # Cancer (extended)
    "melanoma", "pituitary", "hESCtime",    # Additional systems
]

# Layout: 2 panels (a, b), each with 4 metrics × 4 splits.
# fmt: off
_PANEL_A = (
    "Embedding Quality",
    [(m, PROPOSED_SHORT_LABELS[m]) for m in PROPOSED_QUALITY],
)
_PANEL_B = (
    "Clustering",
    [(m, PROPOSED_SHORT_LABELS[m]) for m in PROPOSED_CLUSTERING],
)
# fmt: on

_ALL_BLOCKS = [_PANEL_A, _PANEL_B]

_SPLITS = ["train", "val", "test", "whole"]
_SPLIT_TITLES = {
    "train": "Train",
    "val": "Validation",
    "test": "Test",
    "whole": "Whole set",
}

_FIGURE_SIZE = (16.0, 11.0)

# Font bumps relative to style defaults
_FS_TITLE = FS_TITLE + 2
_FS_AXIS = FS_AXIS + 2
_FS_TICK = FS_TICK + 2
_FS_LEGEND = FS_LEGEND + 2


def _safe_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _load_dataset_summaries(results_dir: Path):
    data: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    datasets: list[str] = []
    metric_keys = [mk for _, metrics in _ALL_BLOCKS for mk, _ in metrics]
    for dataset in _DATASET_ORDER:
        summary_path = results_dir / dataset / "summary_expanded.csv"
        if not summary_path.exists():
            continue
        rows: dict[str, dict[str, dict[str, float]]] = {}
        with summary_path.open() as handle:
            for row in csv.DictReader(handle):
                config = row["config"]
                split = row["split"]
                if config not in _CONFIGS or split not in _SPLITS:
                    continue
                rows.setdefault(config, {})[split] = {
                    mk: _safe_float(row.get(mk)) for mk in metric_keys
                }
        if rows:
            datasets.append(dataset)
            data[dataset] = rows
    return datasets, data


def _metric_bounds(data, metric_key, split=None):
    values = []
    splits_to_check = [split] if split else _SPLITS
    for dataset_data in data.values():
        for config_data in dataset_data.values():
            for s in splits_to_check:
                if s in config_data:
                    v = config_data[s].get(metric_key, np.nan)
                    if np.isfinite(v):
                        values.append(float(v))
    if not values:
        return 0.0, 1.0
    lo, hi = min(values), max(values)
    span = hi - lo
    pad = max(0.025, 0.15 * span) if span > 0 else max(0.05, abs(hi) * 0.15 if hi != 0 else 0.1)
    return lo - pad, hi + pad


def _metric_label(metric_key: str, display_label: str) -> str:
    higher = PROPOSED_DIRECTION.get(metric_key, True)
    arrow = "\u2191" if higher else "\u2193"
    return f"{display_label} {arrow}"


# ── boxplot renderer ──────────────────────────────────────────────────────

def _plot_boxplot(ax, datasets, data, split_name, metric_key,
                  y_limits, show_title, show_xticklabels,
                  show_ylabel, show_yticks):
    positions = np.arange(len(_CONFIGS))
    bp_data = []
    colors = []
    for config in _CONFIGS:
        vals = [
            data[ds].get(config, {}).get(split_name, {}).get(metric_key, np.nan)
            for ds in datasets
        ]
        vals = [v for v in vals if np.isfinite(v)]
        bp_data.append(vals)
        colors.append(_CONFIG_COLORS[config])

    bplot = ax.boxplot(
        bp_data,
        positions=positions,
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="white", linewidth=1.0),
        whiskerprops=dict(linewidth=0.7),
        capprops=dict(linewidth=0.7),
        boxprops=dict(linewidth=0.5),
    )
    for i, (patch, color) in enumerate(zip(bplot["boxes"], colors)):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)
        # Build a per-config boxplot dict for style_boxplot
        bp_single = {
            "boxes": [bplot["boxes"][i]],
            "whiskers": bplot["whiskers"][2 * i: 2 * i + 2],
            "caps": bplot["caps"][2 * i: 2 * i + 2],
            "medians": [bplot["medians"][i]],
        }
        style_boxplot(bp_single, _CONFIGS[i], color)

    rng = np.random.default_rng(42)
    for i, vals in enumerate(bp_data):
        if vals:
            jitter = rng.uniform(-0.10, 0.10, len(vals))
            ax.scatter(
                positions[i] + jitter[: len(vals)],
                vals,
                s=12,
                color=colors[i],
                alpha=0.55,
                zorder=5,
                edgecolors="white",
                linewidth=0.3,
            )

    ax.set_xlim(-0.5, len(_CONFIGS) - 0.5)
    ax.set_ylim(*y_limits)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
    ax.set_xticks(positions)
    if show_xticklabels:
        ax.set_xticklabels(
            [get_tick_name(c) for c in _CONFIGS],
            fontsize=_FS_TICK - 1,
            rotation=90,
            ha="center",
        )
    else:
        ax.set_xticklabels([])
    if show_ylabel:
        ax.set_ylabel(
            f"{_SPLIT_TITLES[split_name]}", fontsize=_FS_AXIS,
        )
    if not show_yticks:
        ax.tick_params(axis="y", labelleft=False)
    ax.tick_params(axis="y", labelsize=_FS_TICK)
    ax.grid(axis="y", alpha=0.22, linestyle="--", linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── composed-figure axes builder ──────────────────────────────────────────

def _build_composed_axes(fig):
    """Build a 2-panel grid: (a) Clustering 4 cols, (b) Quality 4 cols.

    Each panel has 4 rows (one per split).
    """
    left_margin = 0.065
    right_edge = 0.992
    top_edge = 0.92
    bottom_edge = 0.075

    panel_gap = 0.045       # horizontal gap between panel a and b
    col_gap = 0.008         # gap between columns within a block
    row_gap = 0.012         # gap between split rows

    a_cols = len(_PANEL_A[1])   # 4
    b_cols = len(_PANEL_B[1])   # 4

    total_content_w = right_edge - left_margin
    total_inner_col_gap = (a_cols - 1 + b_cols - 1) * col_gap
    col_width = (total_content_w - panel_gap - total_inner_col_gap) / (a_cols + b_cols)

    # Vertical: 4 rows
    total_content_h = top_edge - bottom_edge
    n_row_gaps = 3  # 3 gaps between 4 rows
    row_height = (total_content_h - n_row_gaps * row_gap) / 4

    def _make_block_axes(x_start, y_top, ncols):
        axes = []
        for r in range(4):
            bottom = y_top - (r + 1) * row_height - r * row_gap
            row_axes = []
            x = x_start
            for c in range(ncols):
                row_axes.append(fig.add_axes([x, bottom, col_width, row_height]))
                x += col_width + col_gap
            axes.append(row_axes)
        return axes

    a_left = left_margin
    b_left = a_left + a_cols * col_width + (a_cols - 1) * col_gap + panel_gap

    a_axes = _make_block_axes(a_left, top_edge, a_cols)
    b_axes = _make_block_axes(b_left, top_edge, b_cols)

    return [
        (_PANEL_A, a_axes),
        (_PANEL_B, b_axes),
    ]


# ── main entry points ─────────────────────────────────────────────────────

def build_figure(results_dir: Path, outdir: Path):
    datasets, data = _load_dataset_summaries(results_dir)
    if not datasets:
        print(f"No per-dataset summary_expanded.csv files found in {results_dir}")
        return []

    fig = plt.figure(figsize=_FIGURE_SIZE)
    block_list = _build_composed_axes(fig)

    metric_limits = {}
    for _, metrics in _ALL_BLOCKS:
        for mk, _ in metrics:
            for split in _SPLITS:
                metric_limits[(mk, split)] = _metric_bounds(data, mk, split)

    panel_labels = ["(a)", "(b)"]

    for block_idx, ((block_title, metrics), axes) in enumerate(block_list):
        for row_idx, split_name in enumerate(_SPLITS):
            is_bottom_row = (row_idx == 3)
            for col_idx, (mk, mlabel) in enumerate(metrics):
                _plot_boxplot(
                    axes[row_idx][col_idx],
                    datasets,
                    data,
                    split_name,
                    mk,
                    metric_limits[(mk, split_name)],
                    show_title=(row_idx == 0),
                    show_xticklabels=is_bottom_row,
                    show_ylabel=(col_idx == 0),
                    show_yticks=(col_idx == 0),
                )
                if row_idx == 0:
                    axes[row_idx][col_idx].set_title(
                        _metric_label(mk, mlabel),
                        fontsize=_FS_TITLE,
                        pad=5,
                    )

        # Block title above first column
        first_pos = axes[0][0].get_position()
        fig.text(
            first_pos.x0, first_pos.y1 + 0.020,
            block_title,
            fontsize=_FS_TITLE,
            ha="left",
            va="bottom",
        )

    # Panel labels
    for idx, ((_, _), axes) in enumerate(block_list):
        pos = axes[0][0].get_position()
        fig.text(
            pos.x0, pos.y1 + 0.040,
            panel_labels[idx],
            fontsize=_FS_TITLE + 1,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    # Shared colour-patch legend — centered at top
    legend_patches = [
        Patch(
            facecolor=_CONFIG_COLORS[c],
            edgecolor=_CONFIG_COLORS[c] if c in HIGHLIGHT_CONFIGS else "white",
            linewidth=HIGHLIGHT_EDGE_WIDTH if c in HIGHLIGHT_CONFIGS else 0.5,
            alpha=0.82, label=get_legend_name(c),
        )
        for c in _CONFIGS
    ]
    fig.legend(
        handles=legend_patches,
        loc="upper center",
        fontsize=_FS_LEGEND,
        frameon=False,
        ncol=len(_CONFIGS),
        bbox_to_anchor=(0.5, 0.99),
        handlelength=0.9,
        handletextpad=0.3,
        labelspacing=0.25,
        columnspacing=0.8,
    )

    outpath = outdir / "fig2_ablation_boxplots.png"
    issues = save_figure(
        fig, outpath, vcd_label="fig2_ablation_boxplots", vcd_verbose=True,
    )
    n_warn = sum(1 for i in issues if i.get("severity") == "warning")
    n_err = sum(1 for i in issues if i.get("severity") == "error")
    plt.close(fig)
    print(f"Saved: {outpath}")
    print(f"{n_warn} warnings | {n_err} errors")
    return issues


def main():
    benchmarks_dir = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--resultsdir", default=str(benchmarks_dir / "results"))
    parser.add_argument("--outdir", default=str(benchmarks_dir / "figures"))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    return build_figure(Path(args.resultsdir), outdir)


if __name__ == "__main__":
    main()
