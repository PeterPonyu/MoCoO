#!/usr/bin/env python
"""MoCoO Figure 2 — Per-dataset metric line plots.

Same panel grid and colour scheme as Figure 1 but each subplot is a line plot:
  x-axis  = datasets (ordered low → high by per-metric mean across configs)
  y-axis  = metric value
  lines   = one per model configuration

Panels:
  (a) Upper: Clustering  (6 metrics × 4 splits)
      Lower: LSE         (7 metrics × 4 splits)
  (b) Upper: DRE UMAP    (4 metrics × 4 splits)
      Lower: DRE tSNE    (4 metrics × 4 splits)
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
from matplotlib.lines import Line2D
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
    apply_style,
    get_config_colors,
    get_config_order,
    get_legend_name,
    get_line_style,
    get_line_width,
    save_figure,
)

setup_fonts()
apply_style()

_CONFIGS = get_config_order()
_CONFIG_COLORS = get_config_colors()
_DATASET_ORDER = ["IRALL", "dentate", "endo", "paul", "spinoids"]
_DATASET_DISPLAY = {
    "IRALL": "IR",
    "dentate": "Dent",
    "endo": "Endo",
    "paul": "Paul",
    "spinoids": "Spin",
}

# Layout: 2 panels (a, b), each with upper + lower metric blocks.
# fmt: off
_PANEL_A_UPPER = (
    "Clustering",
    [
        ("ARI", "ARI"),   ("NMI", "NMI"),   ("ASW", "ASW"),
        ("DAV", "DAV"),   ("CAL", "CAL"),   ("COR", "COR"),
    ],
)
_PANEL_A_LOWER = (
    "LSE",
    [
        ("LSE_overall_quality",         "Overall"),
        ("LSE_core_quality",            "Core"),
        ("LSE_manifold_dimensionality", "Dim"),
        ("LSE_spectral_decay_rate",     "Decay"),
        ("LSE_participation_ratio",     "PR"),
        ("LSE_anisotropy_score",        "Aniso"),
        ("LSE_noise_resilience",        "Noise"),
    ],
)
_PANEL_B_UPPER = (
    "DRE — UMAP",
    [
        ("DRE_umap_overall_quality",      "DRE UMAP"),
        ("DRE_umap_Q_local",              "UMAP Qloc"),
        ("DRE_umap_Q_global",             "UMAP Qglob"),
        ("DRE_umap_distance_correlation", "UMAP dCor"),
    ],
)
_PANEL_B_LOWER = (
    "DRE — tSNE",
    [
        ("DRE_tsne_overall_quality",      "DRE tSNE"),
        ("DRE_tsne_Q_local",              "tSNE Qloc"),
        ("DRE_tsne_Q_global",             "tSNE Qglob"),
        ("DRE_tsne_distance_correlation", "tSNE dCor"),
    ],
)
# fmt: on

_ALL_BLOCKS = [
    _PANEL_A_UPPER, _PANEL_A_LOWER,
    _PANEL_B_UPPER, _PANEL_B_LOWER,
]

_SPLITS = ["train", "val", "test", "whole"]
_SPLIT_TITLES = {
    "train": "Train",
    "val": "Validation",
    "test": "Test",
    "whole": "Whole set",
}
_LOWER_IS_BETTER = {"DAV"}

_MARKERS = ["o", "s", "^", "D", "v", "P"]

_FIGURE_SIZE = (20.0, 11.0)

# Font bumps relative to style defaults
_FS_TITLE = FS_TITLE + 2
_FS_AXIS = FS_AXIS + 2
_FS_TICK = FS_TICK + 2
_FS_LEGEND = FS_LEGEND + 2


# ── data loading ──────────────────────────────────────────────────────────

def _safe_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _load_dataset_summaries(results_dir: Path):
    """Load summary_expanded.csv for each dataset.

    Returns (datasets, data) where data[dataset][config][split][metric] = value.
    """
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


def _dataset_order_for_metric(datasets, data, metric_key):
    """Sort datasets low → high by their mean metric value across all configs.

    Uses the 'whole' split for ranking; falls back to mean across all splits.
    """
    means = {}
    for ds in datasets:
        vals = []
        for cfg in _CONFIGS:
            v = data[ds].get(cfg, {}).get("whole", {}).get(metric_key, np.nan)
            if np.isfinite(v):
                vals.append(v)
        if not vals:
            # fall back: mean across all splits
            for cfg in _CONFIGS:
                for sp in _SPLITS:
                    v = data[ds].get(cfg, {}).get(sp, {}).get(metric_key, np.nan)
                    if np.isfinite(v):
                        vals.append(v)
        means[ds] = np.mean(vals) if vals else 0.0
    ascending = metric_key not in _LOWER_IS_BETTER
    return sorted(datasets, key=lambda d: means[d], reverse=not ascending)


def _metric_bounds(data, metric_key):
    values = []
    for dataset_data in data.values():
        for config_data in dataset_data.values():
            for split in _SPLITS:
                if split in config_data:
                    v = config_data[split].get(metric_key, np.nan)
                    if np.isfinite(v):
                        values.append(float(v))
    if not values:
        return 0.0, 1.0
    lo, hi = min(values), max(values)
    span = hi - lo
    pad = max(0.025, 0.15 * span) if span > 0 else max(0.05, abs(hi) * 0.15 if hi != 0 else 0.1)
    return lo - pad, hi + pad


def _metric_label(metric_key: str, display_label: str) -> str:
    arrow = "\u2193" if metric_key in _LOWER_IS_BETTER else "\u2191"
    return f"{display_label} {arrow}"


# ── line-plot renderer ────────────────────────────────────────────────────

def _plot_lines(ax, datasets_sorted, data, split_name, metric_key,
                y_limits, show_title, show_xticklabels,
                show_ylabel, show_yticks):
    """Draw per-config lines across datasets (sorted low→high)."""
    xs = np.arange(len(datasets_sorted))

    for ci, config in enumerate(_CONFIGS):
        ys = []
        for ds in datasets_sorted:
            v = data[ds].get(config, {}).get(split_name, {}).get(metric_key, np.nan)
            ys.append(v)
        ys = np.array(ys, dtype=float)
        mask = np.isfinite(ys)
        ax.plot(
            xs[mask], ys[mask],
            color=_CONFIG_COLORS[config],
            linestyle=get_line_style(config),
            linewidth=get_line_width(config),
            marker=_MARKERS[ci],
            markersize=4.0,
            markeredgecolor="white",
            markeredgewidth=0.4,
            alpha=0.85,
            zorder=5 + ci,
        )

    ax.set_xlim(-0.35, len(datasets_sorted) - 0.65)
    ax.set_ylim(*y_limits)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=2, prune="both"))
    ax.set_xticks(xs)
    if show_xticklabels:
        ax.set_xticklabels(
            [_DATASET_DISPLAY.get(d, d) for d in datasets_sorted],
            fontsize=_FS_TICK - 2,
            rotation=0,
            ha="center",
        )
    else:
        ax.set_xticklabels([])
    if show_ylabel:
        ax.set_ylabel(_SPLIT_TITLES[split_name], fontsize=_FS_AXIS)
    if not show_yticks:
        ax.tick_params(axis="y", labelleft=False)
    ax.tick_params(axis="y", labelsize=_FS_TICK)
    ax.grid(axis="y", alpha=0.22, linestyle="--", linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── composed-figure axes builder ──────────────────────────────────────────

def _build_composed_axes(fig):
    """Build a 2-panel grid identical to Figure 1's layout."""
    left_margin = 0.055
    right_edge = 0.992
    top_edge = 0.92
    bottom_edge = 0.075

    panel_gap = 0.025
    block_gap = 0.090
    col_gap = 0.005
    row_gap = 0.012

    a_cols = max(len(_PANEL_A_UPPER[1]), len(_PANEL_A_LOWER[1]))  # 7
    b_cols = max(len(_PANEL_B_UPPER[1]), len(_PANEL_B_LOWER[1]))  # 4

    total_content_w = right_edge - left_margin
    total_inner_col_gap = (a_cols - 1 + b_cols - 1) * col_gap
    col_width = (total_content_w - panel_gap - total_inner_col_gap) / (a_cols + b_cols)

    total_content_h = top_edge - bottom_edge
    n_row_gaps = 2 * 3
    row_height = (total_content_h - block_gap - n_row_gaps * row_gap) / 8

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

    ab_upper_top = top_edge
    one_block_h = 4 * row_height + 3 * row_gap
    ab_lower_top = ab_upper_top - one_block_h - block_gap

    a_left = left_margin
    b_left = a_left + a_cols * col_width + (a_cols - 1) * col_gap + panel_gap

    a_upper_axes = _make_block_axes(a_left, ab_upper_top, len(_PANEL_A_UPPER[1]))
    a_lower_axes = _make_block_axes(a_left, ab_lower_top, len(_PANEL_A_LOWER[1]))
    b_upper_axes = _make_block_axes(b_left, ab_upper_top, len(_PANEL_B_UPPER[1]))
    b_lower_axes = _make_block_axes(b_left, ab_lower_top, len(_PANEL_B_LOWER[1]))

    return [
        (_PANEL_A_UPPER, a_upper_axes),
        (_PANEL_A_LOWER, a_lower_axes),
        (_PANEL_B_UPPER, b_upper_axes),
        (_PANEL_B_LOWER, b_lower_axes),
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
    dataset_orders = {}
    for _, metrics in _ALL_BLOCKS:
        for mk, _ in metrics:
            metric_limits[mk] = _metric_bounds(data, mk)
            dataset_orders[mk] = _dataset_order_for_metric(datasets, data, mk)

    panel_info = [
        ("(a)", [0, 1]),
        ("(b)", [2, 3]),
    ]

    for block_idx, ((block_title, metrics), axes) in enumerate(block_list):
        for row_idx, split_name in enumerate(_SPLITS):
            is_bottom_row_of_block = (row_idx == 3)
            for col_idx, (mk, mlabel) in enumerate(metrics):
                _plot_lines(
                    axes[row_idx][col_idx],
                    dataset_orders[mk],
                    data,
                    split_name,
                    mk,
                    metric_limits[mk],
                    show_title=(row_idx == 0),
                    show_xticklabels=is_bottom_row_of_block,
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
    for label, block_indices in panel_info:
        first_block_axes = block_list[block_indices[0]][1]
        pos = first_block_axes[0][0].get_position()
        fig.text(
            pos.x0, pos.y1 + 0.040,
            label,
            fontsize=_FS_TITLE + 1,
            ha="left",
            va="bottom",
        )

    # Shared legend in the empty 7th-column slot of the Clustering block
    legend_handles = [
        Line2D(
            [0], [0],
            color=_CONFIG_COLORS[c],
            linestyle=get_line_style(c),
            linewidth=get_line_width(c),
            marker=_MARKERS[i],
            markersize=4.0,
            markeredgecolor="white",
            markeredgewidth=0.4,
            label=get_legend_name(c),
        )
        for i, c in enumerate(_CONFIGS)
    ]
    a_upper_axes = block_list[0][1]
    top_row_last = a_upper_axes[0][-1].get_position()
    bot_row_last = a_upper_axes[-1][-1].get_position()
    legend_x = top_row_last.x1 + 0.012
    legend_y = (top_row_last.y1 + bot_row_last.y0) / 2
    fig.legend(
        handles=legend_handles,
        loc="center left",
        fontsize=8,
        frameon=False,
        ncol=1,
        bbox_to_anchor=(legend_x, legend_y),
        handlelength=1.8,
        handletextpad=0.4,
        labelspacing=0.3,
    )

    outpath = outdir / "fig2_metric_lines.png"
    issues = save_figure(
        fig, outpath, vcd_label="metric_lines", vcd_verbose=True,
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
