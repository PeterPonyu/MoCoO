#!/usr/bin/env python
"""MoCoO Figure 4 — FM-enhanced comparison boxplots.

Same layout as Figure 1 but includes all 12 configurations (6 base + 6 FM
variants). Highlights the effect of Phase-2 Flow Matching refinement on
all metric families across all model variants.

Panels:
  (a) Upper: Clustering (6 metrics × 4 splits)
      Lower: LSE        (7 metrics × 4 splits)
  (b) Upper: DRE UMAP   (4 metrics × 4 splits)
      Lower: DRE tSNE   (4 metrics × 4 splits)
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
    apply_style,
    get_config_colors,
    get_config_order,
    get_legend_name,
    get_tick_name,
    save_figure,
)

setup_fonts()
apply_style()

_CONFIGS = get_config_order()           # includes Full+FM
_CONFIG_COLORS = get_config_colors()
_DATASET_ORDER = [
    "endo", "setty",
    "paul", "IRALL",
    "dentate", "spinoids",
    "lung", "retina", "teeth",
    "hepatoblastoma",
]

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

_ALL_BLOCKS = [_PANEL_A_UPPER, _PANEL_A_LOWER, _PANEL_B_UPPER, _PANEL_B_LOWER]
_SPLITS = ["train", "val", "test", "whole"]
_SPLIT_TITLES = {"train": "Train", "val": "Validation", "test": "Test", "whole": "Whole set"}
_LOWER_IS_BETTER = {"DAV"}

_FIGURE_SIZE = (22.0, 11.0)  # slightly wider to fit 7 configs

_FS_TITLE = FS_TITLE + 2
_FS_AXIS = FS_AXIS + 2
_FS_TICK = FS_TICK
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
    arrow = "\u2193" if metric_key in _LOWER_IS_BETTER else "\u2191"
    return f"{display_label} {arrow}"


def _plot_boxplot(ax, datasets, data, split_name, metric_key,
                  y_limits, show_title, show_xticklabels,
                  show_ylabel, show_yticks):
    # Only include configs that have data
    active_configs = [c for c in _CONFIGS if any(
        c in data[ds] for ds in datasets
    )]
    positions = np.arange(len(active_configs))
    bp_data = []
    colors = []
    for config in active_configs:
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
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)

    rng = np.random.default_rng(42)
    for i, vals in enumerate(bp_data):
        if vals:
            jitter = rng.uniform(-0.10, 0.10, len(vals))
            ax.scatter(
                positions[i] + jitter[: len(vals)],
                vals,
                s=10,
                color=colors[i],
                alpha=0.55,
                zorder=5,
                edgecolors="white",
                linewidth=0.3,
            )

    ax.set_xlim(-0.5, len(active_configs) - 0.5)
    ax.set_ylim(*y_limits)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
    ax.set_xticks(positions)
    if show_xticklabels:
        ax.set_xticklabels(
            [get_tick_name(c) for c in active_configs],
            fontsize=_FS_TICK - 1,
            rotation=90,
            ha="center",
        )
    else:
        ax.set_xticklabels([])
    if show_ylabel:
        ax.set_ylabel(f"{_SPLIT_TITLES[split_name]}", fontsize=_FS_AXIS)
    if not show_yticks:
        ax.tick_params(axis="y", labelleft=False)
    ax.tick_params(axis="y", labelsize=_FS_TICK)
    ax.grid(axis="y", alpha=0.22, linestyle="--", linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _build_composed_axes(fig):
    left_margin = 0.050
    right_edge = 0.992
    top_edge = 0.92
    bottom_edge = 0.075

    panel_gap = 0.032
    block_gap = 0.090
    col_gap = 0.005
    row_gap = 0.012

    a_cols = max(len(_PANEL_A_UPPER[1]), len(_PANEL_A_LOWER[1]))
    b_cols = max(len(_PANEL_B_UPPER[1]), len(_PANEL_B_LOWER[1]))

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

    return [
        (_PANEL_A_UPPER, _make_block_axes(a_left, ab_upper_top, len(_PANEL_A_UPPER[1]))),
        (_PANEL_A_LOWER, _make_block_axes(a_left, ab_lower_top, len(_PANEL_A_LOWER[1]))),
        (_PANEL_B_UPPER, _make_block_axes(b_left, ab_upper_top, len(_PANEL_B_UPPER[1]))),
        (_PANEL_B_LOWER, _make_block_axes(b_left, ab_lower_top, len(_PANEL_B_LOWER[1]))),
    ]


def build_figure(results_dir: Path, outdir: Path):
    datasets, data = _load_dataset_summaries(results_dir)
    if not datasets:
        print(f"No per-dataset summary_expanded.csv files found in {results_dir}")
        return []

    # Check if any +FM data exists
    has_fm = any(
        any(c.endswith("+FM") for c in data[ds])
        for ds in datasets
    )
    if not has_fm:
        print("⚠ No +FM data found in any dataset. "
              "Run run_fm_pipeline.py first to generate FM results.")
        print("  Proceeding with available configs only.")

    fig = plt.figure(figsize=_FIGURE_SIZE)
    block_list = _build_composed_axes(fig)

    metric_limits = {}
    for _, metrics in _ALL_BLOCKS:
        for mk, _ in metrics:
            for split in _SPLITS:
                metric_limits[(mk, split)] = _metric_bounds(data, mk, split)

    panel_info = [("(a)", [0, 1]), ("(b)", [2, 3])]

    for block_idx, ((block_title, metrics), axes) in enumerate(block_list):
        for row_idx, split_name in enumerate(_SPLITS):
            is_bottom_row = (row_idx == 3)
            for col_idx, (mk, mlabel) in enumerate(metrics):
                _plot_boxplot(
                    axes[row_idx][col_idx],
                    datasets, data, split_name, mk,
                    metric_limits[(mk, split_name)],
                    show_title=(row_idx == 0),
                    show_xticklabels=is_bottom_row,
                    show_ylabel=(col_idx == 0),
                    show_yticks=(col_idx == 0),
                )
                if row_idx == 0:
                    axes[row_idx][col_idx].set_title(
                        _metric_label(mk, mlabel),
                        fontsize=_FS_TITLE, pad=5,
                    )

        first_pos = axes[0][0].get_position()
        fig.text(
            first_pos.x0, first_pos.y1 + 0.020,
            block_title, fontsize=_FS_TITLE, ha="left", va="bottom",
        )

    for label, block_indices in panel_info:
        first_block_axes = block_list[block_indices[0]][1]
        pos = first_block_axes[0][0].get_position()
        fig.text(
            pos.x0, pos.y1 + 0.040, label,
            fontsize=_FS_TITLE + 1, fontweight="bold", ha="left", va="bottom",
        )

    # Determine active configs for legend
    active_configs = [c for c in _CONFIGS if any(
        c in data[ds] for ds in datasets
    )]
    legend_patches = [
        Patch(
            facecolor=_CONFIG_COLORS[c], edgecolor="white",
            alpha=0.82, label=get_legend_name(c),
        )
        for c in active_configs
    ]
    a_upper_axes = block_list[0][1]
    top_row_last = a_upper_axes[0][-1].get_position()
    bot_row_last = a_upper_axes[-1][-1].get_position()
    legend_x = top_row_last.x1 + 0.010
    legend_y = (top_row_last.y1 + bot_row_last.y0) / 2
    fig.legend(
        handles=legend_patches,
        loc="center left",
        fontsize=9,
        frameon=False,
        ncol=1,
        bbox_to_anchor=(legend_x, legend_y),
        handlelength=0.9,
        handletextpad=0.3,
        labelspacing=0.25,
    )

    outpath = outdir / "fig4_fm_comparison.png"
    issues = save_figure(
        fig, outpath, vcd_label="fm_comparison", vcd_verbose=True,
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
