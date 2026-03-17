#!/usr/bin/env python
"""MoCoO Figure 3 — FM-enhanced per-dataset metric profiles (whole split).

Same layout as Figure 2 but includes all 12 configurations (6 base + 6 FM
variants).  Highlights the effect of Phase-2 Flow Matching refinement on
all metric families across datasets.

Layout:  6 rows × 7 columns  (33 metrics, some cells blank)
  Row 1: NMI, ARI, ASW, DAV, CAL, COR              (Clustering, 6)
  Row 2: DRE UMAP 4 metrics                         (DRE UMAP, 4)
  Row 3: DRE tSNE 4 metrics                         (DRE tSNE, 4)
  Row 4: LSE 7 metrics                              (LSE, 7)
  Row 5: DREX 7 metrics                             (DREX, 7)
  Row 6: LSEX 5 metrics                             (LSEX, 5)

x-axis = datasets in biological-context order, y-axis = metric value,
coloured lines = model configurations (base + FM variants).
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
    grid_of_axes,
    save_figure,
)

setup_fonts()
apply_style()

_CONFIGS = get_config_order()           # all 12 configs (6 base + 6 FM)
_CONFIG_COLORS = get_config_colors()

_DATASET_ORDER = [
    "endo", "setty",                        # Stem cell / early development
    "paul", "IRALL",                         # Hematopoietic / immune
    "dentate", "spinoids",                   # Neural development
    "lung", "retina", "teeth",               # Organ-specific development
    "hepatoblastoma",                        # Cancer
]
_DATASET_DISPLAY = {
    "IRALL": "IR",
    "dentate": "Dent",
    "endo": "Endo",
    "paul": "Paul",
    "spinoids": "Spin",
    "lung": "Lung",
    "setty": "Setty",
    "retina": "Ret",
    "teeth": "Teeth",
    "hepatoblastoma": "HB",
}
_CONTEXT_BOUNDARIES = [2, 4, 6, 9]

# 6 rows × up to 7 columns of ALL non-diagnostic metrics
# fmt: off
_ROWS = [
    ("Clustering", [
        ("NMI", "NMI \u2191"),
        ("ARI", "ARI \u2191"),
        ("ASW", "ASW \u2191"),
        ("DAV", "DAV \u2193"),
        ("CAL", "CAL \u2191"),
        ("COR", "COR \u2191"),
    ]),
    ("DRE \u2014 UMAP", [
        ("DRE_umap_overall_quality",      "Overall \u2191"),
        ("DRE_umap_Q_local",              "$Q_{local}$ \u2191"),
        ("DRE_umap_Q_global",             "$Q_{global}$ \u2191"),
        ("DRE_umap_distance_correlation", "dCor \u2191"),
    ]),
    ("DRE \u2014 tSNE", [
        ("DRE_tsne_overall_quality",      "Overall \u2191"),
        ("DRE_tsne_Q_local",              "$Q_{local}$ \u2191"),
        ("DRE_tsne_Q_global",             "$Q_{global}$ \u2191"),
        ("DRE_tsne_distance_correlation", "dCor \u2191"),
    ]),
    ("LSE", [
        ("LSE_overall_quality",         "Overall \u2191"),
        ("LSE_core_quality",            "Core \u2191"),
        ("LSE_manifold_dimensionality", "Dim \u2191"),
        ("LSE_spectral_decay_rate",     "Spec Decay \u2191"),
        ("LSE_participation_ratio",     "Part Ratio \u2191"),
        ("LSE_anisotropy_score",        "Anisotropy \u2191"),
        ("LSE_noise_resilience",        "Noise Res \u2191"),
    ]),
    ("DREX", [
        ("DREX_overall_quality",      "Overall \u2191"),
        ("DREX_trustworthiness",      "Trust \u2191"),
        ("DREX_continuity",           "Continuity \u2191"),
        ("DREX_distance_spearman",    "Spearman \u2191"),
        ("DREX_distance_pearson",     "Pearson \u2191"),
        ("DREX_local_scale_quality",  "Scale Qual \u2191"),
        ("DREX_neighborhood_symmetry","Neigh Sym \u2191"),
    ]),
    ("LSEX", [
        ("LSEX_overall_quality",      "Overall \u2191"),
        ("LSEX_two_hop_connectivity", "2-Hop Conn \u2191"),
        ("LSEX_radial_concentration", "Radial Conc \u2191"),
        ("LSEX_local_curvature",      "Curvature \u2191"),
        ("LSEX_entropy_stability",    "Entropy Stab \u2191"),
    ]),
]
# fmt: on

_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "p", "H", "8"]

_FIGURE_SIZE = (26.0, 16.0)

_FS_TITLE = FS_TITLE + 2
_FS_AXIS = FS_AXIS + 1
_FS_TICK = FS_TICK - 2
_FS_LEGEND = FS_LEGEND


def _safe_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _load_whole_metrics(results_dir: Path):
    """Load whole-split metrics for all datasets (all 12 configs)."""
    all_metric_keys = [mk for _, metrics in _ROWS for mk, _ in metrics]
    data: dict[str, dict[str, dict[str, float]]] = {}
    datasets: list[str] = []
    for dataset in _DATASET_ORDER:
        summary_path = results_dir / dataset / "summary_expanded.csv"
        if not summary_path.exists():
            continue
        rows: dict[str, dict[str, float]] = {}
        with summary_path.open() as handle:
            for row in csv.DictReader(handle):
                config = row["config"]
                if config not in _CONFIGS or row["split"] != "whole":
                    continue
                rows[config] = {
                    mk: _safe_float(row.get(mk)) for mk in all_metric_keys
                }
        if rows:
            datasets.append(dataset)
            data[dataset] = rows
    return datasets, data


def _metric_bounds(data, metric_key):
    values = []
    for ds_data in data.values():
        for cfg_data in ds_data.values():
            v = cfg_data.get(metric_key, np.nan)
            if np.isfinite(v):
                values.append(float(v))
    if not values:
        return 0.0, 1.0
    lo, hi = min(values), max(values)
    span = hi - lo
    pad = max(0.03, 0.15 * span) if span > 0 else max(0.05, abs(hi) * 0.15 if hi != 0 else 0.1)
    return lo - pad, hi + pad


def _plot_line_panel(ax, datasets_sorted, data, metric_key,
                     y_limits, title, show_xticklabels, show_ylabel,
                     row_label):
    xs = np.arange(len(datasets_sorted))
    active_configs = [c for c in _CONFIGS if any(c in data[ds] for ds in datasets_sorted)]

    for ci, config in enumerate(active_configs):
        ys = np.array([
            data[ds].get(config, {}).get(metric_key, np.nan)
            for ds in datasets_sorted
        ], dtype=float)
        mask = np.isfinite(ys)
        ax.plot(
            xs[mask], ys[mask],
            color=_CONFIG_COLORS[config],
            linestyle=get_line_style(config),
            linewidth=get_line_width(config),
            marker=_MARKERS[ci % len(_MARKERS)],
            markersize=4.5,
            markeredgecolor="white",
            markeredgewidth=0.4,
            alpha=0.85,
            zorder=5 + ci,
        )

    ax.set_xlim(-0.35, len(datasets_sorted) - 0.65)
    ax.set_ylim(*y_limits)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))

    for bdry in _CONTEXT_BOUNDARIES:
        if bdry < len(datasets_sorted):
            ax.axvline(bdry - 0.5, color="#999999", linewidth=0.4,
                       linestyle=":", alpha=0.45, zorder=1)

    ax.set_xticks(xs)
    if show_xticklabels:
        ax.set_xticklabels(
            [_DATASET_DISPLAY.get(d, d) for d in datasets_sorted],
            fontsize=_FS_TICK, rotation=45, ha="right",
        )
    else:
        ax.set_xticklabels([])
    if show_ylabel:
        ax.set_ylabel(row_label, fontsize=_FS_AXIS, fontweight="medium")
    ax.set_title(title, fontsize=_FS_TITLE, pad=4)
    ax.tick_params(axis="y", labelsize=_FS_TICK)
    ax.grid(axis="y", alpha=0.22, linestyle="--", linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def build_figure(results_dir: Path, outdir: Path):
    datasets, data = _load_whole_metrics(results_dir)
    if not datasets:
        print(f"No per-dataset summary_expanded.csv files found in {results_dir}")
        return []

    ds_order = [d for d in _DATASET_ORDER if d in datasets]
    for d in datasets:
        if d not in ds_order:
            ds_order.append(d)

    nrows = len(_ROWS)
    ncols = max(len(metrics) for _, metrics in _ROWS)

    fig = plt.figure(figsize=_FIGURE_SIZE)
    rect = (0.055, 0.07, 0.82, 0.88)
    axes = grid_of_axes(fig, nrows, ncols, rect, hgap=0.04, wgap=0.025)

    for ri, (row_label, metrics) in enumerate(_ROWS):
        for ci in range(ncols):
            if ci < len(metrics):
                mk, title = metrics[ci]
                ylim = _metric_bounds(data, mk)
                _plot_line_panel(
                    axes[ri][ci], ds_order, data, mk, ylim, title,
                    show_xticklabels=(ri == nrows - 1),
                    show_ylabel=(ci == 0),
                    row_label=row_label,
                )
            else:
                axes[ri][ci].set_visible(False)

    # Legend on the right margin — 12 configs need two columns
    active_configs = [c for c in _CONFIGS if any(c in data[ds] for ds in datasets)]
    legend_handles = [
        Line2D(
            [0], [0],
            color=_CONFIG_COLORS[c],
            linestyle=get_line_style(c),
            linewidth=get_line_width(c),
            marker=_MARKERS[i % len(_MARKERS)],
            markersize=5.0,
            markeredgecolor="white",
            markeredgewidth=0.4,
            label=get_legend_name(c),
        )
        for i, c in enumerate(active_configs)
    ]
    fig.legend(
        handles=legend_handles,
        loc="center right",
        fontsize=_FS_LEGEND,
        frameon=False,
        ncol=1,
        bbox_to_anchor=(0.99, 0.5),
        handlelength=2.0,
        handletextpad=0.4,
        labelspacing=0.4,
    )

    outpath = outdir / "fig3_fm_metric_lines.png"
    issues = save_figure(
        fig, outpath, vcd_label="fm_metric_lines", vcd_verbose=True,
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
