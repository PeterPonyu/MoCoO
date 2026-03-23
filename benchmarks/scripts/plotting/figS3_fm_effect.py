#!/usr/bin/env python
"""MoCoO Supplementary Figure S3 — FM refinement effect across model variants.

Shows the per-metric delta (FM – base) for each of the 6 base model
configurations, aggregated across all 10 datasets.  Each dot is one
dataset; the bar shows the mean delta.  Positive = FM improved.
For DAV lower is better, so the delta sign is flipped.

Panels: 4 rows × 2 columns of 8 proposed metrics.

Reads:  benchmarks/results/<dataset>/summary_expanded.csv
Writes: benchmarks/figures/figS3_fm_effect.{png,pdf}
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
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from benchmarks.scripts.plotting.shared import setup_fonts
from mocoo.visualization.style import (
    ACCENT_POSITIVE,
    ACCENT_NEGATIVE,
    HIGHLIGHT_CONFIGS,
    HIGHLIGHT_EDGE_WIDTH,
    FS_AXIS,
    FS_LEGEND,
    FS_TITLE,
    FS_TICK,
    apply_style,
    get_base_config_order,
    get_config_colors,
    get_tick_name,
    grid_of_axes,
    save_figure,
)

setup_fonts()
apply_style()

# ── Constants ─────────────────────────────────────────────────────────────
_BASE_CONFIGS = get_base_config_order()
_FM_PAIRS = [(c, f"{c}+FM") for c in _BASE_CONFIGS]
_CONFIG_COLORS = get_config_colors()

_METRICS = [
    ("NMI",                       "NMI \u2191"),
    ("ARI",                       "ARI \u2191"),
    ("ASW",                       "ASW \u2191"),
    ("DAV",                       "DAV \u2193"),
    ("DRE_umap_overall_quality",  "DRE \u2191"),
    ("LSE_overall_quality",       "LSE \u2191"),
    ("DREX_overall_quality",      "DREX \u2191"),
    ("LSEX_overall_quality",      "LSEX \u2191"),
]

# DAV is lower-is-better: negate delta so positive = improved
_LOWER_IS_BETTER = {"DAV"}

_DATASET_ORDER = [
    "endo", "setty", "paul", "IRALL", "hemato", "dentate",
    "spinoids", "astrocyte", "lung", "retina", "teeth", "spine",
    "hepatoblastoma", "brainmet", "breast", "gastric",
    "livercancer", "melanoma", "pituitary", "hESCtime",
]

_NCOLS = 2
_NROWS = 4   # ceil(8 / 2)
_FIGURE_SIZE = (14.0, 11.0)
_FS_TITLE_L = FS_TITLE + 2
_FS_AXIS_L = FS_AXIS
_FS_TICK_L = FS_TICK - 1


def _safe_float(v: str) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _load_deltas(results_dir: Path) -> dict:
    """Compute delta = FM - base for every (base_config, metric, dataset).

    Returns
    -------
    deltas : dict[metric_key][base_config] = list[float]  (one per dataset)
    """
    metric_keys = [mk for mk, _ in _METRICS]
    deltas: dict[str, dict[str, list]] = {mk: {c: [] for c in _BASE_CONFIGS}
                                           for mk in metric_keys}

    for ds in _DATASET_ORDER:
        csv_path = results_dir / ds / "summary_expanded.csv"
        if not csv_path.exists():
            continue
        rows: dict[str, dict[str, float]] = {}
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                if row["split"] != "whole":
                    continue
                rows[row["config"]] = {
                    mk: _safe_float(row.get(mk)) for mk in metric_keys
                }

        for base, fm in _FM_PAIRS:
            if base not in rows or fm not in rows:
                continue
            for mk in metric_keys:
                v_base = rows[base].get(mk, np.nan)
                v_fm = rows[fm].get(mk, np.nan)
                if np.isfinite(v_base) and np.isfinite(v_fm):
                    d = v_fm - v_base
                    if mk in _LOWER_IS_BETTER:
                        d = -d  # flip so positive = improved
                    deltas[mk][base].append(d)

    return deltas


def main(results_dir: Path, outdir: Path) -> int:
    deltas = _load_deltas(results_dir)

    n_metrics = len(_METRICS)
    n_configs = len(_BASE_CONFIGS)

    fig = plt.figure(figsize=_FIGURE_SIZE)
    rect = (0.08, 0.10, 0.88, 0.82)
    axes = grid_of_axes(fig, _NROWS, _NCOLS, rect, hgap=0.04, wgap=0.10)

    xs = np.arange(n_configs)
    bar_width = 0.55

    for idx, (mk, label) in enumerate(_METRICS):
        ri, ci = divmod(idx, _NCOLS)
        ax = axes[ri][ci]

        means = []
        all_pts = []
        for cfg_i, cfg in enumerate(_BASE_CONFIGS):
            pts = np.array(deltas[mk][cfg])
            mean_val = np.nanmean(pts) if len(pts) > 0 else 0.0
            means.append(mean_val)
            all_pts.append(pts)

        means = np.array(means)
        colors = [ACCENT_POSITIVE if m >= 0 else ACCENT_NEGATIVE for m in means]
        edge_colors = [
            _CONFIG_COLORS.get(cfg, "0.4") if cfg in HIGHLIGHT_CONFIGS else "white"
            for cfg in _BASE_CONFIGS
        ]
        edge_widths = [
            HIGHLIGHT_EDGE_WIDTH if cfg in HIGHLIGHT_CONFIGS else 0.5
            for cfg in _BASE_CONFIGS
        ]

        bars = ax.bar(xs, means, width=bar_width, color=colors, alpha=0.75,
               edgecolor=edge_colors, linewidth=edge_widths, zorder=3)

        # Scatter individual dataset points
        for cfg_i, pts in enumerate(all_pts):
            if len(pts) == 0:
                continue
            jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(pts))
            ax.scatter(
                xs[cfg_i] + jitter, pts,
                color=_CONFIG_COLORS.get(_BASE_CONFIGS[cfg_i], "0.4"),
                s=14, alpha=0.55, edgecolors="white", linewidths=0.3,
                zorder=5,
            )

        ax.axhline(0, color="0.3", linewidth=0.6, zorder=2)
        ax.set_ylabel(f"$\\Delta$ {label}", fontsize=_FS_AXIS_L)
        ax.tick_params(axis="both", labelsize=_FS_TICK_L, pad=1)
        ax.yaxis.set_major_locator(plt.MaxNLocator(3, symmetric=True, prune="both"))
        ax.grid(axis="y", alpha=0.2, linestyle="--", linewidth=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # x-tick labels on every subplot
        ax.set_xticks(xs)
        tick_labels = ax.set_xticklabels(
            [get_tick_name(c) for c in _BASE_CONFIGS],
            fontsize=_FS_TICK_L, rotation=35, ha="right",
        )
        for tl, cfg in zip(tick_labels, _BASE_CONFIGS):
            if cfg in HIGHLIGHT_CONFIGS:
                tl.set_fontweight("bold")

        # Panel letter
        letter = chr(ord("a") + idx)
        ax.text(-0.12, 1.05, f"({letter})", transform=ax.transAxes,
                fontsize=_FS_TITLE_L + 1, fontweight="bold", va="bottom")

    fig.suptitle(
        "Flow Matching refinement effect (FM $-$ base)",
        fontsize=_FS_TITLE_L + 2, fontweight="bold", y=0.965,
    )

    outpath = outdir / "figS3_fm_effect.png"
    issues = save_figure(fig, outpath, vcd_label="figS3_fm_effect", vcd_verbose=True)
    n_err = sum(1 for i in issues if i.get("severity") == "error")
    plt.close(fig)

    print(f"\n\u2713 figS3_fm_effect saved to {outpath}")
    print(f"  VCD: {len(issues)} issues, {n_err} errors")
    return n_err


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resultsdir",
        default=str(Path(__file__).resolve().parent.parent.parent / "results"),
    )
    parser.add_argument(
        "--outdir",
        default=str(Path(__file__).resolve().parent.parent.parent / "figures"),
    )
    args = parser.parse_args()
    sys.exit(main(Path(args.resultsdir), Path(args.outdir)))
