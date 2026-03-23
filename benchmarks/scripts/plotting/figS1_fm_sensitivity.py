#!/usr/bin/env python
"""MoCoO Supplementary Figure S1 — FM hyperparameter sensitivity analysis.

Panel layout: 8 rows × 5 columns (fm_t_start, fm_epochs, fm_lr,
fm_hidden_dim, fm_steps).  Each subplot shows thin grey lines per dataset
and a bold coloured mean line with \u00b1 1 s.d. shading.  A vertical dashed
line marks the default value.

Reads: benchmarks/results/fm_sensitivity/sensitivity.csv
Writes: benchmarks/figures/figS1_fm_sensitivity.{png,pdf}
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
    FS_AXIS,
    FS_LEGEND,
    FS_TITLE,
    FS_TICK,
    apply_style,
    grid_of_axes,
    save_figure,
)

setup_fonts()
apply_style()

# ── Layout constants ──────────────────────────────────────────────────────
_METRICS = [
    "NMI", "ARI", "ASW", "DAV",
    "DRE_umap_overall_quality", "LSE_overall_quality",
    "DREX_overall_quality", "LSEX_overall_quality",
]
_METRIC_LABELS = {
    "NMI": "NMI \u2191", "ARI": "ARI \u2191", "ASW": "ASW \u2191",
    "DAV": "DAV \u2193",
    "DRE_umap_overall_quality": "DRE \u2191",
    "LSE_overall_quality": "LSE \u2191",
    "DREX_overall_quality": "DREX \u2191",
    "LSEX_overall_quality": "LSEX \u2191",
}

_PARAM_ORDER = ["fm_t_start", "fm_epochs", "fm_lr", "fm_hidden_dim", "fm_steps"]
_PARAM_LABELS = {
    "fm_t_start": "$t_{\\mathrm{start}}$",
    "fm_epochs": "Epochs",
    "fm_lr": "Learning rate (log)",
    "fm_hidden_dim": "Hidden dim",
    "fm_steps": "Euler steps",
}
_PARAM_DEFAULTS = {
    "fm_t_start": 0.9,
    "fm_epochs": 200,
    "fm_lr": 1e-3,
    "fm_hidden_dim": 128,
    "fm_steps": 100,
}
# Parameters that need log-scale x-axis
_LOG_PARAMS = {"fm_lr"}

_MEAN_COLOR = "#D55E00"   # vermilion
_SHADE_COLOR = "#D55E00"
_BASELINE_COLOR = "#0072B2"  # blue for contrast
_LINE_ALPHA = 0.18
_SHADE_ALPHA = 0.18

_FIGURE_SIZE = (18.0, 18.0)
_FS_TITLE = FS_TITLE + 3
_FS_AXIS = FS_AXIS + 2
_FS_TICK = FS_TICK + 2
_FS_LEGEND = FS_LEGEND + 2


def _load_sensitivity(csv_path: Path) -> dict:
    """Load sensitivity CSV → nested dict[param][dataset] = {values, metrics}.

    Returns
    -------
    data : dict
        data[param][dataset] = {"values": [...], metric: [...], ...}
    """
    data: dict = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            param = row["param"]
            ds = row["dataset"]
            if param not in data:
                data[param] = {}
            if ds not in data[param]:
                entry = {"values": []}
                for m in _METRICS:
                    entry[m] = []
                data[param][ds] = entry
            data[param][ds]["values"].append(float(row["value"]))
            for m in _METRICS:
                data[param][ds][m].append(float(row.get(m, 0)))
    return data


def _load_baselines(results_dir: Path) -> dict:
    """Load VAE+ODE (no FM) 'whole'-split metrics as baselines.

    Returns
    -------
    baselines : dict[metric] = {"mean": float, "per_ds": dict[ds, float]}
    """
    per_ds: dict[str, dict[str, float]] = {}
    for ds_dir in sorted(results_dir.iterdir()):
        csv_path = ds_dir / "summary_expanded.csv"
        if not csv_path.exists():
            continue
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if row["config"] == "VAE+ODE" and row["split"] == "whole":
                    per_ds[ds_dir.name] = {
                        m: float(row.get(m, "nan")) for m in _METRICS
                    }
                    break
    baselines: dict[str, dict] = {}
    for m in _METRICS:
        vals = [per_ds[ds][m] for ds in per_ds if np.isfinite(per_ds[ds].get(m, np.nan))]
        baselines[m] = {
            "mean": float(np.mean(vals)) if vals else np.nan,
            "per_ds": {ds: per_ds[ds][m] for ds in per_ds},
        }
    return baselines


def main(csv_path: Path, outdir: Path, results_dir: Path | None = None):
    data = _load_sensitivity(csv_path)

    # Load baselines from main benchmark results
    if results_dir is None:
        results_dir = csv_path.parent.parent
    baselines = _load_baselines(results_dir)

    fig = plt.figure(figsize=_FIGURE_SIZE)

    nrows, ncols = len(_METRICS), len(_PARAM_ORDER)
    rect = (0.08, 0.07, 0.87, 0.84)
    axes = grid_of_axes(fig, nrows, ncols, rect, hgap=0.04, wgap=0.055)

    for ri, metric in enumerate(_METRICS):
        for ci, param in enumerate(_PARAM_ORDER):
            ax = axes[ri][ci]
            pdata = data.get(param, {})

            # Collect all dataset curves
            all_values = None
            all_curves = []
            for ds, dsd in pdata.items():
                xvals = np.array(dsd["values"])
                yvals = np.array(dsd[metric])
                # Sort by x
                order = np.argsort(xvals)
                xvals = xvals[order]
                yvals = yvals[order]
                if param == "fm_t_start":
                    baseline_val = baselines[metric]["per_ds"].get(ds, np.nan)
                    if np.isfinite(baseline_val):
                        xvals = np.append(xvals, 1.0)
                        yvals = np.append(yvals, baseline_val)
                all_curves.append(yvals)
                if all_values is None:
                    all_values = xvals
                elif param == "fm_t_start" and len(xvals) > len(all_values):
                    all_values = xvals
                # Plot individual dataset line
                if param in _LOG_PARAMS:
                    ax.semilogx(xvals, yvals, color="0.5", alpha=_LINE_ALPHA,
                                linewidth=0.8, zorder=1)
                else:
                    ax.plot(xvals, yvals, color="0.5", alpha=_LINE_ALPHA,
                            linewidth=0.8, zorder=1)

            if all_values is not None and len(all_curves) > 0:
                mat = np.array(all_curves)
                mean = mat.mean(axis=0)
                std = mat.std(axis=0)

                if param in _LOG_PARAMS:
                    ax.semilogx(all_values, mean, color=_MEAN_COLOR,
                                linewidth=2.2, zorder=3)
                else:
                    ax.plot(all_values, mean, color=_MEAN_COLOR,
                            linewidth=2.2, zorder=3)
                ax.fill_between(all_values, mean - std, mean + std,
                                color=_SHADE_COLOR, alpha=_SHADE_ALPHA, zorder=2)

                # Default value line
                default = _PARAM_DEFAULTS[param]
                ax.axvline(default, color="0.3", linestyle="--",
                           linewidth=0.8, alpha=0.6, zorder=1)

            # Baseline horizontal line (VAE+ODE without FM)
            bl_mean = baselines[metric]["mean"]
            if np.isfinite(bl_mean):
                ax.axhline(bl_mean, color=_BASELINE_COLOR, linestyle=":",
                           linewidth=1.4, alpha=0.7, zorder=2)

            # Labels
            if ri == 0:
                ax.set_title(_PARAM_LABELS[param], fontsize=_FS_TITLE)
            if ri == nrows - 1:
                ax.set_xlabel(_PARAM_LABELS[param], fontsize=_FS_AXIS)
            else:
                ax.set_xticklabels([])
            if ci == 0:
                ax.set_ylabel(_METRIC_LABELS[metric], fontsize=_FS_AXIS)
            ax.tick_params(labelsize=_FS_TICK)
            ax.yaxis.set_major_locator(plt.MaxNLocator(4, prune="both"))
            ax.set_ylim(bottom=0)

            # Tighten log-scale lr ticks to avoid cross-axes bleed
            if param in _LOG_PARAMS:
                from matplotlib.ticker import LogLocator, NullFormatter, FixedLocator, FixedFormatter
                ax.set_xlim(5e-5, 2e-2)
                if ri == nrows - 1:
                    ax.xaxis.set_major_locator(FixedLocator([1e-4, 1e-3, 1e-2]))
                    ax.xaxis.set_major_formatter(FixedFormatter(["$10^{-4}$", "$10^{-3}$", "$10^{-2}$"]))
                else:
                    ax.xaxis.set_major_locator(FixedLocator([]))
                    ax.xaxis.set_major_formatter(NullFormatter())
                ax.xaxis.set_minor_locator(LogLocator(subs=[], numticks=1))
                ax.xaxis.set_minor_formatter(NullFormatter())
            elif param == "fm_t_start":
                ticks = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0]
                ax.set_xlim(0.08, 1.01)
                ax.set_xticks(ticks)
                if ri == nrows - 1:
                    ax.set_xticklabels(["0.1", "0.3", "0.5", "0.7", "0.9", "0.99", "1.0"])
            elif param == "fm_epochs":
                ticks = [25, 50, 100, 200, 400]
                ax.set_xlim(15, 410)
                ax.set_xticks(ticks)
            elif param == "fm_hidden_dim":
                ticks = [32, 64, 128, 256, 512]
                ax.set_xlim(20, 530)
                ax.set_xticks(ticks)
            elif param == "fm_steps":
                ticks = [5, 10, 25, 50, 100, 200]
                ax.set_xlim(0, 220)
                ax.set_xticks(ticks)

            if ri != nrows - 1 and param != "fm_lr":
                ax.tick_params(axis="x", labelbottom=False)

    # Panel letters — placed as figure-level text to avoid tick overlap
    for ri, metric in enumerate(_METRICS):
        ax0 = axes[ri][0]
        pos = ax0.get_position()
        letter = chr(ord("a") + ri)
        fig.text(pos.x0 - 0.04, pos.y1 + 0.015, f"({letter})",
                 fontsize=_FS_TITLE + 2, fontweight="bold", va="bottom")

    # Figure-level legend for line types
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=_MEAN_COLOR, linewidth=2.2, label="FM mean ± s.d."),
        Line2D([0], [0], color=_BASELINE_COLOR, linewidth=1.4,
             linestyle=":", label="VAE+ODE baseline (= $t_{start}=1.0$)"),
        Line2D([0], [0], color="0.3", linewidth=0.8,
               linestyle="--", label="Default value"),
    ]
    fig.legend(handles=legend_handles, loc="upper center",
             ncol=3, fontsize=_FS_LEGEND, frameon=False,
               bbox_to_anchor=(0.5, 0.98))

    outpath = outdir / "figS1_fm_sensitivity.png"
    issues = save_figure(fig, outpath, vcd_label="figS1_fm_sensitivity",
                         vcd_verbose=True)
    n_err = sum(1 for i in issues if i.get("severity") == "error")
    plt.close(fig)

    print(f"\n\u2713 figS1_fm_sensitivity saved to {outpath}")
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
    results_dir = Path(args.resultsdir)
    csv_path = results_dir / "fm_sensitivity" / "sensitivity.csv"
    if not csv_path.exists():
        csv_path = results_dir / "sensitivity.csv"
    sys.exit(main(csv_path, Path(args.outdir), results_dir=results_dir))
