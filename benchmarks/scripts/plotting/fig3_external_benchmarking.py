#!/usr/bin/env python
"""MoCoO Figure 3 — External Benchmarking (unified, 5-metric).

Comprehensive comparison of MoCoO (Full+FM) against external baselines on
all 5 proposed metrics: ASW, DAV, CAL, DRE, DREX.

Layout: 1 row x 5 metric subplots (matching Fig 2 boxplot style).
Each panel shows only methods that have data for that metric, so the
clustering panels (ASW/DAV/CAL) may show 19 methods while DRE/DREX panels
show fewer if some DL methods lack embedding-quality data.

Reads (in priority order, first-seen wins):
  benchmarks/results/baselines/extended_baselines_v2.csv  (with DRE/DREX)
  benchmarks/results/baselines/external_baselines.csv     (original 5 external)
  benchmarks/results/baselines/extended_baselines.csv     (original ML+DL)
  benchmarks/results/<dataset>/summary_expanded.csv       (MoCoO Full+FM)
Writes:
  benchmarks/figures/fig3_external_benchmarking.{png,pdf}
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
from matplotlib.patches import Patch

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from benchmarks.scripts.plotting.shared import setup_fonts
from mocoo.visualization.style import (
    FS_AXIS, FS_LEGEND, FS_SMALL, FS_TICK, FS_TITLE,
    FIG_WIDTH_IN,
    apply_style, save_figure,
)

setup_fonts()
apply_style()

# ── Method ordering & colours ─────────────────────────────────────────────

METHOD_GROUPS = {
    "Classical ML": ["PCA", "ICA", "NMF", "FA", "TruncatedSVD"],
    "Trad. Tools": ["DPT", "PCA+KMeans", "Harmony"],
    "Deep Learning": [
        "scVI", "scANVI", "CellBLAST", "CLEAR", "GM-VAE",
        "SCALEX", "scDAC", "scDeepCluster", "scSMD", "siVAE",
    ],
    "Ours": ["MoCoO"],
}
METHOD_ORDER = [m for methods in METHOD_GROUPS.values() for m in methods]

_GROUP_PALETTES = {
    "Classical ML": ["#56B4E9", "#4A9BD9", "#3E82C9", "#326AB9", "#2651A9"],
    "Trad. Tools": ["#009E73", "#00B880", "#00D28D"],
    "Deep Learning": [
        "#7570B3", "#8B7EC0", "#A18CCD", "#B79ADA",
        "#CC79A7", "#D68AB5", "#E09BC3",
        "#EAA0C8", "#C27091", "#9A4060",
    ],
    "Ours": ["#D55E00"],
}
METHOD_COLORS = {}
for _group, _methods in METHOD_GROUPS.items():
    _palette = _GROUP_PALETTES[_group]
    for _i, _m in enumerate(_methods):
        METHOD_COLORS[_m] = _palette[_i % len(_palette)]

# All 5 proposed metrics: (ext_csv_col, internal_col, higher_is_better, label)
ALL_METRICS = [
    ("ASW",  "ASW",                       True,  "ASW \u2191"),
    ("DB",   "DAV",                       False, "DAV \u2193"),
    ("CH",   "CAL",                       True,  "CAL \u2191"),
    ("DRE",  "DRE_umap_overall_quality",  True,  "DRE \u2191"),
    ("DREX", "DREX_overall_quality",      True,  "DREX \u2191"),
]

ALL_DATASETS = ["IRALL", "paul", "endo", "dentate", "hemato", "lung", "spinoids"]


# ── Data loading ──────────────────────────────────────────────────────────

def _load_external_csv(fpath: Path, metrics, methods_filter=None):
    """Load a baselines CSV -> {method: {dataset: {ext_col: [values]}}}."""
    if not fpath.exists():
        return {}
    acc = {}
    with open(fpath) as f:
        for row in csv.DictReader(f):
            method = row.get("method", "").strip()
            ds = row.get("dataset", "").strip()
            if methods_filter and method not in methods_filter:
                continue
            if ds not in ALL_DATASETS:
                continue
            bucket = acc.setdefault(method, {}).setdefault(ds, {})
            for ext_col, _int_col, _higher, _lbl in metrics:
                try:
                    v = float(row[ext_col])
                    if np.isfinite(v):
                        bucket.setdefault(ext_col, []).append(v)
                except (KeyError, ValueError):
                    pass
    return acc


def _load_mocoo(results_dir: Path, metrics):
    """Load Full+FM whole-split metrics -> {dataset: {int_col: value}}."""
    data = {}
    for ds in ALL_DATASETS:
        fp = results_dir / ds / "summary_expanded.csv"
        if not fp.exists():
            continue
        with open(fp) as f:
            for row in csv.DictReader(f):
                if row.get("config", "").strip() != "Full+FM":
                    continue
                if row.get("split", "").strip() != "whole":
                    continue
                entry = {}
                for _ext_col, int_col, _higher, _lbl in metrics:
                    try:
                        entry[int_col] = float(row[int_col])
                    except (KeyError, ValueError):
                        pass
                if entry:
                    data[ds] = entry
    return data


def _build_unified_data(results_dir: Path, metrics):
    """Merge all sources -> {method: {metric_label: [per-dataset value]}}."""
    baselines_dir = results_dir / "baselines"

    v2 = _load_external_csv(baselines_dir / "extended_baselines_v2.csv", metrics)
    mocoo = _load_mocoo(results_dir, metrics)

    unified = {}
    seen_pairs = set()

    for method, ds_dict in v2.items():
        if method not in METHOD_ORDER:
            continue
        method_data = unified.setdefault(
            method, {lbl: [] for _, _, _, lbl in metrics})
        for ds in ALL_DATASETS:
            if ds not in ds_dict:
                continue
            if (method, ds) in seen_pairs:
                continue
            seen_pairs.add((method, ds))
            for ext_col, _int_col, _higher, lbl in metrics:
                vals = ds_dict[ds].get(ext_col, [])
                if vals:
                    method_data[lbl].append(float(np.mean(vals)))

    mocoo_data = {lbl: [] for _, _, _, lbl in metrics}
    for ds in ALL_DATASETS:
        if ds not in mocoo:
            continue
        for _ext_col, int_col, _higher, lbl in metrics:
            v = mocoo[ds].get(int_col)
            if v is not None and np.isfinite(v):
                mocoo_data[lbl].append(v)
    unified["MoCoO"] = mocoo_data

    return unified


def _method_group(method):
    for group, methods in METHOD_GROUPS.items():
        if method in methods:
            return group
    return "Other"


def _detect_available_metrics(results_dir: Path):
    """Check if DRE/DREX columns exist in any CSV."""
    baselines_dir = results_dir / "baselines"
    for fname in ["extended_baselines_v2.csv", "extended_baselines.csv",
                   "external_baselines.csv"]:
        fp = baselines_dir / fname
        if not fp.exists():
            continue
        with open(fp) as f:
            reader = csv.DictReader(f)
            headers = set(reader.fieldnames or [])
            if "DRE" in headers and "DREX" in headers:
                return ALL_METRICS
    return [m for m in ALL_METRICS if m[0] in ("ASW", "DB", "CH")]


def _render_panel(ax, panel_methods, data, title, _int_col, rng):
    """Render a single metric panel with per-panel method filtering."""
    n = len(panel_methods)
    positions = np.arange(n)
    box_data = []
    colors = []

    for m in panel_methods:
        vals = [v for v in data[m].get(title, []) if np.isfinite(v)]
        box_data.append(vals if vals else [np.nan])
        colors.append(METHOD_COLORS.get(m, "#999999"))

    bp = ax.boxplot(
        box_data, positions=positions, widths=0.55,
        patch_artist=True, showfliers=False,
        medianprops=dict(color="white", linewidth=1.2),
        whiskerprops=dict(linewidth=0.6),
        capprops=dict(linewidth=0.6),
        boxprops=dict(linewidth=0.7),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)
        patch.set_edgecolor("#444444")

    # Scatter overlay
    for pos, vals_raw, color in zip(positions, box_data, colors):
        vals = [v for v in vals_raw if np.isfinite(v)]
        if vals:
            jitter = rng.uniform(-0.15, 0.15, len(vals))
            ax.scatter(pos + jitter, vals, color=color, s=14,
                       alpha=0.55, edgecolors="white", linewidths=0.3,
                       zorder=5)

    # Highlight MoCoO — bold edge
    if "MoCoO" in panel_methods:
        idx = panel_methods.index("MoCoO")
        bp["boxes"][idx].set_edgecolor("#D55E00")
        bp["boxes"][idx].set_linewidth(1.6)

    # Group separators
    prev_group = None
    for i, m in enumerate(panel_methods):
        g = _method_group(m)
        if prev_group and g != prev_group:
            ax.axvline(i - 0.5, color="0.80", linewidth=0.6, zorder=1,
                       linestyle="--")
        prev_group = g

    # Formatting
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [m.replace("+", "\n+") for m in panel_methods],
        fontsize=FS_SMALL - 0.5, rotation=55, ha="right",
    )
    ax.tick_params(axis="y", labelsize=FS_TICK)

    if _int_col == "CAL":
        ax.set_yscale("log")
    else:
        ax.yaxis.set_major_locator(plt.MaxNLocator(5, prune="both"))

    ax.set_title(title, fontsize=FS_TITLE, pad=4)
    ax.grid(axis="y", alpha=0.18, linestyle="--", linewidth=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Main ──────────────────────────────────────────────────────────────────

def main(results_dir: Path, outdir: Path) -> int:
    metrics = _detect_available_metrics(results_dir)
    n_metrics = len(metrics)
    print(f"Detected {n_metrics} metrics: {[m[3] for m in metrics]}")

    data = _build_unified_data(results_dir, metrics)

    # Global active methods (any metric)
    all_active = [m for m in METHOD_ORDER if m in data
                  and any(len(data[m][lbl]) > 0 for _, _, _, lbl in metrics)]
    if not all_active:
        print("No data found. Run benchmarks first.")
        return 1

    # Per-metric active methods: only methods with data for that metric
    per_metric_methods = {}
    for _ext_col, _int_col, _higher, lbl in metrics:
        per_metric_methods[lbl] = [
            m for m in METHOD_ORDER if m in data
            and len([v for v in data[m].get(lbl, []) if np.isfinite(v)]) > 0
        ]

    print(f"Methods per metric:")
    for _, _, _, lbl in metrics:
        pm = per_metric_methods[lbl]
        print(f"  {lbl}: {len(pm)} methods")

    # Figure layout: proportional widths based on method count per panel
    fig_w = FIG_WIDTH_IN * 2.4
    fig_h = FIG_WIDTH_IN * 0.88

    fig = plt.figure(figsize=(fig_w, fig_h))

    left_margin = 0.05
    right_margin = 0.98
    bottom = 0.28
    top = 0.84
    h = top - bottom
    gap = 0.025
    total_w = right_margin - left_margin - gap * (n_metrics - 1)

    # Width proportional to number of methods (min 3 to avoid tiny panels)
    raw_widths = [max(len(per_metric_methods[lbl]), 3) for _, _, _, lbl in metrics]
    sum_w = sum(raw_widths)
    col_widths = [total_w * (w / sum_w) for w in raw_widths]

    axes = []
    x = left_margin
    for cw in col_widths:
        axes.append(fig.add_axes([x, bottom, cw, h]))
        x += cw + gap

    rng = np.random.default_rng(42)

    for ax, (_ext_col, _int_col, _higher, title) in zip(axes, metrics):
        panel_methods = per_metric_methods[title]
        _render_panel(ax, panel_methods, data, title, _int_col, rng)

    # ── Legend by group (union of all active methods) ──
    legend_methods = list(dict.fromkeys(
        m for lbl_methods in per_metric_methods.values() for m in lbl_methods
    ))
    legend_handles = []
    for group, methods in METHOD_GROUPS.items():
        added_any = False
        for m in methods:
            if m in legend_methods:
                lw = 1.6 if m == "MoCoO" else 0.7
                ec = "#D55E00" if m == "MoCoO" else "#444444"
                legend_handles.append(
                    Patch(facecolor=METHOD_COLORS[m], edgecolor=ec,
                          linewidth=lw, alpha=0.82, label=m))
                added_any = True
        if added_any and group != "Ours":
            legend_handles.append(
                Patch(facecolor="none", edgecolor="none", label=""))

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=min(len(legend_methods) + 3, 11),
        fontsize=FS_LEGEND - 0.5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
        handletextpad=0.3,
        columnspacing=0.6,
        handlelength=1.0,
    )

    # ── Save ──
    outdir.mkdir(parents=True, exist_ok=True)
    stem = "fig3_external_benchmarking"
    save_figure(fig, outdir / f"{stem}.png")
    save_figure(fig, outdir / f"{stem}.pdf")
    print(f"Saved to {outdir / stem}.{{png,pdf}}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Figure 3")
    parser.add_argument(
        "--resultsdir", type=str,
        default=str(Path(__file__).resolve().parent.parent.parent / "results"),
    )
    parser.add_argument(
        "--outdir", type=str,
        default=str(Path(__file__).resolve().parent.parent.parent / "figures"),
    )
    args = parser.parse_args()
    sys.exit(main(Path(args.resultsdir), Path(args.outdir)))
