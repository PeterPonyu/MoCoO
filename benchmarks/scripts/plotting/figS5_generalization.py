#!/usr/bin/env python
"""MoCoO Supplementary — Cross-dataset evaluation & generalization.

Four-panel figure using the full proposed metric set:
  (A) Proposed-metric heatmap (config x dataset, column-normalised)
  (B) Clustering (ARI) vs embedding quality (DREX overall) scatter
  (C) Proposed-quality heatmap (DRE/LSE/DREX/LSEX per config x dataset)
  (D) Radar profile across all 8 proposed metrics
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from benchmarks.scripts.plotting.shared import setup_fonts
from mocoo.visualization.style import (
    FS_AXIS, FS_LEGEND, FS_SMALL, FS_TICK, FS_TITLE,
    FIG_WIDTH_IN, DPI,
    PROPOSED_METRICS, PROPOSED_CLUSTERING, PROPOSED_QUALITY,
    PROPOSED_SHORT_LABELS, PROPOSED_DIRECTION,
    apply_style, save_figure, add_panel_label,
    get_config_colors, get_display_name, get_short_name,
    get_base_config_order,
)
from mocoo.visualization.direct_layout import bind_figure_region

setup_fonts()
apply_style()


def _load_expanded(results_dir: Path):
    """Load summary_expanded.csv from each dataset into a unified DataFrame."""
    # Exclude known non-dataset directories
    _EXCLUDE = {
        "baselines", "fm_sensitivity", "multiseed", "beta_ablation",
        "pseudotime_validation", "trajectory_baselines",
    }
    rows = []
    for ds in sorted(os.listdir(results_dir)):
        if ds in _EXCLUDE:
            continue
        fp = results_dir / ds / "summary_expanded.csv"
        if not fp.exists():
            continue
        df = pd.read_csv(fp)
        df["dataset"] = ds
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def make_figure(results_dir: Path, out_path: Path):
    df = _load_expanded(results_dir)
    if df.empty:
        print("No cross-dataset data found.")
        return

    # Filter to whole split only
    df = df[df["split"] == "whole"].copy()

    config_colors = get_config_colors()
    base_cfgs = get_base_config_order()
    datasets = sorted(df["dataset"].unique())

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * 1.05))
    root = bind_figure_region(fig, (0.12, 0.06, 0.88, 0.92))
    (r_top, r_bot) = root.split_rows([1, 1], gap=0.12)
    (r_a, r_b) = r_top.split_cols([3, 2], gap=0.12)
    (r_c, r_d) = r_bot.split_cols([3, 2], gap=0.12)

    # ── Panel (A): Clustering metrics heatmap (config x dataset) ──
    ax_a = r_a.add_axes(fig)
    # Average of proposed clustering metrics (normalised per metric, then averaged)
    n_cfgs = len(base_cfgs)
    n_ds = len(datasets)
    mat = np.full((n_cfgs, n_ds), np.nan)
    for i, cfg in enumerate(base_cfgs):
        for j, ds in enumerate(datasets):
            sel = df[(df["config"] == cfg) & (df["dataset"] == ds)]
            if sel.empty:
                continue
            score_parts = []
            for m in PROPOSED_CLUSTERING:
                if m in sel.columns:
                    v = sel[m].values[0]
                    if not PROPOSED_DIRECTION.get(m, True):
                        v = -v  # flip DAV so higher is better for averaging
                    score_parts.append(v)
            if score_parts:
                mat[i, j] = np.mean(score_parts)

    # Column-normalise for display
    disp_mat = mat.copy()
    for j in range(n_ds):
        col = disp_mat[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) > 1:
            mn, mx = valid.min(), valid.max()
            if mx - mn > 1e-12:
                disp_mat[:, j] = (col - mn) / (mx - mn)
    im = ax_a.imshow(disp_mat, aspect="auto", cmap="YlOrRd")
    ax_a.set_xticks(range(n_ds))
    ax_a.set_xticklabels([d[:5] for d in datasets], fontsize=FS_SMALL - 2,
                         rotation=90, ha="center")
    ax_a.set_yticks(range(n_cfgs))
    ax_a.set_yticklabels([get_short_name(c) for c in base_cfgs], fontsize=FS_SMALL)
    ax_a.set_title("Clustering Score (col-norm)", fontsize=FS_TITLE)
    add_panel_label(ax_a, "A", x=-0.20, y=1.06)

    # ── Panel (B): ARI vs DREX scatter ──
    ax_b = r_b.add_axes(fig)
    for cfg in base_cfgs:
        sel = df[df["config"] == cfg]
        if sel.empty or "DREX_overall_quality" not in sel.columns:
            continue
        ari_mean = sel["ARI"].mean()
        drex_mean = sel["DREX_overall_quality"].mean()
        color = config_colors.get(cfg, "#888888")
        ax_b.scatter(drex_mean, ari_mean,
                     s=60, c=color, edgecolors="black", linewidth=0.5,
                     zorder=3, label=get_short_name(cfg))
    ax_b.set_xlabel("DREX Overall", fontsize=FS_AXIS)
    ax_b.set_ylabel("ARI", fontsize=FS_AXIS)
    ax_b.set_title("Clustering vs Embedding", fontsize=FS_TITLE)
    ax_b.legend(fontsize=FS_SMALL - 1, loc="lower right", markerscale=0.7)
    add_panel_label(ax_b, "B", x=-0.30, y=1.10)

    # ── Panel (C): Quality metrics heatmap (config x dataset) ──
    ax_c = r_c.add_axes(fig)
    mat_q = np.full((n_cfgs, n_ds), np.nan)
    for i, cfg in enumerate(base_cfgs):
        for j, ds in enumerate(datasets):
            sel = df[(df["config"] == cfg) & (df["dataset"] == ds)]
            if sel.empty:
                continue
            score_parts = []
            for m in PROPOSED_QUALITY:
                if m in sel.columns:
                    score_parts.append(sel[m].values[0])
            if score_parts:
                mat_q[i, j] = np.mean(score_parts)

    im2 = ax_c.imshow(mat_q, aspect="auto", cmap="YlGnBu")
    ax_c.set_xticks(range(n_ds))
    ax_c.set_xticklabels([d[:5] for d in datasets], fontsize=FS_SMALL - 2,
                         rotation=90, ha="center")
    ax_c.set_yticks(range(n_cfgs))
    ax_c.set_yticklabels([get_short_name(c) for c in base_cfgs], fontsize=FS_SMALL)
    ax_c.set_title("Quality Score (DRE/LSE/DREX/LSEX avg)", fontsize=FS_TITLE)
    add_panel_label(ax_c, "C", x=-0.20, y=1.06)

    # ── Panel (D): Radar profile across all 8 proposed metrics ──
    ax_d = r_d.add_axes(fig, polar=True)
    N = len(PROPOSED_METRICS)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles.append(angles[0])

    for cfg in base_cfgs:
        sel = df[df["config"] == cfg]
        if sel.empty:
            continue
        vals = []
        for m in PROPOSED_METRICS:
            if m in sel.columns:
                v = sel[m].mean()
                # Normalise DAV: flip and rescale to [0,1] range
                if not PROPOSED_DIRECTION.get(m, True):
                    # DAV typically 0.5-3.0; map so lower→higher
                    v = max(0, 1.0 - v / 3.0)
                vals.append(v)
            else:
                vals.append(0)
        vals.append(vals[0])
        color = config_colors.get(cfg, "#888888")
        ax_d.plot(angles, vals, linewidth=1.2, color=color,
                  label=get_short_name(cfg))
        ax_d.fill(angles, vals, alpha=0.06, color=color)

    labels = [PROPOSED_SHORT_LABELS[m] for m in PROPOSED_METRICS]
    ax_d.set_xticks(angles[:-1])
    ax_d.set_xticklabels(labels, fontsize=FS_SMALL)
    ax_d.set_title("Config Profile", fontsize=FS_TITLE, pad=14)
    ax_d.legend(fontsize=FS_SMALL - 1, loc="lower right",
                bbox_to_anchor=(1.30, -0.15))
    add_panel_label(ax_d, "D", x=-0.22, y=1.06)

    save_figure(fig, str(out_path), vcd_label="figS5_generalization",
                vcd_verbose=True)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="MoCoO: Cross-Dataset Generalization (Proposed Metrics)")
    parser.add_argument("--resultsdir", type=Path,
                        default=Path(__file__).resolve().parent.parent.parent
                        / "results")
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()
    outdir = args.outdir or (args.resultsdir.parent / "figures")
    Path(outdir).mkdir(parents=True, exist_ok=True)
    make_figure(args.resultsdir, Path(outdir) / "figS5_generalization.png")


if __name__ == "__main__":
    main()
