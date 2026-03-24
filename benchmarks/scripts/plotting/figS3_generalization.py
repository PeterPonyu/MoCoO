#!/usr/bin/env python
"""MoCoO Supplementary -- Cross-dataset evaluation & generalization.

Two-panel figure:
  (A) All-8-metric heatmap (config x dataset, column-normalised, DAV flipped)
  (B) Radar profile across all 8 proposed metrics
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
    PROPOSED_METRICS, PROPOSED_DIRECTION,
    PROPOSED_SHORT_LABELS,
    HIGHLIGHT_CONFIGS, HEATMAP_DARK_THRESHOLD,
    FMT_SCORE_SHORT,
    apply_style, save_figure, add_panel_label,
    get_config_colors, get_short_name,
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

    n_cfgs = len(base_cfgs)
    n_ds = len(datasets)
    n_metrics = len(PROPOSED_METRICS)

    # Slightly wider than tall for a 1-row, 2-col layout
    fig = plt.figure(figsize=(FIG_WIDTH_IN * 1.15, FIG_WIDTH_IN * 0.55))
    root = bind_figure_region(fig, (0.10, 0.10, 0.92, 0.90))
    (r_a, r_b) = root.split_cols([3, 2], gap=0.10)

    # ── Panel (A): All-8-metric heatmap (config x dataset, col-normalised) ──
    ax_a = r_a.add_axes(fig)

    # Build matrix: rows = configs, cols = datasets * metrics (but we want
    # a single averaged score per config x dataset).  Average across all 8
    # proposed metrics after flipping DAV.
    mat = np.full((n_cfgs, n_ds), np.nan)
    for i, cfg in enumerate(base_cfgs):
        for j, ds in enumerate(datasets):
            sel = df[(df["config"] == cfg) & (df["dataset"] == ds)]
            if sel.empty:
                continue
            score_parts = []
            for m in PROPOSED_METRICS:
                if m in sel.columns:
                    v = sel[m].values[0]
                    if not PROPOSED_DIRECTION.get(m, True):
                        # DAV: lower is better; flip so higher = better
                        v = max(0, 1.0 - v / 3.0)
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

    # Annotate cells with raw values
    for i in range(n_cfgs):
        for j in range(n_ds):
            v = disp_mat[i, j]
            if np.isnan(v):
                continue
            txt_color = "white" if v > HEATMAP_DARK_THRESHOLD else "black"
            ax_a.text(j, i, f"{v:{FMT_SCORE_SHORT}}",
                      ha="center", va="center",
                      fontsize=FS_SMALL - 1, color=txt_color)

    ax_a.set_xticks(range(n_ds))
    ax_a.set_xticklabels([d[:6] for d in datasets], fontsize=FS_SMALL,
                         rotation=45, ha="right")
    ax_a.set_yticks(range(n_cfgs))
    ax_a.set_yticklabels([get_short_name(c) for c in base_cfgs],
                         fontsize=FS_SMALL)
    ax_a.set_title("Composite Score (8 metrics, col-norm)", fontsize=FS_TITLE)
    add_panel_label(ax_a, "A", x=-0.22, y=1.08)

    # ── Panel (B): Radar profile across all 8 proposed metrics ──
    ax_b = r_b.add_axes(fig, polar=True)
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
                    # DAV typically 0.5-3.0; map so lower -> higher
                    v = max(0, 1.0 - v / 3.0)
                vals.append(v)
            else:
                vals.append(0)
        vals.append(vals[0])
        color = config_colors.get(cfg, "#888888")
        is_highlight = cfg in HIGHLIGHT_CONFIGS
        lw = 2.2 if is_highlight else 1.2
        fill_alpha = 0.15 if is_highlight else 0.04
        ax_b.plot(angles, vals, linewidth=lw, color=color,
                  label=get_short_name(cfg))
        ax_b.fill(angles, vals, alpha=fill_alpha, color=color)

    labels = [PROPOSED_SHORT_LABELS[m] for m in PROPOSED_METRICS]
    ax_b.set_xticks(angles[:-1])
    ax_b.set_xticklabels(labels, fontsize=FS_SMALL)
    # Fix label overlap: adjust horizontal alignment per angular position
    for lbl, angle in zip(ax_b.get_xticklabels(), angles[:-1]):
        deg = np.degrees(angle) % 360
        if deg < 10 or deg > 350:
            lbl.set_ha("center")
        elif deg < 180:
            lbl.set_ha("left")
        else:
            lbl.set_ha("right")
    ax_b.tick_params(axis="x", pad=10)  # extra radial padding for labels
    ax_b.set_title("Config Profile", fontsize=FS_TITLE, pad=14)
    ax_b.legend(fontsize=FS_SMALL - 1, loc="lower right",
                bbox_to_anchor=(1.30, -0.15))
    add_panel_label(ax_b, "B", x=-0.22, y=1.08)

    save_figure(fig, str(out_path), vcd_label="figS3_generalization",
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
    make_figure(args.resultsdir, Path(outdir) / "figS3_generalization.png")


if __name__ == "__main__":
    main()
