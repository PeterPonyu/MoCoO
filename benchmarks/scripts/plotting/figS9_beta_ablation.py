#!/usr/bin/env python
"""MoCoO Figure — Beta (KL weight) ablation heatmap.

Single heatmap showing the full proposed metric set across three beta values
(0.01, 0.1, 1.0) for each base model configuration.
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from benchmarks.scripts.plotting.shared import setup_fonts
from mocoo.visualization.style import (
    FS_AXIS, FS_SMALL, FS_TICK, FS_TITLE,
    HEATMAP_CMAP, HEATMAP_DARK_THRESHOLD,
    FIG_WIDTH_IN, DPI,
    HIGHLIGHT_CONFIGS,
    PROPOSED_METRICS, PROPOSED_SHORT_LABELS, PROPOSED_DIRECTION,
    apply_style, save_figure, add_panel_label,
    get_base_config_order, get_tick_name,
)
from mocoo.visualization.direct_layout import bind_figure_region

setup_fonts()
apply_style()


def _load_beta_ablation(results_dir: Path):
    """Load beta ablation results from summary_expanded.csv for all betas."""
    betas = ["0.01", "0.1", "1.0"]
    configs = get_base_config_order()

    labels_y = []
    matrix = []
    row_configs = []

    for beta in betas:
        beta_dir = results_dir / f"beta_{beta}"
        if not beta_dir.exists():
            beta_dir = results_dir / "beta_ablation" / f"beta_{beta}"
        # Prefer summary_expanded.csv for full metric set
        csv_path = beta_dir / "summary_expanded.csv"
        if not csv_path.exists():
            csv_path = beta_dir / "summary.csv"
        if not csv_path.exists():
            continue

        cfg_data = {}
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                cfg = row.get("config", "").strip()
                cfg_data[cfg] = row

        for cfg in configs:
            if cfg not in cfg_data:
                continue
            vals = []
            for m in PROPOSED_METRICS:
                try:
                    vals.append(float(cfg_data[cfg].get(m, 0)))
                except ValueError:
                    vals.append(0.0)
            matrix.append(vals)
            labels_y.append(f"{get_tick_name(cfg)} (\u03b2={beta})")
            row_configs.append(cfg)

    x_labels = [PROPOSED_SHORT_LABELS[m] for m in PROPOSED_METRICS]
    return np.array(matrix) if matrix else None, labels_y, x_labels, row_configs


def make_figure(results_dir: Path, out_path: Path):
    matrix, y_labels, x_labels, row_configs = _load_beta_ablation(results_dir)
    if matrix is None or matrix.size == 0:
        print("No beta ablation data found.")
        return

    # Column-normalise for display (each metric has different scale)
    norm_matrix = np.zeros_like(matrix)
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        lo, hi = col.min(), col.max()
        rng = hi - lo if hi > lo else 1.0
        norm_val = (col - lo) / rng
        # Flip DAV (lower is better)
        if not PROPOSED_DIRECTION.get(PROPOSED_METRICS[j], True):
            norm_val = 1.0 - norm_val
        norm_matrix[:, j] = norm_val

    fig = plt.figure(figsize=(FIG_WIDTH_IN * 0.85, FIG_WIDTH_IN * 0.85))
    root = bind_figure_region(fig, (0.18, 0.08, 0.95, 0.88))
    ax = root.add_axes(fig)

    im = ax.imshow(norm_matrix, aspect="auto", cmap=HEATMAP_CMAP)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=FS_TICK, rotation=45, ha="right")
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=FS_SMALL)
    ax.set_title("Beta Ablation \u2014 Proposed Metrics", fontsize=FS_TITLE)

    # Cell annotations (show raw values)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            norm_val = norm_matrix[i, j]
            color = "white" if norm_val > HEATMAP_DARK_THRESHOLD else "black"
            raw = matrix[i, j]
            fmt = f"{raw:.2f}" if abs(raw) < 10 else f"{raw:.1f}"
            ax.text(j, i, fmt, ha="center", va="center",
                    fontsize=FS_SMALL - 1, color=color)

    add_panel_label(ax, "a", x=-0.22)

    # Bold y-tick labels for MoCoO (Full) rows
    for tl, cfg in zip(ax.get_yticklabels(), row_configs):
        if cfg in HIGHLIGHT_CONFIGS:
            tl.set_fontweight("bold")

    save_figure(fig, str(out_path), vcd_label="figS9_beta_ablation", vcd_verbose=True)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="MoCoO: Beta Ablation (Proposed Metrics)")
    parser.add_argument("--resultsdir", type=Path,
                        default=Path(__file__).resolve().parent.parent.parent
                        / "results")
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()
    outdir = args.outdir or (args.resultsdir.parent / "figures")
    Path(outdir).mkdir(parents=True, exist_ok=True)
    make_figure(args.resultsdir, Path(outdir) / "figS9_beta_ablation.png")


if __name__ == "__main__":
    main()
