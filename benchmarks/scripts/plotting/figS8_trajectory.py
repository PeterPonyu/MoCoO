#!/usr/bin/env python
"""MoCoO Supplementary Figure S8 — Trajectory and pseudotime analysis.

Two-panel figure:
  (a)  Pseudotime: Spearman |ρ| per ODE-containing config (bar chart)
  (b)  Trajectory baselines: MoCoO vs DPT vs Palantir (grouped bars)
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
    FS_AXIS, FS_LEGEND, FS_SMALL, FS_TICK, FS_TITLE,
    FIG_WIDTH_IN, DPI,
    apply_style, save_figure, add_panel_label,
    get_config_colors, get_tick_name,
)
from mocoo.visualization.direct_layout import bind_figure_region

setup_fonts()
apply_style()

_TRAJ_COLORS = {
    "DPT": "#66A61E",
    "Palantir": "#7570B3",
}


def _load_pseudotime(results_dir: Path):
    """Load pseudotime validation CSV → per-config mean |ρ|."""
    fp = results_dir / "pseudotime_validation" / "pseudotime_validation.csv"
    if not fp.exists():
        return {}
    acc = {}
    with open(fp) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cfg = row["config"].strip()
            try:
                rho = abs(float(row["spearman_rho"]))
            except (KeyError, ValueError):
                continue
            acc.setdefault(cfg, []).append(rho)
    return {c: np.mean(v) for c, v in acc.items()}


def _load_trajectory(results_dir: Path):
    """Load trajectory baselines CSV → per-method mean |ρ| across datasets."""
    fp = results_dir / "trajectory_baselines" / "trajectory_baselines.csv"
    if not fp.exists():
        return {}
    acc = {}
    with open(fp) as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["method"].strip()
            try:
                rho = float(row["spearman_abs"])
            except (KeyError, ValueError):
                continue
            acc.setdefault(method, []).append(rho)
    return {m: np.mean(v) for m, v in acc.items()}


def make_figure(results_dir: Path, out_path: Path):
    pseudo = _load_pseudotime(results_dir)
    traj = _load_trajectory(results_dir)
    if not pseudo and not traj:
        print("No trajectory data found.")
        return

    config_colors = get_config_colors()

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * 0.50))
    root = bind_figure_region(fig, (0.10, 0.14, 0.90, 0.84))
    (r_a, r_b) = root.split_cols([1, 1], gap=0.10)

    # --- Panel (a): Pseudotime Spearman |ρ| per config ---
    ax_a = r_a.add_axes(fig)
    if pseudo:
        configs = sorted(pseudo.keys())
        x = np.arange(len(configs))
        vals = [pseudo[c] for c in configs]
        colors = [config_colors.get(c, "#888888") for c in configs]
        ax_a.bar(x, vals, 0.6, color=colors, zorder=3)
        ax_a.set_xticks(x)
        ax_a.set_xticklabels([get_tick_name(c) for c in configs],
                             fontsize=FS_SMALL, rotation=45, ha="right")
        ax_a.set_ylabel("Mean |Spearman ρ|", fontsize=FS_AXIS)
        ax_a.set_title("Pseudotime Correlation", fontsize=FS_TITLE)
        # Annotate values
        ymax = ax_a.get_ylim()[1]
        for i, v in enumerate(vals):
            if v + 0.005 < ymax * 0.92:
                ax_a.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom",
                          fontsize=FS_SMALL)
            else:
                # Place inside bar for tallest bars
                ax_a.text(i, v - 0.008, f"{v:.3f}", ha="center", va="top",
                          fontsize=FS_SMALL, color="white", fontweight="bold")
    add_panel_label(ax_a, "a", x=-0.18, y=1.12)

    # --- Panel (b): Trajectory baselines comparison ---
    ax_b = r_b.add_axes(fig)
    if traj:
        methods = sorted(traj.keys())
        x = np.arange(len(methods))
        vals = [traj[m] for m in methods]
        colors = []
        for m in methods:
            if m in _TRAJ_COLORS:
                colors.append(_TRAJ_COLORS[m])
            elif m.startswith("MoCoO_"):
                cfg = m.replace("MoCoO_", "")
                colors.append(config_colors.get(cfg, "#D55E00"))
            else:
                colors.append("#888888")
        ax_b.barh(x, vals, 0.6, color=colors, zorder=3)
        ax_b.set_yticks(x)
        labels = [m.replace("MoCoO_", "") for m in methods]
        ax_b.set_yticklabels(labels, fontsize=FS_TICK)
        ax_b.set_xlabel("Mean |Spearman ρ|", fontsize=FS_AXIS)
        ax_b.set_title("Trajectory Methods", fontsize=FS_TITLE)
        # Annotate
        for i, v in enumerate(vals):
            ax_b.text(v + 0.01, i, f"{v:.3f}", ha="left", va="center",
                      fontsize=FS_SMALL)
    add_panel_label(ax_b, "b")

    save_figure(fig, str(out_path), vcd_label="figS8_trajectory", vcd_verbose=True)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="MoCoO Fig S8: Trajectory")
    parser.add_argument("--resultsdir", type=Path,
                        default=Path(__file__).resolve().parent.parent.parent
                        / "results")
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()
    outdir = args.outdir or (args.resultsdir.parent / "figures")
    Path(outdir).mkdir(parents=True, exist_ok=True)
    make_figure(args.resultsdir, Path(outdir) / "figS8_trajectory.png")


if __name__ == "__main__":
    main()
