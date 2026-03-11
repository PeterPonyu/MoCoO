#!/usr/bin/env python
"""
MoCoO Figure 6 — Beta Sensitivity Analysis
============================================
3×3 grid of line charts showing how each aggregate metric varies
across β ∈ {0.01, 0.1, 1.0} for all 6 configurations.

Usage:
    python -m benchmarks.scripts.plotting.plot_beta_sensitivity
"""
from __future__ import annotations

import argparse
import json
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

from benchmarks.scripts.plotting.shared import setup_fonts, panel_label, add_config_legend_footnote
from benchmarks.scripts.pipeline.visual_conflict_detector import detect_all_conflicts
from mocoo.visualization.style import (
    FIG_WIDTH_IN, FIG_HEIGHT_IN, DPI,
    FS_LABEL, FS_TITLE, FS_AXIS, FS_TICK, FS_LEGEND, FS_SMALL,
    apply_style, get_config_order, get_config_colors, get_short_name, get_tick_name,
    get_line_style, get_line_width,
)

setup_fonts()
apply_style()

_CONFIGS = get_config_order()
_COLORS = get_config_colors()
_BETAS = [0.01, 0.1, 1.0]

_METRICS = [
    ("ARI", "ARI ↑", True),
    ("NMI", "NMI ↑", True),
    ("ASW", "ASW ↑", True),
    ("DRE_umap_overall_quality", "DRE (UMAP) ↑", True),
    ("DRE_tsne_overall_quality", "DRE (tSNE) ↑", True),
    ("DREX_overall_quality", "DREX ↑", True),
    ("LSE_overall_quality", "LSE ↑", True),
    ("LSEX_overall_quality", "LSEX ↑", True),
    ("COR", "COR ↑", True),
]


def _load_all_betas(results_base: Path) -> dict:
    """Load metrics across all beta values. Returns {beta: {config: dict}}."""
    all_data = {}
    for beta in _BETAS:
        beta_dir = results_base / "beta_ablation" / f"beta_{beta}"
        if not beta_dir.exists():
            continue
        beta_data = {}
        for cfg in _CONFIGS:
            key = cfg.replace("+", "_")
            jf = beta_dir / f"{key}.json"
            if jf.exists():
                with open(jf) as f:
                    beta_data[cfg] = json.load(f)
        if beta_data:
            all_data[beta] = beta_data
    return all_data


def build_figure(results_base: Path, outdir: Path):
    all_data = _load_all_betas(results_base)
    if not all_data:
        print("No beta ablation data found.")
        return []

    betas_present = sorted(all_data.keys())

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN * 0.80))
    # 3×3 grid — wider gutters, more footer room
    _cw = (0.86 - 0.06 * 2) / 3   # ~0.2467
    _rh = (0.78 - 0.06 * 2) / 3   # ~0.2200
    axes = np.array([
        [fig.add_axes([0.10 + c * (_cw + 0.06),
                       0.16 + 0.78 - (r + 1) * _rh - r * 0.06,
                       _cw, _rh])
         for c in range(3)]
        for r in range(3)
    ])

    for idx, (metric_key, metric_label, higher_better) in enumerate(_METRICS):
        row, col = divmod(idx, 3)
        ax = axes[row, col]

        for cfg in _CONFIGS:
            vals = []
            betas_for_cfg = []
            for beta in betas_present:
                if cfg in all_data[beta]:
                    v = all_data[beta][cfg].get(metric_key, np.nan)
                    if v is not None and np.isfinite(v):
                        vals.append(v)
                        betas_for_cfg.append(beta)

            if vals:
                ax.plot(betas_for_cfg, vals,
                        color=_COLORS[cfg],
                        linestyle=get_line_style(cfg),
                        linewidth=get_line_width(cfg),
                        marker="o", markersize=3,
                        label=get_short_name(cfg))

        ax.set_xscale("log")
        ax.set_xticks(_BETAS)
        # Only show x-tick labels on the bottom row to reduce clutter
        if row == 2:
            ax.set_xticklabels([str(b) for b in _BETAS], fontsize=FS_TICK)
        else:
            ax.set_xticklabels([])
        ax.tick_params(axis="both", labelsize=FS_TICK)
        ax.set_title(metric_label, fontsize=FS_TITLE, pad=3)
        ax.grid(alpha=0.22, linestyle="--", linewidth=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if col == 0:
            ax.set_ylabel("Score", fontsize=FS_AXIS)
        if row == 2:
            ax.set_xlabel("β", fontsize=FS_AXIS)

    # Single legend above the grid (outside the matrix)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=FS_LEGEND, ncol=6,
               loc="upper center", bbox_to_anchor=(0.50, 0.98),
               frameon=False, handlelength=1.5, columnspacing=1.0)

    add_config_legend_footnote(fig, y_pos=0.012)

    # Panel labels — only on leftmost column, shifted outward
    letters = "ABC"
    for idx in range(min(3, len(_METRICS))):
        row = idx
        panel_label(fig, axes[row, 0], letters[row],
                   x_off=-0.07, y_off=0.012)

    outpath = outdir / "fig6_beta_sensitivity.png"

    print("\n── Conflict Detection ──")
    issues = detect_all_conflicts(fig, label="beta_sensitivity", verbose=True)
    n_warn = sum(1 for i in issues if i.get("severity") == "warning")
    n_err = sum(1 for i in issues if i.get("severity") == "error")

    from mocoo.visualization.style import save_figure
    save_figure(fig, outpath)
    plt.close(fig)
    print(f"Saved: {outpath}")
    print(f"{n_warn} warnings | {n_err} errors")
    return issues


def main():
    _benchmarks = Path(__file__).resolve().parent.parent.parent
    p = argparse.ArgumentParser()
    p.add_argument("--resultsdir", default=str(_benchmarks / "results"))
    p.add_argument("--outdir", default=str(_benchmarks / "figures"))
    args = p.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    build_figure(Path(args.resultsdir), outdir)


if __name__ == "__main__":
    main()
