#!/usr/bin/env python
"""
MoCoO Figure 4 — Training Dynamics & Convergence Analysis
==========================================================
Layout (17 × 21 cm):
  Row 0 (A): Training loss convergence curves (all 6 configs).
             Dual panel: train loss (left) + val loss (right).
  Row 1 (B): Resource efficiency scatter — ARI vs train_time_s (bubble size =
             peak_mem_gb). Pareto frontier labelled.

Note: Validation metric evolution panels (ARI/NMI/ASW over epochs) are only
drawn when val_scores data is available in the benchmark NPZ file.
If val_scores is empty, the figure uses a compact 2-row layout.

Usage:
    python benchmarks/scripts/plotting/plot_training_dynamics.py
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
from vcd import detect_all_conflicts
from benchmarks.scripts.plotting.shared import setup_fonts, load_benchmark_npz, load_config_metrics, export_subpanels, panel_label, add_config_legend_footnote

# ── Import centralized style ────────────────────────────────────────────────
from mocoo.visualization.style import (
    FIG_WIDTH_IN as FIG_W, FIG_HEIGHT_IN as FIG_H, DPI,
    FS_LABEL, FS_TITLE, FS_AXIS, FS_TICK, FS_LEGEND as FS_LEG, FS_SMALL,
    get_config_colors, get_config_order, get_short_name, get_line_style,
    get_line_width, apply_style, get_tick_name, get_legend_name,
)

apply_style()

# ── Fonts ──────────────────────────────────────────────────────────────────
setup_fonts()

# ── Style constants from centralized module ──────────────────────────────────
_CONFIGS = get_config_order()
_CONFIG_COLOR = get_config_colors()
_XSHORT = {c: get_tick_name(c) for c in _CONFIGS}


def _load_data(rdir: Path):
    data = load_benchmark_npz(rdir)
    configs = data["configs"]
    train_losses = data.get("train_losses", [])
    val_losses = data.get("val_losses", [])
    val_scores = data.get("val_scores", [])
    metrics = load_config_metrics(rdir, configs)
    return configs, train_losses, val_losses, val_scores, metrics


def _has_val_scores(val_scores):
    """Check whether val_scores contains any actual data."""
    for vs in val_scores:
        if vs.ndim == 2 and vs.shape[0] > 0 and vs.shape[1] > 0:
            return True
    return False


# ── Panel A: Loss curves ───────────────────────────────────────────────────

def _draw_loss_curves(axes, fig, configs, train_losses, val_losses):
    _max_ep = max((len(tl) for tl in train_losses), default=50)
    ax_train = axes[0]
    ax_val   = axes[1]

    for i, cfg in enumerate(configs):
        tl = train_losses[i]
        vl = val_losses[i]
        ep_t = np.arange(len(tl))
        val_epochs = np.linspace(0, len(tl)-1, len(vl)).astype(int)

        c  = _CONFIG_COLOR[cfg]
        ls = get_line_style(cfg)
        lw = get_line_width(cfg)

        ax_train.plot(ep_t, tl,  color=c, ls=ls, lw=lw, alpha=0.85, label=cfg)
        ax_val.plot(val_epochs, vl, color=c, ls=ls, lw=lw, alpha=0.85, label=cfg)

    for ax, title, ylabel in [
        (ax_train, "Training Loss", "ELBO Loss"),
        (ax_val,   "Validation Loss",  "Val. ELBO Loss"),
    ]:
        ax.set_title(title, fontsize=FS_TITLE, pad=1)
        ax.set_xlabel("Epoch", fontsize=FS_AXIS)
        ax.set_ylabel(ylabel, fontsize=FS_AXIS)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(alpha=0.22, linestyle="--", linewidth=0.4)
        ax.set_xlim(0, _max_ep * 1.02)
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5, integer=True, prune="both"))

    return ax_train


# ── Panel: Validation metric evolution (only used when data exists) ──────

def _draw_val_metric_evolution(axes, fig, configs, val_losses, val_scores):
    """val_scores: (n_configs, n_checkpoints, 6) -> 0=ARI,1=NMI,2=ASW,3=CAL,4=DAV,5=COR"""
    _max_ep = max((len(vl) for vl in val_losses), default=50)
    score_defs = [
        (0, "Val ARI \u2191", True),
        (1, "Val NMI \u2191", True),
        (2, "Val ASW \u2191", True),
    ]
    ax_first = None
    for j, (si, title, higher) in enumerate(score_defs):
        ax = axes[j]
        if j == 0:
            ax_first = ax
        for i, cfg in enumerate(configs):
            vs = val_scores[i]
            if vs.ndim == 2 and vs.shape[1] > si:
                curve = vs[:, si]
                epochs = np.linspace(0, _max_ep - 1, len(curve)).astype(int)
                ax.plot(epochs, curve,
                        color=_CONFIG_COLOR[cfg], ls=get_line_style(cfg),
                        lw=get_line_width(cfg), alpha=0.85, label=cfg)
        ax.set_title(title, fontsize=FS_AXIS, pad=1)
        if j == 0:
            ax.set_ylabel("Score", fontsize=FS_AXIS)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(alpha=0.22, linestyle="--", linewidth=0.4)
        ax.set_xlim(0, _max_ep * 1.02)
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5, integer=True, prune="both"))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4, prune="both"))
        ylo, yhi = ax.get_ylim()
        ax.set_ylim(max(-0.01, ylo), yhi)
    return ax_first


# ── Panel: Efficiency scatter ──────────────────────────────────────────────

def _draw_efficiency(ax, fig, configs, metrics):
    """Scatter: ARI vs train_time_s — bubble = peak_mem_gb."""
    ari   = np.array([metrics[c].get("ARI",    0) for c in configs])
    time_ = np.array([metrics[c].get("train_time_s", 1) for c in configs])
    mem   = np.array([metrics[c].get("peak_mem_gb",  0.05) for c in configs])
    colors = [_CONFIG_COLOR[c] for c in configs]

    # Bubble size proportional to memory
    sizes = ((mem - mem.min()) / (mem.max() - mem.min() + 1e-6) + 0.2) * 280

    ax.scatter(time_, ari, s=sizes, c=colors, alpha=0.85,
               edgecolors="black", linewidths=0.6, zorder=3)

    # Smart label placement: detect clusters and offset accordingly
    # Sort by time to identify spatial groups
    pts = list(zip(time_, ari, configs, colors))
    placed = []  # list of (x, y) of placed label anchors

    for i, (t, a, cfg, col) in enumerate(pts):
        short = _XSHORT.get(cfg, cfg)
        # Default offset: right and slightly up
        dx, dy = 8, 8

        # Check if this point is close to any already-placed label
        for px, py in placed:
            if abs(t - px) < 15 and abs(a - py) < 0.015:
                # Collision: alternate placement
                dy = -14
                break

        # Special handling for known tight clusters at high time values
        if t > 60:
            # These are the ODE configs clustered together
            if "Full" in cfg:
                dx, dy = -20, 12
            elif "ODE+MoCo" in cfg:
                dx, dy = 6, -14
            elif "ODE" in cfg and "MoCo" not in cfg:
                dx, dy = 6, 8

        # Prevent labels from overlapping axes edges
        ymin, ymax = ax.get_ylim() if ax.get_ylim()[0] != ax.get_ylim()[1] else (0, 1)
        yrange = ymax - ymin
        if a - ymin < 0.12 * yrange and dy < 0:
            dy = 8  # push label above when near bottom axis
        if ymax - a < 0.12 * yrange and dy > 0:
            dy = -10  # push label below when near top axis

        ax.annotate(short, (t, a),
                    textcoords="offset points", xytext=(dx, dy),
                    fontsize=FS_SMALL, color="black",
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.4, alpha=0.5)
                    if abs(dx) > 5 or abs(dy) > 10 else None)
        placed.append((t, a))

    # Pareto frontier
    order = np.argsort(time_)
    pareto_x, pareto_y = [time_[order[0]]], [ari[order[0]]]
    best_ari = pareto_y[0]
    for idx in order[1:]:
        if ari[idx] > best_ari:
            pareto_x.append(time_[idx])
            pareto_y.append(ari[idx])
            best_ari = ari[idx]
    if len(pareto_x) > 1:
        ax.plot(pareto_x, pareto_y, "k--", lw=0.8, alpha=0.5,
                zorder=2)

    ax.set_xlabel("Training Time (s)", fontsize=FS_AXIS)
    ax.set_ylabel("ARI \u2191", fontsize=FS_AXIS, labelpad=1)
    ax.set_title("Efficiency",
                 fontsize=FS_AXIS, pad=1)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(alpha=0.22, linestyle="--", linewidth=0.4)
    ax.margins(0.08)
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    ax.set_xlim(max(0, xlo), xhi)
    ax.set_ylim(max(-0.01, ylo), yhi)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5, prune="both"))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4, prune="both"))

    # Memory legend (minimal)
    ax.legend([], [], frameon=False)
    return ax


# ── Main ───────────────────────────────────────────────────────────────────

def build_figure(rdir: Path, outdir: Path):
    configs, train_losses, val_losses, val_scores, metrics = _load_data(rdir)

    has_scores = _has_val_scores(val_scores)

    if has_scores:
        # Full layout: loss curves + val metrics + efficiency
        fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
        # Row A: 2 loss panels — wider gap for tick label clearance
        _aw_A = (0.86 - 0.12) / 2  # 0.37
        axes_A = [
            fig.add_axes([0.10, 0.72, _aw_A, 0.20]),
            fig.add_axes([0.10 + _aw_A + 0.12, 0.72, _aw_A, 0.20]),
        ]
        # Row B: 3 val-metric panels — wider gap from row A
        _aw_B = (0.86 - 0.05 * 2) / 3  # ~0.2533
        axes_B = [
            fig.add_axes([0.10, 0.40, _aw_B, 0.22]),
            fig.add_axes([0.10 + _aw_B + 0.05, 0.40, _aw_B, 0.22]),
            fig.add_axes([0.10 + 2 * (_aw_B + 0.05), 0.40, _aw_B, 0.22]),
        ]
        # Row C: single efficiency panel — lift above footer
        ax_C = fig.add_axes([0.14, 0.08, 0.82, 0.22])

        print("  Drawing Panel A (Loss convergence)...")
        ax_A = _draw_loss_curves(axes_A, fig, configs, train_losses, val_losses)
        print("  Drawing Panel B (ARI/NMI/ASW evolution)...")
        ax_B = _draw_val_metric_evolution(axes_B, fig, configs, val_losses, val_scores)
        print("  Drawing Panel C (Efficiency)...")
        ax_C = _draw_efficiency(ax_C, fig, configs, metrics)

        add_config_legend_footnote(fig, y_pos=0.005)

        handles, labels = ax_A.get_legend_handles_labels()
        fig.legend(
            handles, labels,
            fontsize=FS_LEG, frameon=False, ncol=len(handles),
            loc="upper center", bbox_to_anchor=(0.5, 0.995),
            handlelength=1.0, labelspacing=0.2, columnspacing=0.6,
            borderpad=0.3,
        )

        panel_label(fig, ax_A, "A", x_off=-0.08, y_off=0.036)
        panel_label(fig, ax_B, "B", x_off=-0.08, y_off=0.036)
        panel_label(fig, ax_C, "C", x_off=-0.10, y_off=0.040)

        sub_panels = [(ax_A, "panelA_train_loss"),
                      (ax_B, "panelB_val_metrics"),
                      (ax_C, "panelC_efficiency")]
    else:
        # Compact layout: loss curves + efficiency (no val_scores data)
        print("  Note: val_scores is empty, using compact 2-row layout.")
        fig_h = FIG_W * 1.0  # balanced 2-panel layout
        fig = plt.figure(figsize=(FIG_W, fig_h), dpi=DPI)
        # Row A: 2 loss panels — explicit per-subplot geometry
        _aw_A2 = (0.86 - 0.10) / 2  # 0.38
        axes_A = [
            fig.add_axes([0.10, 0.54, _aw_A2, 0.38]),
            fig.add_axes([0.10 + _aw_A2 + 0.10, 0.54, _aw_A2, 0.38]),
        ]
        # Row B: single efficiency panel
        ax_B = fig.add_axes([0.10, 0.08, 0.86, 0.38])

        print("  Drawing Panel A (Loss convergence)...")
        ax_A = _draw_loss_curves(axes_A, fig, configs, train_losses, val_losses)

        print("  Drawing Panel B (Efficiency)...")
        ax_B = _draw_efficiency(ax_B, fig, configs, metrics)

        add_config_legend_footnote(fig, y_pos=0.005)

        handles, labels = ax_A.get_legend_handles_labels()
        fig.legend(
            handles, labels,
            fontsize=FS_LEG, frameon=False, ncol=len(handles),
            loc="upper center", bbox_to_anchor=(0.5, 0.995),
            handlelength=1.0, labelspacing=0.2, columnspacing=0.6,
            borderpad=0.3,
        )

        panel_label(fig, ax_A, "A", x_off=-0.08, y_off=0.040)
        panel_label(fig, ax_B, "B", x_off=-0.08, y_off=0.040)

        sub_panels = [(ax_A, "panelA_train_loss"),
                      (ax_B, "panelB_efficiency")]

    print("\n-- Conflict Detection --")
    issues = detect_all_conflicts(fig, label="training_dynamics", verbose=True)

    outpath = outdir / "fig4_training_dynamics.png"
    from mocoo.visualization.style import save_figure
    save_figure(fig, outpath)

    # Export individual panel sub-figures
    sub_dir = outdir / "fig4_training_dynamics"
    sub_dir.mkdir(parents=True, exist_ok=True)
    export_subpanels(fig, sub_dir, sub_panels)
    plt.close(fig)

    n_warn = sum(1 for x in issues if x.get("severity") == "warning")
    n_err  = sum(1 for x in issues if x.get("severity") == "error")
    print(f"\nSaved -> {outpath}")
    print(f"{n_warn} warnings | {n_err} errors")
    return issues


def main():
    _benchmarks = Path(__file__).resolve().parent.parent.parent  # benchmarks/
    p = argparse.ArgumentParser()
    p.add_argument("--resultsdir",
                   default=str(_benchmarks / "results" / "single_dataset"))
    p.add_argument("--outdir",
                   default=str(_benchmarks / "figures"))
    args = p.parse_args()
    rdir   = Path(args.resultsdir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    return build_figure(rdir, outdir)


if __name__ == "__main__":
    main()
