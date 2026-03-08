#!/usr/bin/env python
"""
MoCoO Figure 4 — Training Dynamics & Convergence Analysis
==========================================================
Layout (17 × 21 cm):
  Row 0 (A): Training loss convergence curves (all 6 configs).
             Dual panel: train loss (left) + val loss (right).
  Row 1 (B): Validation metric evolution — ARI, NMI, ASW over epochs.
             Shows that adding ODE/MoCo improves *rate* of convergence.
  Row 2 (C): Latent space quality metrics during training evolution
             (Val score dims 3-5: CAL, DAV, COR).
  Row 3 (D): Resource efficiency scatter — ARI vs train_time_s (bubble size =
             peak_mem_gb). Pareto frontier labelled. Includes per-component
             breakdown annotation.

Usage:
    python benchmarks/plot_training_dynamics.py
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
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from benchmarks.scripts.pipeline.visual_conflict_detector import detect_all_conflicts

# ── Fonts ──────────────────────────────────────────────────────────────────
_FONT_DIR = Path(__file__).resolve().parent.parent.parent / "fonts"
for _fp in (_FONT_DIR / "Arial.ttf", _FONT_DIR / "Arial Bold.ttf"):
    if _fp.exists():
        fm.fontManager.addfont(str(_fp))
if (_FONT_DIR / "Arial.ttf").exists():
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["Arial"] + list(
        matplotlib.rcParams.get("font.sans-serif", []))

FIG_W = 17 / 2.54
FIG_H = 21 / 2.54
DPI   = 300
FS_LABEL = 9
FS_TITLE = 7
FS_AXIS  = 6
FS_TICK  = 5
FS_LEG   = 4.5
FS_SMALL = 3.8

_CONFIGS  = ["VAE", "VAE+ODE", "VAE+MoCo", "VAE+MoCo+Proto", "VAE+ODE+MoCo", "Full"]
_PALETTE  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
_CONFIG_COLOR = dict(zip(_CONFIGS, _PALETTE))
_CONFIG_LS    = {
    "VAE":          "-",
    "VAE+ODE":      "--",
    "VAE+MoCo":     "-.",
    "VAE+MoCo+Proto": ":",
    "VAE+ODE+MoCo": (0, (3, 1, 1, 1)),
    "Full":         "-",
}
_CONFIG_LW = {c: (1.8 if c == "Full" else 1.1) for c in _CONFIGS}


def _export_subpanels(fig, sub_dir: Path, panels: list) -> None:
    """Save each panel (axes) as a standalone PNG cropped tightly."""
    renderer = fig.canvas.get_renderer()
    for ax, name in panels:
        if ax is None:
            continue
        try:
            bbox = ax.get_tightbbox(renderer)
            if bbox is None:
                continue
            extent = bbox.transformed(fig.dpi_scale_trans.inverted())
            sp = sub_dir / f"{name}.png"
            fig.savefig(sp, dpi=DPI, bbox_inches=extent)
        except Exception as exc:
            print(f"  sub-panel {name}: skipped ({exc})")


def _panel_label(fig, ax, letter):
    pos = ax.get_position()
    fig.text(pos.x0 - 0.042, pos.y1 + 0.006,
             f"({letter})", fontsize=FS_LABEL, fontweight="bold",
             va="bottom", ha="right", clip_on=False)


def _unify_metric_keys(m: dict) -> dict:
    """Normalise JSON metric keys so downstream code uses short names."""
    _MAP = {
        "full_ARI": "ARI", "full_NMI": "NMI", "full_ASW": "ASW",
        "full_CH": "CAL", "full_DB": "DAV", "corr": "COR",
        "CH": "CAL", "DB": "DAV",
    }
    for src, dst in _MAP.items():
        if src in m and dst not in m:
            m[dst] = m[src]
    return m


def _load_data(rdir: Path):
    npz = np.load(rdir / "benchmark_data.npz", allow_pickle=True)
    configs     = [str(c) for c in npz["configs"]]
    train_losses = [np.asarray(x, dtype=np.float32) for x in npz["train_losses"]]
    val_losses   = [np.asarray(x, dtype=np.float32) for x in npz["val_losses"]]
    val_scores   = [np.asarray(x, dtype=np.float32) for x in npz["val_scores"]]
    metrics = {}
    for cfg in configs:
        key = cfg.replace("+", "_")
        jf  = rdir / f"{key}.json"
        if jf.exists():
            with open(jf) as f:
                metrics[cfg] = _unify_metric_keys(json.load(f))
    return configs, train_losses, val_losses, val_scores, metrics


# ── Panel A: Loss curves ───────────────────────────────────────────────────

def _draw_loss_curves(gs, fig, configs, train_losses, val_losses):
    ax_train = fig.add_subplot(gs[0])
    ax_val   = fig.add_subplot(gs[1])

    for i, cfg in enumerate(configs):
        tl = train_losses[i]
        vl = val_losses[i]
        ep_t = np.arange(len(tl))
        # Sample val at every 5 epochs
        val_epochs = np.linspace(0, len(tl)-1, len(vl)).astype(int)

        c  = _CONFIG_COLOR[cfg]
        ls = _CONFIG_LS[cfg]
        lw = _CONFIG_LW[cfg]

        ax_train.plot(ep_t, tl,  color=c, ls=ls, lw=lw, alpha=0.85, label=cfg)
        ax_val.plot(val_epochs, vl, color=c, ls=ls, lw=lw, alpha=0.85, label=cfg)

    for ax, title, ylabel in [
        (ax_train, "Training Loss", "ELBO Loss"),
        (ax_val,   "Validation Loss",  "Val. ELBO Loss"),
    ]:
        ax.set_title(title, fontsize=FS_TITLE, pad=3)
        ax.set_xlabel("Epoch", fontsize=FS_AXIS)
        ax.set_ylabel(ylabel, fontsize=FS_AXIS)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(alpha=0.22, linestyle="--", linewidth=0.4)
        ax.set_xlim(0, 305)
        ax.xaxis.set_major_locator(plt.FixedLocator([0, 50, 100, 150, 200, 250, 300]))

    # legend added at figure level; no per-axis legend

    # Shade epochs >= 100 — NO text annotation to avoid overlaps
    for ax in (ax_train, ax_val):
        ax.axvspan(100, 305, alpha=0.05, color="gray")

    return ax_train


# ── Panel B: Validation metric evolution ──────────────────────────────────

def _draw_val_metric_evolution(gs, fig, configs, val_losses, val_scores):
    """val_scores: (n_configs, n_checkpoints, 6) → 0=ARI,1=NMI,2=ASW,3=CAL,4=DAV,5=COR"""
    score_defs = [
        (0, "Val ARI ↑", True),
        (1, "Val NMI ↑", True),
        (2, "Val ASW ↑", True),
    ]
    ax_first = None
    for j, (si, title, higher) in enumerate(score_defs):
        ax = fig.add_subplot(gs[j])
        if j == 0:
            ax_first = ax
        for i, cfg in enumerate(configs):
            vs = val_scores[i]
            if vs.ndim == 2 and vs.shape[1] > si:
                curve = vs[:, si]
                epochs = np.linspace(0, 299, len(curve)).astype(int)
                ax.plot(epochs, curve,
                        color=_CONFIG_COLOR[cfg], ls=_CONFIG_LS[cfg],
                        lw=_CONFIG_LW[cfg], alpha=0.85, label=cfg)
        ax.set_title(title, fontsize=FS_TITLE, pad=3)
        ax.set_xlabel("Epoch", fontsize=FS_AXIS)
        if j == 0:
            ax.set_ylabel("Score", fontsize=FS_AXIS)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(alpha=0.22, linestyle="--", linewidth=0.4)
        ax.set_xlim(0, 305)
        ax.xaxis.set_major_locator(plt.FixedLocator([0, 50, 100, 150, 200, 250, 300]))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4, prune="upper"))
    return ax_first


# ── Panel C: Advanced metric evolution ────────────────────────────────────

def _draw_advanced_metrics(gs, fig, configs, val_scores):
    """CAL, DAV, COR — dims 3,4,5 of val_scores."""
    score_defs = [
        (3, "Cal.-Harabasz ↑", True),
        (4, "Davies-Bouldin ↓", False),
        (5, "Silh. Correlation ↑", True),
    ]
    ax_first = None
    for j, (si, title, higher) in enumerate(score_defs):
        ax = fig.add_subplot(gs[j])
        if j == 0:
            ax_first = ax
        for i, cfg in enumerate(configs):
            vs = val_scores[i]
            if vs.ndim == 2 and vs.shape[1] > si:
                curve = vs[:, si]
                epochs = np.linspace(0, 299, len(curve)).astype(int)
                ax.plot(epochs, curve,
                        color=_CONFIG_COLOR[cfg], ls=_CONFIG_LS[cfg],
                        lw=_CONFIG_LW[cfg], alpha=0.85, label=cfg)
        ax.set_title(title, fontsize=FS_TITLE, pad=3)
        ax.set_xlabel("Epoch", fontsize=FS_AXIS)
        if j == 0:
            ax.set_ylabel("Score", fontsize=FS_AXIS)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(alpha=0.22, linestyle="--", linewidth=0.4)
        ax.set_xlim(0, 305)
        ax.xaxis.set_major_locator(plt.FixedLocator([0, 50, 100, 150, 200, 250, 300]))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4, prune="upper"))
    return ax_first


# ── Panel D: Efficiency scatter ────────────────────────────────────────────

def _draw_efficiency(gs, fig, configs, metrics):
    """Scatter: ARI vs train_time_s — bubble = peak_mem_gb.
    Shows Pareto frontier (best performance per unit time).
    """
    ax = fig.add_subplot(gs[:])
    ari   = np.array([metrics[c].get("ARI",    0) for c in configs])
    time_ = np.array([metrics[c].get("train_time_s", 1) for c in configs])
    mem   = np.array([metrics[c].get("peak_mem_gb",  0.05) for c in configs])
    colors = [_CONFIG_COLOR[c] for c in configs]

    # Bubble size proportional to memory
    sizes = ((mem - mem.min()) / (mem.max() - mem.min() + 1e-6) + 0.2) * 280

    sc = ax.scatter(time_, ari, s=sizes, c=colors, alpha=0.85,
                    edgecolors="black", linewidths=0.6, zorder=3)

    # Per-point offsets to avoid label collisions for close-ARI configs
    _xyoff = {
        "VAE":           ( 5,  14),
        "VAE+ODE":       ( 5,  4),
        "VAE+MoCo":      ( 5,  4),
        "VAE+MoCo+Proto":( 5,  14),
        "VAE+ODE+MoCo":  ( 5, -10),
        "Full":          ( 5,  4),
    }
    for i, cfg in enumerate(configs):
        short = cfg.replace("VAE+", "V+").replace("VAE", "VAE")
        ax.annotate(short, (time_[i], ari[i]),
                    textcoords="offset points", xytext=_xyoff.get(cfg, (5, 4)),
                    fontsize=FS_SMALL, color=colors[i])

    # Pareto frontier (minimum time for given ARI)
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
                label="Pareto frontier", zorder=2)

    ax.set_xlabel("Training Time (s)", fontsize=FS_AXIS)
    ax.set_ylabel("Validation ARI ↑", fontsize=FS_AXIS)
    ax.set_title("Efficiency: ARI vs Training Time\n(Bubble size = peak GPU memory)",
                 fontsize=FS_TITLE, pad=3)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(alpha=0.22, linestyle="--", linewidth=0.4)
    ax.margins(0.18)  # prevent scatter markers clipping at edges

    # Memory legend
    for mem_val, label in [(min(mem), f"{min(mem):.2f} GB"), (max(mem), f"{max(mem):.2f} GB")]:
        sz = ((mem_val - mem.min()) / (mem.max() - mem.min() + 1e-6) + 0.2) * 280
        ax.scatter([], [], s=sz, c="gray", alpha=0.5, edgecolors="black",
                   linewidths=0.6, label=f"Mem={label}")
    ax.legend(fontsize=FS_LEG, frameon=False, loc="lower right",
              handlelength=1.0, labelspacing=0.2)
    return ax


# ── Main ───────────────────────────────────────────────────────────────────

def build_figure(rdir: Path, outdir: Path):
    configs, train_losses, val_losses, val_scores, metrics = _load_data(rdir)

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    outer = gridspec.GridSpec(
        4, 1,
        height_ratios=[2.5, 2.5, 2.5, 3.0],
        hspace=0.42,
        figure=fig,
    )

    gs_A = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.30)
    gs_B = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], wspace=0.30)
    gs_C = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2], wspace=0.30)
    gs_D = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[3], wspace=0.20)

    print("  Drawing Panel A (Loss convergence)...")
    ax_A = _draw_loss_curves(gs_A, fig, configs, train_losses, val_losses)

    print("  Drawing Panel B (ARI/NMI/ASW evolution)...")
    ax_B = _draw_val_metric_evolution(gs_B, fig, configs, val_losses, val_scores)

    print("  Drawing Panel C (CAL/DAV/COR evolution)...")
    ax_C = _draw_advanced_metrics(gs_C, fig, configs, val_scores)

    print("  Drawing Panel D (Efficiency)...")
    ax_D = _draw_efficiency(gs_D, fig, configs, metrics)

    fig.subplots_adjust(left=0.13, right=0.94, top=0.92, bottom=0.05)

    # Shared legend at top of figure
    handles, labels = ax_A.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        fontsize=FS_LEG, frameon=True, ncol=len(handles),
        loc="upper center", bbox_to_anchor=(0.5, 0.995),
        handlelength=1.0, labelspacing=0.2, columnspacing=0.6,
        framealpha=0.92, edgecolor="#cccccc", borderpad=0.3,
    )

    _panel_label(fig, ax_A, "A")
    _panel_label(fig, ax_B, "B")
    _panel_label(fig, ax_C, "C")
    _panel_label(fig, ax_D, "D")

    print("\n── Conflict Detection ──")
    issues = detect_all_conflicts(fig, label="training_dynamics", verbose=True)

    outpath = outdir / "training_dynamics.png"
    fig.savefig(outpath, dpi=DPI)

    # Export individual panel sub-figures
    sub_dir = outdir / "fig4_training_dynamics"
    sub_dir.mkdir(parents=True, exist_ok=True)
    _export_subpanels(fig, sub_dir, [(ax_A, "panelA_train_loss"),
                                     (ax_B, "panelB_val_ari"),
                                     (ax_C, "panelC_val_elbo"),
                                     (ax_D, "panelD_efficiency")])
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
                   default=str(_benchmarks / "results" / "dataset_default"))
    p.add_argument("--outdir",
                   default=str(_benchmarks / "figures"))
    args = p.parse_args()
    rdir   = Path(args.resultsdir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    build_figure(rdir, outdir)


if __name__ == "__main__":
    main()
