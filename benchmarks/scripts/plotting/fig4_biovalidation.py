#!/usr/bin/env python
"""MoCoO Figure 4 — Biological Validation (3-panel).

(a) Annotation transfer: kNN accuracy + F1 macro across 5 datasets.
(b) Pseudotime--marker validation: lollipop chart with gene annotations.
(c) Generation quality: authenticity + diversity across 5 datasets.

Reads:
  benchmarks/results/<dataset>/downstream/annotation_transfer.json
  benchmarks/results/<dataset>/downstream/generation_quality.json
  (Hardcoded pseudotime--marker data from Table V)
Writes:
  benchmarks/figures/fig4_biovalidation.{png,pdf}
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

from benchmarks.scripts.plotting.shared import setup_fonts
from mocoo.visualization.style import (
    FS_AXIS, FS_LEGEND, FS_SMALL, FS_TICK, FS_TITLE,
    FIG_WIDTH_IN,
    apply_style, save_figure, add_panel_label,
)

setup_fonts()
apply_style()

# ── Constants ─────────────────────────────────────────────────────────────

_DATASETS = ["IRALL", "dentate", "endo", "paul", "spinoids"]
_DISPLAY_NAMES = {
    "IRALL": "IRALL", "dentate": "Dentate", "endo": "Endo",
    "paul": "Paul", "spinoids": "Spinoids",
}

# Pseudotime-marker data (canonical, from appendix marker tables)
_PSEUDOTIME_DATA = [
    ("IRALL", "Hbb-bs", 0.275, "HSC \u2192 erythroid"),
    ("Dentate", "Slc1a3", 0.542, "RGC \u2192 neuroblast"),
    ("Endo", "Ins2", 0.463, "Prog. \u2192 \u03B2-cell"),
    ("Paul", "Epor", 0.346, "Prog. \u2192 erythroid"),
    ("Spinoids", "MKI67", 0.492, "Prolif. \u2192 diff."),
]

# Wong colorblind palette assignments
_CLR_ACCURACY = "#0072B2"   # blue
_CLR_F1 = "#E69F00"         # orange
_CLR_STRONG = "#56B4E9"     # sky blue (rho >= 0.4)
_CLR_MODERATE = "#E69F00"   # orange (rho < 0.4)
_CLR_AUTHENTIC = "#009E73"  # green
_CLR_DIVERSITY = "#CC79A7"  # purple


# ── Data loaders ──────────────────────────────────────────────────────────

def _load_annotation_transfer(results_dir: Path):
    """Load kNN annotation transfer metrics for each dataset."""
    data = {}
    for ds in _DATASETS:
        fp = results_dir / ds / "downstream" / "annotation_transfer.json"
        if not fp.exists():
            continue
        with open(fp) as f:
            raw = json.load(f)
        knn = raw.get("knn", {})
        data[ds] = {
            "accuracy": knn.get("accuracy", np.nan),
            "f1_macro": knn.get("f1_macro", np.nan),
            "n_classes": knn.get("n_classes", 0),
        }
    return data


def _load_generation_quality(results_dir: Path):
    """Load generation quality metrics for each dataset."""
    data = {}
    for ds in _DATASETS:
        fp = results_dir / ds / "downstream" / "generation_quality.json"
        if not fp.exists():
            continue
        with open(fp) as f:
            raw = json.load(f)
        data[ds] = {
            "authenticity": raw.get("authenticity", np.nan),
            "diversity": raw.get("diversity", np.nan),
            "nnd_mean": raw.get("nnd_mean", np.nan),
        }
    return data


# ── Panel builders ────────────────────────────────────────────────────────

def _panel_annotation(ax, results_dir: Path):
    """Panel (a): Grouped bars for kNN accuracy + F1 macro."""
    data = _load_annotation_transfer(results_dir)
    datasets = [ds for ds in _DATASETS if ds in data]
    n = len(datasets)
    if n == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        return

    x = np.arange(n)
    w = 0.32

    acc_vals = [data[ds]["accuracy"] for ds in datasets]
    f1_vals = [data[ds]["f1_macro"] for ds in datasets]

    ax.bar(x - w / 2, acc_vals, w, label="Accuracy",
           color=_CLR_ACCURACY, zorder=3, edgecolor="white", linewidth=0.3)
    ax.bar(x + w / 2, f1_vals, w, label="F1 Macro",
           color=_CLR_F1, zorder=3, edgecolor="white", linewidth=0.3)

    # Value annotations on bars
    for i, (a, f) in enumerate(zip(acc_vals, f1_vals)):
        n_cls = data[datasets[i]]["n_classes"]
        if a > 0.15:
            ax.text(i - w / 2, a + 0.015, f"{a:.2f}", ha="center",
                    fontsize=FS_SMALL - 1, color="0.3")
            ax.text(i + w / 2, f + 0.015, f"{f:.2f}", ha="center",
                    fontsize=FS_SMALL - 1, color="0.3")
        else:
            # Very low accuracy (continuous-trajectory dataset) — show n_classes
            ax.text(i, max(a, f) + 0.04, f"{n_cls} classes\n(traj.)",
                    ha="center", fontsize=FS_SMALL - 1, color="0.4",
                    fontstyle="italic")

    labels = [_DISPLAY_NAMES.get(ds, ds) for ds in datasets]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FS_TICK)
    ax.set_ylabel("Score", fontsize=FS_AXIS)
    ax.set_ylim(0, 1.15)
    ax.set_title("Annotation Transfer", fontsize=FS_TITLE, pad=4)
    ax.legend(fontsize=FS_LEGEND, frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.18, linestyle="--", linewidth=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=FS_TICK)


def _panel_pseudotime(ax):
    """Panel (b): Lollipop chart for pseudotime--marker correlations."""
    datasets = [d[0] for d in _PSEUDOTIME_DATA]
    rhos = [d[2] for d in _PSEUDOTIME_DATA]
    genes = [d[1] for d in _PSEUDOTIME_DATA]
    axes_labels = [d[3] for d in _PSEUDOTIME_DATA]

    y = np.arange(len(datasets))

    for i, (rho, gene, axis_lbl) in enumerate(zip(rhos, genes, axes_labels)):
        color = _CLR_STRONG if rho >= 0.4 else _CLR_MODERATE

        # Stem
        ax.plot([0, rho], [i, i], color=color, linewidth=2.0, zorder=3,
                solid_capstyle="round")

        # Dot at tip
        ax.scatter(rho, i, color=color, s=70, zorder=4, edgecolors="white",
                   linewidths=0.8)

        # Gene + axis annotation
        ax.text(rho + 0.015, i, f"{gene} ({axis_lbl})",
                va="center", fontsize=FS_SMALL, color="0.3")

    ax.set_yticks(y)
    ax.set_yticklabels(datasets, fontsize=FS_TICK)
    ax.set_xlabel("Spearman |$\\rho$|", fontsize=FS_AXIS)
    ax.set_xlim(0, 0.78)
    ax.set_ylim(-0.5, len(datasets) - 0.5)
    ax.set_title("Pseudotime Validation", fontsize=FS_TITLE, pad=4)
    ax.axvline(0.3, color="0.7", linewidth=0.6, linestyle="--", zorder=1)
    ax.grid(axis="x", alpha=0.18, linewidth=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=FS_TICK)

    # All p << 0.001
    ax.text(0.98, 0.02, "all $p < 10^{-5}$", transform=ax.transAxes,
            fontsize=FS_SMALL, ha="right", va="bottom", color="0.5",
            fontstyle="italic")


def _panel_generation(ax, results_dir: Path):
    """Panel (c): Grouped bars for generation quality metrics."""
    data = _load_generation_quality(results_dir)
    datasets = [ds for ds in _DATASETS if ds in data]
    n = len(datasets)
    if n == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        return

    x = np.arange(n)
    w = 0.32

    auth_vals = [data[ds]["authenticity"] for ds in datasets]
    # Normalize diversity to [0, 1] for comparable scale
    div_raw = [data[ds]["diversity"] for ds in datasets]
    max_div = max(div_raw) if div_raw else 1.0
    div_norm = [v / max_div for v in div_raw]

    ax.bar(x - w / 2, auth_vals, w, label="Authenticity",
           color=_CLR_AUTHENTIC, zorder=3, edgecolor="white", linewidth=0.3)
    ax.bar(x + w / 2, div_norm, w, label="Diversity (norm.)",
           color=_CLR_DIVERSITY, zorder=3, edgecolor="white", linewidth=0.3)

    # Value annotations
    for i, (a, d) in enumerate(zip(auth_vals, div_norm)):
        ax.text(i - w / 2, a + 0.015, f"{a:.2f}", ha="center",
                fontsize=FS_SMALL - 1, color="0.3")
        ax.text(i + w / 2, d + 0.015, f"{d:.2f}", ha="center",
                fontsize=FS_SMALL - 1, color="0.3")

    labels = [_DISPLAY_NAMES.get(ds, ds) for ds in datasets]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FS_TICK)
    ax.set_ylabel("Score", fontsize=FS_AXIS)
    ax.set_ylim(0, 1.18)
    ax.set_title("Generation Quality", fontsize=FS_TITLE, pad=4)
    ax.legend(fontsize=FS_LEGEND, frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.18, linestyle="--", linewidth=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=FS_TICK)


# ── Main ──────────────────────────────────────────────────────────────────

def main(results_dir: Path, outdir: Path) -> int:
    fig_w = FIG_WIDTH_IN * 2.0
    fig_h = FIG_WIDTH_IN * 0.78

    fig = plt.figure(figsize=(fig_w, fig_h))

    # Layout: 3 panels in one row
    left = 0.06
    bottom = 0.16
    top = 0.88
    h = top - bottom
    gap = 0.07
    total_w = 0.90
    rel = [1.1, 1.0, 1.1]
    sum_rel = sum(rel)
    widths = [(r / sum_rel) * (total_w - gap * 2) for r in rel]

    x_pos = left
    rects = []
    for w in widths:
        rects.append([x_pos, bottom, w, h])
        x_pos += w + gap

    ax_a = fig.add_axes(rects[0])
    ax_b = fig.add_axes(rects[1])
    ax_c = fig.add_axes(rects[2])

    _panel_annotation(ax_a, results_dir)
    _panel_pseudotime(ax_b)
    _panel_generation(ax_c, results_dir)

    add_panel_label(ax_a, "a", x=-0.12, y=1.08)
    add_panel_label(ax_b, "b", x=-0.12, y=1.08)
    add_panel_label(ax_c, "c", x=-0.12, y=1.08)

    outpath = outdir / "fig4_biovalidation.png"
    issues = save_figure(fig, outpath, vcd_label="fig4_biovalidation",
                         vcd_verbose=True)
    n_err = sum(1 for i in issues if i.get("severity") == "error")
    plt.close(fig)

    print(f"\n\u2713 fig4_biovalidation saved to {outpath}")
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
