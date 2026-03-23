#!/usr/bin/env python
"""MoCoO Figure 6 — Downstream biological validation.

Six-panel figure using the direct_layout geometry engine:
  (a)  Annotation transfer — kNN accuracy + F1 (grouped bars per dataset)
  (b)  Uncertainty quantification — mean ± std (bars + error bars)
  (c)  Generation quality — NND, coverage, authenticity, diversity (heatmap)
  (d)  Gene importance — marker recovery rate per dataset (bars)
  (e)  Branching detection — branch cells and divergence per dataset
  (f)  Differential expression — fraction significant genes per dataset
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from benchmarks.scripts.plotting.shared import setup_fonts
from mocoo.visualization.style import (
    FS_AXIS, FS_LEGEND, FS_SMALL, FS_TICK, FS_TITLE,
    ACCENT_POSITIVE, ACCENT_NEGATIVE, HEATMAP_CMAP,
    FIG_WIDTH_IN, DPI,
    apply_style, save_figure, add_panel_label,
)
from mocoo.visualization.direct_layout import bind_figure_region

setup_fonts()
apply_style()

_DATASETS = ["IRALL", "dentate", "endo", "paul", "spinoids"]
_DS_SHORT = {"IRALL": "IRALL", "dentate": "Dentate", "endo": "Endo",
             "paul": "Paul", "spinoids": "Spinoids"}


def _load_downstream(results_dir: Path):
    """Load downstream JSONs for all datasets."""
    data = {}
    for ds in _DATASETS:
        ds_dir = results_dir / ds / "downstream"
        if not ds_dir.exists():
            continue
        rec = {}
        for name in ("annotation_transfer", "uncertainty",
                      "generation_quality", "gene_importance", "branching",
                      "differential_expression"):
            fp = ds_dir / f"{name}.json"
            if fp.exists():
                rec[name] = json.loads(fp.read_text())
        data[ds] = rec
    return data


def make_figure(results_dir: Path, out_path: Path):
    data = _load_downstream(results_dir)
    if not data:
        print("No downstream data found.")
        return

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * 1.30))
    root = bind_figure_region(fig, (0.12, 0.07, 0.95, 0.93))
    (row_top, row_mid, row_bot) = root.split_rows([1, 1, 1], gap=0.11)
    (r_a, r_b) = row_top.split_cols([1, 1], gap=0.10)
    (r_c, r_d) = row_mid.split_cols([1, 1], gap=0.10)
    (r_e, r_f) = row_bot.split_cols([1, 1], gap=0.10)

    # --- Panel (a): Annotation transfer ---
    ax_a = r_a.add_axes(fig)
    ds_labels = []
    accs, f1s = [], []
    for ds in _DATASETS:
        if ds not in data or "annotation_transfer" not in data[ds]:
            continue
        at = data[ds]["annotation_transfer"]
        knn = at.get("knn", {})
        accs.append(knn.get("accuracy", 0))
        f1s.append(knn.get("f1_macro", 0))
        ds_labels.append(_DS_SHORT.get(ds, ds))

    x = np.arange(len(ds_labels))
    w = 0.35
    ax_a.bar(x - w / 2, accs, w, label="Accuracy", color="#0072B2", zorder=3)
    ax_a.bar(x + w / 2, f1s, w, label="F1 macro", color="#E69F00", zorder=3)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(ds_labels, fontsize=FS_TICK)
    ax_a.set_ylabel("Score", fontsize=FS_AXIS)
    ax_a.set_title("Annotation Transfer", fontsize=FS_TITLE)
    ax_a.legend(fontsize=FS_LEGEND, loc="upper left")
    ax_a.set_ylim(0, 1.05)
    add_panel_label(ax_a, "a", x=-0.22, y=1.08)

    # --- Panel (b): Uncertainty ---
    ax_b = r_b.add_axes(fig)
    means, stds, q5s, q95s = [], [], [], []
    unc_labels = []
    for ds in _DATASETS:
        if ds not in data or "uncertainty" not in data[ds]:
            continue
        u = data[ds]["uncertainty"]
        means.append(u["mean"])
        stds.append(u["std"])
        q5s.append(u["q5"])
        q95s.append(u["q95"])
        unc_labels.append(_DS_SHORT.get(ds, ds))

    x = np.arange(len(unc_labels))
    means_a = np.array(means)
    q5_a = np.array(q5s)
    q95_a = np.array(q95s)
    ax_b.bar(x, means_a, 0.6, color="#56B4E9", zorder=3)
    ax_b.errorbar(x, means_a, yerr=[means_a - q5_a, q95_a - means_a],
                  fmt="none", ecolor="black", capsize=3, zorder=4)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(unc_labels, fontsize=FS_TICK)
    ax_b.set_ylabel("Reconstruction Uncertainty", fontsize=FS_AXIS)
    ax_b.set_title("Uncertainty Quantification", fontsize=FS_TITLE)
    add_panel_label(ax_b, "b", x=-0.26, y=1.08)

    # --- Panel (c): Generation quality heatmap ---
    ax_c = r_c.add_axes(fig)
    gen_metrics = ["nnd_mean", "coverage", "authenticity", "diversity"]
    gen_labels = [u"NND \u2193", u"Coverage \u2191", u"Authenticity \u2191",
                  u"Diversity \u2191"]
    gen_matrix = []
    gen_ds = []
    for ds in _DATASETS:
        if ds not in data or "generation_quality" not in data[ds]:
            continue
        g = data[ds]["generation_quality"]
        gen_matrix.append([g.get(m, 0) for m in gen_metrics])
        gen_ds.append(_DS_SHORT.get(ds, ds))

    if gen_matrix:
        mat = np.array(gen_matrix)
        im = ax_c.imshow(mat, aspect="auto", cmap=HEATMAP_CMAP)
        ax_c.set_xticks(range(len(gen_labels)))
        ax_c.set_xticklabels(gen_labels, fontsize=FS_SMALL, rotation=30,
                             ha="right")
        ax_c.set_yticks(range(len(gen_ds)))
        ax_c.set_yticklabels(gen_ds, fontsize=FS_TICK)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                color = "white" if mat[i, j] > mat.max() * 0.6 else "black"
                ax_c.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                          fontsize=FS_SMALL, color=color)
        ax_c.set_title("Generation Quality", fontsize=FS_TITLE)
    add_panel_label(ax_c, "c", x=-0.18, y=1.02)

    # --- Panel (d): Gene importance — marker recovery ---
    ax_d = r_d.add_axes(fig)
    marker_counts = []
    gi_labels = []
    for ds in _DATASETS:
        if ds not in data or "gene_importance" not in data[ds]:
            continue
        gi = data[ds]["gene_importance"]
        mf = gi.get("markers_found", [])
        if isinstance(mf, list):
            mc = sum(1 for m in mf if m.get("in_top50", False))
        elif isinstance(mf, dict):
            mc = mf.get("count", 0)
        else:
            mc = int(mf) if mf else 0
        marker_counts.append(mc)
        gi_labels.append(_DS_SHORT.get(ds, ds))

    x = np.arange(len(gi_labels))
    bars = ax_d.bar(x, marker_counts, 0.6, color="#009E73", zorder=3)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(gi_labels, fontsize=FS_TICK)
    ax_d.set_ylabel("Markers in Top 50", fontsize=FS_AXIS)
    ax_d.set_title("Gene Importance", fontsize=FS_TITLE)
    ymax = max(marker_counts) if marker_counts else 1
    ax_d.set_ylim(0, ymax + 2)
    for bar_i, v in enumerate(marker_counts):
        ax_d.text(bar_i, v + 0.3, str(v), ha="center", va="bottom",
                  fontsize=FS_SMALL)
    add_panel_label(ax_d, "d", x=-0.18, y=1.08)

    # --- Panel (e): Branching detection ---
    ax_e = r_e.add_axes(fig)
    br_cells = []
    br_divs = []
    br_labels = []
    for ds in _DATASETS:
        if ds not in data or "branching" not in data[ds]:
            continue
        br = data[ds]["branching"]
        br_cells.append(br.get("n_branch_cells", 0))
        stats = br.get("divergence_stats", {})
        br_divs.append(stats.get("q90", 0))
        br_labels.append(_DS_SHORT.get(ds, ds))

    x = np.arange(len(br_labels))
    w = 0.35
    ax_e_twin = ax_e.twinx()
    ax_e.bar(x - w / 2, br_cells, w, label="Branch cells",
             color="#CC79A7", zorder=3)
    ax_e_twin.bar(x + w / 2, br_divs, w, label="Divergence (q90)",
                  color="#D55E00", zorder=3)
    ax_e.set_xticks(x)
    ax_e.set_xticklabels(br_labels, fontsize=FS_TICK)
    ax_e.set_ylabel("Branch Cells", fontsize=FS_AXIS, color="#CC79A7")
    ax_e_twin.set_ylabel("Divergence (q90)", fontsize=FS_AXIS, color="#D55E00")
    ax_e.set_title("Branching Detection", fontsize=FS_TITLE)
    lines_e = ax_e.get_legend_handles_labels()
    lines_e2 = ax_e_twin.get_legend_handles_labels()
    ax_e.legend(lines_e[0] + lines_e2[0], lines_e[1] + lines_e2[1],
                fontsize=FS_LEGEND, loc="upper right")
    add_panel_label(ax_e, "e", x=-0.22, y=1.08)

    # --- Panel (f): Differential expression ---
    ax_f = r_f.add_axes(fig)
    de_fracs = []
    de_labels = []
    for ds in _DATASETS:
        if ds not in data or "differential_expression" not in data[ds]:
            continue
        de = data[ds]["differential_expression"]
        # Compute mean fraction of significant genes across all clusters
        fracs = []
        for key, val in de.items():
            if key.startswith("_"):
                continue
            if isinstance(val, dict) and "frac_sig_005" in val:
                fracs.append(val["frac_sig_005"])
        de_fracs.append(np.mean(fracs) if fracs else 0)
        de_labels.append(_DS_SHORT.get(ds, ds))

    x = np.arange(len(de_labels))
    ax_f.bar(x, de_fracs, 0.6, color="#56B4E9", zorder=3)
    ax_f.set_xticks(x)
    ax_f.set_xticklabels(de_labels, fontsize=FS_TICK)
    ax_f.set_ylabel("Mean Frac. Sig. (p<0.05)", fontsize=FS_AXIS)
    ax_f.set_title("Differential Expression", fontsize=FS_TITLE)
    ax_f.set_ylim(0, 1.05)
    for bar_i, v in enumerate(de_fracs):
        ax_f.text(bar_i, v + 0.01, f"{v:.2f}", ha="center", va="bottom",
                  fontsize=FS_SMALL)
    add_panel_label(ax_f, "f", x=-0.22, y=1.08)

    save_figure(fig, str(out_path), vcd_label="fig6_downstream", vcd_verbose=True)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="MoCoO Fig 6: Downstream")
    parser.add_argument("--resultsdir", type=Path,
                        default=Path(__file__).resolve().parent.parent.parent
                        / "results")
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()
    out = args.outdir if args.outdir else (args.resultsdir.parent / "figures")
    out.mkdir(parents=True, exist_ok=True)
    make_figure(args.resultsdir, out / "fig6_downstream.png")


if __name__ == "__main__":
    main()
