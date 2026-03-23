#!/usr/bin/env python
"""MoCoO Supplementary Figure S6 — Biological validation of latent spaces.

Four-panel figure (IRALL, Full model):
  (A) Per-dimension gradient importance bar chart
  (B) Top genes per latent component (horizontal bar chart, top 5 per dim)
  (C) Gene marker recovery summary (markers found vs not found)
  (D) Per-configuration gene importance comparison across configs
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
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from benchmarks.scripts.plotting.shared import setup_fonts
from mocoo.visualization.style import (
    FS_AXIS, FS_LEGEND, FS_SMALL, FS_TICK, FS_TITLE,
    FIG_WIDTH_IN, DPI,
    apply_style, save_figure, add_panel_label,
    get_config_colors, get_display_name, get_short_name,
    get_base_config_order,
)
from mocoo.visualization.direct_layout import bind_figure_region

setup_fonts()
apply_style()

_DIR_MAP = {
    "VAE": "VAE", "VAE+ODE": "VAE_ODE", "VAE+MoCo": "VAE_MoCo",
    "VAE+MoCo+Proto": "VAE_MoCo_Proto", "VAE+ODE+MoCo": "VAE_ODE_MoCo",
    "Full": "Full",
}


def _load_gradients(results_dir: Path, config: str):
    cfg_dir = _DIR_MAP.get(config, config)
    fp = results_dir / "IRALL" / cfg_dir / "gradients.npy"
    if not fp.exists():
        return None
    return np.load(fp)


def _load_gene_importance(results_dir: Path):
    fp = results_dir / "IRALL" / "downstream" / "gene_importance.json"
    if not fp.exists():
        return None
    with open(fp) as f:
        return json.load(f)


def make_figure(results_dir: Path, out_path: Path):
    config_colors = get_config_colors()
    base_cfgs = get_base_config_order()
    gi = _load_gene_importance(results_dir)

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * 1.0))
    root = bind_figure_region(fig, (0.12, 0.06, 0.86, 0.90))
    (r_top, r_bot) = root.split_rows([1, 1], gap=0.14)
    (r_a, r_b) = r_top.split_cols([1, 1], gap=0.12)
    (r_c, r_d) = r_bot.split_cols([1, 1], gap=0.12)

    # ── Panel (A): Per-dimension gradient importance (Full model) ──
    ax_a = r_a.add_axes(fig)
    grads = _load_gradients(results_dir, "Full")
    if grads is not None:
        dim_imp = np.mean(np.abs(grads), axis=0)
        ndims = len(dim_imp)
        x = np.arange(ndims)
        ax_a.bar(x, dim_imp, color=config_colors.get("Full", "#D55E00"),
                 alpha=0.8, zorder=3)
        ax_a.set_xlabel("Latent Dimension", fontsize=FS_AXIS)
        ax_a.set_ylabel("Mean |Gradient|", fontsize=FS_AXIS)
        ax_a.set_title("Dimension Importance (Full)", fontsize=FS_TITLE)
        ax_a.xaxis.set_major_locator(MaxNLocator(integer=True))
    add_panel_label(ax_a, "A", x=-0.22, y=1.14)

    # ── Panel (B): Top genes per latent component ──
    ax_b = r_b.add_axes(fig)
    if gi and "top_genes_per_dim" in gi:
        top_per_dim = gi["top_genes_per_dim"]
        n_show_dims = min(4, len(top_per_dim))
        n_genes = 3
        y_offset = 0
        yticks, ylabels = [], []
        cmap = plt.cm.Set2(np.linspace(0, 1, n_show_dims))
        for di in range(n_show_dims):
            genes = top_per_dim.get(str(di), [])[:n_genes]
            for gi_idx, (gene, score) in enumerate(genes):
                ax_b.barh(y_offset, score, height=0.7, color=cmap[di],
                          alpha=0.8, zorder=3)
                yticks.append(y_offset)
                ylabels.append(f"D{di}:{gene}")
                y_offset += 1
            y_offset += 0.5
        ax_b.set_yticks(yticks)
        ax_b.set_yticklabels(ylabels, fontsize=FS_SMALL - 1)
        ax_b.set_xlabel("Importance Score", fontsize=FS_AXIS)
        ax_b.set_title("Top Genes per Dim", fontsize=FS_TITLE)
        ax_b.invert_yaxis()
    add_panel_label(ax_b, "B", x=-0.22, y=1.08)

    # ── Panel (C): Marker gene recovery ──
    ax_c = r_c.add_axes(fig)
    if gi and "markers_found" in gi:
        markers = gi["markers_found"]
        names = [m["marker"] for m in markers]
        found = [1 if m["in_top50"] else 0 for m in markers]
        colors = ["#009E73" if f else "#CC79A7" for f in found]
        x = np.arange(len(names))
        ax_c.bar(x, [1] * len(names), color=colors, alpha=0.8, zorder=3)
        ax_c.set_xticks(x)
        ax_c.set_xticklabels(names, fontsize=FS_SMALL - 1, rotation=45,
                             ha="right", style="italic")
        ax_c.set_yticks([0, 1])
        ax_c.set_yticklabels(["Not in Top 50", "In Top 50"], fontsize=FS_SMALL)
        ax_c.set_title("Marker Recovery", fontsize=FS_TITLE)
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor="#009E73", label="Found"),
                           Patch(facecolor="#CC79A7", label="Not found")]
        ax_c.legend(handles=legend_elements, fontsize=FS_SMALL,
                    loc="upper right")
    add_panel_label(ax_c, "C", x=-0.18, y=1.08)

    # ── Panel (D): Cross-config gradient comparison ──
    ax_d = r_d.add_axes(fig)
    bar_width = 0.12
    n_cfgs = 0
    for ci, cfg in enumerate(base_cfgs):
        gr = _load_gradients(results_dir, cfg)
        if gr is None:
            continue
        dim_imp = np.mean(np.abs(gr), axis=0)
        ndims = len(dim_imp)
        x = np.arange(ndims) + ci * bar_width
        color = config_colors.get(cfg, "#888888")
        ax_d.bar(x, dim_imp, width=bar_width, color=color, alpha=0.8,
                 label=get_short_name(cfg), zorder=3)
        n_cfgs += 1
    if n_cfgs > 0:
        ax_d.set_xlabel("Latent Dimension", fontsize=FS_AXIS)
        ax_d.set_ylabel("Mean |Gradient|", fontsize=FS_AXIS)
        ax_d.set_title("Config Comparison", fontsize=FS_TITLE)
        ax_d.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax_d.legend(fontsize=FS_SMALL - 1, loc="upper right", ncol=2)
    add_panel_label(ax_d, "D", x=-0.26, y=1.18)

    save_figure(fig, str(out_path), vcd_label="figS6_biological_validation",
                vcd_verbose=True)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="MoCoO Supp Fig S6: Biological Validation")
    parser.add_argument("--resultsdir", type=Path,
                        default=Path(__file__).resolve().parent.parent.parent
                        / "results")
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=None)
    args = parser.parse_args()
    outdir = args.outdir or (args.resultsdir.parent / "figures")
    Path(outdir).mkdir(parents=True, exist_ok=True)
    make_figure(args.resultsdir, Path(outdir) / "figS6_biological_validation.png")


if __name__ == "__main__":
    main()
