"""Plot subcategory metric heatmap across configurations.

This module primarily provides the subcategory diagnostic block embedded in the
integrated Figure 5 builder. It can still be run standalone to export just that
block into the Figure 5 subpanel directory.

Usage:
    python plot_subcategory_heatmap.py [--resultsdir DIR] [--outdir DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from benchmarks.scripts.plotting.shared import setup_fonts, panel_label
from vcd import detect_all_conflicts
from mocoo.visualization.style import (
    FIG_WIDTH_IN, FIG_HEIGHT_IN, DPI,
    FS_LABEL, FS_TITLE, FS_AXIS, FS_TICK, FS_LEGEND, FS_SMALL,
    HEATMAP_DARK_THRESHOLD,
    apply_style, get_config_order, get_config_colors, get_short_name,
    get_tick_name, FMT_SCORE_SHORT,
)

CONFIGS = get_config_order()

# Metric families with (key, label, higher_is_better)
PANELS = {
    "Clustering": [
        ("NMI", "NMI", True),
        ("ARI", "ARI", True),
        ("ASW", "ASW", True),
        ("DAV", "DAV", False),
        ("CAL", "Cal-H", True),
        ("COR", "Corr", True),
    ],
    "DRE": [
        ("DRE_umap_distance_correlation", "U-DC", True),
        ("DRE_umap_Q_local", "U-QL", True),
        ("DRE_umap_Q_global", "U-QG", True),
        ("DRE_tsne_distance_correlation", "T-DC", True),
        ("DRE_tsne_Q_local", "T-QL", True),
        ("DRE_tsne_Q_global", "T-QG", True),
    ],
    "DREX": [
        ("DREX_trustworthiness", "Trst", True),
        ("DREX_continuity", "Cont", True),
        ("DREX_distance_spearman", "Spr", True),
        ("DREX_distance_pearson", "Pr", True),
        ("DREX_local_scale_quality", "LSc", True),
        ("DREX_neighborhood_symmetry", "NbrS", True),
        ("DREX_knn_rank_correlation", "Rnk", True),
    ],
    "LSE": [
        ("LSE_manifold_dimensionality", "MDim", True),
        ("LSE_spectral_decay_rate", "SDec", True),
        ("LSE_participation_ratio", "PRat", True),
        ("LSE_anisotropy_score", "Aniso", False),
        ("LSE_noise_resilience", "NoiseR", True),
        ("LSE_core_quality", "Core", True),
    ],
    "LSEX": [
        ("LSEX_two_hop_connectivity", "2Hop", True),
        ("LSEX_radial_concentration", "RadC", True),
        ("LSEX_local_curvature", "Curv", True),
        ("LSEX_cluster_compactness", "Comp", True),
        ("LSEX_neighbor_purity", "NPur", True),
        ("LSEX_sampling_stability", "SStb", True),
        ("LSEX_inter_cluster_gap", "Gap", True),
    ],
}

SHORT_NAMES = {c: get_tick_name(c) for c in CONFIGS}


def load_metrics(rdir: Path) -> dict:
    data = {}
    for cfg in CONFIGS:
        key = cfg.replace("+", "_")
        jf = rdir / f"{key}.json"
        if jf.exists():
            with open(jf) as f:
                data[cfg] = json.load(f)
    return data


def make_heatmap(ax, data, panel_name, metrics_spec, configs, *, show_ylabels=True):
    """Draw a column-normalised heatmap and return per-config win counts."""
    n_cfg = len(configs)
    n_met = len(metrics_spec)

    mat = np.full((n_cfg, n_met), np.nan)
    for i, cfg in enumerate(configs):
        if cfg not in data:
            continue
        for j, (key, _, _) in enumerate(metrics_spec):
            mat[i, j] = data[cfg].get(key, np.nan)

    # Column-normalise for colouring
    col_min = np.nanmin(mat, axis=0)
    col_max = np.nanmax(mat, axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1
    norm_mat = (mat - col_min) / col_range

    # For lower-is-better metrics, invert normalization
    for j, (_, _, higher_better) in enumerate(metrics_spec):
        if not higher_better:
            norm_mat[:, j] = 1.0 - norm_mat[:, j]

    im = ax.imshow(norm_mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

    # Track wins per config
    wins = np.zeros(n_cfg, dtype=int)

    for j, (_, _, higher_better) in enumerate(metrics_spec):
        col = mat[:, j]
        valid = ~np.isnan(col)
        if valid.any():
            best_idx = np.nanargmax(col) if higher_better else np.nanargmin(col)
            wins[best_idx] += 1

    ax.set_xticks(range(n_met))
    ax.set_xticklabels([m[1] for m in metrics_spec], fontsize=max(FS_SMALL - 2, 5.5), rotation=45, ha="right")
    ax.set_yticks(range(n_cfg))
    ylabels = [SHORT_NAMES.get(c, c) for i, c in enumerate(configs)]
    if show_ylabels:
        ax.set_yticklabels(ylabels, fontsize=max(FS_SMALL - 2, 5.5))
    else:
        ax.set_yticklabels([])
    # Extra padding so labels don't overlap with heatmap boundaries
    ax.tick_params(axis="y", pad=8, labelleft=True)
    ax.tick_params(axis="x", pad=2, labelbottom=True)
    ax.set_title(panel_name, fontsize=FS_TITLE - 1, pad=2)
    return im, wins


def draw_subcategory_block(fig, axes_list, data, configs=None):
    """Draw the 5-panel subcategory diagnostic block on the supplied axes."""
    active_configs = list(configs or CONFIGS)
    total_wins = np.zeros(len(active_configs), dtype=int)
    for idx, (pname, metrics_spec) in enumerate(PANELS.items()):
        _, wins = make_heatmap(
            axes_list[idx],
            data,
            pname,
            metrics_spec,
            active_configs,
            show_ylabels=True,
        )
        total_wins += wins
    return total_wins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resultsdir",
                        default="benchmarks/results/beta_ablation/beta_0.1")
    parser.add_argument("--outdir",
                        default="benchmarks/figures")
    args = parser.parse_args()

    rdir = Path(args.resultsdir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    setup_fonts()
    apply_style()

    data = load_metrics(rdir)
    if not data:
        print(f"No JSON files found in {rdir}")
        return

    fig = plt.figure(figsize=(FIG_WIDTH_IN * 1.35, FIG_HEIGHT_IN * 0.72), dpi=DPI)

    top_gap = 0.04
    top_w = (0.82 - 2 * top_gap) / 3
    bot_gap = 0.05
    bot_w = (0.82 - bot_gap) / 2
    axes_list = [
        fig.add_axes([0.12, 0.58, top_w, 0.24]),
        fig.add_axes([0.12 + top_w + top_gap, 0.58, top_w, 0.24]),
        fig.add_axes([0.12 + 2 * (top_w + top_gap), 0.58, top_w, 0.24]),
        fig.add_axes([0.12, 0.16, bot_w, 0.24]),
        fig.add_axes([0.12 + bot_w + bot_gap, 0.16, bot_w, 0.24]),
    ]

    total_wins = draw_subcategory_block(fig, axes_list, data, CONFIGS)

    print("Win counts per config (merged subcategory heatmap):")
    for i, cfg in enumerate(CONFIGS):
        print(f"  {cfg}: {total_wins[i]} wins")

    for idx, ax in enumerate(axes_list):
        panel_label(fig, ax, "ABCDE"[idx], x_off=-0.04, y_off=0.028)

    out_path = outdir / "fig5_composed_benchmark" / "panelD_subcategory_block.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n── Conflict Detection (merged subcategory heatmap) ──")
    issues = detect_all_conflicts(fig, label="subcategory_heatmap", verbose=True)
    n_warn = sum(1 for i in issues if i["severity"] == "warning")
    n_err = sum(1 for i in issues if i["severity"] == "error")

    from mocoo.visualization.style import save_figure
    save_figure(fig, out_path, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(f"{n_warn} warnings | {n_err} errors")
    return issues


if __name__ == "__main__":
    main()
