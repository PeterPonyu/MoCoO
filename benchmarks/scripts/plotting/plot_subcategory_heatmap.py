"""Plot subcategory metric heatmap across configurations.

Generates a multi-panel heatmap showing individual subcategory scores
for all metric families (Clustering, DRE, DREX, LSE, LSEX) across
6 configurations. Column-normalised with best-in-column highlighting.

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
from mocoo.visualization.style import (
    FIG_WIDTH_IN, FIG_HEIGHT_IN, DPI, SAVEFIG_KW,
    FS_LABEL, FS_TITLE, FS_AXIS, FS_TICK, FS_LEGEND, FS_SMALL,
    HEATMAP_DARK_THRESHOLD,
    apply_style, get_config_order, get_config_colors, get_short_name,
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
        ("DRE_umap_distance_correlation", "UMAP\nDistCorr", True),
        ("DRE_umap_Q_local", "UMAP\nQ_loc", True),
        ("DRE_umap_Q_global", "UMAP\nQ_glob", True),
        ("DRE_tsne_distance_correlation", "tSNE\nDistCorr", True),
        ("DRE_tsne_Q_local", "tSNE\nQ_loc", True),
        ("DRE_tsne_Q_global", "tSNE\nQ_glob", True),
    ],
    "DREX": [
        ("DREX_trustworthiness", "Trust", True),
        ("DREX_continuity", "Cont", True),
        ("DREX_distance_spearman", "Spear", True),
        ("DREX_distance_pearson", "Pearson", True),
        ("DREX_local_scale_quality", "LocScale", True),
        ("DREX_neighborhood_symmetry", "NbrSym", True),
        ("DREX_knn_rank_correlation", "RankCorr", True),
    ],
    "LSE": [
        ("LSE_manifold_dimensionality", "ManDim", True),
        ("LSE_spectral_decay_rate", "SpDecay", True),
        ("LSE_participation_ratio", "PartRat", True),
        ("LSE_anisotropy_score", "Aniso", False),
        ("LSE_noise_resilience", "NoiseR", True),
        ("LSE_core_quality", "Core", True),
    ],
    "LSEX": [
        ("LSEX_two_hop_connectivity", "2Hop", True),
        ("LSEX_radial_concentration", "RadConc", True),
        ("LSEX_local_curvature", "LocCurv", True),
        ("LSEX_cluster_compactness", "Compact", True),
        ("LSEX_neighbor_purity", "NbrPur", True),
        ("LSEX_sampling_stability", "SampStab", True),
        ("LSEX_inter_cluster_gap", "Gap", True),
    ],
}

SHORT_NAMES = {c: get_short_name(c) for c in CONFIGS}


def load_metrics(rdir: Path) -> dict:
    data = {}
    for cfg in CONFIGS:
        key = cfg.replace("+", "_")
        jf = rdir / f"{key}.json"
        if jf.exists():
            with open(jf) as f:
                data[cfg] = json.load(f)
    return data


def make_heatmap(ax, data, panel_name, metrics_spec, configs):
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

    # Annotate with raw values, bold the best, add rank superscript
    for j, (_, _, higher_better) in enumerate(metrics_spec):
        col = mat[:, j]
        valid = ~np.isnan(col)
        if valid.any():
            best_idx = np.nanargmax(col) if higher_better else np.nanargmin(col)
            wins[best_idx] += 1
            # Compute ranks for this column
            valid_vals = col[valid]
            if higher_better:
                order = np.argsort(-valid_vals)
            else:
                order = np.argsort(valid_vals)
            ranks = np.zeros(len(valid_vals), dtype=int)
            ranks[order] = np.arange(1, len(valid_vals) + 1)
            valid_indices = np.where(valid)[0]
        else:
            best_idx = -1
            valid_indices = np.array([], dtype=int)
            ranks = np.array([], dtype=int)

        for i in range(n_cfg):
            val = mat[i, j]
            if np.isnan(val):
                continue
            # Format: use integer for large values, 3 decimal otherwise
            txt = f"{val:.0f}" if abs(val) > 10 else f"{val:.3f}"
            color = "white" if norm_mat[i, j] > HEATMAP_DARK_THRESHOLD else "black"
            # Add rank as superscript
            rank_idx = np.where(valid_indices == i)[0]
            if len(rank_idx) > 0:
                rank = ranks[rank_idx[0]]
                txt = f"{txt} ({rank})"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=FS_SMALL, fontweight="normal", color=color)

    ax.set_xticks(range(n_met))
    ax.set_xticklabels([m[1] for m in metrics_spec], fontsize=FS_TICK, rotation=45, ha="right")
    ax.set_yticks(range(n_cfg))
    # Append win count to y-labels
    ylabels = [f"{SHORT_NAMES.get(c, c)} [{wins[i]}W]" for i, c in enumerate(configs)]
    ax.set_yticklabels(ylabels, fontsize=FS_TICK)
    # Extra padding so labels don't overlap with heatmap boundaries
    ax.tick_params(axis="y", pad=6)
    ax.tick_params(axis="x", pad=4)
    ax.set_title(panel_name, fontsize=FS_TITLE, pad=4)
    return im, wins


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

    panel_names = list(PANELS.keys())
    n_panels = len(panel_names)

    # Grid layout: 3 columns x 2 rows (5 panels + 1 empty) for better space use
    n_cols = 3
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(FIG_WIDTH_IN * 1.35, FIG_HEIGHT_IN * 0.85),
                              gridspec_kw={"hspace": 0.32, "wspace": 0.28})

    letters = "ABCDE"
    total_wins = np.zeros(len(CONFIGS), dtype=int)
    last_im = None
    for idx, (pname, metrics_spec) in enumerate(PANELS.items()):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        im, wins = make_heatmap(ax, data, pname, metrics_spec, CONFIGS)
        last_im = im
        total_wins += wins

    # Use empty cell for colorbar instead of hiding
    for idx in range(n_panels, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        cbar_ax = axes[row, col]
        cbar_ax.set_visible(True)
        # Add colorbar in the empty cell
        cb = fig.colorbar(last_im, ax=cbar_ax, shrink=0.7, pad=0.05,
                          label="Column-normalised score")
        cb.ax.tick_params(labelsize=FS_TICK)
        cbar_ax.set_axis_off()

    # Print win summary
    print("Win counts per config (across all panels):")
    for i, cfg in enumerate(CONFIGS):
        print(f"  {cfg}: {total_wins[i]} wins")

    fig.suptitle("Subcategory Metric Breakdown ($\\beta = 0.1$, IRALL, 3000 cells)",
                 fontsize=FS_LABEL, y=0.99)

    # Finalise layout BEFORE adding panel labels so that ax.get_position()
    # returns the correct post-adjustment coordinates.
    fig.subplots_adjust(left=0.09, right=0.97, top=0.93, bottom=0.08)

    # Add panel labels after layout adjustment to avoid overlap with cells.
    for idx in range(n_panels):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        panel_label(fig, ax, letters[idx], x_off=-0.04, y_off=0.035)

    out_path = outdir / "fig5_subcategory_heatmap.png"
    fig.savefig(out_path, **SAVEFIG_KW, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
