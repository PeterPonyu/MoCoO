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
from benchmarks.scripts.plotting.shared import setup_fonts, panel_label, add_config_legend_footnote, add_metric_footnote
from benchmarks.scripts.pipeline.visual_conflict_detector import detect_all_conflicts
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
            txt = f"{val:.0f}" if abs(val) > 10 else f"{val:.2f}"
            color = "white" if norm_mat[i, j] > HEATMAP_DARK_THRESHOLD else "black"
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

    # Split into two figures for better readability:
    #   Figure 1 (5a): Clustering + DRE + DREX (3 panels)
    #   Figure 2 (5b): LSE + LSEX (2 panels)
    all_panels = list(PANELS.items())
    fig1_panels = all_panels[:3]   # Clustering, DRE, DREX
    fig2_panels = all_panels[3:]   # LSE, LSEX

    for fig_idx, (panels_subset, suffix) in enumerate([
        (fig1_panels, "fig5_subcategory_heatmap_a.png"),
        (fig2_panels, "fig5_subcategory_heatmap_b.png"),
    ]):
        n_panels = len(panels_subset)
        # Single row layout for each sub-figure
        n_cols = n_panels
        n_rows = 1
        fig = plt.figure(figsize=(FIG_WIDTH_IN * 1.5, FIG_HEIGHT_IN * 0.50))
        _cw = (0.88 - 0.05 * (n_cols - 1)) / n_cols
        _rh = 0.72
        axes_list = [
            fig.add_axes([0.08 + c * (_cw + 0.05),
                          0.18, _cw, _rh])
            for c in range(n_cols)
        ]

        letters = "ABCDE" if fig_idx == 0 else "DE"
        if fig_idx == 1:
            letters = "AB"  # restart lettering for second figure
        total_wins = np.zeros(len(CONFIGS), dtype=int)
        last_im = None
        for idx, (pname, metrics_spec) in enumerate(panels_subset):
            ax = axes_list[idx]
            im, wins = make_heatmap(ax, data, pname, metrics_spec, CONFIGS)
            last_im = im
            total_wins += wins

        # Print win summary
        label = "Clustering/DRE/DREX" if fig_idx == 0 else "LSE/LSEX"
        print(f"Win counts per config ({label}):")
        for i, cfg in enumerate(CONFIGS):
            print(f"  {cfg}: {total_wins[i]} wins")

        add_config_legend_footnote(fig, y_pos=0.010)

        for idx in range(n_panels):
            ax = axes_list[idx]
            panel_label(fig, ax, letters[idx], x_off=-0.04, y_off=0.025)

        out_path = outdir / suffix

        print(f"\n── Conflict Detection ({label}) ──")
        issues = detect_all_conflicts(fig, label=f"subcategory_heatmap_{fig_idx}", verbose=True)
        n_warn = sum(1 for i in issues if i["severity"] == "warning")
        n_err = sum(1 for i in issues if i["severity"] == "error")

        from mocoo.visualization.style import save_figure
        save_figure(fig, out_path, facecolor="white")
        plt.close(fig)
        print(f"Saved: {out_path}")
        print(f"{n_warn} warnings | {n_err} errors")


if __name__ == "__main__":
    main()
