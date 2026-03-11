#!/usr/bin/env python
"""
MoCoO Figure 7 — Batch Integration & Cross-Dataset Generalization
=================================================================
Layout (17 × 21 cm):
  Row 0 (A): Grouped bar chart — iLISI, bASW, cLISI across all 6 configs
             (IRALL dataset, 8 time-point batches).
  Row 1 (B): Bio-conservation vs batch-correction scatter — shows the
             trade-off between preserving biological variance and removing
             batch effects.  Each point is one config.
  Row 2 (C): Cross-dataset comparison heatmap — ARI/NMI/ASW per config
             across IRALL, dentate, endo datasets (normalised within metric).
  Row 3 (D): Cross-dataset radar chart — config performance profiles
             averaged across datasets.

Usage:
    python -m benchmarks.scripts.plotting.plot_batch_integration
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
from mocoo.visualization.style import (
    FIG_WIDTH_IN, FIG_HEIGHT_IN, DPI,
    FS_LABEL, FS_TITLE, FS_AXIS, FS_TICK, FS_LEGEND, FS_SMALL,
    HEATMAP_DARK_THRESHOLD,
    apply_style, get_config_order, get_config_colors, get_short_name, get_tick_name,
)

setup_fonts()
apply_style()
FS_LEG = FS_LEGEND

_CONFIGS = get_config_order()
_CONFIG_COLOR = get_config_colors()
_SHORT = {c: get_tick_name(c) for c in _CONFIGS}

_DATASETS = ["IRALL", "dentate", "endo"]
_DS_SHORT = {"IRALL": "IRALL", "dataset_default": "IRALL",
             "dentate": "Dentate", "endo": "Endo"}


def _resolve_dataset_dir(results_base: Path, ds_name: str) -> Path | None:
    """Find the actual directory for a dataset, checking multiple layouts.

    Priority:
      1. results/cross_dataset/{ds_name}/   (new layout)
      2. results/single_dataset/            (for IRALL single-dataset runs)
      3. results/{ds_name}/                 (flat old layout)
      4. results/_legacy_50ep/{ds_name}/    (archived legacy, old naming)
      5. results/dataset_default/           (old IRALL alias)
    """
    candidates = [
        results_base / "cross_dataset" / ds_name,
    ]
    if ds_name == "IRALL":
        candidates.append(results_base / "single_dataset")
    candidates.extend([
        results_base / ds_name,
        results_base / "_legacy_50ep" / ds_name,
    ])
    if ds_name == "IRALL":
        candidates.extend([
            results_base / "dataset_default",
            results_base / "_legacy_50ep" / "dataset_default",
        ])
    for c in candidates:
        if c.exists():
            return c
    return None


def _load_batch_metrics(rdir: Path) -> dict:
    """Load per-config batch metrics from summary_batch.csv."""
    metrics = {}
    csv_file = rdir / "summary_batch.csv"
    if csv_file.exists():
        import pandas as pd
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            cfg = row['config']
            if cfg in _CONFIGS:
                metrics[cfg] = row.to_dict()
    else:
        # Fallback to JSON if CSV doesn't exist
        metrics = load_config_metrics(rdir, _CONFIGS)
    return metrics


def _load_cross_dataset_metrics(results_base: Path) -> dict:
    """Load metrics across all datasets."""
    cross = {}
    for ds_name in _DATASETS:
        ds_path = _resolve_dataset_dir(results_base, ds_name)
        if ds_path is None:
            continue
        ds_key = ds_name  # Always use canonical name as key
        ds_metrics = load_config_metrics(ds_path, _CONFIGS)
        if ds_metrics:
            cross[ds_key] = ds_metrics
    return cross


# ═══════════════════════════════════════════════════════════════════════════════
# Panel A: Batch integration grouped bar chart
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_batch_bars(ax, metrics):
    """Grouped bar chart for iLISI, bASW, cLISI."""
    batch_keys = ["iLISI", "bASW", "cLISI", "graph_conn", "iso_label_ASW"]
    batch_labels = ["iLISI ↑", "bASW ↑", "cLISI ↑", "G.Conn ↑", "Iso.ASW ↑"]

    configs_present = [c for c in _CONFIGS if c in metrics]
    n_cfg = len(configs_present)
    n_metrics = len(batch_keys)

    # Check if ANY config has actual batch metric data
    has_any_data = False
    for cfg in configs_present:
        m = metrics[cfg]
        for k in batch_keys:
            v = m.get(k, np.nan)
            if v is not None and not np.isnan(v) and v > 1e-6:
                has_any_data = True
                break
        if has_any_data:
            break

    if not has_any_data:
        ax.set_visible(False)
        return

    x = np.arange(n_metrics)
    width = 0.12

    for i, cfg in enumerate(configs_present):
        m = metrics[cfg]
        vals = [m.get(k, np.nan) for k in batch_keys]
        # Replace NaN with 0 for bar heights but track which are missing
        bar_vals = [0 if np.isnan(v) else v for v in vals]
        offset = (i - n_cfg / 2 + 0.5) * width
        bars = ax.bar(x + offset, bar_vals, width * 0.9,
                      label=_SHORT[cfg], color=_CONFIG_COLOR[cfg],
                      edgecolor="white", linewidth=0.3)
        # Hatch bars with missing data
        for bar_obj, raw_val, bv in zip(bars, vals, bar_vals):
            if np.isnan(raw_val):
                bar_obj.set_hatch("//")
                bar_obj.set_alpha(0.25)

    ax.set_xticks(x)
    ax.set_xticklabels(batch_labels, fontsize=FS_SMALL, rotation=0)
    ax.set_ylabel("Score", fontsize=FS_AXIS)
    ax.set_title("Batch Metrics",
                 fontsize=FS_AXIS, pad=1)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.set_ylim(0, 1.25)
    ax.set_xlim(-0.5, len(batch_keys) - 0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.22, linestyle="--", linewidth=0.4, axis="y")


# ═══════════════════════════════════════════════════════════════════════════════
# Panel B: Bio-conservation vs Batch-correction scatter
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_bio_batch_scatter(ax, metrics):
    """Scatter: x=batch_correction, y=bio_conservation, sized by overall."""
    configs_present = [c for c in _CONFIGS if c in metrics]

    plotted = False
    for cfg in configs_present:
        m = metrics[cfg]
        bc = m.get("batch_correction", np.nan)
        bio = m.get("bio_conservation", np.nan)
        overall = m.get("overall_score", 0.5)
        if np.isnan(bc) or np.isnan(bio):
            continue
        ax.scatter(bc, bio, s=max(30, overall * 200), c=_CONFIG_COLOR[cfg],
                   edgecolors="black", linewidths=0.4, zorder=5, alpha=0.85,
                   label=_SHORT[cfg])
        plotted = True

    if not plotted:
        ax.set_visible(False)
        return

    # Reference diagonal
    ax.plot([0, 1], [0, 1], "--", color="#cccccc", linewidth=0.5, zorder=1)

    ax.set_xlabel("Batch Corr.", fontsize=FS_SMALL)
    ax.set_ylabel("Bio Conservation \u2192", fontsize=FS_AXIS)
    ax.set_title("Integration Trade-off", fontsize=FS_TITLE, pad=1)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.22, linestyle="--", linewidth=0.4)


# ═══════════════════════════════════════════════════════════════════════════════
# Panel C: Cross-dataset heatmap
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_cross_dataset_heatmap(ax, cross_data):
    """Heatmap: rows = configs, cols = datasets × metrics."""
    metric_keys = ["ARI", "NMI", "ASW"]
    # Fallback keys for dataset_default which uses full_ARI etc.
    fallback = {"ARI": "full_ARI", "NMI": "full_NMI", "ASW": "full_ASW"}

    datasets = [d for d in _DATASETS if d in cross_data]
    configs_present = [c for c in _CONFIGS
                       if any(c in cross_data[d] for d in datasets)]

    if not datasets or not configs_present:
        ax.set_visible(False)
        return

    n_ds = len(datasets)
    n_met = len(metric_keys)
    n_cols = n_ds * n_met
    n_rows = len(configs_present)

    data = np.full((n_rows, n_cols), np.nan)
    for i, cfg in enumerate(configs_present):
        for j, ds in enumerate(datasets):
            if cfg not in cross_data[ds]:
                continue
            m = cross_data[ds][cfg]
            for k, mk in enumerate(metric_keys):
                v = m.get(mk, m.get(fallback.get(mk, ""), np.nan))
                data[i, j * n_met + k] = v

    # Normalise per column (metric × dataset) for heatmap
    data_norm = data.copy()
    for c in range(n_cols):
        col = data[:, c]
        mn, mx = np.nanmin(col), np.nanmax(col)
        if mx > mn:
            data_norm[:, c] = (col - mn) / (mx - mn)

    im = ax.imshow(data_norm, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

    # Labels
    row_labels = [_SHORT[c] for c in configs_present]
    col_labels = []
    for ds in datasets:
        for mk in metric_keys:
            col_labels.append(f"{mk}")

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(col_labels, fontsize=FS_SMALL-2, ha="center", rotation=90)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(row_labels, fontsize=FS_SMALL)

    # Annotate cells with raw values
    for i in range(n_rows):
        for j in range(n_cols):
            v = data[i, j]
            if not np.isnan(v):
                color = "white" if data_norm[i, j] > HEATMAP_DARK_THRESHOLD else "black"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=FS_SMALL, color=color)

    ax.set_title("Cross-Dataset Perf.",
                 fontsize=FS_AXIS, pad=1)

    # Vertical separators between datasets
    for sep in range(1, n_ds):
        ax.axvline(sep * n_met - 0.5, color="white", linewidth=1.5)


# ═══════════════════════════════════════════════════════════════════════════════
# Panel D: Cross-dataset radar
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_cross_radar(ax, cross_data):
    """Radar chart averaging ARI/NMI/ASW across datasets for each config."""
    metric_keys = ["ARI", "NMI", "ASW"]
    fallback = {"ARI": "full_ARI", "NMI": "full_NMI", "ASW": "full_ASW"}
    datasets = [d for d in _DATASETS if d in cross_data]
    configs_present = [c for c in _CONFIGS
                       if any(c in cross_data[d] for d in datasets)]

    # Add dataset-level labels
    labels = []
    for ds in datasets:
        for mk in metric_keys:
            labels.append(f"{mk}")

    n_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    angles += angles[:1]

    for cfg in configs_present:
        vals = []
        for ds in datasets:
            m = cross_data.get(ds, {}).get(cfg, {})
            for mk in metric_keys:
                v = m.get(mk, m.get(fallback.get(mk, ""), 0))
                vals.append(max(0, v) if (v is not None and not np.isnan(v)) else 0)
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=0.8, color=_CONFIG_COLOR[cfg],
                label=_SHORT[cfg])
        ax.fill(angles, vals, alpha=0.06, color=_CONFIG_COLOR[cfg])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=FS_SMALL - 1)
    ax.set_ylim(0, 0.7)
    ax.set_yticks([0.2, 0.5])
    ax.tick_params(axis="y", labelsize=FS_SMALL)
    ax.set_title("Cross-Dataset Profile", fontsize=FS_AXIS, pad=18)
    ax.legend(fontsize=FS_SMALL - 1, loc="upper right",
              bbox_to_anchor=(1.32, 1.02), frameon=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Main figure builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_figure(results_base: Path, outdir: Path):
    """Build the complete Figure 7."""
    # Load batch metrics (IRALL)
    irall_dir = _resolve_dataset_dir(results_base, "IRALL")
    batch_metrics = _load_batch_metrics(irall_dir) if irall_dir else {}

    # Load cross-dataset metrics
    cross_data = _load_cross_dataset_metrics(results_base)

    # ── Create figure with more vertical room ──
    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN * 1.10))

    # Panel A: Batch integration bars (full width) — more headroom for labels
    ax_A = fig.add_axes([0.10, 0.82, 0.86, 0.14])
    _draw_batch_bars(ax_A, batch_metrics)

    # Panel B: Bio vs Batch scatter — separate from B2 with real gap
    ax_B = fig.add_axes([0.10, 0.58, 0.38, 0.18])
    _draw_bio_batch_scatter(ax_B, batch_metrics)

    # Panel B2: Overall score bar — proper horizontal gap
    ax_B2 = fig.add_axes([0.56, 0.58, 0.40, 0.18])
    configs_present = [c for c in _CONFIGS if c in batch_metrics]
    overall = [batch_metrics[c].get("overall_score", 0) for c in configs_present]
    if not configs_present or all(v == 0 for v in overall):
        ax_B2.set_visible(False)
    else:
        bars = ax_B2.barh(
            [_SHORT[c] for c in configs_present], overall,
            color=[_CONFIG_COLOR[c] for c in configs_present],
            edgecolor="white", linewidth=0.3, height=0.6,
        )
        for bar_obj, val in zip(bars, overall):
            ax_B2.text(bar_obj.get_width() + 0.005, bar_obj.get_y() + bar_obj.get_height() / 2,
                       f"{val:.3f}", ha="left", va="center", fontsize=FS_SMALL)
        ax_B2.set_xlabel("Overall Score", fontsize=FS_SMALL)
        ax_B2.set_title("scIB Score", fontsize=FS_AXIS, pad=1)
        ax_B2.tick_params(axis="both", labelsize=FS_TICK)
        ax_B2.set_xlim(0, 1.0)
        ax_B2.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax_B2.spines["top"].set_visible(False)
        ax_B2.spines["right"].set_visible(False)

    # Panel C: Cross-dataset heatmap (full width) — lower, with gap from B
    ax_C = fig.add_axes([0.10, 0.38, 0.86, 0.14])
    _draw_cross_dataset_heatmap(ax_C, cross_data)

    # Panel D: Cross-dataset radar (polar) — narrower to leave room for legend
    ax_D = fig.add_axes([0.10, 0.08, 0.48, 0.22], polar=True)
    _draw_cross_radar(ax_D, cross_data)

    panel_label(fig, ax_A, "A", x_off=-0.05, y_off=0.020)
    panel_label(fig, ax_B, "B", x_off=-0.05, y_off=0.020)
    panel_label(fig, ax_C, "C", x_off=-0.05, y_off=0.020)
    panel_label(fig, ax_D, "D", x_off=-0.05, y_off=0.020)

    fig.canvas.draw()

    print("\n── Conflict Detection ──")
    issues = detect_all_conflicts(fig, label="batch_integration", verbose=True)

    outpath = outdir / "supp_batch_integration.png"
    from mocoo.visualization.style import save_figure
    save_figure(fig, outpath)

    # Export individual panels
    sub_dir = outdir / "supp_batch_integration"
    sub_dir.mkdir(parents=True, exist_ok=True)
    export_subpanels(fig, sub_dir, [
        (ax_A, "panelA_batch_bars"),
        (ax_B, "panelB_bio_batch_scatter"),
        (ax_B2, "panelB2_overall_score"),
        (ax_C, "panelC_cross_dataset_heatmap"),
        (ax_D, "panelD_cross_radar"),
    ])
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
                   default=str(_benchmarks / "results"))
    p.add_argument("--outdir",
                   default=str(_benchmarks / "figures"))
    args = p.parse_args()
    rdir   = Path(args.resultsdir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    return build_figure(rdir, outdir)


if __name__ == "__main__":
    main()
