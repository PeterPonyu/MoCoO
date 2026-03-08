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

# ── Style constants (17 cm × 21 cm) ────────────────────────────────────────
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
_SHORT = {
    "VAE": "VAE", "VAE+ODE": "V+O", "VAE+MoCo": "V+M",
    "VAE+MoCo+Proto": "V+MP", "VAE+ODE+MoCo": "V+OM", "Full": "Full",
}

_DATASETS = ["IRALL", "dentate", "endo"]
_DS_SHORT = {"IRALL": "IRALL", "dataset_default": "IRALL",
             "dentate": "Dentate", "endo": "Endo"}


def _export_subpanels(fig, sub_dir: Path, panels: list) -> None:
    renderer = fig.canvas.get_renderer()
    for ax, name in panels:
        if ax is None:
            continue
        try:
            bbox = ax.get_tightbbox(renderer)
            if bbox is None:
                continue
            extent = bbox.transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(sub_dir / f"{name}.png", dpi=DPI, bbox_inches=extent)
        except Exception:
            pass


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
        for cfg in _CONFIGS:
            key = cfg.replace("+", "_")
            jf = rdir / f"{key}.json"
            if jf.exists():
                with open(jf) as f:
                    metrics[cfg] = _unify_metric_keys(json.load(f))
    return metrics


def _load_cross_dataset_metrics(results_base: Path) -> dict:
    """Load metrics across all datasets."""
    cross = {}
    for ds_dir in ["dataset_default", "dentate", "endo"]:
        ds_path = results_base / ds_dir
        if not ds_path.exists():
            continue
        ds_metrics = {}
        for cfg in _CONFIGS:
            key = cfg.replace("+", "_")
            jf = ds_path / f"{key}.json"
            if jf.exists():
                with open(jf) as f:
                    ds_metrics[cfg] = _unify_metric_keys(json.load(f))
        if ds_metrics:
            cross[ds_dir] = ds_metrics
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
            elif bv > 1e-6:
                ax.text(bar_obj.get_x() + bar_obj.get_width() / 2,
                        bar_obj.get_height() + 0.005,
                        f"{bv:.2f}", ha="center", va="bottom",
                        fontsize=FS_SMALL - 0.3, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(batch_labels, fontsize=FS_TICK, rotation=0)
    ax.set_ylabel("Score", fontsize=FS_AXIS)
    ax.set_title("Batch Integration Metrics (IRALL, 8 batches)",
                 fontsize=FS_TITLE, pad=4)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.set_ylim(0, 1.25) # Increased y-limit to prevent text overlap with legend/title
    ax.legend(fontsize=FS_LEG, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), frameon=False) # Moved legend down slightly
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.22, linestyle="--", linewidth=0.4, axis="y") # Added grid for readability


# ═══════════════════════════════════════════════════════════════════════════════
# Panel B: Bio-conservation vs Batch-correction scatter
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_bio_batch_scatter(ax, metrics):
    """Scatter: x=batch_correction, y=bio_conservation, sized by overall."""
    configs_present = [c for c in _CONFIGS if c in metrics]

    for cfg in configs_present:
        m = metrics[cfg]
        bc = m.get("batch_correction", np.nan)
        bio = m.get("bio_conservation", np.nan)
        overall = m.get("overall_score", 0.5)
        if np.isnan(bc) or np.isnan(bio):
            continue
        ax.scatter(bc, bio, s=max(30, overall * 200), c=_CONFIG_COLOR[cfg],
                   edgecolors="black", linewidths=0.4, zorder=5, alpha=0.85)
        ax.annotate(_SHORT[cfg], (bc, bio), fontsize=FS_SMALL,
                    xytext=(3, 3), textcoords="offset points")

    # Reference diagonal
    ax.plot([0, 1], [0, 1], "--", color="#cccccc", linewidth=0.5, zorder=1)

    ax.set_xlabel("Batch Correction →", fontsize=FS_AXIS)
    ax.set_ylabel("Bio Conservation →", fontsize=FS_AXIS)
    ax.set_title("Integration Trade-off", fontsize=FS_TITLE, pad=4)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.set_xlim(0.45, 0.60) # Adjusted limits to better fit the data points
    ax.set_ylim(0.75, 0.90) # Adjusted limits to better fit the data points
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.22, linestyle="--", linewidth=0.4) # Added grid for readability


# ═══════════════════════════════════════════════════════════════════════════════
# Panel C: Cross-dataset heatmap
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_cross_dataset_heatmap(ax, cross_data):
    """Heatmap: rows = configs, cols = datasets × metrics."""
    metric_keys = ["ARI", "NMI", "ASW"]
    # Fallback keys for dataset_default which uses full_ARI etc.
    fallback = {"ARI": "full_ARI", "NMI": "full_NMI", "ASW": "full_ASW"}

    datasets = [d for d in ["dataset_default", "dentate", "endo"] if d in cross_data]
    configs_present = [c for c in _CONFIGS
                       if any(c in cross_data[d] for d in datasets)]

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
            col_labels.append(f"{_DS_SHORT.get(ds, ds)}\n{mk}")

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(col_labels, fontsize=FS_SMALL, ha="center")
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(row_labels, fontsize=FS_TICK)

    # Annotate cells with raw values
    for i in range(n_rows):
        for j in range(n_cols):
            v = data[i, j]
            if not np.isnan(v):
                color = "white" if data_norm[i, j] > 0.6 else "black"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=FS_SMALL - 0.3, color=color)

    ax.set_title("Cross-Dataset Performance (normalised)",
                 fontsize=FS_TITLE, pad=4)

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
    datasets = [d for d in ["dataset_default", "dentate", "endo"] if d in cross_data]
    configs_present = [c for c in _CONFIGS
                       if any(c in cross_data[d] for d in datasets)]

    # Add dataset-level labels
    labels = []
    for ds in datasets:
        for mk in metric_keys:
            labels.append(f"{_DS_SHORT.get(ds, ds)} {mk}")

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
    ax.set_xticklabels(labels, fontsize=FS_SMALL - 0.5)
    ax.set_ylim(0, 0.7)
    ax.tick_params(axis="y", labelsize=FS_SMALL - 0.5)
    ax.set_title("Cross-Dataset Profile", fontsize=FS_TITLE, pad=12)
    ax.legend(fontsize=FS_LEG, loc="upper right",
              bbox_to_anchor=(1.3, 1.1), frameon=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Main figure builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_figure(results_base: Path, outdir: Path):
    """Build the complete Figure 7."""
    # Load batch metrics (IRALL)
    irall_dir = results_base / "dataset_default"
    batch_metrics = _load_batch_metrics(irall_dir)

    # Load cross-dataset metrics
    cross_data = _load_cross_dataset_metrics(results_base)

    # ── Create figure ──
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.50, wspace=0.35,
                           height_ratios=[1, 0.9, 1.1, 1.1])

    # Panel A: Batch integration bars (span full width)
    ax_A = fig.add_subplot(gs[0, :])
    _draw_batch_bars(ax_A, batch_metrics)

    # Panel B: Bio vs Batch scatter
    ax_B = fig.add_subplot(gs[1, 0])
    _draw_bio_batch_scatter(ax_B, batch_metrics)

    # Panel B2: Overall score bar
    ax_B2 = fig.add_subplot(gs[1, 1])
    configs_present = [c for c in _CONFIGS if c in batch_metrics]
    overall = [batch_metrics[c].get("overall_score", 0) for c in configs_present]
    bars = ax_B2.barh(
        [_SHORT[c] for c in configs_present], overall,
        color=[_CONFIG_COLOR[c] for c in configs_present],
        edgecolor="white", linewidth=0.3, height=0.6,
    )
    for bar_obj, val in zip(bars, overall):
        ax_B2.text(bar_obj.get_width() + 0.005, bar_obj.get_y() + bar_obj.get_height() / 2,
                   f"{val:.3f}", ha="left", va="center", fontsize=FS_SMALL)
    ax_B2.set_xlabel("Overall Score (0.4·bio + 0.6·batch)", fontsize=FS_AXIS)
    ax_B2.set_title("scIB Overall Score", fontsize=FS_TITLE, pad=4)
    ax_B2.tick_params(axis="both", labelsize=FS_TICK)
    ax_B2.spines["top"].set_visible(False)
    ax_B2.spines["right"].set_visible(False)

    # Panel C: Cross-dataset heatmap (span full width)
    ax_C = fig.add_subplot(gs[2, :])
    _draw_cross_dataset_heatmap(ax_C, cross_data)

    # Panel D: Cross-dataset radar
    ax_D = fig.add_subplot(gs[3, :], polar=True)
    _draw_cross_radar(ax_D, cross_data)

    fig.subplots_adjust(left=0.12, right=0.94, top=0.96, bottom=0.06)

    _panel_label(fig, ax_A, "A")
    _panel_label(fig, ax_B, "B")
    _panel_label(fig, ax_C, "C")
    _panel_label(fig, ax_D, "D")

    fig.canvas.draw()

    print("\n── Conflict Detection ──")
    issues = detect_all_conflicts(fig, label="batch_integration", verbose=True)

    outpath = outdir / "batch_integration.png"
    fig.savefig(outpath, dpi=DPI)

    # Export individual panels
    sub_dir = outdir / "fig7_batch_integration"
    sub_dir.mkdir(parents=True, exist_ok=True)
    _export_subpanels(fig, sub_dir, [
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
    build_figure(rdir, outdir)


if __name__ == "__main__":
    main()
