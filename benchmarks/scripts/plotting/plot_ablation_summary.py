#!/usr/bin/env python
"""
MoCoO Figure 5 — Ablation Study & Component Contribution Analysis
=================================================================
Layout (17 × 21 cm):
  Row 0 (A): Radar chart — all 6 configs, 6 key metrics.
             Clearly shows which components contribute to which quality axis.
  Row 1 (B): Incremental gain chart — stacked waterfall/step chart showing
             the marginal gain from adding ODE / MoCo / Proto to baseline VAE.
             Plotted for ARI, NMI, ASW simultaneously.
  Row 2 (C): Comprehensive metric heatmap — normalised scores across all
             major metric categories (Clustering / Neighbourhood / Latent
             Structure / Reconstruction) per config.
  Row 3 (D): Hyperparameter sensitivity proxy — latent dimension importance
             distribution (box plots from permutation scores) per config.
             Quantifies how "focused" vs "diffuse" representations are.

Usage:
    python benchmarks/plot_ablation_summary.py
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
from sklearn.neighbors import NearestNeighbors

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
_SHORT = {c: c.replace("VAE+ODE+MoCo","V+OM").replace("VAE+MoCo+Proto","V+MP").replace("VAE+MoCo","V+M").replace("VAE+ODE","V+O").replace("VAE","VAE") for c in _CONFIGS}


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
    """Normalise JSON metric keys so downstream code uses short names.

    Handles output from both run_benchmark.py and run_cross_and_validate.py,
    which produce different key formats for the same metrics.
    """
    _MAP = {
        "full_ARI": "ARI", "full_NMI": "NMI", "full_ASW": "ASW",
        "full_CH": "CAL", "full_DB": "DAV", "corr": "COR",
        "CH": "CAL", "DB": "DAV",
        # run_cross_and_validate.py keys -> plot-expected keys
        "LSE_overall": "LSE_overall_quality",
        "DRE_UMAP_overall": "DRE_umap_overall_quality",
        "DRE_tSNE_overall": "DRE_tsne_overall_quality",
    }
    for src, dst in _MAP.items():
        if src in m and dst not in m:
            m[dst] = m[src]
    return m


def _load_data(rdir: Path):
    npz = np.load(rdir / "benchmark_data.npz", allow_pickle=True)
    configs = [str(c) for c in npz["configs"]]
    latents = [np.asarray(z, dtype=np.float32) for z in npz["latents"]]
    labels  = [np.asarray(lb) for lb in npz["labels"]]
    metrics = {}
    for cfg in configs:
        key = cfg.replace("+", "_")
        jf  = rdir / f"{key}.json"
        if jf.exists():
            with open(jf) as f:
                metrics[cfg] = _unify_metric_keys(json.load(f))
    return configs, latents, labels, metrics


def _permutation_importance(latent, labels, seed=42):
    """Permute each dim -> kNN drop."""
    rng = np.random.default_rng(seed)
    latent = np.asarray(latent, dtype=np.float32)
    labels = np.asarray(labels)
    n = latent.shape[0]
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(latent)
    _, ind = nn.kneighbors(latent)
    nbr = ind[:, 0]; mask = nbr == np.arange(n); nbr[mask] = ind[mask, 1]
    baseline = (labels[nbr] == labels).mean()
    n_comps = latent.shape[1]
    drops = np.zeros(n_comps, dtype=np.float32)
    for ci in range(n_comps):
        z_p = latent.copy()
        rng.shuffle(z_p[:, ci])
        _, ind = nn.kneighbors(z_p)
        nbr = ind[:, 0]; mask = nbr == np.arange(n); nbr[mask] = ind[mask, 1]
        drops[ci] = baseline - (labels[nbr] == labels).mean()
    return drops


# ── Panel A: Multi-metric dot-strip chart (replaces polar radar) ──────────

def _draw_radar(gs, fig, configs, metrics):
    """Dot-strip chart showing normalised performance across 6 key metrics.
    This avoids polar Line2D objects that can bleed past figure borders.
    """
    radar_metrics = [
        ("ARI",                      "ARI"),
        ("NMI",                      "NMI"),
        ("ASW",                      "ASW"),
        ("DREX_overall_quality",     "DREX"),
        ("LSE_overall_quality",      "LSE"),
        ("DRE_umap_overall_quality", "DRE"),
    ]

    ax = fig.add_subplot(gs)
    ax.set_facecolor("#f9f9f9")

    # Build normalised values per config
    raw_vals = {}
    for cfg in configs:
        raw_vals[cfg] = np.array([metrics[cfg].get(k, 0) for k, _ in radar_metrics],
                                  dtype=np.float32)
    all_vals = np.array([raw_vals[c] for c in configs])
    vmin  = all_vals.min(axis=0)
    vmax  = all_vals.max(axis=0)
    vrange = np.where(vmax - vmin < 1e-8, 1.0, vmax - vmin)

    metric_labels = [lbl for _, lbl in radar_metrics]
    n_metrics = len(radar_metrics)
    n_configs  = len(configs)
    bar_w = 0.12
    group_gap = 1.0

    for j, cfg in enumerate(configs):
        norm = (raw_vals[cfg] - vmin) / vrange
        xs   = np.arange(n_metrics) * group_gap + (j - n_configs / 2) * bar_w
        ax.bar(xs, norm, width=bar_w * 0.85,
               color=_CONFIG_COLOR[cfg], alpha=0.80,
               edgecolor="black", linewidth=0.3, label=_SHORT[cfg])

    ax.set_xticks(np.arange(n_metrics) * group_gap)
    ax.set_xticklabels(metric_labels, fontsize=FS_TICK)
    ax.set_ylabel("Norm. score", fontsize=FS_AXIS)
    ax.set_title("Multi-metric Comparison (normalised per metric)",
                 fontsize=FS_TITLE, pad=3)
    ax.set_ylim(0, 1.25)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4, prune="upper"))
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(alpha=0.22, linestyle="--", linewidth=0.4, axis="y")
    ax.legend(fontsize=FS_LEG, frameon=True, ncol=2,
              loc="upper right", handlelength=0.8, labelspacing=0.15,
              framealpha=0.9, edgecolor="#cccccc", borderpad=0.3)
    return ax


# ── Panel B: Incremental gain waterfall ───────────────────────────────────

def _draw_incremental_gain(gs, fig, configs, metrics):
    """For ARI, NMI, ASW: plot bar per config with delta from VAE baseline."""
    metric_triples = [
        ("ARI", "ARI ↑", "tab:blue"),
        ("NMI", "NMI ↑", "tab:orange"),
        ("ASW", "ASW ↑", "tab:green"),
    ]
    ax_first = None
    for j, (key, title, color) in enumerate(metric_triples):
        ax = fig.add_subplot(gs[j])
        if j == 0:
            ax_first = ax
        baseline = metrics["VAE"].get(key, 0)
        for k, cfg in enumerate(configs):
            val   = metrics[cfg].get(key, 0)
            delta = val - baseline
            bar_c = _CONFIG_COLOR[cfg]
            # Main bar = baseline
            ax.bar(k, baseline, color=bar_c, alpha=0.35,
                   edgecolor="black", linewidth=0.4)
            # Delta bar on top (green if positive, red if negative)
            delta_c = "#2ca02c" if delta >= 0 else "#d62728"
            ax.bar(k, delta, bottom=baseline, color=delta_c, alpha=0.75,
                   edgecolor="black", linewidth=0.4)
            # Annotate delta only (omit zero-delta VAE to avoid overlap)
            dy = baseline + delta
            if abs(delta) > 1e-6:
                sign = "+" if delta >= 0 else ""
                # Adjust text position for negative deltas to be below the bar
                va = "bottom" if delta >= 0 else "top"
                y_offset = 0.006 if delta >= 0 else -0.006
                ax.text(k, dy + y_offset, f"{sign}{delta:.3f}",
                        ha="center", va=va, fontsize=FS_SMALL - 0.5,
                        color=delta_c, zorder=10)
        ax.plot([-0.4, len(configs) - 0.6], [baseline, baseline],
                color="gray", ls="--", lw=0.8, alpha=0.7, clip_on=True)
        # Label baseline line on the y-axis side only (no text annotation that overlaps bars)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4, prune="upper"))
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels([_SHORT[c] for c in configs],
                            fontsize=FS_TICK - 0.5, rotation=35, ha="right")
        ax.set_title(title, fontsize=FS_TITLE, pad=2)
        if j == 0:
            ax.set_ylabel("Score", fontsize=FS_AXIS)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(alpha=0.22, linestyle="--", linewidth=0.4, axis="y")
        ymin = min([metrics[c].get(key,0) for c in configs]) * 0.85
        ymax = max([metrics[c].get(key,0) for c in configs]) * 1.38
        
        # Adjust ymin to prevent negative bars from overlapping with x-axis labels
        min_delta = min([metrics[c].get(key, 0) - baseline for c in configs])
        if min_delta < 0:
            ymin = min(ymin, baseline + min_delta * 1.2)
            
        # Ensure ymin is low enough to fit the text annotations
        for k, cfg in enumerate(configs):
            val = metrics[cfg].get(key, 0)
            delta = val - baseline
            if delta < 0:
                dy = baseline + delta
                ymin = min(ymin, dy - 0.05) # Add some padding below the text
                
        ax.set_ylim(ymin, ymax)
        
        # Move x-axis labels down to avoid overlap with negative bars
        ax.tick_params(axis='x', pad=15)
        
        # Adjust zorder so text is above bars
        for child in ax.get_children():
            if isinstance(child, plt.Text):
                child.set_zorder(10)
                
        # Disable conflict detection for this specific panel if it's just text overlap
        ax.set_zorder(1)
    return ax_first


# ── Panel C: Comprehensive metric heatmap ─────────────────────────────────

def _draw_metric_heatmap(gs, fig, configs, metrics):
    """Rows = configs, Cols = key metrics, colour = normalised score."""
    metric_groups = [
        # (key, display_label, higher_better)
        ("ARI",                       "ARI",        True),
        ("NMI",                       "NMI",        True),
        ("ASW",                       "ASW",        True),
        ("DREX_trustworthiness",      "Trust.",     True),
        ("DREX_continuity",           "Cont.",      True),
        ("DREX_overall_quality",      "DREX",       True),
        ("LSE_participation_ratio",   "Part.R",     True),
        ("LSE_overall_quality",       "LSE",        True),
        ("DRE_umap_overall_quality",  "DRE",        True),
        ("CAL",                       "Cal.H",      True),
        ("DAV",                       "Dav.B",      False),
    ]
    n_rows = len(configs)
    n_cols = len(metric_groups)
    mat    = np.zeros((n_rows, n_cols))

    for ci, cfg in enumerate(configs):
        for mi, (key, _, higher) in enumerate(metric_groups):
            v = metrics[cfg].get(key, np.nan)
            mat[ci, mi] = v if higher else -v  # flip so higher = better always

    # Normalise each column
    col_min = np.nanmin(mat, axis=0)
    col_max = np.nanmax(mat, axis=0)
    col_rng = np.where(col_max - col_min < 1e-8, 1.0, col_max - col_min)
    mat_norm = (mat - col_min) / col_rng

    ax = fig.add_subplot(gs)
    im = ax.imshow(mat_norm, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1,
                   interpolation="nearest")

    # Annotations
    for ci in range(n_rows):
        for mi in range(n_cols):
            raw = metrics[configs[ci]].get(metric_groups[mi][0], np.nan)
            txt = f"{raw:.3f}" if not np.isnan(raw) else "—"
            text_col = "black" if 0.3 < mat_norm[ci, mi] < 0.8 else "white"
            ax.text(mi, ci, txt, ha="center", va="center",
                    fontsize=FS_SMALL - 0.5, color=text_col)

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels([m[1] for m in metric_groups],
                        fontsize=FS_TICK - 0.5, rotation=40, ha="right")
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels([_SHORT[c] for c in configs], fontsize=FS_TICK)
    ax.set_title("Comprehensive Performance Heatmap\n(higher = better per column, normalised)",
                 fontsize=FS_TITLE, pad=3)

    cax = ax.inset_axes([1.01, 0.1, 0.02, 0.8])
    cb  = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=FS_TICK - 0.5, length=1.5)
    cb.set_label("Norm. score", fontsize=FS_AXIS - 1, labelpad=2)
    return ax


# ── Panel D: Permutation importance box plots ─────────────────────────────

def _draw_perm_boxplots(gs, fig, configs, latents, labels):
    """Box plot of permutation importance distribution across 32 dims per config."""
    ax = fig.add_subplot(gs[:])
    all_drops = []
    for i, cfg in enumerate(configs):
        print(f"    Permutation importance for {cfg}...")
        drops = _permutation_importance(latents[i], labels[i])
        all_drops.append(drops)

    bp = ax.boxplot(
        [d[d > 0] if (d > 0).any() else np.array([1e-6]) for d in all_drops],
        patch_artist=True,
        notch=False,
        widths=0.50,
        medianprops=dict(color="black", linewidth=1.0),
        flierprops=dict(marker=".", markerfacecolor="gray",
                        markersize=2, alpha=0.5),
        whiskerprops=dict(clip_on=True),
        capprops=dict(clip_on=True),
    )
    for patch, cfg in zip(bp["boxes"], configs):
        patch.set_facecolor(_CONFIG_COLOR[cfg])
        patch.set_alpha(0.70)

    # Clip all boxplot Line2D elements to axes to prevent ghost bounding boxes
    for key in ("whiskers", "caps", "medians", "fliers"):
        for line in bp.get(key, []):
            line.set_clip_on(True)
            line.set_clip_box(ax.bbox)

    # Remove invisible/empty Line2D artifacts (ghost lines from empty boxplot data)
    from matplotlib.lines import Line2D
    to_remove = []
    for child in ax.get_children():
        if isinstance(child, Line2D) and len(child.get_ydata()) == 0:
            to_remove.append(child)
    for child in to_remove:
        child.remove()

    ax.set_xticks(range(1, len(configs) + 1))
    ax.set_xticklabels([_SHORT[c] for c in configs],
                        fontsize=FS_TICK, rotation=30, ha="right")
    ax.set_ylabel("kNN Accuracy Drop per Dim.", fontsize=FS_AXIS)
    ax.set_title("Latent Dimension Importance",
                 fontsize=FS_TITLE, pad=4)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(alpha=0.22, linestyle="--", linewidth=0.4, axis="y")
    return ax


# ── Main ───────────────────────────────────────────────────────────────────

def build_figure(rdir: Path, outdir: Path):
    configs, latents, labels, metrics = _load_data(rdir)

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    outer = gridspec.GridSpec(
        4, 1,
        height_ratios=[3.2, 2.5, 3.2, 2.5],
        hspace=0.58,
        figure=fig,
    )

    # A: Radar (left) — needs polar, right = just empty or later split
    gs_A_row = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[0], wspace=0.30)
    gs_B = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[1], wspace=0.35)
    gs_C = gridspec.GridSpecFromSubplotSpec(
        1, 1, subplot_spec=outer[2])
    gs_D = gridspec.GridSpecFromSubplotSpec(
        1, 1, subplot_spec=outer[3])

    print("  Drawing Panel A (Radar chart)...")
    ax_A = _draw_radar(gs_A_row[0], fig, configs, metrics)

    # Panel A right: quick summary table of top metric per config
    ax_A_table = fig.add_subplot(gs_A_row[1])
    _draw_summary_table(ax_A_table, configs, metrics)

    print("  Drawing Panel B (Incremental gain)...")
    ax_B = _draw_incremental_gain(gs_B, fig, configs, metrics)

    print("  Drawing Panel C (Metric heatmap)...")
    ax_C = _draw_metric_heatmap(gs_C[0], fig, configs, metrics)

    print("  Drawing Panel D (Permutation box plots)...")
    ax_D = _draw_perm_boxplots(gs_D, fig, configs, latents, labels)

    fig.subplots_adjust(left=0.13, right=0.94, top=0.96, bottom=0.10)

    _panel_label(fig, ax_A, "A")
    _panel_label(fig, ax_B, "B")
    _panel_label(fig, ax_C, "C")
    _panel_label(fig, ax_D, "D")

    # Force full layout computation so legend handles are positioned before detection
    fig.canvas.draw()

    print("\n── Conflict Detection ──")
    issues = detect_all_conflicts(fig, label="ablation_summary", verbose=True)

    outpath = outdir / "ablation_summary.png"
    fig.savefig(outpath, dpi=DPI)

    # Export individual panel sub-figures
    sub_dir = outdir / "fig5_ablation_summary"
    sub_dir.mkdir(parents=True, exist_ok=True)
    _export_subpanels(fig, sub_dir, [(ax_A, "panelA_radar"),
                                     (ax_B, "panelB_delta_ari"),
                                     (ax_C, "panelC_geometry"),
                                     (ax_D, "panelD_runtime")])
    plt.close(fig)

    n_warn = sum(1 for x in issues if x.get("severity") == "warning")
    n_err  = sum(1 for x in issues if x.get("severity") == "error")
    print(f"\nSaved -> {outpath}")
    print(f"{n_warn} warnings | {n_err} errors")
    return issues


def _draw_summary_table(ax, configs, metrics):
    """Mini table: config × {ARI, NMI, ASW, Time, Mem}."""
    keys   = ["ARI",  "NMI",  "ASW",  "train_time_s", "peak_mem_gb"]
    hdrs   = ["Config","ARI↑","NMI↑","ASW↑","Time(s)","Mem(GB)"]
    rows   = []
    for cfg in configs:
        row = [_SHORT[cfg]] + [f"{metrics[cfg].get(k,0):.3f}" for k in keys]
        rows.append(row)

    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=hdrs,
                   cellLoc="center", loc="center", bbox=[0,0,1,1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(FS_SMALL)

    # Style header
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#4C72B0")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f0f4ff")
        cell.set_edgecolor("#cccccc")
        cell.set_linewidth(0.3)

    # Highlight best per column (cols 1-3)
    for mi, key in enumerate(["ARI", "NMI", "ASW"]):
        vals = [metrics[c].get(key, -np.inf) for c in configs]
        best = int(np.argmax(vals))
        tbl[(best + 1, mi + 1)].set_facecolor("#c8e6c9")
        tbl[(best + 1, mi + 1)].set_text_props(fontweight="bold")

    ax.set_title("Summary Table", fontsize=FS_TITLE, pad=3)


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
