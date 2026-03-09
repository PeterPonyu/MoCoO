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

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from benchmarks.scripts.pipeline.visual_conflict_detector import detect_all_conflicts

# ── Import centralized style ────────────────────────────────────────────────
from mocoo.visualization.style import (
    FIG_WIDTH_IN as FIG_W, FIG_HEIGHT_IN as FIG_H, DPI, SAVEFIG_KW,
    FS_LABEL, FS_TITLE, FS_AXIS, FS_TICK, FS_LEGEND as FS_LEG, FS_SMALL,
    HEATMAP_DARK_THRESHOLD,
    get_config_colors, get_config_order, get_short_name, apply_style,
)

apply_style()

# ── Fonts ──────────────────────────────────────────────────────────────────
_FONT_DIR = Path(__file__).resolve().parent.parent.parent / "fonts"
for _fp in (_FONT_DIR / "Arial.ttf", _FONT_DIR / "Arial Bold.ttf"):
    if _fp.exists():
        fm.fontManager.addfont(str(_fp))
if (_FONT_DIR / "Arial.ttf").exists():
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["Arial"] + list(
        matplotlib.rcParams.get("font.sans-serif", []))

_CONFIGS = get_config_order()
_CONFIG_COLOR = get_config_colors()
_SHORT = {c: get_short_name(c) for c in _CONFIGS}


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


# ── Panel A: ODE × MoCo Synergy Heatmap ──────────────────────────────────

def _load_beta_metrics(rdir: Path) -> dict:
    """Load metrics from beta ablation subdirectories.

    Returns {beta_label: {config_name: metrics_dict}}.
    Searches in order:
      1. results/beta_ablation/beta_{value}/  (new layout)
      2. results/beta{value}/                 (old layout)
      3. results/_legacy_50ep/beta{value}/    (archived legacy)
    Falls back to the single rdir if no beta subdirectories are found.
    """
    results_root = rdir.parent  # e.g. benchmarks/results/
    beta_values = ["1.0", "0.1", "0.01"]
    out = {}
    for bval in beta_values:
        label = rf"$\beta$={bval}"
        # Try each candidate path in priority order
        candidates = [
            results_root / "beta_ablation" / f"beta_{bval}",
            results_root / f"beta{bval}",
            results_root / "_legacy_50ep" / f"beta{bval}",
        ]
        bdir = None
        for c in candidates:
            if c.exists():
                bdir = c
                break
        if bdir is None:
            continue
        out[label] = {}
        for jf in bdir.glob("*.json"):
            cfg_key = jf.stem.replace("_", "+")
            if cfg_key == "Full":
                cfg_key = "Full"
            with open(jf) as f:
                out[label][cfg_key] = _unify_metric_keys(json.load(f))
    return out


def _compute_synergy(beta_metrics: dict) -> tuple:
    """Compute ODE×MoCo interaction term for each metric × beta.

    synergy = (VAE+ODE+MoCo) - (VAE+ODE) - (VAE+MoCo) + VAE

    Returns (matrix, metric_labels, beta_labels).
    """
    synergy_metrics = [
        ("ARI",  "ARI",  True),
        ("NMI",  "NMI",  True),
        ("ASW",  "ASW",  True),
        ("DAV",  "DB\u2193", False),
        ("DRE_umap_overall_quality", "DRE", True),
        ("DREX_overall_quality", "DREX", True),
    ]
    beta_labels = list(beta_metrics.keys())
    metric_labels = [lbl for _, lbl, _ in synergy_metrics]
    mat = np.full((len(synergy_metrics), len(beta_labels)), np.nan)

    for bi, blabel in enumerate(beta_labels):
        bm = beta_metrics[blabel]
        for mi, (key, _, higher) in enumerate(synergy_metrics):
            vae      = bm.get("VAE",          {}).get(key, np.nan)
            vae_ode  = bm.get("VAE+ODE",      {}).get(key, np.nan)
            vae_moco = bm.get("VAE+MoCo",     {}).get(key, np.nan)
            vae_om   = bm.get("VAE+ODE+MoCo", {}).get(key, np.nan)
            if any(np.isnan(v) for v in [vae, vae_ode, vae_moco, vae_om]):
                continue
            syn = vae_om - vae_ode - vae_moco + vae
            # For DB (lower-is-better), negate so positive = good synergy
            if not higher:
                syn = -syn
            mat[mi, bi] = syn
    return mat, metric_labels, beta_labels


def _draw_synergy_heatmap(gs, fig, rdir):
    """ODE × MoCo synergy heatmap across metrics and beta values."""
    beta_metrics = _load_beta_metrics(rdir)
    if not beta_metrics:
        # Fallback: empty axes with note
        ax = fig.add_subplot(gs)
        ax.text(0.5, 0.5, "Beta sweep data\nnot available",
                ha="center", va="center", fontsize=FS_AXIS,
                transform=ax.transAxes, color="gray")
        ax.set_axis_off()
        return ax

    mat, metric_labels, beta_labels = _compute_synergy(beta_metrics)

    ax = fig.add_subplot(gs)

    # Use diverging colormap: blue = negative, white = 0, red = positive
    vabs = np.nanmax(np.abs(mat))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vabs, vmax=vabs,
                   interpolation="nearest")

    # Annotations
    for mi in range(mat.shape[0]):
        for bi in range(mat.shape[1]):
            v = mat[mi, bi]
            if np.isnan(v):
                continue
            sign = "+" if v >= 0 else ""
            txt = f"{sign}{v:.3f}"
            text_col = "white" if abs(v) > vabs * 0.65 else "black"
            ax.text(bi, mi, txt, ha="center", va="center",
                    fontsize=FS_SMALL + 0.5, color=text_col, fontweight="bold")

    ax.set_xticks(np.arange(len(beta_labels)))
    ax.set_xticklabels(beta_labels, fontsize=FS_TICK)
    ax.set_yticks(np.arange(len(metric_labels)))
    ax.set_yticklabels(metric_labels, fontsize=FS_TICK)
    ax.set_title("ODE \u00d7 MoCo Synergy\n(positive = super-additive)",
                 fontsize=FS_TITLE, pad=3)

    cax = ax.inset_axes([1.03, 0.1, 0.03, 0.8])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=FS_TICK, length=1.5)
    cb.set_label("Interaction term", fontsize=FS_AXIS - 1, labelpad=2)
    return ax


# ── Panel B: Incremental gain waterfall ───────────────────────────────────

def _draw_incremental_gain(gs, fig, configs, metrics):
    """For ARI, NMI, ASW: plot bar per config with delta from VAE baseline.

    Negative deltas are drawn below the baseline; annotations are placed
    inside the delta bar to avoid overlap with x-axis labels.
    """
    metric_triples = [
        ("ARI", "ARI \u2191"),
        ("NMI", "NMI \u2191"),
        ("ASW", "ASW \u2191"),
    ]
    ax_first = None
    for j, (key, title) in enumerate(metric_triples):
        ax = fig.add_subplot(gs[j])
        if j == 0:
            ax_first = ax
        baseline = metrics["VAE"].get(key, 0)
        all_vals = [metrics[c].get(key, 0) for c in configs]

        for k, cfg in enumerate(configs):
            val   = all_vals[k]
            delta = val - baseline
            bar_c = _CONFIG_COLOR[cfg]
            # Baseline portion
            ax.bar(k, baseline, color=bar_c, alpha=0.35,
                   edgecolor="black", linewidth=0.4)
            # Delta portion
            delta_c = "#2ca02c" if delta >= 0 else "#d62728"
            ax.bar(k, delta, bottom=baseline, color=delta_c, alpha=0.75,
                   edgecolor="black", linewidth=0.4)
            # Annotate delta inside the delta bar (above for positive, below baseline for negative)
            if abs(delta) > 1e-6:
                sign = "+" if delta >= 0 else ""
                txt_y = baseline + delta * 0.5  # midpoint of delta bar
                ax.text(k, txt_y, f"{sign}{delta:.3f}",
                        ha="center", va="center", fontsize=FS_SMALL,
                        color=delta_c, fontweight="bold", zorder=10)

        ax.axhline(baseline, color="gray", ls="--", lw=0.8, alpha=0.7, zorder=1)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4, prune="upper"))
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels([_SHORT[c] for c in configs],
                            fontsize=FS_TICK, rotation=35, ha="right")
        ax.set_title(title, fontsize=FS_TITLE, pad=2)
        if j == 0:
            ax.set_ylabel("Score", fontsize=FS_AXIS)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(alpha=0.22, linestyle="--", linewidth=0.4, axis="y")

        # Set y-limits with padding for annotations
        vmin, vmax = min(all_vals), max(all_vals)
        margin = (vmax - vmin) * 0.25 if vmax > vmin else 0.05
        ax.set_ylim(vmin - margin, vmax + margin)
    return ax_first


# ── Panel C: Comprehensive metric heatmap ─────────────────────────────────

def _draw_metric_heatmap(gs, fig, configs, metrics):
    """Rows = configs, Cols = key metrics, colour = normalised score.

    Uses a focused set of 8 metrics for readability at journal column width.
    """
    metric_groups = [
        # (key, display_label, higher_better)
        ("ARI",                       "ARI",     True),
        ("NMI",                       "NMI",     True),
        ("ASW",                       "ASW",     True),
        ("DREX_trustworthiness",      "Trust.",   True),
        ("DREX_overall_quality",      "DREX",    True),
        ("LSE_overall_quality",       "LSE",     True),
        ("DRE_umap_overall_quality",  "DRE",     True),
        ("LSEX_overall_quality",      "LSEX",    True),
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
            txt = f"{raw:.3f}" if not np.isnan(raw) else "\u2014"
            text_col = "white" if mat_norm[ci, mi] > HEATMAP_DARK_THRESHOLD else "black"
            ax.text(mi, ci, txt, ha="center", va="center",
                    fontsize=FS_SMALL + 0.5, color=text_col)

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels([m[1] for m in metric_groups],
                        fontsize=FS_TICK, rotation=40, ha="right")
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels([_SHORT[c] for c in configs], fontsize=FS_TICK)
    ax.set_title("Comprehensive Performance Heatmap\n(higher = better per column, normalised)",
                 fontsize=FS_TITLE, pad=3)

    cax = ax.inset_axes([1.01, 0.1, 0.02, 0.8])
    cb  = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=FS_TICK, length=1.5)
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

    # A: Synergy heatmap (left) + summary table (right)
    gs_A_row = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[0], wspace=0.30)
    gs_B = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[1], wspace=0.35)
    gs_C = gridspec.GridSpecFromSubplotSpec(
        1, 1, subplot_spec=outer[2])
    gs_D = gridspec.GridSpecFromSubplotSpec(
        1, 1, subplot_spec=outer[3])

    print("  Drawing Panel A (Synergy heatmap + summary table)...")
    ax_A = _draw_synergy_heatmap(gs_A_row[0], fig, rdir)

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

    outpath = outdir / "fig3_ablation_summary.png"
    fig.savefig(outpath, **SAVEFIG_KW)

    # Export individual panel sub-figures
    sub_dir = outdir / "fig3_ablation_summary"
    sub_dir.mkdir(parents=True, exist_ok=True)
    _export_subpanels(fig, sub_dir, [(ax_A, "panelA_synergy"),
                                     (ax_B, "panelB_delta_ari"),
                                     (ax_C, "panelC_heatmap"),
                                     (ax_D, "panelD_permutation")])
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
                   default=str(_benchmarks / "results" / "single_dataset"))
    p.add_argument("--outdir",
                   default=str(_benchmarks / "figures"))
    args = p.parse_args()
    rdir   = Path(args.resultsdir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    build_figure(rdir, outdir)


if __name__ == "__main__":
    main()
