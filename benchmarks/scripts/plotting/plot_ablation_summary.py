#!/usr/bin/env python
"""
MoCoO Figure 3 — Ablation Study & Component Contribution Analysis
=================================================================
Layout (17 × 21 cm):
    Row 0 (A): Two-part component-effect summary.
                         Left: ODE×MoCo synergy heatmap across metrics and β values.
                         Right: Normalized metric profiles across configurations.
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
import numpy as np
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from vcd import detect_all_conflicts
from benchmarks.scripts.plotting.shared import (
    setup_fonts, unify_metric_keys, load_benchmark_npz, load_config_metrics,
    export_subpanels, panel_label,
    add_config_legend_footnote, add_metric_footnote, load_multiseed_stats,
)

# ── Import centralized style ────────────────────────────────────────────────
from mocoo.visualization.style import (
    FIG_WIDTH_IN as FIG_W, FIG_HEIGHT_IN as FIG_H, DPI,
    FS_LABEL, FS_TITLE, FS_AXIS, FS_TICK, FS_LEGEND as FS_LEG, FS_SMALL,
    HEATMAP_DARK_THRESHOLD, FMT_SCORE_SHORT,
    get_config_colors, get_config_order, get_short_name, apply_style,
    get_tick_name, get_legend_name, metric_title,
)

apply_style()

# ── Fonts ──────────────────────────────────────────────────────────────────
setup_fonts()

_CONFIGS = get_config_order()
_CONFIG_COLOR = get_config_colors()
_SHORT = {c: get_tick_name(c) for c in _CONFIGS}


def _load_data(rdir: Path):
    data = load_benchmark_npz(rdir)
    configs, latents, labels = data["configs"], data["latents"], data["labels"]
    metrics = load_config_metrics(rdir, configs)
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
                out[label][cfg_key] = unify_metric_keys(json.load(f))
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


def _draw_synergy_heatmap(ax, fig, rdir, cbar_rect=None):
    """ODE × MoCo synergy heatmap across metrics and beta values."""
    beta_metrics = _load_beta_metrics(rdir)
    if not beta_metrics:
        # Fallback: empty axes with note
        ax.text(0.5, 0.5, "Beta sweep data\nnot available",
                ha="center", va="center", fontsize=FS_AXIS,
                transform=ax.transAxes, color="gray")
        ax.set_axis_off()
        return ax

    mat, metric_labels, beta_labels = _compute_synergy(beta_metrics)

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
                    fontsize=FS_SMALL, color=text_col)

    ax.set_xticks(np.arange(len(beta_labels)))
    ax.set_xticklabels(beta_labels, fontsize=FS_TICK)
    ax.set_yticks(np.arange(len(metric_labels)))
    ax.set_yticklabels(metric_labels, fontsize=FS_TICK)
    ax.set_title("A1  ODE \u00d7 MoCo Synergy",
                 fontsize=FS_AXIS, pad=3)

    if cbar_rect is None:
        pos = ax.get_position()
        cbar_rect = [pos.x1 + 0.010, pos.y0 + pos.height * 0.08,
                     0.012, pos.height * 0.24]
    cax = fig.add_axes(cbar_rect)
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=max(FS_SMALL - 1, 6), length=1.2, pad=0.4)
    cb.ax.yaxis.set_ticks_position("right")
    cb.ax.set_title("Int.", fontsize=max(FS_SMALL - 1, 6), pad=1.5, y=1.06)
    return ax


# ── Panel B: Incremental gain waterfall ───────────────────────────────────

def _draw_incremental_gain(axes, fig, configs, metrics, multiseed_stats=None):
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
        ax = axes[j]
        if j == 0:
            ax_first = ax
        baseline = metrics["VAE"].get(key, 0)
        all_vals = [metrics[c].get(key, 0) for c in configs]
        vmin, vmax = min(all_vals), max(all_vals)
        margin = (vmax - vmin) * 0.35 if vmax > vmin else 0.05
        y_lo = vmin - margin
        y_hi = vmax + margin
        ax.set_ylim(y_lo, y_hi)

        for k, cfg in enumerate(configs):
            val   = all_vals[k]
            delta = val - baseline
            bar_c = _CONFIG_COLOR[cfg]
            # Baseline portion — start from y_lo to keep bars within axes
            ax.bar(k, baseline - y_lo, bottom=y_lo, color=bar_c, alpha=0.35,
                   edgecolor="black", linewidth=0.4)
            # Delta portion
            delta_c = "#2ca02c" if delta >= 0 else "#d62728"
            ax.bar(k, delta, bottom=baseline, color=delta_c, alpha=0.75,
                   edgecolor="black", linewidth=0.4)

            # Error bar from multiseed variance
            if multiseed_stats and cfg in multiseed_stats and key in multiseed_stats[cfg]:
                _, std = multiseed_stats[cfg][key]
                ax.errorbar(k, val, yerr=std, fmt="none",
                            ecolor="black", capsize=2, capthick=0.6, elinewidth=0.6, zorder=10)

        ax.axhline(baseline, color="gray", ls="--", lw=0.8, alpha=0.7, zorder=1)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4, prune="both"))
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels([_SHORT[c] for c in configs],
                            fontsize=FS_SMALL, rotation=90, ha="center")
        ax.set_xlim(-0.5, len(configs) - 0.5)
        ax.set_title(title, fontsize=FS_TITLE, pad=1)
        if j == 0:
            ax.set_ylabel("Score", fontsize=FS_AXIS)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(alpha=0.22, linestyle="--", linewidth=0.4, axis="y")
    return ax_first


# ── Panel C: Comprehensive metric heatmap ─────────────────────────────────

def _draw_metric_heatmap(ax, fig, configs, metrics, cbar_rect=None):
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

    im = ax.imshow(mat_norm, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1,
                   interpolation="nearest")

    # Annotations
    for ci in range(n_rows):
        for mi in range(n_cols):
            raw = metrics[configs[ci]].get(metric_groups[mi][0], np.nan)
            txt = f"{raw:.2f}" if not np.isnan(raw) else "\u2014"
            text_col = "white" if mat_norm[ci, mi] > HEATMAP_DARK_THRESHOLD else "black"
            ax.text(mi, ci, txt, ha="center", va="center",
                    fontsize=FS_SMALL, color=text_col)

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels([m[1] for m in metric_groups],
                        fontsize=FS_SMALL, rotation=90, ha="center")
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels([_SHORT[c] for c in configs], fontsize=FS_TICK)
    ax.set_title("Performance Heatmap\n(column-normalised; darker = better)",
                 fontsize=FS_TITLE, pad=3)

    if cbar_rect is None:
        pos = ax.get_position()
        cbar_rect = [min(pos.x1 + 0.008, 0.972), pos.y0 + pos.height * 0.08,
                     0.012, pos.height * 0.22]
    cax = fig.add_axes(cbar_rect)
    cb  = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=max(FS_SMALL - 1, 6), length=1.2, pad=0.4)
    cb.ax.yaxis.set_ticks_position("right")
    cb.ax.set_title("Norm.", fontsize=max(FS_SMALL - 1, 6), pad=1.5, y=1.08)
    cb.set_ticks([0.0, 0.5, 1.0])
    return ax


# ── Panel D: Permutation importance box plots ─────────────────────────────

def _draw_perm_boxplots(ax, fig, configs, latents, labels):
    """Box plot of permutation importance distribution across 32 dims per config."""
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
                        fontsize=FS_SMALL, rotation=90, ha="center")
    ax.set_xlim(0.5, len(configs) + 0.5)
    ax.set_ylabel("kNN Acc. Drop", fontsize=FS_AXIS)
    ax.set_title("Latent Dimension Importance",
                 fontsize=FS_TITLE, pad=1)
    ax.tick_params(labelsize=FS_TICK)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4, prune="both"))
    ax.grid(alpha=0.22, linestyle="--", linewidth=0.4, axis="y")
    return ax


# ── Main ───────────────────────────────────────────────────────────────────

def build_figure(rdir: Path, outdir: Path, multiseed_stats=None):
    configs, latents, labels, metrics = _load_data(rdir)

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)

    # ── absolute geometry (replaces GridSpec) ────────────────────────────
    L, R, TOP, BOT = 0.12, 0.94, 0.91, 0.11
    W_all = R - L
    H_all = TOP - BOT

    _ratios = np.array([3.4, 2.5, 3.2, 2.5])
    _hspace = 0.85
    _gap_r  = _hspace * _ratios.mean()
    _unit   = H_all / (_ratios.sum() + 3 * _gap_r)
    row_h   = _ratios * _unit
    gap     = _gap_r * _unit

    row_b = np.empty(4)
    _y = TOP
    for _i in range(4):
        _y -= row_h[_i]
        row_b[_i] = _y
        if _i < 3:
            _y -= gap

    # Row A: two panels (synergy heatmap + summary table), wspace~0.22
    _gapA = 0.34 * (W_all / 2)
    _cwA  = (W_all - _gapA) / 2
    ax_A       = fig.add_axes([L,                  row_b[0], _cwA, row_h[0]])
    ax_A_table = fig.add_axes([L + _cwA + _gapA,   row_b[0], _cwA, row_h[0]])

    # Row B: three panels (incremental gain), wspace~0.35
    _gapB = 0.35 * (W_all / 3)
    _cwB  = (W_all - 2 * _gapB) / 3
    axes_B = [
        fig.add_axes([L,                          row_b[1], _cwB, row_h[1]]),
        fig.add_axes([L + _cwB + _gapB,           row_b[1], _cwB, row_h[1]]),
        fig.add_axes([L + 2 * (_cwB + _gapB),     row_b[1], _cwB, row_h[1]]),
    ]

    # Row C: metric heatmap (full width)
    ax_C = fig.add_axes([L, row_b[2], W_all, row_h[2]])

    # Row D: permutation boxplots (full width)
    ax_D = fig.add_axes([L, row_b[3], W_all, row_h[3]])

    pos_A = ax_A.get_position()
    pos_A_table = ax_A_table.get_position()
    gap_A = pos_A_table.x0 - pos_A.x1
    synergy_cbar_rect = [
        pos_A.x1 + gap_A * 0.18,
        pos_A.y0 + pos_A.height * 0.08,
        min(gap_A * 0.11, 0.010),
        pos_A.height * 0.24,
    ]

    print("  Drawing Panel A (Synergy heatmap + summary profile)...")
    _draw_synergy_heatmap(ax_A, fig, rdir, cbar_rect=synergy_cbar_rect)

    _draw_summary_stats(ax_A_table, configs, metrics)

    print("  Drawing Panel B (Incremental gain)...")
    ax_B = _draw_incremental_gain(axes_B, fig, configs, metrics, multiseed_stats=multiseed_stats)

    pos_C = ax_C.get_position()
    metric_cbar_rect = [
        pos_C.x1 + 0.010,
        pos_C.y0 + pos_C.height * 0.08,
        0.010,
        pos_C.height * 0.22,
    ]
    print("  Drawing Panel C (Metric heatmap)...")
    _draw_metric_heatmap(ax_C, fig, configs, metrics, cbar_rect=metric_cbar_rect)

    print("  Drawing Panel D (Permutation box plots)...")
    _draw_perm_boxplots(ax_D, fig, configs, latents, labels)

    panel_label(fig, ax_A, "A", x_off=-0.07)
    panel_label(fig, ax_B, "B")
    panel_label(fig, ax_C, "C")
    panel_label(fig, ax_D, "D", y_off=0.025)

    # Force full layout computation so legend handles are positioned before detection
    fig.canvas.draw()

    print("\n── Conflict Detection ──")
    issues = detect_all_conflicts(fig, label="ablation_summary", verbose=True)

    outpath = outdir / "fig3_ablation_summary.png"
    from mocoo.visualization.style import save_figure
    save_figure(fig, outpath)

    # Export individual panel sub-figures
    sub_dir = outdir / "fig3_ablation_summary"
    sub_dir.mkdir(parents=True, exist_ok=True)
    export_subpanels(fig, sub_dir, [(ax_A, "panelA_synergy"),
                                     (ax_B, "panelB_delta_ari"),
                                     (ax_C, "panelC_heatmap"),
                                     (ax_D, "panelD_permutation")])
    plt.close(fig)

    n_warn = sum(1 for x in issues if x.get("severity") == "warning")
    n_err  = sum(1 for x in issues if x.get("severity") == "error")
    print(f"\nSaved -> {outpath}")
    print(f"{n_warn} warnings | {n_err} errors")
    return issues


def _draw_summary_stats(ax, configs, metrics):
    """Compact normalised metric profile plot replacing the old table."""
    metric_specs = [
        ("ARI", "ARI"),
        ("NMI", "NMI"),
        ("ASW", "ASW"),
        ("DREX_overall_quality", "DREX"),
        ("LSE_overall_quality", "LSE"),
    ]
    x = np.arange(len(metric_specs))
    raw = np.array([
        [metrics[cfg].get(key, np.nan) for key, _ in metric_specs]
        for cfg in configs
    ], dtype=float)
    col_min = np.nanmin(raw, axis=0)
    col_max = np.nanmax(raw, axis=0)
    col_rng = np.where(col_max - col_min < 1e-8, 1.0, col_max - col_min)
    norm = (raw - col_min) / col_rng

    for cfg_idx, cfg in enumerate(configs):
        ax.plot(x, norm[cfg_idx], marker="o", ms=3.2, lw=1.0,
                color=_CONFIG_COLOR[cfg], alpha=0.85, label=_SHORT[cfg])

    ax.set_xlim(-0.25, len(metric_specs) - 0.75)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metric_specs], fontsize=FS_SMALL)
    ax.set_ylabel("")
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.tick_params(axis="y", pad=-5)
    ax.text(0.965, 0.5, "Norm. score", transform=ax.transAxes,
            rotation=90, va="center", ha="right", fontsize=FS_AXIS)
    ax.set_title("A2  Metric Profiles", fontsize=FS_TITLE, pad=3)
    ax.grid(alpha=0.22, linestyle="--", linewidth=0.4, axis="y")
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=max(FS_SMALL - 1, 6), frameon=False, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.18),
              handlelength=1.0, columnspacing=0.6)


def main():
    _benchmarks = Path(__file__).resolve().parent.parent.parent  # benchmarks/
    p = argparse.ArgumentParser()
    p.add_argument("--resultsdir",
                   default=str(_benchmarks / "results" / "single_dataset"))
    p.add_argument("--outdir",
                   default=str(_benchmarks / "figures"))
    p.add_argument("--multiseed-csv", default=None)
    args = p.parse_args()
    rdir   = Path(args.resultsdir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    multiseed_stats = None
    if args.multiseed_csv:
        multiseed_stats = load_multiseed_stats(Path(args.multiseed_csv))
    return build_figure(rdir, outdir, multiseed_stats=multiseed_stats)


if __name__ == "__main__":
    main()
