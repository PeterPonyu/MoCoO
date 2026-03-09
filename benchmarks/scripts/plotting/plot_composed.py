#!/usr/bin/env python
"""
MoCoO Benchmark — Composed Figure

All evaluation panels in a single publication-quality figure with ~17:21
aspect ratio.  Integrated visual-conflict detection: the final PNG passes
through the 7-pass detector (plus composed-specific checks) before export.

Usage:
    python benchmarks/scripts/plotting/plot_composed.py
    python benchmarks/scripts/plotting/plot_composed.py --resultsdir benchmarks/results \
           --outdir benchmarks/figures
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

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from benchmarks.scripts.pipeline.visual_conflict_detector import detect_all_conflicts, summarize_issues
from benchmarks.scripts.plotting.shared import setup_fonts, unify_metric_keys, load_benchmark_npz, load_config_metrics
from mocoo.visualization.style import (
    FIG_WIDTH_IN, FIG_HEIGHT_IN, DPI, SAVEFIG_KW,
    FS_LABEL, FS_TITLE, FS_AXIS, FS_TICK, FS_LEGEND, FS_SMALL,
    apply_style, get_config_colors, get_short_name,
)

setup_fonts()
apply_style()

# ═══════════════════════════════════════════════════════════════════════════════
# Style constants — from centralized style module (17 cm × 21 cm)
# ═══════════════════════════════════════════════════════════════════════════════
FIG_W, FIG_H = FIG_WIDTH_IN, FIG_HEIGHT_IN  # ~6.693 × 8.268 inches
FONT_PANEL_LABEL = FS_LABEL    # (A) (B) …
FONT_PANEL_TITLE = FS_TITLE    # panel title
FONT_AXIS_LABEL  = FS_AXIS     # axis labels
FONT_TICK         = FS_TICK    # tick labels
FONT_LEGEND       = FS_LEGEND  # legend entries
FONT_ANNOT        = FS_SMALL   # heatmap cell annotations

PALETTE = list(get_config_colors().values())

def _s(cfg):
    return get_short_name(str(cfg))


# ═══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_results(rdir: Path):
    data = load_benchmark_npz(rdir)
    configs      = data["configs"]
    latents      = data["latents"]
    labels       = data["labels"]
    val_losses   = data.get("val_losses", [])
    val_scores   = data.get("val_scores", [])
    train_losses = data.get("train_losses", [])
    gradients    = data.get("gradients")

    metrics_dict = load_config_metrics(rdir, configs)
    metrics = [metrics_dict.get(c) for c in configs]

    return dict(configs=configs, latents=latents, labels=labels,
                val_losses=val_losses, val_scores=val_scores,
                train_losses=train_losses, metrics=metrics,
                gradients=gradients)


# ═══════════════════════════════════════════════════════════════════════════════
# Per-panel drawing helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _bar(ax, configs, mets, specs, title, *, rotation=40):
    """Grouped bar chart inside *ax*.
    specs: list of (key, short_label, higher_is_better)
    """
    n = len(configs)
    avail = [(k, l, h) for k, l, h in specs
             if any(np.isfinite(m.get(k, np.nan)) for m in mets if m)]
    if not avail:
        ax.set_title(title, fontsize=FONT_PANEL_TITLE)
        ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                ha="center", va="center", fontsize=FONT_AXIS_LABEL)
        return

    ncols = len(avail)
    width = 0.7 / ncols
    x = np.arange(n)
    short_cfgs = [_s(c) for c in configs]

    for j, (key, label, hib) in enumerate(avail):
        vals = [m.get(key, np.nan) if m else np.nan for m in mets]
        offset = (j - ncols / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.92, label=label,
                      color=PALETTE[j % len(PALETTE)],
                      edgecolor="white", linewidth=0.3)
        # highlight best
        finite_idx = [i for i, v in enumerate(vals) if np.isfinite(v)]
        if finite_idx:
            best = max(finite_idx, key=lambda i: vals[i]) if hib else \
                   min(finite_idx, key=lambda i: vals[i])
            bars[best].set_edgecolor("black")
            bars[best].set_linewidth(1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(short_cfgs, rotation=rotation, ha="right",
                       fontsize=FONT_TICK)
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    ax.set_title(title, fontsize=FONT_PANEL_TITLE)
    ax.set_ylabel("Score", fontsize=FONT_AXIS_LABEL)
    ax.legend(fontsize=FONT_LEGEND, frameon=False,
              loc="upper right",
              handlelength=1, handletextpad=0.3, labelspacing=0.2,
              ncol=len(avail), borderaxespad=0.3)
    # Add vertical headroom so bars don't reach the legend
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.55 * max(ymax - ymin, 0.01))
    ax.grid(axis="y", alpha=0.12)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _compute_umaps(configs, latents, cache_path=None):
    """Pre-compute or load cached UMAP embeddings."""
    try:
        import umap as _umap
    except ImportError:
        return None

    # Try loading cache
    if cache_path and Path(cache_path).exists():
        print(f"  Loading cached UMAP embeddings from {cache_path}")
        cached = np.load(cache_path, allow_pickle=True)
        embeddings = list(cached["embeddings"])
        if len(embeddings) == len(configs):
            return embeddings
        print("  Cache size mismatch, recomputing...")

    embeddings = []
    for i, cfg in enumerate(configs):
        print(f"  UMAP {i+1}/{len(configs)}: {cfg} ({latents[i].shape[0]} cells)...",
              end=" ", flush=True)
        reducer = _umap.UMAP(n_components=2, random_state=42,
                              n_neighbors=30, min_dist=0.3,
                              n_jobs=1, verbose=False)
        emb = reducer.fit_transform(np.asarray(latents[i], dtype=np.float32))
        embeddings.append(emb)
        print("done", flush=True)

    # Save cache
    if cache_path:
        np.savez_compressed(cache_path, embeddings=np.array(embeddings, dtype=object))
        print(f"  Cached UMAP embeddings → {cache_path}")

    return embeddings


def _draw_umap_panel(fig, gs_parent, configs, latents, labels_arr,
                     umap_embeddings=None):
    """Draw 2×3 UMAP grid inside *gs_parent*."""
    if umap_embeddings is None:
        return

    inner = gs_parent.subgridspec(2, 3, wspace=0.04, hspace=0.18)
    n = len(configs)
    for i in range(n):
        ax = fig.add_subplot(inner[i // 3, i % 3])
        emb = umap_embeddings[i]
        unique = np.unique(labels_arr[i])
        n_types = len(unique)
        cmap = plt.colormaps.get_cmap("tab20")
        for j, lbl in enumerate(unique):
            mask = labels_arr[i] == lbl
            ax.scatter(emb[mask, 0], emb[mask, 1], s=1.5, alpha=0.45,
                       c=[cmap(j / n_types)], label=str(lbl), rasterized=True)
        ax.set_title(configs[i], fontsize=FONT_PANEL_TITLE)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    # shared legend below the UMAP block — centred in the whitespace
    # between Row 0 (UMAP) and Row 1
    handles, lbls = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center",
               bbox_to_anchor=(0.5, 0.685),
               ncol=min(6, len(lbls)), markerscale=3,
               fontsize=FONT_LEGEND, frameon=False)


def _draw_radar(ax, configs, mets):
    """Radar/spider chart on *ax*."""
    cats = [
        ("ARI",                      "ARI"),
        ("NMI",                      "NMI"),
        ("ASW",                      "ASW"),
        ("DRE_umap_overall_quality", "DRE"),
        ("LSE_overall_quality",      "LSE"),
        ("DREX_overall_quality",     "DREX"),
        ("LSEX_overall_quality",     "LSEX"),
    ]
    avail = [(k,l) for k,l in cats
             if any(np.isfinite(m.get(k, np.nan)) for m in mets if m)]
    if len(avail) < 3:
        ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                ha="center", va="center")
        return
    nc = len(avail)
    angles = np.linspace(0, 2*np.pi, nc, endpoint=False).tolist() + [0]

    # convert ax to polar — remove the non-polar placeholder
    pos = ax.get_position()
    parent_fig = ax.figure
    ax.remove()
    pax = parent_fig.add_axes(pos, polar=True)

    for i, (cfg, m) in enumerate(zip(configs, mets)):
        if m is None:
            continue
        vals = [m.get(k, 0) for k, _ in avail]
        vals = [v if np.isfinite(v) else 0 for v in vals]
        vals += vals[:1]
        c = PALETTE[i % len(PALETTE)]
        pax.plot(angles, vals, "o-", lw=1.2, color=c, label=_s(cfg), ms=3)
        pax.fill(angles, vals, alpha=0.08, color=c)

    pax.set_xticks(angles[:-1])
    pax.set_xticklabels([l for _, l in avail], fontsize=FONT_TICK)
    pax.set_yticks([])  # hide radial gridlines entirely to prevent tick overlap
    pax.set_rlabel_position(0)
    pax.set_title("Radar Summary", fontsize=FONT_PANEL_TITLE, pad=12)
    pax.legend(fontsize=FONT_LEGEND - 0.5, frameon=False,
               loc="upper left", bbox_to_anchor=(1.55, 0.95),
               handlelength=1, labelspacing=0.2, ncol=1)
    return pax  # caller may need ref


def _draw_training_curves(fig, gs_parent, configs, val_losses, val_scores):
    """Val loss + validation metrics over training.

    val_scores columns (from MoCoO agent.py):
        0=ARI, 1=NMI, 2=ASW, 3=CAL, 4=DAV, 5=COR

    Gracefully handles empty val_scores by showing only the loss panel
    with a note about missing data in remaining cells.
    """
    # Check if val_scores has actual data
    has_scores = any(
        np.asarray(vs).ndim == 2 and np.asarray(vs).shape[0] > 0
        for vs in val_scores
    )

    if has_scores:
        # Full layout: 2x4 subgrid
        inner = gs_parent.subgridspec(2, 4, wspace=0.40, hspace=0.65)
        panels = [
            ("Val Loss",  "loss"),
            ("Val ARI",   0),
            ("Val NMI",   1),
            ("Val ASW",   2),
            ("Val CAL",   3),
            ("Val DAV",   4),
            ("Val COR",   5),
        ]
    else:
        # Compact layout: just loss curve + legend
        inner = gs_parent.subgridspec(1, 2, wspace=0.35)
        panels = [("Val Loss", "loss")]

    for pidx, (title, src) in enumerate(panels):
        if has_scores:
            row, col = divmod(pidx, 4)
            ax = fig.add_subplot(inner[row, col])
        else:
            ax = fig.add_subplot(inner[0, 0])

        for i, (cfg, vl, vs) in enumerate(zip(configs, val_losses, val_scores)):
            c = PALETTE[i % len(PALETTE)]
            if src == "loss":
                ax.plot(range(len(vl)), vl, color=c, lw=1, label=_s(cfg))
            else:
                vs_a = np.array(vs) if len(vs) > 0 else np.empty((0, 6))
                if vs_a.shape[0] > 0 and vs_a.shape[1] > src:
                    ax.plot(range(vs_a.shape[0]), vs_a[:, src], color=c,
                            lw=1, label=_s(cfg))
        ax.set_title(title, fontsize=FONT_PANEL_TITLE)
        if src == "loss":
            ax.set_ylabel("Loss", fontsize=FONT_AXIS_LABEL - 1)
        if has_scores and pidx // 4 == 1:
            ax.set_xlabel("Epoch", fontsize=FONT_AXIS_LABEL - 1)
        elif not has_scores:
            ax.set_xlabel("Epoch", fontsize=FONT_AXIS_LABEL - 1)
        ax.tick_params(labelsize=FONT_TICK)
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=4, integer=True,
                                                     prune='both'))
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5, prune='both'))
        ax.set_xlim(left=0)
        ax.grid(alpha=0.12)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    # Place legend
    if has_scores:
        ax_leg = fig.add_subplot(inner[1, 3])
    else:
        ax_leg = fig.add_subplot(inner[0, 1])
    ax_leg.set_axis_off()
    handles, lbls = [], []
    for ax_check in fig.axes:
        if ax_check.get_title() == "Val Loss":
            handles, lbls = ax_check.get_legend_handles_labels()
            break
    if handles:
        ax_leg.legend(handles, lbls, loc="center",
                      fontsize=FONT_LEGEND, frameon=False,
                      handlelength=1.5, labelspacing=0.3, ncol=2)
        ax_leg._is_legend_cell = True


def _draw_heatmap(ax, configs, mets):
    """Curated metrics heatmap with column-normalised colours.

    Shows a focused set of ~18 representative metrics spanning all evaluation
    categories (clustering, DRE, LSE, DREX, LSEX, diagnostics, resource) for
    readability at publication scale.
    """
    # Curated metric list: (key, display_label)
    # Reduced to ~9 most important metrics for readability at publication scale.
    CURATED = [
        ("ARI",  "ARI"),  ("NMI",  "NMI"),  ("ASW",  "ASW"),
        ("DRE_umap_overall_quality",  "DRE"),
        ("DREX_overall_quality",      "DREX"),
        ("LSE_overall_quality",       "LSE"),
        ("LSEX_overall_quality",      "LSEX"),
        ("train_time_s",              "Time(s)"),
        ("peak_mem_gb",               "Mem(GB)"),
    ]
    # Filter to keys present in all configs
    keys, labels = [], []
    for k, lbl in CURATED:
        if all(m is not None and np.isfinite(m.get(k, np.nan)) for m in mets):
            keys.append(k)
            labels.append(lbl)
    if len(keys) < 3:
        ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                ha="center", va="center")
        return

    matrix = np.array([[m.get(k, np.nan) for k in keys] for m in mets if m])
    cmin = np.nanmin(matrix, axis=0)
    cmax = np.nanmax(matrix, axis=0)
    crng = cmax - cmin; crng[crng == 0] = 1
    norm = (matrix - cmin) / crng

    im = ax.imshow(norm, aspect="auto", cmap="YlOrRd")
    short_c = [_s(c) for c in configs if mets[configs.index(c)] is not None]
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, rotation=45, ha="right",
                       fontsize=FONT_TICK)
    ax.set_yticks(range(len(short_c)))
    ax.set_yticklabels(short_c, fontsize=FONT_TICK)
    # Annotate cell values
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            v = matrix[r, c]
            fmt = f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}"
            clr = "white" if norm[r, c] > 0.45 else "black"
            ax.text(c, r, fmt, ha="center", va="center",
                    fontsize=FONT_ANNOT + 2.5, color=clr)
    ax.set_title("Key Metrics Heatmap (col-normalized)",
                 fontsize=FONT_PANEL_TITLE)
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.55, pad=0.02)
    cb.ax.tick_params(labelsize=FONT_TICK)


def _project_velocity_to_umap(z, dz, umap_emb, k=30):
    """Project latent-space velocity *dz* onto 2-D UMAP using scVelo-style
    neighbour-based cosine-similarity weighting.

    For each cell *i*, find its *k* nearest neighbours in latent space,
    compute cosine similarity between (z_j − z_i) and dz_i, then weight
    the UMAP displacements (E_j − E_i) by these cosine similarities.
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=min(k + 1, len(z)), metric="euclidean")
    nn.fit(z)
    _, indices = nn.kneighbors(z)

    n = z.shape[0]
    umap_vel = np.zeros((n, 2))

    for i in range(n):
        nbrs = indices[i, 1:]  # exclude self
        disp_z = z[nbrs] - z[i]  # (k, d)
        disp_u = umap_emb[nbrs] - umap_emb[i]  # (k, 2)

        norm_dz = np.linalg.norm(disp_z, axis=1) + 1e-8
        norm_v = np.linalg.norm(dz[i]) + 1e-8
        cos_sim = (disp_z @ dz[i]) / (norm_dz * norm_v)
        cos_sim = np.clip(cos_sim, 0, 1)

        if cos_sim.sum() > 0:
            w = cos_sim / cos_sim.sum()
            umap_vel[i] = (w[:, None] * disp_u).sum(axis=0)

    return umap_vel


def _velocity_on_grid(emb, vel, grid_n=50, smooth=0.5, cutoff_perc=5):
    """Compute velocity field on a regular grid — scVelo / scTour style.

    Parameters
    ----------
    emb : ndarray (n, 2)
        2-D embedding (e.g. UMAP).
    vel : ndarray (n, 2)
        Per-cell velocity projected onto the embedding.
    grid_n : int
        Grid resolution per axis.
    smooth : float
        Gaussian bandwidth = mean(grid_spacing) * smooth.
    cutoff_perc : float
        Percentile below which grid velocities are masked.

    Returns
    -------
    X_grid, Y_grid : 1-D arrays of length *grid_n*
    U_grid, V_grid : 2-D arrays of shape (grid_n, grid_n)  (may contain NaN)
    """
    from scipy.stats import norm as _norm
    from sklearn.neighbors import NearestNeighbors

    n_obs = emb.shape[0]
    # 1 % padding on each side (scVelo convention)
    pad = 0.01
    x_min, x_max = emb[:, 0].min(), emb[:, 0].max()
    y_min, y_max = emb[:, 1].min(), emb[:, 1].max()
    rx, ry = x_max - x_min, y_max - y_min
    x_min -= pad * rx;  x_max += pad * rx
    y_min -= pad * ry;  y_max += pad * ry

    X_grid = np.linspace(x_min, x_max, grid_n)
    Y_grid = np.linspace(y_min, y_max, grid_n)
    xx, yy = np.meshgrid(X_grid, Y_grid)
    grid_pts = np.column_stack([xx.ravel(), yy.ravel()])  # (grid_n², 2)

    # k-neighbours for Gaussian smoothing (scVelo: n_obs / 50)
    k = max(1, int(n_obs / 50))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(emb)
    dists, inds = nn.kneighbors(grid_pts)

    # Gaussian kernel bandwidth
    spacing = np.mean([(x_max - x_min) / grid_n, (y_max - y_min) / grid_n])
    scale = spacing * smooth

    weights = _norm.pdf(x=dists, scale=scale)          # (grid_n², k)
    vel_x = vel[inds, 0]                               # (grid_n², k)
    vel_y = vel[inds, 1]                               # (grid_n², k)

    w_sum = np.maximum(weights.sum(axis=1, keepdims=True), 1.0)
    Ug = (weights * vel_x).sum(axis=1) / w_sum.ravel()
    Vg = (weights * vel_y).sum(axis=1) / w_sum.ravel()

    U_grid = Ug.reshape(grid_n, grid_n)
    V_grid = Vg.reshape(grid_n, grid_n)

    # Mass / length cutoff — mask sparse regions
    mass = weights.sum(axis=1).reshape(grid_n, grid_n)
    min_mass = 1e-5
    min_mass = min(min_mass, np.percentile(mass, 99) * 0.01)

    speed = np.sqrt(U_grid ** 2 + V_grid ** 2)
    length_thr = np.percentile(speed[speed > 0], cutoff_perc) if (speed > 0).any() else 0
    mask = (mass < min_mass) | (speed < length_thr)
    U_grid[mask] = np.nan
    V_grid[mask] = np.nan

    return X_grid, Y_grid, U_grid, V_grid


def _draw_vector_field(ax, configs, latents, labels_arr, gradients,
                       umap_embeddings, mets=None):
    """Draw ODE vector field as a streamplot on UMAP (scVelo / scTour style).

    Uses Gaussian-kernel interpolation onto a regular grid, then matplotlib
    ``ax.streamplot``.  No neighbourhood graph edges — only scatter + streams.
    """
    ode_cfgs = [(i, c) for i, c in enumerate(configs)
                if gradients is not None and i < len(gradients)
                and gradients[i] is not None]
    if not ode_cfgs or umap_embeddings is None:
        ax.text(0.5, 0.5, "N/A\n(no gradients saved;\nre-run benchmark)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=FONT_AXIS_LABEL, color="grey")
        ax.set_title("ODE Vector Field", fontsize=FONT_PANEL_TITLE)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
        return

    # Policy: prefer Full; otherwise highest NMI among ODE configs
    preferred = [p for p in ode_cfgs if str(p[1]) == "Full"]
    if preferred:
        idx, cfg_name = preferred[0]
    elif mets is not None:
        def _nmi(pair):
            i, _ = pair
            try:
                return float(mets[i].get("NMI", -np.inf))
            except Exception:
                return -np.inf
        idx, cfg_name = max(ode_cfgs, key=_nmi)
    else:
        idx, cfg_name = ode_cfgs[-1]
    emb = umap_embeddings[idx]
    z = np.asarray(latents[idx], dtype=np.float32)
    dz = np.asarray(gradients[idx], dtype=np.float32)
    lbl = labels_arr[idx]

    # --- per-cell velocity projected to UMAP ---
    vel_umap = _project_velocity_to_umap(z, dz, emb, k=30)

    # --- background scatter (cell type colours, drawn first) ---
    unique = np.unique(lbl)
    n_types = len(unique)
    cmap = plt.colormaps.get_cmap("tab20")
    for j, lab in enumerate(unique):
        mask = lbl == lab
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   s=5, alpha=0.40,
                   c=[cmap(j / n_types)], rasterized=True, zorder=0)

    # --- grid-based velocity & streamplot (scVelo / scTour style) ---
    X_grid, Y_grid, U_grid, V_grid = _velocity_on_grid(
        emb, vel_umap, grid_n=50, smooth=0.5, cutoff_perc=5)

    # Speed-proportional linewidth (scVelo: base * 2 * speed / max_speed)
    speed = np.sqrt(np.nan_to_num(U_grid) ** 2 + np.nan_to_num(V_grid) ** 2)
    max_speed = speed.max() if speed.max() > 0 else 1.0
    lw = 1.2 + 2.0 * speed / max_speed          # range ~ [1.2, 3.2]

    ax.streamplot(X_grid, Y_grid, U_grid, V_grid,
                  density=2, linewidth=lw,
                  color="black", arrowsize=1.2,
                  arrowstyle="-|>", maxlength=4,
                  integration_direction="both", zorder=3)

    # Add 3 % margin so scatter dots are not clipped at edge
    rx = emb[:, 0].max() - emb[:, 0].min()
    ry = emb[:, 1].max() - emb[:, 1].min()
    ax.set_xlim(emb[:, 0].min() - 0.03 * rx, emb[:, 0].max() + 0.03 * rx)
    ax.set_ylim(emb[:, 1].min() - 0.03 * ry, emb[:, 1].max() + 0.03 * ry)

    ax.set_title(f"ODE Vector Field ({cfg_name}; stream)",
                 fontsize=FONT_PANEL_TITLE)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


# ═══════════════════════════════════════════════════════════════════════════════
# Panel label painter
# ═══════════════════════════════════════════════════════════════════════════════

def _label(fig, ax, letter, *, x_off=-0.06, y_off=1.06):
    """Place '(A)' etc. above-left of *ax*."""
    fig.text(
        ax.get_position().x0 + x_off,
        ax.get_position().y1 + (y_off - 1.0) * ax.get_position().height,
        f"({letter})", fontsize=FONT_PANEL_LABEL, fontweight="bold",
        va="bottom", ha="left",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — build composed figure
# ═══════════════════════════════════════════════════════════════════════════════

def build_composed(data, outpath: Path, cache_dir: Path | None = None):
    """Build the single composed figure and save with conflict detection.

    Compact layout (5 rows) showing only non-redundant panels:
      Row 0 (A): UMAP grid
      Row 1 (B, C): Radar summary | Training curves (val loss)
      Row 2 (D, E): Latent diagnostics | ODE Vector Field
      Row 3 (F): All-metrics heatmap

    Panels from Figs 2-4 (clustering bars, neighbourhood quality, resource
    efficiency, incremental deltas) are intentionally excluded to avoid
    redundancy with the standalone figures.
    """
    configs    = data["configs"]
    mets       = data["metrics"]
    latents    = data["latents"]
    labels     = data["labels"]
    val_losses = data["val_losses"]
    val_scores = data["val_scores"]
    gradients  = data.get("gradients")

    n = len(configs)

    # Pre-compute UMAPs (outside figure creation for progress visibility)
    print("Pre-computing UMAP embeddings...")
    cache_path = None
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "umap_cache.npz"
    umap_embeddings = _compute_umaps(configs, latents, cache_path)

    # Metric specs
    diag_specs = [
        ("diag_mean_norm",          "μ-Norm", True),
        ("diag_std_mean",           "σ-Mean", True),
        ("diag_near_zero_dims",     "0-Dim↓", False),
        ("diag_pairwise_dist_mean", "P-Dst",  True),
    ]

    # ── Figure and GridSpec (4 rows) ─────────────────────────────────────
    fig = plt.figure(figsize=(FIG_W, FIG_H * 0.62), dpi=DPI)
    outer = gridspec.GridSpec(
        4, 1, figure=fig,
        height_ratios=[4.2, 2.8, 3.2, 2.8],
        hspace=0.42,
    )

    # Row 0 — UMAP grid (A)
    gs_umap = outer[0].subgridspec(1, 1)
    _draw_umap_panel(fig, gs_umap[0], configs, latents, labels,
                     umap_embeddings=umap_embeddings)

    # Row 1 — Radar (B) | Training curves (C)
    gs_r1 = outer[1].subgridspec(1, 2, wspace=0.35,
                                  width_ratios=[1.0, 1.4])
    ax_radar = fig.add_subplot(gs_r1[0, 0])
    _draw_radar(ax_radar, configs, mets)
    _draw_training_curves(fig, gs_r1[0, 1], configs, val_losses, val_scores)

    # Row 2 — Diagnostics (D) | Vector Field (E)
    gs_r2 = outer[2].subgridspec(1, 2, wspace=0.40,
                                  width_ratios=[1.0, 1.5])
    ax_diag = fig.add_subplot(gs_r2[0, 0])
    _bar(ax_diag, configs, mets, diag_specs, "Latent Diag.")
    ax_vf = fig.add_subplot(gs_r2[0, 1])
    _draw_vector_field(ax_vf, configs, latents, labels, gradients,
                       umap_embeddings, mets)

    # Row 3 — Heatmap (F)
    ax_hm = fig.add_subplot(outer[3])
    _draw_heatmap(ax_hm, configs, mets)

    # ── Panel labels (A–F) ──────────────────────────────────────────────
    panel_axes = []
    # (A): first UMAP axis
    umap_axes = [a for a in fig.axes
                 if a.get_title() and a.get_title() in configs]
    if umap_axes:
        panel_axes.append(("A", umap_axes[0]))
    # (B) radar — find polar axes
    polar_axes = [a for a in fig.axes if hasattr(a, "set_theta_offset")]
    if polar_axes:
        panel_axes.append(("B", polar_axes[0]))
    # (C) training curves first axis
    tc_axes = [a for a in fig.axes
               if a.get_title() in ("Val Loss", "Val ARI", "Val NMI")]
    if tc_axes:
        panel_axes.append(("C", tc_axes[0]))
    panel_axes += [
        ("D", ax_diag), ("E", ax_vf), ("F", ax_hm),
    ]

    for letter, ax in panel_axes:
        try:
            pos = ax.get_position()
            fig.text(pos.x0 - 0.025, pos.y1 + 0.012,
                     f"({letter})", fontsize=FONT_PANEL_LABEL,
                     fontweight="bold", va="bottom", ha="right")
        except Exception:
            pass

    # ── Conflict detection ──────────────────────────────────────────────
    print("\n── Conflict Detection on Composed Figure ──")
    issues = detect_all_conflicts(fig, label="composed", verbose=True)

    has_trunc = any(i["type"].endswith("_truncation") and
                    i["severity"] == "warning" for i in issues)
    pad = 0.3 if has_trunc else 0.15

    save_kw = dict(SAVEFIG_KW)
    save_kw["pad_inches"] = pad
    fig.savefig(str(outpath), **save_kw)
    plt.close(fig)

    n_warn = sum(1 for i in issues if i["severity"] == "warning")
    n_err  = sum(1 for i in issues if i["severity"] == "error")
    print(f"\nComposed figure saved → {outpath}")
    print(f"  Warnings: {n_warn}  |  Errors: {n_err}")
    if n_warn + n_err == 0:
        print("  ✓ CLEAN — no conflicts detected")
    return issues


# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resultsdir", default=None)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    _benchmarks = Path(__file__).resolve().parent.parent.parent  # benchmarks/
    rdir = Path(args.resultsdir) if args.resultsdir else (_benchmarks / "results" / "single_dataset")
    odir = Path(args.outdir) if args.outdir else (_benchmarks / "figures")
    odir.mkdir(parents=True, exist_ok=True)

    if not (rdir / "benchmark_data.npz").exists():
        print(f"No benchmark_data.npz in {rdir}")
        sys.exit(1)

    data = load_results(rdir)
    print(f"Loaded {len(data['configs'])} configs: {data['configs']}")

    build_composed(data, odir / "fig5_composed_benchmark.png", cache_dir=rdir)


if __name__ == "__main__":
    main()
