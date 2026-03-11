#!/usr/bin/env python
"""
MoCoO Benchmark — Composed Figure

Streamlined 3-panel figure: UMAP grid, training curves, and key metrics heatmap.
Integrated visual-conflict detection before export.

Usage:
    python benchmarks/scripts/plotting/plot_composed.py
    python benchmarks/scripts/plotting/plot_composed.py --resultsdir benchmarks/results \
           --outdir benchmarks/figures
"""

from __future__ import annotations

import argparse
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

from benchmarks.scripts.pipeline.visual_conflict_detector import detect_all_conflicts
from benchmarks.scripts.plotting.shared import setup_fonts, load_benchmark_npz, load_config_metrics, add_config_legend_footnote
from mocoo.visualization.style import (
    FIG_WIDTH_IN, FIG_HEIGHT_IN, DPI,
    FS_LABEL, FS_TITLE, FS_AXIS, FS_TICK, FS_LEGEND, FS_SMALL,
    apply_style, get_config_colors, get_short_name, get_tick_name,
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
    return get_tick_name(str(cfg))


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

    metrics_dict = load_config_metrics(rdir, configs)
    metrics = [metrics_dict.get(c) for c in configs]

    return dict(configs=configs, latents=latents, labels=labels,
                val_losses=val_losses, val_scores=val_scores,
                train_losses=train_losses, metrics=metrics)


# ═══════════════════════════════════════════════════════════════════════════════
# Per-panel drawing helpers
# ═══════════════════════════════════════════════════════════════════════════════

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


def _draw_umap_panel(fig, umap_axes, configs, latents, labels_arr,
                     umap_embeddings=None):
    """Draw 2×3 UMAP grid on pre-created *umap_axes*."""
    if umap_embeddings is None:
        return

    n = len(configs)
    for i in range(n):
        ax = umap_axes[i // 3][i % 3]
        emb = umap_embeddings[i]
        unique = np.unique(labels_arr[i])
        n_types = len(unique)
        cmap = plt.colormaps.get_cmap("tab20")
        for j, lbl in enumerate(unique):
            mask = labels_arr[i] == lbl
            ax.scatter(emb[mask, 0], emb[mask, 1], s=1.2, alpha=0.60,
                       c=[cmap(j / n_types)], label=str(lbl), rasterized=True)
        ax.set_title(configs[i], fontsize=FONT_PANEL_TITLE)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    # shared legend below the UMAP block
    handles, lbls = fig.axes[0].get_legend_handles_labels()
    # Use more columns for datasets with many cell types to keep legend compact
    n_cols = min(max(6, len(lbls) // 4), 10)
    fig.legend(handles, lbls, loc="lower center",
               bbox_to_anchor=(0.5, 0.685),
               ncol=n_cols, markerscale=2.5,
               fontsize=FONT_ANNOT, frameon=False,
               handlelength=0.8, columnspacing=0.5, labelspacing=0.2)


def _draw_training_curves(fig, curve_axes, configs, val_losses, val_scores):
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
        panels = [("Val Loss", "loss")]

    for pidx, (title, src) in enumerate(panels):
        ax = curve_axes[pidx]

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
            ax.set_ylabel("Loss", fontsize=FONT_AXIS_LABEL)
        if has_scores and pidx // 4 == 1:
            ax.set_xlabel("Epoch", fontsize=FONT_AXIS_LABEL)
        elif not has_scores:
            ax.set_xlabel("Epoch", fontsize=FONT_AXIS_LABEL)
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
        ax_leg = curve_axes[7]
    else:
        ax_leg = curve_axes[1]
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
    """Curated metrics heatmap with column-normalised colours."""
    CURATED = [
        ("ARI",  "ARI"),  ("NMI",  "NMI"),  ("ASW",  "ASW"),
        ("DRE_umap_overall_quality",  "DRE"),
        ("DREX_overall_quality",      "DREX"),
        ("LSE_overall_quality",       "LSE"),
        ("LSEX_overall_quality",      "LSEX"),
        ("train_time_s",              "Time(s)"),
        ("peak_mem_gb",               "Mem(GB)"),
    ]
    keys, labels = [], []
    for k, lbl in CURATED:
        if all(m is not None and np.isfinite(m.get(k, np.nan)) for m in mets):
            keys.append(k)
            labels.append(lbl)
    if len(keys) < 3:
        ax.set_visible(False)
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
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            v = matrix[r, c]
            fmt = f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}"
            clr = "white" if norm[r, c] > 0.45 else "black"
            ax.text(c, r, fmt, ha="center", va="center",
                    fontsize=FONT_ANNOT, color=clr)
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.55, pad=0.02)
    cb.ax.tick_params(labelsize=FONT_TICK)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — build composed figure
# ═══════════════════════════════════════════════════════════════════════════════

def build_composed(data, outpath: Path, cache_dir: Path | None = None):
    """Build the single composed figure and save with conflict detection.

    Streamlined layout (3 rows) showing the most informative panels:
      Row 0 (A): UMAP grid
      Row 1 (B): Training curves (val loss)
      Row 2 (C): All-metrics heatmap
    """
    configs    = data["configs"]
    mets       = data["metrics"]
    latents    = data["latents"]
    labels     = data["labels"]
    val_losses = data["val_losses"]
    val_scores = data["val_scores"]

    # Pre-compute UMAPs (outside figure creation for progress visibility)
    print("Pre-computing UMAP embeddings...")
    cache_path = None
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "umap_cache.npz"
    umap_embeddings = _compute_umaps(configs, latents, cache_path)

    # ── Figure with absolute-geometry layout ─────────────────────────────
    fig = plt.figure(figsize=(FIG_W, FIG_H * 0.55), dpi=DPI)

    # Row 0 — UMAP grid (A): 2×3 explicit per-subplot geometry
    _u_cw = (0.92 - 0.03 * 2) / 3   # ~0.2867
    _u_rh = (0.28 - 0.04) / 2        # 0.12
    umap_axes = [
        [fig.add_axes([0.04 + c * (_u_cw + 0.03),
                       0.68 + 0.28 - (r + 1) * _u_rh - r * 0.04,
                       _u_cw, _u_rh])
         for c in range(3)]
        for r in range(2)
    ]
    _draw_umap_panel(fig, umap_axes, configs, latents, labels,
                     umap_embeddings=umap_embeddings)

    # Row 1 — Training curves (B)
    has_scores = any(
        np.asarray(vs).ndim == 2 and np.asarray(vs).shape[0] > 0
        for vs in val_scores
    )
    if has_scores:
        # 2×4 grid of training curve panels — explicit per-subplot geometry
        _tc_cw = (0.86 - 0.05 * 3) / 4   # ~0.1775
        _tc_rh = (0.24 - 0.05) / 2        # 0.095
        _cg = [
            [fig.add_axes([0.10 + c * (_tc_cw + 0.05),
                           0.38 + 0.24 - (r + 1) * _tc_rh - r * 0.05,
                           _tc_cw, _tc_rh])
             for c in range(4)]
            for r in range(2)
        ]
        curve_axes = [ax for row in _cg for ax in row]
    else:
        # 2 panels in a row
        _tc2_aw = (0.86 - 0.05) / 2  # 0.405
        curve_axes = [
            fig.add_axes([0.10, 0.38, _tc2_aw, 0.24]),
            fig.add_axes([0.10 + _tc2_aw + 0.05, 0.38, _tc2_aw, 0.24]),
        ]
    _draw_training_curves(fig, curve_axes, configs, val_losses, val_scores)

    # Row 2 — Heatmap (C)
    ax_hm = fig.add_axes([0.10, 0.06, 0.86, 0.26])
    _draw_heatmap(ax_hm, configs, mets)

    # ── Panel labels (A–C) ──────────────────────────────────────────────
    panel_axes = []
    umap_axes = [a for a in fig.axes
                 if a.get_title() and a.get_title() in configs]
    if umap_axes:
        panel_axes.append(("A", umap_axes[0]))
    tc_axes = [a for a in fig.axes
               if a.get_title() in ("Val Loss", "Val ARI", "Val NMI")]
    if tc_axes:
        panel_axes.append(("B", tc_axes[0]))
    panel_axes.append(("C", ax_hm))

    for letter, ax in panel_axes:
        try:
            pos = ax.get_position()
            fig.text(pos.x0 - 0.025, pos.y1 + 0.012,
                     f"({letter})", fontsize=FONT_PANEL_LABEL,
                     fontweight="bold", va="bottom", ha="right")
        except Exception:
            pass

    # ── Conflict detection ──────────────────────────────────────────────
    add_config_legend_footnote(fig, y_pos=0.005)
    print("\n── Conflict Detection on Composed Figure ──")
    issues = detect_all_conflicts(fig, label="composed", verbose=True)

    has_trunc = any(i["type"].endswith("_truncation") and
                    i["severity"] == "warning" for i in issues)
    pad = 0.3 if has_trunc else 0.15

    from mocoo.visualization.style import save_figure
    save_figure(fig, str(outpath), pad_inches=pad)
    plt.close(fig)

    n_warn = sum(1 for i in issues if i["severity"] == "warning")
    n_err  = sum(1 for i in issues if i["severity"] == "error")
    print(f"\nComposed figure saved → {outpath}")
    print(f"  Warnings: {n_warn}  |  Errors: {n_err}")
    if n_warn + n_err == 0:
        print("  CLEAN — no conflicts detected")
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
