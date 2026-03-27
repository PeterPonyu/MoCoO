#!/usr/bin/env python
"""MoCoO Figure 2 — Absolute performance landscape with FM improvement arrows.

All 12 configs (6 base + 6 FM variants) shown as paired boxplots.  FM
improvement is made explicit via diagonal connector arrows between each
base/FM pair — green when FM improved, red when FM degraded.  Subplot
titles include the FM win-rate so the universal-refinement claim is
readable at a glance.

Layout: 1 row × 5 subplots (ASW, DAV, CAL, DRE, DREX).

Reads:  benchmarks/results/<dataset>/summary_expanded.csv
Writes: benchmarks/figures/fig2_ablation_hero.{png,pdf}
"""
from __future__ import annotations

import argparse
import csv
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

from benchmarks.scripts.plotting.shared import setup_fonts, DATASET_ORDER
from mocoo.visualization.style import (
    ACCENT_POSITIVE,
    ACCENT_NEGATIVE,
    FS_AXIS,
    FS_LABEL,
    FS_LEGEND,
    FS_SMALL,
    FS_TICK,
    FS_TITLE,
    FIG_WIDTH_IN,
    HIGHLIGHT_CONFIGS,
    PROPOSED_DIRECTION,
    PROPOSED_METRICS,
    PROPOSED_SHORT_LABELS,
    apply_style,
    get_base_config_order,
    get_config_colors,
    get_config_order,
    get_tick_name,
    save_figure,
    style_boxplot,
)

setup_fonts()
apply_style()

# ── Constants ─────────────────────────────────────────────────────────────
_ALL_CONFIGS = get_config_order()   # 12 configs
_BASE_CONFIGS = get_base_config_order()  # 6 base
_FM_PAIRS = [(c, f"{c}+FM") for c in _BASE_CONFIGS]
_CONFIG_COLORS = get_config_colors()

_LOWER_IS_BETTER = {mk for mk, up in PROPOSED_DIRECTION.items() if not up}

_FIGURE_WIDTH = FIG_WIDTH_IN * 2.0
_FIGURE_HEIGHT = FIG_WIDTH_IN * 0.72


def _safe_float(v: str) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _load_data(results_dir: Path):
    """Load whole-split metrics preserving dataset identity.

    Returns
    -------
    raw : dict[dataset][config][metric] = float
    """
    all_keys = list(PROPOSED_METRICS)
    raw: dict = {}
    for ds in DATASET_ORDER:
        csv_path = results_dir / ds / "summary_expanded.csv"
        if not csv_path.exists():
            continue
        rows: dict = {}
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                if row["split"] != "whole":
                    continue
                cfg = row["config"]
                if cfg in set(_ALL_CONFIGS):
                    rows[cfg] = {mk: _safe_float(row.get(mk)) for mk in all_keys}
        if rows:
            raw[ds] = rows
    return raw


def _flatten_whole(raw):
    """Flatten to data[config][metric] = list[float]."""
    data: dict[str, dict[str, list]] = {
        c: {m: [] for m in PROPOSED_METRICS} for c in _ALL_CONFIGS
    }
    for ds, cfg_data in raw.items():
        for cfg, metric_vals in cfg_data.items():
            if cfg not in data:
                continue
            for mk, v in metric_vals.items():
                if np.isfinite(v):
                    data[cfg][mk].append(v)
    return data


def _compute_win_rates(raw):
    """Per-metric FM win-rate (% of base-config × dataset pairs where FM improved)."""
    rates = {}
    for mk in PROPOSED_METRICS:
        n_total = n_improved = 0
        for base, fm in _FM_PAIRS:
            for ds, cfg_data in raw.items():
                if base not in cfg_data or fm not in cfg_data:
                    continue
                v_base = cfg_data[base].get(mk, np.nan)
                v_fm = cfg_data[fm].get(mk, np.nan)
                if not (np.isfinite(v_base) and np.isfinite(v_fm)):
                    continue
                n_total += 1
                improved = (v_fm > v_base) if mk not in _LOWER_IS_BETTER else (v_fm < v_base)
                if improved:
                    n_improved += 1
        rates[mk] = (n_improved / n_total * 100) if n_total > 0 else 0.0
    return rates


def _metric_ylim(data, mk):
    vals = []
    for cfg in _ALL_CONFIGS:
        vals.extend(v for v in data[cfg][mk] if np.isfinite(v))
    if not vals:
        return 0.0, 1.0
    lo, hi = min(vals), max(vals)
    span = hi - lo
    pad = max(0.02, 0.08 * span)
    return lo - pad, hi + pad


def _render_subplot(ax, data, mk, win_rate: float) -> None:
    """Draw paired boxplots + FM improvement arrows for one metric."""
    positions = []
    box_data = []
    box_colors = []
    box_configs = []

    for pi, base in enumerate(_BASE_CONFIGS):
        fm = f"{base}+FM"
        pos_base = pi * 2.8
        pos_fm = pi * 2.8 + 1.0

        base_vals = [v for v in data[base][mk] if np.isfinite(v)]
        fm_vals = [v for v in data[fm][mk] if np.isfinite(v)]

        if base_vals:
            positions.append(pos_base)
            box_data.append(base_vals)
            box_colors.append(_CONFIG_COLORS[base])
            box_configs.append(base)
        if fm_vals:
            positions.append(pos_fm)
            box_data.append(fm_vals)
            box_colors.append(_CONFIG_COLORS[fm])
            box_configs.append(fm)

    if not box_data:
        return

    bp = ax.boxplot(
        box_data, positions=positions, widths=0.8,
        patch_artist=True, showfliers=False,
        medianprops=dict(color="white", linewidth=1.4),
        whiskerprops=dict(linewidth=0.6),
        capprops=dict(linewidth=0.6),
    )

    for patch, color, cfg in zip(bp["boxes"], box_colors, box_configs):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)
        style_boxplot(
            {"boxes": [patch], "whiskers": [], "caps": [], "medians": []},
            cfg, color,
        )

    # Overlaid scatter dots
    rng = np.random.default_rng(42)
    for pos, vals, cfg in zip(positions, box_data, box_configs):
        jitter = rng.uniform(-0.22, 0.22, len(vals))
        ax.scatter(
            pos + jitter, vals, color=_CONFIG_COLORS[cfg],
            s=8, alpha=0.40, edgecolors="white", linewidths=0.2,
            zorder=5,
        )

    # Extract median y values from the boxplot objects
    # bp["medians"] is ordered the same as positions/box_data
    median_ys = [line.get_ydata()[0] for line in bp["medians"]]
    # Build lookup: config -> (position, median_y)
    median_map = {cfg: (pos, med) for cfg, pos, med in zip(box_configs, positions, median_ys)}

    # FM delta annotations: colored marker + delta text above each pair
    ylims = _metric_ylim(data, mk)
    y_annot = ylims[1] - 0.03 * (ylims[1] - ylims[0])  # near top
    for pi, (base, fm) in enumerate(_FM_PAIRS):
        if base not in median_map or fm not in median_map:
            continue
        _pos_b, med_b = median_map[base]
        _pos_f, med_f = median_map[fm]
        delta = med_f - med_b
        # Improvement check (sign-aware)
        if mk in _LOWER_IS_BETTER:
            improved = delta < 0
        else:
            improved = delta > 0
        color = ACCENT_POSITIVE if improved else ACCENT_NEGATIVE
        marker = "\u25B2" if improved else "\u25BC"  # ▲ or ▼
        pair_center = pi * 2.8 + 0.5
        ax.text(
            pair_center, y_annot, marker,
            color=color, fontsize=7, ha="center", va="top",
            zorder=8,
        )

    # Vertical separators between pairs
    for pi in range(1, len(_BASE_CONFIGS)):
        x_sep = pi * 2.8 - 0.9
        ax.axvline(x_sep, color="0.88", linewidth=0.5, zorder=1)

    # Y-axis limits
    ax.set_ylim(*_metric_ylim(data, mk))

    # X ticks: base-config labels at pair centers
    pair_centers = [pi * 2.8 + 0.5 for pi in range(len(_BASE_CONFIGS))]
    ax.set_xticks(pair_centers)
    ax.set_xticklabels(
        [get_tick_name(c) for c in _BASE_CONFIGS],
        fontsize=FS_TICK, rotation=25, ha="right",
    )

    # Metric title with direction arrow and win-rate
    direction = PROPOSED_DIRECTION[mk]
    arrow = "\u2191" if direction else "\u2193"
    short = PROPOSED_SHORT_LABELS[mk]
    # Highlight win-rate in color
    rate_str = f"{win_rate:.0f}%"
    ax.set_title(f"{short} {arrow}   FM: {rate_str}", fontsize=FS_TITLE, pad=4)

    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5, prune="both"))
    ax.grid(axis="y", alpha=0.18, linestyle="--", linewidth=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main(results_dir: Path, outdir: Path) -> int:
    raw = _load_data(results_dir)
    if not raw:
        print(f"No data found in {results_dir}")
        return 1

    data = _flatten_whole(raw)
    win_rates = _compute_win_rates(raw)

    n_ds = len(raw)
    n_configs = max(len(data[c][PROPOSED_METRICS[0]]) for c in _ALL_CONFIGS)
    print(f"Loaded data for {n_ds} datasets, {len(_ALL_CONFIGS)} configs")

    fig = plt.figure(figsize=(_FIGURE_WIDTH, _FIGURE_HEIGHT))

    top_margin = 0.84
    bottom_margin = 0.20
    left = 0.05
    total_w = 0.92
    ncols = len(PROPOSED_METRICS)
    col_gap = 0.035
    row_height = top_margin - bottom_margin
    col_w = (total_w - col_gap * (ncols - 1)) / ncols

    axes = []
    for ci, mk in enumerate(PROPOSED_METRICS):
        x = left + ci * (col_w + col_gap)
        rect = [x, bottom_margin, col_w, row_height]
        ax = fig.add_axes(rect)
        _render_subplot(ax, data, mk, win_rates[mk])
        axes.append(ax)

    # Compact legend: paired colour swatches (base | +FM)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = []
    for base in _BASE_CONFIGS:
        fm = f"{base}+FM"
        handles.append(
            Patch(facecolor=_CONFIG_COLORS[base], alpha=0.82,
                  label=get_tick_name(base), edgecolor="0.3", linewidth=0.3))
        handles.append(
            Patch(facecolor=_CONFIG_COLORS[fm], alpha=0.82,
                  label=f"{get_tick_name(fm)}", edgecolor="0.3", linewidth=0.3))
    # Delta indicator legend entries
    handles.append(Line2D([0], [0], color=ACCENT_POSITIVE, lw=0,
                           marker="^", markersize=5, label="\u25B2 FM improved"))
    handles.append(Line2D([0], [0], color=ACCENT_NEGATIVE, lw=0,
                           marker="v", markersize=5, label="\u25BC FM degraded"))

    fig.legend(
        handles=handles, loc="upper center",
        bbox_to_anchor=(0.52, 0.995), ncol=7,
        fontsize=FS_LEGEND, frameon=False,
        handlelength=1.0, handletextpad=0.25, columnspacing=0.6,
    )

    outpath = outdir / "fig2_ablation_hero.png"
    issues = save_figure(fig, outpath, vcd_label="fig2_ablation_hero", vcd_verbose=True)
    n_err = sum(1 for i in issues if i.get("severity") == "error")
    plt.close(fig)

    print(f"\n\u2713 fig2_ablation_hero saved to {outpath}")
    print(f"  Win-rates: " + ", ".join(
        f"{PROPOSED_SHORT_LABELS[mk]}={win_rates[mk]:.0f}%"
        for mk in PROPOSED_METRICS
    ))
    print(f"  VCD: {len(issues)} issues, {n_err} errors")
    return n_err


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resultsdir",
        default=str(Path(__file__).resolve().parent.parent.parent / "results"),
    )
    parser.add_argument(
        "--outdir",
        default=str(Path(__file__).resolve().parent.parent.parent / "figures"),
    )
    args = parser.parse_args()
    sys.exit(main(Path(args.resultsdir), Path(args.outdir)))
