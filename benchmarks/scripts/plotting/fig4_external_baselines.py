#!/usr/bin/env python
"""MoCoO Figure 4 — External baselines comparison (4-panel layout).

Four-panel figure (2x2) comparing against deep-learning baselines:
  (a)  Cluster Geometry: ASW + inverted-DAV grouped bars (all methods)
  (b)  Embedding Quality: DRE + DREX grouped bars (MoCoO configs only)
  (c)  Latent-Space Quality: LSE + LSEX grouped bars (MoCoO configs only)
  (d)  Multi-Criteria Radar: all 6 proposed metrics, polar overlay

External baselines limited to deep-learning methods (scVI, scANVI, Harmony)
that operate in comparable latent-space dimensions. DPT and PCA+KMeans
excluded as they lack learned latent representations.
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from benchmarks.scripts.plotting.shared import setup_fonts
from mocoo.visualization.style import (
    FS_AXIS, FS_LEGEND, FS_SMALL, FS_TICK, FS_TITLE,
    FIG_WIDTH_IN, DPI,
    apply_style, save_figure, add_panel_label,
    get_config_colors, get_legend_name, get_short_name,
)
from mocoo.visualization.direct_layout import bind_figure_region

setup_fonts()
apply_style()

# ── External method colors (deep-learning methods only) ────────────────
_EXTERNAL_COLORS = {
    "scVI": "#7570B3",
    "scANVI": "#1B9E77",
    "Harmony": "#E7298A",
}

# MoCoO configs to display (ODE ablation progression + FM)
_MOCOO_CONFIGS = ["VAE", "VAE+ODE", "VAE+ODE+MoCo", "VAE+ODE+MoCo+FM"]

# External methods display order (DL-based only; DPT/PCA+KMeans excluded)
_EXT_ORDER = ["scVI", "scANVI", "Harmony"]

# Metrics needed from internal results
_INTERNAL_METRICS = [
    "ASW", "DAV",
    "DRE_umap_overall_quality", "DREX_overall_quality",
    "LSE_overall_quality", "LSEX_overall_quality",
]


# ── Data loaders ───────────────────────────────────────────────────────

def _load_baselines(results_dir: Path):
    """Load external_baselines.csv -> per-method mean (seeds -> dataset -> grand)."""
    fp = results_dir / "baselines" / "external_baselines.csv"
    if not fp.exists():
        return {}
    # Accumulate per method per dataset: method -> dataset -> metric -> [values]
    acc: dict[str, dict[str, dict[str, list]]] = {}
    with open(fp) as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["method"].strip()
            # Skip non-DL methods
            if method in ("DPT", "PCA+KMeans"):
                continue
            dataset = row["dataset"].strip()
            bucket = acc.setdefault(method, {}).setdefault(
                dataset, {"ASW": [], "DAV": []},
            )
            try:
                bucket["ASW"].append(float(row["ASW"]))
            except (KeyError, ValueError):
                pass
            # DB column -> DAV
            try:
                bucket["DAV"].append(float(row["DB"]))
            except (KeyError, ValueError):
                pass
    # Aggregate: mean across seeds per dataset, then mean across datasets
    result: dict[str, dict[str, float]] = {}
    for method, ds_dict in acc.items():
        result[method] = {}
        for metric in ("ASW", "DAV"):
            ds_means = []
            for _dataset, met_dict in ds_dict.items():
                vals = met_dict[metric]
                if vals:
                    ds_means.append(np.mean(vals))
            result[method][metric] = np.mean(ds_means) if ds_means else np.nan
    return result


def _load_internal(results_dir: Path):
    """Load internal config means from summary_expanded.csv files (split=whole)."""
    scores: dict[str, dict[str, list]] = {}
    for d in sorted(results_dir.iterdir()):
        fp = d / "summary_expanded.csv"
        if not fp.exists():
            continue
        with open(fp) as f:
            reader = csv.DictReader(f)
            for row in reader:
                cfg = row.get("config", "").strip()
                split = row.get("split", "").strip()
                if split != "whole":
                    continue
                if cfg not in scores:
                    scores[cfg] = {m: [] for m in _INTERNAL_METRICS}
                for m in _INTERNAL_METRICS:
                    try:
                        scores[cfg][m].append(float(row[m]))
                    except (KeyError, ValueError):
                        pass
    return {
        c: {k: np.mean(v) if v else np.nan for k, v in vals.items()}
        for c, vals in scores.items()
    }


# ── Helpers ────────────────────────────────────────────────────────────

def _method_color(method, config_colors):
    """Return colour for a method (internal config or external baseline)."""
    if method in config_colors:
        return config_colors[method]
    return _EXTERNAL_COLORS.get(method, "#888888")


def _method_label(method, internal, short=False):
    """Return display label for a method."""
    if method in internal:
        return get_short_name(method) if short else get_legend_name(method)
    return method


def _safe_val(source, method, metric, default=0.0):
    """Safely retrieve a metric value, returning *default* on missing/NaN."""
    v = source.get(method, {}).get(metric, np.nan)
    return v if np.isfinite(v) else default


# ── Main figure builder ───────────────────────────────────────────────

def make_figure(results_dir: Path, out_path: Path):
    ext = _load_baselines(results_dir)
    internal = _load_internal(results_dir)
    if not ext and not internal:
        print("No data found — skipping fig4.")
        return

    config_colors = get_config_colors()

    # Build ordered method lists
    mocoo_show = [c for c in _MOCOO_CONFIGS if c in internal]
    ext_methods = [m for m in _EXT_ORDER if m in ext]
    all_methods = mocoo_show + ext_methods
    n_all = len(all_methods)

    if n_all == 0:
        print("No methods with data — skipping fig4.")
        return

    # ── Figure layout: 2x2 grid ──────────────────────────────────────
    fig = plt.figure(figsize=(FIG_WIDTH_IN * 1.1, FIG_WIDTH_IN * 1.0))
    root = bind_figure_region(fig, (0.08, 0.10, 0.94, 0.92))
    (r_top, r_bot) = root.split_rows([1, 1], gap=0.16)
    (r_a, r_b) = r_top.split_cols([1, 1], gap=0.12)
    (r_c, r_d) = r_bot.split_cols([1, 1.3], gap=0.12)

    # ════════════════════════════════════════════════════════════════════
    # Panel (a): Cluster Geometry — ASW + inverted-DAV, all methods
    # ════════════════════════════════════════════════════════════════════
    ax_a = r_a.add_axes(fig)
    x_a = np.arange(n_all)
    w_a = 0.35

    # Collect ASW values
    asw_vals = []
    for method in all_methods:
        src = internal if method in internal else ext
        asw_vals.append(_safe_val(src, method, "ASW"))

    # Collect DAV values and compute inverted display
    dav_raw = []
    for method in all_methods:
        src = internal if method in internal else ext
        v = src.get(method, {}).get("DAV", np.nan)
        dav_raw.append(v if np.isfinite(v) else np.nan)

    finite_dav = [v for v in dav_raw if np.isfinite(v)]
    max_dav = max(finite_dav) if finite_dav else 1.0
    dav_inverted = [
        max_dav - v if np.isfinite(v) else 0.0 for v in dav_raw
    ]

    ax_a.bar(
        x_a - w_a / 2, asw_vals, w_a,
        label="ASW", color="#009E73",
        zorder=3, edgecolor="white", linewidth=0.3,
    )
    ax_a.bar(
        x_a + w_a / 2, dav_inverted, w_a,
        label="inv-DAV", color="#CC79A7",
        zorder=3, edgecolor="white", linewidth=0.3,
    )

    ax_a.set_xticks(x_a)
    ax_a.set_xticklabels(
        [_method_label(m, internal, short=True) for m in all_methods],
        fontsize=FS_SMALL, rotation=35, ha="right",
    )
    ax_a.set_ylabel("Score", fontsize=FS_AXIS)
    ax_a.set_title("Cluster Geometry", fontsize=FS_TITLE)
    ax_a.legend(fontsize=FS_LEGEND, frameon=False, loc="upper right")
    ax_a.grid(axis="y", alpha=0.22)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    add_panel_label(ax_a, "a", x=-0.12, y=1.08)

    # ════════════════════════════════════════════════════════════════════
    # Panel (b): Embedding Quality — DRE + DREX, MoCoO configs only
    # ════════════════════════════════════════════════════════════════════
    ax_b = r_b.add_axes(fig)
    n_mocoo = len(mocoo_show)
    x_b = np.arange(n_mocoo)
    w_b = 0.35

    dre_vals = [
        _safe_val(internal, c, "DRE_umap_overall_quality") for c in mocoo_show
    ]
    drex_vals = [
        _safe_val(internal, c, "DREX_overall_quality") for c in mocoo_show
    ]

    ax_b.bar(
        x_b - w_b / 2, dre_vals, w_b,
        label="DRE", color="#0072B2",
        zorder=3, edgecolor="white", linewidth=0.3,
    )
    ax_b.bar(
        x_b + w_b / 2, drex_vals, w_b,
        label="DREX", color="#E69F00",
        zorder=3, edgecolor="white", linewidth=0.3,
    )

    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels(
        [get_short_name(c) for c in mocoo_show],
        fontsize=FS_SMALL, rotation=35, ha="right",
    )
    ax_b.set_ylabel("Score", fontsize=FS_AXIS)
    ax_b.set_title("Embedding Quality", fontsize=FS_TITLE)
    ax_b.legend(fontsize=FS_LEGEND, frameon=False, loc="upper right")
    ax_b.grid(axis="y", alpha=0.22)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # Italic gray note about external methods (inside axes, bottom-left)
    ax_b.text(
        0.02, 0.02,
        "External methods lack latent\nspace for these metrics",
        transform=ax_b.transAxes, fontsize=FS_SMALL - 1,
        fontstyle="italic", color="#999999", ha="left", va="bottom",
    )
    add_panel_label(ax_b, "b", x=-0.12, y=1.08)

    # ════════════════════════════════════════════════════════════════════
    # Panel (c): Latent-Space Quality — LSE + LSEX, MoCoO configs only
    # ════════════════════════════════════════════════════════════════════
    ax_c = r_c.add_axes(fig)
    x_c = np.arange(n_mocoo)
    w_c = 0.35

    lse_vals = [
        _safe_val(internal, c, "LSE_overall_quality") for c in mocoo_show
    ]
    lsex_vals = [
        _safe_val(internal, c, "LSEX_overall_quality") for c in mocoo_show
    ]

    ax_c.bar(
        x_c - w_c / 2, lse_vals, w_c,
        label="LSE", color="#009E73",
        zorder=3, edgecolor="white", linewidth=0.3,
    )
    ax_c.bar(
        x_c + w_c / 2, lsex_vals, w_c,
        label="LSEX", color="#CC79A7",
        zorder=3, edgecolor="white", linewidth=0.3,
    )

    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels(
        [get_short_name(c) for c in mocoo_show],
        fontsize=FS_SMALL, rotation=35, ha="right",
    )
    ax_c.set_ylabel("Score", fontsize=FS_AXIS)
    ax_c.set_title("Latent-Space Quality", fontsize=FS_TITLE)
    ax_c.legend(fontsize=FS_LEGEND, frameon=False, loc="upper right")
    ax_c.grid(axis="y", alpha=0.22)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    ax_c.text(
        0.02, 0.02,
        "External methods lack latent\nspace for these metrics",
        transform=ax_c.transAxes, fontsize=FS_SMALL - 1,
        fontstyle="italic", color="#999999", ha="left", va="bottom",
    )
    add_panel_label(ax_c, "c", x=-0.12, y=1.08)

    # ════════════════════════════════════════════════════════════════════
    # Panel (d): Multi-Criteria Radar
    # ════════════════════════════════════════════════════════════════════
    ax_d = r_d.add_axes(fig, projection="polar")

    radar_labels = ["ASW", "1/DAV", "DRE", "LSE", "DREX", "LSEX"]
    N_axes = len(radar_labels)
    angles = np.linspace(0, 2 * np.pi, N_axes, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    # Methods and their visual styles for the radar
    _radar_styles = [
        ("VAE+ODE+MoCo+FM", True, 2.5, 0.25),   # filled, thick, highlight
        ("VAE+ODE+MoCo",    True, 1.5, 0.15),    # filled, thinner
        ("scVI",            False, 1.2, 0.0),     # line only
        ("scANVI",          False, 1.2, 0.0),     # line only
        ("Harmony",         False, 1.2, 0.0),     # line only
    ]

    # Filter to methods that actually exist in the data
    radar_entries = []
    for method, filled, lw, alpha in _radar_styles:
        if method in internal or method in ext:
            radar_entries.append((method, filled, lw, alpha))

    # Collect raw metric values per radar method
    raw_radar: dict[str, dict[str, float]] = {}
    for method, *_ in radar_entries:
        vals: dict[str, float] = {}
        if method in internal:
            vals["ASW"] = internal[method].get("ASW", np.nan)
            vals["DAV"] = internal[method].get("DAV", np.nan)
            vals["DRE"] = internal[method].get("DRE_umap_overall_quality", np.nan)
            vals["LSE"] = internal[method].get("LSE_overall_quality", np.nan)
            vals["DREX"] = internal[method].get("DREX_overall_quality", np.nan)
            vals["LSEX"] = internal[method].get("LSEX_overall_quality", np.nan)
        elif method in ext:
            vals["ASW"] = ext[method].get("ASW", np.nan)
            vals["DAV"] = ext[method].get("DAV", np.nan)
            vals["DRE"] = 0.0
            vals["LSE"] = 0.0
            vals["DREX"] = 0.0
            vals["LSEX"] = 0.0
        else:
            vals = {"ASW": 0.0, "DAV": np.nan, "DRE": 0.0,
                    "LSE": 0.0, "DREX": 0.0, "LSEX": 0.0}
        raw_radar[method] = vals

    # Compute normalisation maxima (only from methods that truly have each metric)
    def _finite_max(values):
        fv = [v for v in values if np.isfinite(v) and v > 0]
        return max(fv) if fv else 1.0

    max_asw = _finite_max(
        [raw_radar[m]["ASW"] for m, *_ in radar_entries]
    )
    max_dav_r = _finite_max(
        [raw_radar[m]["DAV"] for m, *_ in radar_entries]
    )
    # For quality metrics only consider internal methods
    max_dre = _finite_max(
        [raw_radar[m]["DRE"] for m, *_ in radar_entries if m in internal]
    )
    max_lse = _finite_max(
        [raw_radar[m]["LSE"] for m, *_ in radar_entries if m in internal]
    )
    max_drex = _finite_max(
        [raw_radar[m]["DREX"] for m, *_ in radar_entries if m in internal]
    )
    max_lsex = _finite_max(
        [raw_radar[m]["LSEX"] for m, *_ in radar_entries if m in internal]
    )

    # Plot each method on the radar
    for method, filled, lw, fill_alpha in radar_entries:
        v = raw_radar[method]
        norm_asw = v["ASW"] / max_asw if np.isfinite(v["ASW"]) else 0.0
        norm_dav = 1.0 - v["DAV"] / max_dav_r if np.isfinite(v["DAV"]) else 0.0
        norm_dre = v["DRE"] / max_dre if np.isfinite(v["DRE"]) else 0.0
        norm_lse = v["LSE"] / max_lse if np.isfinite(v["LSE"]) else 0.0
        norm_drex = v["DREX"] / max_drex if np.isfinite(v["DREX"]) else 0.0
        norm_lsex = v["LSEX"] / max_lsex if np.isfinite(v["LSEX"]) else 0.0

        values = [norm_asw, norm_dav, norm_dre, norm_lse, norm_drex, norm_lsex]
        values += values[:1]  # close polygon

        color = _method_color(method, config_colors)
        label = _method_label(method, internal)

        if filled:
            ax_d.fill(angles, values, alpha=fill_alpha, color=color)
            ax_d.plot(angles, values, color=color, linewidth=lw, label=label)
        else:
            ax_d.plot(
                angles, values, color=color, linewidth=lw,
                linestyle="--", label=label, marker="o", markersize=3,
            )

    ax_d.set_xticks(angles[:-1])
    ax_d.set_xticklabels(radar_labels, fontsize=FS_TICK)
    ax_d.set_ylim(0, 1.05)
    ax_d.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_d.set_yticklabels(
        ["0.25", "0.50", "0.75", "1.00"], fontsize=FS_SMALL, color="#666666",
    )
    ax_d.set_title("Multi-Criteria Radar", fontsize=FS_TITLE, pad=18)
    ax_d.legend(
        fontsize=FS_LEGEND, frameon=False, loc="lower right",
        bbox_to_anchor=(1.25, -0.02),
    )

    # Note about DAV inversion
    ax_d.text(
        0.5, -0.06, "*DAV inverted (outer = better)",
        transform=ax_d.transAxes, fontsize=FS_SMALL,
        fontstyle="italic", color="gray", ha="center",
    )
    add_panel_label(ax_d, "d", x=-0.12, y=1.08)

    # ── Save ─────────────────────────────────────────────────────────
    save_figure(
        fig, str(out_path),
        vcd_label="fig4_external_baselines", vcd_verbose=True,
    )
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── CLI entry point ────────────────────────────────────────────────────

def main():
    benchmarks_dir = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(
        description="MoCoO: External Baselines (4-panel)",
    )
    parser.add_argument(
        "--resultsdir", "--results-dir", type=Path,
        default=benchmarks_dir / "results",
    )
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.out:
        out = args.out
    elif args.outdir:
        out = args.outdir / "fig4_external_baselines.png"
    else:
        out = benchmarks_dir / "figures" / "fig4_external_baselines.png"
    make_figure(args.resultsdir, out)


if __name__ == "__main__":
    main()
