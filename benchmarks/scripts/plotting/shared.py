"""Shared utilities for MoCoO plotting scripts.

Provides font registration, selection, and the reusable boxplot figure
builder used by fig2 (ablation) and other boxplot figures.
"""
from __future__ import annotations

import csv
import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — ensure project root is importable
# ---------------------------------------------------------------------------

_ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from mocoo.visualization.style import (  # noqa: E402
    FS_AXIS,
    FS_LEGEND,
    FS_TICK,
    FS_TITLE,
    HIGHLIGHT_CONFIGS,
    HIGHLIGHT_EDGE_WIDTH,
    PROPOSED_CLUSTERING,
    PROPOSED_DIRECTION,
    PROPOSED_QUALITY,
    PROPOSED_SHORT_LABELS,
    get_legend_name,
    get_tick_name,
    save_figure,
    style_boxplot,
)

# ---------------------------------------------------------------------------
# Font setup
# ---------------------------------------------------------------------------

_FONT_DIRS = [
    _ROOT_DIR / "vcd" / "fonts",
    _ROOT_DIR / "fonts",
]


def setup_fonts() -> None:
    """Register Arial fonts (if bundled) and set best available sans-serif.

    Preference order: Arial > Liberation Sans > Nimbus Sans > DejaVu Sans.
    Liberation Sans is metrically identical to Arial.
    """
    # Try bundled Arial first (prefer vcd/fonts when available).
    for font_dir in _FONT_DIRS:
        for fp in (font_dir / "Arial.ttf", font_dir / "Arial Bold.ttf",
                   font_dir / "Arial Italic.ttf",
                   font_dir / "Arial Bold Italic.ttf"):
            if fp.exists():
                fm.fontManager.addfont(str(fp))
    # Pick the best available sans-serif
    available = {f.name for f in fm.fontManager.ttflist}
    preferred = [n for n in ("Arial", "Liberation Sans", "Nimbus Sans")
                 if n in available]
    if preferred:
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = preferred + list(
            matplotlib.rcParams.get("font.sans-serif", []))


# ---------------------------------------------------------------------------
# Boxplot figure builder — reusable across figure scripts
# ---------------------------------------------------------------------------

DATASET_ORDER = [
    "endo", "setty",                        # Stem cell / early development
    "paul", "IRALL", "hemato",              # Hematopoietic / immune
    "dentate", "spinoids", "astrocyte",     # Neural development
    "lung", "retina", "teeth", "spine",     # Organ-specific development
    "hepatoblastoma", "brainmet",           # Cancer
    "breast", "gastric", "livercancer",     # Cancer (extended)
    "melanoma", "pituitary", "hESCtime",    # Additional systems
]

_SPLITS = ["train", "val", "test", "whole"]
_SPLIT_TITLES = {
    "train": "Train",
    "val": "Validation",
    "test": "Test",
    "whole": "Whole set",
}


# -- private helpers --------------------------------------------------------

def _safe_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _load_dataset_summaries(
    results_dir: Path,
    configs_set: set[str],
    metric_keys: list[str],
) -> tuple[list[str], dict]:
    """Load per-dataset CSV summaries, returning (dataset_list, nested_data)."""
    data: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    datasets: list[str] = []
    for dataset in DATASET_ORDER:
        summary_path = results_dir / dataset / "summary_expanded.csv"
        if not summary_path.exists():
            continue
        rows: dict[str, dict[str, dict[str, float]]] = {}
        with summary_path.open() as handle:
            for row in csv.DictReader(handle):
                config = row["config"]
                split = row["split"]
                if config not in configs_set or split not in _SPLITS:
                    continue
                rows.setdefault(config, {})[split] = {
                    mk: _safe_float(row.get(mk)) for mk in metric_keys
                }
        if rows:
            datasets.append(dataset)
            data[dataset] = rows
    return datasets, data


def _metric_bounds(
    data: dict, metric_key: str, split: str | None = None,
) -> tuple[float, float]:
    """Return (lo, hi) y-axis limits for *metric_key*, with padding."""
    values: list[float] = []
    splits_to_check = [split] if split else _SPLITS
    for dataset_data in data.values():
        for config_data in dataset_data.values():
            for s in splits_to_check:
                if s in config_data:
                    v = config_data[s].get(metric_key, np.nan)
                    if np.isfinite(v):
                        values.append(float(v))
    if not values:
        return 0.0, 1.0
    lo, hi = min(values), max(values)
    span = hi - lo
    pad = (max(0.025, 0.15 * span) if span > 0
           else max(0.05, abs(hi) * 0.15 if hi != 0 else 0.1))
    return lo - pad, hi + pad


def _metric_label(metric_key: str, display_label: str) -> str:
    higher = PROPOSED_DIRECTION.get(metric_key, True)
    arrow = "\u2191" if higher else "\u2193"
    return f"{display_label} {arrow}"


# -- public entry point -----------------------------------------------------

def build_boxplot_figure(
    results_dir: Path,
    outdir: Path,
    *,
    configs: list[str],
    config_colors: dict[str, str],
    figure_size: tuple[float, float],
    output_name: str,
    vcd_label: str,
    fs_tick_offset: int = 2,
    warn_if_no_fm: bool = False,
) -> list[dict]:
    """Build a 2-panel boxplot figure (Embedding Quality + Clustering).

    Each panel contains 4 metric columns and 4 split rows
    (train / val / test / whole).

    Parameters
    ----------
    results_dir : Path
        Root directory containing per-dataset result folders.
    outdir : Path
        Directory where the output PNG will be saved.
    configs : list[str]
        Ordered list of configuration names to include.
    config_colors : dict[str, str]
        Mapping from config name to hex colour.
    figure_size : tuple[float, float]
        Width and height of the figure in inches.
    output_name : str
        Base filename (without extension) for the saved figure.
    vcd_label : str
        Label passed to ``save_figure`` for visual-checks-dashboard.
    fs_tick_offset : int
        Added to the base ``FS_TICK`` for tick-label font sizing.
        Use 2 for 6-config figures, 0 for 12-config figures.
    warn_if_no_fm : bool
        If *True*, print a warning when no ``+FM`` configs are found.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

    # ── panel definitions ─────────────────────────────────────────────
    panel_a = (
        "Embedding Quality",
        [(m, PROPOSED_SHORT_LABELS[m]) for m in PROPOSED_QUALITY],
    )
    panel_b = (
        "Clustering",
        [(m, PROPOSED_SHORT_LABELS[m]) for m in PROPOSED_CLUSTERING],
    )
    all_blocks = [panel_a, panel_b]

    # ── font sizes ────────────────────────────────────────────────────
    fs_title = FS_TITLE + 2
    fs_axis = FS_AXIS + 2
    fs_tick = FS_TICK + fs_tick_offset
    fs_legend = FS_LEGEND + 2

    # ── derived layout tweaks (scale with config count) ───────────────
    n_configs = len(configs)
    scatter_size = 12 if n_configs <= 6 else 10
    xtick_adj = 1 if n_configs <= 6 else 2
    left_margin = 0.065 if n_configs <= 6 else 0.060
    legend_ncol = n_configs if n_configs <= 6 else min(6, n_configs)
    legend_colspace = 0.8 if n_configs <= 6 else 0.6

    # ── load data ─────────────────────────────────────────────────────
    metric_keys = [mk for _, metrics in all_blocks for mk, _ in metrics]
    datasets, data = _load_dataset_summaries(
        results_dir, set(configs), metric_keys,
    )
    if not datasets:
        print(
            f"No per-dataset summary_expanded.csv files found in {results_dir}"
        )
        return []

    if warn_if_no_fm:
        has_fm = any(
            any(c.endswith("+FM") for c in data[ds]) for ds in datasets
        )
        if not has_fm:
            print(
                "No +FM data found in any dataset. "
                "Run run_fm_pipeline.py first to generate FM results."
            )
            print("  Proceeding with available configs only.")

    # Only include configs that actually appear in at least one dataset.
    active_configs = [
        c for c in configs if any(c in data[ds] for ds in datasets)
    ]

    # ── build axes ────────────────────────────────────────────────────
    fig = plt.figure(figsize=figure_size)

    right_edge = 0.992
    top_edge = 0.92
    bottom_edge = 0.075
    panel_gap = 0.045
    col_gap = 0.008
    row_gap = 0.012

    a_cols = len(panel_a[1])
    b_cols = len(panel_b[1])

    total_content_w = right_edge - left_margin
    total_inner_col_gap = (a_cols - 1 + b_cols - 1) * col_gap
    col_width = (
        (total_content_w - panel_gap - total_inner_col_gap) / (a_cols + b_cols)
    )

    total_content_h = top_edge - bottom_edge
    row_height = (total_content_h - 3 * row_gap) / 4

    def _make_block_axes(x_start, y_top, ncols):
        axes = []
        for r in range(4):
            bottom = y_top - (r + 1) * row_height - r * row_gap
            row_axes = []
            x = x_start
            for c in range(ncols):
                row_axes.append(
                    fig.add_axes([x, bottom, col_width, row_height])
                )
                x += col_width + col_gap
            axes.append(row_axes)
        return axes

    a_left = left_margin
    b_left = (
        a_left + a_cols * col_width + (a_cols - 1) * col_gap + panel_gap
    )

    block_list = [
        (panel_a, _make_block_axes(a_left, top_edge, a_cols)),
        (panel_b, _make_block_axes(b_left, top_edge, b_cols)),
    ]

    # ── metric bounds ─────────────────────────────────────────────────
    metric_limits: dict[tuple[str, str], tuple[float, float]] = {}
    for _, metrics in all_blocks:
        for mk, _ in metrics:
            for split in _SPLITS:
                metric_limits[(mk, split)] = _metric_bounds(data, mk, split)

    # ── boxplot renderer (closure over shared state) ──────────────────

    def _render_boxplot(ax, split_name, metric_key, y_limits,
                        show_xticklabels, show_ylabel, show_yticks):
        positions = np.arange(len(active_configs))
        bp_data = []
        colors = []
        for config in active_configs:
            vals = [
                data[ds].get(config, {}).get(split_name, {}).get(
                    metric_key, np.nan,
                )
                for ds in datasets
            ]
            vals = [v for v in vals if np.isfinite(v)]
            bp_data.append(vals)
            colors.append(config_colors[config])

        bplot = ax.boxplot(
            bp_data,
            positions=positions,
            widths=0.52,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="white", linewidth=1.0),
            whiskerprops=dict(linewidth=0.7),
            capprops=dict(linewidth=0.7),
            boxprops=dict(linewidth=0.5),
        )
        for i, (patch, color) in enumerate(zip(bplot["boxes"], colors)):
            patch.set_facecolor(color)
            patch.set_alpha(0.82)
            bp_single = {
                "boxes": [bplot["boxes"][i]],
                "whiskers": bplot["whiskers"][2 * i : 2 * i + 2],
                "caps": bplot["caps"][2 * i : 2 * i + 2],
                "medians": [bplot["medians"][i]],
            }
            style_boxplot(bp_single, active_configs[i], color)

        rng = np.random.default_rng(42)
        for i, vals in enumerate(bp_data):
            if vals:
                jitter = rng.uniform(-0.10, 0.10, len(vals))
                ax.scatter(
                    positions[i] + jitter[: len(vals)],
                    vals,
                    s=scatter_size,
                    color=colors[i],
                    alpha=0.55,
                    zorder=5,
                    edgecolors="white",
                    linewidth=0.3,
                )

        ax.set_xlim(-0.5, len(active_configs) - 0.5)
        ax.set_ylim(*y_limits)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
        ax.set_xticks(positions)
        if show_xticklabels:
            ax.set_xticklabels(
                [get_tick_name(c) for c in active_configs],
                fontsize=fs_tick - xtick_adj,
                rotation=90,
                ha="center",
            )
        else:
            ax.set_xticklabels([])
        if show_ylabel:
            ax.set_ylabel(
                _SPLIT_TITLES[split_name], fontsize=fs_axis,
            )
        if not show_yticks:
            ax.tick_params(axis="y", labelleft=False)
        ax.tick_params(axis="y", labelsize=fs_tick)
        ax.grid(axis="y", alpha=0.22, linestyle="--", linewidth=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ── render panels ─────────────────────────────────────────────────
    panel_labels = ["(a)", "(b)"]

    for block_idx, ((block_title, metrics), axes) in enumerate(block_list):
        for row_idx, split_name in enumerate(_SPLITS):
            is_bottom_row = row_idx == 3
            for col_idx, (mk, mlabel) in enumerate(metrics):
                _render_boxplot(
                    axes[row_idx][col_idx],
                    split_name,
                    mk,
                    metric_limits[(mk, split_name)],
                    show_xticklabels=is_bottom_row,
                    show_ylabel=(col_idx == 0),
                    show_yticks=(col_idx == 0),
                )
                if row_idx == 0:
                    axes[row_idx][col_idx].set_title(
                        _metric_label(mk, mlabel),
                        fontsize=fs_title,
                        pad=5,
                    )

        first_pos = axes[0][0].get_position()
        fig.text(
            first_pos.x0,
            first_pos.y1 + 0.020,
            block_title,
            fontsize=fs_title,
            ha="left",
            va="bottom",
        )

    for idx, ((_, _), axes) in enumerate(block_list):
        pos = axes[0][0].get_position()
        fig.text(
            pos.x0,
            pos.y1 + 0.040,
            panel_labels[idx],
            fontsize=fs_title + 1,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    # ── legend ────────────────────────────────────────────────────────
    legend_patches = [
        Patch(
            facecolor=config_colors[c],
            edgecolor=(
                config_colors[c] if c in HIGHLIGHT_CONFIGS else "white"
            ),
            linewidth=(
                HIGHLIGHT_EDGE_WIDTH if c in HIGHLIGHT_CONFIGS else 0.5
            ),
            alpha=0.82,
            label=get_legend_name(c),
        )
        for c in active_configs
    ]
    actual_ncol = (
        min(legend_ncol, len(active_configs)) if active_configs else 1
    )
    fig.legend(
        handles=legend_patches,
        loc="upper center",
        fontsize=fs_legend,
        frameon=False,
        ncol=actual_ncol,
        bbox_to_anchor=(0.5, 1.01),
        handlelength=0.9,
        handletextpad=0.3,
        labelspacing=0.25,
        columnspacing=legend_colspace,
    )

    # ── save ──────────────────────────────────────────────────────────
    outpath = outdir / f"{output_name}.png"
    issues = save_figure(fig, outpath, vcd_label=vcd_label, vcd_verbose=True)
    n_warn = sum(1 for i in issues if i.get("severity") == "warning")
    n_err = sum(1 for i in issues if i.get("severity") == "error")
    plt.close(fig)
    print(f"Saved: {outpath}")
    print(f"{n_warn} warnings | {n_err} errors")
    return issues
