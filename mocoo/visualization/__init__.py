"""
MoCoO Visualization Module
===========================

Modular, publication-quality visualization pipeline for MoCoO benchmark
results.  Provides a clean API that wraps the plotting logic formerly
spread across seven standalone scripts in ``benchmarks/scripts/plotting/``.

Quick start
-----------
::

    from mocoo.visualization import (
        apply_style,
        get_config_colors,
        get_config_order,
        plot_ablation_radar,
        plot_metric_bars,
        plot_umap_grid,
        plot_training_curves,
        plot_pseudotime_markers,
        plot_beta_sensitivity,
        FigurePipeline,
    )

    # Generate all paper figures from benchmark output
    pipe = FigurePipeline("benchmarks/results/IRALL", "figures/")
    pipe.load_results()
    pipe.generate_all()

Submodules
----------
style
    Centralized rcParams, color palette, and config display mappings.
plots
    Core plotting functions (each returns a matplotlib Figure).
pipeline
    High-level :class:`FigurePipeline` for batch figure generation.
"""
from __future__ import annotations

# Style API
from .style import (
    apply_style,
    get_config_colors,
    get_config_order,
    get_display_name,
    get_short_name,
    get_line_style,
    get_line_width,
    place_axes,
    row_of_axes,
    col_of_axes,
    grid_of_axes,
    DPI,
    FIG_WIDTH_IN,
    FIG_HEIGHT_IN,
    FIG_WIDTH_CM,
    FIG_HEIGHT_CM,
    FS_LABEL,
    FS_TITLE,
    FS_AXIS,
    FS_TICK,
    FS_LEGEND,
    FS_SMALL,
)

# Core plotting functions
from .plots import (
    plot_ablation_radar,
    plot_metric_bars,
    plot_umap_grid,
    plot_training_curves,
    plot_pseudotime_markers,
    plot_beta_sensitivity,
)

# Pipeline
from .pipeline import FigurePipeline, FIGURE_NAMES

__all__ = [
    # Style
    "apply_style",
    "get_config_colors",
    "get_config_order",
    "get_display_name",
    "get_short_name",
    "get_line_style",
    "get_line_width",
    "place_axes",
    "row_of_axes",
    "col_of_axes",
    "grid_of_axes",
    "DPI",
    "FIG_WIDTH_IN",
    "FIG_HEIGHT_IN",
    "FIG_WIDTH_CM",
    "FIG_HEIGHT_CM",
    "FS_LABEL",
    "FS_TITLE",
    "FS_AXIS",
    "FS_TICK",
    "FS_LEGEND",
    "FS_SMALL",
    # Plots
    "plot_ablation_radar",
    "plot_metric_bars",
    "plot_umap_grid",
    "plot_training_curves",
    "plot_pseudotime_markers",
    "plot_beta_sensitivity",
    # Pipeline
    "FigurePipeline",
    "FIGURE_NAMES",
]
