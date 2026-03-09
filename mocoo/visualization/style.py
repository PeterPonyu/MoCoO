"""
Centralized style configuration for MoCoO publication figures.

Provides consistent matplotlib rcParams, color palettes, and config display
mappings used across all visualization functions. All style constants are
derived from the existing benchmark plotting scripts to ensure visual
consistency with the published figures (17 x 21 cm, 300 DPI, Arial font).
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Figure geometry (17 cm x 21 cm page, matching journal column width)
# ---------------------------------------------------------------------------
FIG_WIDTH_CM = 17.0
FIG_HEIGHT_CM = 21.0
FIG_WIDTH_IN = FIG_WIDTH_CM / 2.54   # ~6.693 in
FIG_HEIGHT_IN = FIG_HEIGHT_CM / 2.54  # ~8.268 in
DPI = 300

# Standard savefig keyword arguments for all scripts
SAVEFIG_KW = dict(dpi=DPI, bbox_inches="tight", pad_inches=0.08)

# Threshold for heatmap text colour: above this normalised value, use white text
HEATMAP_DARK_THRESHOLD = 0.45

# Default heatmap colormap (consistent across all heatmap panels)
HEATMAP_CMAP = "YlOrRd"

# Accent colours for highlights and annotations
ACCENT_POSITIVE = "#2ca02c"  # green — positive deltas / improvements
ACCENT_NEGATIVE = "#d62728"  # red   — negative deltas / degradation
ACCENT_BEST     = "crimson"  # best-value highlight edge

# ---------------------------------------------------------------------------
# Font sizes (calibrated for the 17 x 21 cm canvas)
# ---------------------------------------------------------------------------
FS_LABEL = 9     # Panel letters (A), (B), ...
FS_TITLE = 7     # Subplot titles
FS_AXIS = 6      # Axis labels
FS_TICK = 5      # Tick labels
FS_LEGEND = 4.5  # Legend text
FS_SMALL = 4.5   # Annotations / fine print (journal min ~5pt)

# ---------------------------------------------------------------------------
# Model configurations: canonical order and display names
# ---------------------------------------------------------------------------
_CONFIG_ORDER: List[str] = [
    "VAE",
    "VAE+ODE",
    "VAE+MoCo",
    "VAE+MoCo+Proto",
    "VAE+ODE+MoCo",
    "Full",
]

_PALETTE: List[str] = [
    "#4C72B0",  # VAE            — muted blue
    "#DD8452",  # VAE+ODE        — soft orange
    "#55A868",  # VAE+MoCo       — forest green
    "#C44E52",  # VAE+MoCo+Proto — brick red
    "#8172B3",  # VAE+ODE+MoCo   — muted purple
    "#937860",  # Full           — warm brown
]

_CONFIG_COLORS: Dict[str, str] = OrderedDict(
    zip(_CONFIG_ORDER, _PALETTE)
)

# Display name mapping (internal key -> label used in figures)
_DISPLAY_NAMES: Dict[str, str] = {
    "VAE": "VAE",
    "VAE_ODE": "VAE+ODE",
    "VAE+ODE": "VAE+ODE",
    "VAE_MoCo": "VAE+MoCo",
    "VAE+MoCo": "VAE+MoCo",
    "VAE_MoCo_Proto": "VAE+MoCo+Proto",
    "VAE+MoCo+Proto": "VAE+MoCo+Proto",
    "VAE_ODE_MoCo": "VAE+ODE+MoCo",
    "VAE+ODE+MoCo": "VAE+ODE+MoCo",
    "Full": "Full",
}

# Ultra-short abbreviations for tight x-tick labels
_SHORT_NAMES: Dict[str, str] = {
    "VAE": "VAE",
    "VAE+ODE": "V+O",
    "VAE+MoCo": "V+M",
    "VAE+MoCo+Proto": "V+MP",
    "VAE+ODE+MoCo": "V+OM",
    "Full": "Full",
}

# Per-config line styles for training curve differentiation
_LINE_STYLES: Dict[str, object] = {
    "VAE": "-",
    "VAE+ODE": "--",
    "VAE+MoCo": "-.",
    "VAE+MoCo+Proto": ":",
    "VAE+ODE+MoCo": (0, (3, 1, 1, 1)),
    "Full": (0, (5, 1)),       # long-dash (distinct from VAE solid)
}

# Per-config line widths (Full model gets emphasis)
_LINE_WIDTHS: Dict[str, float] = {
    c: (1.8 if c == "Full" else 1.1) for c in _CONFIG_ORDER
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_style() -> None:
    """Apply publication-quality matplotlib rcParams.

    Sets Arial font (if available), figure size, DPI, tick formatting, and
    other parameters to match the MoCoO benchmark figure style. Safe to
    call multiple times; subsequent calls are idempotent.
    """
    params = {
        # Figure
        "figure.figsize": (FIG_WIDTH_IN, FIG_HEIGHT_IN),
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,

        # Font
        "font.family": "sans-serif",
        "font.size": FS_AXIS,

        # Axes
        "axes.titlesize": FS_TITLE,
        "axes.labelsize": FS_AXIS,
        "axes.linewidth": 0.5,
        "axes.grid": True,
        "axes.grid.which": "major",
        "axes.facecolor": "white",

        # Grid
        "grid.alpha": 0.22,
        "grid.linestyle": "--",
        "grid.linewidth": 0.4,

        # Ticks
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,

        # Legend
        "legend.fontsize": FS_LEGEND,
        "legend.frameon": True,
        "legend.framealpha": 0.65,
        "legend.borderpad": 0.3,

        # Lines
        "lines.linewidth": 1.1,
        "lines.markersize": 3,

        # Scatter / patch
        "patch.linewidth": 0.4,
    }

    # Try to set Arial; fall back gracefully
    try:
        import matplotlib.font_manager as fm
        available = {f.name for f in fm.fontManager.ttflist}
        if "Arial" in available:
            params["font.sans-serif"] = ["Arial"] + list(
                matplotlib.rcParams.get("font.sans-serif", [])
            )
    except Exception:
        pass

    matplotlib.rcParams.update(params)


def get_config_colors() -> Dict[str, str]:
    """Return an ordered dict mapping config names to hex color strings.

    Example::

        >>> colors = get_config_colors()
        >>> colors["Full"]
        '#937860'
    """
    return OrderedDict(_CONFIG_COLORS)


def get_config_order() -> List[str]:
    """Return the canonical display order of model configurations.

    Example::

        >>> get_config_order()
        ['VAE', 'VAE+ODE', 'VAE+MoCo', 'VAE+MoCo+Proto', 'VAE+ODE+MoCo', 'Full']
    """
    return list(_CONFIG_ORDER)


def get_display_name(config: str) -> str:
    """Map an internal config key to its display label.

    Handles both ``'VAE_ODE'`` (underscore) and ``'VAE+ODE'`` (plus) forms.
    Returns the input unchanged if no mapping exists.
    """
    return _DISPLAY_NAMES.get(config, config)


def get_short_name(config: str) -> str:
    """Return an ultra-short abbreviation for tight x-tick labels."""
    return _SHORT_NAMES.get(config, config)


def get_line_style(config: str):
    """Return the matplotlib linestyle for the given config."""
    return _LINE_STYLES.get(config, "-")


def get_line_width(config: str) -> float:
    """Return the matplotlib linewidth for the given config."""
    return _LINE_WIDTHS.get(config, 1.1)
