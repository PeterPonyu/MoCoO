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
SAVEFIG_KW = dict(dpi=DPI)

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
FS_LABEL = 16    # Panel letters (A), (B), ...
FS_TITLE = 13    # Subplot titles
FS_AXIS = 11     # Axis labels
FS_TICK = 10     # Tick labels
FS_LEGEND = 9    # Legend text
FS_SMALL = 8     # Annotations / heatmap cells (≥6pt for readability)

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
    "#0072B2",  # VAE            — blue (Wong)
    "#E69F00",  # VAE+ODE        — orange (Wong)
    "#009E73",  # VAE+MoCo       — bluish green (Wong)
    "#CC79A7",  # VAE+MoCo+Proto — reddish purple (Wong)
    "#56B4E9",  # VAE+ODE+MoCo   — sky blue (Wong)
    "#D55E00",  # Full           — vermilion (Wong)
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

# ---------------------------------------------------------------------------
# Metric glossary and direction indicators
# ---------------------------------------------------------------------------
METRIC_GLOSSARY: Dict[str, str] = {
    "ARI": "Adj. Rand Index",
    "NMI": "Norm. Mutual Info.",
    "ASW": "Avg. Silhouette Width",
    "CAL": "Calinski\u2013Harabasz",
    "DAV": "Davies\u2013Bouldin",
    "COR": "Pearson Corr.",
    "DRE": "Dim. Red. Eval.",
    "DREX": "Ext. Dim. Red. Eval.",
    "LSE": "Latent Space Eval.",
    "LSEX": "Ext. Latent Space Eval.",
    "iLISI": "Integration LISI",
    "bASW": "Batch ASW",
    "cLISI": "Cell-type LISI",
}

METRIC_DIRECTION: Dict[str, bool] = {
    "ARI": True, "NMI": True, "ASW": True, "CAL": True,
    "DAV": False, "COR": True, "DRE": True, "DREX": True,
    "LSE": True, "LSEX": True, "iLISI": True, "bASW": True,
    "cLISI": True,
}

# ---------------------------------------------------------------------------
# Standardized decimal format constants
# ---------------------------------------------------------------------------
FMT_SCORE = ".3f"        # Metric scores in [0,1]
FMT_SCORE_SHORT = ".2f"  # Heatmap cells (space-constrained)
FMT_LARGE = ".1f"        # Values > 10
FMT_DELTA = "+.3f"       # Signed delta annotations

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
    c: (2.2 if c == "Full" else 1.4) for c in _CONFIG_ORDER
}


# ---------------------------------------------------------------------------
# Absolute-geometry layout helpers
# ---------------------------------------------------------------------------
# All coordinates are normalised figure fractions [0, 1].  The helpers below
# free every figure script from tight_layout / bbox_inches="tight" by
# computing exact axes positions up-front.

def place_axes(fig, rect):
    """Create an axes at an exact position.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    rect : tuple (left, bottom, width, height) in figure-normalised coords.

    Returns
    -------
    matplotlib.axes.Axes
    """
    return fig.add_axes(rect)


def row_of_axes(fig, n, rect, gap=0.04, widths=None):
    """Distribute *n* axes horizontally inside *rect*.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    n : int
    rect : tuple (left, bottom, width, height)
    gap : float  — horizontal gap between adjacent axes (figure fraction).
    widths : list of float, optional
        Relative widths for each axes.  If *None*, all are equal.

    Returns
    -------
    list of matplotlib.axes.Axes
    """
    left, bottom, total_w, height = rect
    if widths is None:
        widths = [1.0] * n
    wsum = sum(widths)
    usable = total_w - gap * (n - 1)
    axes = []
    x = left
    for i, w in enumerate(widths):
        aw = usable * (w / wsum)
        axes.append(fig.add_axes([x, bottom, aw, height]))
        x += aw + gap
    return axes


def col_of_axes(fig, n, rect, gap=0.04, heights=None):
    """Distribute *n* axes vertically inside *rect* (top to bottom).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    n : int
    rect : tuple (left, bottom, width, height)
    gap : float  — vertical gap between adjacent axes (figure fraction).
    heights : list of float, optional
        Relative heights (top-to-bottom order).  If *None*, all equal.

    Returns
    -------
    list of matplotlib.axes.Axes  — ordered top to bottom.
    """
    left, bottom, width, total_h = rect
    if heights is None:
        heights = [1.0] * n
    hsum = sum(heights)
    usable = total_h - gap * (n - 1)
    axes = []
    # Build from top downward
    y = bottom + total_h
    for i, h in enumerate(heights):
        ah = usable * (h / hsum)
        y -= ah
        axes.append(fig.add_axes([left, y, width, ah]))
        y -= gap
    return axes


def grid_of_axes(fig, nrows, ncols, rect, hgap=0.04, wgap=0.04,
                 heights=None, widths=None):
    """Create a *nrows* × *ncols* grid of axes inside *rect*.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    nrows, ncols : int
    rect : tuple (left, bottom, width, height)
    hgap : float — vertical gap between rows (figure fraction).
    wgap : float — horizontal gap between columns (figure fraction).
    heights : list of float, optional
        Relative row heights (top-to-bottom).
    widths : list of float, optional
        Relative column widths (left-to-right).

    Returns
    -------
    list of list of matplotlib.axes.Axes
        ``axes[row][col]``, row 0 is the top row.
    """
    left, bottom, total_w, total_h = rect
    if heights is None:
        heights = [1.0] * nrows
    if widths is None:
        widths = [1.0] * ncols
    hsum = sum(heights)
    wsum = sum(widths)
    usable_h = total_h - hgap * (nrows - 1)
    usable_w = total_w - wgap * (ncols - 1)

    # Pre-compute row bottoms (top-to-bottom order)
    row_bottoms = []
    row_heights = []
    y = bottom + total_h
    for rh in heights:
        ah = usable_h * (rh / hsum)
        y -= ah
        row_bottoms.append(y)
        row_heights.append(ah)
        y -= hgap

    # Pre-compute column lefts
    col_lefts = []
    col_widths = []
    x = left
    for cw in widths:
        aw = usable_w * (cw / wsum)
        col_lefts.append(x)
        col_widths.append(aw)
        x += aw + wgap

    axes = []
    for ri in range(nrows):
        row = []
        for ci in range(ncols):
            ax = fig.add_axes([col_lefts[ci], row_bottoms[ri],
                               col_widths[ci], row_heights[ri]])
            row.append(ax)
        axes.append(row)
    return axes


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

        # Font — Arial normal weight throughout
        "font.family": "sans-serif",
        "font.size": FS_AXIS,
        "font.weight": "normal",
        "font.style": "normal",

        # Axes — normal weight titles and labels
        "axes.titlesize": FS_TITLE,
        "axes.titleweight": "normal",
        "axes.labelsize": FS_AXIS,
        "axes.labelweight": "normal",
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
        "lines.linewidth": 1.4,
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


def save_figure(fig, path, **extra_kw) -> None:
    """Save figure as both PNG (raster) and PDF (vector) for publication.

    Given ``path = 'foo/bar.png'``, saves:
      - ``foo/bar.png``  (300 DPI raster)
      - ``foo/bar.pdf``  (vector for journal submission)
    """
    from pathlib import Path as _Path

    p = _Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    kw = dict(SAVEFIG_KW, **extra_kw)
    fig.savefig(str(p), **kw)
    # Also save vector PDF
    pdf_path = p.with_suffix(".pdf")
    pdf_kw = dict(kw)
    pdf_kw.pop("dpi", None)  # PDF is vector; DPI is irrelevant
    fig.savefig(str(pdf_path), **pdf_kw)


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
    return _LINE_WIDTHS.get(config, 1.4)


def get_tick_name(config: str) -> str:
    """Short name for x-tick labels (V+O, V+M, etc.)."""
    return _SHORT_NAMES.get(config, config)


def get_legend_name(config: str) -> str:
    """Full display name for legend entries (VAE+ODE, etc.)."""
    return _DISPLAY_NAMES.get(config, config)


def metric_label(abbrev: str, include_direction: bool = True) -> str:
    """Return display label like 'ARI (Adj. Rand Index) ↑' for axis labels."""
    arrow = ""
    if include_direction and abbrev in METRIC_DIRECTION:
        arrow = " \u2191" if METRIC_DIRECTION[abbrev] else " \u2193"
    full = METRIC_GLOSSARY.get(abbrev, abbrev)
    return f"{abbrev} ({full}){arrow}"


def metric_title(abbrev: str) -> str:
    """Short title like 'ARI ↑' for subplot titles."""
    arrow = ""
    if abbrev in METRIC_DIRECTION:
        arrow = " \u2191" if METRIC_DIRECTION[abbrev] else " \u2193"
    return f"{abbrev}{arrow}"
