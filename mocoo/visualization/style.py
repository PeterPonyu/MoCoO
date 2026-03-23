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
    "VAE+FM",
    "VAE+ODE",
    "VAE+ODE+FM",
    "VAE+MoCo",
    "VAE+MoCo+FM",
    "VAE+MoCo+Proto",
    "VAE+MoCo+Proto+FM",
    "VAE+ODE+MoCo",
    "VAE+ODE+MoCo+FM",
    "Full",
    "Full+FM",
]

_PALETTE: List[str] = [
    "#0072B2",  # VAE                — blue (Wong)
    "#59A3CD",  # VAE+FM             — light blue
    "#E69F00",  # VAE+ODE            — orange (Wong)
    "#EEC059",  # VAE+ODE+FM         — light orange
    "#009E73",  # VAE+MoCo           — bluish green (Wong)
    "#59BFA4",  # VAE+MoCo+FM        — light green
    "#CC79A7",  # VAE+MoCo+Proto     — reddish purple (Wong)
    "#DDA7C5",  # VAE+MoCo+Proto+FM  — light purple
    "#56B4E9",  # VAE+ODE+MoCo       — sky blue (Wong)
    "#91CEF0",  # VAE+ODE+MoCo+FM    — light sky blue
    "#D55E00",  # Full               — vermilion (Wong)
    "#E39659",  # Full+FM            — light vermilion
]

_CONFIG_COLORS: Dict[str, str] = OrderedDict(
    zip(_CONFIG_ORDER, _PALETTE)
)

# Display name mapping (internal key -> label used in figures)
_DISPLAY_NAMES: Dict[str, str] = {
    "VAE": "VAE",
    "VAE_FM": "VAE+FM",
    "VAE+FM": "VAE+FM",
    "VAE_ODE": "VAE+ODE",
    "VAE+ODE": "VAE+ODE",
    "VAE_ODE_FM": "VAE+ODE+FM",
    "VAE+ODE+FM": "VAE+ODE+FM",
    "VAE_MoCo": "VAE+MoCo",
    "VAE+MoCo": "VAE+MoCo",
    "VAE_MoCo_FM": "VAE+MoCo+FM",
    "VAE+MoCo+FM": "VAE+MoCo+FM",
    "VAE_MoCo_Proto": "VAE+MoCo+Proto",
    "VAE+MoCo+Proto": "VAE+MoCo+Proto",
    "VAE_MoCo_Proto_FM": "VAE+MoCo+Proto+FM",
    "VAE+MoCo+Proto+FM": "VAE+MoCo+Proto+FM",
    "VAE_ODE_MoCo": "VAE+ODE+MoCo",
    "VAE+ODE+MoCo": "VAE+ODE+MoCo",
    "VAE_ODE_MoCo_FM": "VAE+ODE+MoCo+FM",
    "VAE+ODE+MoCo+FM": "VAE+ODE+MoCo+FM",
    "Full": "MoCoO",
    "Full_FM": "MoCoO+FM",
    "Full+FM": "MoCoO+FM",
}

# Ultra-short abbreviations for tight x-tick labels
_SHORT_NAMES: Dict[str, str] = {
    "VAE": "VAE",
    "VAE+FM": "V.f",
    "VAE+ODE": "V+O",
    "VAE+ODE+FM": "VO.f",
    "VAE+MoCo": "V+M",
    "VAE+MoCo+FM": "VM.f",
    "VAE+MoCo+Proto": "V+MP",
    "VAE+MoCo+Proto+FM": "VMP.f",
    "VAE+ODE+MoCo": "V+OM",
    "VAE+ODE+MoCo+FM": "VOM.f",
    "Full": "Full",
    "Full+FM": "F.f",
}

# ---------------------------------------------------------------------------
# Metric glossary and direction indicators
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Proposed metric set — canonical metrics for all comparisons
# ---------------------------------------------------------------------------
# Clustering core (4) + overall quality scores (4) = 8 summary metrics.
# These are the metrics where MoCoO (Full / FM) shows clear advantages and
# form the evaluation framework proposed in the paper.

PROPOSED_CLUSTERING = ["NMI", "ARI", "ASW", "DAV"]
PROPOSED_QUALITY = [
    "DRE_umap_overall_quality",
    "LSE_overall_quality",
    "DREX_overall_quality",
    "LSEX_overall_quality",
]
PROPOSED_METRICS = PROPOSED_CLUSTERING + PROPOSED_QUALITY

# Direction for proposed metrics (True = higher is better)
PROPOSED_DIRECTION: Dict[str, bool] = {
    "NMI": True, "ARI": True, "ASW": True, "DAV": False,
    "DRE_umap_overall_quality": True, "LSE_overall_quality": True,
    "DREX_overall_quality": True, "LSEX_overall_quality": True,
}

# Short display labels for proposed metrics (used in bar charts / axes)
PROPOSED_SHORT_LABELS: Dict[str, str] = {
    "NMI": "NMI",
    "ARI": "ARI",
    "ASW": "ASW",
    "DAV": "DAV",
    "DRE_umap_overall_quality": "DRE",
    "LSE_overall_quality": "LSE",
    "DREX_overall_quality": "DREX",
    "LSEX_overall_quality": "LSEX",
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
    "VAE+FM": (0, (1, 1)),
    "VAE+ODE": "--",
    "VAE+ODE+FM": (0, (1, 1)),
    "VAE+MoCo": "-.",
    "VAE+MoCo+FM": (0, (1, 1)),
    "VAE+MoCo+Proto": ":",
    "VAE+MoCo+Proto+FM": (0, (1, 1)),
    "VAE+ODE+MoCo": (0, (3, 1, 1, 1)),
    "VAE+ODE+MoCo+FM": (0, (1, 1)),
    "Full": (0, (5, 1)),       # long-dash (distinct from VAE solid)
    "Full+FM": (0, (1, 1)),     # dotted (FM variants)
}

# Per-config line widths (Full/FM variants get slight emphasis)
_LINE_WIDTHS: Dict[str, float] = {
    c: (2.2 if c in ("Full", "Full+FM") else 1.4) for c in _CONFIG_ORDER
}

# ---------------------------------------------------------------------------
# Visual emphasis for MoCoO (Full / Full+FM) in boxplot & bar figures
# ---------------------------------------------------------------------------
HIGHLIGHT_CONFIGS = {"Full", "Full+FM"}
HIGHLIGHT_EDGE_WIDTH = 2.0   # boxplot edge width for highlighted configs
DEFAULT_EDGE_WIDTH = 0.8     # all other configs


def style_boxplot(bp, config: str, color: str) -> None:
    """Apply visual emphasis to a boxplot dict *bp* based on *config*.

    For configs in HIGHLIGHT_CONFIGS, the box edge becomes thicker and uses
    the config's own colour at full opacity.  Other configs get a thin edge.
    """
    is_highlight = config in HIGHLIGHT_CONFIGS
    ew = HIGHLIGHT_EDGE_WIDTH if is_highlight else DEFAULT_EDGE_WIDTH
    edge_color = color if is_highlight else "#444444"
    for element in ("boxes", "whiskers", "caps"):
        for line in bp.get(element, []):
            line.set_linewidth(ew)
            if is_highlight:
                line.set_color(edge_color)
    for line in bp.get("medians", []):
        line.set_linewidth(ew + 0.4 if is_highlight else 1.0)
        if is_highlight:
            line.set_color("white")
            line.set_zorder(4)


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
        "savefig.bbox": None,          # absolute geometry — never use tight
        "savefig.pad_inches": 0.0,     # no extra padding
        "savefig.facecolor": "white",
        "savefig.edgecolor": "none",
        "savefig.transparent": False,

        # Font — Arial / Liberation Sans, normal weight throughout
        "font.family": "sans-serif",
        "font.size": FS_AXIS,
        "font.weight": "normal",
        "font.style": "normal",

        # Axes — normal weight titles and labels
        "axes.titlesize": FS_TITLE,
        "axes.titleweight": "normal",
        "axes.titlepad": 4.0,
        "axes.labelsize": FS_AXIS,
        "axes.labelweight": "normal",
        "axes.labelpad": 3.0,
        "axes.linewidth": 0.5,
        "axes.grid": True,
        "axes.grid.which": "major",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "figure.edgecolor": "none",

        # Layout — disabled (we use fig.add_axes exclusively)
        "figure.autolayout": False,
        "figure.constrained_layout.use": False,

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
        "xtick.direction": "out",
        "ytick.direction": "out",

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

        # Math — use same sans-serif for math text
        "mathtext.default": "regular",
    }

    # Set font preference: Arial > Liberation Sans > DejaVu Sans
    try:
        import matplotlib.font_manager as fm
        available = {f.name for f in fm.fontManager.ttflist}
        # Liberation Sans is metrically identical to Arial
        preferred = [n for n in ("Arial", "Liberation Sans", "Nimbus Sans")
                     if n in available]
        if preferred:
            params["font.sans-serif"] = preferred + list(
                matplotlib.rcParams.get("font.sans-serif", [])
            )
    except Exception:
        pass

    matplotlib.rcParams.update(params)


def save_figure(fig, path, vcd_label: str | None = None,
                vcd_verbose: bool = False, **extra_kw):
    """Save figure as both PNG (raster) and PDF (vector) for publication.

    Given ``path = 'foo/bar.png'``, saves:
      - ``foo/bar.png``  (300 DPI raster)
      - ``foo/bar.pdf``  (vector for journal submission)

    When ``vcd_label`` is provided, VCD conflict detection is executed as part
    of the save flow and the resulting issue list is returned.
    """
    from pathlib import Path as _Path

    p = _Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    kw = dict(SAVEFIG_KW, **extra_kw)
    issues = []
    if vcd_label is not None:
        from vcd import detect_all_conflicts

        print("\n── Conflict Detection ──")
        issues = detect_all_conflicts(fig, label=vcd_label, verbose=vcd_verbose)
    fig.savefig(str(p), **kw)
    # Also save vector PDF
    pdf_path = p.with_suffix(".pdf")
    pdf_kw = dict(kw)
    pdf_kw.pop("dpi", None)  # PDF is vector; DPI is irrelevant
    fig.savefig(str(pdf_path), **pdf_kw)
    return issues


def add_panel_label(ax, label: str, x: float = -0.08, y: float = 1.06,
                    fontsize: float | None = None) -> None:
    """Place a bold panel label (a), (b), ... outside *ax*.

    Uses a white stroke path effect so the letter is readable against
    any background.
    """
    import matplotlib.patheffects as pe

    fs = fontsize or FS_LABEL
    ax.text(
        x, y, f"({label})",
        transform=ax.transAxes,
        fontsize=fs, fontweight="bold", va="bottom", ha="left",
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
    )


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


def get_base_config_order() -> List[str]:
    """Return only the 6 base model configurations (no +FM variants)."""
    return [c for c in _CONFIG_ORDER if "+FM" not in c]


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
