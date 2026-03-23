"""
Direct-layout geometry engine for MoCoO publication figures.

Provides a ``LayoutRegion`` abstraction that splits a normalised figure area
into sub-regions without using ``GridSpec`` or ``tight_layout``.  Inspired by
CLOP-DiT's ``direct_layout.py`` approach: every axes position is computed
analytically and placed with ``fig.add_axes``.

Usage
-----
>>> root = bind_figure_region(fig, (0.06, 0.06, 0.94, 0.94))
>>> left, right = root.split_cols([2, 1], gap=0.05)
>>> top, bot = left.split_rows([1, 1], gap=0.04)
>>> ax_tl = top.add_axes(fig)
>>> ax_bl = bot.add_axes(fig)
>>> ax_r  = right.add_axes(fig)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class LayoutRegion:
    """An axis-aligned rectangle in normalised figure coordinates [0, 1].

    Attributes
    ----------
    left, bottom : float
        Lower-left corner.
    width, height : float
        Extent.
    """
    left: float
    bottom: float
    width: float
    height: float

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------
    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def top(self) -> float:
        return self.bottom + self.height

    @property
    def rect(self) -> Tuple[float, float, float, float]:
        """``(left, bottom, width, height)`` tuple for ``fig.add_axes``."""
        return (self.left, self.bottom, self.width, self.height)

    # ------------------------------------------------------------------
    # Alternate constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_bounds(cls, left: float, bottom: float,
                    right: float, top: float) -> "LayoutRegion":
        """Create from ``(left, bottom, right, top)`` bounds."""
        return cls(left, bottom, right - left, top - bottom)

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------
    def split_cols(self, widths: Sequence[float],
                   gap: float = 0.03) -> List["LayoutRegion"]:
        """Split horizontally into *len(widths)* regions.

        Parameters
        ----------
        widths : sequence of float
            Relative column widths (arbitrary scale).
        gap : float
            Normalised gap between adjacent columns.
        """
        n = len(widths)
        total = sum(widths)
        usable = self.width - gap * max(n - 1, 0)
        regions: List[LayoutRegion] = []
        x = self.left
        for i, w in enumerate(widths):
            aw = usable * (w / total)
            regions.append(LayoutRegion(x, self.bottom, aw, self.height))
            x += aw + gap
        return regions

    def split_rows(self, heights: Sequence[float],
                   gap: float = 0.03) -> List["LayoutRegion"]:
        """Split vertically into *len(heights)* regions (top-to-bottom order).

        Parameters
        ----------
        heights : sequence of float
            Relative row heights (arbitrary scale, top-to-bottom).
        gap : float
            Normalised gap between adjacent rows.
        """
        n = len(heights)
        total = sum(heights)
        usable = self.height - gap * max(n - 1, 0)
        regions: List[LayoutRegion] = []
        y = self.top  # start from top
        for h in heights:
            ah = usable * (h / total)
            y -= ah
            regions.append(LayoutRegion(self.left, y, self.width, ah))
            y -= gap
        return regions

    def grid(self, nrows: int, ncols: int, *,
             hgap: float = 0.03, wgap: float = 0.03,
             heights: Optional[Sequence[float]] = None,
             widths: Optional[Sequence[float]] = None,
             ) -> List[List["LayoutRegion"]]:
        """Create an *nrows* × *ncols* grid of sub-regions.

        Returns ``regions[row][col]`` with row 0 = top.
        """
        if heights is None:
            heights = [1.0] * nrows
        if widths is None:
            widths = [1.0] * ncols
        rows = self.split_rows(heights, gap=hgap)
        return [row.split_cols(widths, gap=wgap) for row in rows]

    def inset(self, *, left: float = 0.0, bottom: float = 0.0,
              right: float = 0.0, top: float = 0.0) -> "LayoutRegion":
        """Return a new region trimmed inward by the given fractions of *self*.

        Each parameter is a fraction of the corresponding dimension:
        ``left`` and ``right`` are fractions of ``self.width``;
        ``bottom`` and ``top`` are fractions of ``self.height``.
        """
        dx_l = self.width * left
        dx_r = self.width * right
        dy_b = self.height * bottom
        dy_t = self.height * top
        return LayoutRegion(
            self.left + dx_l,
            self.bottom + dy_b,
            self.width - dx_l - dx_r,
            self.height - dy_b - dy_t,
        )

    # ------------------------------------------------------------------
    # Axes creation
    # ------------------------------------------------------------------
    def add_axes(self, fig: plt.Figure, **kwargs) -> plt.Axes:
        """Add a matplotlib axes at this region's exact position."""
        ax = fig.add_axes(self.rect, **kwargs)
        # Track managed axes on figure for downstream VCD / export
        managed = getattr(fig, "_layout_managed", [])
        managed.append(ax)
        fig._layout_managed = managed
        return ax


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------
def bind_figure_region(
    fig: plt.Figure,
    bounds: Tuple[float, float, float, float] = (0.06, 0.06, 0.94, 0.94),
) -> LayoutRegion:
    """Bind a layout root region to *fig*.

    Parameters
    ----------
    fig : Figure
    bounds : (left, bottom, right, top)
        Normalised figure-coordinate bounds of the usable area.

    Returns
    -------
    LayoutRegion
    """
    region = LayoutRegion.from_bounds(*bounds)
    fig._layout_rect = bounds
    return region
