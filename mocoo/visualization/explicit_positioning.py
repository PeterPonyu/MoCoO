"""
Post-hoc axes positioning helpers for MoCoO publication figures.

These helpers reposition or add axes relative to existing ones — useful for
colorbars, shared legends, and post-layout tweaks.  Inspired by CLOP-DiT's
``explicit_positioning.py``.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def get_axes_rect(ax: plt.Axes) -> Tuple[float, float, float, float]:
    """Return the current ``(left, bottom, width, height)`` of *ax*."""
    pos = ax.get_position()
    return (pos.x0, pos.y0, pos.width, pos.height)


def set_axes_rect(ax: plt.Axes, rect: Tuple[float, float, float, float]) -> None:
    """Set *ax* position to ``(left, bottom, width, height)``."""
    ax.set_position(rect)


def rect_next_to_axes(
    ax: plt.Axes,
    side: str = "right",
    size: float = 0.02,
    pad: float = 0.01,
) -> Tuple[float, float, float, float]:
    """Compute a rect adjacent to *ax* on the given *side*.

    Parameters
    ----------
    side : {"right", "left", "top", "bottom"}
    size : float — width (right/left) or height (top/bottom) of the new rect.
    pad : float — gap between *ax* and the new rect.
    """
    l, b, w, h = get_axes_rect(ax)
    if side == "right":
        return (l + w + pad, b, size, h)
    if side == "left":
        return (l - pad - size, b, size, h)
    if side == "top":
        return (l, b + h + pad, w, size)
    if side == "bottom":
        return (l, b - pad - size, w, size)
    raise ValueError(f"Unknown side {side!r}")


def add_axes_next_to(
    fig: plt.Figure,
    ax: plt.Axes,
    side: str = "right",
    size: float = 0.02,
    pad: float = 0.01,
    **kwargs,
) -> plt.Axes:
    """Add a new axes adjacent to *ax* on the given *side*.

    Commonly used for colorbars (``side='right', size=0.015``).
    """
    rect = rect_next_to_axes(ax, side=side, size=size, pad=pad)
    return fig.add_axes(rect, **kwargs)


def add_shared_legend_axes(
    fig: plt.Figure,
    rect: Tuple[float, float, float, float],
) -> plt.Axes:
    """Add an invisible axes at *rect* for a standalone shared legend."""
    ax = fig.add_axes(rect)
    ax.set_axis_off()
    return ax


def union_axes_rect(
    axes: Sequence[plt.Axes],
) -> Tuple[float, float, float, float]:
    """Compute the bounding rect enclosing all *axes*."""
    rects = [get_axes_rect(ax) for ax in axes]
    ls = [r[0] for r in rects]
    bs = [r[1] for r in rects]
    rs = [r[0] + r[2] for r in rects]
    ts = [r[1] + r[3] for r in rects]
    left, bottom = min(ls), min(bs)
    right, top = max(rs), max(ts)
    return (left, bottom, right - left, top - bottom)
