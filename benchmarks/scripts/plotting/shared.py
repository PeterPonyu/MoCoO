"""Shared utilities for MoCoO plotting scripts.

Provides font registration and selection used by figure scripts.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm


# ---------------------------------------------------------------------------
# Font setup
# ---------------------------------------------------------------------------

_ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
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
