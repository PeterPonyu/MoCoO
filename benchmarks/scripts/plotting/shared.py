"""Shared utilities for MoCoO plotting scripts.

Consolidates duplicated helper functions used across all 7 figure scripts:
  - Font setup
  - Metric key normalization
  - NPZ + JSON data loading
  - Sub-panel export
  - Panel labelling
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.font_manager as fm
import numpy as np


# ---------------------------------------------------------------------------
# Font setup
# ---------------------------------------------------------------------------

_FONT_DIR = Path(__file__).resolve().parent.parent.parent / "fonts"


def setup_fonts() -> None:
    """Register Arial fonts (if available) and set as default sans-serif."""
    for fp in (_FONT_DIR / "Arial.ttf", _FONT_DIR / "Arial Bold.ttf"):
        if fp.exists():
            fm.fontManager.addfont(str(fp))
    if (_FONT_DIR / "Arial.ttf").exists():
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = ["Arial"] + list(
            matplotlib.rcParams.get("font.sans-serif", []))


# ---------------------------------------------------------------------------
# Metric key normalization
# ---------------------------------------------------------------------------

_METRIC_KEY_MAP = {
    "full_ARI": "ARI", "full_NMI": "NMI", "full_ASW": "ASW",
    "full_CH": "CAL", "full_DB": "DAV", "corr": "COR",
    "CH": "CAL", "DB": "DAV",
    "LSE_overall": "LSE_overall_quality",
    "DRE_UMAP_overall": "DRE_umap_overall_quality",
    "DRE_tSNE_overall": "DRE_tsne_overall_quality",
}


def unify_metric_keys(m: dict) -> dict:
    """Normalise JSON metric keys so downstream code uses short names.

    Handles output from various pipeline scripts which produce different
    key formats for the same metrics.
    """
    for src, dst in _METRIC_KEY_MAP.items():
        if src in m and dst not in m:
            m[dst] = m[src]
    return m


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_benchmark_npz(rdir: Path) -> dict:
    """Load benchmark_data.npz and return a dict of parsed arrays.

    Returns a dict with keys: ``configs``, ``latents``, ``labels``, and
    optionally ``train_losses``, ``val_losses``, ``val_scores``,
    ``gradients`` (when present in the npz).
    """
    npz = np.load(rdir / "benchmark_data.npz", allow_pickle=True)
    result: Dict[str, Any] = {
        "configs": [str(c) for c in npz["configs"]],
        "latents": [np.asarray(z, dtype=np.float32) for z in npz["latents"]],
        "labels": [np.asarray(lb) for lb in npz["labels"]],
    }
    for key in ("train_losses", "val_losses", "val_scores", "gradients"):
        if key in npz:
            result[key] = [np.asarray(x, dtype=np.float32) for x in npz[key]]
    return result


def load_config_metrics(
    rdir: Path,
    configs: List[str],
) -> Dict[str, dict]:
    """Load per-config JSON metric files and normalize keys.

    Parameters
    ----------
    rdir : Path
        Directory containing ``<config>.json`` files.
    configs : list of str
        Config names (e.g. ``["VAE", "VAE+ODE", "Full"]``).

    Returns
    -------
    dict
        Mapping from config name to its normalized metric dict.
    """
    metrics: Dict[str, dict] = {}
    for cfg in configs:
        key = cfg.replace("+", "_")
        jf = rdir / f"{key}.json"
        if jf.exists():
            with open(jf) as f:
                metrics[cfg] = unify_metric_keys(json.load(f))
    return metrics


# ---------------------------------------------------------------------------
# Sub-panel export
# ---------------------------------------------------------------------------

def export_subpanels(
    fig: Any,
    sub_dir: Path,
    panels: List[Tuple[Any, str]],
    dpi: int = 300,
) -> None:
    """Save each panel (axes) as a standalone PNG cropped tightly."""
    renderer = fig.canvas.get_renderer()
    for ax, name in panels:
        if ax is None:
            continue
        try:
            bbox = ax.get_tightbbox(renderer)
            if bbox is None:
                continue
            extent = bbox.transformed(fig.dpi_scale_trans.inverted())
            sp = sub_dir / f"{name}.png"
            fig.savefig(sp, dpi=dpi, bbox_inches=extent)
        except Exception as exc:
            print(f"  sub-panel {name}: skipped ({exc})")


# ---------------------------------------------------------------------------
# Panel labelling
# ---------------------------------------------------------------------------

def panel_label(
    fig: Any,
    ax: Any,
    letter: str,
    x_off: float = -0.042,
    y_off: float = 0.006,
    fontsize: float = 9,
) -> None:
    """Place a bold panel label (e.g. '(A)') at the top-left of *ax*."""
    pos = ax.get_position()
    fig.text(
        pos.x0 + x_off, pos.y1 + y_off,
        f"({letter})", fontsize=fontsize, fontweight="bold",
        va="bottom", ha="right", clip_on=False,
    )
