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
    """Register Arial fonts (if bundled) and set best available sans-serif.

    Preference order: Arial > Liberation Sans > Nimbus Sans > DejaVu Sans.
    Liberation Sans is metrically identical to Arial.
    """
    # Try bundled Arial first
    for fp in (_FONT_DIR / "Arial.ttf", _FONT_DIR / "Arial Bold.ttf"):
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
            sp_pdf = sub_dir / f"{name}.pdf"
            fig.savefig(sp_pdf, bbox_inches=extent)
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
    fontsize: float = None,
) -> None:
    """Place a bold panel label (e.g. '(A)') at the top-left of *ax*."""
    if fontsize is None:
        try:
            from mocoo.visualization.style import FS_LABEL
            fontsize = FS_LABEL
        except ImportError:
            fontsize = 9
    pos = ax.get_position()
    fig.text(
        pos.x0 + x_off, pos.y1 + y_off,
        f"({letter})", fontsize=fontsize, fontweight="bold",
        va="bottom", ha="right", clip_on=False,
    )


# ---------------------------------------------------------------------------
# Multiseed data loading
# ---------------------------------------------------------------------------

def load_multiseed_stats(
    csv_path: Path,
    metrics: Optional[List[str]] = None,
) -> Dict[str, Dict[str, "tuple[float, float]"]]:
    """Load multiseed CSV and compute mean +/- std per config per metric.

    Parameters
    ----------
    csv_path : Path
        Path to multiseed_IRALL.csv (cols: config, seed, ARI, NMI, ...).
    metrics : list of str, optional
        Subset of metric columns. Defaults to all numeric columns.

    Returns
    -------
    dict
        ``{config_name: {metric_name: (mean, std)}}``
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    if metrics is None:
        skip = {"config", "seed", "dataset", "actual_epochs"}
        metrics = [c for c in df.columns if c not in skip]
    grouped = df.groupby("config")
    result: Dict[str, Dict[str, tuple]] = {}
    for cfg, grp in grouped:
        result[str(cfg)] = {}
        for m in metrics:
            if m in grp.columns:
                vals = grp[m].dropna()
                if len(vals) > 0:
                    result[str(cfg)][m] = (float(vals.mean()), float(vals.std()))
    return result


# ---------------------------------------------------------------------------
# Figure footnotes (config key + metric abbreviations)
# ---------------------------------------------------------------------------

def add_config_legend_footnote(fig: Any, y_pos: float = 0.01) -> None:
    """Add a footnote mapping short names → full names at the figure bottom."""
    from mocoo.visualization.style import (
        get_config_order, get_tick_name, get_legend_name, FS_SMALL,
    )
    entries = [
        f"{get_tick_name(c)} = {get_legend_name(c)}"
        for c in get_config_order()
        if get_tick_name(c) != get_legend_name(c)
    ]
    if not entries:
        return
    footnote = "Config key: " + " | ".join(entries)
    fig.text(0.50, y_pos, footnote,
             fontsize=max(FS_SMALL - 1, 7), ha="center", va="bottom",
             style="italic", color="#555555")


def add_metric_footnote(
    fig: Any,
    metrics_used: List[str],
    y_pos: float = -0.01,
) -> None:
    """Add a footnote defining metric abbreviations at the figure bottom.

    Example output::

        ARI = Adj. Rand Index ↑; NMI = Norm. Mutual Info. ↑; ...
    """
    from mocoo.visualization.style import (
        METRIC_GLOSSARY, METRIC_DIRECTION, FS_SMALL,
    )
    entries = []
    for m in metrics_used:
        full = METRIC_GLOSSARY.get(m, None)
        if full is None:
            continue
        direction = "\u2191" if METRIC_DIRECTION.get(m, True) else "\u2193"
        entries.append(f"{m} = {full} {direction}")
    if not entries:
        return
    footnote_text = ";  ".join(entries)
    fig.text(0.50, y_pos, footnote_text,
             fontsize=max(FS_SMALL - 1, 7), ha="center", va="top",
             color="#666666", style="italic")
