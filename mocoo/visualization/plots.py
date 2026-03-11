"""
Core plotting functions for MoCoO benchmark visualization.

Each function accepts structured data (pandas DataFrames, dicts, or numpy
arrays), returns a matplotlib ``Figure``, and optionally saves to *outpath*.
All functions apply the centralized style from :mod:`mocoo.visualization.style`
automatically so callers need not configure matplotlib before use.

Functions
---------
plot_ablation_radar
    Normalized multi-metric dot-strip chart across model configurations.
plot_metric_bars
    Grouped bar chart comparing selected metrics across configs.
plot_umap_grid
    2-row x 3-col UMAP scatter grid colored by cell-type labels.
plot_training_curves
    Training / validation loss convergence curves for all configs.
plot_pseudotime_markers
    Marker-gene correlation with pseudotime (line plot with ranking).
plot_beta_sensitivity
    Beta (KL weight) sensitivity sweep: metric vs. beta value per config.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]

from . import style as _style

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_style() -> None:
    """Apply the MoCoO publication style (idempotent)."""
    _style.apply_style()


def _save_and_return(fig: plt.Figure, outpath: Optional[Union[str, Path]]) -> plt.Figure:
    """Save *fig* to *outpath* when provided, then return the figure."""
    if outpath is not None:
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(outpath), dpi=_style.DPI)
    return fig


def _resolve_configs(configs: Optional[Sequence[str]]) -> List[str]:
    """Return the requested config list, falling back to canonical order."""
    if configs is not None:
        return list(configs)
    return _style.get_config_order()


def _df_to_dict(data, key_col: str = "config") -> dict:
    """Convert a DataFrame with a *key_col* column to a dict of dicts."""
    if pd is not None and isinstance(data, pd.DataFrame):
        result = {}
        for _, row in data.iterrows():
            cfg = row[key_col]
            result[cfg] = {
                k: v for k, v in row.items() if k != key_col
            }
        return result
    return data


# ═══════════════════════════════════════════════════════════════════════════
# 1. Ablation radar / dot-strip chart
# ═══════════════════════════════════════════════════════════════════════════

def plot_ablation_radar(
    metrics_df,
    configs: Optional[Sequence[str]] = None,
    metrics: Optional[Sequence[str]] = None,
    outpath: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Normalized multi-metric comparison across model configurations.

    Renders a grouped bar chart where each metric is min-max normalized
    across configs so that relative strengths are immediately visible --
    the modern, non-polar replacement for radar charts used in the MoCoO
    ablation study (Figure 5A).

    Parameters
    ----------
    metrics_df : pandas.DataFrame or dict
        If a DataFrame, must contain a ``"config"`` column and one column
        per metric.  If a dict, maps ``config_name -> {metric: value}``.
    configs : sequence of str, optional
        Subset / order of configs to plot.  Defaults to the canonical 6.
    metrics : sequence of str, optional
        Which metrics to include.  Defaults to
        ``["ARI", "NMI", "ASW", "DREX", "LSE", "DRE"]``.
    outpath : str or Path, optional
        If given, save the figure to this path (PNG/PDF/SVG).

    Returns
    -------
    matplotlib.figure.Figure
    """
    _ensure_style()
    colors = _style.get_config_colors()
    configs = _resolve_configs(configs)

    data = _df_to_dict(metrics_df)

    if not data:
        fig = plt.figure(figsize=(_style.FIG_WIDTH_IN, 3))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.text(0.5, 0.5, "No ablation data available", transform=ax.transAxes,
                ha="center", va="center", fontsize=_style.FS_LABEL)
        ax.set_axis_off()
        return _save_and_return(fig, outpath)

    if metrics is None:
        metrics = ["ARI", "NMI", "ASW", "DREX", "LSE", "DRE"]
    metrics = list(metrics)

    # Collect raw values  (configs x metrics)
    raw = np.zeros((len(configs), len(metrics)), dtype=np.float64)
    for i, cfg in enumerate(configs):
        cfg_data = data.get(cfg, {})
        for j, m in enumerate(metrics):
            raw[i, j] = float(cfg_data.get(m, 0.0))

    # Min-max normalize per metric
    vmin = raw.min(axis=0)
    vmax = raw.max(axis=0)
    vrange = np.where(vmax - vmin < 1e-8, 1.0, vmax - vmin)
    normed = (raw - vmin) / vrange

    n_metrics = len(metrics)
    n_configs = len(configs)
    bar_w = 0.12
    group_gap = 1.0

    fig = plt.figure(figsize=(_style.FIG_WIDTH_IN, _style.FIG_HEIGHT_IN * 0.3))
    ax = fig.add_axes([0.10, 0.20, 0.86, 0.70])
    ax.set_facecolor("#f9f9f9")

    for j, cfg in enumerate(configs):
        xs = np.arange(n_metrics) * group_gap + (j - n_configs / 2) * bar_w
        ax.bar(
            xs, normed[j], width=bar_w * 0.85,
            color=colors.get(cfg, "#888888"), alpha=0.80,
            edgecolor="black", linewidth=0.3,
            label=_style.get_short_name(cfg),
        )

    ax.set_xticks(np.arange(n_metrics) * group_gap)
    ax.set_xticklabels(metrics, fontsize=_style.FS_TICK)
    ax.set_ylabel("Norm. score", fontsize=_style.FS_AXIS)
    ax.set_title(
        "Multi-metric Comparison (normalised per metric)",
        fontsize=_style.FS_TITLE, pad=3,
    )
    ax.set_ylim(0, 1.25)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4, prune="upper"))
    ax.tick_params(labelsize=_style.FS_TICK)
    ax.grid(alpha=0.22, linestyle="--", linewidth=0.4, axis="y")
    ax.legend(
        fontsize=_style.FS_LEGEND, frameon=True, ncol=3,
        loc="upper right", handlelength=0.8, labelspacing=0.15,
        columnspacing=0.6, borderpad=0.3,
    )

    return _save_and_return(fig, outpath)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Metric bar charts (grouped, val + test overlay)
# ═══════════════════════════════════════════════════════════════════════════

def plot_metric_bars(
    metrics_df,
    metric_names: Optional[Sequence[str]] = None,
    configs: Optional[Sequence[str]] = None,
    outpath: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Grouped bar charts for selected clustering metrics across configs.

    For each metric in *metric_names*, draws a set of bars (one per config).
    When the data contains both a base key (e.g. ``"ARI"``) and a
    ``"test_<key>"`` variant, overlays test-set bars with hatching.

    Parameters
    ----------
    metrics_df : pandas.DataFrame or dict
        Config -> metric mapping (same format as :func:`plot_ablation_radar`).
    metric_names : sequence of str, optional
        Metrics to plot.  Defaults to ``["ARI", "NMI", "ASW"]``.
    configs : sequence of str, optional
        Subset / order of configs.
    outpath : str or Path, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _ensure_style()
    colors = _style.get_config_colors()
    configs = _resolve_configs(configs)
    data = _df_to_dict(metrics_df)

    if metric_names is None:
        metric_names = ["ARI", "NMI", "ASW"]
    metric_names = list(metric_names)

    n_panels = len(metric_names)
    fig = plt.figure(figsize=(_style.FIG_WIDTH_IN, _style.FIG_HEIGHT_IN * 0.25))
    # Explicit per-axis geometry: each bar panel in a horizontal row
    _aw = (0.86 - 0.06 * (n_panels - 1)) / n_panels
    axes = [fig.add_axes([0.10 + i * (_aw + 0.06), 0.22, _aw, 0.68])
            for i in range(n_panels)]

    x = np.arange(len(configs))
    w = 0.38

    for idx, metric in enumerate(metric_names):
        ax = axes[idx]
        test_key = f"test_{metric}"
        vals = [float(data.get(c, {}).get(metric, 0)) for c in configs]
        tvals = [float(data.get(c, {}).get(test_key, 0)) for c in configs]
        bar_colors = [colors.get(c, "#888888") for c in configs]

        has_test = any(v != 0 for v in tvals)
        if has_test:
            ax.bar(
                x - w / 2, vals, w, color=bar_colors, alpha=0.85,
                edgecolor="black", linewidth=0.4, label="Val",
            )
            ax.bar(
                x + w / 2, tvals, w, color=bar_colors, alpha=0.4,
                edgecolor="black", linewidth=0.4, hatch="//", label="Test",
            )
        else:
            ax.bar(
                x, vals, w * 1.5, color=bar_colors, alpha=0.85,
                edgecolor="black", linewidth=0.4,
            )

        # Mark the best config
        best_i = int(np.argmax(vals))
        offset_x = -w / 2 if has_test else 0
        ax.annotate(
            "*", xy=(best_i + offset_x, vals[best_i]),
            xytext=(0, 2), textcoords="offset points",
            ha="center", fontsize=_style.FS_TITLE, fontweight="bold",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [_style.get_short_name(c) for c in configs],
            fontsize=_style.FS_TICK, rotation=45, ha="right",
        )
        ax.set_title(metric, fontsize=_style.FS_TITLE, pad=3)
        ax.tick_params(labelsize=_style.FS_TICK)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4, prune="upper"))
        ax.grid(alpha=0.22, linestyle="--", linewidth=0.4, axis="y")

        if idx == 0 and has_test:
            ax.legend(fontsize=_style.FS_LEGEND, loc="upper left")

    return _save_and_return(fig, outpath)


# ═══════════════════════════════════════════════════════════════════════════
# 3. UMAP grid
# ═══════════════════════════════════════════════════════════════════════════

def plot_umap_grid(
    latents_dict: Dict[str, np.ndarray],
    labels: Union[np.ndarray, Dict[str, np.ndarray]],
    configs: Optional[Sequence[str]] = None,
    outpath: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """2-row x 3-col UMAP scatter grid colored by cell-type labels.

    Unlike the standalone script, this function expects **pre-computed**
    2-D UMAP embeddings so it does not depend on the ``umap-learn`` package.
    Pass the raw 2-D arrays in *latents_dict*.

    Parameters
    ----------
    latents_dict : dict[str, ndarray]
        Mapping ``config_name -> (n_cells, 2)`` UMAP coordinates.
    labels : ndarray or dict[str, ndarray]
        Cell-type labels.  If a single array, it is broadcast to all configs.
        If a dict, maps ``config_name -> label_array``.
    configs : sequence of str, optional
        Which configs to plot (and in what order).
    outpath : str or Path, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _ensure_style()
    colors = _style.get_config_colors()
    configs = _resolve_configs(configs)
    # Only keep configs we actually have data for
    configs = [c for c in configs if c in latents_dict]
    n = len(configs)
    if n == 0:
        fig = plt.figure(figsize=(_style.FIG_WIDTH_IN, 3))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.text(0.5, 0.5, "No UMAP data available", transform=ax.transAxes,
                ha="center", va="center", fontsize=_style.FS_LABEL)
        ax.set_axis_off()
        return _save_and_return(fig, outpath)

    ncols = min(n, 3)
    nrows = max(1, (n + ncols - 1) // ncols)

    fig = plt.figure(figsize=(_style.FIG_WIDTH_IN, _style.FIG_HEIGHT_IN * 0.38 * nrows))
    # Explicit per-axis geometry: UMAP grid
    _cw = (0.92 - 0.04 * (ncols - 1)) / ncols
    _rh = (0.88 - 0.06 * (nrows - 1)) / nrows
    axes_grid = [
        [fig.add_axes([0.04 + c * (_cw + 0.04),
                       0.06 + 0.88 - (r + 1) * _rh - r * 0.06,
                       _cw, _rh])
         for c in range(ncols)]
        for r in range(nrows)
    ]
    axes = np.array(axes_grid)

    cmap = plt.colormaps.get_cmap("tab20")

    # Resolve labels
    if isinstance(labels, dict):
        labels_map = labels
    else:
        labels_map = {c: np.asarray(labels) for c in configs}

    # Determine global unique label set for consistent coloring
    all_labels = np.unique(
        np.concatenate([np.asarray(labels_map.get(c, np.zeros(len(latents_dict[c])))) for c in configs])
    )
    label_to_idx = {lb: i for i, lb in enumerate(all_labels)}

    for j, cfg in enumerate(configs):
        r, c = divmod(j, ncols)
        ax = axes[r, c]
        emb = np.asarray(latents_dict[cfg])
        lbl = np.asarray(labels_map.get(cfg, np.zeros(len(emb))))
        uniq = np.unique(lbl)

        for lb in uniq:
            mask = lbl == lb
            ax.scatter(
                emb[mask, 0], emb[mask, 1],
                color=cmap(label_to_idx[lb] % 20),
                s=0.5, alpha=0.55, linewidths=0, rasterized=True,
            )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            cfg, fontsize=_style.FS_TITLE, pad=2,
            color=colors.get(cfg, "#333333"), fontweight="bold",
        )
        for spine in ax.spines.values():
            spine.set_edgecolor(colors.get(cfg, "#333333"))
            spine.set_linewidth(1.0)

    # Hide unused axes
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].set_visible(False)

    # Legend in first panel
    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=cmap(label_to_idx[lb] % 20),
            markersize=2.5, linewidth=0,
        )
        for lb in all_labels
    ]
    axes[0, 0].legend(
        handles, [str(lb) for lb in all_labels],
        fontsize=_style.FS_LEGEND, ncol=2, loc="lower left",
        framealpha=0.65, handletextpad=0.1,
        borderpad=0.2, markerscale=0.9, columnspacing=0.4,
    )

    return _save_and_return(fig, outpath)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Training curves
# ═══════════════════════════════════════════════════════════════════════════

def plot_training_curves(
    loss_histories: Dict[str, dict],
    configs: Optional[Sequence[str]] = None,
    outpath: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Training and validation loss convergence curves.

    Parameters
    ----------
    loss_histories : dict[str, dict]
        Mapping ``config_name -> {"train": array, "val": array}``.
        Each array is a 1-D sequence of per-epoch loss values.
        Optionally includes ``"val_scores"`` as an ``(n_checkpoints, K)``
        array where columns are [ARI, NMI, ASW, ...] evaluated at
        equally-spaced epochs.
    configs : sequence of str, optional
        Subset / order of configs.
    outpath : str or Path, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _ensure_style()
    colors = _style.get_config_colors()
    configs = _resolve_configs(configs)
    configs = [c for c in configs if c in loss_histories]

    # Determine whether we have val_scores for metric evolution panels
    has_scores = any(
        "val_scores" in loss_histories[c] for c in configs
    )
    n_score_cols = 0
    if has_scores:
        for c in configs:
            vs = loss_histories[c].get("val_scores")
            if vs is not None:
                vs = np.asarray(vs)
                if vs.ndim == 2:
                    n_score_cols = vs.shape[1]
                    break
        n_score_cols = min(n_score_cols, 3)  # ARI, NMI, ASW

    n_top_panels = 2  # train + val loss
    n_bottom_panels = n_score_cols if has_scores else 0
    total_rows = 1 + (1 if n_bottom_panels > 0 else 0)

    fig = plt.figure(
        figsize=(_style.FIG_WIDTH_IN, _style.FIG_HEIGHT_IN * 0.22 * total_rows)
    )

    if total_rows == 1:
        row0_rect = [0.10, 0.18, 0.86, 0.74]
    else:
        row0_rect = [0.10, 0.56, 0.86, 0.38]

    # Explicit per-axis geometry: loss curve panels
    _r0_l, _r0_b, _r0_w, _r0_h = row0_rect
    _aw0 = (_r0_w - 0.08) / 2
    ax_train = fig.add_axes([_r0_l, _r0_b, _aw0, _r0_h])
    ax_val = fig.add_axes([_r0_l + _aw0 + 0.08, _r0_b, _aw0, _r0_h])

    max_epoch = 0
    for cfg in configs:
        h = loss_histories[cfg]
        tl = np.asarray(h.get("train", []), dtype=np.float64)
        vl = np.asarray(h.get("val", []), dtype=np.float64)

        c = colors.get(cfg, "#888888")
        ls = _style.get_line_style(cfg)
        lw = _style.get_line_width(cfg)

        if len(tl) > 0:
            ep_t = np.arange(len(tl))
            ax_train.plot(ep_t, tl, color=c, ls=ls, lw=lw, alpha=0.85, label=cfg)
            max_epoch = max(max_epoch, len(tl))

        if len(vl) > 0:
            if len(tl) > 0:
                val_epochs = np.linspace(0, len(tl) - 1, len(vl)).astype(int)
            else:
                val_epochs = np.arange(len(vl))
            ax_val.plot(val_epochs, vl, color=c, ls=ls, lw=lw, alpha=0.85, label=cfg)

    xlim = max_epoch + 5 if max_epoch > 0 else 305
    for ax, title, ylabel in [
        (ax_train, "Training Loss", "ELBO Loss"),
        (ax_val, "Validation Loss", "Val. ELBO Loss"),
    ]:
        ax.set_title(title, fontsize=_style.FS_TITLE, pad=3)
        ax.set_xlabel("Epoch", fontsize=_style.FS_AXIS)
        ax.set_ylabel(ylabel, fontsize=_style.FS_AXIS)
        ax.tick_params(labelsize=_style.FS_TICK)
        ax.grid(alpha=0.22, linestyle="--", linewidth=0.4)
        ax.set_xlim(0, xlim)

        # Shade convergence region (beyond epoch 100)
        if xlim > 100:
            ax.axvspan(100, xlim, alpha=0.05, color="gray")

    ax_train.legend(
        fontsize=_style.FS_LEGEND, loc="upper right",
        ncol=2, framealpha=0.65, handlelength=1.0,
    )

    # ── Row 1: val metric evolution (optional) ──
    if n_bottom_panels > 0:
        score_labels = ["Val ARI", "Val NMI", "Val ASW", "Val CAL", "Val DAV", "Val COR"]
        row1_rect = [0.10, 0.08, 0.86, 0.38]
        _r1_l, _r1_b, _r1_w, _r1_h = row1_rect
        _aw1 = (_r1_w - 0.06 * (n_bottom_panels - 1)) / n_bottom_panels
        score_axes = [fig.add_axes([_r1_l + i * (_aw1 + 0.06), _r1_b, _aw1, _r1_h])
                      for i in range(n_bottom_panels)]
        for si in range(n_bottom_panels):
            ax = score_axes[si]
            for cfg in configs:
                vs = loss_histories[cfg].get("val_scores")
                if vs is None:
                    continue
                vs = np.asarray(vs, dtype=np.float64)
                if vs.ndim == 2 and vs.shape[1] > si:
                    curve = vs[:, si]
                    tl = np.asarray(loss_histories[cfg].get("train", []))
                    n_train = len(tl) if len(tl) > 0 else 300
                    epochs = np.linspace(0, n_train - 1, len(curve)).astype(int)

                    c = colors.get(cfg, "#888888")
                    ls = _style.get_line_style(cfg)
                    lw = _style.get_line_width(cfg)
                    ax.plot(epochs, curve, color=c, ls=ls, lw=lw, alpha=0.85, label=cfg)

            title = score_labels[si] if si < len(score_labels) else f"Score {si}"
            ax.set_title(title, fontsize=_style.FS_TITLE, pad=3)
            ax.set_xlabel("Epoch", fontsize=_style.FS_AXIS)
            if si == 0:
                ax.set_ylabel("Score", fontsize=_style.FS_AXIS)
            ax.tick_params(labelsize=_style.FS_TICK)
            ax.grid(alpha=0.22, linestyle="--", linewidth=0.4)
            ax.set_xlim(0, xlim)
            ax.yaxis.set_major_locator(plt.MaxNLocator(4, prune="upper"))

    return _save_and_return(fig, outpath)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Pseudotime marker correlations
# ═══════════════════════════════════════════════════════════════════════════

def plot_pseudotime_markers(
    correlations_df,
    n_top: int = 8,
    outpath: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Marker-gene expression along pseudotime (smoothed line plots).

    Plots the top *n_top* genes ranked by their correlation with pseudotime.
    Each gene is shown as a smoothed line (rolling mean) of expression vs.
    pseudotime bin.

    Parameters
    ----------
    correlations_df : pandas.DataFrame or dict
        If a DataFrame, expected columns: ``"gene"``, ``"pseudotime"``,
        ``"expression"``, and optionally ``"correlation"`` (for ranking).
        If a dict, maps ``gene_name -> {"pseudotime": array, "expression":
        array, "correlation": float}``.
    n_top : int
        Number of top-correlated genes to plot.
    outpath : str or Path, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _ensure_style()

    # Normalize input
    if pd is not None and isinstance(correlations_df, pd.DataFrame):
        df = correlations_df
        genes = df["gene"].unique()
        # Rank by absolute mean correlation
        if "correlation" in df.columns:
            gene_corr = (
                df.groupby("gene")["correlation"]
                .mean().abs().sort_values(ascending=False)
            )
            top_genes = list(gene_corr.index[:n_top])
        else:
            top_genes = list(genes[:n_top])

        gene_data = {}
        for g in top_genes:
            gdf = df[df["gene"] == g].sort_values("pseudotime")
            gene_data[g] = {
                "pseudotime": gdf["pseudotime"].values,
                "expression": gdf["expression"].values,
                "correlation": gdf["correlation"].mean() if "correlation" in gdf.columns else 0.0,
            }
    else:
        # Dict input
        gene_data_raw = dict(correlations_df)
        ranked = sorted(
            gene_data_raw.items(),
            key=lambda kv: abs(float(kv[1].get("correlation", 0))),
            reverse=True,
        )
        gene_data = {k: v for k, v in ranked[:n_top]}

    n_genes = len(gene_data)
    if n_genes == 0:
        fig = plt.figure(figsize=(_style.FIG_WIDTH_IN, 2))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.text(0.5, 0.5, "No gene data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=_style.FS_TITLE)
        return _save_and_return(fig, outpath)

    ncols = min(n_genes, 4)
    nrows = max(1, (n_genes + ncols - 1) // ncols)

    fig = plt.figure(figsize=(_style.FIG_WIDTH_IN, _style.FIG_HEIGHT_IN * 0.2 * nrows))
    # Explicit per-axis geometry: pseudotime gene grid
    _cw = (0.86 - 0.05 * (ncols - 1)) / ncols
    _rh = (0.82 - 0.06 * (nrows - 1)) / nrows
    axes_grid = [
        [fig.add_axes([0.10 + c * (_cw + 0.05),
                       0.10 + 0.82 - (r + 1) * _rh - r * 0.06,
                       _cw, _rh])
         for c in range(ncols)]
        for r in range(nrows)
    ]
    axes = np.array(axes_grid)

    cmap = plt.colormaps.get_cmap("viridis")

    for idx, (gene, gd) in enumerate(gene_data.items()):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]

        pt = np.asarray(gd["pseudotime"], dtype=np.float64)
        expr = np.asarray(gd["expression"], dtype=np.float64)
        corr_val = float(gd.get("correlation", 0.0))

        # Sort by pseudotime
        order = np.argsort(pt)
        pt = pt[order]
        expr = expr[order]

        # Rolling mean smoothing (window = 5% of data or min 5)
        win = max(5, len(expr) // 20)
        if len(expr) >= win:
            kernel = np.ones(win) / win
            expr_smooth = np.convolve(expr, kernel, mode="same")
        else:
            expr_smooth = expr

        color = cmap(0.3 + 0.5 * abs(corr_val))
        ax.plot(pt, expr_smooth, color=color, lw=1.2, alpha=0.85)
        ax.fill_between(pt, expr_smooth, alpha=0.12, color=color)
        ax.set_title(
            f"{gene} (r={corr_val:+.2f})",
            fontsize=_style.FS_TITLE, pad=2,
        )
        ax.tick_params(labelsize=_style.FS_TICK)
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))

    # Common labels
    for c_idx in range(ncols):
        if nrows - 1 < axes.shape[0]:
            axes[nrows - 1, c_idx].set_xlabel("Pseudotime", fontsize=_style.FS_AXIS)
    for r_idx in range(nrows):
        axes[r_idx, 0].set_ylabel("Expression", fontsize=_style.FS_AXIS)

    # Hide unused axes
    for idx in range(n_genes, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle(
        f"Top {n_genes} Marker Genes vs. Pseudotime",
        fontsize=_style.FS_TITLE + 1, y=0.97,
    )
    return _save_and_return(fig, outpath)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Beta (KL weight) sensitivity sweep
# ═══════════════════════════════════════════════════════════════════════════

def plot_beta_sensitivity(
    sweep_results,
    metric_name: str = "ARI",
    configs: Optional[Sequence[str]] = None,
    outpath: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Metric vs. beta (KL weight) sensitivity sweep per config.

    Parameters
    ----------
    sweep_results : pandas.DataFrame or dict
        If a DataFrame, expected columns: ``"config"``, ``"beta"``,
        and one or more metric columns.
        If a dict, maps ``config_name -> {"beta": array, metric_name: array}``.
    metric_name : str
        Which metric column to plot on the y-axis.
    configs : sequence of str, optional
        Subset / order of configs.
    outpath : str or Path, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _ensure_style()
    colors = _style.get_config_colors()
    configs = _resolve_configs(configs)

    # Normalize input
    if pd is not None and isinstance(sweep_results, pd.DataFrame):
        df = sweep_results
        data = {}
        for cfg in configs:
            cdf = df[df["config"] == cfg].sort_values("beta")
            if len(cdf) > 0:
                data[cfg] = {
                    "beta": cdf["beta"].values,
                    metric_name: cdf[metric_name].values if metric_name in cdf.columns else np.zeros(len(cdf)),
                }
    else:
        data = dict(sweep_results)

    configs = [c for c in configs if c in data]

    if not configs:
        fig = plt.figure(figsize=(_style.FIG_WIDTH_IN, 3))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.text(0.5, 0.5, "No beta sweep data available", transform=ax.transAxes,
                ha="center", va="center", fontsize=_style.FS_LABEL)
        ax.set_axis_off()
        return _save_and_return(fig, outpath)

    fig = plt.figure(figsize=(_style.FIG_WIDTH_IN, _style.FIG_HEIGHT_IN * 0.28))
    ax = fig.add_axes([0.12, 0.18, 0.84, 0.72])

    for cfg in configs:
        d = data[cfg]
        beta = np.asarray(d["beta"], dtype=np.float64)
        vals = np.asarray(d.get(metric_name, np.zeros_like(beta)), dtype=np.float64)

        c = colors.get(cfg, "#888888")
        ls = _style.get_line_style(cfg)
        lw = _style.get_line_width(cfg)

        ax.plot(beta, vals, color=c, ls=ls, lw=lw, alpha=0.85, label=cfg, marker="o", markersize=3)

    ax.set_xlabel("Beta (KL weight)", fontsize=_style.FS_AXIS)
    ax.set_ylabel(metric_name, fontsize=_style.FS_AXIS)
    ax.set_title(
        f"{metric_name} vs. Beta Sensitivity",
        fontsize=_style.FS_TITLE, pad=3,
    )
    ax.tick_params(labelsize=_style.FS_TICK)
    ax.grid(alpha=0.22, linestyle="--", linewidth=0.4)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))

    # Log scale for beta if range spans > 1 order of magnitude
    all_betas = np.concatenate([np.asarray(data[c]["beta"]) for c in configs])
    if all_betas.min() > 0 and all_betas.max() / all_betas.min() > 10:
        ax.set_xscale("log")

    ax.legend(
        fontsize=_style.FS_LEGEND, loc="best",
        ncol=2, framealpha=0.65, handlelength=1.0,
    )

    return _save_and_return(fig, outpath)
