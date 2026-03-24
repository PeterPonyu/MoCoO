#!/usr/bin/env python
"""MoCoO Supplementary Figure S4 -- Trajectory & pseudotime analysis (merged).

Six-panel figure combining ODE-driven trajectory diagnostics (old S4) with
pseudotime correlation and trajectory-baseline comparisons (old S8).

Layout (2 rows x 3 columns):
  Top row:
    (A) PCA of latent spaces coloured by cell type (3 configs side by side)
    (B) Pseudotime Spearman |rho| per ODE-containing config (bar chart)
    (C) Trajectory baselines: MoCoO vs DPT vs Palantir (horizontal bars)
  Bottom row:
    (D) Per-cell-type pseudotime distribution (violin plots)
    (E) Per-dimension gradient magnitude (gene expression importance)
    (F) Trajectory smoothness: sequential pairwise distance histograms
"""
from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from benchmarks.scripts.plotting.shared import setup_fonts
from mocoo.visualization.style import (
    FS_AXIS, FS_LEGEND, FS_SMALL, FS_TICK, FS_TITLE,
    FIG_WIDTH_IN, DPI,
    apply_style, save_figure, add_panel_label,
    get_config_colors, get_display_name, get_tick_name,
)
from mocoo.visualization.direct_layout import bind_figure_region

setup_fonts()
apply_style()

# ── Constants ──────────────────────────────────────────────────────────────
_CONFIGS = ["VAE", "VAE+ODE", "Full"]
_DIR_MAP = {"VAE": "VAE", "VAE+ODE": "VAE_ODE", "Full": "Full"}

_TRAJ_COLORS = {
    # DPT and Palantir removed — not deep-learning methods
}

# ── Data-loading helpers ──────────────────────────────────

def _load_latents(results_dir: Path, config: str):
    """Load latent embeddings and labels for a given config (IRALL dataset)."""
    cfg_dir = _DIR_MAP.get(config, config)
    fp = results_dir / "IRALL" / cfg_dir / "latents.npz"
    if not fp.exists():
        return None, None
    d = np.load(fp)
    return d["whole_latent"], d["whole_labels"]


def _load_gradients(results_dir: Path, config: str):
    """Load gradient magnitudes for a given config (IRALL dataset)."""
    cfg_dir = _DIR_MAP.get(config, config)
    fp = results_dir / "IRALL" / cfg_dir / "gradients.npy"
    if not fp.exists():
        return None
    return np.load(fp)


def _compute_pseudotime(latent: np.ndarray) -> np.ndarray:
    """Infer pseudotime as PC1 projection, normalised to [0, 1]."""
    from sklearn.decomposition import PCA
    pc1 = PCA(n_components=1).fit_transform(latent).ravel()
    mn, mx = pc1.min(), pc1.max()
    if mx - mn < 1e-12:
        return np.zeros_like(pc1)
    return (pc1 - mn) / (mx - mn)


def _pca2d(latent: np.ndarray) -> np.ndarray:
    from sklearn.decomposition import PCA
    return PCA(n_components=2).fit_transform(latent)


def _pairwise_dists(latent: np.ndarray, n_sample: int = 2000) -> np.ndarray:
    """Compute sequential pairwise distances after pseudotime ordering."""
    pt = _compute_pseudotime(latent)
    order = np.argsort(pt)
    z = latent[order]
    if len(z) > n_sample:
        idx = np.linspace(0, len(z) - 1, n_sample, dtype=int)
        z = z[idx]
    return np.linalg.norm(np.diff(z, axis=0), axis=1)


# ── Data-loading helpers (from old figS8) ──────────────────────────────────

def _load_pseudotime(results_dir: Path):
    """Load pseudotime validation CSV -> per-config mean |rho|."""
    fp = results_dir / "pseudotime_validation" / "pseudotime_validation.csv"
    if not fp.exists():
        return {}
    acc = {}
    with open(fp) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cfg = row["config"].strip()
            try:
                rho = abs(float(row["spearman_rho"]))
            except (KeyError, ValueError):
                continue
            acc.setdefault(cfg, []).append(rho)
    return {c: np.mean(v) for c, v in acc.items()}


def _load_trajectory(results_dir: Path):
    """Load trajectory baselines CSV -> per-method mean |rho| across datasets."""
    fp = results_dir / "trajectory_baselines" / "trajectory_baselines.csv"
    if not fp.exists():
        return {}
    acc = {}
    with open(fp) as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["method"].strip()
            try:
                rho = float(row["spearman_abs"])
            except (KeyError, ValueError):
                continue
            acc.setdefault(method, []).append(rho)
    return {m: np.mean(v) for m, v in acc.items()}


# ── Main figure ────────────────────────────────────────────────────────────

def make_figure(results_dir: Path, out_path: Path):
    config_colors = get_config_colors()
    pseudo = _load_pseudotime(results_dir)
    traj = _load_trajectory(results_dir)

    # 2-row x 3-col figure; slightly taller than wide
    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * 0.95))
    root = bind_figure_region(fig, (0.08, 0.06, 0.92, 0.92))
    (r_top, r_bot) = root.split_rows([1, 1], gap=0.12)

    # Top row: A (PCA, wider), B, C
    (r_a, r_b, r_c) = r_top.split_cols([3, 2, 2], gap=0.08)
    # Bottom row: D, E, F
    (r_d, r_e, r_f) = r_bot.split_cols([1, 1, 1], gap=0.10)

    # ── Panel A: PCA coloured by cell type (3 configs side by side) ────────
    pca_sub = r_a.split_cols([1] * len(_CONFIGS), gap=0.04)

    all_labels = set()
    for cfg in _CONFIGS:
        lat, lbl = _load_latents(results_dir, cfg)
        if lbl is not None:
            all_labels.update(np.unique(lbl))
    label_list = sorted(all_labels)
    label_cmap = plt.cm.tab20(np.linspace(0, 1, max(len(label_list), 1)))

    ax_a_first = None
    for i, cfg in enumerate(_CONFIGS):
        lat, lbl = _load_latents(results_dir, cfg)
        if lat is None:
            continue
        pca = _pca2d(lat)
        ax = pca_sub[i].add_axes(fig)
        if ax_a_first is None:
            ax_a_first = ax
        for j, lab in enumerate(label_list):
            mask = lbl == lab
            ax.scatter(pca[mask, 0], pca[mask, 1], s=1.5,
                       c=[label_cmap[j % len(label_cmap)]], alpha=0.5,
                       rasterized=True)
        ax.set_title(get_display_name(cfg), fontsize=FS_TICK)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("Cell type", fontsize=FS_SMALL)

    if ax_a_first is not None:
        add_panel_label(ax_a_first, "A", x=-0.30, y=1.14)

    # ── Panel B: Pseudotime Spearman |rho| per config ──────────────────────
    ax_b = r_b.add_axes(fig)
    if pseudo:
        configs = sorted(pseudo.keys())
        x = np.arange(len(configs))
        vals = [pseudo[c] for c in configs]
        colors = [config_colors.get(c, "#888888") for c in configs]
        ax_b.bar(x, vals, 0.6, color=colors, zorder=3)
        ax_b.set_xticks(x)
        ax_b.set_xticklabels([get_tick_name(c) for c in configs],
                             fontsize=FS_SMALL, rotation=45, ha="right")
        ax_b.set_ylabel("Mean |Spearman rho|", fontsize=FS_AXIS)
        ax_b.set_title("Pseudotime Correlation", fontsize=FS_TITLE)
        # Annotate values
        ymax = ax_b.get_ylim()[1]
        for i, v in enumerate(vals):
            if v + 0.005 < ymax * 0.92:
                ax_b.text(i, v + 0.005, f"{v:.3f}", ha="center",
                          va="bottom", fontsize=FS_SMALL)
            else:
                ax_b.text(i, v - 0.008, f"{v:.3f}", ha="center",
                          va="top", fontsize=FS_SMALL, color="white",
                          fontweight="bold")
    add_panel_label(ax_b, "B", x=-0.18, y=1.14)

    # ── Panel C: Trajectory baselines comparison ───────────────────────────
    ax_c = r_c.add_axes(fig)
    if traj:
        # Filter out non-DL methods (DPT, Palantir)
        _skip = {"DPT", "Palantir"}
        traj = {k: v for k, v in traj.items() if k not in _skip}
        methods = sorted(traj.keys())
        x = np.arange(len(methods))
        vals = [traj[m] for m in methods]
        colors = []
        for m in methods:
            if m in _TRAJ_COLORS:
                colors.append(_TRAJ_COLORS[m])
            elif m.startswith("MoCoO_"):
                cfg = m.replace("MoCoO_", "")
                colors.append(config_colors.get(cfg, "#D55E00"))
            else:
                colors.append("#888888")
        ax_c.barh(x, vals, 0.6, color=colors, zorder=3)
        ax_c.set_yticks(x)
        labels = [m.replace("MoCoO_", "") for m in methods]
        ax_c.set_yticklabels(labels, fontsize=FS_TICK)
        ax_c.set_xlabel("Mean |Spearman rho|", fontsize=FS_AXIS)
        ax_c.set_title("Trajectory Methods", fontsize=FS_TITLE)
        for i, v in enumerate(vals):
            ax_c.text(v + 0.01, i, f"{v:.3f}", ha="left", va="center",
                      fontsize=FS_SMALL)
    add_panel_label(ax_c, "C", x=-0.18, y=1.14)

    # ── Panel D: Per-cell-type pseudotime violins (Full config) ────────────
    ax_d = r_d.add_axes(fig)
    lat_full, lbl_full = _load_latents(results_dir, "Full")
    if lat_full is not None:
        pt_full = _compute_pseudotime(lat_full)
        unique_labels = sorted(set(lbl_full) if isinstance(lbl_full[0], str)
                               else np.unique(lbl_full).tolist())
        vdata = [pt_full[lbl_full == lab] for lab in unique_labels]
        parts = ax_d.violinplot(vdata, positions=range(len(unique_labels)),
                                showmeans=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_alpha(0.6)
        ax_d.set_xticks(range(len(unique_labels)))
        tick_labels = [str(l)[:8] for l in unique_labels]
        ax_d.set_xticklabels(tick_labels, fontsize=FS_SMALL - 1,
                             rotation=45, ha="right")
        ax_d.set_ylabel("Pseudotime", fontsize=FS_AXIS)
        ax_d.set_title("Per-type Dist. (V+OMP)", fontsize=FS_TITLE)
    add_panel_label(ax_d, "D", x=-0.22, y=1.14)

    # ── Panel E: Gradient magnitude per latent dimension ───────────────────
    ax_e = r_e.add_axes(fig)
    for cfg in _CONFIGS:
        grads = _load_gradients(results_dir, cfg)
        if grads is None:
            continue
        dim_importance = np.mean(np.abs(grads), axis=0)
        color = config_colors.get(cfg, "#888888")
        ax_e.bar(np.arange(len(dim_importance)) + _CONFIGS.index(cfg) * 0.25,
                 dim_importance, width=0.22, color=color, alpha=0.8,
                 label=get_display_name(cfg))
    ax_e.set_xlabel("Latent Dimension", fontsize=FS_AXIS)
    ax_e.set_ylabel("Mean |Gradient|", fontsize=FS_AXIS)
    ax_e.set_title("Gene Expr. Importance", fontsize=FS_TITLE)
    ax_e.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_e.legend(fontsize=FS_SMALL, loc="upper left", frameon=False)
    add_panel_label(ax_e, "E", x=-0.22, y=1.14)

    # ── Panel F: Trajectory smoothness histograms ──────────────────────────
    ax_f = r_f.add_axes(fig)
    for cfg in _CONFIGS:
        lat, _ = _load_latents(results_dir, cfg)
        if lat is None:
            continue
        dists = _pairwise_dists(lat)
        color = config_colors.get(cfg, "#888888")
        ax_f.hist(dists, bins=50, alpha=0.5, color=color, density=True,
                  label=get_display_name(cfg))
    ax_f.set_xlabel("Sequential Pairwise Dist.", fontsize=FS_AXIS)
    ax_f.set_ylabel("Density", fontsize=FS_AXIS)
    ax_f.set_title("Trajectory Smoothness", fontsize=FS_TITLE)
    ax_f.legend(fontsize=FS_SMALL, loc="upper left", frameon=False)
    add_panel_label(ax_f, "F", x=-0.18, y=1.14)

    save_figure(fig, str(out_path), vcd_label="figS2_trajectory",
                vcd_verbose=True)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="MoCoO Supp Fig S4: Trajectory & Pseudotime (merged)")
    parser.add_argument("--resultsdir", type=Path,
                        default=Path(__file__).resolve().parent.parent.parent
                        / "results")
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=None)
    args = parser.parse_args()
    outdir = args.outdir or (args.resultsdir.parent / "figures")
    Path(outdir).mkdir(parents=True, exist_ok=True)
    make_figure(args.resultsdir, Path(outdir) / "figS2_trajectory.png")


if __name__ == "__main__":
    main()
