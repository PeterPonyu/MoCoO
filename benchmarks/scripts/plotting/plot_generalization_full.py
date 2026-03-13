#!/usr/bin/env python
"""
MoCoO Figure 7-Full — All-Metrics Generalization Comparison
=============================================================
Paired bar chart showing metrics on four splits (train / val / test / full)
for every configuration and **all** 37 scorable metrics.

This script first recomputes split-level metrics from the saved latent
embeddings in ``benchmark_data.npz`` (using the same 70/15/15 random
split as ``recompute_metrics.py``), then produces a multi-row figure
so the user can visually compare every metric.

Usage:
    python -m benchmarks.scripts.plotting.plot_generalization_full \
        --resultsdir benchmarks/results/beta_ablation/beta_0.1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from mocoo.evaluation import compute_all_metrics
from mocoo.visualization.style import (
    DPI, FS_TITLE, FS_AXIS, FS_TICK, FS_SMALL,
    apply_style, get_config_order, get_config_colors, get_tick_name,
    save_figure,
)

apply_style()

_CONFIGS = get_config_order()
_COLORS = get_config_colors()

# ── Metrics to plot (key, display label, higher_is_better) ─────────────
_METRICS = [
    # Clustering
    ("ARI",  "ARI ↑",  True),
    ("NMI",  "NMI ↑",  True),
    ("ASW",  "ASW ↑",  True),
    ("CAL",  "CAL ↓",  False),
    ("DAV",  "DAV ↓",  False),
    ("COR",  "COR ↑",  True),
    # DRE – UMAP
    ("DRE_umap_overall_quality",     "DRE UMAP ↑",  True),
    ("DRE_umap_Q_global",            "DRE UMAP Qg ↑", True),
    ("DRE_umap_Q_local",             "DRE UMAP Ql ↑", True),
    ("DRE_umap_distance_correlation","DRE UMAP dcor ↑", True),
    # DRE – tSNE
    ("DRE_tsne_overall_quality",     "DRE tSNE ↑",  True),
    ("DRE_tsne_Q_global",            "DRE tSNE Qg ↑", True),
    ("DRE_tsne_Q_local",             "DRE tSNE Ql ↑", True),
    ("DRE_tsne_distance_correlation","DRE tSNE dcor ↑", True),
    # DREX
    ("DREX_overall_quality",         "DREX ↑",  True),
    ("DREX_trustworthiness",         "DREX trust ↑", True),
    ("DREX_continuity",              "DREX cont ↑",  True),
    ("DREX_distance_pearson",        "DREX dpear ↑", True),
    ("DREX_distance_spearman",       "DREX dspear ↑", True),
    ("DREX_knn_rank_correlation",    "DREX knn ↑",   True),
    ("DREX_local_scale_quality",     "DREX lscale ↑", True),
    ("DREX_neighborhood_symmetry",   "DREX nsym ↑",  True),
    # LSE
    ("LSE_overall_quality",          "LSE ↑",  True),
    ("LSE_core_quality",             "LSE core ↑",   True),
    ("LSE_manifold_dimensionality",  "LSE manif ↑",  True),
    ("LSE_noise_resilience",         "LSE noise ↑",  True),
    ("LSE_spectral_decay_rate",      "LSE decay",    True),
    ("LSE_anisotropy_score",         "LSE aniso",    True),
    ("LSE_participation_ratio",      "LSE part",     True),
    # LSEX
    ("LSEX_overall_quality",         "LSEX ↑",  True),
    ("LSEX_cluster_compactness",     "LSEX compact ↑", True),
    ("LSEX_inter_cluster_gap",       "LSEX gap ↑",    True),
    ("LSEX_local_curvature",         "LSEX curv ↑",   True),
    ("LSEX_neighbor_purity",         "LSEX purity ↑", True),
    ("LSEX_radial_concentration",    "LSEX radial ↑", True),
    ("LSEX_sampling_stability",      "LSEX stab ↑",   True),
    ("LSEX_two_hop_connectivity",    "LSEX 2hop ↑",   True),
]


# ── Split indices (same seed as recompute_metrics.py) ─────────────────
def _make_splits(n: int):
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    n_train = int(0.7 * n)
    n_val   = int(0.15 * n)
    return {
        "train": perm[:n_train],
        "val":   perm[n_train:n_train + n_val],
        "test":  perm[n_train + n_val:],
        "full":  np.arange(n),
    }


def _compute_split_metrics(rdir: Path) -> dict:
    """Return {config: {split: {metric_key: value}}}."""
    cache_path = rdir / "split_metrics_cache.json"
    if cache_path.exists():
        print(f"Loading cached split metrics from {cache_path.name}")
        with open(cache_path) as f:
            return json.load(f)

    npz_path = rdir / "benchmark_data.npz"
    if not npz_path.exists():
        print(f"No benchmark_data.npz found in {rdir}")
        return {}

    npz = np.load(npz_path, allow_pickle=True)
    configs = list(npz["configs"])
    latents = list(npz["latents"])
    labels_all = list(npz["labels"])

    n = latents[0].shape[0]
    splits = _make_splits(n)
    print(f"Split sizes: train={len(splits['train'])}, val={len(splits['val'])}, "
          f"test={len(splits['test'])}, full={n}")

    results = {}
    for i, (cfg, latent, labels) in enumerate(zip(configs, latents, labels_all)):
        print(f"  [{i+1}/{len(configs)}] {cfg}")
        cfg_results = {}
        for split_name, idx in splits.items():
            z = latent[idx]
            y = labels[idx]
            if len(z) < 20:
                continue
            t0 = time.time()
            m = compute_all_metrics(z, y, dre_k=15)
            elapsed = time.time() - t0
            # Keep only numeric, non-private keys
            clean = {}
            for k, v in m.items():
                if k.startswith("_"):
                    continue
                if isinstance(v, (int, float, np.integer, np.floating)):
                    clean[k] = round(float(v), 6)
            cfg_results[split_name] = clean
            print(f"    {split_name}: {len(clean)} metrics in {elapsed:.1f}s")
        results[cfg] = cfg_results

    # Cache
    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Cached to {cache_path.name}")
    return results


def _draw_grouped_bars(ax, split_data, metric_key, metric_label, configs):
    """Draw grouped bars (train / val / test / full) for one metric."""
    splits = ["train", "val", "test", "full"]
    hatches = ["",  "//",  "\\\\",  "xx"]
    alphas  = [0.85, 0.70, 0.55, 0.40]

    n = len(configs)
    n_splits = len(splits)
    width = 0.8 / n_splits
    x = np.arange(n)

    all_vals = []
    for s in splits:
        for cfg in configs:
            v = split_data.get(cfg, {}).get(s, {}).get(metric_key, np.nan)
            if np.isfinite(v):
                all_vals.append(v)

    if all_vals:
        ymin, ymax = min(all_vals), max(all_vals)
        margin = (ymax - ymin) * 0.30 if ymax > ymin else 0.1
        y_lo = max(0, ymin - margin * 0.5)
        y_hi = ymax + margin
    else:
        y_lo, y_hi = 0, 1

    for si, s in enumerate(splits):
        vals = []
        for cfg in configs:
            v = split_data.get(cfg, {}).get(s, {}).get(metric_key, np.nan)
            vals.append(v)
        offset = (si - n_splits / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            [v - y_lo if np.isfinite(v) else 0 for v in vals],
            width, bottom=y_lo,
            color=[_COLORS[c] for c in configs],
            edgecolor="white", linewidth=0.3,
            hatch=hatches[si], alpha=alphas[si],
            label=s if configs[0] == configs[0] else None,
        )

    ax.set_ylim(y_lo, y_hi)
    ax.set_xticks(x)
    ax.set_xticklabels([get_tick_name(c) for c in configs],
                        fontsize=max(FS_SMALL - 2, 5), rotation=90, ha="center")
    ax.set_title(metric_label, fontsize=FS_SMALL, pad=2)
    ax.tick_params(axis="y", labelsize=max(FS_SMALL - 2, 5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=3, prune="both"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.18, linestyle="--", linewidth=0.3, axis="y")


def build_figure(rdir: Path, outdir: Path):
    split_data = _compute_split_metrics(rdir)
    if not split_data:
        return

    configs_present = [c for c in _CONFIGS if c in split_data]
    n_metrics = len(_METRICS)
    n_cols = 6
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 2.8, n_rows * 2.4),
                              constrained_layout=True)
    axes = axes.flatten()

    for j, (mk, ml, _) in enumerate(_METRICS):
        _draw_grouped_bars(axes[j], split_data, mk, ml, configs_present)

    # Hide unused axes
    for j in range(n_metrics, len(axes)):
        axes[j].set_visible(False)

    # Legend
    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(facecolor="#888888", alpha=0.85, label="Train (70%)"),
        mpatches.Patch(facecolor="#888888", alpha=0.70, hatch="//",  label="Val (15%)"),
        mpatches.Patch(facecolor="#888888", alpha=0.55, hatch="\\\\", label="Test (15%)"),
        mpatches.Patch(facecolor="#888888", alpha=0.40, hatch="xx",  label="Full (3000)"),
    ]
    fig.legend(handles=legend_patches, loc="upper center",
               ncol=4, fontsize=FS_SMALL, frameon=False,
               bbox_to_anchor=(0.5, 1.02))

    outpath = outdir / "fig7_generalization_full.png"
    save_figure(fig, outpath, bbox_inches='tight', pad_inches=0.06)
    plt.close(fig)
    print(f"Saved: {outpath}")


def main():
    _benchmarks = Path(__file__).resolve().parent.parent.parent
    p = argparse.ArgumentParser()
    p.add_argument("--resultsdir",
                   default=str(_benchmarks / "results" / "beta_ablation" / "beta_0.1"))
    p.add_argument("--outdir",
                   default=str(_benchmarks / "figures"))
    args = p.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    build_figure(Path(args.resultsdir), outdir)


if __name__ == "__main__":
    main()
