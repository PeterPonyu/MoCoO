#!/usr/bin/env python
"""Generate LaTeX tables for the MoCoO paper from Figure 1 ablation CSV data.

Reads summary_expanded.csv from all 5 datasets (split=whole), computes
cross-dataset means, and emits LaTeX table environments to stdout.

Tables produced:
  1. Cross-dataset mean — Clustering (6 metrics)
  2. Cross-dataset mean — Embedding quality (DRE UMAP + DRE tSNE + DREX = 11)
  3–7. Per-dataset tables (appendix) — all 17 metrics per dataset
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

_DATASETS = ["IRALL", "dentate", "endo", "paul", "spinoids"]
_CONFIGS = ["VAE", "VAE+ODE", "VAE+MoCo", "VAE+MoCo+Proto", "VAE+ODE+MoCo", "Full"]

_CLUSTER_METRICS = ["ARI", "NMI", "ASW", "DAV", "CAL", "COR"]
_DRE_UMAP = [
    "DRE_umap_distance_correlation",
    "DRE_umap_Q_local",
    "DRE_umap_Q_global",
    "DRE_umap_overall_quality",
]
_DRE_TSNE = [
    "DRE_tsne_distance_correlation",
    "DRE_tsne_Q_local",
    "DRE_tsne_Q_global",
    "DRE_tsne_overall_quality",
]
_DREX = [
    "DREX_distance_spearman",
    "DREX_local_scale_quality",
    "DREX_overall_quality",
]
_EMBED_METRICS = _DRE_UMAP + _DRE_TSNE + _DREX

# Lower-is-better metrics
_LOWER_BETTER = {"DAV"}

_CLUSTER_HEADERS = {
    "ARI": "ARI", "NMI": "NMI", "ASW": "ASW",
    "DAV": r"DAV$\downarrow$", "CAL": "CAL", "COR": "COR",
}
_EMBED_HEADERS = {
    "DRE_umap_distance_correlation": r"\makecell{UMAP\\dist.c.}",
    "DRE_umap_Q_local": r"\makecell{UMAP\\$Q_l$}",
    "DRE_umap_Q_global": r"\makecell{UMAP\\$Q_g$}",
    "DRE_umap_overall_quality": r"\makecell{UMAP\\ovr.}",
    "DRE_tsne_distance_correlation": r"\makecell{tSNE\\dist.c.}",
    "DRE_tsne_Q_local": r"\makecell{tSNE\\$Q_l$}",
    "DRE_tsne_Q_global": r"\makecell{tSNE\\$Q_g$}",
    "DRE_tsne_overall_quality": r"\makecell{tSNE\\ovr.}",
    "DREX_distance_spearman": r"\makecell{DREX\\Spear.}",
    "DREX_local_scale_quality": r"\makecell{DREX\\local}",
    "DREX_overall_quality": r"\makecell{DREX\\ovr.}",
}

_CONFIG_DISPLAY = {
    "VAE": "VAE",
    "VAE+ODE": "VAE+ODE",
    "VAE+MoCo": "VAE+MoCo",
    "VAE+MoCo+Proto": "VAE+MoCo+Proto",
    "VAE+ODE+MoCo": "VAE+ODE+MoCo",
    "Full": "Full (MoCoO)",
}

_DATASET_DISPLAY = {
    "IRALL": "IRALL",
    "dentate": "Dentate",
    "endo": "Endo",
    "paul": "Paul",
    "spinoids": "Spinoids",
}


def _load_whole_rows(results_dir: Path) -> dict[str, dict[str, dict[str, float]]]:
    """Return {dataset: {config: {metric: value}}}."""
    data = {}
    for ds in _DATASETS:
        csv_path = results_dir / ds / "summary_expanded.csv"
        if not csv_path.exists():
            print(f"WARNING: {csv_path} not found, skipping", file=sys.stderr)
            continue
        data[ds] = {}
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if row["split"] != "whole":
                    continue
                cfg = row["config"]
                data[ds][cfg] = {k: float(row[k]) for k in row if k not in ("config", "split")}
    return data


def _cross_dataset_mean(data: dict) -> dict[str, dict[str, float]]:
    """Compute mean metric values across datasets per config."""
    means = {}
    for cfg in _CONFIGS:
        vals: dict[str, list[float]] = {}
        for ds in _DATASETS:
            if ds not in data or cfg not in data[ds]:
                continue
            for m, v in data[ds][cfg].items():
                vals.setdefault(m, []).append(v)
        means[cfg] = {m: sum(vs) / len(vs) for m, vs in vals.items()}
    return means


def _format_val(val: float, metric: str, is_best: bool) -> str:
    """Format a numeric value; bold if best."""
    if metric == "CAL":
        s = f"{val:.0f}"
    elif metric == "COR":
        s = f"{val:.3f}"
    else:
        s = f"{val:.3f}"
    return rf"\textbf{{{s}}}" if is_best else s


def _find_best(rows: dict[str, dict[str, float]], metrics: list[str]) -> dict[str, str]:
    """Return {metric: config_name} of the best config per metric."""
    best = {}
    for m in metrics:
        vals = {cfg: rows[cfg][m] for cfg in _CONFIGS if cfg in rows and m in rows[cfg]}
        if not vals:
            continue
        if m in _LOWER_BETTER:
            best[m] = min(vals, key=vals.get)
        else:
            best[m] = max(vals, key=vals.get)
    return best


def _emit_table(rows: dict[str, dict[str, float]], metrics: list[str],
                headers: dict[str, str], caption: str, label: str,
                star: bool = False):
    """Print a LaTeX table to stdout."""
    env = "table*" if star else "table"
    ncols = len(metrics) + 1
    col_spec = "l" + "c" * len(metrics)
    best = _find_best(rows, metrics)

    print(f"\\begin{{{env}}}[!t]")
    print("    \\centering")
    print(f"    \\caption{{{caption}}}")
    print(f"    \\label{{{label}}}")
    if star or len(metrics) > 6:
        print("    \\resizebox{\\textwidth}{!}{%")
    else:
        print("    \\resizebox{\\columnwidth}{!}{%")
    print(f"    \\begin{{tabular}}{{{col_spec}}}")
    print("        \\toprule")

    # Header row
    hdr = "        Config"
    for m in metrics:
        hdr += f" & {headers.get(m, m)}"
    hdr += " \\\\"
    print(hdr)
    print("        \\midrule")

    # Data rows
    for cfg in _CONFIGS:
        if cfg not in rows:
            continue
        row_str = f"        {_CONFIG_DISPLAY[cfg]}"
        for m in metrics:
            val = rows[cfg].get(m, float("nan"))
            is_best = best.get(m) == cfg
            row_str += f" & {_format_val(val, m, is_best)}"
        row_str += " \\\\"
        print(row_str)

    print("        \\bottomrule")
    print("    \\end{tabular}%")
    if star or len(metrics) > 6:
        print("    }")
    else:
        print("    }")
    print(f"\\end{{{env}}}")
    print()


def _emit_per_dataset(data: dict, ds: str, label_prefix: str):
    """Emit a combined 17-metric table for one dataset (appendix)."""
    if ds not in data:
        return
    rows = data[ds]
    metrics = _CLUSTER_METRICS + _EMBED_METRICS
    headers = {**_CLUSTER_HEADERS, **_EMBED_HEADERS}
    disp = _DATASET_DISPLAY[ds]
    _emit_table(
        rows, metrics, headers,
        caption=f"{disp} --- all 17 evaluation metrics (split=whole). Best per column in \\textbf{{bold}}. DAV$\\downarrow$ indicates lower is better.",
        label=f"tab:{label_prefix}_{ds.lower()}",
        star=True,
    )


def main():
    results_dir = Path(__file__).resolve().parent.parent.parent / "results"
    data = _load_whole_rows(results_dir)
    means = _cross_dataset_mean(data)

    print("%" * 72)
    print("% AUTO-GENERATED TABLES — do not edit by hand")
    print("%" * 72)
    print()

    # 1. Cross-dataset mean clustering
    _emit_table(
        means, _CLUSTER_METRICS, _CLUSTER_HEADERS,
        caption="Cross-dataset mean clustering metrics (5 datasets, split=whole). Best per column in \\textbf{bold}. DAV$\\downarrow$ indicates lower is better.",
        label="tab:mean_clustering",
        star=False,
    )

    # 2. Cross-dataset mean embedding
    _emit_table(
        means, _EMBED_METRICS, _EMBED_HEADERS,
        caption="Cross-dataset mean embedding quality metrics (5 datasets, split=whole). Best per column in \\textbf{bold}.",
        label="tab:mean_embedding",
        star=True,
    )

    # 3–7. Per-dataset appendix tables
    print("%" * 72)
    print("% Per-dataset appendix tables")
    print("%" * 72)
    print()
    for ds in _DATASETS:
        _emit_per_dataset(data, ds, "ablation")


if __name__ == "__main__":
    main()
