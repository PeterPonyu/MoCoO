#!/usr/bin/env python
"""Generate LaTeX tables for the MoCoO paper from ablation CSV data.

Reads summary_expanded.csv from all 20 datasets (split=whole), computes
cross-dataset means, and emits LaTeX table environments to stdout.

Tables produced:
  1. Cross-dataset mean — Clustering (3 proposed metrics: ASW, DAV, CAL)
     — 6 base configs, 20-dataset average
  2. Cross-dataset mean — Embedding quality (2 proposed: DRE, DREX)
     — 6 base configs, 20-dataset average
  3. FM Enhancement — Clustering (12 configs, 20-dataset average)
  4. FM Enhancement — Quality   (12 configs, 20-dataset average)
  5. FM delta summary (mean improvement per metric across 20 datasets)
  6-25. Per-dataset appendix tables — proposed 5 metrics per dataset
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

# All 20 datasets with results
_DATASETS = [
    "IRALL", "astrocyte", "brainmet", "breast", "dentate",
    "endo", "gastric", "hemato", "hepatoblastoma", "hESCtime",
    "livercancer", "lung", "melanoma", "paul", "pituitary",
    "retina", "setty", "spine", "spinoids", "teeth",
]

_BASE_CONFIGS = [
    "VAE", "VAE+ODE", "VAE+MoCo", "VAE+MoCo+Proto", "VAE+ODE+MoCo", "Full",
]
_FM_CONFIGS = [
    "VAE+FM", "VAE+ODE+FM", "VAE+MoCo+FM", "VAE+MoCo+Proto+FM",
    "VAE+ODE+MoCo+FM", "Full+FM",
]
_ALL_CONFIGS = _BASE_CONFIGS + _FM_CONFIGS

# FM pairing: base -> FM variant
_FM_PAIRS = {
    "VAE": "VAE+FM",
    "VAE+ODE": "VAE+ODE+FM",
    "VAE+MoCo": "VAE+MoCo+FM",
    "VAE+MoCo+Proto": "VAE+MoCo+Proto+FM",
    "VAE+ODE+MoCo": "VAE+ODE+MoCo+FM",
    "Full": "Full+FM",
}

# Proposed 5-metric set (canonical: ASW, DAV, CAL + DRE, DREX)
_CLUSTER_METRICS = ["ASW", "DAV", "CAL"]
_QUALITY_METRICS = [
    "DRE_umap_overall_quality",
    "DREX_overall_quality",
]
_ALL_PROPOSED = _CLUSTER_METRICS + _QUALITY_METRICS

# Lower-is-better metrics
_LOWER_BETTER = {"DAV"}

_CLUSTER_HEADERS = {
    "ASW": "ASW",
    "DAV": r"DAV$\downarrow$",
    "CAL": "CAL",
}
_QUALITY_HEADERS = {
    "DRE_umap_overall_quality": r"\makecell{DRE\\overall}",
    "DREX_overall_quality": r"\makecell{DREX\\overall}",
}
_ALL_HEADERS = {**_CLUSTER_HEADERS, **_QUALITY_HEADERS}

_CONFIG_DISPLAY = {
    "VAE": "VAE",
    "VAE+ODE": "VAE+ODE",
    "VAE+MoCo": "VAE+MoCo",
    "VAE+MoCo+Proto": "VAE+MoCo+Proto",
    "VAE+ODE+MoCo": "VAE+ODE+MoCo",
    "Full": "Full (MoCoO)",
    "VAE+FM": "VAE+FM",
    "VAE+ODE+FM": "VAE+ODE+FM",
    "VAE+MoCo+FM": "VAE+MoCo+FM",
    "VAE+MoCo+Proto+FM": "VAE+MoCo+Proto+FM",
    "VAE+ODE+MoCo+FM": "VAE+ODE+MoCo+FM",
    "Full+FM": "Full+FM (MoCoO+FM)",
}

_DATASET_DISPLAY = {
    "IRALL": "IRALL", "astrocyte": "Astrocyte", "brainmet": "Brain Met.",
    "breast": "Breast", "dentate": "Dentate", "endo": "Endo",
    "gastric": "Gastric", "hemato": "Hemato", "hepatoblastoma": "Hepatoblastoma",
    "hESCtime": "hESC-time", "livercancer": "Liver Cancer",
    "lung": "Lung", "melanoma": "Melanoma", "paul": "Paul",
    "pituitary": "Pituitary", "retina": "Retina", "setty": "Setty",
    "spine": "Spine", "spinoids": "Spinoids", "teeth": "Teeth",
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


def _cross_dataset_mean(data: dict, configs: list[str]) -> dict[str, dict[str, float]]:
    """Compute mean metric values across datasets per config."""
    means = {}
    for cfg in configs:
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
    s = f"{val:.3f}"
    return rf"\textbf{{{s}}}" if is_best else s


def _find_best(rows: dict[str, dict[str, float]], metrics: list[str],
               configs: list[str]) -> dict[str, str]:
    """Return {metric: config_name} of the best config per metric."""
    best = {}
    for m in metrics:
        vals = {cfg: rows[cfg][m] for cfg in configs if cfg in rows and m in rows[cfg]}
        if not vals:
            continue
        if m in _LOWER_BETTER:
            best[m] = min(vals, key=vals.get)
        else:
            best[m] = max(vals, key=vals.get)
    return best


def _emit_table(rows: dict[str, dict[str, float]], metrics: list[str],
                headers: dict[str, str], caption: str, label: str,
                configs: list[str] | None = None, star: bool = False):
    """Print a LaTeX table to stdout."""
    if configs is None:
        configs = _BASE_CONFIGS
    env = "table*" if star else "table"
    col_spec = "l" + "c" * len(metrics)
    best = _find_best(rows, metrics, configs)

    print(f"\\begin{{{env}}}[!t]")
    print("    \\centering")
    print(f"    \\caption{{{caption}}}")
    print(f"    \\label{{{label}}}")
    if star:
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
    for cfg in configs:
        if cfg not in rows:
            continue
        row_str = f"        {_CONFIG_DISPLAY.get(cfg, cfg)}"
        for m in metrics:
            val = rows[cfg].get(m, float("nan"))
            is_best = best.get(m) == cfg
            row_str += f" & {_format_val(val, m, is_best)}"
        row_str += " \\\\"
        print(row_str)

    print("        \\bottomrule")
    print("    \\end{tabular}%")
    print("    }")
    print(f"\\end{{{env}}}")
    print()


def _emit_per_dataset(data: dict, ds: str, label_prefix: str):
    """Emit a proposed 8-metric table for one dataset (appendix)."""
    if ds not in data:
        return
    rows = data[ds]
    disp = _DATASET_DISPLAY.get(ds, ds)
    _emit_table(
        rows, _ALL_PROPOSED, _ALL_HEADERS,
        caption=(
            f"{disp} --- proposed five-metric evaluation suite (split=whole, "
            f"all 12 configurations). Best per column in \\textbf{{bold}}. "
            f"DAV$\\downarrow$ indicates lower is better."
        ),
        label=f"tab:{label_prefix}_{ds.lower()}",
        configs=_ALL_CONFIGS,
        star=True,
    )


def _emit_fm_delta_table(data: dict):
    """Print a table showing mean FM improvement (delta) per metric across datasets."""
    print("%" * 72)
    print("% FM improvement summary (delta = FM - base, averaged over datasets)")
    print("%" * 72)
    print()

    col_spec = "l" + "c" * len(_ALL_PROPOSED)

    print("\\begin{table}[!t]")
    print("    \\centering")
    print(
        f"    \\caption{{Mean Flow Matching improvement ($\\Delta$ = FM $-$ base) "
        f"across {len([d for d in _DATASETS if d in data])} datasets "
        f"(split=whole). Positive values indicate FM enhancement. "
        f"For DAV$\\downarrow$, negative $\\Delta$ is better.}}"
    )
    print("    \\label{tab:fm_delta}")
    print("    \\resizebox{\\columnwidth}{!}{%")
    print(f"    \\begin{{tabular}}{{{col_spec}}}")
    print("        \\toprule")

    # Header
    hdr = "        Base Config"
    for m in _ALL_PROPOSED:
        hdr += f" & {_ALL_HEADERS.get(m, m)}"
    hdr += " \\\\"
    print(hdr)
    print("        \\midrule")

    # Compute deltas per base config
    for base_cfg, fm_cfg in _FM_PAIRS.items():
        deltas: dict[str, list[float]] = {m: [] for m in _ALL_PROPOSED}
        for ds in _DATASETS:
            if ds not in data:
                continue
            if base_cfg not in data[ds] or fm_cfg not in data[ds]:
                continue
            for m in _ALL_PROPOSED:
                base_val = data[ds][base_cfg].get(m)
                fm_val = data[ds][fm_cfg].get(m)
                if base_val is not None and fm_val is not None:
                    deltas[m].append(fm_val - base_val)

        row_str = f"        {_CONFIG_DISPLAY.get(base_cfg, base_cfg)}"
        for m in _ALL_PROPOSED:
            if deltas[m]:
                mean_d = sum(deltas[m]) / len(deltas[m])
                sign = "+" if mean_d >= 0 else ""
                s = f"{sign}{mean_d:.3f}"
                # Bold if improvement direction is correct
                improved = (mean_d < 0) if m in _LOWER_BETTER else (mean_d > 0)
                if improved:
                    row_str += rf" & \textbf{{{s}}}"
                else:
                    row_str += f" & {s}"
            else:
                row_str += " & ---"
        row_str += " \\\\"
        print(row_str)

    print("        \\bottomrule")
    print("    \\end{tabular}%")
    print("    }")
    print("\\end{table}")
    print()


def _compute_fm_stats(data: dict) -> dict[str, dict[str, float]]:
    """Compute overall FM delta statistics for inline use in the paper."""
    stats = {}
    for m in _ALL_PROPOSED:
        all_deltas = []
        for base_cfg, fm_cfg in _FM_PAIRS.items():
            for ds in _DATASETS:
                if ds not in data:
                    continue
                if base_cfg not in data[ds] or fm_cfg not in data[ds]:
                    continue
                base_val = data[ds][base_cfg].get(m)
                fm_val = data[ds][fm_cfg].get(m)
                if base_val is not None and fm_val is not None:
                    all_deltas.append(fm_val - base_val)
        if all_deltas:
            mean_d = sum(all_deltas) / len(all_deltas)
            std_d = (sum((x - mean_d) ** 2 for x in all_deltas) / len(all_deltas)) ** 0.5
            n_improved = sum(
                1 for d in all_deltas
                if (d < 0 if m in _LOWER_BETTER else d > 0)
            )
            stats[m] = {
                "mean": mean_d,
                "std": std_d,
                "n_improved": n_improved,
                "n_total": len(all_deltas),
                "pct_improved": 100 * n_improved / len(all_deltas),
            }
    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate LaTeX tables")
    parser.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent.parent.parent.parent / "paper")
    args = parser.parse_args()

    results_dir = Path(__file__).resolve().parent.parent.parent / "results"
    data = _load_whole_rows(results_dir)
    n_datasets = len(data)

    base_means = _cross_dataset_mean(data, _BASE_CONFIGS)
    all_means = _cross_dataset_mean(data, _ALL_CONFIGS)

    # === File 1: FM tables (for main body) ===
    import io
    fm_buf = io.StringIO()
    _old_stdout = sys.stdout

    sys.stdout = fm_buf
    print("%" * 72)
    print(f"% AUTO-GENERATED FM TABLES --- do not edit by hand")
    print(f"% {n_datasets} datasets, {len(_ALL_CONFIGS)} configurations")
    print("%" * 72)
    print()

    # FM comparison — Clustering (all 12 configs)
    _emit_table(
        all_means, _CLUSTER_METRICS, _CLUSTER_HEADERS,
        caption=(
            f"Cross-dataset mean clustering metrics with FM enhancement "
            f"({n_datasets} datasets, split=whole, all 12 configurations). "
            f"Best per column in \\textbf{{bold}}. "
            f"DAV$\\downarrow$ indicates lower is better."
        ),
        label="tab:fm_clustering",
        configs=_ALL_CONFIGS,
        star=False,
    )

    # FM comparison — Quality (all 12 configs)
    _emit_table(
        all_means, _QUALITY_METRICS, _QUALITY_HEADERS,
        caption=(
            f"Cross-dataset mean embedding quality with FM enhancement "
            f"({n_datasets} datasets, split=whole, all 12 configurations). "
            f"Best per column in \\textbf{{bold}}."
        ),
        label="tab:fm_embedding",
        configs=_ALL_CONFIGS,
        star=False,
    )

    # FM delta summary table
    _emit_fm_delta_table(data)

    # FM stats for inline reporting
    fm_stats = _compute_fm_stats(data)
    print("%" * 72)
    print("% FM improvement summary statistics (for inline \\newcommand use)")
    print("%" * 72)
    _SHORT = {
        "ASW": "ASW", "DAV": "DAV", "CAL": "CAL",
        "DRE_umap_overall_quality": "DRE",
        "DREX_overall_quality": "DREX",
    }
    for m, s in fm_stats.items():
        short = _SHORT.get(m, m.split("_")[0])
        sign = "+" if s["mean"] >= 0 else ""
        print(
            f"% {short}: mean delta = {sign}{s['mean']:.4f} "
            f"(std {s['std']:.4f}), "
            f"improved in {s['n_improved']}/{s['n_total']} "
            f"({s['pct_improved']:.1f}%) dataset-config pairs"
        )
    print()

    sys.stdout = _old_stdout
    fm_path = args.outdir / "tables_fm.tex"
    fm_path.write_text(fm_buf.getvalue())
    print(f"Wrote {fm_path} ({fm_buf.getvalue().count(chr(10))} lines)", file=sys.stderr)

    # === File 2: Per-dataset appendix tables ===
    pd_buf = io.StringIO()
    sys.stdout = pd_buf
    print("%" * 72)
    print(f"% AUTO-GENERATED PER-DATASET TABLES --- do not edit by hand")
    print(f"% {n_datasets} datasets, {len(_ALL_CONFIGS)} configurations")
    print("%" * 72)
    print()
    for ds in _DATASETS:
        _emit_per_dataset(data, ds, "ablation")

    sys.stdout = _old_stdout
    pd_path = args.outdir / "tables_perdataset.tex"
    pd_path.write_text(pd_buf.getvalue())
    print(f"Wrote {pd_path} ({pd_buf.getvalue().count(chr(10))} lines)", file=sys.stderr)


if __name__ == "__main__":
    main()
