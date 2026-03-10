#!/usr/bin/env python3
"""
Metric sensitivity analysis for DRE/DREX/LSE/LSEX.

Tests metric stability across subsample sizes, latent dimensionalities,
and random seeds. Addresses reviewer Concern 5 about custom metric validation.

Usage:
    python benchmarks/scripts/evaluation/metric_sensitivity.py \
        --resultsdir benchmarks/results/single_dataset
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from mocoo.evaluation import (
    compute_dre_metrics,
    compute_lse_metrics,
    compute_drex_metrics,
    compute_lsex_metrics,
)
from mocoo.evaluation._projections import compute_2d_projections
from benchmarks.scripts.plotting.shared import load_benchmark_npz


def _run_sensitivity(latent, labels, subsample_sizes, seeds):
    """Run metric computation across subsample sizes and seeds."""
    rows = []
    for n_sub in subsample_sizes:
        for seed in seeds:
            rng = np.random.RandomState(seed)
            if n_sub >= len(latent):
                idx = np.arange(len(latent))
            else:
                idx = rng.choice(len(latent), n_sub, replace=False)

            lat_sub = latent[idx]
            lab_sub = labels[idx]

            umap_2d, tsne_2d = compute_2d_projections(lat_sub)
            row = {"n_cells": n_sub, "seed": seed}

            # DRE
            if umap_2d is not None:
                row.update(compute_dre_metrics(lat_sub, umap_2d, 15, "DRE_umap"))
            if tsne_2d is not None:
                row.update(compute_dre_metrics(lat_sub, tsne_2d, 15, "DRE_tsne"))

            # LSE
            row.update(compute_lse_metrics(lat_sub))

            # DREX
            if umap_2d is not None:
                row.update(compute_drex_metrics(lat_sub, umap_2d, 15))

            # LSEX
            row.update(compute_lsex_metrics(lat_sub, lab_sub, 15))

            rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Metric sensitivity analysis")
    parser.add_argument("--resultsdir", type=str, required=True)
    parser.add_argument("--config", type=str, default="Full",
                        help="Which config to test (default: Full)")
    parser.add_argument("--subsample_sizes", type=int, nargs="+",
                        default=[500, 1000, 2000, 3000])
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 123, 456, 789, 1024])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    rdir = Path(args.resultsdir)
    data = load_benchmark_npz(rdir)
    configs = data["configs"]

    if args.config not in configs:
        print(f"Config '{args.config}' not found. Available: {configs}")
        sys.exit(1)

    ci = configs.index(args.config)
    latent = data["latents"][ci]
    labels = data["labels"][ci]

    print(f"Running metric sensitivity for config={args.config} "
          f"(n={len(latent)}, d={latent.shape[1]})")
    print(f"  Subsample sizes: {args.subsample_sizes}")
    print(f"  Seeds: {args.seeds}")

    df = _run_sensitivity(latent, labels, args.subsample_sizes, args.seeds)

    # Summary: mean and std per subsample size
    metric_cols = [c for c in df.columns if c not in ("n_cells", "seed")]
    summary = df.groupby("n_cells")[metric_cols].agg(["mean", "std"])

    print("\n" + "=" * 70)
    print("METRIC SENSITIVITY ANALYSIS")
    print("=" * 70)

    for col in metric_cols:
        vals = summary[col]
        cv_values = (vals["std"] / (vals["mean"].abs() + 1e-10)).values
        mean_cv = np.mean(cv_values)
        print(f"  {col:40s}  mean_CV={mean_cv:.4f}  "
              f"{'STABLE' if mean_cv < 0.1 else 'MODERATE' if mean_cv < 0.2 else 'VARIABLE'}")

    out_path = Path(args.output) if args.output else rdir / "metric_sensitivity.csv"
    df.to_csv(out_path, index=False)
    summary_path = out_path.parent / "metric_sensitivity_summary.csv"
    summary.to_csv(summary_path)
    print(f"\nSaved: {out_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
