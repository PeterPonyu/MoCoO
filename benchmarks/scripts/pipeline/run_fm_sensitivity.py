#!/usr/bin/env python
"""
FM Sensitivity Analysis — Sweep each FM hyperparameter independently.

Trains the best model (VAE+ODE) once per dataset, then re-runs Phase-2
Flow Matching with varied parameter settings, evaluating on whole-split
metrics (ARI, NMI, ASW).

Output:
    benchmarks/results/fm_sensitivity/sensitivity.csv
    Columns: dataset, param, value, ARI, NMI, ASW

Usage:
    python benchmarks/scripts/pipeline/run_fm_sensitivity.py --dataset all
    python benchmarks/scripts/pipeline/run_fm_sensitivity.py --dataset IRALL
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from mocoo import MoCoO
from mocoo.configs import (
    load_config,
    get_shared_params,
    get_model_configs,
    get_training_params,
)
from mocoo.evaluation import compute_all_metrics
from benchmarks.scripts.pipeline.fig1_training_pipeline import (
    DATASET_SPECS,
    load_dataset,
)

# ═══════════════════════════════════════════════════════════════════════════
# Default FM parameters (held constant when not being swept)
# ═══════════════════════════════════════════════════════════════════════════
_FM_DEFAULTS = {
    "fm_epochs": 200,
    "fm_lr": 1e-3,
    "fm_hidden_dim": 128,
    "fm_steps": 100,
    "fm_t_start": 0.9,
}

# ═══════════════════════════════════════════════════════════════════════════
# Sweep grid: each parameter → list of values to try
# ═══════════════════════════════════════════════════════════════════════════
_SWEEP_GRID = {
    "fm_t_start": [0.1, 0.3, 0.5, 0.7, 0.9, 0.99],
    "fm_epochs": [25, 50, 100, 200, 400],
    "fm_lr": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    "fm_hidden_dim": [32, 64, 128, 256, 512],
    "fm_steps": [5, 10, 25, 50, 100, 200],
}

# 10 core benchmark datasets
_CORE_DATASETS = [
    "endo", "setty", "paul", "IRALL", "dentate",
    "spinoids", "lung", "retina", "teeth", "hepatoblastoma",
]


_METRIC_KEYS = [
    "ARI", "NMI", "ASW",
    "LSE_overall_quality", "DRE_umap_overall_quality",
    "DREX_overall_quality", "LSEX_overall_quality",
]


def _evaluate_fm(model, t_start: float, steps: int) -> dict:
    """Compute whole-split metrics on FM-refined latents."""
    latent = model.take_fm_refined(model.X, t_start=t_start, steps=steps)
    metrics = compute_all_metrics(latent, model.labels)
    return {mk: round(float(metrics.get(mk, 0)), 6) for mk in _METRIC_KEYS}


def run_sensitivity_dataset(
    dataset_name: str,
    data_dir: str,
    vae_ode_params: dict,
    shared: dict,
    phase1_epochs: int,
    phase1_patience: int,
    val_every: int,
) -> list[dict]:
    """Run full sensitivity sweep for one dataset.

    Returns list of dicts with columns: dataset, param, value, ARI, NMI, ASW
    """
    spec = DATASET_SPECS[dataset_name]
    data_path = os.path.join(data_dir, spec["path"])

    print(f"\n{'=' * 70}")
    print(f"  Sensitivity: {dataset_name}")
    print(f"{'=' * 70}")

    adata = load_dataset(data_path, spec["max_cells"], spec["hvg"])

    # ── Phase 1: Train VAE+ODE once ──
    params = {**shared, **vae_ode_params}
    print(f"\n  Phase 1: Training VAE+ODE ...")
    model = MoCoO(adata, **params)
    model.fit(epochs=phase1_epochs, patience=phase1_patience, val_every=val_every)

    res = model.get_resource_metrics()
    print(f"    Epochs: {int(res['actual_epochs'])}, Time: {res['train_time']:.1f}s")

    rows = []

    # ── Sweep each FM parameter ──
    for param_name, values in _SWEEP_GRID.items():
        print(f"\n  Sweeping {param_name}: {values}")

        for val in values:
            # Build FM kwargs: defaults + override the swept param
            fm_kw = dict(_FM_DEFAULTS)
            fm_kw[param_name] = val

            # Phase 2: Train FM
            t0 = time.time()
            model.train_fm(
                epochs=fm_kw["fm_epochs"],
                lr=fm_kw["fm_lr"],
                hidden_dim=fm_kw["fm_hidden_dim"],
            )
            fm_time = time.time() - t0

            # Evaluate
            metrics = _evaluate_fm(
                model,
                t_start=fm_kw["fm_t_start"],
                steps=fm_kw["fm_steps"],
            )

            row = {
                "dataset": dataset_name,
                "param": param_name,
                "value": val,
                **metrics,
            }
            rows.append(row)

            print(f"    {param_name}={val:>8} → ARI={metrics['ARI']:.4f}  "
                  f"NMI={metrics['NMI']:.4f}  ASW={metrics['ASW']:.4f}  "
                  f"LSE={metrics['LSE_overall_quality']:.4f}  "
                  f"DRE={metrics['DRE_umap_overall_quality']:.4f}  "
                  f"({fm_time:.1f}s)")

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="FM sensitivity analysis — sweep FM hyperparameters"
    )
    parser.add_argument(
        "--dataset", default="all",
        help="Dataset name or 'all' (default: all 10 core datasets)",
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("MOCOO_DATA_DIR", os.path.expanduser("~")),
    )
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--val-every", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config("default")
    training = get_training_params(cfg)
    shared = get_shared_params(cfg)
    configs = get_model_configs(cfg)

    epochs = args.epochs or training["epochs"]
    patience = args.patience or training["patience"]

    # VAE+ODE is the best-performing base model
    vae_ode_params = configs["VAE+ODE"]

    outdir = Path(args.outdir) if args.outdir else (
        Path(__file__).resolve().parent.parent.parent / "results" / "fm_sensitivity"
    )
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "sensitivity.csv"

    datasets = _CORE_DATASETS if args.dataset == "all" else [args.dataset]

    print(f"\n{'#' * 70}")
    print(f"  FM Sensitivity Analysis")
    print(f"  Base model: VAE+ODE")
    print(f"  Datasets: {datasets}")
    print(f"  Sweep grid: {sum(len(v) for v in _SWEEP_GRID.values())} settings")
    print(f"  Output: {csv_path}")
    print(f"{'#' * 70}")

    all_rows = []

    # Load any existing results to support incremental runs
    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            existing = list(reader)
        done_datasets = {r["dataset"] for r in existing}
        all_rows.extend(existing)
        print(f"\n  Loaded {len(existing)} existing rows "
              f"({len(done_datasets)} datasets done)")
    else:
        done_datasets = set()

    for ds in datasets:
        if ds not in DATASET_SPECS:
            print(f"  ⚠ Unknown dataset '{ds}', skipping.")
            continue
        if ds in done_datasets:
            print(f"\n  ✓ {ds} already done — skipping")
            continue

        rows = run_sensitivity_dataset(
            ds, args.data_dir, vae_ode_params, shared,
            epochs, patience, args.val_every,
        )
        all_rows.extend(rows)

        # Write after each dataset (incremental save)
        fieldnames = ["dataset", "param", "value"] + _METRIC_KEYS
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_rows)

        print(f"\n  Saved {len(all_rows)} total rows to {csv_path}")

    print(f"\n✓ Sensitivity analysis complete. Results: {csv_path}")


if __name__ == "__main__":
    main()
