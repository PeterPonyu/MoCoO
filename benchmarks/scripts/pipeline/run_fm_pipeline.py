#!/usr/bin/env python
"""
FM Pipeline — Train all 6 MoCoO configs + Phase-2 Flow Matching per dataset.

For each dataset, for each of the 6 model configurations:
  1. Train the model (Phase 1)
  2. Train a Flow Matching velocity network on frozen VAE posteriors (Phase 2)
  3. Compute FM-refined latents via the learned velocity field
  4. Evaluate all metrics on the FM-refined latents (4-split)
  5. Save latents.npz, metrics.json, training_history.json
  6. Append "<config>+FM" rows to summary_expanded.csv

The pipeline reuses the exact same dataset loading, preprocessing, and
metric evaluation as fig1_training_pipeline.py to ensure comparability.

Usage:
    python benchmarks/scripts/pipeline/run_fm_pipeline.py --dataset IRALL
    python benchmarks/scripts/pipeline/run_fm_pipeline.py --dataset all
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import scanpy as sc
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

# Reuse dataset loader from fig1 pipeline
from benchmarks.scripts.pipeline.fig1_training_pipeline import (
    DATASET_SPECS,
    load_dataset,
)


# ═════════════════════════════════════════════════════════════════════════════
# FM-specific metric computation
# ═════════════════════════════════════════════════════════════════════════════

def compute_fm_4split_metrics(model, t_start: float, steps: int):
    """Compute metrics on FM-refined latents for all 4 splits.

    Returns
    -------
    split_metrics : dict[str, dict]
    split_latents : dict[str, np.ndarray]
    split_labels  : dict[str, np.ndarray]
    """
    splits = {
        "train": (model.X_train, model.labels_train),
        "val": (model.X_val, model.labels_val),
        "test": (model.X_test, model.labels_test),
        "whole": (model.X, model.labels),
    }

    split_metrics = {}
    split_latents = {}
    split_labels = {}

    for split_name, (X_split, labels_split) in splits.items():
        latent = model.take_fm_refined(X_split, t_start=t_start, steps=steps)
        split_latents[split_name] = latent
        split_labels[split_name] = labels_split

        print(f"    Computing FM metrics on {split_name} ({len(labels_split)} cells)...")
        metrics = compute_all_metrics(latent, labels_split)

        clean_metrics = {
            k: (round(float(v), 6) if isinstance(v, (float, np.floating)) else v)
            for k, v in metrics.items()
            if not k.startswith("_") and not isinstance(v, np.ndarray)
        }
        split_metrics[split_name] = clean_metrics

    return split_metrics, split_latents, split_labels


# ═════════════════════════════════════════════════════════════════════════════
# CSV append helper
# ═════════════════════════════════════════════════════════════════════════════

def _append_fm_to_expanded_csv(outdir: Path, fm_metrics: dict, config_name: str):
    """Append <config>+FM rows to the dataset's summary_expanded.csv."""
    csv_path = outdir / "summary_expanded.csv"
    if not csv_path.exists():
        print(f"    \u26a0 {csv_path} not found \u2014 creating new file")
        existing_rows = []
        fieldnames = None
    else:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            existing_rows = [
                r for r in reader if r.get("config") != config_name
            ]

    # Build new rows
    new_rows = []
    for split_name, split_m in fm_metrics.items():
        row = {"config": config_name, "split": split_name}
        row.update(split_m)
        new_rows.append(row)

    all_rows = existing_rows + new_rows

    # Determine fieldnames from existing + new
    if fieldnames is None:
        fieldnames = list(new_rows[0].keys())
    for r in new_rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    print(f"    ✓ Appended {config_name} rows to {csv_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Main per-dataset runner
# ═════════════════════════════════════════════════════════════════════════════

def run_fm_dataset(
    dataset_name: str,
    data_dir: str,
    outdir_base: str,
    shared: dict,
    configs: dict,
    phase1_epochs: int,
    phase1_patience: int,
    val_every: int,
    fm_epochs: int,
    fm_lr: float,
    fm_hidden_dim: int,
    fm_t_start: float,
    fm_steps: int,
):
    spec = DATASET_SPECS[dataset_name]
    data_path = os.path.join(data_dir, spec["path"])

    print(f"\n{'=' * 70}")
    print(f"  FM Pipeline: {dataset_name}")
    print(f"  Path: {data_path}")
    print(f"  Configs: {list(configs.keys())}")
    print(f"{'=' * 70}")

    adata = load_dataset(data_path, spec["max_cells"], spec["hvg"])

    outdir = Path(outdir_base) / dataset_name
    outdir.mkdir(parents=True, exist_ok=True)

    for config_name, config_params in configs.items():
        fm_config_name = f"{config_name}+FM"
        safe_name = config_name.replace("+", "_") + "_FM"

        params = {**shared, **config_params}
        print(f"\n  ── Phase 1: Training {config_name} ──")
        model = MoCoO(adata, **params)
        model.fit(epochs=phase1_epochs, patience=phase1_patience, val_every=val_every)

        res = model.get_resource_metrics()
        print(f"    Epochs: {int(res['actual_epochs'])}, Time: {res['train_time']:.1f}s")

        # ── Phase 2: Train Flow Matching ──
        print(f"\n  ── Phase 2: Training Flow Matching for {config_name} ({fm_epochs} epochs) ──")
        t0 = time.time()
        model.train_fm(
            epochs=fm_epochs,
            lr=fm_lr,
            hidden_dim=fm_hidden_dim,
        )
        fm_time = time.time() - t0
        print(f"    FM training time: {fm_time:.1f}s")

        # ── Compute FM-refined metrics ──
        print(f"\n  ── Computing FM-refined metrics for {fm_config_name} ──")
        fm_split_metrics, fm_split_latents, fm_split_labels = compute_fm_4split_metrics(
            model, t_start=fm_t_start, steps=fm_steps
        )

        # ── Save outputs ──
        config_dir = outdir / safe_name
        config_dir.mkdir(parents=True, exist_ok=True)

        # Latents
        latent_save = {f"{s}_latent": fm_split_latents[s] for s in fm_split_latents}
        latent_save.update({f"{s}_labels": fm_split_labels[s] for s in fm_split_labels})
        latent_save["train_idx"] = model.train_idx
        latent_save["val_idx"] = model.val_idx
        latent_save["test_idx"] = model.test_idx
        np.savez(config_dir / "latents.npz", **latent_save)

        # Metrics JSON
        metrics_out = {
            "config": fm_config_name,
            "resource": {
                "phase1_epochs": int(res["actual_epochs"]),
                "phase1_time_s": round(float(res["train_time"]), 1),
                "fm_epochs": fm_epochs,
                "fm_time_s": round(fm_time, 1),
                "total_time_s": round(float(res["train_time"]) + fm_time, 1),
            },
            "fm_params": {
                "fm_lr": fm_lr,
                "fm_hidden_dim": fm_hidden_dim,
                "fm_t_start": fm_t_start,
                "fm_steps": fm_steps,
            },
            "splits": fm_split_metrics,
        }
        with open(config_dir / "metrics.json", "w") as f:
            json.dump(metrics_out, f, indent=2)

        # FM loss history
        fm_losses = model.get_fm_loss_history()
        history = {"fm_losses": [float(x) for x in fm_losses]}
        with open(config_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        # ODE gradients (only if model uses ODE)
        if config_params.get("use_ode", False):
            try:
                gradients = model.get_velocity()
                np.save(config_dir / "gradients.npy", gradients)
            except Exception as e:
                print(f"    ODE gradients skipped: {e}")

        # Append to summary_expanded.csv
        _append_fm_to_expanded_csv(outdir, fm_split_metrics, fm_config_name)

        # Print summary
        whole = fm_split_metrics["whole"]
        print(f"\n  {fm_config_name}: ARI={whole.get('ARI', 0):.3f} "
              f"NMI={whole.get('NMI', 0):.3f} "
              f"ASW={whole.get('ASW', 0):.3f} "
              f"DRE_umap={whole.get('DRE_umap_overall_quality', 0):.3f} "
              f"LSE={whole.get('LSE_overall_quality', 0):.3f}")
        print(f"  Phase 1: {int(res['actual_epochs'])} epochs, {res['train_time']:.1f}s")
        print(f"  Phase 2: {fm_epochs} epochs, {fm_time:.1f}s")
        print(f"  Saved to: {config_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="FM Pipeline — All configs + Flow Matching refinement"
    )
    parser.add_argument(
        "--dataset", default="IRALL",
        help="Dataset name or 'all'",
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("MOCOO_DATA_DIR", os.path.expanduser("~")),
    )
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Phase 1 epochs")
    parser.add_argument("--patience", type=int, default=None, help="Phase 1 patience")
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--fm-epochs", type=int, default=200)
    parser.add_argument("--fm-lr", type=float, default=1e-3)
    parser.add_argument("--fm-hidden-dim", type=int, default=128)
    parser.add_argument("--fm-t-start", type=float, default=0.9)
    parser.add_argument("--fm-steps", type=int, default=100)
    args = parser.parse_args()

    # Load config
    cfg = load_config("default")
    training = get_training_params(cfg)
    shared = get_shared_params(cfg)
    configs = get_model_configs(cfg)

    epochs = args.epochs or training["epochs"]
    patience = args.patience or training["patience"]

    outdir_base = args.outdir or str(
        Path(__file__).resolve().parent.parent.parent / "results"
    )

    datasets = list(DATASET_SPECS.keys()) if args.dataset == "all" else [args.dataset]

    print(f"\n{'#' * 70}")
    print(f"  MoCoO FM Pipeline — All Configs")
    print(f"  Datasets: {datasets}")
    print(f"  Configs: {list(configs.keys())}")
    print(f"  Phase 1: epochs={epochs}, patience={patience}")
    print(f"  Phase 2: FM epochs={args.fm_epochs}, lr={args.fm_lr}, "
          f"hidden_dim={args.fm_hidden_dim}")
    print(f"  FM inference: t_start={args.fm_t_start}, steps={args.fm_steps}")
    print(f"{'#' * 70}")

    for ds in datasets:
        if ds not in DATASET_SPECS:
            print(f"  ⚠ Unknown dataset '{ds}', skipping.")
            continue
        run_fm_dataset(
            ds, args.data_dir, outdir_base, shared, configs,
            epochs, patience, args.val_every,
            args.fm_epochs, args.fm_lr, args.fm_hidden_dim,
            args.fm_t_start, args.fm_steps,
        )

    print(f"\n✓ FM pipeline complete. Results: {outdir_base}")


if __name__ == "__main__":
    main()
