"""
Clean Training Runner — Single Dataset, All 6 Configs.

Trains all 6 MoCoO ablation configurations on one dataset with:
  - latent_dim=10, epochs=400, patience=60
  - KMeans pseudo-labels (n_clusters = latent_dim) for all datasets
  - 4-split metric evaluation (train, val, test, whole)
  - Intermediate data saved for regeneration
  - Config snapshot saved for reproducibility

Output structure:
  benchmarks/results/<dataset>/
    config_snapshot.json      — full hyperparameters for reproducibility
    <ConfigName>/
      latents.npz             — latent embeddings for all 4 splits
      metrics.json            — all metrics for all 4 splits
      training_history.json   — loss / score curves
    summary.csv               — one row per config, key metrics
    summary_expanded.csv      — full metric battery per config

Usage:
    python benchmarks/scripts/pipeline/run_training.py --dataset IRALL
    python benchmarks/scripts/pipeline/run_training.py --dataset paul
    python benchmarks/scripts/pipeline/run_training.py --dataset all
"""

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


# ═════════════════════════════════════════════════════════════════════════════
# Dataset loading
# ═════════════════════════════════════════════════════════════════════════════

def load_dataset(path: str, max_cells: int, hvg: int, seed: int = 42):
    """Load, subsample, HVG-filter, ensure counts layer."""
    adata = sc.read_h5ad(path)
    print(f"  Loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")

    if adata.n_obs > max_cells:
        sc.pp.subsample(adata, n_obs=max_cells, random_state=seed)
        print(f"  Subsampled -> {adata.n_obs} cells")

    sc.pp.filter_genes(adata, min_cells=10)

    if "counts" not in adata.layers:
        from scipy.sparse import issparse
        X = adata.X.toarray() if issparse(adata.X) else adata.X.copy()
        adata.layers["counts"] = X

    if adata.n_vars > hvg:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        try:
            sc.pp.highly_variable_genes(
                adata, n_top_genes=hvg, flavor="seurat_v3", layer="counts"
            )
        except (ImportError, Exception):
            sc.pp.highly_variable_genes(adata, n_top_genes=hvg)
        adata = adata[:, adata.var["highly_variable"]].copy()
        print(f"  HVG filtered -> {adata.n_vars} genes")

    return adata


# ═════════════════════════════════════════════════════════════════════════════
# Single config training
# ═════════════════════════════════════════════════════════════════════════════

def run_single(name, adata, config, shared, epochs, patience, val_every):
    """Train one configuration, return model + metadata."""
    params = {**shared, **config}
    print(f"\n  ── {name} ──")

    model = MoCoO(adata, **params)
    model.fit(epochs=epochs, patience=patience, val_every=val_every)

    res = model.get_resource_metrics()
    print(f"    Epochs: {int(res['actual_epochs'])}, Time: {res['train_time']:.1f}s")

    return model, res


# ═════════════════════════════════════════════════════════════════════════════
# 4-split metric computation
# ═════════════════════════════════════════════════════════════════════════════

def compute_4split_metrics(model):
    """Compute metrics on train, val, test, and whole splits.

    Returns
    -------
    split_metrics : dict[str, dict]
        Keys: 'train', 'val', 'test', 'whole'. Each value is a metric dict.
    split_latents : dict[str, np.ndarray]
        Latent arrays for each split.
    split_labels : dict[str, np.ndarray]
        Label arrays for each split.
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
        latent = model.take_latent(X_split)
        split_latents[split_name] = latent
        split_labels[split_name] = labels_split

        print(f"    Computing metrics on {split_name} ({len(labels_split)} cells)...")
        metrics = compute_all_metrics(latent, labels_split)

        # Strip private keys (numpy arrays)
        clean_metrics = {
            k: (round(float(v), 6) if isinstance(v, (float, np.floating)) else v)
            for k, v in metrics.items()
            if not k.startswith("_") and not isinstance(v, np.ndarray)
        }
        split_metrics[split_name] = clean_metrics

    return split_metrics, split_latents, split_labels


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

DATASET_SPECS = {
    "IRALL": {
        "path": "LAB/scRL/IRALL.h5ad",
        "max_cells": 3000,
        "hvg": 3000,
    },
    "dentate": {
        "path": "vGAE_LAB/data/dentate.h5ad",
        "max_cells": 3000,
        "hvg": 3000,
    },
    "endo": {
        "path": "vGAE_LAB/data/endo.h5ad",
        "max_cells": 2500,
        "hvg": 3000,
    },
    "paul": {
        "path": "LAB/data/paul.h5ad",
        "max_cells": 2700,
        "hvg": 3000,
    },
    "spinoids": {
        "path": "LAB/data/spinoids.h5ad",
        "max_cells": 3000,
        "hvg": 3000,
    },
}


def run_dataset(dataset_name, data_dir, outdir_base, shared, configs, epochs, patience, val_every):
    """Run all 6 configs on one dataset."""
    spec = DATASET_SPECS[dataset_name]
    data_path = os.path.join(data_dir, spec["path"])

    print(f"\n{'=' * 70}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Path: {data_path}")
    print(f"{'=' * 70}")

    adata = load_dataset(data_path, spec["max_cells"], spec["hvg"])

    outdir = Path(outdir_base) / dataset_name
    outdir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot
    snapshot = {
        "dataset": dataset_name,
        "data_path": data_path,
        "n_cells": adata.n_obs,
        "n_genes": adata.n_vars,
        "shared_params": shared,
        "epochs": epochs,
        "patience": patience,
        "val_every": val_every,
        "configs": {name: cfg for name, cfg in configs.items()},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(outdir / "config_snapshot.json", "w") as f:
        json.dump(snapshot, f, indent=2, default=str)

    summary_rows = []

    for name, cfg in configs.items():
        model, res = run_single(name, adata, cfg, shared, epochs, patience, val_every)

        # 4-split metrics
        split_metrics, split_latents, split_labels = compute_4split_metrics(model)

        # Per-config output directory
        safe_name = name.replace("+", "_")
        config_dir = outdir / safe_name
        config_dir.mkdir(parents=True, exist_ok=True)

        # Save latents for all 4 splits
        latent_save = {
            f"{s}_latent": split_latents[s] for s in split_latents
        }
        latent_save.update({
            f"{s}_labels": split_labels[s] for s in split_labels
        })
        latent_save["train_idx"] = model.train_idx
        latent_save["val_idx"] = model.val_idx
        latent_save["test_idx"] = model.test_idx
        np.savez(config_dir / "latents.npz", **latent_save)

        # Save ODE gradients if available
        if cfg.get("use_ode", False):
            try:
                gradients = model.get_velocity()
                np.save(config_dir / "gradients.npy", gradients)
            except Exception as e:
                print(f"    ODE gradients skipped: {e}")

        # Save metrics
        metrics_out = {
            "config": name,
            "resource": {
                "actual_epochs": int(res["actual_epochs"]),
                "train_time_s": round(float(res["train_time"]), 1),
                "peak_mem_gb": round(float(res["peak_memory_gb"]), 2),
                "best_val_loss": round(float(model.best_val_loss), 2),
            },
            "splits": split_metrics,
        }
        with open(config_dir / "metrics.json", "w") as f:
            json.dump(metrics_out, f, indent=2)

        # Save training history
        history = {
            "train_losses": [float(x) for x in model.train_losses],
            "val_losses": [float(x) for x in model.val_losses],
        }
        with open(config_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        # Build summary row (whole-split metrics + resource)
        whole = split_metrics["whole"]
        test = split_metrics["test"]
        row = {
            "config": name,
            "whole_ARI": whole.get("ARI", np.nan),
            "whole_NMI": whole.get("NMI", np.nan),
            "whole_ASW": whole.get("ASW", np.nan),
            "whole_DAV": whole.get("DAV", np.nan),
            "whole_CAL": whole.get("CAL", np.nan),
            "whole_COR": whole.get("COR", np.nan),
            "test_ARI": test.get("ARI", np.nan),
            "test_NMI": test.get("NMI", np.nan),
            "test_ASW": test.get("ASW", np.nan),
            "whole_DRE_umap_overall": whole.get("DRE_umap_overall_quality", np.nan),
            "whole_LSE_overall": whole.get("LSE_overall_quality", np.nan),
            "whole_DREX_overall": whole.get("DREX_overall_quality", np.nan),
            "whole_LSEX_overall": whole.get("LSEX_overall_quality", np.nan),
            "actual_epochs": int(res["actual_epochs"]),
            "train_time_s": round(float(res["train_time"]), 1),
            "peak_mem_gb": round(float(res["peak_memory_gb"]), 2),
            "best_val_loss": round(float(model.best_val_loss), 2),
        }
        summary_rows.append(row)

        print(f"    ✓ {name}: ARI={whole.get('ARI', 0):.3f} NMI={whole.get('NMI', 0):.3f}")

    # Write summary CSV
    if summary_rows:
        fields = list(summary_rows[0].keys())
        with open(outdir / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(summary_rows)

        # Write expanded CSV (all metrics, all splits)
        expanded_rows = []
        for name in configs:
            safe_name = name.replace("+", "_")
            mpath = outdir / safe_name / "metrics.json"
            if mpath.exists():
                with open(mpath) as f:
                    mdata = json.load(f)
                for split_name, split_m in mdata.get("splits", {}).items():
                    erow = {"config": name, "split": split_name}
                    erow.update(split_m)
                    expanded_rows.append(erow)

        if expanded_rows:
            efields = list(expanded_rows[0].keys())
            # Ensure all fields are captured
            for r in expanded_rows:
                for k in r:
                    if k not in efields:
                        efields.append(k)
            with open(outdir / "summary_expanded.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=efields, extrasaction="ignore")
                w.writeheader()
                w.writerows(expanded_rows)

    # Print summary table
    print(f"\n{'=' * 90}")
    print(f"  Results for {dataset_name}")
    print(f"{'=' * 90}")
    header = f"{'Config':<20} {'ARI':>6} {'NMI':>6} {'ASW':>6} {'DRE':>6} {'LSE':>6} {'DREX':>6} {'LSEX':>6} | {'Ep':>4} {'Time':>6}"
    print(header)
    print(f"{'─' * 90}")
    for r in summary_rows:
        print(
            f"{r['config']:<20} {r['whole_ARI']:>6.3f} {r['whole_NMI']:>6.3f} "
            f"{r['whole_ASW']:>6.3f} {r['whole_DRE_umap_overall']:>6.3f} "
            f"{r['whole_LSE_overall']:>6.3f} {r['whole_DREX_overall']:>6.3f} "
            f"{r['whole_LSEX_overall']:>6.3f} | {r['actual_epochs']:>4d} {r['train_time_s']:>6.1f}"
        )
    print(f"{'=' * 90}")
    print(f"  Saved to: {outdir}")


def main():
    parser = argparse.ArgumentParser(
        description="Clean Training Runner — all 6 configs on one or all datasets"
    )
    parser.add_argument(
        "--dataset", default="IRALL",
        help="Dataset name (IRALL, dentate, endo, paul, spinoids) or 'all'",
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("MOCOO_DATA_DIR", os.path.expanduser("~")),
        help="Base data directory (default: MOCOO_DATA_DIR or ~)",
    )
    parser.add_argument(
        "--outdir", default=None,
        help="Output base directory (default: benchmarks/results)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override max training epochs",
    )
    parser.add_argument(
        "--patience", type=int, default=None,
        help="Override early stopping patience",
    )
    parser.add_argument(
        "--val-every", type=int, default=5,
        help="Validate every N epochs",
    )
    parser.add_argument(
        "--configs", nargs="*", default=None,
        help="Subset of configs to run (default: all 6)",
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config("default")
    training = get_training_params(cfg)
    SHARED = get_shared_params(cfg)
    CONFIGS = get_model_configs(cfg)

    epochs = args.epochs or training["epochs"]
    patience = args.patience or training["patience"]

    if args.configs:
        CONFIGS = {k: v for k, v in CONFIGS.items() if k in args.configs}

    outdir_base = args.outdir or str(
        Path(__file__).resolve().parent.parent.parent / "results"
    )

    datasets = list(DATASET_SPECS.keys()) if args.dataset == "all" else [args.dataset]

    print(f"\n{'#' * 70}")
    print(f"  MoCoO Training Runner")
    print(f"  Datasets: {datasets}")
    print(f"  Configs: {list(CONFIGS.keys())}")
    print(f"  latent_dim={SHARED['latent_dim']}, epochs={epochs}, patience={patience}")
    print(f"  Labels: KMeans (n_clusters={SHARED['latent_dim']})")
    print(f"  Metrics: 4-split (train, val, test, whole)")
    print(f"{'#' * 70}")

    for ds in datasets:
        if ds not in DATASET_SPECS:
            print(f"  ⚠ Unknown dataset '{ds}', skipping. Available: {list(DATASET_SPECS.keys())}")
            continue
        run_dataset(ds, args.data_dir, outdir_base, SHARED, CONFIGS, epochs, patience, args.val_every)

    print(f"\n✓ All training complete. Results: {outdir_base}")


if __name__ == "__main__":
    main()
