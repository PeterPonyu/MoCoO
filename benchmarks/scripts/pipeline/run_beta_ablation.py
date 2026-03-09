"""
Series 1: Beta Ablation Study (Paper Tables I-V).

Runs ALL 6 configurations across beta={0.01, 0.1, 1.0} with proper
training settings (200 epochs, patience 40, 3000 cells) to allow
complex models (Full MoCoO) to converge.

Outputs per-beta subdirectories with per-config JSON, summary.csv,
benchmark_data.npz, and summary_expanded.csv.

Usage:
    python benchmarks/scripts/pipeline/run_beta_ablation.py
    python benchmarks/scripts/pipeline/run_beta_ablation.py --betas 0.01 0.1 1.0
    python benchmarks/scripts/pipeline/run_beta_ablation.py --epochs 300 --patience 50
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
    get_sweep_params,
)
from mocoo.evaluation import compute_clustering_metrics, compute_all_metrics


def load_dataset(path: str, max_cells: int, hvg: int, seed: int = 42):
    """Load, subsample, HVG-filter, ensure counts layer."""
    adata = sc.read_h5ad(path)
    print(f"Loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")

    if adata.n_obs > max_cells:
        sc.pp.subsample(adata, n_obs=max_cells, random_state=seed)
        print(f"Subsampled -> {adata.n_obs} cells")

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
        print(f"HVG filtered -> {adata.n_vars} genes")

    return adata


def run_single(name, adata, config, shared, epochs, patience, val_every):
    """Train one configuration, return metrics dict."""
    params = {**shared, **config}
    print(f"\n  -- {name} --")

    model = MoCoO(adata, **params)
    model.fit(epochs=epochs, patience=patience, val_every=val_every)

    # Collect latent embeddings
    latent = model.get_latent()
    test_latent = model.get_test_latent()
    labels_all = model.labels

    # Full-data clustering metrics via package API
    full_metrics = compute_clustering_metrics(latent, labels_all)

    # Test-set clustering metrics
    labels_test = labels_all[model.test_idx]
    test_metrics = compute_clustering_metrics(test_latent, labels_test)

    # Resource metrics
    res = model.get_resource_metrics()

    result = {
        "config": name,
        "full_ARI": round(float(full_metrics["ARI"]), 4),
        "full_NMI": round(float(full_metrics["NMI"]), 4),
        "full_ASW": round(float(full_metrics["ASW"]), 4),
        "full_CH": round(float(full_metrics.get("CAL", 0)), 2),
        "full_DB": round(float(full_metrics.get("DAV", 0)), 4),
        "test_ARI": round(float(test_metrics["ARI"]), 4),
        "test_NMI": round(float(test_metrics["NMI"]), 4),
        "test_ASW": round(float(test_metrics["ASW"]), 4),
        "corr": round(float(full_metrics.get("COR", 0)), 4),
        "best_val_loss": round(float(model.best_val_loss), 2),
        "actual_epochs": int(res["actual_epochs"]),
        "train_time_s": round(float(res["train_time"]), 1),
        "peak_mem_gb": round(float(res["peak_memory_gb"]), 2),
    }

    # Internal data for NPZ and expanded metrics
    result["_val_losses"] = model.val_losses
    result["_val_scores"] = model.val_scores
    result["_train_losses"] = model.train_losses
    result["_latent"] = latent
    result["_labels"] = labels_all

    # ODE gradients
    if config.get("use_ode", False):
        try:
            result["_gradients"] = model.get_velocity()
            print(f"  ODE gradients saved ({result['_gradients'].shape})")
        except Exception as e:
            print(f"  ODE gradients skipped: {e}")
            result["_gradients"] = None
    else:
        result["_gradients"] = None

    return result


def compute_expanded_metrics(results, outdir):
    """Compute full metric battery for each config and save summary_expanded.csv."""
    print("\n  Computing expanded metrics...")
    expanded_rows = []

    for r in results:
        latent = r["_latent"]
        labels = r["_labels"]

        all_metrics = compute_all_metrics(latent, labels)

        # Remove private keys before saving to JSON
        saveable_metrics = {
            k: round(float(v), 6) if isinstance(v, (float, np.floating)) else v
            for k, v in all_metrics.items()
            if not k.startswith("_") and not isinstance(v, np.ndarray)
        }
        saveable_metrics["config"] = r["config"]

        # Merge with existing per-config JSON
        json_path = outdir / f"{r['config'].replace('+', '_')}.json"
        if json_path.exists():
            with open(json_path) as f:
                existing = json.load(f)
            existing.update(saveable_metrics)
            saveable_metrics = existing

        with open(json_path, "w") as f:
            json.dump(saveable_metrics, f, indent=2)

        expanded_rows.append(saveable_metrics)

    # Save summary_expanded.csv
    if expanded_rows:
        fields = [k for k in expanded_rows[0] if not isinstance(expanded_rows[0][k], (list, dict, type(None)))]
        csv_path = outdir / "summary_expanded.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for row in expanded_rows:
                w.writerow({k: v for k, v in row.items() if not isinstance(v, (list, dict, type(None)))})
        print(f"  Expanded metrics saved to {csv_path}")


def print_summary(results, beta):
    """Print tabular summary."""
    header = (
        f"{'Config':<20} {'ARI':>6} {'NMI':>6} {'ASW':>6} "
        f"{'CH':>8} {'DB':>6} {'Corr':>6} | "
        f"{'tARI':>6} {'tNMI':>6} {'tASW':>6} | "
        f"{'Ep':>4} {'Time':>6} {'Mem':>5}"
    )
    print(f"\n{'=' * 120}")
    print(f"  Beta = {beta}")
    print(f"{'=' * 120}")
    print(header)
    print(f"{'─' * 120}")
    for r in results:
        print(
            f"{r['config']:<20} {r['full_ARI']:>6.3f} {r['full_NMI']:>6.3f} "
            f"{r['full_ASW']:>6.3f} {r['full_CH']:>8.1f} {r['full_DB']:>6.3f} "
            f"{r['corr']:>6.3f} | {r['test_ARI']:>6.3f} {r['test_NMI']:>6.3f} "
            f"{r['test_ASW']:>6.3f} | {r['actual_epochs']:>4d} "
            f"{r['train_time_s']:>6.1f} {r['peak_mem_gb']:>5.2f}"
        )
    print(f"{'=' * 120}")


def main():
    parser = argparse.ArgumentParser(
        description="Series 1: Beta Ablation Study (Paper Tables I-V)"
    )
    parser.add_argument(
        "--data",
        default=os.environ.get("MOCOO_DATA_DIR", "data") + "/LAB/scRL/IRALL.h5ad",
        help="Path to .h5ad dataset",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override max training epochs (default: from beta_ablation config = 200)",
    )
    parser.add_argument(
        "--patience", type=int, default=None,
        help="Override early stopping patience (default: from config = 40)",
    )
    parser.add_argument(
        "--val-every", type=int, default=5,
        help="Validate every N epochs",
    )
    parser.add_argument(
        "--max-cells", type=int, default=3000,
        help="Max cells to subsample",
    )
    parser.add_argument(
        "--hvg", type=int, default=3000,
        help="Number of highly variable genes",
    )
    parser.add_argument(
        "--betas", nargs="*", type=float, default=None,
        help="Beta values to sweep (default: [0.01, 0.1, 1.0] from config)",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output base directory (default: benchmarks/results/beta_ablation)",
    )
    parser.add_argument(
        "--configs", nargs="*", default=None,
        help="Subset of configs to run (default: all 6)",
    )
    parser.add_argument(
        "--expanded", action="store_true", default=True,
        help="Compute expanded metrics (DRE, LSE, DREX, LSEX) after training",
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config("beta_ablation")
    training = get_training_params(cfg)
    sweep = get_sweep_params(cfg)
    SHARED = get_shared_params(cfg)
    CONFIGS = get_model_configs(cfg)

    epochs = args.epochs or training["epochs"]
    patience = args.patience or training["patience"]
    betas = args.betas or sweep["values"]

    outdir_base = (
        Path(args.outdir) if args.outdir
        else Path(__file__).resolve().parent.parent.parent / "results" / "beta_ablation"
    )

    configs_to_run = args.configs or list(CONFIGS.keys())

    # Load dataset once
    adata = load_dataset(args.data, args.max_cells, args.hvg)

    print(f"\n{'=' * 60}")
    print(f"  Beta Ablation Study")
    print(f"  Betas: {betas}")
    print(f"  Configs: {configs_to_run}")
    print(f"  Epochs: {epochs}, Patience: {patience}")
    print(f"  Cells: {adata.n_obs}, Genes: {adata.n_vars}")
    print(f"{'=' * 60}")

    for beta in betas:
        print(f"\n{'#' * 60}")
        print(f"  Beta = {beta}")
        print(f"{'#' * 60}")

        SHARED["beta"] = beta
        outdir = outdir_base / f"beta_{beta}"
        outdir.mkdir(parents=True, exist_ok=True)

        results = []
        for name in configs_to_run:
            if name not in CONFIGS:
                print(f"Unknown config: {name}, skipping")
                continue
            r = run_single(
                name, adata, CONFIGS[name], SHARED,
                epochs, patience, args.val_every,
            )
            results.append(r)

            # Save per-config JSON (without numpy arrays)
            saveable = {k: v for k, v in r.items() if not k.startswith("_")}
            with open(outdir / f"{name.replace('+', '_')}.json", "w") as f:
                json.dump(saveable, f, indent=2)

        # Save benchmark_data.npz
        save_dict = dict(
            configs=np.array([r["config"] for r in results], dtype=object),
            val_losses=np.array([r["_val_losses"] for r in results], dtype=object),
            val_scores=np.array([r["_val_scores"] for r in results], dtype=object),
            train_losses=np.array([r["_train_losses"] for r in results], dtype=object),
            latents=np.array([r["_latent"] for r in results], dtype=object),
            labels=np.array([r["_labels"] for r in results], dtype=object),
            gradients=np.array([r["_gradients"] for r in results], dtype=object),
        )
        if "batch" in adata.obs.columns:
            save_dict["batch_labels"] = adata.obs["batch"].values.astype(str)
        if "cell_type" in adata.obs.columns:
            save_dict["cell_type_labels"] = adata.obs["cell_type"].values.astype(str)
        np.savez(outdir / "benchmark_data.npz", **save_dict, allow_pickle=True)

        # Summary CSV
        fields = [k for k in results[0] if not k.startswith("_")]
        csv_path = outdir / "summary.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in results:
                w.writerow({k: v for k, v in r.items() if not k.startswith("_")})

        # Expanded metrics
        if args.expanded:
            compute_expanded_metrics(results, outdir)

        print_summary(results, beta)
        print(f"\nResults saved to: {outdir}")

    print(f"\n{'=' * 60}")
    print(f"  Beta ablation study complete.")
    print(f"  Results: {outdir_base}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
