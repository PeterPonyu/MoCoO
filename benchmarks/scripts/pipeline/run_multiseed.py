#!/usr/bin/env python3
"""
Multi-seed benchmark runner for MoCoO ablation study.
Addresses Major Concern M1: single-seed evaluation.

Wraps the existing run_benchmark.py logic to run each configuration
across N seeds on each dataset, then outputs seed-level results
that can be fed into significance_tests.py.

Usage (GPU required):
    conda run -p /home/zeyufu/Desktop/.conda python run_multiseed.py --seeds 5 --datasets IRALL
    conda run -p /home/zeyufu/Desktop/.conda python run_multiseed.py --seeds 5 --datasets IRALL dentate endo --epochs 300

Estimated GPU time:
    1 dataset × 6 configs × 5 seeds ≈ 1-2 hours (RTX 4090)
    3 datasets × 6 configs × 5 seeds ≈ 4-6 hours
"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Ensure MoCoO is importable
BASE_DIR = Path("/home/zeyufu/Desktop")
sys.path.insert(0, str(BASE_DIR / "MoCoO"))

# Import configs from run_benchmark
sys.path.insert(0, str(BASE_DIR / "MoCoO" / "benchmarks" / "scripts" / "pipeline"))


# ── Shared hyperparameters ──────────────────────────────────────────────────
SHARED = dict(
    latent_dim=32,
    hidden_dim=128,
    i_dim=4,
    lr=1e-4,
    batch_size=128,
    beta=1.0,
    recon=1.0,
    loss_mode="nb",
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
)

CONFIGS = {
    "VAE": dict(use_ode=False, use_moco=False, use_prototype=False),
    "VAE+ODE": dict(
        use_ode=True, use_moco=False, use_prototype=False,
        vae_reg=0.5, ode_reg=0.5,
    ),
    "VAE+MoCo": dict(
        use_ode=False, use_moco=True, use_prototype=False,
        moco_weight=1.0, moco_T=0.2, moco_K=4096,
    ),
    "VAE+MoCo+Proto": dict(
        use_ode=False, use_moco=True, use_prototype=True,
        n_prototypes=12, moco_weight=1.0, moco_T=0.2, moco_K=4096,
        proto_weight=0.1,
    ),
    "VAE+ODE+MoCo": dict(
        use_ode=True, use_moco=True, use_prototype=False,
        vae_reg=0.5, ode_reg=0.5,
        moco_weight=1.0, moco_T=0.2, moco_K=4096,
    ),
    "MoCoO": dict(
        use_ode=True, use_moco=True, use_prototype=True,
        n_prototypes=12, vae_reg=0.5, ode_reg=0.5,
        moco_weight=1.0, moco_T=0.2, moco_K=4096,
        proto_weight=0.1,
    ),
}

DATASET_SPECS = {
    "IRALL": {
        "path": str(BASE_DIR / "datasets" / "IRALL.h5ad"),
        "cell_type_col": "cell_type",
        "batch_col": "batch",
        "epochs_default": 300,
    },
    "dentate": {
        "path": str(BASE_DIR / "datasets" / "DevelopmentDatasets" / "dentate.h5ad"),
        "cell_type_col": "Clusters",
        "batch_col": None,
        "epochs_default": 100,
    },
    "endo": {
        "path": str(BASE_DIR / "datasets" / "DevelopmentDatasets" / "endo.h5ad"),
        "cell_type_col": "clusters_fine",
        "batch_col": "day",
        "epochs_default": 100,
    },
}


def load_dataset(path, max_cells, hvg, seed):
    """Load, subsample, HVG-filter."""
    import scanpy as sc
    from scipy.sparse import issparse

    adata = sc.read_h5ad(path)
    if adata.n_obs > max_cells:
        sc.pp.subsample(adata, n_obs=max_cells, random_state=seed)
    sc.pp.filter_genes(adata, min_cells=10)

    if "counts" not in adata.layers:
        X = adata.X.toarray() if issparse(adata.X) else adata.X.copy()
        adata.layers["counts"] = X

    if adata.n_vars > hvg:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        try:
            sc.pp.highly_variable_genes(adata, n_top_genes=hvg, flavor="seurat_v3", layer="counts")
        except Exception:
            sc.pp.highly_variable_genes(adata, n_top_genes=hvg)
        adata = adata[:, adata.var["highly_variable"]].copy()

    return adata


def train_and_evaluate(adata, config_name, config, seed, epochs, patience, val_every):
    """Train one config with one seed, return metrics dict."""
    import torch
    from mocoo import MoCoO
    from sklearn.cluster import KMeans
    from sklearn.metrics import (
        adjusted_rand_score, normalized_mutual_info_score,
        silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    )

    # Set all random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    params = {**SHARED, **config, "random_seed": seed}
    model = MoCoO(adata, **params)

    t0 = time.time()
    model.fit(epochs=epochs, patience=patience, val_every=val_every)
    train_time = time.time() - t0

    # Get latents
    latent = model.get_latent()
    test_latent = model.get_test_latent()
    labels_all = model.labels
    labels_test = labels_all[model.test_idx]
    n_clusters = len(np.unique(labels_all))

    # Full-data metrics
    pred = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(latent)
    # Test-set metrics
    pred_test = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(test_latent)

    res = model.get_resource_metrics()

    return {
        "config": config_name,
        "seed": seed,
        "ARI": round(adjusted_rand_score(labels_all, pred), 4),
        "NMI": round(normalized_mutual_info_score(labels_all, pred), 4),
        "ASW": round(silhouette_score(latent, pred), 4),
        "CH": round(calinski_harabasz_score(latent, pred), 2),
        "DB": round(davies_bouldin_score(latent, pred), 4),
        "test_ARI": round(adjusted_rand_score(labels_test, pred_test), 4),
        "test_NMI": round(normalized_mutual_info_score(labels_test, pred_test), 4),
        "test_ASW": round(silhouette_score(test_latent, pred_test), 4),
        "train_time_s": round(train_time, 1),
        "peak_mem_gb": round(res.get("peak_memory_gb", 0), 2),
        "actual_epochs": res.get("actual_epochs", epochs),
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-seed MoCoO benchmark")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds (0, 1, ..., n-1)")
    parser.add_argument("--datasets", nargs="+", default=["IRALL"],
                        choices=list(DATASET_SPECS.keys()))
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Subset of configs to run (default: all)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override default epochs for all datasets")
    parser.add_argument("--max_cells", type=int, default=3000)
    parser.add_argument("--hvg", type=int, default=3000)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--val_every", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Skip configs/seeds already in output CSV")
    args = parser.parse_args()

    configs_to_run = args.configs or list(CONFIGS.keys())
    out_dir = Path(args.output_dir) if args.output_dir else \
        BASE_DIR / "MoCoO" / "benchmarks" / "results" / "multiseed"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds_name in args.datasets:
        ds_spec = DATASET_SPECS[ds_name]
        epochs = args.epochs or ds_spec["epochs_default"]
        out_csv = out_dir / f"multiseed_{ds_name}.csv"

        # Resume support
        existing = set()
        if args.resume and out_csv.exists():
            prev = pd.read_csv(out_csv)
            existing = set(zip(prev["config"], prev["seed"]))
            print(f"Resuming: {len(existing)} runs already completed for {ds_name}")

        print(f"\n{'='*70}")
        print(f"DATASET: {ds_name} | epochs={epochs} | seeds={args.seeds}")
        print(f"{'='*70}")

        # Load once per dataset (same subsample per seed? No, vary seed)
        all_rows = []

        total = len(configs_to_run) * args.seeds
        done = 0

        for config_name in configs_to_run:
            for seed in range(args.seeds):
                done += 1
                if (config_name, seed) in existing:
                    print(f"  [{done}/{total}] {config_name} seed={seed} — SKIPPED (exists)")
                    continue

                print(f"\n  [{done}/{total}] {config_name} seed={seed}")

                # Reload per seed to get different subsample
                adata = load_dataset(ds_spec["path"], args.max_cells, args.hvg, seed=seed)
                adata.obs["cell_type"] = adata.obs[ds_spec["cell_type_col"]].values

                try:
                    metrics = train_and_evaluate(
                        adata, config_name, CONFIGS[config_name],
                        seed, epochs, args.patience, args.val_every
                    )
                    metrics["dataset"] = ds_name
                    all_rows.append(metrics)

                    print(f"    ARI={metrics['ARI']:.4f}  NMI={metrics['NMI']:.4f}  "
                          f"ASW={metrics['ASW']:.4f}  time={metrics['train_time_s']:.0f}s")

                    # Incremental save
                    df_new = pd.DataFrame([metrics])
                    if out_csv.exists():
                        df_new.to_csv(out_csv, mode="a", header=False, index=False)
                    else:
                        df_new.to_csv(out_csv, index=False)

                except Exception as e:
                    print(f"    ERROR: {e}")
                    all_rows.append({
                        "config": config_name, "seed": seed, "dataset": ds_name,
                        "error": str(e),
                    })

        # Summary
        if all_rows:
            df = pd.DataFrame([r for r in all_rows if "error" not in r])
            if len(df) > 0:
                print(f"\n{'='*70}")
                print(f"SUMMARY — {ds_name}")
                print(f"{'='*70}")
                summary = df.groupby("config")[["ARI", "NMI", "ASW"]].agg(["mean", "std"])
                print(summary.to_string())

    print(f"\nAll results saved to {out_dir}/")
    print("\nNext: Run significance tests:")
    print(f"  python significance_tests.py --input {out_dir}/multiseed_IRALL.csv")


if __name__ == "__main__":
    main()
