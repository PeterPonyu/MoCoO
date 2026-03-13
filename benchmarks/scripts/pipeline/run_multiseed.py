#!/usr/bin/env python3
"""
Multi-seed benchmark runner for MoCoO ablation study.
Addresses Major Concern M1: single-seed evaluation.

Runs each configuration across N seeds on each dataset, then outputs
seed-level results that can be fed into significance_tests.py.

All hyperparameters are loaded from the canonical YAML config
(default.yaml) via the mocoo.configs API.

Usage (GPU required):
    python run_multiseed.py --seeds 5 --datasets IRALL
    python run_multiseed.py --seeds 5 --datasets IRALL dentate endo
"""
import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from mocoo.configs import (
    load_config,
    get_shared_params,
    get_model_configs,
    get_training_params,
    get_dataset_paths,
)

# ── Load all params from canonical YAML config ─────────────────────────────
_CFG = load_config("default")
SHARED = get_shared_params(_CFG)
CONFIGS = get_model_configs(_CFG)
TRAINING = get_training_params(_CFG)
DATASET_SPECS = get_dataset_paths(_CFG)


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

    # Leiden clustering (reclustering-free alternative to KMeans)
    try:
        from mocoo.evaluation import compute_leiden_metrics, compute_neighborhood_metrics
        leiden = compute_leiden_metrics(latent, labels_all)
        nbr = compute_neighborhood_metrics(latent, labels_all)
    except Exception:
        leiden = {"Leiden_ARI_best": np.nan, "Leiden_NMI_best": np.nan}
        nbr = {"kNN_purity": np.nan}

    return {
        "config": config_name,
        "seed": seed,
        "ARI": round(adjusted_rand_score(labels_all, pred), 4),
        "NMI": round(normalized_mutual_info_score(labels_all, pred), 4),
        "ASW": round(silhouette_score(latent, pred), 4),
        "CH": round(calinski_harabasz_score(latent, pred), 2),
        "DB": round(davies_bouldin_score(latent, pred), 4),
        "leiden_ARI": round(leiden.get("Leiden_ARI_best", np.nan), 4),
        "leiden_NMI": round(leiden.get("Leiden_NMI_best", np.nan), 4),
        "knn_purity": round(nbr.get("kNN_purity", np.nan), 4),
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
                        help="Override epochs (default: from YAML config)")
    parser.add_argument("--max-cells", type=int, default=None,
                        help="Override max cells (default: per-dataset from YAML)")
    parser.add_argument("--hvg", type=int, default=None,
                        help="Override HVG count (default: per-dataset from YAML)")
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--val_every", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Skip configs/seeds already in output CSV")
    args = parser.parse_args()

    configs_to_run = args.configs or list(CONFIGS.keys())
    epochs = args.epochs or TRAINING.get("epochs", 400)
    patience = args.patience or TRAINING.get("patience", 60)
    val_every = args.val_every or TRAINING.get("val_every", 5)
    out_dir = Path(args.output_dir) if args.output_dir else \
        _REPO_ROOT / "benchmarks" / "results" / "multiseed"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds_name in args.datasets:
        ds_spec = DATASET_SPECS[ds_name]
        out_csv = out_dir / f"multiseed_{ds_name}.csv"

        # Resume support
        existing = set()
        if args.resume and out_csv.exists():
            prev = pd.read_csv(out_csv)
            existing = set(zip(prev["config"], prev["seed"]))
            print(f"Resuming: {len(existing)} runs already completed for {ds_name}")

        max_cells = args.max_cells or ds_spec.get("max_cells", 3000)
        hvg = args.hvg or ds_spec.get("hvg", 3000)

        print(f"\n{'='*70}")
        print(f"DATASET: {ds_name} | epochs={epochs} | seeds={args.seeds}")
        print(f"{'='*70}")

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

                adata = load_dataset(ds_spec["path"], max_cells, hvg, seed=seed)
                adata.obs["cell_type"] = adata.obs[ds_spec["cell_type_col"]].values

                try:
                    metrics = train_and_evaluate(
                        adata, config_name, CONFIGS[config_name],
                        seed, epochs, patience, val_every
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
    print(f"  python benchmarks/scripts/evaluation/significance_tests.py "
          f"--input {out_dir}/multiseed_IRALL.csv")


if __name__ == "__main__":
    main()
