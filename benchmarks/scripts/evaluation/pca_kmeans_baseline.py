#!/usr/bin/env python3
"""
PCA + k-means baseline for MoCoO benchmark comparison.
Addresses Major Concern M6: lack of external baselines.

Runs PCA dimensionality reduction followed by k-means clustering on the
same preprocessed data used by MoCoO, using identical train/val/test splits
and evaluation metrics.

Usage:
    python pca_kmeans_baseline.py
    python pca_kmeans_baseline.py --n_components 50 --datasets IRALL dentate endo
"""
import argparse
import sys
import os
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.model_selection import train_test_split
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Dataset registry (mirrors run_benchmark.py) ────────────────────────────
DATASETS = {
    "IRALL": {
        "path": "datasets/IRALL.h5ad",
        "cell_type_col": "cell_type",
        "batch_col": "batch",
    },
    "dentate": {
        "path": "datasets/DevelopmentDatasets/dentate.h5ad",
        "cell_type_col": "Clusters",
        "batch_col": None,
    },
    "endo": {
        "path": "datasets/DevelopmentDatasets/endo.h5ad",
        "cell_type_col": "clusters_fine",
        "batch_col": "day",
    },
}

BASE_DIR = Path(os.environ.get("MOCOO_DATA_DIR", "data"))


def load_and_preprocess(dataset_name: str, max_cells: int = 3000, n_hvg: int = 3000):
    """Load + preprocess exactly as MoCoO benchmark does."""
    spec = DATASETS[dataset_name]
    adata = sc.read_h5ad(str(BASE_DIR / spec["path"]))

    # Subsample
    if adata.n_obs > max_cells:
        sc.pp.subsample(adata, n_obs=max_cells, random_state=42)

    # Filter + HVG + normalise
    sc.pp.filter_genes(adata, min_cells=10)
    if adata.n_vars > n_hvg:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat_v3", span=1.0)
        adata = adata[:, adata.var.highly_variable].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    labels = adata.obs[spec["cell_type_col"]].values
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.nan_to_num(X, nan=0.0)

    return X, labels


def run_pca_kmeans(X, labels, n_components: int = 50, seed: int = 42):
    """PCA -> k-means and evaluate."""
    n_clusters = len(np.unique(labels))

    # Train/test split (same ratio as MoCoO: 70/15/15)
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.30, random_state=seed, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.50, random_state=seed, stratify=y_test
    )

    # Fit PCA on train, transform test
    nc = min(n_components, X_train.shape[1], X_train.shape[0] - 1)
    pca = PCA(n_components=nc, random_state=seed)
    Z_train = pca.fit_transform(X_train)
    Z_test = pca.transform(X_test)
    Z_all = pca.transform(X)

    # k-means on full PCA embedding
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    pred_test = km.fit_predict(Z_test)

    metrics = {
        "ARI": adjusted_rand_score(y_test, pred_test),
        "NMI": normalized_mutual_info_score(y_test, pred_test),
        "ASW": silhouette_score(Z_test, y_test),
        "CH": calinski_harabasz_score(Z_test, y_test),
        "DB": davies_bouldin_score(Z_test, y_test),
        "n_components": nc,
        "var_explained": float(pca.explained_variance_ratio_.sum()),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="PCA+k-means baseline")
    parser.add_argument("--n_components", type=int, default=50)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--datasets", nargs="+", default=["IRALL", "dentate", "endo"])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    rows = []
    for ds in args.datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds}")
        print(f"{'='*60}")

        X, labels = load_and_preprocess(ds)
        print(f"  Shape: {X.shape}, Types: {len(np.unique(labels))}")

        seed_results = []
        for seed in range(args.n_seeds):
            m = run_pca_kmeans(X, labels, n_components=args.n_components, seed=seed)
            seed_results.append(m)
            print(f"  Seed {seed}: ARI={m['ARI']:.3f}  NMI={m['NMI']:.3f}  ASW={m['ASW']:.3f}")

        # Aggregate
        for key in ["ARI", "NMI", "ASW", "CH", "DB"]:
            vals = [r[key] for r in seed_results]
            mean, std = np.mean(vals), np.std(vals)
            print(f"  {key}: {mean:.4f} ± {std:.4f}")

        for sr in seed_results:
            sr["dataset"] = ds
            sr["method"] = "PCA+k-means"
            rows.append(sr)

    df = pd.DataFrame(rows)
    out_path = args.output or str(
        BASE_DIR / "MoCoO" / "benchmarks" / "results" / "pca_kmeans_baseline.csv"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY (mean ± std across seeds)")
    print("=" * 70)
    summary = df.groupby("dataset")[["ARI", "NMI", "ASW", "CH", "DB"]].agg(["mean", "std"])
    print(summary.to_string())


if __name__ == "__main__":
    main()
