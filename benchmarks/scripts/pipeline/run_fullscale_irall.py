#!/usr/bin/env python3
"""
Full-scale IRALL evaluation for top 3 configurations.
Addresses W5: subsample-only ablation concern.

Runs VAE+ODE, VAE+ODE+MoCo, and Full at β=0.1 on the complete IRALL dataset
(no subsampling) with 3 seeds each.

Usage (GPU required):
    python benchmarks/scripts/pipeline/run_fullscale_irall.py
"""
import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

BASE_DIR = Path(os.environ.get("MOCOO_DATA_DIR", str(Path.home())))

SHARED = dict(
    latent_dim=32,
    hidden_dim=128,
    i_dim=4,
    lr=1e-4,
    batch_size=128,
    beta=0.1,
    recon=1.0,
    loss_mode="nb",
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
)

CONFIGS = {
    "VAE+ODE": dict(
        use_ode=True, use_moco=False, use_prototype=False,
        vae_reg=0.6, ode_reg=0.4,
    ),
    "VAE+ODE+MoCo": dict(
        use_ode=True, use_moco=True, use_prototype=False,
        vae_reg=0.6, ode_reg=0.4,
        moco_weight=0.3, moco_T=0.2, moco_K=4096,
    ),
    "Full": dict(
        use_ode=True, use_moco=True, use_prototype=True,
        n_prototypes=12, vae_reg=0.6, ode_reg=0.4,
        moco_weight=0.3, moco_T=0.2, moco_K=4096,
        proto_weight=0.1,
    ),
}


def load_irall(max_cells=None, hvg=3000, seed=42):
    import scanpy as sc
    from scipy.sparse import issparse

    path = str(BASE_DIR / "LAB" / "scRL" / "IRALL.h5ad")
    adata = sc.read_h5ad(path)

    if max_cells and adata.n_obs > max_cells:
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

    adata.obs["cell_type"] = adata.obs["cell_type"].values
    return adata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-cells", type=int, default=None,
                        help="Max cells (None = full scale)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=30)
    args = parser.parse_args()

    outdir = Path(_REPO_ROOT / "benchmarks" / "results" / "fullscale_irall")
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / "fullscale_results.csv"

    import torch
    from mocoo import MoCoO
    from sklearn.cluster import KMeans
    from sklearn.metrics import (
        adjusted_rand_score, normalized_mutual_info_score,
        silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    )

    rows = []
    total = len(CONFIGS) * args.seeds
    done = 0

    for config_name, config in CONFIGS.items():
        for seed in range(args.seeds):
            done += 1
            print(f"\n[{done}/{total}] {config_name} seed={seed}")

            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            adata = load_irall(max_cells=args.max_cells, seed=seed)
            n_cells = adata.n_obs
            print(f"  Loaded {n_cells} cells, {adata.n_vars} genes")

            params = {**SHARED, **config, "random_seed": seed}

            try:
                model = MoCoO(adata, **params)
                t0 = time.time()
                model.fit(epochs=args.epochs, patience=args.patience, val_every=5)
                train_time = time.time() - t0

                latent = model.get_latent()
                labels = model.labels
                n_clusters = len(np.unique(labels))

                pred = KMeans(n_clusters=n_clusters, n_init=10,
                              random_state=seed).fit_predict(latent)

                row = {
                    "config": config_name,
                    "seed": seed,
                    "n_cells": n_cells,
                    "ARI": round(adjusted_rand_score(labels, pred), 4),
                    "NMI": round(normalized_mutual_info_score(labels, pred), 4),
                    "ASW": round(silhouette_score(latent, pred), 4),
                    "CH": round(calinski_harabasz_score(latent, pred), 2),
                    "DB": round(davies_bouldin_score(latent, pred), 4),
                    "train_time_s": round(train_time, 1),
                }
                rows.append(row)

                print(f"  ARI={row['ARI']:.4f}  NMI={row['NMI']:.4f}  "
                      f"ASW={row['ASW']:.4f}  CH={row['CH']:.1f}  "
                      f"time={row['train_time_s']:.0f}s")

                # Incremental save
                pd.DataFrame([row]).to_csv(
                    out_csv, mode="a",
                    header=not out_csv.exists(),
                    index=False)

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(outdir / "fullscale_summary.csv", index=False)
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(df.groupby("config")[["ARI", "NMI", "ASW", "CH", "DB", "train_time_s"]].agg(["mean", "std"]).to_string())


if __name__ == "__main__":
    main()
