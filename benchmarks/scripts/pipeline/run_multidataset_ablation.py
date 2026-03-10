#!/usr/bin/env python3
"""
Multi-dataset ablation: run the 6-config ablation at beta=0.1 on IRALL, Paul, Dentate.
Produces a unified comparison table. Addresses reviewer Concern 3 (single-dataset reliance).

Usage (GPU required):
    python benchmarks/scripts/pipeline/run_multidataset_ablation.py
    python benchmarks/scripts/pipeline/run_multidataset_ablation.py --datasets IRALL paul dentate
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

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

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
    "VAE": dict(use_ode=False, use_moco=False, use_prototype=False),
    "VAE+ODE": dict(
        use_ode=True, use_moco=False, use_prototype=False,
        vae_reg=0.6, ode_reg=0.4,
    ),
    "VAE+MoCo": dict(
        use_ode=False, use_moco=True, use_prototype=False,
        moco_weight=0.5, moco_T=0.2, moco_K=4096,
    ),
    "VAE+MoCo+Proto": dict(
        use_ode=False, use_moco=True, use_prototype=True,
        n_prototypes=12, moco_weight=0.5, moco_T=0.2, moco_K=4096,
        proto_weight=0.1,
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

BASE_DIR = Path(os.environ.get("MOCOO_DATA_DIR", str(Path.home())))

DATASET_SPECS = {
    "IRALL": {
        "path": str(BASE_DIR / "LAB" / "scRL" / "IRALL.h5ad"),
        "cell_type_col": "cell_type",
        "epochs": 150,
        "max_cells": 3000,
    },
    "paul": {
        "path": str(BASE_DIR / "LAB" / "data" / "paul.h5ad"),
        "cell_type_col": "paul15_clusters",
        "epochs": 100,
        "max_cells": 2000,
    },
    "dentate": {
        "path": str(BASE_DIR / "vGAE_LAB" / "data" / "dentate.h5ad"),
        "cell_type_col": "Clusters",
        "epochs": 100,
        "max_cells": 3000,
    },
}


def load_dataset(path, max_cells, hvg=2000, seed=42):
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


def train_and_evaluate(adata, config_name, config, epochs, cell_type_col, seed=42):
    import torch
    from mocoo import MoCoO
    from mocoo.evaluation import compute_clustering_metrics

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    params = {**SHARED, **config, "random_seed": seed}
    model = MoCoO(adata, **params)

    t0 = time.time()
    model.fit(epochs=epochs, patience=20, val_every=5)
    train_time = time.time() - t0

    latent = model.get_latent()
    labels = model.labels
    metrics = compute_clustering_metrics(latent, labels, random_state=seed)
    metrics["train_time_s"] = round(train_time, 1)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Multi-dataset ablation")
    parser.add_argument("--datasets", nargs="+", default=["IRALL", "paul", "dentate"])
    parser.add_argument("--outdir", type=str, default=str(_REPO_ROOT / "benchmarks" / "results" / "multidataset"))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for ds_name in args.datasets:
        if ds_name not in DATASET_SPECS:
            print(f"WARNING: Unknown dataset '{ds_name}', skipping")
            continue

        spec = DATASET_SPECS[ds_name]
        print(f"\n{'='*60}")
        print(f"DATASET: {ds_name}")
        print(f"{'='*60}")

        adata = load_dataset(spec["path"], spec["max_cells"])

        for cfg_name, cfg in CONFIGS.items():
            print(f"\n  Config: {cfg_name}")
            try:
                metrics = train_and_evaluate(
                    adata, cfg_name, cfg, spec["epochs"],
                    spec["cell_type_col"],
                )
                row = {"dataset": ds_name, "config": cfg_name, **metrics}
                all_rows.append(row)
                print(f"    ARI={metrics['ARI']:.4f}  NMI={metrics['NMI']:.4f}  "
                      f"ASW={metrics.get('ASW', float('nan')):.4f}")
            except Exception as e:
                print(f"    ERROR: {e}")
                all_rows.append({"dataset": ds_name, "config": cfg_name, "error": str(e)})

    df = pd.DataFrame(all_rows)
    df.to_csv(outdir / "multidataset_ablation.csv", index=False)

    # Summary pivot
    if "ARI" in df.columns:
        pivot = df.pivot_table(index="config", columns="dataset", values="ARI", aggfunc="first")
        print(f"\n{'='*60}")
        print("ARI PIVOT TABLE")
        print(f"{'='*60}")
        print(pivot.to_string())
        pivot.to_csv(outdir / "multidataset_ari_pivot.csv")

    print(f"\nResults saved to {outdir}/")


if __name__ == "__main__":
    main()
