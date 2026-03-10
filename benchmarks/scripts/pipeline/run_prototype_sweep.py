#!/usr/bin/env python3
"""
Prototype count sweep: test P ∈ {8, 12, 16, 20} on IRALL and Paul.
Addresses reviewer Concern 8 (fixed P=12).

Usage (GPU required):
    python benchmarks/scripts/pipeline/run_prototype_sweep.py
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

DATASET_SPECS = {
    "IRALL": {
        "path": str(BASE_DIR / "LAB" / "scRL" / "IRALL.h5ad"),
        "cell_type_col": "cell_type",
        "n_types": 12,
        "max_cells": 3000,
        "epochs": 150,
    },
    "paul": {
        "path": str(BASE_DIR / "LAB" / "data" / "paul.h5ad"),
        "cell_type_col": "paul15_clusters",
        "n_types": 19,
        "max_cells": 2000,
        "epochs": 100,
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


def main():
    parser = argparse.ArgumentParser(description="Prototype count sweep")
    parser.add_argument("--datasets", nargs="+", default=["IRALL", "paul"])
    parser.add_argument("--prototypes", type=int, nargs="+", default=[8, 12, 16, 20])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str,
                        default=str(_REPO_ROOT / "benchmarks" / "results" / "prototype_sweep"))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []

    for ds_name in args.datasets:
        if ds_name not in DATASET_SPECS:
            print(f"WARNING: Unknown dataset '{ds_name}', skipping")
            continue

        spec = DATASET_SPECS[ds_name]
        print(f"\n{'='*60}")
        print(f"DATASET: {ds_name} (n_types={spec['n_types']})")
        print(f"{'='*60}")

        adata = load_dataset(spec["path"], spec["max_cells"])

        for n_proto in args.prototypes:
            print(f"\n  P={n_proto}")

            import torch
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)

            from mocoo import MoCoO
            from mocoo.evaluation import compute_clustering_metrics

            config = dict(
                use_ode=True, use_moco=True, use_prototype=True,
                n_prototypes=n_proto, vae_reg=0.6, ode_reg=0.4,
                moco_weight=0.3, moco_T=0.2, moco_K=4096,
                proto_weight=0.1, random_seed=args.seed,
            )
            params = {**SHARED, **config}

            try:
                model = MoCoO(adata, **params)
                t0 = time.time()
                model.fit(epochs=spec["epochs"], patience=20, val_every=5)
                train_time = time.time() - t0

                latent = model.get_latent()
                labels = model.labels
                metrics = compute_clustering_metrics(latent, labels, random_state=args.seed)

                row = {
                    "dataset": ds_name,
                    "n_prototypes": n_proto,
                    "n_cell_types": spec["n_types"],
                    "ratio_P_to_K": round(n_proto / spec["n_types"], 2),
                    "train_time_s": round(train_time, 1),
                    **metrics,
                }
                rows.append(row)
                print(f"    ARI={metrics['ARI']:.4f}  NMI={metrics['NMI']:.4f}  "
                      f"ASW={metrics.get('ASW', float('nan')):.4f}  "
                      f"DAV={metrics.get('DAV', float('nan')):.3f}")
            except Exception as e:
                print(f"    ERROR: {e}")
                rows.append({"dataset": ds_name, "n_prototypes": n_proto, "error": str(e)})

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "prototype_sweep.csv", index=False)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    if "ARI" in df.columns:
        print(df[["dataset", "n_prototypes", "ARI", "NMI", "ASW", "DAV"]].to_string(index=False))

    print(f"\nResults saved to {outdir}/")


if __name__ == "__main__":
    main()
