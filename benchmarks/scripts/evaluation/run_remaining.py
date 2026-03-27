#!/usr/bin/env python3
"""Run missing dataset×method combos for seed=0 only.

Clears GPU cache between each model to avoid OOM when other processes
are running.
"""
import gc
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.sparse import issparse
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

warnings.filterwarnings("ignore")

SKILL_PARENT = str(Path.home() / ".copilot" / "skills")
sys.path.insert(0, SKILL_PARENT)

BASE_DIR = Path.home()
DATASETS = {
    "dentate": {
        "path": str(BASE_DIR / "vGAE_LAB" / "data" / "dentate.h5ad"),
        "cell_type_col": "Clusters",
        "max_cells": 3000,
    },
    "hemato": {
        "path": str(BASE_DIR / "Downloads" / "DevelopmentDatasets" / "hemato.h5ad"),
        "cell_type_col": "Cell type",
        "max_cells": 3000,
    },
    "lung": {
        "path": str(BASE_DIR / "Downloads" / "DevelopmentDatasets" / "lung.h5ad"),
        "cell_type_col": "clusters",
        "max_cells": 3000,
    },
    "spinoids": {
        "path": str(BASE_DIR / "LAB" / "data" / "spinoids.h5ad"),
        "cell_type_col": "seurat_clusters",
        "max_cells": 3000,
    },
}

UNIFIED_MODELS = {
    "CellBLAST": ("create_cellblast_model", {"latent_dim": 10, "hidden_dims": [256, 128]}),
    "CLEAR": ("create_clear_model", {"latent_dim": 128, "hidden_dims": [512, 256]}),
    "GM-VAE": ("create_gmvae_model", {"latent_dim": 10, "hidden_dims": [512, 256], "distribution": "euclidean"}),
    "SCALEX": ("create_scalex_model", {"latent_dim": 10, "hidden_dims": [512, 256]}),
    "scDAC": ("create_scdac_model", {"latent_dim": 32}),
    "scDeepCluster": ("create_scdeepcluster_model", {"latent_dim": 32, "n_clusters": 10}),
    "scSMD": ("create_scsmd_model", {"latent_dim": 10, "n_clusters": 10}),
    "siVAE": ("create_sivae_model", {"latent_dim": 50, "hidden_dims": [512, 256]}),
}

_NEEDS_TUPLE = {"CLEAR"}


def load_dataset(ds_name, seed=0, hvg=3000):
    spec = DATASETS[ds_name]
    adata = sc.read_h5ad(spec["path"])
    adata.var_names_make_unique()
    if adata.n_obs > spec["max_cells"]:
        sc.pp.subsample(adata, n_obs=spec["max_cells"], random_state=seed)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=min(hvg, adata.n_vars))
    adata = adata[:, adata.var["highly_variable"]].copy()
    adata.obs["cell_type"] = adata.obs[spec["cell_type_col"]].astype(str)
    return adata


def compute_metrics(latent, labels, seed=0):
    km = KMeans(n_clusters=len(np.unique(labels)), random_state=seed, n_init=10)
    pred = km.fit_predict(latent)
    return {
        "ARI": round(adjusted_rand_score(labels, pred), 4),
        "NMI": round(normalized_mutual_info_score(labels, pred), 4),
        "ASW": round(silhouette_score(latent, labels), 4),
        "CH": round(calinski_harabasz_score(latent, labels), 2),
        "DB": round(davies_bouldin_score(latent, labels), 4),
    }


def run_ml_baselines(X, seed=0):
    from sklearn.decomposition import PCA, FastICA, NMF, FactorAnalysis, TruncatedSVD
    results = {}
    n_comp = min(50, X.shape[1])
    for name, Cls, kw in [
        ("PCA", PCA, {"n_components": n_comp}),
        ("ICA", FastICA, {"n_components": n_comp, "random_state": seed, "max_iter": 500}),
        ("NMF", NMF, {"n_components": n_comp, "random_state": seed, "max_iter": 500}),
        ("FA", FactorAnalysis, {"n_components": n_comp, "random_state": seed}),
        ("TruncatedSVD", TruncatedSVD, {"n_components": n_comp, "random_state": seed}),
    ]:
        X_in = np.clip(X, 0, None) if name == "NMF" else X
        results[name] = Cls(**kw).fit_transform(X_in)
    return results


def run_unified_model(model_name, X, seed=0, epochs=50, device="cuda", n_clusters=10):
    import importlib
    um = importlib.import_module("external-benchmarker.unified_models")

    factory_name, kwargs = UNIFIED_MODELS[model_name]
    kwargs = {**kwargs, "input_dim": X.shape[1]}
    if "n_clusters" in kwargs:
        kwargs["n_clusters"] = n_clusters

    torch.manual_seed(seed)
    np.random.seed(seed)

    factory = getattr(um, factory_name)
    model = factory(**kwargs)

    X_t = torch.FloatTensor(X)
    as_tuple = model_name in _NEEDS_TUPLE

    class _TensorLoader:
        def __init__(self, data, batch_size=128, shuffle=True, as_tuple=False):
            self._data = data
            self._bs = batch_size
            self._shuffle = shuffle
            self._as_tuple = as_tuple
        def __iter__(self):
            n = self._data.shape[0]
            idx = torch.randperm(n) if self._shuffle else torch.arange(n)
            for i in range(0, n, self._bs):
                batch = self._data[idx[i:i + self._bs]]
                if self._as_tuple:
                    yield (batch, batch)
                else:
                    yield batch
        def __len__(self):
            return (self._data.shape[0] + self._bs - 1) // self._bs

    train_loader = _TensorLoader(X_t, batch_size=128, shuffle=True, as_tuple=as_tuple)

    model.fit(train_loader, epochs=epochs, lr=1e-3, device=device, verbose=False)

    eval_loader = _TensorLoader(X_t, batch_size=256, shuffle=False, as_tuple=as_tuple)
    latent = model.encode(eval_loader, device=device)
    if isinstance(latent, torch.Tensor):
        latent = latent.detach().cpu().numpy()

    return latent


def main():
    out_dir = Path(__file__).resolve().parent.parent.parent / "results" / "baselines"
    csv_path = out_dir / "remaining_datasets_v2.csv"

    # Force CPU to avoid GPU contention with other processes
    device = "cpu"
    print(f"Device: {device} (forced to avoid GPU contention)")

    all_rows = []

    for ds_name in ["dentate", "hemato", "lung", "spinoids"]:
        print(f"\n{'='*60}\nDATASET: {ds_name}\n{'='*60}")

        adata = load_dataset(ds_name, seed=0)
        labels = adata.obs["cell_type"].values
        X = adata.X.toarray() if issparse(adata.X) else np.array(adata.X)
        X = np.nan_to_num(X, nan=0.0)
        n_clusters = len(np.unique(labels))

        # ML baselines (no GPU needed)
        print("\n  [ML baselines]")
        ml_results = run_ml_baselines(X, seed=0)
        for name, latent in ml_results.items():
            m = compute_metrics(latent, labels, seed=0)
            m.update({"method": name, "dataset": ds_name, "seed": 0})
            all_rows.append(m)
            print(f"    {name}: ARI={m['ARI']:.4f} NMI={m['NMI']:.4f} ASW={m['ASW']:.4f}")

        # Unified DL models — one at a time with cache clearing
        for model_name in UNIFIED_MODELS:
            print(f"\n  [{model_name}]")
            # Clear GPU before each model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            try:
                t0 = time.time()
                latent = run_unified_model(model_name, X, seed=0, epochs=50,
                                           device=device, n_clusters=n_clusters)
                elapsed = time.time() - t0
                m = compute_metrics(latent, labels, seed=0)
                m.update({"method": model_name, "dataset": ds_name, "seed": 0,
                          "train_time_s": round(elapsed, 1)})
                all_rows.append(m)
                print(f"    ARI={m['ARI']:.4f} NMI={m['NMI']:.4f} ASW={m['ASW']:.4f} time={elapsed:.0f}s")
            except Exception as e:
                print(f"    ERROR: {e}")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()

        # Save incrementally
        pd.DataFrame(all_rows).to_csv(csv_path, index=False)
        print(f"  [saved {len(all_rows)} rows to {csv_path}]")

    print(f"\nDone. {len(all_rows)} total rows saved to {csv_path}")


if __name__ == "__main__":
    main()
