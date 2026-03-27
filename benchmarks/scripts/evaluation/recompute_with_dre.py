#!/usr/bin/env python3
"""
Recompute baselines with DRE/DREX metrics.

Lightweight script that re-runs ML baselines (fast) and existing external
methods (DPT, PCA+KMeans, Harmony, scVI, scANVI) on all available datasets,
now computing DRE/DREX in addition to the original 5 clustering metrics.

Usage:
    python benchmarks/scripts/evaluation/recompute_with_dre.py --ml-only
    python benchmarks/scripts/evaluation/recompute_with_dre.py --all
"""
import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA, NMF, FactorAnalysis, TruncatedSVD
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

BASE_DIR = Path(os.environ.get("MOCOO_DATA_DIR", str(Path.home())))

DATASETS = {
    "IRALL": {
        "path": str(BASE_DIR / "LAB" / "scRL" / "IRALL.h5ad"),
        "cell_type_col": "cell_type",
        "batch_col": "batch",
        "max_cells": 3000,
    },
    "paul": {
        "path": str(BASE_DIR / "LAB" / "data" / "paul.h5ad"),
        "cell_type_col": "paul15_clusters",
        "batch_col": None,
        "max_cells": 2700,
    },
    "endo": {
        "path": str(BASE_DIR / "vGAE_LAB" / "data" / "endo.h5ad"),
        "cell_type_col": "clusters_fine",
        "batch_col": "day",
        "max_cells": 2500,
    },
    "dentate": {
        "path": str(BASE_DIR / "vGAE_LAB" / "data" / "dentate.h5ad"),
        "cell_type_col": "Clusters",
        "batch_col": None,
        "max_cells": 3000,
    },
    "hemato": {
        "path": str(BASE_DIR / "Downloads" / "DevelopmentDatasets" / "hemato.h5ad"),
        "cell_type_col": "Cell type",
        "batch_col": "batch",
        "max_cells": 3000,
    },
    "lung": {
        "path": str(BASE_DIR / "Downloads" / "DevelopmentDatasets" / "lung.h5ad"),
        "cell_type_col": "clusters",
        "batch_col": "batch",
        "max_cells": 3000,
    },
    "spinoids": {
        "path": str(BASE_DIR / "LAB" / "data" / "spinoids.h5ad"),
        "cell_type_col": "seurat_clusters",
        "batch_col": None,
        "max_cells": 3000,
    },
}


def load_dataset(ds_name, seed=42, hvg=3000):
    """Load and preprocess dataset identically to MoCoO pipeline."""
    spec = DATASETS[ds_name]
    adata = sc.read_h5ad(spec["path"])
    if adata.n_obs > spec["max_cells"]:
        sc.pp.subsample(adata, n_obs=spec["max_cells"], random_state=seed)
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
    adata.obs["cell_type"] = adata.obs[spec["cell_type_col"]].values
    return adata


def compute_metrics(latent, labels, seed=42):
    """Compute all 7 metrics: ARI, NMI, ASW, CH, DB, DRE, DREX."""
    n_clusters = len(np.unique(labels))
    pred = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(latent)
    m = {
        "ARI": round(adjusted_rand_score(labels, pred), 4),
        "NMI": round(normalized_mutual_info_score(labels, pred), 4),
        "ASW": round(silhouette_score(latent, pred), 4),
        "CH": round(calinski_harabasz_score(latent, pred), 2),
        "DB": round(davies_bouldin_score(latent, pred), 4),
    }
    try:
        from mocoo.evaluation._projections import compute_2d_projections
        from mocoo.evaluation.bench import compute_dre_metrics
        from mocoo.evaluation.drex import compute_drex_metrics

        # Subsample for DRE/DREX (O(n^2) pairwise distances) — cap at 1500 cells
        _DRE_MAX = 1500
        if latent.shape[0] > _DRE_MAX:
            rng = np.random.default_rng(seed)
            idx = rng.choice(latent.shape[0], _DRE_MAX, replace=False)
            lat_sub = latent[idx]
        else:
            lat_sub = latent

        umap_2d, _ = compute_2d_projections(lat_sub)
        if umap_2d is not None:
            dre = compute_dre_metrics(lat_sub, umap_2d, k=15, prefix="DRE_umap")
            m["DRE"] = round(dre.get("DRE_umap_overall_quality", float("nan")), 4)
            drex = compute_drex_metrics(lat_sub, umap_2d, k=15)
            m["DREX"] = round(drex.get("DREX_overall_quality", float("nan")), 4)
        else:
            m["DRE"] = float("nan")
            m["DREX"] = float("nan")
    except Exception as e:
        print(f"    DRE/DREX failed: {e}")
        m["DRE"] = float("nan")
        m["DREX"] = float("nan")
    return m


def run_ml_baselines(X, seed=42):
    """Run all ML decomposition baselines → dict[name → latent]."""
    results = {}
    nc = 10
    for name, cls, kwargs in [
        ("PCA", PCA, {}),
        ("ICA", FastICA, {"max_iter": 500}),
        ("FA", FactorAnalysis, {}),
        ("TruncatedSVD", TruncatedSVD, {}),
    ]:
        try:
            n = min(nc, X.shape[1] - 1, X.shape[0] - 1)
            results[name] = cls(n_components=n, random_state=seed, **kwargs).fit_transform(X)
        except Exception as e:
            print(f"    {name} ERROR: {e}")
    # NMF needs non-negative
    try:
        X_nn = X.copy()
        X_nn[X_nn < 0] = 0
        n = min(nc, X_nn.shape[1] - 1, X_nn.shape[0] - 1)
        results["NMF"] = NMF(n_components=n, random_state=seed, max_iter=500).fit_transform(X_nn)
    except Exception as e:
        print(f"    NMF ERROR: {e}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--ml-only", action="store_true", help="Only ML baselines (no DL)")
    parser.add_argument("--all", action="store_true", help="All methods including scVI/scANVI/DL models")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent.parent.parent / "results" / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "extended_baselines_v2.csv"

    # Load existing data to avoid overwriting previous runs
    rows = []
    existing_keys = set()
    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        rows = existing_df.to_dict("records")
        for r in rows:
            existing_keys.add((r.get("method", ""), r.get("dataset", ""), r.get("seed", "")))
        print(f"Loaded {len(rows)} existing rows from {csv_path}")

    for ds_name in args.datasets:
        if not Path(DATASETS[ds_name]["path"]).exists():
            print(f"SKIP {ds_name}: data file not found")
            continue
        print(f"\n{'='*60}\n{ds_name}\n{'='*60}")

        for seed in range(args.n_seeds):
            adata = load_dataset(ds_name, seed=seed)
            labels = adata.obs["cell_type"].values
            X = adata.X.toarray() if issparse(adata.X) else np.array(adata.X)
            X = np.nan_to_num(X, nan=0.0)

            # ML baselines
            ml_needed = [n for n in ["PCA","ICA","NMF","FA","TruncatedSVD"]
                         if (n, ds_name, seed) not in existing_keys]
            if ml_needed:
                print(f"  [ML] seed={seed} ({len(ml_needed)} methods)")
                ml = run_ml_baselines(X, seed=seed)
                for name, latent in ml.items():
                    if (name, ds_name, seed) in existing_keys:
                        continue
                    t0 = time.time()
                    m = compute_metrics(latent, labels, seed=seed)
                    m.update({"method": name, "dataset": ds_name, "seed": seed,
                              "train_time_s": round(time.time() - t0, 1)})
                    rows.append(m)
                    print(f"    {name}: ASW={m['ASW']:.4f} DRE={m.get('DRE',float('nan')):.4f} "
                          f"DREX={m.get('DREX',float('nan')):.4f}")

            # DPT + PCA+KMeans (always fast)
            print(f"  [DPT] seed={seed}")
            try:
                adata_dpt = adata.copy()
                n_pcs = min(50, adata_dpt.n_vars - 1, adata_dpt.n_obs - 1)
                sc.pp.pca(adata_dpt, n_comps=n_pcs, random_state=seed)
                sc.pp.neighbors(adata_dpt, n_neighbors=15, n_pcs=n_pcs)
                sc.tl.diffmap(adata_dpt, n_comps=15)
                cell_types = adata_dpt.obs["cell_type"].values
                most_common = pd.Series(cell_types).value_counts().index[0]
                root_idx = np.where(cell_types == most_common)[0]
                adata_dpt.uns["iroot"] = root_idx[seed % len(root_idx)]
                sc.tl.dpt(adata_dpt)
                latent = adata_dpt.obsm["X_diffmap"][:, :15]
                m = compute_metrics(latent, labels, seed=seed)
                m.update({"method": "DPT", "dataset": ds_name, "seed": seed, "train_time_s": 0})
                rows.append(m)
                print(f"    DPT: ASW={m['ASW']:.4f} DRE={m.get('DRE',float('nan')):.4f}")
            except Exception as e:
                print(f"    DPT ERROR: {e}")

            print(f"  [PCA+KMeans] seed={seed}")
            try:
                nc = min(50, X.shape[1] - 1, X.shape[0] - 1)
                latent = PCA(n_components=nc, random_state=seed).fit_transform(X)
                m = compute_metrics(latent, labels, seed=seed)
                m.update({"method": "PCA+KMeans", "dataset": ds_name, "seed": seed, "train_time_s": 0})
                rows.append(m)
                print(f"    PCA+KMeans: ASW={m['ASW']:.4f} DRE={m.get('DRE',float('nan')):.4f}")
            except Exception as e:
                print(f"    PCA+KMeans ERROR: {e}")

            # Harmony (needs batch column)
            batch_col = DATASETS[ds_name].get("batch_col")
            if batch_col is not None:
                print(f"  [Harmony] seed={seed}")
                try:
                    import harmonypy
                    adata_h = adata.copy()
                    n_pcs = min(50, adata_h.n_vars - 1, adata_h.n_obs - 1)
                    sc.pp.pca(adata_h, n_comps=n_pcs, random_state=seed)
                    ho = harmonypy.run_harmony(adata_h.obsm["X_pca"], adata_h.obs, batch_col, random_state=seed)
                    latent = ho.Z_corr
                    if latent.shape[0] != adata_h.n_obs:
                        latent = latent.T
                    m = compute_metrics(latent, labels, seed=seed)
                    m.update({"method": "Harmony", "dataset": ds_name, "seed": seed, "train_time_s": 0})
                    rows.append(m)
                    print(f"    Harmony: ASW={m['ASW']:.4f} DRE={m.get('DRE',float('nan')):.4f}")
                except Exception as e:
                    print(f"    Harmony ERROR: {e}")

            # Incremental save
            pd.DataFrame([r for r in rows if "error" not in r]).to_csv(csv_path, index=False)

            # scVI / scANVI (slower, GPU)
            if args.all:
                print(f"  [scVI] seed={seed}")
                try:
                    import scvi as scvi_pkg
                    import torch
                    torch.manual_seed(seed)
                    adata_scvi = adata.copy()
                    if "counts" in adata_scvi.layers:
                        adata_scvi.X = adata_scvi.layers["counts"].copy()
                        if issparse(adata_scvi.X):
                            adata_scvi.X = adata_scvi.X.toarray()
                    scvi_pkg.model.SCVI.setup_anndata(adata_scvi, layer=None)
                    model = scvi_pkg.model.SCVI(adata_scvi, n_latent=32, gene_likelihood="nb")
                    model.train(max_epochs=200, early_stopping=True, train_size=0.85)
                    latent = model.get_latent_representation()
                    m = compute_metrics(latent, labels, seed=seed)
                    m.update({"method": "scVI", "dataset": ds_name, "seed": seed, "train_time_s": 0})
                    rows.append(m)
                    print(f"    scVI: ASW={m['ASW']:.4f} DRE={m.get('DRE',float('nan')):.4f}")
                except Exception as e:
                    print(f"    scVI ERROR: {e}")

                print(f"  [scANVI] seed={seed}")
                try:
                    ct_col = DATASETS[ds_name]["cell_type_col"]
                    adata_scanvi = adata.copy()
                    if "counts" in adata_scanvi.layers:
                        adata_scanvi.X = adata_scanvi.layers["counts"].copy()
                        if issparse(adata_scanvi.X):
                            adata_scanvi.X = adata_scanvi.X.toarray()
                    scvi_pkg.model.SCVI.setup_anndata(adata_scanvi, layer=None)
                    scvi_model = scvi_pkg.model.SCVI(adata_scanvi, n_latent=32, gene_likelihood="nb")
                    scvi_model.train(max_epochs=100, early_stopping=True, train_size=0.85)
                    scanvi_model = scvi_pkg.model.SCANVI.from_scvi_model(
                        scvi_model, unlabeled_category="unknown", labels_key=ct_col)
                    scanvi_model.train(max_epochs=100, early_stopping=True, train_size=0.85)
                    latent = scanvi_model.get_latent_representation()
                    m = compute_metrics(latent, labels, seed=seed)
                    m.update({"method": "scANVI", "dataset": ds_name, "seed": seed, "train_time_s": 0})
                    rows.append(m)
                    print(f"    scANVI: ASW={m['ASW']:.4f} DRE={m.get('DRE',float('nan')):.4f}")
                except Exception as e:
                    print(f"    scANVI ERROR: {e}")

                # Unified DL models (CellBLAST, CLEAR, GM-VAE, etc.)
                import importlib, torch as _torch
                try:
                    um = importlib.import_module("external-benchmarker.unified_models")
                except ImportError:
                    um = None
                    print("    external-benchmarker not available, skipping DL models")

                if um is not None:
                    _UNIFIED = {
                        "CellBLAST": ("create_cellblast_model", {"latent_dim": 10, "hidden_dims": [256, 128]}),
                        "CLEAR": ("create_clear_model", {"latent_dim": 128, "hidden_dims": [512, 256]}),
                        "GM-VAE": ("create_gmvae_model", {"latent_dim": 10, "hidden_dims": [512, 256], "distribution": "euclidean"}),
                        "SCALEX": ("create_scalex_model", {"latent_dim": 10, "hidden_dims": [512, 256]}),
                        "scDAC": ("create_scdac_model", {"latent_dim": 32}),
                        "scDeepCluster": ("create_scdeepcluster_model", {"latent_dim": 32, "n_clusters": 10}),
                        "scSMD": ("create_scsmd_model", {"latent_dim": 10, "n_clusters": 10}),
                        "siVAE": ("create_sivae_model", {"latent_dim": 50, "hidden_dims": [512, 256]}),
                    }
                    n_clusters = len(np.unique(labels))
                    device = "cuda" if _torch.cuda.is_available() else "cpu"

                    for model_name, (factory_name, kwargs) in _UNIFIED.items():
                        key = (model_name, ds_name, seed)
                        if key in existing_keys:
                            continue
                        print(f"  [{model_name}] seed={seed}")
                        try:
                            _torch.cuda.empty_cache()
                            np.random.seed(seed)
                            _torch.manual_seed(seed)
                            factory_fn = getattr(um, factory_name)
                            kw = {**kwargs}
                            if "n_clusters" in kw:
                                kw["n_clusters"] = n_clusters
                            model = factory_fn(input_dim=X.shape[1], **kw)

                            X_t = _torch.FloatTensor(X)
                            _NEEDS_TUPLE = {"CLEAR"}
                            as_tuple = model_name in _NEEDS_TUPLE

                            class _Loader:
                                def __init__(self, data, bs=128, tup=False):
                                    self._d, self._bs, self._t = data, bs, tup
                                def __iter__(self):
                                    n = self._d.shape[0]
                                    idx = _torch.randperm(n)
                                    for i in range(0, n, self._bs):
                                        b = self._d[idx[i:i+self._bs]]
                                        yield (b, b) if self._t else b
                                def __len__(self):
                                    return (self._d.shape[0] + self._bs - 1) // self._bs

                            loader = _Loader(X_t, tup=as_tuple)
                            t0 = time.time()
                            model.fit(loader, epochs=200, lr=1e-3, device=device,
                                      patience=25, verbose=0)
                            model.eval()
                            with _torch.no_grad():
                                latent = model.encode(X_t.to(device)).cpu().numpy()
                            elapsed = time.time() - t0
                            m = compute_metrics(latent, labels, seed=seed)
                            m.update({"method": model_name, "dataset": ds_name,
                                      "seed": seed, "train_time_s": round(elapsed, 1)})
                            rows.append(m)
                            print(f"    {model_name}: ASW={m['ASW']:.4f} "
                                  f"DRE={m.get('DRE',float('nan')):.4f} time={elapsed:.0f}s")
                        except Exception as e:
                            print(f"    {model_name} ERROR: {e}")

                pd.DataFrame([r for r in rows if "error" not in r]).to_csv(csv_path, index=False)

            pd.DataFrame([r for r in rows if "error" not in r]).to_csv(csv_path, index=False)

    df = pd.DataFrame([r for r in rows if "error" not in r])
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} rows to {csv_path}")
    print(f"Methods: {sorted(df['method'].unique())}")
    print(f"Datasets: {sorted(df['dataset'].unique())}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
