#!/usr/bin/env python3
"""
Extended external baselines for enriched Figure 3.

Adds unified deep-learning models (CellBLAST, CLEAR, GM-VAE, SCALEX, scDAC,
scDeepCluster, scDHMap, scGCC, scGNN, scSMD, siVAE) and classical ML baselines
(ICA, NMF, FA, TruncatedSVD) on top of the existing methods (scVI, scANVI,
DPT, PCA+KMeans, Harmony).

Runs all on 7 datasets × 5 seeds, producing a single CSV that can be merged
with existing external_baselines.csv.

Usage:
    python benchmarks/scripts/evaluation/run_extended_baselines.py
    python benchmarks/scripts/evaluation/run_extended_baselines.py --datasets IRALL paul --n_seeds 3
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
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────
SKILL_PARENT = str(Path.home() / ".copilot" / "skills")
sys.path.insert(0, SKILL_PARENT)

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
            sc.pp.highly_variable_genes(
                adata, n_top_genes=hvg, flavor="seurat_v3", layer="counts"
            )
        except Exception:
            sc.pp.highly_variable_genes(adata, n_top_genes=hvg)
        adata = adata[:, adata.var["highly_variable"]].copy()

    adata.obs["cell_type"] = adata.obs[spec["cell_type_col"]].values
    return adata


def compute_metrics(latent, labels, seed=42):
    """Compute clustering + embedding-quality metrics on latent embeddings."""
    n_clusters = len(np.unique(labels))
    pred = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(
        latent
    )
    m = {
        "ARI": round(adjusted_rand_score(labels, pred), 4),
        "NMI": round(normalized_mutual_info_score(labels, pred), 4),
        "ASW": round(silhouette_score(latent, pred), 4),
        "CH": round(calinski_harabasz_score(latent, pred), 2),
        "DB": round(davies_bouldin_score(latent, pred), 4),
    }

    # DRE / DREX — embedding quality (only needs latent coordinates + UMAP)
    try:
        from mocoo.evaluation._projections import compute_2d_projections
        from mocoo.evaluation.bench import compute_dre_metrics
        from mocoo.evaluation.drex import compute_drex_metrics

        umap_2d, _ = compute_2d_projections(latent)
        if umap_2d is not None:
            dre = compute_dre_metrics(latent, umap_2d, k=15, prefix="DRE_umap")
            m["DRE"] = round(dre.get("DRE_umap_overall_quality", float("nan")), 4)
            drex = compute_drex_metrics(latent, umap_2d, k=15)
            m["DREX"] = round(drex.get("DREX_overall_quality", float("nan")), 4)
        else:
            m["DRE"] = float("nan")
            m["DREX"] = float("nan")
    except Exception as e:
        print(f"    DRE/DREX computation failed: {e}")
        m["DRE"] = float("nan")
        m["DREX"] = float("nan")

    return m


# ── Unified deep-learning baselines ──────────────────────────────────────

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


def run_unified_model(model_name, X, seed=42, epochs=200, device="cuda", n_clusters=10):
    """Train a unified baseline model and return latent embedding."""
    import importlib
    um = importlib.import_module("external-benchmarker.unified_models")

    factories = {
        "create_cellblast_model": um.create_cellblast_model,
        "create_clear_model": um.create_clear_model,
        "create_gmvae_model": um.create_gmvae_model,
        "create_scalex_model": um.create_scalex_model,
        "create_scdac_model": um.create_scdac_model,
        "create_scdeepcluster_model": um.create_scdeepcluster_model,
        "create_scdhmap_model": um.create_scdhmap_model,
        "create_scgcc_model": um.create_scgcc_model,
        "create_scgnn_model": um.create_scgnn_model,
        "create_scsmd_model": um.create_scsmd_model,
        "create_sivae_model": um.create_sivae_model,
    }

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    factory_name, kwargs = UNIFIED_MODELS[model_name]
    factory_fn = factories[factory_name]
    # Override n_clusters for cluster-based models
    if "n_clusters" in kwargs:
        kwargs = {**kwargs, "n_clusters": n_clusters}
    model = factory_fn(input_dim=X.shape[1], **kwargs)

    X_t = torch.FloatTensor(X)
    # Models with custom _prepare_batch may need tuple batches.
    # Use a tuple-yielding loader for models that expect (x, x_raw) format.
    _NEEDS_TUPLE = {"CLEAR", "scDHMap"}

    class _TensorLoader:
        """Minimal loader that yields tensors or (tensor, tensor) tuples."""
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

    as_tuple = model_name in _NEEDS_TUPLE
    train_loader = _TensorLoader(X_t, batch_size=128, shuffle=True,
                                 as_tuple=as_tuple)

    if not torch.cuda.is_available():
        device = "cpu"

    model.fit(train_loader, epochs=epochs, lr=1e-3, device=device,
              patience=25, verbose=0)

    model.eval()
    with torch.no_grad():
        z = model.encode(X_t.to(device)).cpu().numpy()

    return z


# ── Classical ML baselines ────────────────────────────────────────────────

def run_ml_baselines(X, seed=42):
    """Run all ML decomposition baselines. Returns dict of name -> latent."""
    from sklearn.decomposition import (
        PCA, FastICA, NMF, FactorAnalysis, TruncatedSVD,
    )
    results = {}
    n_components = 10

    # PCA (already in existing baselines but we compute here for completeness)
    try:
        nc = min(n_components, X.shape[1] - 1, X.shape[0] - 1)
        results["PCA"] = PCA(n_components=nc, random_state=seed).fit_transform(X)
    except Exception as e:
        print(f"    PCA ERROR: {e}")

    # ICA
    try:
        nc = min(n_components, X.shape[1] - 1, X.shape[0] - 1)
        results["ICA"] = FastICA(n_components=nc, random_state=seed,
                                 max_iter=500).fit_transform(X)
    except Exception as e:
        print(f"    ICA ERROR: {e}")

    # NMF (needs non-negative input)
    try:
        X_nn = X.copy()
        X_nn[X_nn < 0] = 0
        nc = min(n_components, X_nn.shape[1] - 1, X_nn.shape[0] - 1)
        results["NMF"] = NMF(n_components=nc, random_state=seed,
                             max_iter=500).fit_transform(X_nn)
    except Exception as e:
        print(f"    NMF ERROR: {e}")

    # Factor Analysis
    try:
        nc = min(n_components, X.shape[1] - 1, X.shape[0] - 1)
        results["FA"] = FactorAnalysis(n_components=nc,
                                       random_state=seed).fit_transform(X)
    except Exception as e:
        print(f"    FA ERROR: {e}")

    # Truncated SVD
    try:
        nc = min(n_components, X.shape[1] - 1, X.shape[0] - 1)
        results["TruncatedSVD"] = TruncatedSVD(
            n_components=nc, random_state=seed
        ).fit_transform(X)
    except Exception as e:
        print(f"    TruncatedSVD ERROR: {e}")

    return results


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extended external baselines")
    parser.add_argument(
        "--datasets", nargs="+",
        default=list(DATASETS.keys()),
        choices=list(DATASETS.keys()),
    )
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--hvg", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--ml_only", action="store_true",
        help="Run only ML baselines (fast, no GPU needed)",
    )
    parser.add_argument(
        "--unified_only", action="store_true",
        help="Run only unified DL models",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Specific unified models to run (subset of UNIFIED_MODELS keys)",
    )
    args = parser.parse_args()

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(__file__).resolve().parent.parent.parent / "results" / "baselines"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Determine which unified models to run
    if args.models:
        unified_to_run = [m for m in args.models if m in UNIFIED_MODELS]
    else:
        unified_to_run = list(UNIFIED_MODELS.keys())

    all_rows = []
    csv_path = out_dir / "extended_baselines.csv"

    for ds_name in args.datasets:
        print(f"\n{'='*70}")
        print(f"DATASET: {ds_name}")
        print(f"{'='*70}")

        for seed in range(args.n_seeds):
            adata = load_dataset(ds_name, seed=seed, hvg=args.hvg)
            labels = adata.obs["cell_type"].values
            X = adata.X.toarray() if issparse(adata.X) else np.array(adata.X)
            X = np.nan_to_num(X, nan=0.0)
            n_clusters = len(np.unique(labels))

            # ── ML baselines ──
            if not args.unified_only:
                print(f"\n  [ML baselines] seed={seed}")
                t0 = time.time()
                ml_results = run_ml_baselines(X, seed=seed)
                for name, latent in ml_results.items():
                    m = compute_metrics(latent, labels, seed=seed)
                    m.update({
                        "method": name,
                        "dataset": ds_name,
                        "seed": seed,
                        "train_time_s": round(time.time() - t0, 1),
                    })
                    all_rows.append(m)
                    print(f"    {name}: ARI={m['ARI']:.4f} ASW={m['ASW']:.4f} "
                          f"DRE={m.get('DRE', float('nan')):.4f}")

            # ── Unified DL models ──
            if not args.ml_only:
                for model_name in unified_to_run:
                    print(f"\n  [{model_name}] seed={seed}")
                    try:
                        t0 = time.time()
                        latent = run_unified_model(
                            model_name, X, seed=seed,
                            epochs=args.epochs, device=device,
                            n_clusters=n_clusters,
                        )
                        elapsed = time.time() - t0
                        m = compute_metrics(latent, labels, seed=seed)
                        m.update({
                            "method": model_name,
                            "dataset": ds_name,
                            "seed": seed,
                            "train_time_s": round(elapsed, 1),
                        })
                        all_rows.append(m)
                        print(f"    ARI={m['ARI']:.4f} ASW={m['ASW']:.4f} "
                              f"DRE={m.get('DRE', float('nan')):.4f} time={elapsed:.0f}s")
                    except Exception as e:
                        print(f"    {model_name} ERROR: {e}")
                        all_rows.append({
                            "method": model_name,
                            "dataset": ds_name,
                            "seed": seed,
                            "error": str(e),
                        })

            # Incremental save after each seed
            df_inc = pd.DataFrame([r for r in all_rows if "error" not in r])
            df_inc.to_csv(csv_path, index=False)
            print(f"  [saved {len(df_inc)} rows to {csv_path}]")

    # Final summary
    df = pd.DataFrame([r for r in all_rows if "error" not in r])
    df.to_csv(csv_path, index=False)

    print(f"\n{'='*70}")
    print("SUMMARY (mean ± std across seeds)")
    print(f"{'='*70}")
    if len(df) > 0:
        for ds in sorted(df["dataset"].unique()):
            print(f"\n  {ds}:")
            ds_df = df[df["dataset"] == ds]
            summary = ds_df.groupby("method")[["ARI", "NMI", "ASW", "DRE", "DREX"]].agg(
                ["mean", "std"]
            )
            print(summary.to_string())

    errors = [r for r in all_rows if "error" in r]
    if errors:
        print(f"\n  {len(errors)} errors encountered:")
        for e in errors:
            print(f"    {e['method']} on {e['dataset']} seed={e['seed']}: {e['error'][:80]}")

    print(f"\nResults saved to {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
