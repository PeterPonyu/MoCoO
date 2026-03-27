#!/usr/bin/env python3
"""
External baselines for MoCoO benchmark comparison.
Addresses Critical Concern: lack of external baselines (scVI, DPT).

Runs scVI (VAE baseline) and scanpy Diffusion Pseudotime (DPT) on the same
preprocessed data used by MoCoO, using identical evaluation metrics.

Usage:
    python benchmarks/scripts/evaluation/external_baselines.py
    python benchmarks/scripts/evaluation/external_baselines.py --datasets IRALL paul --n_seeds 5
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
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

warnings.filterwarnings("ignore")

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
    """Load and preprocess dataset identically to MoCoO."""
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


# ── scVI Baseline ──────────────────────────────────────────────────────────


def run_scvi(adata, seed=42, n_latent=32, max_epochs=200):
    """Run scVI and return latent + metrics."""
    import scvi as scvi_pkg

    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    adata_scvi = adata.copy()

    # scVI needs raw counts
    if "counts" in adata_scvi.layers:
        adata_scvi.X = adata_scvi.layers["counts"].copy()
        if issparse(adata_scvi.X):
            adata_scvi.X = adata_scvi.X.toarray()

    scvi_pkg.model.SCVI.setup_anndata(adata_scvi, layer=None)
    model = scvi_pkg.model.SCVI(
        adata_scvi, n_latent=n_latent, gene_likelihood="nb"
    )
    model.train(max_epochs=max_epochs, early_stopping=True, train_size=0.85)

    latent = model.get_latent_representation()
    return latent


# ── DPT (Diffusion Pseudotime) Baseline ───────────────────────────────────


def run_dpt(adata, seed=42, n_pcs=50, n_neighbors=15):
    """Run scanpy DPT pipeline and return PCA latent + pseudotime."""
    np.random.seed(seed)

    adata_dpt = adata.copy()

    # PCA
    n_pcs_actual = min(n_pcs, adata_dpt.n_vars - 1, adata_dpt.n_obs - 1)
    sc.pp.pca(adata_dpt, n_comps=n_pcs_actual, random_state=seed)
    sc.pp.neighbors(adata_dpt, n_neighbors=n_neighbors, n_pcs=n_pcs_actual)

    # Diffusion map
    sc.tl.diffmap(adata_dpt, n_comps=15)

    # DPT needs a root cell -- pick a random cell from the most common type
    cell_types = adata_dpt.obs["cell_type"].values
    most_common = pd.Series(cell_types).value_counts().index[0]
    root_candidates = np.where(cell_types == most_common)[0]
    adata_dpt.uns["iroot"] = root_candidates[seed % len(root_candidates)]

    sc.tl.dpt(adata_dpt)

    # Use diffusion map embedding as latent space for clustering eval
    latent = adata_dpt.obsm["X_diffmap"][:, :15]
    pseudotime = adata_dpt.obs["dpt_pseudotime"].values

    return latent, pseudotime


# ── Harmony Baseline (batch correction) ───────────────────────────────────


def run_harmony(adata, seed=42, n_pcs=50, batch_col="batch"):
    """Run Harmony integration and return corrected PCA."""
    import harmonypy

    np.random.seed(seed)

    adata_h = adata.copy()
    n_pcs_actual = min(n_pcs, adata_h.n_vars - 1, adata_h.n_obs - 1)
    sc.pp.pca(adata_h, n_comps=n_pcs_actual, random_state=seed)

    ho = harmonypy.run_harmony(
        adata_h.obsm["X_pca"], adata_h.obs, batch_col, random_state=seed
    )
    # ho.Z_corr shape is (n_cells, n_pcs) in harmonypy >= 0.0.5
    latent = ho.Z_corr
    if latent.shape[0] != adata_h.n_obs:
        latent = latent.T  # fallback for older harmonypy where shape is (n_pcs, n_cells)
    return latent


# ── scANVI Baseline (semi-supervised) ─────────────────────────────────────


def run_scanvi(adata, seed=42, n_latent=32, max_epochs=200, cell_type_col="cell_type"):
    """Run scANVI (semi-supervised extension of scVI) and return latent."""
    import scvi as scvi_pkg
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    adata_scanvi = adata.copy()

    if "counts" in adata_scanvi.layers:
        adata_scanvi.X = adata_scanvi.layers["counts"].copy()
        if issparse(adata_scanvi.X):
            adata_scanvi.X = adata_scanvi.X.toarray()

    scvi_pkg.model.SCVI.setup_anndata(adata_scanvi, layer=None)
    scvi_model = scvi_pkg.model.SCVI(adata_scanvi, n_latent=n_latent, gene_likelihood="nb")
    scvi_model.train(max_epochs=max_epochs // 2, early_stopping=True, train_size=0.85)

    scanvi_model = scvi_pkg.model.SCANVI.from_scvi_model(
        scvi_model, unlabeled_category="unknown",
        labels_key=cell_type_col,
    )
    scanvi_model.train(max_epochs=max_epochs // 2, early_stopping=True, train_size=0.85)

    latent = scanvi_model.get_latent_representation()
    return latent


def main():
    parser = argparse.ArgumentParser(description="External baselines for MoCoO")
    parser.add_argument(
        "--datasets", nargs="+", default=["IRALL"], choices=list(DATASETS.keys())
    )
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--hvg", type=int, default=3000)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["scVI", "DPT", "PCA+KMeans", "Harmony", "scANVI"],
        help="Methods to run",
    )
    args = parser.parse_args()

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(__file__).resolve().parent.parent.parent / "results" / "baselines"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for ds_name in args.datasets:
        print(f"\n{'='*70}")
        print(f"DATASET: {ds_name}")
        print(f"{'='*70}")

        for seed in range(args.n_seeds):
            adata = load_dataset(ds_name, seed=seed, hvg=args.hvg)
            labels = adata.obs["cell_type"].values

            # ── scVI ──
            if "scVI" in args.methods:
                print(f"\n  [scVI] seed={seed}")
                try:
                    t0 = time.time()
                    latent = run_scvi(adata, seed=seed)
                    elapsed = time.time() - t0
                    m = compute_metrics(latent, labels, seed=seed)
                    m.update(
                        {
                            "method": "scVI",
                            "dataset": ds_name,
                            "seed": seed,
                            "train_time_s": round(elapsed, 1),
                        }
                    )
                    all_rows.append(m)
                    print(
                        f"    ARI={m['ARI']:.4f}  NMI={m['NMI']:.4f}  "
                        f"ASW={m['ASW']:.4f}  time={elapsed:.0f}s"
                    )
                except Exception as e:
                    print(f"    scVI ERROR: {e}")
                    all_rows.append(
                        {
                            "method": "scVI",
                            "dataset": ds_name,
                            "seed": seed,
                            "error": str(e),
                        }
                    )

            # ── DPT ──
            if "DPT" in args.methods:
                print(f"\n  [DPT] seed={seed}")
                try:
                    t0 = time.time()
                    latent, pseudotime = run_dpt(adata, seed=seed)
                    elapsed = time.time() - t0
                    m = compute_metrics(latent, labels, seed=seed)
                    m.update(
                        {
                            "method": "DPT",
                            "dataset": ds_name,
                            "seed": seed,
                            "train_time_s": round(elapsed, 1),
                        }
                    )
                    all_rows.append(m)
                    print(
                        f"    ARI={m['ARI']:.4f}  NMI={m['NMI']:.4f}  "
                        f"ASW={m['ASW']:.4f}  time={elapsed:.0f}s"
                    )
                except Exception as e:
                    print(f"    DPT ERROR: {e}")
                    all_rows.append(
                        {
                            "method": "DPT",
                            "dataset": ds_name,
                            "seed": seed,
                            "error": str(e),
                        }
                    )

            # ── PCA+KMeans ──
            if "PCA+KMeans" in args.methods:
                print(f"\n  [PCA+KMeans] seed={seed}")
                try:
                    from sklearn.decomposition import PCA

                    t0 = time.time()
                    X = adata.X.toarray() if issparse(adata.X) else adata.X.copy()
                    X = np.nan_to_num(X, nan=0.0)
                    nc = min(50, X.shape[1] - 1, X.shape[0] - 1)
                    pca = PCA(n_components=nc, random_state=seed)
                    latent = pca.fit_transform(X)
                    elapsed = time.time() - t0
                    m = compute_metrics(latent, labels, seed=seed)
                    m.update(
                        {
                            "method": "PCA+KMeans",
                            "dataset": ds_name,
                            "seed": seed,
                            "train_time_s": round(elapsed, 1),
                        }
                    )
                    all_rows.append(m)
                    print(
                        f"    ARI={m['ARI']:.4f}  NMI={m['NMI']:.4f}  "
                        f"ASW={m['ASW']:.4f}  time={elapsed:.0f}s"
                    )
                except Exception as e:
                    print(f"    PCA+KMeans ERROR: {e}")

            # ── Harmony ──
            if "Harmony" in args.methods:
                spec = DATASETS[ds_name]
                batch_col = spec.get("batch_col")
                if batch_col is not None:
                    print(f"\n  [Harmony] seed={seed}")
                    try:
                        t0 = time.time()
                        latent = run_harmony(adata, seed=seed, batch_col=batch_col)
                        elapsed = time.time() - t0
                        m = compute_metrics(latent, labels, seed=seed)
                        m.update({
                            "method": "Harmony",
                            "dataset": ds_name,
                            "seed": seed,
                            "train_time_s": round(elapsed, 1),
                        })
                        all_rows.append(m)
                        print(f"    ARI={m['ARI']:.4f}  NMI={m['NMI']:.4f}  "
                              f"ASW={m['ASW']:.4f}  time={elapsed:.0f}s")
                    except Exception as e:
                        print(f"    Harmony ERROR: {e}")
                        all_rows.append({
                            "method": "Harmony", "dataset": ds_name,
                            "seed": seed, "error": str(e),
                        })
                else:
                    print(f"\n  [Harmony] skipped (no batch column for {ds_name})")

            # ── scANVI ──
            if "scANVI" in args.methods:
                print(f"\n  [scANVI] seed={seed}")
                try:
                    t0 = time.time()
                    latent = run_scanvi(adata, seed=seed,
                                        cell_type_col=DATASETS[ds_name]["cell_type_col"])
                    elapsed = time.time() - t0
                    m = compute_metrics(latent, labels, seed=seed)
                    m.update({
                        "method": "scANVI",
                        "dataset": ds_name,
                        "seed": seed,
                        "train_time_s": round(elapsed, 1),
                    })
                    all_rows.append(m)
                    print(f"    ARI={m['ARI']:.4f}  NMI={m['NMI']:.4f}  "
                          f"ASW={m['ASW']:.4f}  time={elapsed:.0f}s")
                except Exception as e:
                    print(f"    scANVI ERROR: {e}")
                    all_rows.append({
                        "method": "scANVI", "dataset": ds_name,
                        "seed": seed, "error": str(e),
                    })

    # Save results
    df = pd.DataFrame([r for r in all_rows if "error" not in r])
    csv_path = out_dir / "external_baselines.csv"
    df.to_csv(csv_path, index=False)

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY (mean +/- std across seeds)")
    print(f"{'='*70}")
    if len(df) > 0:
        for ds in df["dataset"].unique():
            print(f"\n  {ds}:")
            ds_df = df[df["dataset"] == ds]
            summary = ds_df.groupby("method")[["ARI", "NMI", "ASW"]].agg(
                ["mean", "std"]
            )
            print(summary.to_string())

    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
