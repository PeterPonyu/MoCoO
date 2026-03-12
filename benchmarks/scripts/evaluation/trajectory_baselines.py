#!/usr/bin/env python3
"""
Trajectory baseline comparison: MoCoO vs DPT vs Palantir.
Addresses W2: Missing trajectory baselines.

Compares pseudotime estimates from:
1. MoCoO (Neural ODE predicted pseudotime)
2. DPT (Diffusion Pseudotime)
3. Palantir

Against the ground truth 'pseudotime' or 'collection_day' column.
Metric: Spearman rank correlation |ρ|.

Usage:
    python benchmarks/scripts/evaluation/trajectory_baselines.py
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

BASE_DIR = Path(os.environ.get("MOCOO_DATA_DIR", str(Path.home())))


DATASETS = {
    "IRALL": {
        "path": str(BASE_DIR / "LAB" / "scRL" / "IRALL.h5ad"),
        "time_col": "pseudotime",
        "cell_type_col": "cell_type",
        "max_cells": 3000,
    },
}

SHARED = dict(
    latent_dim=32, hidden_dim=128, i_dim=4, lr=1e-4,
    batch_size=128, beta=0.1, recon=1.0, loss_mode="nb",
    train_size=0.7, val_size=0.15, test_size=0.15,
)

MOCOO_CONFIGS = {
    "VAE+ODE": dict(use_ode=True, use_moco=False, use_prototype=False,
                    vae_reg=0.6, ode_reg=0.4),
    "VAE+ODE+MoCo": dict(use_ode=True, use_moco=True, use_prototype=False,
                         vae_reg=0.6, ode_reg=0.4,
                         moco_weight=0.3, moco_T=0.2, moco_K=4096),
    "Full": dict(use_ode=True, use_moco=True, use_prototype=True,
                 n_prototypes=12, vae_reg=0.6, ode_reg=0.4,
                 moco_weight=0.3, moco_T=0.2, moco_K=4096,
                 proto_weight=0.1),
}


def load_dataset(path, max_cells, seed=42):
    import scanpy as sc
    from scipy.sparse import issparse

    adata = sc.read_h5ad(path)
    if max_cells and adata.n_obs > max_cells:
        sc.pp.subsample(adata, n_obs=max_cells, random_state=seed)
    sc.pp.filter_genes(adata, min_cells=10)

    if "counts" not in adata.layers:
        X = adata.X.toarray() if issparse(adata.X) else adata.X.copy()
        adata.layers["counts"] = X

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3", layer="counts")
    except Exception:
        sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var["highly_variable"]].copy()
    return adata


def compute_dpt_pseudotime(adata, n_comps=50, n_neighbors=30, seed=42):
    """Compute DPT pseudotime using scanpy."""
    import scanpy as sc

    ad = adata.copy()
    sc.pp.pca(ad, n_comps=min(n_comps, ad.n_vars - 1))
    sc.pp.neighbors(ad, n_neighbors=n_neighbors, random_state=seed)
    sc.tl.diffmap(ad)

    # Pick root cell with earliest ground-truth pseudotime (if available)
    if "pseudotime" in ad.obs.columns:
        root = int(ad.obs["pseudotime"].idxmin() if hasattr(ad.obs["pseudotime"].idxmin(), '__int__') else
                    np.argmin(ad.obs["pseudotime"].values))
    elif "collection_day" in ad.obs.columns:
        root = int(np.argmin(ad.obs["collection_day"].values))
    else:
        root = 0
    ad.uns["iroot"] = root

    sc.tl.dpt(ad)
    return ad.obs["dpt_pseudotime"].values


def compute_palantir_pseudotime(adata, n_comps=50, seed=42):
    """Compute Palantir pseudotime."""
    import palantir
    import scanpy as sc

    ad = adata.copy()
    sc.pp.pca(ad, n_comps=min(n_comps, ad.n_vars - 1))

    # Use diffusion maps
    dm_res = palantir.utils.run_diffusion_maps(ad)

    # Determine multiscale space
    ms_data = palantir.utils.determine_multiscale_space(ad)

    # Pick root cell
    if "pseudotime" in ad.obs.columns:
        root = ad.obs["pseudotime"].idxmin()
    elif "collection_day" in ad.obs.columns:
        root = ad.obs["collection_day"].idxmin()
    else:
        root = ad.obs_names[0]

    pr_res = palantir.core.run_palantir(ad, root, use_early_cell_as_start=True)
    return pr_res.pseudotime.values


def compute_mocoo_pseudotime(adata, config_name, config, seed=42):
    """Train MoCoO and extract ODE pseudotime."""
    import torch
    from mocoo import MoCoO

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    params = {**SHARED, **config, "random_seed": seed}
    model = MoCoO(adata.copy(), **params)
    model.fit(epochs=150, patience=50, val_every=5)

    # Extract pseudotime from ODE
    ptime = model.get_pseudotime()
    return ptime


def spearman_abs(x, y):
    """Absolute Spearman correlation, handling NaN."""
    mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
    if mask.sum() < 10:
        return np.nan
    rho, _ = stats.spearmanr(x[mask], y[mask])
    return abs(rho)


def main():
    outdir = _REPO_ROOT / "benchmarks" / "results" / "trajectory_baselines"
    outdir.mkdir(parents=True, exist_ok=True)

    seeds = [0, 1, 2]
    rows = []

    for ds_name, ds_spec in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"DATASET: {ds_name}")
        print(f"{'='*60}")

        for seed in seeds:
            print(f"\n  --- Seed {seed} ---")
            adata = load_dataset(ds_spec["path"], ds_spec["max_cells"], seed=seed)
            gt = adata.obs[ds_spec["time_col"]].values.astype(float)
            print(f"  Loaded {adata.n_obs} cells, GT range: [{np.nanmin(gt):.3f}, {np.nanmax(gt):.3f}]")

            # DPT
            print("  Computing DPT...", end=" ", flush=True)
            try:
                dpt_pt = compute_dpt_pseudotime(adata, seed=seed)
                rho_dpt = spearman_abs(gt, dpt_pt)
                print(f"|ρ| = {rho_dpt:.4f}")
                rows.append({"dataset": ds_name, "seed": seed, "method": "DPT",
                             "spearman_abs": round(rho_dpt, 4)})
            except Exception as e:
                print(f"ERROR: {e}")
                rows.append({"dataset": ds_name, "seed": seed, "method": "DPT",
                             "spearman_abs": np.nan, "error": str(e)})

            # Palantir
            print("  Computing Palantir...", end=" ", flush=True)
            try:
                pal_pt = compute_palantir_pseudotime(adata, seed=seed)
                rho_pal = spearman_abs(gt, pal_pt)
                print(f"|ρ| = {rho_pal:.4f}")
                rows.append({"dataset": ds_name, "seed": seed, "method": "Palantir",
                             "spearman_abs": round(rho_pal, 4)})
            except Exception as e:
                print(f"ERROR: {e}")
                rows.append({"dataset": ds_name, "seed": seed, "method": "Palantir",
                             "spearman_abs": np.nan, "error": str(e)})

            # MoCoO variants
            for config_name, config in MOCOO_CONFIGS.items():
                print(f"  Computing MoCoO {config_name}...", end=" ", flush=True)
                try:
                    moc_pt = compute_mocoo_pseudotime(adata, config_name, config, seed=seed)
                    rho_moc = spearman_abs(gt, moc_pt)
                    print(f"|ρ| = {rho_moc:.4f}")
                    rows.append({"dataset": ds_name, "seed": seed,
                                 "method": f"MoCoO_{config_name}",
                                 "spearman_abs": round(rho_moc, 4)})
                except Exception as e:
                    print(f"ERROR: {e}")
                    rows.append({"dataset": ds_name, "seed": seed,
                                 "method": f"MoCoO_{config_name}",
                                 "spearman_abs": np.nan, "error": str(e)})

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "trajectory_baselines.csv", index=False)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY (mean ± std |ρ|)")
    print(f"{'='*60}")
    summary = df.groupby("method")["spearman_abs"].agg(["mean", "std", "count"])
    print(summary.to_string())
    summary.to_csv(outdir / "trajectory_summary.csv")

    print(f"\nResults saved to {outdir}/")


if __name__ == "__main__":
    main()
