#!/usr/bin/env python3
"""
Pseudotime validation against known collection time-points (IRALL d0-d30).
Addresses reviewer Concern #8: velocity consistency circularity.

Trains ODE-containing configs, extracts pseudotime, and computes Spearman
correlation between predicted pseudotime and known collection day.

Usage (GPU required):
    python benchmarks/scripts/evaluation/pseudotime_validation.py
"""
import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

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


def load_dataset(path, max_cells=3000, hvg=3000, seed=42):
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


def extract_batch_day(adata):
    """Extract numeric collection day from IRALL batch labels (e.g., 'd0' -> 0)."""
    batch_labels = adata.obs["batch"].values.astype(str)
    days = []
    for b in batch_labels:
        b = b.strip().lower()
        if b.startswith("d"):
            try:
                days.append(float(b[1:]))
            except ValueError:
                days.append(np.nan)
        else:
            try:
                days.append(float(b))
            except ValueError:
                days.append(np.nan)
    return np.array(days)


def get_pseudotime(model):
    """Extract pseudotime from a trained MoCoO model."""
    # Use ODE-derived pseudotime if available
    if model.use_ode:
        return model.get_pseudotime()
    # Fallback: use PC1 of latent space as proxy pseudotime
    latent = model.get_latent()
    from sklearn.decomposition import PCA
    pc1 = PCA(n_components=1, random_state=42).fit_transform(latent).ravel()
    return (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-9)


def main():
    parser = argparse.ArgumentParser(description="Pseudotime vs. collection day validation")
    parser.add_argument("--data", type=str,
                        default=str(BASE_DIR / "LAB" / "scRL" / "IRALL.h5ad"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--max-cells", type=int, default=3000)
    parser.add_argument("--outdir", type=str,
                        default=str(_REPO_ROOT / "benchmarks" / "results" / "pseudotime_validation"))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []

    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"SEED: {seed}")
        print(f"{'='*60}")

        adata = load_dataset(args.data, max_cells=args.max_cells, seed=seed)
        adata.obs["cell_type"] = adata.obs["cell_type"].values

        # Extract collection day
        days = extract_batch_day(adata)
        valid_mask = ~np.isnan(days)
        n_valid = valid_mask.sum()
        print(f"  Valid day labels: {n_valid}/{len(days)}")

        if n_valid < 100:
            print("  Too few valid day labels, skipping this seed")
            continue

        for cfg_name, cfg in CONFIGS.items():
            print(f"\n  Config: {cfg_name}")
            import torch
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            from mocoo import MoCoO

            params = {**SHARED, **cfg, "random_seed": seed}
            model = MoCoO(adata, **params)

            t0 = time.time()
            model.fit(epochs=args.epochs, patience=30, val_every=5)
            train_time = time.time() - t0

            pt = get_pseudotime(model)

            # Compute correlation between pseudotime and collection day
            pt_valid = pt[valid_mask]
            days_valid = days[valid_mask]

            rho, p_val = stats.spearmanr(pt_valid, days_valid)

            # Also compute per-day mean pseudotime
            unique_days = sorted(np.unique(days_valid))
            day_means = {d: np.mean(pt_valid[days_valid == d]) for d in unique_days}

            # Check monotonicity: are day means increasing?
            means_list = [day_means[d] for d in unique_days]
            monotonic = all(means_list[i] <= means_list[i+1] for i in range(len(means_list)-1))
            # Also check reverse monotonicity
            rev_monotonic = all(means_list[i] >= means_list[i+1] for i in range(len(means_list)-1))

            row = {
                "config": cfg_name,
                "seed": seed,
                "spearman_rho": round(rho, 4),
                "p_value": p_val,
                "abs_rho": round(abs(rho), 4),
                "monotonic": monotonic or rev_monotonic,
                "n_valid": n_valid,
                "train_time_s": round(train_time, 1),
            }
            rows.append(row)

            print(f"    Spearman rho = {rho:.4f}, p = {p_val:.2e}")
            print(f"    Monotonic: {monotonic or rev_monotonic}")
            print(f"    Day means: {', '.join(f'd{d:.0f}={m:.3f}' for d, m in day_means.items())}")

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "pseudotime_validation.csv", index=False)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY (mean |rho| across seeds)")
    print(f"{'='*60}")
    if len(df) > 0:
        summary = df.groupby("config").agg(
            mean_abs_rho=("abs_rho", "mean"),
            std_abs_rho=("abs_rho", "std"),
            mean_rho=("spearman_rho", "mean"),
            pct_monotonic=("monotonic", "mean"),
        ).round(4)
        print(summary.to_string())
        summary.to_csv(outdir / "pseudotime_validation_summary.csv")

    print(f"\nResults saved to {outdir}/")


if __name__ == "__main__":
    main()
