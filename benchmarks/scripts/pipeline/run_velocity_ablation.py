#!/usr/bin/env python3
"""
Velocity consistency ablation: compare Full model with and without velocity loss.
Addresses reviewer Concern 7 (pseudotime circularity).

Trains Full model with ode_reg={0, 0.2, 0.4} across 5 seeds and reports
pseudotime stability (rank correlation between seeds).

Usage (GPU required):
    python benchmarks/scripts/pipeline/run_velocity_ablation.py
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


def load_dataset(path, max_cells=3000, hvg=2000, seed=42):
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


def get_pseudotime(model):
    """Extract pseudotime from a trained MoCoO model."""
    import torch
    model.model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(model._get_input_data()).to(model.device)
        out = model.model.encode(X_tensor)
        if hasattr(out, "pseudotime") and out.pseudotime is not None:
            return out.pseudotime.cpu().numpy().ravel()
        # Fallback: use PC1 of latent
        latent = model.get_latent()
        from sklearn.decomposition import PCA
        pc1 = PCA(n_components=1, random_state=42).fit_transform(latent).ravel()
        return (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-9)


def main():
    parser = argparse.ArgumentParser(description="Velocity consistency ablation")
    parser.add_argument("--data", type=str,
                        default=str(BASE_DIR / "LAB" / "scRL" / "IRALL.h5ad"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024])
    parser.add_argument("--ode_regs", type=float, nargs="+", default=[0.0, 0.2, 0.4])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--outdir", type=str,
                        default=str(_REPO_ROOT / "benchmarks" / "results" / "velocity_ablation"))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    adata = load_dataset(args.data)

    rows = []
    pseudotimes = {}  # key = (ode_reg, seed) -> pseudotime array

    for ode_reg in args.ode_regs:
        for seed in args.seeds:
            print(f"\n  ode_reg={ode_reg}, seed={seed}")
            import torch
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            from mocoo import MoCoO
            from mocoo.evaluation import compute_clustering_metrics

            config = dict(
                use_ode=True, use_moco=True, use_prototype=True,
                n_prototypes=12, vae_reg=0.6, ode_reg=ode_reg,
                moco_weight=0.3, moco_T=0.2, moco_K=4096,
                proto_weight=0.1, random_seed=seed,
            )
            params = {**SHARED, **config}
            model = MoCoO(adata, **params)

            t0 = time.time()
            model.fit(epochs=args.epochs, patience=20, val_every=5)
            train_time = time.time() - t0

            latent = model.get_latent()
            labels = model.labels
            metrics = compute_clustering_metrics(latent, labels, random_state=seed)

            pt = get_pseudotime(model)
            pseudotimes[(ode_reg, seed)] = pt

            row = {
                "ode_reg": ode_reg,
                "seed": seed,
                "train_time_s": round(train_time, 1),
                **metrics,
            }
            rows.append(row)
            print(f"    ARI={metrics['ARI']:.4f}  NMI={metrics['NMI']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "velocity_ablation.csv", index=False)

    # Compute pseudotime rank stability: pairwise Spearman between seeds
    print(f"\n{'='*60}")
    print("PSEUDOTIME RANK STABILITY (pairwise Spearman between seeds)")
    print(f"{'='*60}")

    stability_rows = []
    for ode_reg in args.ode_regs:
        correlations = []
        for i, s1 in enumerate(args.seeds):
            for s2 in args.seeds[i+1:]:
                pt1 = pseudotimes.get((ode_reg, s1))
                pt2 = pseudotimes.get((ode_reg, s2))
                if pt1 is not None and pt2 is not None:
                    n = min(len(pt1), len(pt2))
                    rho, _ = stats.spearmanr(pt1[:n], pt2[:n])
                    correlations.append(rho)

        mean_rho = np.mean(correlations) if correlations else np.nan
        std_rho = np.std(correlations) if correlations else np.nan
        print(f"  ode_reg={ode_reg:.1f}: mean_rho={mean_rho:.4f} +/- {std_rho:.4f}")
        stability_rows.append({
            "ode_reg": ode_reg,
            "mean_spearman": mean_rho,
            "std_spearman": std_rho,
            "n_pairs": len(correlations),
        })

    pd.DataFrame(stability_rows).to_csv(outdir / "pseudotime_stability.csv", index=False)
    print(f"\nResults saved to {outdir}/")


if __name__ == "__main__":
    main()
