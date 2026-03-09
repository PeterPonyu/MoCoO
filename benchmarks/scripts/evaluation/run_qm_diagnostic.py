"""
Diagnostic: q_z vs q_m evaluation fairness.

Trains all 6 configurations at a single beta and compares metrics computed on
stochastic q_z vs deterministic q_m latent representations.

Usage:
    python benchmarks/scripts/evaluation/run_qm_diagnostic.py
    python benchmarks/scripts/evaluation/run_qm_diagnostic.py --beta 0.1
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import scanpy as sc

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from mocoo import MoCoO
from mocoo.configs import (
    load_config,
    get_shared_params,
    get_model_configs,
    get_training_params,
)
from mocoo.evaluation import compute_clustering_metrics


def load_dataset(path, max_cells, hvg, seed=42):
    from scipy.sparse import issparse

    adata = sc.read_h5ad(path)
    print(f"Loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")
    if adata.n_obs > max_cells:
        sc.pp.subsample(adata, n_obs=max_cells, random_state=seed)
        print(f"Subsampled -> {adata.n_obs} cells")
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
        print(f"HVG filtered -> {adata.n_vars} genes")
    return adata


def main():
    parser = argparse.ArgumentParser(description="q_z vs q_m diagnostic")
    parser.add_argument("--data", default=None, help="Path to .h5ad file")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--max-cells", type=int, default=3000)
    parser.add_argument("--hvg", type=int, default=3000)
    args = parser.parse_args()

    # Find dataset
    data_path = args.data
    if data_path is None:
        data_dir = Path(os.environ.get("MOCOO_DATA_DIR", "/home/zeyufu"))
        data_path = str(data_dir / "LAB" / "scRL" / "IRALL.h5ad")

    adata = load_dataset(data_path, args.max_cells, args.hvg)

    cfg = load_config("beta_ablation")
    SHARED = get_shared_params(cfg)
    CONFIGS = get_model_configs(cfg)
    TRAINING = get_training_params(cfg)

    SHARED["beta"] = args.beta
    epochs = args.epochs
    patience = args.patience
    val_every = TRAINING.get("val_every", 5)

    METRIC_KEYS = ["ARI", "NMI", "ASW", "CAL", "DAV"]
    results = []

    for name, config in CONFIGS.items():
        params = {**SHARED, **config}
        print(f"\n{'='*60}")
        print(f"  Training: {name} (beta={args.beta})")
        print(f"{'='*60}")

        model = MoCoO(adata, **params)
        model.fit(epochs=epochs, patience=patience, val_every=val_every)

        labels = model.labels

        # Extract q_z latent (stochastic, default)
        latent_qz = model.get_latent()
        metrics_qz = compute_clustering_metrics(latent_qz, labels)

        # Extract q_m latent (deterministic posterior mean)
        latent_qm = model.get_latent_qm()
        metrics_qm = compute_clustering_metrics(latent_qm, labels)

        has_ode = config.get("use_ode", False)

        deltas = {}
        for k in METRIC_KEYS:
            qz_val = float(metrics_qz.get(k, 0))
            qm_val = float(metrics_qm.get(k, 0))
            deltas[k] = qm_val - qz_val

        results.append({
            "name": name,
            "has_ode": has_ode,
            "metrics_qz": {k: float(metrics_qz.get(k, 0)) for k in METRIC_KEYS},
            "metrics_qm": {k: float(metrics_qm.get(k, 0)) for k in METRIC_KEYS},
            "deltas": deltas,
        })

    # Print results
    print("\n" + "=" * 100)
    print(f"  DIAGNOSTIC: q_z vs q_m Evaluation (beta={args.beta})")
    print("=" * 100)

    # Table 1: q_z metrics
    print(f"\n{'Config':<20} {'ODE':>4} | {'ARI_qz':>8} {'NMI_qz':>8} {'ASW_qz':>8} {'CH_qz':>8} {'DB_qz':>8}")
    print("-" * 80)
    for r in results:
        m = r["metrics_qz"]
        print(f"{r['name']:<20} {'Y' if r['has_ode'] else 'N':>4} | "
              f"{m['ARI']:>8.4f} {m['NMI']:>8.4f} {m['ASW']:>8.4f} "
              f"{m['CAL']:>8.1f} {m['DAV']:>8.4f}")

    # Table 2: q_m metrics
    print(f"\n{'Config':<20} {'ODE':>4} | {'ARI_qm':>8} {'NMI_qm':>8} {'ASW_qm':>8} {'CH_qm':>8} {'DB_qm':>8}")
    print("-" * 80)
    for r in results:
        m = r["metrics_qm"]
        print(f"{r['name']:<20} {'Y' if r['has_ode'] else 'N':>4} | "
              f"{m['ARI']:>8.4f} {m['NMI']:>8.4f} {m['ASW']:>8.4f} "
              f"{m['CAL']:>8.1f} {m['DAV']:>8.4f}")

    # Table 3: Deltas (q_m - q_z)
    print(f"\n{'Config':<20} {'ODE':>4} | {'dARI':>8} {'dNMI':>8} {'dASW':>8} {'dCH':>8} {'dDB':>8}")
    print("-" * 80)
    for r in results:
        d = r["deltas"]
        print(f"{r['name']:<20} {'Y' if r['has_ode'] else 'N':>4} | "
              f"{d['ARI']:>+8.4f} {d['NMI']:>+8.4f} {d['ASW']:>+8.4f} "
              f"{d['CAL']:>+8.1f} {d['DAV']:>+8.4f}")

    # Summary: average deltas for ODE vs non-ODE
    ode_results = [r for r in results if r["has_ode"]]
    non_ode_results = [r for r in results if not r["has_ode"]]

    print(f"\n{'='*80}")
    print("  SUMMARY: Average delta (q_m - q_z) by model type")
    print(f"{'='*80}")
    for k in ["ARI", "NMI", "ASW"]:
        avg_ode = np.mean([r["deltas"][k] for r in ode_results])
        avg_non = np.mean([r["deltas"][k] for r in non_ode_results])
        diff = avg_non - avg_ode
        print(f"  {k}: ODE avg delta={avg_ode:+.4f}, non-ODE avg delta={avg_non:+.4f}"
              f"  -> asymmetry={diff:+.4f}"
              f"  {'SIGNIFICANT' if abs(diff) > 0.02 else 'small'}")

    print(f"\nIf non-ODE delta >> ODE delta, non-ODE models are penalized by q_z evaluation.")
    print(f"Recommendation: use q_m if asymmetry > 0.02 on ARI.")


import os

if __name__ == "__main__":
    main()
