"""
MoCoO Benchmark: Ablation study across model configurations.

Tests 6 configurations on IRALL (41252 cells, 12 cell types) with
equivalent hyperparameters, subsampled to max_cells for speed.
Outputs per-config metrics JSON and a summary CSV.

Usage:
    python benchmarks/run_benchmark.py
    python benchmarks/run_benchmark.py --epochs 200 --max-cells 2000
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import scanpy as sc
import torch
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from mocoo import MoCoO


# ── Shared hyperparameters (fixed across all variants) ──────────────────────
SHARED = dict(
    latent_dim=32,
    hidden_dim=128,
    i_dim=4,
    lr=1e-4,
    batch_size=128,
    beta=1.0,
    recon=1.0,
    loss_mode="nb",
    random_seed=42,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
)

# ── Configurations to benchmark ────────────────────────────────────────────
CONFIGS = {
    "VAE": dict(
        use_ode=False, use_moco=False, use_prototype=False,
    ),
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
        n_prototypes=12,
        vae_reg=0.6, ode_reg=0.4,
        moco_weight=0.3, moco_T=0.2, moco_K=4096,
        proto_weight=0.1,
    ),
}


def load_dataset(path: str, max_cells: int, hvg: int, seed: int = 42):
    """Load, subsample, HVG-filter, ensure counts layer."""
    adata = sc.read_h5ad(path)
    print(f"Loaded: {adata.shape[0]} cells × {adata.shape[1]} genes")

    if adata.n_obs > max_cells:
        sc.pp.subsample(adata, n_obs=max_cells, random_state=seed)
        print(f"Subsampled → {adata.n_obs} cells")

    sc.pp.filter_genes(adata, min_cells=10)

    if "counts" not in adata.layers:
        from scipy.sparse import issparse
        X = adata.X.toarray() if issparse(adata.X) else adata.X.copy()
        adata.layers["counts"] = X

    # HVG selection
    if adata.n_vars > hvg:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        try:
            sc.pp.highly_variable_genes(adata, n_top_genes=hvg, flavor="seurat_v3",
                                        layer="counts")
        except (ImportError, Exception):
            sc.pp.highly_variable_genes(adata, n_top_genes=hvg)
        adata = adata[:, adata.var["highly_variable"]].copy()
        print(f"HVG filtered → {adata.n_vars} genes")

    return adata


def run_single(name: str, adata, config: dict, epochs: int,
               patience: int, val_every: int, track_metrics: bool = False):
    """Train one configuration, return metrics dict."""
    params = {**SHARED, **config}
    print(f"\n  ── {name} ──")

    model = MoCoO(adata, **params)
    model.fit(epochs=epochs, patience=patience, val_every=val_every,
              track_metrics=track_metrics)

    # ── Collect final metrics on full data ──
    latent = model.get_latent()
    test_latent = model.get_test_latent()

    from sklearn.cluster import KMeans
    from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                                 silhouette_score, calinski_harabasz_score,
                                 davies_bouldin_score)

    labels_all = model.labels
    n_clusters = len(np.unique(labels_all))

    # Full-data metrics
    pred = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(latent)
    full_ari = adjusted_rand_score(labels_all, pred)
    full_nmi = normalized_mutual_info_score(labels_all, pred)
    full_asw = silhouette_score(latent, pred)
    full_ch = calinski_harabasz_score(latent, pred)
    full_db = davies_bouldin_score(latent, pred)

    # Test-set metrics
    labels_test = labels_all[model.test_idx]
    pred_test = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(test_latent)
    test_ari = adjusted_rand_score(labels_test, pred_test)
    test_nmi = normalized_mutual_info_score(labels_test, pred_test)
    test_asw = silhouette_score(test_latent, pred_test)

    # Disentanglement (cross-correlation)
    acorr = np.abs(np.corrcoef(latent.T))
    corr_metric = float(acorr.sum(axis=1).mean()) - 1

    # Resource
    res = model.get_resource_metrics()

    result = {
        "config": name,
        "full_ARI": round(float(full_ari), 4),
        "full_NMI": round(float(full_nmi), 4),
        "full_ASW": round(float(full_asw), 4),
        "full_CH": round(float(full_ch), 2),
        "full_DB": round(float(full_db), 4),
        "test_ARI": round(float(test_ari), 4),
        "test_NMI": round(float(test_nmi), 4),
        "test_ASW": round(float(test_asw), 4),
        "corr": round(float(corr_metric), 4),
        "best_val_loss": round(float(model.best_val_loss), 2),
        "actual_epochs": int(res["actual_epochs"]),
        "train_time_s": round(float(res["train_time"]), 1),
        "peak_mem_gb": round(float(res["peak_memory_gb"]), 2),
    }

    # Histories for plotting
    result["_val_losses"] = model.val_losses
    result["_val_scores"] = model.val_scores
    result["_train_losses"] = model.train_losses
    result["_latent"] = latent
    result["_labels"] = labels_all

    # ODE gradients (vector field) — only for ODE configs
    if config.get("use_ode", False):
        try:
            result["_gradients"] = model.get_velocity()
            print(f"  ODE gradients saved ({result['_gradients'].shape})")
        except Exception as e:
            print(f"  ODE gradients skipped: {e}")
            result["_gradients"] = None
    else:
        result["_gradients"] = None

    return result


def print_summary(results):
    """Print tabular summary."""
    header = f"{'Config':<20} {'ARI':>6} {'NMI':>6} {'ASW':>6} {'CH':>8} {'DB':>6} {'Corr':>6} | {'tARI':>6} {'tNMI':>6} {'tASW':>6} | {'Ep':>4} {'Time':>6} {'Mem':>5}"
    print(f"\n{'='*120}")
    print(header)
    print(f"{'─'*120}")
    for r in results:
        print(f"{r['config']:<20} {r['full_ARI']:>6.3f} {r['full_NMI']:>6.3f} "
              f"{r['full_ASW']:>6.3f} {r['full_CH']:>8.1f} {r['full_DB']:>6.3f} "
              f"{r['corr']:>6.3f} | {r['test_ARI']:>6.3f} {r['test_NMI']:>6.3f} "
              f"{r['test_ASW']:>6.3f} | {r['actual_epochs']:>4d} "
              f"{r['train_time_s']:>6.1f} {r['peak_mem_gb']:>5.2f}")
    print(f"{'='*120}")


def main():
    parser = argparse.ArgumentParser(description="MoCoO ablation benchmark")
    parser.add_argument("--data", default=os.environ.get("MOCOO_DATA_DIR", "data") + "/IRALL.h5ad",
                        help="Path to .h5ad dataset")
    parser.add_argument("--epochs", type=int, default=150,
                        help="Max training epochs per config")
    parser.add_argument("--patience", type=int, default=30,
                        help="Early stopping patience")
    parser.add_argument("--val-every", type=int, default=5,
                        help="Validate every N epochs")
    parser.add_argument("--max-cells", type=int, default=3000,
                        help="Max cells to subsample")
    parser.add_argument("--hvg", type=int, default=3000,
                        help="Number of highly variable genes")
    parser.add_argument("--outdir", default=None,
                        help="Output directory (default: benchmarks/results)")
    parser.add_argument("--configs", nargs="*", default=None,
                        help="Subset of configs to run (default: all)")
    parser.add_argument("--beta", type=float, default=None,
                        help="Override KL beta weight (default: use SHARED value)")
    args = parser.parse_args()

    # Override beta in SHARED if specified
    if args.beta is not None:
        SHARED["beta"] = args.beta
        print(f"\n*** Beta overridden to {args.beta} ***\n")

    outdir = Path(args.outdir) if args.outdir else Path(__file__).parent / "results"
    outdir.mkdir(parents=True, exist_ok=True)

    adata = load_dataset(args.data, args.max_cells, args.hvg)

    configs_to_run = args.configs if args.configs else list(CONFIGS.keys())
    results = []

    for name in configs_to_run:
        if name not in CONFIGS:
            print(f"Unknown config: {name}, skipping")
            continue
        r = run_single(name, adata, CONFIGS[name],
                       args.epochs, args.patience, args.val_every)
        results.append(r)

        # Save per-config (without numpy arrays)
        saveable = {k: v for k, v in r.items() if not k.startswith("_")}
        with open(outdir / f"{name.replace('+', '_')}.json", "w") as f:
            json.dump(saveable, f, indent=2)

    # Save full results for visualization
    # Also save batch labels if available for batch integration metrics
    save_dict = dict(
        configs=np.array([r["config"] for r in results], dtype=object),
        val_losses=np.array([r["_val_losses"] for r in results], dtype=object),
        val_scores=np.array([r["_val_scores"] for r in results], dtype=object),
        train_losses=np.array([r["_train_losses"] for r in results], dtype=object),
        latents=np.array([r["_latent"] for r in results], dtype=object),
        labels=np.array([r["_labels"] for r in results], dtype=object),
        gradients=np.array([r["_gradients"] for r in results], dtype=object),
    )
    if "batch" in adata.obs.columns:
        save_dict["batch_labels"] = adata.obs["batch"].values.astype(str)
    if "cell_type" in adata.obs.columns:
        save_dict["cell_type_labels"] = adata.obs["cell_type"].values.astype(str)
    np.savez(
        outdir / "benchmark_data.npz",
        **save_dict,
        allow_pickle=True,
    )

    # Summary CSV
    import csv
    csv_path = outdir / "summary.csv"
    fields = [k for k in results[0] if not k.startswith("_")]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: v for k, v in r.items() if not k.startswith("_")})

    print_summary(results)
    print(f"\nResults saved to: {outdir}")


if __name__ == "__main__":
    main()
