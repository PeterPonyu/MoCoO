"""Compute ODE gradients for benchmark configs that use ODE.

Trains ODE-enabled configs (VAE+ODE, VAE+ODE+MoCo, Full) and extracts
the ODE vector field gradients dz/dt, then patches them into
benchmark_data.npz so the composed figure can draw Panel L.

Usage:
    python benchmarks/compute_gradients.py
    python benchmarks/compute_gradients.py --epochs 300 --patience 50
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import scanpy as sc

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from mocoo import MoCoO

# Same shared params as run_benchmark.py
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

# Only ODE-enabled configs
ODE_CONFIGS = {
    "VAE+ODE": dict(
        use_ode=True, use_moco=False, use_prototype=False,
        vae_reg=0.5, ode_reg=0.5,
    ),
    "VAE+ODE+MoCo": dict(
        use_ode=True, use_moco=True, use_prototype=False,
        vae_reg=0.5, ode_reg=0.5,
        moco_weight=1.0, moco_T=0.2, moco_K=4096,
    ),
    "Full": dict(
        use_ode=True, use_moco=True, use_prototype=True,
        vae_reg=0.5, ode_reg=0.5,
        moco_weight=1.0, moco_T=0.2, moco_K=4096,
        n_prototypes=12,
    ),
}


def load_dataset(path, max_cells, hvg):
    adata = sc.read_h5ad(path)
    if max_cells and adata.n_obs > max_cells:
        sc.pp.subsample(adata, n_obs=max_cells, random_state=42)
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=hvg,
                                    flavor="seurat_v3", layer="counts")
    except Exception:
        sc.pp.highly_variable_genes(adata, n_top_genes=hvg)
    adata = adata[:, adata.var["highly_variable"]].copy()
    return adata


def _check_epoch_sufficiency(configs, val_losses, val_every, target_cfgs):
    """Heuristic check whether additional epochs are likely useful.

    If the best validation checkpoint occurs in the final 20% of training
    checkpoints, recommend extra epochs for more robust gradient extraction.
    """
    print("\nEpoch sufficiency audit (for robust ODE gradients):")
    recommendations = {}
    for cfg in target_cfgs:
        if cfg not in configs:
            continue
        idx = configs.index(cfg)
        vals = np.asarray(val_losses[idx], dtype=float)
        if vals.size == 0 or not np.isfinite(vals).any():
            recommendations[cfg] = 50
            print(f"  {cfg:<14} : no val history -> recommend +50 epochs")
            continue

        best_ckpt = int(np.nanargmin(vals))
        frac = (best_ckpt + 1) / max(1, vals.size)
        best_epoch = (best_ckpt + 1) * val_every
        total_epoch = vals.size * val_every

        if frac >= 0.8:
            extra = max(50, int(0.25 * total_epoch))
            recommendations[cfg] = extra
            print(f"  {cfg:<14} : best@{best_epoch}/{total_epoch} (late) -> recommend +{extra} epochs")
        else:
            recommendations[cfg] = 0
            print(f"  {cfg:<14} : best@{best_epoch}/{total_epoch} (stable) -> no extra epochs needed")
    return recommendations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(os.environ.get("MOCOO_DATA_DIR", "data"), "IRALL.h5ad"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--max-cells", type=int, default=3000)
    parser.add_argument("--hvg", type=int, default=3000)
    parser.add_argument("--mc-samples", type=int, default=5,
                        help="Monte-Carlo samples for robust ODE gradients")
    parser.add_argument("--extra-epochs", type=int, default=0,
                        help="Force additional epochs for ODE gradient retrain")
    parser.add_argument("--resultsdir", default=None)
    args = parser.parse_args()

    resultsdir = Path(args.resultsdir) if args.resultsdir \
                 else Path(__file__).parent / "results"

    npz_path = resultsdir / "benchmark_data.npz"
    if not npz_path.exists():
        print(f"No benchmark_data.npz in {resultsdir}")
        sys.exit(1)

    # Load existing data
    data = np.load(npz_path, allow_pickle=True)
    configs = list(data["configs"])
    latents = list(data["latents"])
    labels = list(data["labels"])
    val_losses = list(data["val_losses"])
    val_scores = list(data["val_scores"])
    train_losses = list(data["train_losses"])

    # Initialize gradients array (None for non-ODE configs)
    gradients = [None] * len(configs)

    print(f"Loaded {len(configs)} configs: {configs}")
    print(f"ODE configs to compute: {list(ODE_CONFIGS.keys())}")

    epoch_reco = _check_epoch_sufficiency(
        configs, val_losses, args.val_every, list(ODE_CONFIGS.keys()))

    adata = load_dataset(args.data, args.max_cells, args.hvg)
    raw_X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)

    for cfg_name, cfg_params in ODE_CONFIGS.items():
        if cfg_name not in configs:
            print(f"  Skipping {cfg_name} — not in saved configs")
            continue

        idx = configs.index(cfg_name)
        print(f"\n{'─'*60}")
        print(f"Training {cfg_name} for ODE gradients...")
        print(f"{'─'*60}")

        train_epochs = int(args.epochs + args.extra_epochs + epoch_reco.get(cfg_name, 0))
        print(f"  policy: epochs={train_epochs} (base={args.epochs}, extra={args.extra_epochs}, reco={epoch_reco.get(cfg_name, 0)})")
        print(f"  policy: robust gradient = mean of {args.mc_samples} MC samples")

        params = {**SHARED, **cfg_params}
        model = MoCoO(adata, **params)
        model.fit(epochs=train_epochs, patience=args.patience,
                  val_every=args.val_every)

        try:
            if args.mc_samples <= 1:
                grads = model.get_velocity()
                grad_std = np.zeros_like(grads)
            else:
                mc_grads = []
                for _ in range(args.mc_samples):
                    mc_grads.append(model.take_grad(raw_X))
                mc_grads = np.stack(mc_grads, axis=0)
                grads = mc_grads.mean(axis=0)
                grad_std = mc_grads.std(axis=0)

            print(f"  ODE gradients computed: {grads.shape}")
            print(f"  mean MC std (cell x dim): {float(np.mean(grad_std)):.6f}")
            gradients[idx] = grads
        except Exception as e:
            print(f"  ODE gradients failed: {e}")
            gradients[idx] = None

    # Save updated npz — use object array for mixed None/ndarray gradients
    grad_arr = np.empty(len(gradients), dtype=object)
    for i, g in enumerate(gradients):
        grad_arr[i] = g

    np.savez(
        npz_path,
        configs=configs,
        val_losses=np.array(val_losses, dtype=object),
        val_scores=np.array(val_scores, dtype=object),
        train_losses=np.array(train_losses, dtype=object),
        latents=np.array(latents, dtype=object),
        labels=np.array(labels, dtype=object),
        gradients=grad_arr,
        allow_pickle=True,
    )
    print(f"\n✓ Updated {npz_path} with ODE gradients")
    for c, g in zip(configs, gradients):
        status = f"{g.shape}" if g is not None else "None"
        print(f"  {c}: {status}")


if __name__ == "__main__":
    main()
