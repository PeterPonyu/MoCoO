#!/usr/bin/env python3
"""
Extended multi-seed benchmark: computes ALL subcategory metrics (32 total)
including distance-preservation subcategories (DRE, DREX, LSE, LSEX).

This script extends run_multiseed.py by saving latent embeddings and
computing the full metric suite from mocoo.evaluation.bench, enabling
statistical validation of the Full model's distance-preservation advantage
identified in the subcategory analysis (Tables XII and XIII in the paper).

Usage (GPU required):
    python run_multiseed_extended.py --seeds 5 --datasets IRALL
    python run_multiseed_extended.py --seeds 10 --datasets IRALL --resume

Output:
    benchmarks/results/multiseed/multiseed_extended_{dataset}.csv
        — All 32+ subcategory metrics per config × seed
"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "benchmarks" / "scripts" / "pipeline"))

from run_multiseed import SHARED, CONFIGS, DATASET_SPECS, BASE_DIR, load_dataset


def train_and_evaluate_extended(adata, config_name, config, seed, epochs, patience, val_every):
    """Train one config with one seed, return full subcategory metrics dict."""
    import torch
    from mocoo import MoCoO
    from mocoo.evaluation.bench import compute_all_metrics
    from sklearn.cluster import KMeans

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    params = {**SHARED, **config, "random_seed": seed}
    model = MoCoO(adata, **params)

    t0 = time.time()
    model.fit(epochs=epochs, patience=patience, val_every=val_every)
    train_time = time.time() - t0

    latent = model.get_latent()
    labels_all = model.labels
    n_clusters = len(np.unique(labels_all))
    pred = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(latent)

    # Compute ALL subcategory metrics via the unified evaluation suite
    all_metrics = compute_all_metrics(latent, labels_all, dre_k=15)

    # Flatten: prefix keys with category for clarity
    row = {
        "config": config_name,
        "seed": seed,
        "train_time_s": round(train_time, 1),
    }
    for k, v in all_metrics.items():
        if isinstance(v, (int, float, np.floating, np.integer)):
            row[k] = round(float(v), 6)

    return row


def main():
    parser = argparse.ArgumentParser(description="Extended multi-seed MoCoO benchmark (all subcategory metrics)")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--datasets", nargs="+", default=["IRALL"],
                        choices=list(DATASET_SPECS.keys()))
    parser.add_argument("--configs", nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-cells", type=int, default=3000)
    parser.add_argument("--hvg", type=int, default=3000)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--val_every", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    configs_to_run = args.configs or list(CONFIGS.keys())
    if args.beta is not None:
        SHARED["beta"] = args.beta
    out_dir = Path(args.output_dir) if args.output_dir else \
        _REPO_ROOT / "benchmarks" / "results" / "multiseed"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds_name in args.datasets:
        ds_spec = DATASET_SPECS[ds_name]
        epochs = args.epochs or ds_spec["epochs_default"]
        out_csv = out_dir / f"multiseed_extended_{ds_name}.csv"

        existing = set()
        if args.resume and out_csv.exists():
            prev = pd.read_csv(out_csv)
            existing = set(zip(prev["config"], prev["seed"]))
            print(f"Resuming: {len(existing)} runs already completed for {ds_name}")

        print(f"\n{'='*70}")
        print(f"EXTENDED MULTI-SEED: {ds_name} | epochs={epochs} | seeds={args.seeds}")
        print(f"{'='*70}")

        total = len(configs_to_run) * args.seeds
        done = 0

        for config_name in configs_to_run:
            for seed in range(args.seeds):
                done += 1
                if (config_name, seed) in existing:
                    print(f"  [{done}/{total}] {config_name} seed={seed} — SKIPPED")
                    continue

                print(f"\n  [{done}/{total}] {config_name} seed={seed}")

                adata = load_dataset(ds_spec["path"], args.max_cells, args.hvg, seed=seed)
                adata.obs["cell_type"] = adata.obs[ds_spec["cell_type_col"]].values

                try:
                    metrics = train_and_evaluate_extended(
                        adata, config_name, CONFIGS[config_name],
                        seed, epochs, args.patience, args.val_every
                    )
                    metrics["dataset"] = ds_name

                    # Print key distance-preservation metrics
                    dp_keys = [k for k in metrics if "distance" in k.lower() or "spearman" in k.lower() or "pearson" in k.lower()]
                    dp_str = "  ".join(f"{k}={metrics[k]:.4f}" for k in dp_keys[:4])
                    print(f"    {dp_str}  time={metrics['train_time_s']:.0f}s")

                    # Incremental save
                    df_new = pd.DataFrame([metrics])
                    if out_csv.exists():
                        df_new.to_csv(out_csv, mode="a", header=False, index=False)
                    else:
                        df_new.to_csv(out_csv, index=False)

                except Exception as e:
                    print(f"    ERROR: {e}")

        # Summary: show distance-preservation metrics
        if out_csv.exists():
            df = pd.read_csv(out_csv)
            dp_cols = [c for c in df.columns if "distance" in c.lower() or "spearman" in c.lower() or "pearson" in c.lower()]
            if dp_cols:
                print(f"\n{'='*70}")
                print(f"DISTANCE PRESERVATION SUMMARY — {ds_name}")
                print(f"{'='*70}")
                summary = df.groupby("config")[dp_cols].agg(["mean", "std"])
                print(summary.to_string())


if __name__ == "__main__":
    main()
