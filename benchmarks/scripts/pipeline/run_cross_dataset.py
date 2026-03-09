"""Cross-dataset benchmark — run MoCoO ablation on multiple datasets.

For each registered dataset, trains all 6 configurations (or a subset)
and saves per-config metrics + latent embeddings to
``results/<dataset_name>/``.  Also produces a meta-analysis summary
aggregating results across all datasets.

Usage:
    python benchmarks/scripts/pipeline/run_cross_dataset.py
    python benchmarks/scripts/pipeline/run_cross_dataset.py \\
        --datasets IRALL dentate endo \\
        --configs VAE Full \\
        --epochs 100
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from mocoo import MoCoO
from mocoo.configs import load_config, get_shared_params, get_model_configs
from mocoo.evaluation import compute_clustering_metrics
from benchmarks.scripts.pipeline.dataset_registry import get_registry
from benchmarks.scripts.evaluation.compute_batch_metrics import (
    compute_batch_integration,
)

# Load config definitions from centralized YAML
_cfg = load_config("default")
SHARED = get_shared_params(_cfg)
CONFIGS = get_model_configs(_cfg)


def run_single(name: str, adata, config: dict, epochs: int,
               patience: int, val_every: int, track_metrics: bool = False):
    """Train one configuration, return metrics dict + latent."""
    params = {**SHARED, **config}
    model = MoCoO(adata, **params)
    model.fit(epochs=epochs, patience=patience, val_every=val_every,
              track_metrics=track_metrics)

    latent = model.get_latent()
    labels_all = model.labels

    clustering = compute_clustering_metrics(latent, labels_all)
    result = {
        "config": name,
        "ARI": round(clustering["ARI"], 4),
        "NMI": round(clustering["NMI"], 4),
        "ASW": round(clustering["ASW"], 4),
        "CH": round(float(clustering["CAL"]), 2),
        "DB": round(clustering["DAV"], 4),
        "actual_epochs": model.actual_epochs,
        "train_time_s": round(model.train_time, 1),
    }

    # ODE gradients
    gradients = None
    if config.get("use_ode", False):
        try:
            gradients = model.get_velocity()
        except Exception:
            pass

    return result, latent, labels_all, gradients, model


def run_dataset(dataset_name: str, configs_to_run: list[str],
                epochs: int, patience: int, val_every: int,
                outdir: Path, max_cells: int, hvg: int):
    """Run all configs on one dataset, save results."""
    reg = get_registry()
    adata, meta = reg.load(dataset_name, max_cells=max_cells, hvg=hvg)

    ds_outdir = outdir / dataset_name
    ds_outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*70}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Cells: {meta['cells']} | Types: {meta['n_cell_types']} | "
          f"Batches: {meta['n_batches']} | Genes: {meta['genes']}")
    print(f"{'═'*70}")

    results = []
    all_latents = []
    all_labels = []
    all_gradients = []

    for cfg_name in configs_to_run:
        if cfg_name not in CONFIGS:
            print(f"  Unknown config: {cfg_name}, skipping")
            continue

        print(f"\n  ── {cfg_name} ──")
        t0 = time.time()
        try:
            r, latent, labels, grads, model = run_single(
                cfg_name, adata, CONFIGS[cfg_name],
                epochs, patience, val_every,
            )
            r["dataset"] = dataset_name
            elapsed = time.time() - t0

            # Batch integration metrics if batch labels available
            if meta["has_batch"] and meta["n_batches"] > 1:
                batch_labels = adata.obs["batch"].astype(str).values
                cell_type_labels = adata.obs["cell_type"].astype(str).values
                try:
                    batch_m = compute_batch_integration(
                        latent, cell_type_labels, batch_labels,
                    )
                    r.update(batch_m)
                except Exception as e:
                    print(f"    Batch metrics failed: {e}")

            results.append(r)
            all_latents.append(latent)
            all_labels.append(labels)
            all_gradients.append(grads)

            # Save per-config JSON (ensure native Python types for JSON)
            r_json = {k: (float(v) if hasattr(v, 'item') else v) for k, v in r.items()}
            with open(ds_outdir / f"{cfg_name.replace('+', '_')}.json", "w") as f:
                json.dump(r_json, f, indent=2)

            print(f"    ARI={r['ARI']:.3f} NMI={r['NMI']:.3f} "
                  f"ASW={r['ASW']:.3f} ({elapsed:.1f}s)")
            if "overall_score" in r:
                print(f"    Batch: iLISI={r.get('iLISI','?'):.3f} "
                      f"bASW={r.get('bASW','?'):.3f} "
                      f"overall={r.get('overall_score','?'):.3f}")

        except Exception as e:
            print(f"    FAILED: {e}")
            import traceback; traceback.print_exc()

    # Save combined results
    if results:
        np.savez(
            ds_outdir / "benchmark_data.npz",
            configs=np.array([r["config"] for r in results]),
            latents=np.array(all_latents, dtype=object),
            labels=np.array(all_labels, dtype=object),
            gradients=np.array(all_gradients, dtype=object),
            allow_pickle=True,
        )

        # CSV
        fields = [k for k in results[0] if not isinstance(results[0][k], (list, dict))]
        with open(ds_outdir / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in results:
                w.writerow({k: v for k, v in r.items()
                            if not isinstance(v, (list, dict))})

    return results, meta


def build_meta_analysis(all_results: list[dict], outdir: Path):
    """Create a cross-dataset meta-analysis summary."""
    meta_path = outdir / "meta_analysis.csv"
    if not all_results:
        return

    fields = sorted({k for r in all_results for k in r.keys()
                     if not isinstance(r.get(k), (list, dict))})
    with open(meta_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in all_results:
            row = {k: v for k, v in r.items()
                   if not isinstance(v, (list, dict))}
            w.writerow(row)

    # Aggregate: mean ± std per config across datasets
    from collections import defaultdict
    by_config = defaultdict(list)
    for r in all_results:
        by_config[r["config"]].append(r)

    print(f"\n{'═'*70}")
    print(f"  META-ANALYSIS SUMMARY")
    print(f"{'═'*70}")
    print(f"{'Config':<20} {'ARI':>10} {'NMI':>10} {'ASW':>10}")
    print(f"{'─'*70}")
    for cfg, runs in by_config.items():
        aris = [r["ARI"] for r in runs]
        nmis = [r["NMI"] for r in runs]
        asws = [r["ASW"] for r in runs]
        print(f"{cfg:<20} {np.mean(aris):>5.3f}±{np.std(aris):.3f} "
              f"{np.mean(nmis):>5.3f}±{np.std(nmis):.3f} "
              f"{np.mean(asws):>5.3f}±{np.std(asws):.3f}")
    print(f"{'═'*70}")
    print(f"Meta-analysis saved to {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset MoCoO benchmark")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Datasets to run (default: all registered)")
    parser.add_argument("--configs", nargs="*", default=None,
                        help="Configs to run (default: all)")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--max-cells", type=int, default=3000)
    parser.add_argument("--hvg", type=int, default=3000)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    reg = get_registry()
    datasets = args.datasets if args.datasets else reg.list()
    configs = args.configs if args.configs else list(CONFIGS.keys())
    outdir = Path(args.outdir) if args.outdir else (
        Path(__file__).resolve().parent.parent.parent / "results"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Cross-dataset benchmark")
    print(f"  Datasets: {datasets}")
    print(f"  Configs: {configs}")
    print(f"  Output: {outdir}")

    all_results = []
    for ds_name in datasets:
        try:
            results, meta = run_dataset(
                ds_name, configs, args.epochs, args.patience,
                args.val_every, outdir, args.max_cells, args.hvg,
            )
            all_results.extend(results)
        except Exception as e:
            print(f"\nFAILED on dataset {ds_name}: {e}")
            import traceback; traceback.print_exc()

    build_meta_analysis(all_results, outdir)


if __name__ == "__main__":
    main()
