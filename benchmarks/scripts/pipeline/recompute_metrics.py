#!/usr/bin/env python
"""Recompute metrics from saved latents without retraining.

Reads latents.npz for every config×dataset, calls compute_all_metrics
with the gold-standard evaluators, and overwrites metrics.json +
summary_expanded.csv.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from mocoo.evaluation import compute_all_metrics

DATASETS = ["IRALL", "dentate", "endo", "paul", "spinoids"]
CONFIGS = [
    "VAE",
    "VAE+ODE",
    "VAE+MoCo",
    "VAE+MoCo+Proto",
    "VAE+ODE+MoCo",
    "Full",
]
SPLITS = ["train", "val", "test", "whole"]


def safe_name(cfg: str) -> str:
    return cfg.replace("+", "_")


def recompute_dataset(results_dir: Path, dataset: str) -> None:
    ds_dir = results_dir / dataset
    if not ds_dir.exists():
        print(f"  Skipping {dataset}: directory not found")
        return

    print(f"\n{'=' * 60}")
    print(f"  Dataset: {dataset}")
    print(f"{'=' * 60}")

    for cfg in CONFIGS:
        cfg_dir = ds_dir / safe_name(cfg)
        npz_path = cfg_dir / "latents.npz"
        if not npz_path.exists():
            print(f"  [{cfg}] latents.npz not found, skipping")
            continue

        data = np.load(str(npz_path))

        split_metrics = {}
        for split in SPLITS:
            lat_key = f"{split}_latent"
            lbl_key = f"{split}_labels"
            if lat_key not in data or lbl_key not in data:
                print(f"  [{cfg}] missing {split} split, skipping")
                continue

            latent = data[lat_key]
            labels = data[lbl_key]
            print(f"  [{cfg}] {split}: {latent.shape[0]} cells, {latent.shape[1]} dims ...", end=" ", flush=True)
            t0 = time.time()
            metrics = compute_all_metrics(latent, labels)
            dt = time.time() - t0
            print(f"{dt:.1f}s")

            clean = {
                k: (round(float(v), 6) if isinstance(v, (float, np.floating)) else v)
                for k, v in metrics.items()
                if not k.startswith("_") and not isinstance(v, np.ndarray)
            }
            split_metrics[split] = clean

        # Overwrite metrics.json (preserve resource block if present)
        mpath = cfg_dir / "metrics.json"
        resource = {}
        if mpath.exists():
            with open(mpath) as f:
                old = json.load(f)
            resource = old.get("resource", {})

        metrics_out = {
            "config": cfg,
            "resource": resource,
            "splits": split_metrics,
        }
        with open(mpath, "w") as f:
            json.dump(metrics_out, f, indent=2)

    # Rebuild summary_expanded.csv from all metrics.json files
    expanded_rows = []
    for cfg in CONFIGS:
        mpath = ds_dir / safe_name(cfg) / "metrics.json"
        if not mpath.exists():
            continue
        with open(mpath) as f:
            mdata = json.load(f)
        for split_name, split_m in mdata.get("splits", {}).items():
            erow = {"config": cfg, "split": split_name}
            erow.update(split_m)
            expanded_rows.append(erow)

    if expanded_rows:
        efields = list(expanded_rows[0].keys())
        for r in expanded_rows:
            for k in r:
                if k not in efields:
                    efields.append(k)
        with open(ds_dir / "summary_expanded.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=efields, extrasaction="ignore")
            w.writeheader()
            w.writerows(expanded_rows)
        print(f"  Wrote {ds_dir / 'summary_expanded.csv'}")


def main():
    results_dir = Path(__file__).resolve().parent.parent.parent / "results"

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all",
                        help="Dataset name or 'all'")
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        recompute_dataset(results_dir, ds)

    print("\n✓ All metrics recomputed.")


if __name__ == "__main__":
    main()
