"""Compute batch integration metrics from saved benchmark latent embeddings.

Loads benchmark_data.npz, reproduces the exact IRALL subsample (seed=42)
to recover batch labels, then computes iLISI, bASW, and cLISI for each
configuration.  Results are saved to per-config JSON files and a
summary_batch.csv.

Metrics
-------
- **iLISI** (integration LISI): How well batches are mixed in the latent
  space.  Higher = better integration.
- **bASW** (batch-aware ASW): Silhouette score at batch level within each
  cell type.  Higher = better integration.
- **cLISI** (cell-type LISI): How well cell-type structure is preserved
  after integration.  Higher = better bio-conservation.

Usage:
    python benchmarks/scripts/evaluation/compute_batch_metrics.py
    python benchmarks/scripts/evaluation/compute_batch_metrics.py \\
        --data /path/to/IRALL.h5ad \\
        --resultsdir benchmarks/results/dataset_default
"""

import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


def reproduce_subsample(data_path: str, max_cells: int = 3000,
                        seed: int = 42) -> sc.AnnData:
    """Reproduce the exact subsample used by run_benchmark.py."""
    adata = sc.read_h5ad(data_path)
    if adata.n_obs > max_cells:
        sc.pp.subsample(adata, n_obs=max_cells, random_state=seed)
    return adata


def compute_batch_integration(latent: np.ndarray,
                              cell_type_labels: np.ndarray,
                              batch_labels: np.ndarray) -> dict:
    """Compute iLISI, bASW, and cLISI from latent embeddings.

    Uses scib (single-cell integration benchmarking) under the hood
    via a lightweight AnnData wrapper.
    """
    import scib

    # Build an AnnData with the embedding + metadata
    adata = sc.AnnData(
        X=latent.astype(np.float32),
        obs=pd.DataFrame({
            "cell_type": pd.Categorical(cell_type_labels),
            "batch": pd.Categorical(batch_labels),
        }),
    )
    adata.obsm["X_emb"] = latent.astype(np.float32)
    sc.pp.neighbors(adata, use_rep="X_emb", n_neighbors=15)

    metrics = {}

    # iLISI — integration LISI (batch mixing quality)
    try:
        ilisi = scib.metrics.ilisi_graph(
            adata, batch_key="batch", type_="embed",
            use_rep="X_emb", n_cores=1,
        )
        metrics["iLISI"] = round(float(ilisi), 6)
    except Exception as e:
        print(f"    iLISI failed: {e}")
        metrics["iLISI"] = float("nan")

    # bASW — batch-aware silhouette width
    try:
        basw = scib.metrics.silhouette_batch(
            adata, batch_key="batch", group_key="cell_type",
            embed="X_emb",
        )
        metrics["bASW"] = round(float(basw), 6)
    except Exception as e:
        print(f"    bASW failed: {e}")
        metrics["bASW"] = float("nan")

    # cLISI — cell-type LISI (biological conservation)
    try:
        clisi = scib.metrics.clisi_graph(
            adata, label_key="cell_type", type_="embed",
            use_rep="X_emb", n_cores=1,
        )
        metrics["cLISI"] = round(float(clisi), 6)
    except Exception as e:
        print(f"    cLISI failed: {e}")
        metrics["cLISI"] = float("nan")

    # Graph connectivity (bio conservation)
    try:
        gc = scib.metrics.graph_connectivity(
            adata, label_key="cell_type",
        )
        metrics["graph_conn"] = round(float(gc), 6)
    except Exception as e:
        print(f"    graph_conn failed: {e}")
        metrics["graph_conn"] = float("nan")

    # Isolated label silhouette (bio conservation)
    try:
        iso_asw = scib.metrics.isolated_labels_asw(
            adata, label_key="cell_type", batch_key="batch",
            embed="X_emb",
        )
        metrics["iso_label_ASW"] = round(float(iso_asw), 6)
    except Exception as e:
        print(f"    iso_label_ASW failed: {e}")
        metrics["iso_label_ASW"] = float("nan")

    # Overall batch score: 0.4 * bio_conservation + 0.6 * batch_correction
    # (following scIB convention)
    bio = np.nanmean([
        metrics.get("cLISI", np.nan),
        metrics.get("graph_conn", np.nan),
        metrics.get("iso_label_ASW", np.nan),
    ])
    batch = np.nanmean([
        metrics.get("iLISI", np.nan),
        metrics.get("bASW", np.nan),
    ])
    metrics["bio_conservation"] = round(float(bio), 6)
    metrics["batch_correction"] = round(float(batch), 6)
    metrics["overall_score"] = round(0.4 * bio + 0.6 * batch, 6)

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Compute batch integration metrics from saved latents",
    )
    parser.add_argument(
        "--data",
        default="/home/zeyufu/Desktop/datasets/IRALL.h5ad",
        help="Path to original .h5ad (to recover batch labels)",
    )
    parser.add_argument(
        "--resultsdir",
        default=None,
        help="Results directory containing benchmark_data.npz",
    )
    parser.add_argument(
        "--max-cells", type=int, default=3000,
        help="Must match the value used in run_benchmark.py",
    )
    args = parser.parse_args()

    resultsdir = (
        Path(args.resultsdir)
        if args.resultsdir
        else Path(__file__).resolve().parent.parent.parent / "results" / "dataset_default"
    )

    npz_path = resultsdir / "benchmark_data.npz"
    if not npz_path.exists():
        print(f"No benchmark_data.npz in {resultsdir}")
        sys.exit(1)

    # ── Reproduce subsample to recover batch labels ──
    print("Reproducing exact subsample to recover batch labels…")
    adata_sub = reproduce_subsample(args.data, args.max_cells)
    batch_labels_raw = adata_sub.obs["batch"].values.astype(str)
    cell_type_labels_raw = adata_sub.obs["cell_type"].values.astype(str)

    # Verify the cell-type labels match the saved integer labels
    le = LabelEncoder()
    cell_type_int = le.fit_transform(cell_type_labels_raw)
    print(f"Recovered {len(np.unique(cell_type_int))} cell types, "
          f"{len(np.unique(batch_labels_raw))} batches "
          f"from {len(cell_type_int)} cells")

    # ── Load saved latents ──
    print("Loading saved latent embeddings…")
    npz = np.load(npz_path, allow_pickle=True)
    configs = list(npz["configs"])
    latents = list(npz["latents"])
    saved_labels = list(npz["labels"])

    # Quick sanity check
    if not np.array_equal(cell_type_int, saved_labels[0]):
        print("WARNING: Reproduced labels do not match saved labels — "
              "subsample may not be identical.  Continuing anyway.")
    else:
        print("✓ Label verification passed — subsample is deterministic.")

    print(f"Found {len(configs)} configurations: {configs}")

    # ── Compute batch metrics for each config ──
    all_results = []
    for i, (cfg, latent) in enumerate(zip(configs, latents)):
        print(f"\n{'─'*60}")
        print(f"  [{i+1}/{len(configs)}] Batch metrics for: {cfg}")
        print(f"  Latent shape: {latent.shape}")
        print(f"{'─'*60}")

        t0 = time.time()
        batch_metrics = compute_batch_integration(
            latent, cell_type_labels_raw, batch_labels_raw,
        )
        elapsed = time.time() - t0

        batch_metrics["config"] = cfg
        batch_metrics["compute_time_s"] = round(elapsed, 1)

        # Merge into existing per-config JSON
        json_path = resultsdir / f"{cfg.replace('+', '_')}.json"
        if json_path.exists():
            with open(json_path) as f:
                existing = json.load(f)
            existing.update(batch_metrics)
            batch_metrics_full = existing
        else:
            batch_metrics_full = batch_metrics

        # Convert numpy types
        for k, v in batch_metrics_full.items():
            if isinstance(v, (np.floating, np.integer)):
                batch_metrics_full[k] = round(float(v), 6)

        with open(json_path, "w") as f:
            json.dump(batch_metrics_full, f, indent=2)

        all_results.append(batch_metrics)

        print(f"  Done in {elapsed:.1f}s")
        print(f"  iLISI={batch_metrics['iLISI']:.4f}  "
              f"bASW={batch_metrics['bASW']:.4f}  "
              f"cLISI={batch_metrics['cLISI']:.4f}")
        print(f"  bio_conservation={batch_metrics['bio_conservation']:.4f}  "
              f"batch_correction={batch_metrics['batch_correction']:.4f}  "
              f"overall={batch_metrics['overall_score']:.4f}")

    # ── Summary CSV ──
    csv_path = resultsdir / "summary_batch.csv"
    if all_results:
        fields = [k for k in all_results[0].keys()
                  if not isinstance(all_results[0][k], (list, dict))]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in all_results:
                w.writerow({k: v for k, v in r.items()
                            if not isinstance(v, (list, dict))})

    print(f"\n{'='*60}")
    print(f"Batch integration metrics saved to {resultsdir}")
    print(f"Updated {len(all_results)} JSON files + summary_batch.csv")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
