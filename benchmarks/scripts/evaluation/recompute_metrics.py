"""Recompute expanded metrics from saved benchmark latent embeddings.

Loads benchmark_data.npz, computes the full PanODE-LAB metric battery
(clustering + DRE + LSE + DREX + LSEX + diagnostics), and updates
the per-config JSON files and summary CSV.

Usage:
    python benchmarks/recompute_metrics.py
    python benchmarks/recompute_metrics.py --resultsdir benchmarks/results
"""

import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_expanded import compute_all_metrics


def main():
    parser = argparse.ArgumentParser(description="Recompute expanded metrics")
    parser.add_argument("--resultsdir", default=None)
    args = parser.parse_args()

    resultsdir = Path(args.resultsdir) if args.resultsdir else Path(__file__).parent / "results"

    npz_path = resultsdir / "benchmark_data.npz"
    if not npz_path.exists():
        print(f"No benchmark_data.npz in {resultsdir}")
        sys.exit(1)

    print("Loading saved latent embeddings...")
    npz = np.load(npz_path, allow_pickle=True)
    configs = list(npz["configs"])
    latents = list(npz["latents"])
    labels_all = list(npz["labels"])

    print(f"Found {len(configs)} configurations: {configs}")

    all_results = []

    for i, (cfg, latent, labels) in enumerate(zip(configs, latents, labels_all)):
        print(f"\n{'─'*60}")
        print(f"  [{i+1}/{len(configs)}] Computing metrics for: {cfg}")
        print(f"  Latent shape: {latent.shape}, unique labels: {len(np.unique(labels))}")
        print(f"{'─'*60}")

        t0 = time.time()
        metrics = compute_all_metrics(latent, labels, dre_k=15)
        elapsed = time.time() - t0

        # Remove internal keys
        save_metrics = {k: v for k, v in metrics.items()
                        if not k.startswith("_")}

        # Load existing JSON to preserve resource data
        json_path = resultsdir / f"{cfg.replace('+', '_')}.json"
        if json_path.exists():
            with open(json_path) as f:
                existing = json.load(f)
            # Preserve resource/training metadata
            for k in ('config', 'best_val_loss', 'actual_epochs',
                      'train_time_s', 'peak_mem_gb',
                      'test_ARI', 'test_NMI', 'test_ASW'):
                if k in existing:
                    save_metrics[k] = existing[k]

        save_metrics['config'] = cfg
        save_metrics['metrics_compute_time_s'] = round(elapsed, 1)

        # Convert numpy types
        for k, v in save_metrics.items():
            if isinstance(v, (np.floating, np.integer)):
                save_metrics[k] = round(float(v), 6)
            elif isinstance(v, np.ndarray):
                save_metrics[k] = v.tolist()

        with open(json_path, "w") as f:
            json.dump(save_metrics, f, indent=2)

        all_results.append(save_metrics)
        print(f"  Done in {elapsed:.1f}s — saved {json_path.name}")

        # Print key metrics
        print(f"  NMI={save_metrics.get('NMI', '?'):.4f}  "
              f"ARI={save_metrics.get('ARI', '?'):.4f}  "
              f"ASW={save_metrics.get('ASW', '?'):.4f}  "
              f"DAV={save_metrics.get('DAV', '?'):.4f}")
        dre_umap = save_metrics.get('DRE_umap_overall_quality', np.nan)
        lse = save_metrics.get('LSE_overall_quality', np.nan)
        drex = save_metrics.get('DREX_overall_quality', np.nan)
        lsex = save_metrics.get('LSEX_overall_quality', np.nan)
        print(f"  DRE_UMAP={dre_umap:.4f}  LSE={lse:.4f}  "
              f"DREX={drex:.4f}  LSEX={lsex:.4f}")

    # Summary CSV
    csv_path = resultsdir / "summary_expanded.csv"
    if all_results:
        fields = [k for k in all_results[0].keys()
                  if not isinstance(all_results[0][k], (list, dict))]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            w.writeheader()
            for r in all_results:
                w.writerow({k: v for k, v in r.items()
                            if not isinstance(v, (list, dict))})

    print(f"\n{'='*60}")
    print(f"Expanded metrics saved to {resultsdir}")
    print(f"Updated {len(all_results)} JSON files + summary_expanded.csv")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
