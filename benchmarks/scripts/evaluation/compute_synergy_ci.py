#!/usr/bin/env python3
"""
Compute synergy interaction terms with bootstrap CIs from multi-seed data.
Addresses W3: statistical power for synergy claims.

For each metric M, the interaction term is:
    Δ_int = M(Full) - M(VAE+ODE) - M(VAE+MoCo) + M(VAE)

Positive Δ_int for upward-good metrics (ARI, NMI) = super-additive.
Negative Δ_int for downward-good metrics (DB) = super-additive (better compactness).

Usage:
    python compute_synergy_ci.py --input multiseed_beta001.csv
"""
import argparse
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")


def bootstrap_ci(values, n_boot=10000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    n = len(values)
    boot_means = [np.mean(rng.choice(values, size=n, replace=True)) for _ in range(n_boot)]
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return float(lo), float(np.mean(values)), float(hi)


def compute_interaction(df, metric, seed_col="seed"):
    """Compute per-seed interaction terms and bootstrap CIs."""
    seeds = sorted(df[seed_col].unique())
    interactions = []

    for s in seeds:
        ds = df[df[seed_col] == s]
        vals = {}
        for config in ["VAE", "VAE+ODE", "VAE+MoCo", "Full"]:
            row = ds[ds["config"] == config]
            if len(row) == 0:
                # Try alternative names
                alt = {"Full": "VAE+ODE+MoCo+Proto"}
                row = ds[ds["config"] == alt.get(config, config)]
            if len(row) == 0:
                break
            vals[config] = row[metric].values[0]

        if len(vals) == 4:
            interaction = vals["Full"] - vals["VAE+ODE"] - vals["VAE+MoCo"] + vals["VAE"]
            interactions.append(interaction)

    if not interactions:
        return None

    interactions = np.array(interactions)
    lo, mean, hi = bootstrap_ci(interactions)
    n = len(interactions)

    return {
        "metric": metric,
        "n_seeds": n,
        "mean_interaction": round(mean, 4),
        "ci95_lo": round(lo, 4),
        "ci95_hi": round(hi, 4),
        "std": round(np.std(interactions, ddof=1) if n > 1 else 0, 4),
        "zero_excluded": "yes" if (lo > 0 or hi < 0) else "no",
        "direction": "positive" if mean > 0 else "negative",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--metrics", nargs="+",
                        default=["ARI", "NMI", "ASW", "CH", "DB"])
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.input).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = [m for m in args.metrics if m in df.columns]
    print(f"Computing synergy CIs from {len(df)} rows, "
          f"{df['seed'].nunique()} seeds, metrics: {metrics}")

    results = []
    for m in metrics:
        res = compute_interaction(df, m)
        if res:
            results.append(res)
            star = " *" if res["zero_excluded"] == "yes" else ""
            print(f"  {m:6s}: Δ_int = {res['mean_interaction']:+.4f} "
                  f"[{res['ci95_lo']:+.4f}, {res['ci95_hi']:+.4f}]{star}")

    if results:
        rdf = pd.DataFrame(results)
        rdf.to_csv(out_dir / "synergy_ci.csv", index=False)
        print(f"\nSaved to {out_dir / 'synergy_ci.csv'}")

        n_sig = sum(1 for r in results if r["zero_excluded"] == "yes")
        print(f"\n{n_sig}/{len(results)} metrics have CIs excluding zero")


if __name__ == "__main__":
    main()
