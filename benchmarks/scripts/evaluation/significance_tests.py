#!/usr/bin/env python3
"""
Statistical significance tests for MoCoO ablation study.
Addresses Major Concern M1: single-seed results lack confidence intervals.

Given multi-seed results (CSV with columns: config, seed, ARI, NMI, ASW, ...),
performs pairwise Wilcoxon signed-rank tests and bootstrap confidence intervals.

Usage (after multi-seed runs are complete):
    python significance_tests.py --input multiseed_results.csv
    python significance_tests.py --input multiseed_results.csv --baseline VAE --alpha 0.05
"""
import argparse
import sys
import warnings
import itertools
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

warnings.filterwarnings("ignore")


def bootstrap_ci(values, n_boot=10000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval for the mean."""
    rng = np.random.RandomState(seed)
    n = len(values)
    boot_means = np.array([np.mean(rng.choice(values, size=n, replace=True)) for _ in range(n_boot)])
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return float(lo), float(np.mean(values)), float(hi)


def pairwise_wilcoxon(df, metric, configs, alpha=0.05):
    """Pairwise Wilcoxon signed-rank test for a given metric across configs."""
    results = []
    pairs = list(itertools.combinations(configs, 2))
    # Bonferroni correction
    n_tests = len(pairs)

    for c1, c2 in pairs:
        v1 = df[df["config"] == c1][metric].values
        v2 = df[df["config"] == c2][metric].values

        if len(v1) < 3 or len(v2) < 3:
            results.append({
                "config_1": c1, "config_2": c2, "metric": metric,
                "mean_1": np.mean(v1), "mean_2": np.mean(v2),
                "diff": np.mean(v2) - np.mean(v1),
                "p_value": np.nan, "significant": False,
                "note": f"Too few seeds (n1={len(v1)}, n2={len(v2)}); need ≥3"
            })
            continue

        # Align by seed
        min_n = min(len(v1), len(v2))
        try:
            stat, p = stats.wilcoxon(v1[:min_n], v2[:min_n], alternative="two-sided")
        except ValueError:
            stat, p = np.nan, 1.0

        # Bonferroni-corrected
        p_adj = min(p * n_tests, 1.0)

        results.append({
            "config_1": c1,
            "config_2": c2,
            "metric": metric,
            "mean_1": np.mean(v1),
            "mean_2": np.mean(v2),
            "diff": np.mean(v2) - np.mean(v1),
            "statistic": stat,
            "p_raw": p,
            "p_bonferroni": p_adj,
            "significant": p_adj < alpha,
            "note": "",
        })

    return pd.DataFrame(results)


def baseline_comparison(df, metric, baseline, configs, alpha=0.05):
    """Compare each config against a baseline using Mann-Whitney U."""
    results = []
    v_base = df[df["config"] == baseline][metric].values
    non_base = [c for c in configs if c != baseline]
    n_tests = len(non_base)

    for c in non_base:
        v = df[df["config"] == c][metric].values

        if len(v_base) < 2 or len(v) < 2:
            results.append({
                "baseline": baseline, "config": c, "metric": metric,
                "mean_baseline": np.mean(v_base), "mean_config": np.mean(v),
                "diff": np.mean(v) - np.mean(v_base),
                "p_value": np.nan, "significant": False,
            })
            continue

        stat, p = stats.mannwhitneyu(v_base, v, alternative="two-sided")
        p_adj = min(p * n_tests, 1.0)

        lo, mean, hi = bootstrap_ci(v - v_base[:len(v)] if len(v) <= len(v_base) else v[:len(v_base)] - v_base)

        results.append({
            "baseline": baseline,
            "config": c,
            "metric": metric,
            "mean_baseline": np.mean(v_base),
            "mean_config": np.mean(v),
            "diff": np.mean(v) - np.mean(v_base),
            "ci_lo": lo,
            "ci_hi": hi,
            "p_raw": p,
            "p_bonferroni": p_adj,
            "significant": p_adj < alpha,
        })

    return pd.DataFrame(results)


def summary_table(df, metrics, configs):
    """Generate mean ± std table for all configs and metrics."""
    rows = []
    for c in configs:
        row = {"config": c, "n_seeds": len(df[df["config"] == c])}
        for m in metrics:
            vals = df[df["config"] == c][m].values
            lo, mean, hi = bootstrap_ci(vals) if len(vals) >= 2 else (np.nan, np.mean(vals), np.nan)
            row[f"{m}_mean"] = mean
            row[f"{m}_std"] = np.std(vals)
            row[f"{m}_ci95_lo"] = lo
            row[f"{m}_ci95_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Statistical significance tests")
    parser.add_argument("--input", type=str, required=True, help="CSV with columns: config, seed, ARI, NMI, ASW, ...")
    parser.add_argument("--baseline", type=str, default="VAE")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--metrics", nargs="+", default=["ARI", "NMI", "ASW"])
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    configs = sorted(df["config"].unique())
    metrics = [m for m in args.metrics if m in df.columns]

    if not metrics:
        print(f"ERROR: None of {args.metrics} found in columns: {list(df.columns)}")
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else Path(args.input).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SUMMARY TABLE (mean ± std [95% CI])")
    print("=" * 70)
    summary = summary_table(df, metrics, configs)
    print(summary.to_string(index=False))
    summary.to_csv(out_dir / "significance_summary.csv", index=False)

    # Pairwise tests
    all_pairwise = []
    for m in metrics:
        pw = pairwise_wilcoxon(df, m, configs, alpha=args.alpha)
        all_pairwise.append(pw)
        sig = pw[pw["significant"]]
        print(f"\n{'='*70}")
        print(f"Pairwise Wilcoxon — {m} ({len(sig)}/{len(pw)} significant at α={args.alpha})")
        print(f"{'='*70}")
        for _, r in pw.iterrows():
            star = " *" if r.get("significant") else ""
            note = f"  [{r['note']}]" if r.get("note") else ""
            print(f"  {r['config_1']:20s} vs {r['config_2']:20s}: "
                  f"Δ={r['diff']:+.4f}  p={r.get('p_bonferroni', r.get('p_value', float('nan'))):.4f}{star}{note}")

    pw_all = pd.concat(all_pairwise, ignore_index=True)
    pw_all.to_csv(out_dir / "pairwise_wilcoxon.csv", index=False)

    # Baseline comparison
    if args.baseline in configs:
        all_baseline = []
        for m in metrics:
            bl = baseline_comparison(df, m, args.baseline, configs, alpha=args.alpha)
            all_baseline.append(bl)
            print(f"\n{'='*70}")
            print(f"vs {args.baseline} — {m}")
            print(f"{'='*70}")
            for _, r in bl.iterrows():
                star = " *" if r["significant"] else ""
                print(f"  {r['config']:20s}: Δ={r['diff']:+.4f}  "
                      f"95%CI=[{r.get('ci_lo', float('nan')):.4f}, {r.get('ci_hi', float('nan')):.4f}]  "
                      f"p={r.get('p_bonferroni', float('nan')):.4f}{star}")

        bl_all = pd.concat(all_baseline, ignore_index=True)
        bl_all.to_csv(out_dir / "baseline_comparison.csv", index=False)

    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
