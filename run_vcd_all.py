#!/usr/bin/env python
"""Detailed VCD audit — runs each figure's main() and prints ALL warnings."""
import subprocess, sys, re
from pathlib import Path

scripts = [
    ("fig6_beta_sensitivity",
     "python benchmarks/scripts/plotting/plot_beta_sensitivity.py --resultsdir benchmarks/results --outdir /tmp/vcd"),
    ("fig4_training_dynamics",
     "python benchmarks/scripts/plotting/plot_training_dynamics.py --resultsdir benchmarks/results/single_dataset --outdir /tmp/vcd"),
    ("fig5_composed",
     "python benchmarks/scripts/plotting/plot_composed.py --resultsdir benchmarks/results/single_dataset --outdir /tmp/vcd"),
    ("fig2_quant_comparison",
     "python benchmarks/scripts/plotting/plot_quant_comparison.py --resultsdir benchmarks/results/single_dataset --outdir /tmp/vcd"),
    ("fig3_ablation_summary",
     "python benchmarks/scripts/plotting/plot_ablation_summary.py --resultsdir benchmarks/results/single_dataset --outdir /tmp/vcd"),
    ("fig7_generalization",
     "python benchmarks/scripts/plotting/plot_generalization.py --resultsdir benchmarks/results/beta_ablation/beta_0.1 --outdir /tmp/vcd"),
    ("supp_ode_trajectory",
     "python benchmarks/scripts/plotting/plot_ode_trajectory.py --resultsdir benchmarks/results/single_dataset --outdir /tmp/vcd --data /home/zeyufu/LAB/scRL/IRALL.h5ad"),
    ("supp_batch_integration",
     "python benchmarks/scripts/plotting/plot_batch_integration.py --resultsdir benchmarks/results --outdir /tmp/vcd"),
    ("supp_bio_validation",
     "python benchmarks/scripts/plotting/plot_biological_validation.py --resultsdir benchmarks/results/single_dataset --outdir /tmp/vcd --data /home/zeyufu/LAB/scRL/IRALL.h5ad"),
]

Path("/tmp/vcd").mkdir(exist_ok=True)

for name, cmd in scripts:
    print(f"\n{'#'*70}")
    print(f"# {name}")
    print(f"{'#'*70}")
    # Run and capture, then filter warnings
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                          cwd=str(Path(__file__).resolve().parent))
    combined = result.stdout + result.stderr
    # Print just the VCD section
    in_vcd = False
    for line in combined.split('\n'):
        if 'Conflict Detection' in line:
            in_vcd = True
        if in_vcd:
            print(line)
        if in_vcd and ('warnings |' in line or 'ISSUES' in line or 'OK' in line):
            in_vcd = False
    sys.stdout.flush()
