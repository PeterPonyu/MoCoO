#!/usr/bin/env python
"""Detailed VCD audit — prints warning-level issues grouped by figure."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import all build_figure functions
from benchmarks.scripts.plotting.plot_quant_comparison import build_figure as bf_fig2
from benchmarks.scripts.plotting.plot_ablation_summary import build_figure as bf_fig3
from benchmarks.scripts.plotting.plot_training_dynamics import build_figure as bf_fig4
from benchmarks.scripts.plotting.plot_composed import build_figure as bf_fig5
from benchmarks.scripts.plotting.plot_subcategory_heatmap import build_figure as bf_fig5h
from benchmarks.scripts.plotting.plot_beta_sensitivity import build_figure as bf_fig6
from benchmarks.scripts.plotting.plot_generalization import build_figure as bf_fig7
from benchmarks.scripts.plotting.plot_ode_trajectory import build_figure as bf_ode
from benchmarks.scripts.plotting.plot_batch_integration import build_figure as bf_batch
from benchmarks.scripts.plotting.plot_biological_validation import build_figure as bf_bio

OUT = Path("/tmp/vcd_audit")
OUT.mkdir(exist_ok=True)
R = Path("benchmarks/results")
SD = R / "single_dataset"
BA = R / "beta_ablation" / "beta_0.1"
DATA = "/home/zeyufu/LAB/scRL/IRALL.h5ad"

FIGURES = [
    ("fig2_quant_comparison", lambda: bf_fig2(SD, OUT)),
    ("fig3_ablation_summary", lambda: bf_fig3(SD, OUT)),
    ("fig4_training_dynamics", lambda: bf_fig4(SD, OUT)),
    ("fig5_composed_benchmark", lambda: bf_fig5(SD, OUT)),
    ("fig5_subcategory_heatmap", lambda: bf_fig5h(BA, OUT)),
    ("fig6_beta_sensitivity", lambda: bf_fig6(R, OUT)),
    ("fig7_generalization", lambda: bf_fig7(BA, OUT)),
    ("supp_ode_trajectory", lambda: bf_ode(SD, OUT, DATA)),
    ("supp_batch_integration", lambda: bf_batch(R, OUT)),
    ("supp_biological_validation", lambda: bf_bio(SD, OUT, DATA)),
]

all_issues = {}
for name, fn in FIGURES:
    print(f"\n{'#'*70}")
    print(f"# {name}")
    print(f"{'#'*70}")
    try:
        issues = fn()
        if issues is None:
            issues = []
    except Exception as e:
        print(f"  ERROR: {e}")
        issues = []
    all_issues[name] = issues
    warnings = [i for i in issues if i.get("severity") == "warning"]
    # Group by type
    by_type = {}
    for w in warnings:
        by_type.setdefault(w["type"], []).append(w)
    for typ, ws in sorted(by_type.items()):
        print(f"\n  {typ} ({len(ws)} warnings):")
        for w in ws[:5]:
            print(f"    - {w['detail'][:150]}")
        if len(ws) > 5:
            print(f"    ... and {len(ws)-5} more")
    plt.close("all")

# Final summary
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
for name, issues in all_issues.items():
    nw = sum(1 for x in issues if x.get("severity") == "warning")
    print(f"  {name:40s}  {nw:4d} warnings")
total = sum(sum(1 for x in v if x.get("severity") == "warning") for v in all_issues.values())
print(f"{'='*70}")
print(f"  TOTAL: {total} warnings")
