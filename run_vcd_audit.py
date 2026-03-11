#!/usr/bin/env python
"""Run visual conflict detector on ALL generated figures and report."""
import sys, json
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from benchmarks.scripts.pipeline.visual_conflict_detector import detect_all_conflicts

FIGURES_DIR = Path("benchmarks/figures")

PNG_FILES = [
    "fig2_quant_comparison.png",
    "fig3_ablation_summary.png",
    "fig4_training_dynamics.png",
    "fig5_composed_benchmark.png",
    "fig5_subcategory_heatmap_a.png",
    "fig5_subcategory_heatmap_b.png",
    "fig6_beta_sensitivity.png",
    "fig7_generalization.png",
    "supp_ode_trajectory.png",
    "supp_batch_integration.png",
    "supp_biological_validation.png",
]

all_results = {}
for fname in PNG_FILES:
    fpath = FIGURES_DIR / fname
    if not fpath.exists():
        print(f"SKIP {fname} — not found")
        continue
    print(f"\n{'='*60}")
    print(f"  VCD: {fname}")
    print(f"{'='*60}")
    img = mpimg.imread(str(fpath))
    fig, ax = plt.figure(dpi=300), None
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.set_axis_off()
    issues = detect_all_conflicts(fig, label=fname, verbose=True)
    all_results[fname] = issues
    plt.close(fig)

# Summary
print(f"\n{'='*60}")
print("GLOBAL VCD AUDIT SUMMARY")
print(f"{'='*60}")
total_w = 0
total_i = 0
for fname, issues in all_results.items():
    nw = sum(1 for x in issues if x.get("severity") == "warning")
    ni = sum(1 for x in issues if x.get("severity") == "info")
    total_w += nw
    total_i += ni
    status = "CLEAN" if nw == 0 else "WARNINGS"
    print(f"  {fname:45s}  {nw} warnings  {ni} info  [{status}]")
print(f"{'='*60}")
print(f"  TOTAL: {total_w} warnings, {total_i} info across {len(all_results)} figures")
if total_w == 0:
    print("  ✓ ALL FIGURES CLEAN")
else:
    print(f"  ✗ {total_w} warnings need fixing")
print(f"{'='*60}")
