#!/usr/bin/env python
"""Run visual conflict detector on ALL figures by building them from source."""
import json, sys, importlib, traceback
from datetime import datetime
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")

# Each entry: (module_path, display_name)
SCRIPTS = [
    ("benchmarks.scripts.plotting.plot_quant_comparison",    "fig2_quant_comparison"),
    ("benchmarks.scripts.plotting.plot_ablation_summary",    "fig3_ablation_summary"),
    ("benchmarks.scripts.plotting.plot_training_dynamics",   "fig4_training_dynamics"),
    ("benchmarks.scripts.plotting.plot_composed",            "fig5_composed"),
    ("benchmarks.scripts.plotting.plot_beta_sensitivity",    "fig6_beta_sensitivity"),
    ("benchmarks.scripts.plotting.plot_generalization",      "fig7_generalization"),
    ("benchmarks.scripts.plotting.plot_ode_trajectory",      "fig8_ode_trajectory"),
    ("benchmarks.scripts.plotting.plot_batch_integration",   "fig9_batch_integration"),
    ("benchmarks.scripts.plotting.plot_biological_validation", "fig10_biological_validation"),
]

all_results = {}
for mod_path, label in SCRIPTS:
    print(f"\n{'='*60}")
    print(f"  Building: {label}")
    print(f"{'='*60}")
    try:
        mod = importlib.import_module(mod_path)
        importlib.reload(mod)
        sys.argv = [sys.argv[0]]
        issues = mod.main()
        all_results[label] = issues if issues else []
    except Exception:
        traceback.print_exc()
        all_results[label] = []

# Summary
print(f"\n{'='*60}")
print("GLOBAL VCD AUDIT SUMMARY")
print(f"{'='*60}")
total_w = 0
total_i = 0
for label, issues in all_results.items():
    nw = sum(1 for x in issues if x.get("severity") == "warning")
    ni = sum(1 for x in issues if x.get("severity") == "info")
    total_w += nw
    total_i += ni
    status = "CLEAN" if nw == 0 else "WARNINGS"
    print(f"  {label:45s}  {nw:2d} warnings  {ni:2d} info  [{status}]")
print(f"{'='*60}")
print(f"  TOTAL: {total_w} warnings, {total_i} info across {len(all_results)} figures")
if total_w == 0:
    print("  ✓ ALL FIGURES CLEAN")
else:
    print(f"  ✗ {total_w} warnings need fixing")
print(f"{'='*60}")

# ── Save results to JSON report ──────────────────────────────────────────
report_dir = Path(__file__).resolve().parent / "benchmarks" / "results"
report_dir.mkdir(parents=True, exist_ok=True)
report_path = report_dir / "vcd_audit_report.json"
report = {
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "total_warnings": total_w,
    "total_info": total_i,
    "figures": {},
}
for label, issues in all_results.items():
    report["figures"][label] = {
        "warnings": sum(1 for x in issues if x.get("severity") == "warning"),
        "info": sum(1 for x in issues if x.get("severity") == "info"),
        "issues": [
            {"type": x.get("type"), "severity": x.get("severity"), "detail": x.get("detail")}
            for x in issues
        ],
    }
report_path.write_text(json.dumps(report, indent=2))
print(f"\n  Report saved → {report_path}")
