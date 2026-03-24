#!/usr/bin/env python
"""MoCoO Figure 2 — Ablation boxplots on proposed metrics.

Single composed figure with two panels of 4 metrics each:
  (a) Embedding Quality:  DRE, LSE, DREX, LSEX
  (b) Clustering:         ASW, DAV

Each subplot shows per-config boxplot distributions across all available
datasets.  Four rows per panel correspond to train/val/test/whole splits.
Shows all 12 configurations (6 base + 6 FM variants) to restore FM theme.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from benchmarks.scripts.plotting.shared import build_boxplot_figure, setup_fonts
from mocoo.visualization.style import (
    apply_style,
    get_config_order,
    get_config_colors,
)

setup_fonts()
apply_style()


def main():
    benchmarks_dir = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--resultsdir", default=str(benchmarks_dir / "results"))
    parser.add_argument("--outdir", default=str(benchmarks_dir / "figures"))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    return build_boxplot_figure(
        Path(args.resultsdir),
        outdir,
        configs=get_config_order(),
        config_colors=get_config_colors(),
        figure_size=(22.0, 11.0),
        output_name="fig2_ablation_boxplots",
        vcd_label="fig2_ablation_boxplots",
        fs_tick_offset=0,
        warn_if_no_fm=True,
    )


if __name__ == "__main__":
    main()
