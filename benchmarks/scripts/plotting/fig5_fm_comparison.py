#!/usr/bin/env python
"""MoCoO Figure 5 — FM-enhanced comparison boxplots (proposed metrics).

Same layout as Figure 2 but includes all 12 configurations (6 base + 6 FM
variants). Highlights the effect of Phase-2 Flow Matching refinement on
the proposed metric set.

Panels:
  (a) Embedding Quality:  DRE, LSE, DREX, LSEX  (x 4 splits)
  (b) Clustering:         NMI, ARI, ASW, DAV     (x 4 splits)
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
    get_config_colors,
    get_config_order,
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
        figure_size=(20.0, 11.0),
        output_name="fig5_fm_comparison",
        vcd_label="fig5_fm_comparison",
        fs_tick_offset=0,
        warn_if_no_fm=True,
    )


if __name__ == "__main__":
    main()
