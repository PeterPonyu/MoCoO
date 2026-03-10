#!/bin/bash
# Regenerate all MoCoO figures
set -e
source .venv/bin/activate
export MOCOO_DATA_DIR=/home/zeyufu

echo "=== Fig 2: Quantitative Comparison ==="
python benchmarks/scripts/plotting/plot_quant_comparison.py \
    --resultsdir benchmarks/results/single_dataset \
    --outdir benchmarks/figures

echo "=== Fig 3: Ablation Summary ==="
python benchmarks/scripts/plotting/plot_ablation_summary.py \
    --resultsdir benchmarks/results/single_dataset \
    --outdir benchmarks/figures

echo "=== Fig 4: Training Dynamics ==="
python benchmarks/scripts/plotting/plot_training_dynamics.py \
    --resultsdir benchmarks/results/single_dataset \
    --outdir benchmarks/figures

echo "=== Fig 5a: Composed Benchmark ==="
python benchmarks/scripts/plotting/plot_composed.py \
    --resultsdir benchmarks/results/single_dataset \
    --outdir benchmarks/figures

echo "=== Fig 5b: Subcategory Heatmap ==="
python benchmarks/scripts/plotting/plot_subcategory_heatmap.py \
    --resultsdir benchmarks/results/beta_ablation/beta_0.1 \
    --outdir benchmarks/figures

echo "=== Fig 6: Beta Sensitivity ==="
python benchmarks/scripts/plotting/plot_beta_sensitivity.py \
    --resultsdir benchmarks/results \
    --outdir benchmarks/figures

echo "=== Fig 7: Generalization ==="
python benchmarks/scripts/plotting/plot_generalization.py \
    --resultsdir benchmarks/results/beta_ablation/beta_0.1 \
    --outdir benchmarks/figures

echo "=== Supp: ODE Trajectory ==="
python benchmarks/scripts/plotting/plot_ode_trajectory.py \
    --resultsdir benchmarks/results/single_dataset \
    --outdir benchmarks/figures \
    --data /home/zeyufu/LAB/scRL/IRALL.h5ad

echo "=== Supp: Batch Integration ==="
python benchmarks/scripts/plotting/plot_batch_integration.py \
    --resultsdir benchmarks/results \
    --outdir benchmarks/figures

echo "=== Supp: Biological Validation ==="
python benchmarks/scripts/plotting/plot_biological_validation.py \
    --resultsdir benchmarks/results/single_dataset \
    --outdir benchmarks/figures \
    --data /home/zeyufu/LAB/scRL/IRALL.h5ad

echo "=== All figures regenerated ==="
ls -la benchmarks/figures/*.png

echo ""
echo "=== Significance Tests ==="
mkdir -p benchmarks/results/multiseed/significance
python benchmarks/scripts/evaluation/significance_tests.py \
    --input benchmarks/results/multiseed/multiseed_IRALL.csv \
    --baseline VAE \
    --output_dir benchmarks/results/multiseed/significance

echo ""
echo "=== Build Paper ==="
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main 2>/dev/null || true
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
echo "=== Paper built: paper/main.pdf ==="
