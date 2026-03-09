#\!/bin/bash
export MPLBACKEND=Agg
export PYTHONPATH=/home/zeyufu/Desktop/MoCoO
cd /home/zeyufu/Desktop/MoCoO
python benchmarks/scripts/plotting/plot_subcategory_heatmap.py --resultsdir benchmarks/results/beta_ablation/beta_0.1 --outdir benchmarks/figures
echo FIRST_DONE
python benchmarks/scripts/plotting/plot_subcategory_heatmap.py --resultsdir benchmarks/results/single_dataset --outdir benchmarks/figures
echo SECOND_DONE
