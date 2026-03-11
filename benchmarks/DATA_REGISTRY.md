# MoCoO Figure Data Registry

All data paths required to regenerate each figure from scratch.

## Figure 1 — Architecture Diagram
- **Source**: `paper/fig_architecture.tex`  
- **Output**: `paper/fig_architecture.pdf`  
- **Regenerate**: `cd paper && pdflatex fig_architecture.tex`

## Figure 2 — Quantitative Comparison
- **Script**: `benchmarks/scripts/plotting/plot_quant_comparison.py`
- **Data**: `benchmarks/results/single_dataset/benchmark_data.npz` (latents, labels, configs)
- **Metrics**: `benchmarks/results/single_dataset/{VAE,VAE_ODE,VAE_MoCo,VAE_MoCo_Proto,VAE_ODE_MoCo,Full}.json`
- **UMAP cache**: `benchmarks/results/single_dataset/qc_umap_*.npz`
- **Output**: `benchmarks/figures/fig2_quant_comparison.png` + sub-panels in `fig2_quant_comparison/`
- **Regenerate**: `python benchmarks/scripts/plotting/plot_quant_comparison.py`

## Figure 3 — Ablation Summary
- **Script**: `benchmarks/scripts/plotting/plot_ablation_summary.py`
- **Data**: `benchmarks/results/single_dataset/benchmark_data.npz`
- **Metrics**: `benchmarks/results/single_dataset/*.json`
- **Beta ablation**: `benchmarks/results/beta_ablation/beta_{0.01,0.1,1.0}/*.json`
- **Output**: `benchmarks/figures/fig3_ablation_summary.png` + sub-panels
- **Regenerate**: `python benchmarks/scripts/plotting/plot_ablation_summary.py`

## Figure 4 — Training Dynamics
- **Script**: `benchmarks/scripts/plotting/plot_training_dynamics.py`
- **Data**: `benchmarks/results/single_dataset/benchmark_data.npz` (train_losses, val_losses, val_scores)
- **Metrics**: `benchmarks/results/single_dataset/*.json` (train_time_s, peak_mem_gb)
- **Output**: `benchmarks/figures/fig4_training_dynamics.png` + sub-panels
- **Regenerate**: `python benchmarks/scripts/plotting/plot_training_dynamics.py`

## Figure 5 — Integrated Benchmark Overview
- **Script**: `benchmarks/scripts/plotting/plot_composed.py`
- **Data**: `benchmarks/results/single_dataset/benchmark_data.npz`
- **Metrics**: `benchmarks/results/single_dataset/*.json`
- **UMAP cache**: `benchmarks/results/single_dataset/umap_cache.npz`
- **Beta sweep input**: `benchmarks/results/beta_ablation/beta_0.1/*.json`
- **Output**: `benchmarks/figures/fig5_composed_benchmark.png` + sub-panels in `fig5_composed_benchmark/`
- **Regenerate**: `python benchmarks/scripts/plotting/plot_composed.py`

### Figure 5 helper block — Subcategory Heatmap
- **Script**: `benchmarks/scripts/plotting/plot_subcategory_heatmap.py`
- **Data**: `benchmarks/results/beta_ablation/beta_0.1/*.json`
- **Output**: `benchmarks/figures/fig5_composed_benchmark/panelD_subcategory_block.png`
- **Regenerate**: `python benchmarks/scripts/plotting/plot_subcategory_heatmap.py`

## Figure 6a — Beta Sensitivity
- **Script**: `benchmarks/scripts/plotting/plot_beta_sensitivity.py`
- **Data**: `benchmarks/results/beta_ablation/beta_{0.01,0.1,1.0}/*.json`
- **Output**: `benchmarks/figures/fig6_beta_sensitivity.png`
- **Regenerate**: `python -m benchmarks.scripts.plotting.plot_beta_sensitivity`

## Figure 6b — ODE Trajectory (Supplementary)
- **Script**: `benchmarks/scripts/plotting/plot_ode_trajectory.py`
- **Data**: `benchmarks/results/single_dataset/benchmark_data.npz`
- **Expression data**: IRALL h5ad file (path from `mocoo/configs/paths.env` or `--adata` argument)
- **Output**: `benchmarks/figures/supp_ode_trajectory.png` + sub-panels in `supp_ode_trajectory/`
- **Regenerate**: `python benchmarks/scripts/plotting/plot_ode_trajectory.py`

## Figure 7a — Generalization (Val vs Test)
- **Script**: `benchmarks/scripts/plotting/plot_generalization.py`
- **Data**: `benchmarks/results/single_dataset/*.json` (contains both val and test_ext_ metrics)
- **Output**: `benchmarks/figures/fig7_generalization.png`
- **Regenerate**: `python -m benchmarks.scripts.plotting.plot_generalization`

## Figure 7b — Batch Integration (Supplementary)
- **Script**: `benchmarks/scripts/plotting/plot_batch_integration.py`
- **Data**: **NOT YET COMPUTED** — requires `summary_batch.csv` with iLISI, bASW, cLISI, graph_conn, iso_label_ASW
- **Cross-dataset**: Requires `benchmarks/results/cross_dataset/{IRALL,dentate,endo}/` with per-config JSONs
- **Output**: `benchmarks/figures/supp_batch_integration.png` + sub-panels in `supp_batch_integration/`
- **Status**: Data absent. Panels will be hidden automatically.

## Supplementary — Biological Validation
- **Script**: `benchmarks/scripts/plotting/plot_biological_validation.py`
- **Data**: `benchmarks/results/single_dataset/benchmark_data.npz`
- **Expression data**: IRALL h5ad file
- **UMAP cache**: `benchmarks/results/bv_umap_cache.npz`
- **Output**: `benchmarks/figures/supp_biological_validation.png` + sub-panels in `supp_biological_validation/`
- **Regenerate**: `python -m benchmarks.scripts.plotting.plot_biological_validation`

## Tables (in paper)
- Tables I–III (Beta sweep): `benchmarks/results/beta_ablation/beta_{1.0,0.1,0.01}/*.json`
- Table IV (Component effects): Computed from deltas between configs in beta ablation JSONs
- Table V (Synergy): Computed from interaction terms across beta ablation JSONs
- Table VI (Full model beta): Extracted from `Full.json` in each beta directory
- Table VII (Win counts): Computed by comparing all configs per beta
- Tables VIII–XII (Biovalidation markers): Require ODE pseudotime + expression data (not stored as JSON)
- Table XIII (Biovalidation summary): Aggregated from per-dataset marker tables
- Table XIV (Component summary): Manual editorial summary
