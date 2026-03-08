# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of MoCoO package
- VAE with multiple likelihood modes (MSE, NB, ZINB, Poisson, ZIP)
- Neural ODE for trajectory modeling
- Momentum Contrast (MoCo) for representation learning
- Information bottleneck regularization
- Disentanglement losses (DIP-VAE, β-TC-VAE, InfoVAE)
- Vector field analysis for RNA velocity
- Comprehensive test suite (87 tests)
- 6-config ablation benchmark pipeline
- PCA+k-means baseline comparison
- Dataset registry for standardised data loading
- **Cross-dataset generalization benchmark** (`run_cross_and_validate.py`):
  runs all 6 configs on dentate and endo datasets with pseudotime-marker
  correlation analysis and latent smoothness metrics
- **Pseudotime extraction API** (`agent.get_pseudotime()`): returns per-cell
  pseudotime ordering from the ODE component
- **Latent smoothness API** (`agent.get_latent_smoothness()`): returns k-NN
  graph entropy and effective dimensionality of the learned latent space
- **Pseudotime–marker correlation API** (`agent.pseudotime_marker_correlation()`):
  computes Spearman/Pearson correlations between learned pseudotime and
  marker gene expression with significance testing
- **LSE integration** (`SingleCellLatentSpaceEvaluator`): manifold
  dimensionality, spectral decay, participation ratio, anisotropy,
  trajectory directionality, noise resilience — computed inline in the
  benchmark pipeline for all configs
- **DRE integration** (`DimensionalityReductionEvaluator`): distance
  correlation, Q_local, Q_global via co-ranking matrix against UMAP and
  t-SNE embeddings of the latent space
- **Paul dataset** (2,730 cells, 19 myeloid/erythroid types) registered in
  `dataset_registry.py` with marker genes for pseudotime biovalidation
- **Spinoids dataset** (9,619 cells, 8 spinal cord organoid types) registered
  with marker genes for pseudotime biovalidation
- **5-dataset benchmark** (IRALL, dentate, endo, paul, spinoids) × 6 configs
  @ 150 epochs with LSE/DRE columns in all summary CSVs

### Changed
- Restructured codebase into proper Python package
- Updated to modern packaging standards (pyproject.toml)
- Rebalanced loss weights for multi-objective training:
  MoCo weight 0.6, prototype weight 0.1, VAE/ODE blend 0.8/0.2
- **Beta sweep infrastructure**: `--beta` CLI flag in `run_benchmark.py`
  supports systematic KL weight sweep (1.0, 0.1, 0.01)
- **Article rewritten** with beta sweep results: Tables I–VII documenting
  6 configs × 3 beta values on IRALL (1000 cells, 50 epochs)
- **Component effect analysis**: ODE, MoCo, and Proto deltas computed per
  beta value with ODE×MoCo synergy interaction terms
- **README updated** with expanded metrics list and key findings table

### Fixed
- ODE solver now runs on GPU, preserving computation graph (was forcing CPU)
- Removed double reconstruction loss for ODE configs (VAE recon only)
- Time deduplication replaced with jittering to preserve all cells
- Eliminated redundant re-encoding in update(); reuse sort_idx from forward pass
- DataLoader now provides paired (normalised, raw) data for correct NB loss
- JSON serialisation of numpy float32 values in benchmark output
- Cross-dataset runner (`run_cross_dataset.py`) now uses corrected loss weights
  (was still using old pre-fix weights: moco_weight=1.0, ode_reg=0.5)
- Dataset registry paths updated to correct data locations
- **HSDE-style architectural fixes**: removed double log1p, time-conditioned
  ODE concat mode, LayerNorm in encoder, q_m/q_s clamping, boolean mask
  deduplication, lib_size clamping
- **Stop-gradient on q_z** for ODE-specific losses (qz_div, vel_loss,
  cross_path_contrastive) prevents ODE gradients from distorting encoder clusters
- **Proto weight reduced** (0.3→0.1) and **MoCo weight increased** (0.5→0.6)
  to fix Full model underperformance vs VAE+ODE+MoCo

## [0.0.1] - 2025-12-26

### Added
- Initial implementation of MoCoO framework
- Core VAE, ODE, and MoCo components
- Basic training and inference functionality
- Package structure and configuration files