# MoCoO — Benchmark & Figure Guide

This document describes the benchmark directory layout, experiment series,
figure generation pipeline, and how results map to paper tables and figures.

---

## Directory Layout

```text
benchmarks/
├── FIGURES.md                          ← this document
├── scripts/
│   ├── pipeline/
│   │   ├── run_benchmark.py            ← single-dataset ablation (6 configs)
│   │   ├── run_beta_ablation.py        ← Series 1: beta sweep (Tables I-V)
│   │   ├── run_cross_dataset.py        ← Series 2: cross-dataset (Tables VIII-XIII)
│   │   ├── run_multiseed.py            ← Series 3: multi-seed robustness
│   │   ├── run_cross_and_validate.py   ← cross-dataset + LSE/DRE + pseudotime
│   │   ├── dataset_registry.py         ← DatasetRegistry (5 datasets)
│   │   └── visual_conflict_detector.py ← figure quality checker
│   ├── evaluation/
│   │   ├── recompute_metrics.py        ← recompute expanded metrics from NPZ
│   │   ├── significance_tests.py       ← Series 3b: statistical tests
│   │   ├── pca_kmeans_baseline.py      ← Series 4: external baseline
│   │   ├── compute_batch_metrics.py    ← batch integration metrics
│   │   ├── compute_gradients.py        ← ODE gradient extraction
│   │   └── metrics_expanded.py         ← thin wrapper (uses mocoo.evaluation)
│   └── plotting/
│       ├── plot_quant_comparison.py     ← Fig 2: quantitative latent comparison
│       ├── plot_ablation_summary.py     ← Fig 3: ablation summary
│       ├── plot_training_dynamics.py    ← Fig 4: training dynamics
│       ├── plot_composed.py            ← Fig 5: composed multi-panel
│       ├── plot_ode_trajectory.py       ← Supplemental: ODE trajectory
│       ├── plot_batch_integration.py    ← Supplemental: batch integration
│       └── plot_biological_validation.py ← Supplemental: biological validation
├── results/
│   ├── single_dataset/                 ← default single-dataset ablation (IRALL)
│   │   ├── benchmark_data.npz
│   │   ├── summary.csv
│   │   ├── summary_expanded.csv
│   │   └── {config}.json
│   ├── beta_ablation/                  ← Series 1: beta sweep
│   │   ├── beta_0.01/
│   │   ├── beta_0.1/
│   │   └── beta_1.0/
│   ├── cross_dataset/                  ← Series 2: multi-dataset
│   │   ├── IRALL/
│   │   ├── dentate/
│   │   ├── endo/
│   │   ├── paul/
│   │   ├── spinoids/
│   │   └── meta_analysis.csv
│   ├── multiseed/                      ← Series 3: robustness
│   │   ├── multiseed_IRALL.csv
│   │   └── significance/
│   ├── baselines/                      ← Series 4: external baselines
│   │   └── pca_kmeans.csv
│   └── _legacy_50ep/                   ← archived old 50-epoch results
└── figures/
    ├── fig2_quant_comparison.png        ← Paper Figure 2
    ├── fig2_quant_comparison/           ← individual panels
    ├── fig3_ablation_summary.png        ← Paper Figure 3
    ├── fig3_ablation_summary/
    ├── fig4_training_dynamics.png       ← Paper Figure 4
    ├── fig4_training_dynamics/
    ├── fig5_composed_benchmark.png      ← Paper Figure 5
    ├── supp_ode_trajectory.png          ← Supplemental
    ├── supp_batch_integration.png       ← Supplemental
    └── supp_biological_validation.png   ← Supplemental
```

---

## Experiment Series

| Series | Makefile Target | Script | Paper Tables | Description |
|--------|----------------|--------|-------------|-------------|
| 1 | `make series1` | `run_beta_ablation.py` | I-V | Beta ablation: 6 configs × 3 betas (200 epochs) |
| 2 | `make series2` | `run_cross_dataset.py` | VIII-XIII | Cross-dataset generalization (5 datasets) |
| 3 | `make series3` | `run_multiseed.py` + `significance_tests.py` | — | Multi-seed robustness + significance tests |
| 4 | `make series4` | `pca_kmeans_baseline.py` | — | External baselines (PCA+KMeans) |

---

## Figure-to-Script Mapping

| Paper Figure | Output File | Script | Input Directory |
|-------------|-------------|--------|----------------|
| Fig 1 | `fig_architecture.pdf` | (TikZ, `paper/fig_architecture.tex`) | — |
| Fig 2 | `fig2_quant_comparison.png` | `plot_quant_comparison.py` | `results/single_dataset/` |
| Fig 3 | `fig3_ablation_summary.png` | `plot_ablation_summary.py` | `results/single_dataset/` + `results/beta_ablation/` |
| Fig 4 | `fig4_training_dynamics.png` | `plot_training_dynamics.py` | `results/single_dataset/` |
| Fig 5 | `fig5_composed_benchmark.png` | `plot_composed.py` | `results/single_dataset/` |
| Supp A | `supp_ode_trajectory.png` | `plot_ode_trajectory.py` | `results/single_dataset/` + raw data |
| Supp B | `supp_batch_integration.png` | `plot_batch_integration.py` | `results/` (cross-dataset + single) |
| Supp C | `supp_biological_validation.png` | `plot_biological_validation.py` | `results/single_dataset/` + raw data |

---

## Regenerating Figures

```bash
# All figures at once
make figures

# Individual figures
make fig-comparison    # Fig 2
make fig-ablation      # Fig 3
make fig-dynamics      # Fig 4
make fig-composed      # Fig 5
make fig-trajectory    # Supplemental: ODE trajectory
make fig-batch         # Supplemental: batch integration
make fig-biovalidation # Supplemental: biological validation

# Override results directory
python benchmarks/scripts/plotting/plot_quant_comparison.py \
    --resultsdir benchmarks/results/single_dataset \
    --outdir benchmarks/figures
```

All plotting scripts accept `--resultsdir` and `--outdir` arguments. Defaults
point to the new directory layout (`results/single_dataset/` for most scripts,
`results/` for batch integration which reads multiple subdirectories).

---

## Six Ablation Configurations

| Config | VAE | ODE | MoCo | Proto |
|--------|:---:|:---:|:----:|:-----:|
| VAE | ✓ | | | |
| VAE+ODE | ✓ | ✓ | | |
| VAE+MoCo | ✓ | | ✓ | |
| VAE+MoCo+Proto | ✓ | | ✓ | ✓ |
| VAE+ODE+MoCo | ✓ | ✓ | ✓ | |
| Full (MoCoO) | ✓ | ✓ | ✓ | ✓ |

---

## Training Configuration

| Parameter | Single-Dataset / Cross | Beta Ablation (Series 1) |
|-----------|:---------------------:|:------------------------:|
| Epochs | 150 | 200 |
| Patience | 30 | 40 |
| Latent dim | 32 | 32 |
| Hidden dim | 128 | 128 |
| Bottleneck (i_dim) | 4 | 4 |
| Likelihood | Negative Binomial | Negative Binomial |
| Learning rate | 1×10⁻⁴ | 1×10⁻⁴ |
| Batch size | 128 | 128 |
| Max cells | 3,000 | 3,000 |
| HVG | 3,000 | 3,000 |
| MoCo K / τ | 4,096 / 0.2 | 4,096 / 0.2 |
| Prototypes / weight | 12 / 0.1 | 12 / 0.1 |
| Beta values | 1.0 (default) | 0.01, 0.1, 1.0 |

Configurations are defined in `mocoo/configs/beta_ablation.yaml` and
`mocoo/configs/default.yaml`, loaded via `mocoo.configs.load_config()`.

---

## Metrics

All metrics are computed through the unified `mocoo.evaluation` package:

- **Clustering (CLA):** ARI, NMI, ASW, Calinski-Harabasz (CAL), Davies-Bouldin (DAV), KNN Correlation (COR)
- **DRE:** Distance correlation, Q_local, Q_global, overall quality (UMAP-based)
- **LSE:** Manifold dimensionality, spectral decay, participation ratio, anisotropy, noise resilience
- **DREX:** Trustworthiness, continuity, distance Spearman/Pearson, local scale quality, neighborhood symmetry
- **LSEX:** Two-hop connectivity, radial concentration, local curvature, entropy stability
- **Diagnostics:** Mean norm, std stats, near-zero dims, pairwise distance stats

See `compute_all_metrics()` in `mocoo/evaluation/__init__.py` for the unified API.
