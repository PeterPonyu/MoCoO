<div align="center">
  <a href="https://peterponyu.github.io/">
    <img src="https://peterponyu.github.io/assets/badges/MoCoO.svg" width="64" alt="ZF Lab · MoCoO">
  </a>
</div>

# MoCoO

[![PyPI version](https://img.shields.io/pypi/v/mocoo.svg)](https://pypi.org/project/mocoo/)
[![Python versions](https://img.shields.io/pypi/pyversions/mocoo.svg)](https://pypi.org/project/mocoo/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://github.com/PeterPonyu/MoCoO/actions/workflows/test.yml/badge.svg)](https://github.com/PeterPonyu/MoCoO/actions/workflows/test.yml)

**Mo**mentum **Co**ntrast **O**DE-Regularized VAE for Single-Cell RNA Velocity

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Benchmarking Pipeline](#benchmarking-pipeline)
- [Local Manuscript Workspace](#local-manuscript-workspace)
- [API Reference](#api-reference)
- [Citation](#citation)
- [License](#license)

---

## Overview

MoCoO is a unified deep learning framework that combines Variational Autoencoders (VAE), Neural Ordinary Differential Equations (ODE), and Momentum Contrast (MoCo) for robust single-cell trajectory inference and representation learning. Key capabilities include:

- **VAE-based dimensionality reduction** with multiple count-based likelihoods (MSE, NB, ZINB, Poisson, ZIP)
- **Neural ODE** for continuous trajectory modeling and pseudotime inference
- **Momentum Contrast (MoCo)** for contrastive representation learning
- **Disentanglement losses** (DIP-VAE, beta-TC-VAE, InfoVAE) for interpretable latents
- **Novel evaluation metrics** (DRE, LSE) for latent space quality assessment

---

## Installation

### From PyPI (recommended)

```bash
pip install mocoo
```

### From source

```bash
git clone https://github.com/PeterPonyu/MoCoO.git
cd MoCoO
pip install -e .
```

### Development installation

```bash
git clone https://github.com/PeterPonyu/MoCoO.git
cd MoCoO
pip install -e ".[dev,benchmark]"
```

---

## Quick Start

```python
import scanpy as sc
from mocoo import MoCoO

adata = sc.read_h5ad("data.h5ad")

# Full model: VAE + ODE + MoCo
model = MoCoO(
    adata,
    use_ode=True,
    use_moco=True,
    latent_dim=10,
    loss_mode="nb",
    batch_size=256,
)
model.fit(epochs=400, patience=60)

latent = model.get_latent()          # latent embeddings
velocity = model.get_velocity()      # RNA velocity (ODE)
pseudotime = model.get_time()        # pseudotime (ODE)
transition = model.get_transition()  # transition matrix (ODE)
```

---

## Project Structure

```text
MoCoO/
├── mocoo/                      # Core Python package
│   ├── agent.py                # MoCoO model API (training, inference)
│   ├── model.py                # VAE + ODE + MoCo architecture
│   ├── module.py               # Encoder, decoder, ODE, MoCo modules
│   ├── mixin.py                # ODE trajectory analysis mixins
│   ├── environment.py          # Data loading and preprocessing
│   ├── configs/                # YAML experiment configurations
│   ├── evaluation/             # Latent-space evaluation metrics (DRE, LSE, ...)
│   └── visualization/          # Publication-quality figure generation
├── benchmarks/                 # Benchmark scripts plus local results/figures
│   └── scripts/                # Pipeline, evaluation, and plotting scripts
├── tests/                      # Test suite
├── Makefile                    # Top-level pipeline orchestration
└── pyproject.toml              # Build and dependency configuration
```

Generated benchmark outputs, figures, and any local manuscript workspace are intentionally kept out of the published source tree via `.gitignore`.

---

## Configuration

The `mocoo.configs` module provides centralized, version-controlled experiment configurations stored as YAML files. All hyperparameters for the ablation study are defined once and loaded through a Python API.

```python
from mocoo.configs import load_config, get_shared_params, get_model_configs

cfg = load_config("default")            # or "beta_ablation"
shared = get_shared_params(cfg)          # epochs, patience, lr, etc.
models = get_model_configs(cfg)          # per-config overrides (VAE, VAE+ODE, Full, ...)
full_params = {**shared, **models["Full"]}
```

Additional helpers: `get_training_params`, `get_moco_params`, `get_loss_weights`, `get_dataset_paths`, `get_sweep_params`.

---

## Evaluation

The `mocoo.evaluation` module provides two evaluator classes for quantifying latent-space quality.

**DimensionalityReductionEvaluator (DRE)** -- assesses how well a reduction method preserves high-dimensional structure via distance correlation, Q_local, and Q_global metrics.

```python
from mocoo.evaluation import evaluate_dimensionality_reduction

results = evaluate_dimensionality_reduction(X_high, X_low)
```

**SingleCellLatentSpaceEvaluator (LSE)** -- evaluates latent-space quality for single-cell data, including trajectory directionality, spectral decay, and participation ratio.

```python
from mocoo.evaluation import evaluate_single_cell_latent_space

results = evaluate_single_cell_latent_space(latent, labels, data_type="trajectory")
```

Both evaluators expose `compare_*` convenience functions for multi-method comparison returning a pandas DataFrame.

---

## Visualization

The `mocoo.visualization` module generates publication-quality figures from benchmark results. It provides individual plot functions and a batch `FigurePipeline`.

**Individual plot functions** (each returns a `matplotlib.Figure`):

| Function | Description |
| --- | --- |
| `plot_ablation_radar` | Normalized multi-metric dot-strip chart |
| `plot_metric_bars` | Grouped bar chart (val + test overlay) |
| `plot_umap_grid` | 2-row x 3-col UMAP scatter grid |
| `plot_training_curves` | Loss convergence curves with metric evolution |
| `plot_pseudotime_markers` | Marker-gene correlation with pseudotime |
| `plot_beta_sensitivity` | Beta (KL weight) sensitivity sweep |

**FigurePipeline** -- batch generator for all paper figures:

```python
from mocoo.visualization import FigurePipeline

pipe = FigurePipeline("benchmarks/results/IRALL", "figures/")
pipe.load_results()
pipe.generate_all()           # all six figure groups
pipe.generate_figure("ablation")  # single figure by name
```

---

## Benchmarking Pipeline

The top-level `Makefile` orchestrates the full reproducibility pipeline from data to figures. Run `make help` for the complete target listing. Key targets:

```bash
make all              # Full pipeline: test -> benchmark -> figures -> optional paper step
make test             # Run pytest suite
make figures          # Generate all paper figures
make paper            # Build a local manuscript if paper/main.tex exists
```

All variables are overridable: `make benchmark EPOCHS=300 PATIENCE=50 MAX_CELLS=5000`.

---

## API Reference

### MoCoO (main class)

| Method | Description |
| --- | --- |
| `MoCoO(adata, ...)` | Initialize model with AnnData and hyperparameters |
| `.fit(epochs, patience, val_every)` | Train the model |
| `.get_latent()` | Extract latent embeddings |
| `.get_time()` | Extract pseudotime (ODE only) |
| `.get_velocity()` | Extract RNA velocity (ODE only) |
| `.get_transition(top_k)` | Compute transition matrix (ODE only) |
| `.get_loss_history()` | Training loss history |
| `.get_metrics_history()` | Validation metrics history |
| `.get_resource_metrics()` | Runtime and memory usage |

### Key Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `adata` | AnnData | required | Annotated data matrix |
| `layer` | str | `'counts'` | Layer containing raw counts |
| `loss_mode` | str | `'nb'` | Likelihood: `'mse'`, `'nb'`, `'zinb'`, `'poisson'`, `'zip'` |
| `latent_dim` | int | `10` | Latent space dimension |
| `use_ode` | bool | `False` | Enable Neural ODE |
| `use_moco` | bool | `False` | Enable Momentum Contrast |
| `moco_K` | int | `4096` | MoCo queue size |
| `batch_size` | int | `128` | Mini-batch size |
| `lr` | float | `1e-4` | Learning rate |

See class docstrings for the complete parameter list.

---

## Citation

```bibtex
@article{Fu2026MoCoO,
  author  = {Fu, Zeyu},
  title   = {MoCoO: Momentum Contrast ODE-Regularized VAE for Single-Cell Trajectory Inference},
  year    = {2026},
  note    = {Preprint},
  doi     = {10.64898/2026.03.27.714791},
  url     = {https://doi.org/10.64898/2026.03.27.714791}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
