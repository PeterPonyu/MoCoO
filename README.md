# MoCoO

**Mo**mentum **Co**ntrast **O**DE-Regularized VAE for Single-Cell RNA Velocity

A unified deep learning framework combining Variational Autoencoders (VAE), Neural Ordinary Differential Equations (ODE), and Momentum Contrast (MoCo) for robust single-cell trajectory inference and representation learning.

---

## Features

- **VAE-based dimensionality reduction** with multiple count-based likelihoods (MSE, NB, ZINB, Poisson, ZIP)
- **Neural ODE** for continuous trajectory modeling and pseudotime inference
- **Momentum Contrast (MoCo)** for robust contrastive representation learning
- **Information bottleneck** for hierarchical feature extraction
- **Disentanglement losses** (DIP-VAE, β-TC-VAE, InfoVAE) for interpretable latents
- **Vector field analysis** for RNA velocity visualization

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
pip install -e ".[dev]"
```

### Publishing

The package is automatically published to PyPI when a GitHub release is created.

**To create a new release:**

1. **Bump version:**
   ```bash
   python release.py patch  # For bug fixes (0.0.1 → 0.0.2)
   python release.py minor  # For new features (0.0.1 → 0.1.0)
   python release.py major  # For breaking changes (0.0.1 → 1.0.0)
   ```

2. **Commit and push:**
   ```bash
   git add -A
   git commit -m "Bump version to X.Y.Z"
   git push
   ```

3. **Create GitHub release:**
   - Go to [Releases](https://github.com/PeterPonyu/MoCoO/releases)
   - Click "Create a new release"
   - Tag: `vX.Y.Z` (e.g., `v0.1.0`)
   - Title: `Release X.Y.Z`
   - Description: List changes
   - Click "Publish release"

4. **Automated publishing:**
   - GitHub Actions will automatically build and publish to PyPI
   - Check the Actions tab for build status

---

## Quick Start

### Basic VAE

```python
import scanpy as sc
from mocoo import MoCoO

adata = sc.read_h5ad('data.h5ad')

model = MoCoO(
    adata,
    layer='counts',
    loss_mode='nb',
    batch_size=128
)
model.fit(epochs=100)

latent = model.get_latent()
adata.obsm['X_mocoo'] = latent
```

### With ODE + MoCo

```python
model = MoCoO(
    adata,
    use_ode=True,
    use_moco=True,
    latent_dim=10,
    i_dim=2,
    moco_K=4096,
    aug_prob=0.5,
    batch_size=256
)
model.fit(epochs=400, patience=25)

latent = model.get_latent()
velocity = model.get_velocity()
pseudotime = model.get_time()
transition = model.get_transition(top_k=30)
```

---

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adata` | AnnData | required | Annotated data matrix |
| `layer` | str | `'counts'` | Layer containing raw counts |
| `loss_mode` | str | `'nb'` | Likelihood: `'mse'`, `'nb'`, `'zinb'`, `'poisson'`, `'zip'` |
| `latent_dim` | int | `10` | Latent space dimension |
| `i_dim` | int | `2` | Bottleneck dimension (< latent_dim) |
| `use_ode` | bool | `False` | Enable Neural ODE |
| `use_moco` | bool | `False` | Enable MoCo |
| `moco_K` | int | `4096` | MoCo queue size |
| `batch_size` | int | `128` | Mini-batch size |
| `lr` | float | `1e-4` | Learning rate |

See docstrings for complete parameter list.

---

## API

### Training
```python
model.fit(epochs=400, patience=25, val_every=5)
```

### Inference
```python
latent = model.get_latent()           # Latent embeddings
bottleneck = model.get_bottleneck()   # Bottleneck features
time = model.get_time()               # Pseudotime (ODE only)
velocity = model.get_velocity()       # RNA velocity (ODE only)
transition = model.get_transition()   # Transition matrix (ODE only)
```

### Metrics
```python
loss_hist = model.get_loss_history()
metrics_hist = model.get_metrics_history()
resources = model.get_resource_metrics()
```

---

## Architecture

```
Input (n_genes)
    ↓
Encoder (log1p → MLP → latent_dim)
    ↓
[Optional ODE] Neural ODE dynamics
    ↓
Bottleneck (latent_dim → i_dim → latent_dim)
    ↓
Decoder (MLP → n_genes)
    ↓
Reconstruction (NB/ZINB/MSE/Poisson/ZIP)

[Optional MoCo] Contrastive learning on augmented views
```

---

## Loss Functions

- **Reconstruction**: MSE, NB, ZINB, Poisson, ZIP
- **KL Divergence**: β-weighted regularization
- **Disentanglement**: DIP-VAE, β-TC-VAE, InfoVAE (MMD)
- **ODE Regularization**: MSE between VAE and ODE latents
- **MoCo Contrastive**: InfoNCE loss

---

## Validation Metrics

- **ARI**: Adjusted Rand Index
- **NMI**: Normalized Mutual Information
- **ASW**: Silhouette Score
- **CH**: Calinski-Harabasz Index
- **DB**: Davies-Bouldin Index
- **Corr**: Latent correlation

---

## Citation

```bibtex
@article{mocoo2025,
  title={MoCoO: Momentum Contrast ODE-Regularized VAE for Single-Cell Trajectory Inference},
  author={Ponyu, Peter},
  year={2025}
}
```

---

## License

MIT License

---

## Contact

GitHub: [@PeterPonyu](https://github.com/PeterPonyu)  
Repository: [MoCoO](https://github.com/PeterPonyu/MoCoO)