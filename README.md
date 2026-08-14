# MoCoO

**Mo**mentum **Co**ntrast **O**DE-Regularized VAE for Single-Cell RNA Velocity

PyTorch package: a count VAE with optional Momentum Contrast and optional Neural ODE heads for single-cell embeddings.

---

## Components

- Count VAE with MSE, NB, ZINB, Poisson, ZIP likelihoods
- Optional Neural ODE head (API; not a validated trajectory proof)
- Optional Momentum Contrast (MoCo) on augmented views
- Information bottleneck (`latent_dim` → `i_dim`)
- Optional disentanglement losses (DIP-VAE, β-TC-VAE, InfoVAE)
- Vector-field export for RNA-velocity plots

---

## Installation

### From PyPI

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

## License

MIT License

---

## Contact

GitHub: [@PeterPonyu](https://github.com/PeterPonyu)  
Repository: [MoCoO](https://github.com/PeterPonyu/MoCoO)
