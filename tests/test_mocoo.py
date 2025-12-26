"""Tests for MoCoO package."""

import pytest
import numpy as np
import torch
from anndata import AnnData

from mocoo import MoCoO


@pytest.fixture
def sample_adata():
    """Create sample AnnData for testing."""
    np.random.seed(42)
    torch.manual_seed(42)

    n_cells, n_genes = 100, 50
    X = np.random.poisson(5, size=(n_cells, n_genes)).astype(np.float32)

    adata = AnnData(X=X)
    adata.obs['cell_type'] = np.random.choice(['A', 'B', 'C'], n_cells)

    return adata


def test_mocoo_initialization(sample_adata):
    """Test MoCoO initialization."""
    model = MoCoO(
        adata=sample_adata,
        layer='X',
        latent_dim=10,
        hidden_dim=32,
        batch_size=16,
        use_ode=False,
        use_moco=False,
    )

    assert model is not None
    assert hasattr(model, 'fit')
    assert hasattr(model, 'get_latent')


def test_mocoo_basic_training(sample_adata):
    """Test basic training without ODE/MoCo."""
    model = MoCoO(
        adata=sample_adata,
        layer='X',
        latent_dim=5,
        hidden_dim=16,
        batch_size=16,
        use_ode=False,
        use_moco=False,
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
    )

    # Quick training
    model.fit(epochs=2, patience=5, val_every=1)

    latent = model.get_latent()
    assert latent.shape[0] == sample_adata.n_obs
    assert latent.shape[1] == 5


def test_mocoo_with_ode(sample_adata):
    """Test MoCoO with ODE enabled."""
    model = MoCoO(
        adata=sample_adata,
        layer='X',
        latent_dim=5,
        hidden_dim=16,
        batch_size=16,
        use_ode=True,
        use_moco=False,
        vae_reg=0.5,
        ode_reg=0.5,
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
    )

    # Quick training
    model.fit(epochs=2, patience=5, val_every=1)

    latent = model.get_latent()
    time = model.get_time()
    velocity = model.get_velocity()

    assert latent.shape[0] == sample_adata.n_obs
    assert latent.shape[1] == 5
    assert time.shape[0] == sample_adata.n_obs
    assert velocity.shape[0] == sample_adata.n_obs


def test_mocoo_with_moco(sample_adata):
    """Test MoCoO with MoCo enabled."""
    model = MoCoO(
        adata=sample_adata,
        layer='X',
        latent_dim=5,
        hidden_dim=16,
        batch_size=16,
        use_ode=False,
        use_moco=True,
        moco_K=256,
        aug_prob=0.5,
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
    )

    # Quick training
    model.fit(epochs=2, patience=5, val_every=1)

    latent = model.get_latent()
    assert latent.shape[0] == sample_adata.n_obs
    assert latent.shape[1] == 5


def test_mocoo_api_methods(sample_adata):
    """Test all API methods."""
    model = MoCoO(
        adata=sample_adata,
        layer='X',
        latent_dim=5,
        hidden_dim=16,
        batch_size=16,
        use_ode=False,
        use_moco=False,
    )

    # Test before training
    latent = model.get_latent()
    bottleneck = model.get_bottleneck()
    test_latent = model.get_test_latent()

    assert latent.shape[0] == sample_adata.n_obs
    assert bottleneck.shape[0] == sample_adata.n_obs
    assert test_latent.shape[1] == 5

    # Test metrics
    loss_hist = model.get_loss_history()
    metrics_hist = model.get_metrics_history()
    resources = model.get_resource_metrics()

    assert isinstance(loss_hist, dict)
    assert isinstance(metrics_hist, dict)
    assert isinstance(resources, dict)