"""Comprehensive tests for MoCoO across all model configurations.

Tests cover:
- All 20 architecture configs: use_ode × use_moco × loss_mode
- Simulated and real data
- Metrics computation (ARI, NMI, ASW, CAL, DAV, COR) per config
- API method correctness
- Loss history structure
"""

import pytest
import numpy as np
import torch
import os
from anndata import AnnData
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.cluster import KMeans

from mocoo import MoCoO


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sim_adata():
    """Simulated Poisson count data with 3 cell types."""
    np.random.seed(42)
    torch.manual_seed(42)

    n_cells, n_genes = 120, 50
    labels = np.repeat(['A', 'B', 'C'], n_cells // 3)
    means = {'A': 3, 'B': 7, 'C': 12}
    X = np.vstack([
        np.random.poisson(means[l], size=(1, n_genes)) for l in labels
    ]).astype(np.float32)

    adata = AnnData(X=X)
    adata.obs['cell_type'] = labels
    return adata


@pytest.fixture
def real_adata():
    """Load a real scRNA-seq dataset (dentate gyrus) if available."""
    path = os.environ.get('MOCOO_TEST_DATA')
    if path is None:
        pytest.skip("MOCOO_TEST_DATA environment variable not set; skipping real-data test")
    if not os.path.exists(path):
        pytest.skip(f"Real dataset not found: {path}")

    import scanpy as sc
    adata = sc.read_h5ad(path)

    if adata.n_obs > 500:
        sc.pp.subsample(adata, n_obs=500, random_state=42)

    from scipy.sparse import issparse
    if issparse(adata.X):
        adata.X = adata.X.toarray()
    adata.X = adata.X.astype(np.float32)

    label_key = None
    for k in ['Clusters', 'clusters', 'cell_type', 'celltype']:
        if k in adata.obs.columns:
            label_key = k
            break
    if label_key and label_key != 'cell_type':
        adata.obs['cell_type'] = adata.obs[label_key]
    if label_key is None:
        adata.obs['cell_type'] = 'unknown'

    return adata


# ── Helpers ───────────────────────────────────────────────────────────────────

LOSS_MODES = ['mse', 'nb', 'zinb', 'poisson', 'zip']
ODE_FLAGS = [False, True]
MOCO_FLAGS = [False, True]

COMMON_KWARGS = dict(
    hidden_dim=16,
    latent_dim=5,
    i_dim=2,
    batch_size=16,
    train_size=0.6,
    val_size=0.2,
    test_size=0.2,
    lr=1e-3,
    grad_clip=1.0,
)


def _build_model(adata, use_ode, use_moco, loss_mode, **overrides):
    kw = {**COMMON_KWARGS, **overrides}
    if use_moco:
        kw.setdefault('moco_K', 64)
        kw.setdefault('aug_prob', 0.5)
        kw.setdefault('use_prototype', False)
        kw.setdefault('n_prototypes', 3)
    if use_ode:
        kw.setdefault('vae_reg', 0.5)
        kw.setdefault('ode_reg', 0.5)
    return MoCoO(
        adata=adata,
        layer='X',
        use_ode=use_ode,
        use_moco=use_moco,
        loss_mode=loss_mode,
        **kw,
    )


def _compute_metrics(latent, labels):
    """Compute clustering metrics following PanODE-LAB's compute_metrics pattern."""
    n_clusters = len(np.unique(labels))
    pred = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(latent)

    metrics = {}
    metrics['ARI'] = adjusted_rand_score(labels, pred)
    metrics['NMI'] = normalized_mutual_info_score(labels, pred)

    try:
        metrics['ASW'] = silhouette_score(latent, pred) if len(np.unique(pred)) > 1 else np.nan
    except Exception:
        metrics['ASW'] = np.nan

    try:
        metrics['CAL'] = calinski_harabasz_score(latent, pred)
    except Exception:
        metrics['CAL'] = np.nan

    try:
        metrics['DAV'] = davies_bouldin_score(latent, pred)
    except Exception:
        metrics['DAV'] = np.nan

    try:
        acorr = np.abs(np.corrcoef(latent.T))
        cor = acorr.sum(axis=1).mean() - 1
        metrics['COR'] = cor if np.isfinite(cor) else np.nan
    except Exception:
        metrics['COR'] = np.nan

    return metrics


def _compute_latent_diagnostics(latent):
    """Latent space diagnostics (collapse/redundancy) per PanODE-LAB."""
    z = np.asarray(latent)
    std = z.std(axis=0)
    return {
        'mean_norm': float(np.linalg.norm(z.mean(axis=0))),
        'std_mean': float(std.mean()),
        'std_min': float(std.min()),
        'near_zero_dims': int((std < 1e-3).sum()),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 1. PARAMETRIZED TESTS: all 20 configs on simulated data
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("loss_mode", LOSS_MODES)
@pytest.mark.parametrize("use_moco", MOCO_FLAGS, ids=["no_moco", "moco"])
@pytest.mark.parametrize("use_ode", ODE_FLAGS, ids=["no_ode", "ode"])
def test_config_train_and_latent(sim_adata, use_ode, use_moco, loss_mode):
    """Train 2 epochs and verify latent shape for every config."""
    model = _build_model(sim_adata, use_ode, use_moco, loss_mode)
    model.fit(epochs=2, patience=5, val_every=1)

    latent = model.get_latent()
    assert latent.shape == (sim_adata.n_obs, COMMON_KWARGS['latent_dim']), \
        f"Latent shape mismatch for ode={use_ode}, moco={use_moco}, loss={loss_mode}"
    assert np.isfinite(latent).all(), "Latent contains NaN/Inf"


@pytest.mark.parametrize("loss_mode", LOSS_MODES)
@pytest.mark.parametrize("use_moco", MOCO_FLAGS, ids=["no_moco", "moco"])
@pytest.mark.parametrize("use_ode", ODE_FLAGS, ids=["no_ode", "ode"])
def test_config_metrics(sim_adata, use_ode, use_moco, loss_mode):
    """Compute and validate metrics under every config."""
    from sklearn.preprocessing import LabelEncoder
    model = _build_model(sim_adata, use_ode, use_moco, loss_mode)
    model.fit(epochs=3, patience=5, val_every=1)

    latent = model.get_latent()
    labels = LabelEncoder().fit_transform(sim_adata.obs['cell_type'])
    metrics = _compute_metrics(latent, labels)

    assert -1.0 <= metrics['ARI'] <= 1.0, f"ARI={metrics['ARI']}"
    assert -1.0 <= metrics['NMI'] <= 1.0, f"NMI={metrics['NMI']}"
    if np.isfinite(metrics['ASW']):
        assert -1.0 <= metrics['ASW'] <= 1.0, f"ASW={metrics['ASW']}"
    if np.isfinite(metrics['DAV']):
        assert metrics['DAV'] >= 0, f"DAV={metrics['DAV']}"
    if np.isfinite(metrics['CAL']):
        assert metrics['CAL'] >= 0, f"CAL={metrics['CAL']}"


@pytest.mark.parametrize("loss_mode", LOSS_MODES)
@pytest.mark.parametrize("use_moco", MOCO_FLAGS, ids=["no_moco", "moco"])
@pytest.mark.parametrize("use_ode", ODE_FLAGS, ids=["no_ode", "ode"])
def test_config_loss_history(sim_adata, use_ode, use_moco, loss_mode):
    """Verify loss history structure under every config."""
    model = _build_model(sim_adata, use_ode, use_moco, loss_mode)
    model.fit(epochs=2, patience=5, val_every=1)

    assert len(model.loss) > 0, "No loss entries recorded"
    assert len(model.loss[-1]) == 10, f"Expected 10-tuple, got {len(model.loss[-1])}"

    for i, val in enumerate(model.loss[-1]):
        assert np.isfinite(val), f"Loss component {i} is not finite: {val}"


# ══════════════════════════════════════════════════════════════════════════════
# 2. ODE-SPECIFIC TESTS
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("loss_mode", LOSS_MODES)
def test_ode_outputs(sim_adata, loss_mode):
    """Verify ODE-specific outputs (time, velocity, transition)."""
    model = _build_model(sim_adata, use_ode=True, use_moco=False, loss_mode=loss_mode)
    model.fit(epochs=2, patience=5, val_every=1)

    time = model.get_time()
    velocity = model.get_velocity()
    transition = model.get_transition(top_k=10)

    assert time.shape == (sim_adata.n_obs,)
    assert velocity.shape == (sim_adata.n_obs, COMMON_KWARGS['latent_dim'])
    assert transition.shape == (sim_adata.n_obs, sim_adata.n_obs)

    row_sums = transition.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)


def test_ode_error_without_ode(sim_adata):
    """ODE-specific methods should raise when use_ode=False."""
    model = _build_model(sim_adata, use_ode=False, use_moco=False, loss_mode='mse')

    with pytest.raises(RuntimeError):
        model.get_time()
    with pytest.raises(RuntimeError):
        model.get_velocity()
    with pytest.raises(RuntimeError):
        model.get_transition()


# ══════════════════════════════════════════════════════════════════════════════
# 3. MoCo CONTRASTIVE TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_moco_loss_nonzero(sim_adata):
    """MoCo loss should be non-zero when enabled."""
    model = _build_model(sim_adata, use_ode=False, use_moco=True, loss_mode='nb')
    model.fit(epochs=3, patience=5, val_every=1)

    moco_losses = [l[6] for l in model.loss]
    assert any(m > 0 for m in moco_losses), "All MoCo losses are zero"


def test_cross_path_loss_with_ode_moco(sim_adata):
    """Cross-path contrastive should fire when ODE + MoCo are both on."""
    model = _build_model(sim_adata, use_ode=True, use_moco=True, loss_mode='nb')
    model.fit(epochs=3, patience=5, val_every=1)

    cross_losses = [l[7] for l in model.loss]
    vel_losses = [l[8] for l in model.loss]
    assert any(c > 0 for c in cross_losses), "Cross-path losses all zero with ODE+MoCo"
    assert any(v > 0 for v in vel_losses), "Velocity losses all zero with ODE"


# ══════════════════════════════════════════════════════════════════════════════
# 4. API METHOD TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_bottleneck(sim_adata):
    """Bottleneck should have i_dim columns."""
    model = _build_model(sim_adata, use_ode=False, use_moco=False, loss_mode='nb')
    btl = model.get_bottleneck()
    assert btl.shape == (sim_adata.n_obs, COMMON_KWARGS['i_dim'])


def test_test_latent(sim_adata):
    """Test split latent extraction."""
    model = _build_model(sim_adata, use_ode=False, use_moco=False, loss_mode='nb')
    test_lat = model.get_test_latent()
    n_test = int(COMMON_KWARGS['test_size'] * sim_adata.n_obs)
    assert test_lat.shape[0] == n_test or abs(test_lat.shape[0] - n_test) <= 1
    assert test_lat.shape[1] == COMMON_KWARGS['latent_dim']


def test_resource_metrics(sim_adata):
    """Resource metrics should be populated after fit."""
    model = _build_model(sim_adata, use_ode=False, use_moco=False, loss_mode='mse')
    model.fit(epochs=2, patience=5, val_every=1)

    res = model.get_resource_metrics()
    assert res['train_time'] > 0
    assert res['actual_epochs'] == 2


def test_metrics_history_keys(sim_adata):
    """Metrics history should contain expected keys."""
    model = _build_model(sim_adata, use_ode=False, use_moco=False, loss_mode='nb')
    model.fit(epochs=3, patience=5, val_every=1)

    hist = model.get_metrics_history()
    for key in ['ARI', 'NMI', 'ASW', 'CH', 'DB', 'Corr']:
        assert key in hist, f"Missing key: {key}"
        assert len(hist[key]) > 0


def test_loss_history_keys(sim_adata):
    """Loss history should contain total/train/val."""
    model = _build_model(sim_adata, use_ode=False, use_moco=False, loss_mode='mse')
    model.fit(epochs=3, patience=5, val_every=1)

    hist = model.get_loss_history()
    assert 'total' in hist
    assert 'train' in hist
    assert 'val' in hist


# ══════════════════════════════════════════════════════════════════════════════
# 5. LATENT DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("use_ode", ODE_FLAGS, ids=["no_ode", "ode"])
def test_latent_diagnostics(sim_adata, use_ode):
    """Latent space diagnostics: no collapse after 3 epochs."""
    model = _build_model(sim_adata, use_ode=use_ode, use_moco=False, loss_mode='nb')
    model.fit(epochs=3, patience=5, val_every=1)

    latent = model.get_latent()
    diag = _compute_latent_diagnostics(latent)

    assert diag['std_mean'] > 1e-4, f"Possible latent collapse: std_mean={diag['std_mean']}"
    assert diag['near_zero_dims'] < COMMON_KWARGS['latent_dim'], "Most dims near zero"


# ══════════════════════════════════════════════════════════════════════════════
# 6. REAL DATA TESTS
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("loss_mode", ['nb', 'mse'])
@pytest.mark.parametrize("use_ode", ODE_FLAGS, ids=["no_ode", "ode"])
def test_real_data_train(real_adata, use_ode, loss_mode):
    """Train on real scRNA-seq data and verify metrics."""
    from sklearn.preprocessing import LabelEncoder

    model = _build_model(
        real_adata, use_ode=use_ode, use_moco=False, loss_mode=loss_mode,
        hidden_dim=32, latent_dim=10, i_dim=4, batch_size=32,
    )
    model.fit(epochs=5, patience=10, val_every=2)

    latent = model.get_latent()
    assert latent.shape == (real_adata.n_obs, 10)
    assert np.isfinite(latent).all(), "Latent contains NaN/Inf on real data"

    labels = LabelEncoder().fit_transform(real_adata.obs['cell_type'])
    metrics = _compute_metrics(latent, labels)

    for key in ['ARI', 'NMI', 'ASW', 'CAL', 'DAV', 'COR']:
        assert key in metrics, f"Missing metric: {key}"

    diag = _compute_latent_diagnostics(latent)
    assert diag['std_mean'] > 0, "Latent collapsed on real data"


@pytest.mark.parametrize("use_moco", MOCO_FLAGS, ids=["no_moco", "moco"])
def test_real_data_moco(real_adata, use_moco):
    """Test MoCo on real data."""
    model = _build_model(
        real_adata, use_ode=False, use_moco=use_moco, loss_mode='nb',
        hidden_dim=32, latent_dim=10, i_dim=4, batch_size=32,
        moco_K=64, aug_prob=0.5,
    )
    model.fit(epochs=3, patience=10, val_every=1)

    latent = model.get_latent()
    assert np.isfinite(latent).all()
    assert latent.shape[0] == real_adata.n_obs


# ══════════════════════════════════════════════════════════════════════════════
# 7. COMPREHENSIVE METRICS TABLE (single combined test)
# ══════════════════════════════════════════════════════════════════════════════

def test_metrics_table_all_configs(sim_adata):
    """Run all 20 configs and collect metrics into a summary."""
    from sklearn.preprocessing import LabelEncoder

    labels = LabelEncoder().fit_transform(sim_adata.obs['cell_type'])
    results = []

    for use_ode in ODE_FLAGS:
        for use_moco in MOCO_FLAGS:
            for loss_mode in LOSS_MODES:
                model = _build_model(sim_adata, use_ode, use_moco, loss_mode)
                model.fit(epochs=2, patience=5, val_every=1)

                latent = model.get_latent()
                metrics = _compute_metrics(latent, labels)
                diag = _compute_latent_diagnostics(latent)

                results.append({
                    'use_ode': use_ode,
                    'use_moco': use_moco,
                    'loss_mode': loss_mode,
                    **metrics,
                    **diag,
                })

    assert len(results) == 20, f"Expected 20 configs, got {len(results)}"

    for r in results:
        assert np.isfinite(r['ARI']), f"Non-finite ARI: {r}"
        assert np.isfinite(r['NMI']), f"Non-finite NMI: {r}"
        assert r['std_mean'] > 0, f"Latent collapsed: {r}"


# ══════════════════════════════════════════════════════════════════════════════
# 8. DISENTANGLEMENT REGULARIZER CONFIGS
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("reg_name,reg_kwargs", [
    ("dip", {"dip": 1.0}),
    ("tc", {"tc": 1.0}),
    ("info", {"info": 1.0}),
    ("combined", {"dip": 0.5, "tc": 0.5, "info": 0.5}),
])
def test_regularizer_configs(sim_adata, reg_name, reg_kwargs):
    """Each disentanglement regularizer should train without error."""
    model = _build_model(
        sim_adata, use_ode=False, use_moco=False, loss_mode='nb', **reg_kwargs,
    )
    model.fit(epochs=2, patience=5, val_every=1)

    latent = model.get_latent()
    assert np.isfinite(latent).all(), f"Regularizer {reg_name} produced NaN/Inf"


# ══════════════════════════════════════════════════════════════════════════════
# 9. PanODE-LAB CONTRASTIVE STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════

def test_scgpcl_prototype_contrastive(sim_adata):
    """scGPCL prototype-level contrastive with learnable prototypes."""
    n_proto = 5
    model = _build_model(
        sim_adata, use_ode=False, use_moco=True, loss_mode='nb',
        use_prototype=True, n_prototypes=n_proto,
    )
    model.fit(epochs=2, patience=5, val_every=1)

    # Check prototypes exist and correct shape
    assert hasattr(model.nn.moco, 'prototypes'), "Prototypes should exist"
    assert model.nn.moco.prototypes.shape == (n_proto, COMMON_KWARGS['latent_dim'])

    # Test prototype loss
    z = torch.randn(16, COMMON_KWARGS['latent_dim']).to(model.device)
    loss = model.nn.moco.prototype_contrastive_loss(z)
    assert loss.item() > 0, "Prototype loss should be positive"


def test_cross_path_contrastive_ode_vae(sim_adata):
    """Cross-path contrastive aligns VAE and ODE paths."""
    model = _build_model(sim_adata, use_ode=True, use_moco=True, loss_mode='nb')
    model.fit(epochs=2, patience=5, val_every=1)

    q_z = torch.randn(16, COMMON_KWARGS['latent_dim']).to(model.device)
    q_z_ode = torch.randn(16, COMMON_KWARGS['latent_dim']).to(model.device)
    loss = model.nn.moco.cross_path_contrastive(q_z, q_z_ode)

    assert loss.item() > 0, "Cross-path loss should be positive"


def test_batchnorm_projection_heads(sim_adata):
    """Projection heads should use BatchNorm (per PanODE-LAB)."""
    model = _build_model(sim_adata, use_ode=False, use_moco=True, loss_mode='nb')
    
    # Check projection head structure
    proj_head = model.nn.moco.proj_head_q
    has_batchnorm = any(isinstance(m, torch.nn.BatchNorm1d) for m in proj_head.modules())
    assert has_batchnorm, "Projection heads should use BatchNorm1d"

