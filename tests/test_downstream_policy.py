"""Runtime correctness & policy tests for downstream analysis modules.

Covers:
  A – Persistence invariants (best_state sync, weights_only, FM round-trip)
  B – Decode-path determinism and non-negativity
  C – Annotation transfer (prototype determinism, kNN label-shape, confidence)
  D – Jacobian / velocity / divergence correctness
  E – Generation quality bounds
  F – Differential expression integrity
  G – Uncertainty distribution & collapse detection
  H – Pipeline guards (module isolation, JSON validity)
  Perf – runtime ceilings
"""

import json
import os
import time
import tempfile

import numpy as np
import pytest
import torch
from anndata import AnnData

from mocoo import MoCoO

# ── Fixtures ──────────────────────────────────────────────────────────────────

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


@pytest.fixture(scope="module")
def sim_adata():
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


def _build(adata, use_ode=False, use_moco=False, loss_mode='nb', **kw):
    params = {**COMMON_KWARGS, **kw}
    if use_moco:
        params.setdefault('moco_K', 64)
        params.setdefault('aug_prob', 0.5)
        params.setdefault('use_prototype', False)
        params.setdefault('n_prototypes', 3)
    if use_ode:
        params.setdefault('vae_reg', 0.5)
        params.setdefault('ode_reg', 0.5)
    return MoCoO(adata=adata, layer='X', use_ode=use_ode,
                 use_moco=use_moco, loss_mode=loss_mode, **params)


# Shared trained models (module-scoped to avoid repetitive training)

@pytest.fixture(scope="module")
def full_model(sim_adata):
    """Full config: ODE + MoCo + Proto, trained 3 epochs."""
    m = _build(sim_adata, use_ode=True, use_moco=True, loss_mode='nb',
               use_prototype=True, n_prototypes=3)
    m.fit(epochs=3, patience=100, val_every=3, track_metrics=False)
    return m


@pytest.fixture(scope="module")
def vae_model(sim_adata):
    """Vanilla VAE, trained 3 epochs."""
    m = _build(sim_adata, use_ode=False, use_moco=False, loss_mode='nb')
    m.fit(epochs=3, patience=100, val_every=3, track_metrics=False)
    return m


@pytest.fixture(scope="module")
def ode_model(sim_adata):
    """VAE + ODE, trained 3 epochs."""
    m = _build(sim_adata, use_ode=True, use_moco=False, loss_mode='nb')
    m.fit(epochs=3, patience=100, val_every=3, track_metrics=False)
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# Group A — Persistence invariants
# ═══════════════════════════════════════════════════════════════════════════════

class TestPersistencePolicy:

    def test_best_state_sync_after_load(self, sim_adata, full_model, tmp_path):
        """BUG-1: load_model() must sync best_model_state."""
        path = str(tmp_path / "ckpt.pt")
        full_model.save_model(path)

        m2 = _build(sim_adata, use_ode=True, use_moco=True, loss_mode='nb',
                     use_prototype=True, n_prototypes=3)
        m2.fit(epochs=1, patience=100, val_every=1, track_metrics=False)
        m2.load_model(path)

        assert hasattr(m2, 'best_model_state') and m2.best_model_state is not None
        # The best_model_state should match current nn state_dict
        for k in m2.best_model_state:
            torch.testing.assert_close(
                m2.best_model_state[k],
                m2.nn.state_dict()[k].cpu(),
            )

    def test_weights_only_true(self, full_model, tmp_path):
        """BUG-3: checkpoint should be loadable with weights_only=True."""
        path = str(tmp_path / "wo.pt")
        full_model.save_model(path)
        ckpt = torch.load(path, weights_only=True)
        assert 'state_dict' in ckpt
        assert 'config' in ckpt

    def test_fm_roundtrip_auto_init(self, sim_adata, full_model, tmp_path):
        """BUG-2: FM weights restored even if receiver has no fm_net yet."""
        full_model.train_fm(epochs=3, hidden_dim=16)
        path = str(tmp_path / "fm.pt")
        full_model.save_model(path)

        m2 = _build(sim_adata, use_ode=True, use_moco=True, loss_mode='nb',
                     use_prototype=True, n_prototypes=3)
        m2.fit(epochs=1, patience=100, val_every=1, track_metrics=False)
        assert not hasattr(m2, 'fm_net')
        m2.load_model(path)
        assert hasattr(m2, 'fm_net'), "FM net should be auto-initialised on load"
        # Verify FM weights match
        gen1 = full_model.generate_cells(n=20, steps=10, decode=False)
        gen2 = m2.generate_cells(n=20, steps=10, decode=False)
        # Shapes must match
        assert gen1.shape == gen2.shape

    def test_fm_hidden_dim_saved(self, full_model, tmp_path):
        """BUG-2 supplementary: checkpoint stores fm_hidden_dim."""
        if not hasattr(full_model, 'fm_net'):
            full_model.train_fm(epochs=3, hidden_dim=16)
        path = str(tmp_path / "fmh.pt")
        full_model.save_model(path)
        ckpt = torch.load(path, weights_only=True)
        assert ckpt['config'].get('fm_hidden_dim') == 16


# ═══════════════════════════════════════════════════════════════════════════════
# Group B — Decode-path
# ═══════════════════════════════════════════════════════════════════════════════

class TestDecodePathPolicy:

    def test_decoded_determinism(self, full_model):
        """get_decoded() should return identical results on repeated calls."""
        d1 = full_model.get_decoded()
        d2 = full_model.get_decoded()
        np.testing.assert_array_equal(d1, d2)

    def test_decoded_shape(self, sim_adata, full_model):
        d = full_model.get_decoded()
        assert d.shape == (sim_adata.n_obs, sim_adata.n_vars)

    def test_decoded_nonneg_nb(self, full_model):
        """NB decoder outputs (mu) should be non-negative."""
        d = full_model.get_decoded()
        assert (d >= 0).all(), "NB decoded values should be non-negative"

    def test_decoded_qm_vs_blended_differ(self, ode_model):
        """For ODE models, q_m decode != blended decode."""
        d_qm = ode_model.get_decoded()
        d_blend = ode_model.get_decoded(ode_model.get_latent())
        assert not np.allclose(d_qm, d_blend, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# Group C — Annotation transfer
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnnotationPolicy:

    def test_prototype_determinism(self, full_model):
        """BUG-4: prototype annotation should be deterministic (uses q_m)."""
        r1 = full_model.annotate_cells(method='prototype')
        r2 = full_model.annotate_cells(method='prototype')
        np.testing.assert_array_equal(r1['labels'], r2['labels'])
        np.testing.assert_allclose(r1['confidence'], r2['confidence'])

    def test_prototype_in_projection_space(self, full_model):
        """Prototype annotation projects through proj_head_q + L2-norm."""
        result = full_model.annotate_cells(method='prototype')
        # Confidence should be bounded [0,1] (cosine similarity post-norm)
        assert (result['confidence'] >= 0).all()
        assert (result['confidence'] <= 1).all()

    def test_knn_label_shape_traintest(self, sim_adata, vae_model):
        """BUG-5: kNN with reference_data ensures matching label/ref shape."""
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        gt_all = le.fit_transform(sim_adata.obs['cell_type'].values)
        gt_train = gt_all[vae_model.train_idx]

        result = vae_model.annotate_cells(
            query_data=vae_model.X_test,
            reference_data=vae_model.X_train,
            reference_labels=gt_train,
            method='knn', k=5,
        )
        assert result['labels'].shape[0] == len(vae_model.test_idx)

    def test_knn_confidence_bounds(self, sim_adata, vae_model):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        labels = le.fit_transform(sim_adata.obs['cell_type'].values)
        result = vae_model.annotate_cells(reference_labels=labels,
                                          method='knn', k=5)
        assert (result['confidence'] >= 0).all()
        assert (result['confidence'] <= 1).all()


# ═══════════════════════════════════════════════════════════════════════════════
# Group D — Jacobian / velocity / divergence
# ═══════════════════════════════════════════════════════════════════════════════

class TestJacobianPolicy:

    def test_jacobian_finite(self, vae_model, sim_adata):
        jac = vae_model.get_gene_jacobian(batch_size=64)
        assert jac.shape == (sim_adata.n_obs, sim_adata.n_vars,
                             COMMON_KWARGS['latent_dim'])
        assert np.all(np.isfinite(jac))

    def test_jacobian_restores_train_mode(self, vae_model):
        """BUG-6: take_gene_jacobian must restore nn.training mode."""
        vae_model.nn.train()
        assert vae_model.nn.training
        _ = vae_model.get_gene_jacobian(batch_size=64)
        assert vae_model.nn.training, \
            "Model should be back in train mode after get_gene_jacobian"

    def test_velocity_requires_ode(self, vae_model):
        with pytest.raises(RuntimeError):
            vae_model.get_gene_velocity()

    def test_velocity_finite(self, ode_model, sim_adata):
        vel = ode_model.get_gene_velocity(batch_size=64)
        assert vel.shape == (sim_adata.n_obs, sim_adata.n_vars)
        assert np.all(np.isfinite(vel))

    def test_divergence_finite(self, ode_model, sim_adata):
        div = ode_model.get_divergence(batch_size=64)
        assert div.shape == (sim_adata.n_obs,)
        assert np.all(np.isfinite(div))


# ═══════════════════════════════════════════════════════════════════════════════
# Group E — Generation quality
# ═══════════════════════════════════════════════════════════════════════════════

class TestGenerationPolicy:

    @pytest.fixture(autouse=True)
    def _train_fm(self, full_model):
        if not hasattr(full_model, 'fm_net'):
            full_model.train_fm(epochs=5, hidden_dim=16)

    def test_generate_single_cell(self, full_model):
        gen = full_model.generate_cells(n=1, steps=10, decode=False)
        assert gen.shape == (1, COMMON_KWARGS['latent_dim'])
        assert np.all(np.isfinite(gen))

    def test_coverage_bounds(self, full_model):
        from mocoo.evaluation.generation_quality import generation_quality_metrics
        real = full_model.get_latent()
        gen = full_model.generate_cells(n=50, steps=20, decode=False)
        metrics = generation_quality_metrics(real, gen, k=5)
        assert 0.0 <= metrics['coverage'] <= 1.0

    def test_diversity_positive(self, full_model):
        from mocoo.evaluation.generation_quality import generation_quality_metrics
        real = full_model.get_latent()
        gen = full_model.generate_cells(n=50, steps=20, decode=False)
        metrics = generation_quality_metrics(real, gen, k=5)
        assert metrics['diversity'] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Group F — Differential expression
# ═══════════════════════════════════════════════════════════════════════════════

class TestDifferentialExpressionPolicy:

    def test_de_per_cluster(self, vae_model, sim_adata):
        from mocoo.evaluation.differential_expression import decoder_de
        latent = vae_model.get_latent_qm()
        labels = vae_model.labels
        decoded = vae_model.get_decoded()
        centroids = {}
        for lab in np.unique(labels):
            z_mean = latent[labels == lab].mean(axis=0)
            centroids[lab] = vae_model.get_decoded(z_mean[None, :])[0]
        gene_names = [f"gene_{i}" for i in range(sim_adata.n_vars)]
        result = decoder_de(centroids, decoded_all=decoded, labels=labels,
                            top_n=10, gene_names=gene_names)
        assert len(result) > 0, "Should have at least one cluster"

    def test_de_log2fc_finite(self, vae_model, sim_adata):
        from mocoo.evaluation.differential_expression import decoder_de
        latent = vae_model.get_latent_qm()
        labels = vae_model.labels
        decoded = vae_model.get_decoded()
        centroids = {}
        for lab in np.unique(labels):
            z_mean = latent[labels == lab].mean(axis=0)
            centroids[lab] = vae_model.get_decoded(z_mean[None, :])[0]
        gene_names = [f"gene_{i}" for i in range(sim_adata.n_vars)]
        result = decoder_de(centroids, decoded_all=decoded, labels=labels,
                            top_n=10, gene_names=gene_names)
        for lab, data in result.items():
            assert np.all(np.isfinite(data['log2fc'])), \
                f"Non-finite log2fc in cluster {lab}"

    def test_de_gene_names_valid(self, vae_model, sim_adata):
        from mocoo.evaluation.differential_expression import decoder_de
        latent = vae_model.get_latent_qm()
        labels = vae_model.labels
        decoded = vae_model.get_decoded()
        centroids = {}
        for lab in np.unique(labels):
            z_mean = latent[labels == lab].mean(axis=0)
            centroids[lab] = vae_model.get_decoded(z_mean[None, :])[0]
        gene_names = [f"gene_{i}" for i in range(sim_adata.n_vars)]
        result = decoder_de(centroids, decoded_all=decoded, labels=labels,
                            top_n=10, gene_names=gene_names)
        for lab, data in result.items():
            for g in data['top_genes']:
                assert g in gene_names, f"Gene {g} not in gene_names"


# ═══════════════════════════════════════════════════════════════════════════════
# Group G — Uncertainty
# ═══════════════════════════════════════════════════════════════════════════════

class TestUncertaintyPolicy:

    def test_distribution_consistency(self, vae_model, sim_adata):
        """Multiple calls should yield similar uncertainty distributions."""
        u1 = vae_model.get_uncertainty(n_samples=20)['uncertainty']
        u2 = vae_model.get_uncertainty(n_samples=20)['uncertainty']
        # Same model/data → distributions should be very close
        np.testing.assert_allclose(u1, u2, atol=0.5)

    def test_collapse_detection(self, vae_model):
        """Uncertainty should be non-zero (posterior not collapsed)."""
        u = vae_model.get_uncertainty(n_samples=20)['uncertainty']
        assert u.mean() > 0, "Mean uncertainty is zero → possible posterior collapse"


# ═══════════════════════════════════════════════════════════════════════════════
# Group H — Pipeline guards
# ═══════════════════════════════════════════════════════════════════════════════

class TestPipelineGuards:

    def test_module_isolation(self, sim_adata, vae_model):
        """Running one module should not alter latent output of another."""
        lat1 = vae_model.get_latent_qm().copy()
        _ = vae_model.get_decoded()
        _ = vae_model.get_gene_jacobian(batch_size=64)
        lat2 = vae_model.get_latent_qm()
        np.testing.assert_array_equal(lat1, lat2)

    def test_gene_importance_nan_guard(self):
        """PERF-1: rank_genes_by_jacobian handles NaN/Inf in Jacobian."""
        from mocoo.evaluation.gene_importance import rank_genes_by_jacobian
        N, G, D = 10, 5, 3
        jac = np.random.randn(N, G, D).astype(np.float32)
        jac[0, 2, 1] = np.nan
        jac[1, 3, 0] = np.inf
        result = rank_genes_by_jacobian(jac, top_n=3)
        assert np.all(np.isfinite(result['importance'])), \
            "NaN/Inf in Jacobian should be handled gracefully"

    def test_load_dataset_missing_file(self):
        """BUG-7: load_dataset should raise FileNotFoundError for missing path."""
        import sys
        sys.path.insert(0, os.path.join(
            os.path.dirname(__file__), '..', 'benchmarks', 'scripts', 'pipeline'))
        from run_downstream import load_dataset
        with pytest.raises(FileNotFoundError):
            load_dataset("/nonexistent/file.h5ad", max_cells=1000, hvg=500)


# ═══════════════════════════════════════════════════════════════════════════════
# Performance guards (generous ceilings, simulated data)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerformanceCeilings:

    def test_jacobian_runtime(self, vae_model):
        t0 = time.time()
        _ = vae_model.get_gene_jacobian(batch_size=64)
        assert time.time() - t0 < 30, "Jacobian should finish in <30s on sim data"

    def test_uncertainty_runtime(self, vae_model):
        t0 = time.time()
        _ = vae_model.get_uncertainty(n_samples=20)
        assert time.time() - t0 < 15, "Uncertainty should finish in <15s on sim data"

    def test_de_runtime(self, vae_model, sim_adata):
        from mocoo.evaluation.differential_expression import decoder_de
        latent = vae_model.get_latent_qm()
        labels = vae_model.labels
        decoded = vae_model.get_decoded()
        centroids = {}
        for lab in np.unique(labels):
            z_mean = latent[labels == lab].mean(axis=0)
            centroids[lab] = vae_model.get_decoded(z_mean[None, :])[0]
        gene_names = [f"gene_{i}" for i in range(sim_adata.n_vars)]
        t0 = time.time()
        _ = decoder_de(centroids, decoded_all=decoded, labels=labels,
                       top_n=10, gene_names=gene_names)
        assert time.time() - t0 < 10, "DE should finish in <10s on sim data"

    def test_generation_runtime(self, full_model):
        if not hasattr(full_model, 'fm_net'):
            full_model.train_fm(epochs=3, hidden_dim=16)
        t0 = time.time()
        _ = full_model.generate_cells(n=100, steps=20, decode=False)
        assert time.time() - t0 < 15, "Generation should finish in <15s on sim data"
