"""Tests for mocoo.evaluation subpackage."""
import numpy as np
import pytest
from mocoo.evaluation import (
    DimensionalityReductionEvaluator,
    SingleCellLatentSpaceEvaluator,
    compute_all_metrics,
    compute_clustering_metrics,
    compute_dre_metrics,
    compute_lse_metrics,
    compute_drex_metrics,
    compute_lsex_metrics,
    compute_latent_diagnostics,
    CORE_METRICS,
    ALL_METRIC_GROUPS,
)


class TestDRE:
    """Test DimensionalityReductionEvaluator."""

    def setup_method(self):
        np.random.seed(42)
        self.n_cells = 200
        self.latent_dim = 32
        self.latent = np.random.randn(self.n_cells, self.latent_dim).astype(np.float32)
        self.embedding_2d = np.random.randn(self.n_cells, 2).astype(np.float32)

    def test_instantiation(self):
        evaluator = DimensionalityReductionEvaluator(verbose=False)
        assert evaluator is not None

    def test_evaluate_returns_dict(self):
        evaluator = DimensionalityReductionEvaluator(verbose=False)
        result = evaluator.comprehensive_evaluation(self.latent, self.embedding_2d)
        assert isinstance(result, dict)
        # Check for expected keys
        assert "overall_quality" in result
        assert "distance_correlation" in result
        assert "Q_local" in result
        assert "Q_global" in result

    def test_different_k_values(self):
        evaluator = DimensionalityReductionEvaluator(verbose=False)
        result = evaluator.comprehensive_evaluation(self.latent, self.embedding_2d, k=10)
        assert isinstance(result, dict)

    def test_mismatched_samples_raises(self):
        bad_embedding = np.random.randn(self.n_cells + 10, 2).astype(np.float32)
        evaluator = DimensionalityReductionEvaluator(verbose=False)
        with pytest.raises((ValueError, Exception)):
            evaluator.comprehensive_evaluation(self.latent, bad_embedding)

    def test_distance_correlation_score(self):
        evaluator = DimensionalityReductionEvaluator(verbose=False)
        score = evaluator.distance_correlation_score(self.latent, self.embedding_2d)
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0

    def test_compare_methods(self):
        evaluator = DimensionalityReductionEvaluator(verbose=False)
        emb_a = np.random.randn(self.n_cells, 2).astype(np.float32)
        emb_b = np.random.randn(self.n_cells, 2).astype(np.float32)
        method_dict = {
            "MethodA": (self.latent, emb_a),
            "MethodB": (self.latent, emb_b),
        }
        df = evaluator.compare_methods(method_dict, k=10)
        assert len(df) == 2
        assert "Overall_Quality" in df.columns

    def test_convenience_function(self):
        from mocoo.evaluation import evaluate_dimensionality_reduction
        result = evaluate_dimensionality_reduction(
            self.latent, self.embedding_2d, k=10, verbose=False
        )
        assert isinstance(result, dict)
        assert "overall_quality" in result


class TestLSE:
    """Test SingleCellLatentSpaceEvaluator."""

    def setup_method(self):
        np.random.seed(42)
        self.n_cells = 200
        self.latent_dim = 32
        self.latent = np.random.randn(self.n_cells, self.latent_dim).astype(np.float32)

    def test_instantiation(self):
        evaluator = SingleCellLatentSpaceEvaluator(verbose=False)
        assert evaluator is not None

    def test_evaluate_returns_dict(self):
        evaluator = SingleCellLatentSpaceEvaluator(verbose=False)
        result = evaluator.comprehensive_evaluation(self.latent)
        assert isinstance(result, dict)

    def test_scores_are_numeric(self):
        evaluator = SingleCellLatentSpaceEvaluator(verbose=False)
        result = evaluator.comprehensive_evaluation(self.latent)
        numeric_keys = [
            "manifold_dimensionality",
            "spectral_decay_rate",
            "participation_ratio",
            "anisotropy_score",
            "trajectory_directionality",
            "noise_resilience",
            "core_quality",
            "overall_quality",
        ]
        for key in numeric_keys:
            value = result[key]
            assert isinstance(value, (int, float, np.floating)), (
                f"{key} is not numeric: {type(value)}"
            )
            assert np.isfinite(value), f"{key} is not finite: {value}"

    def test_data_type_trajectory(self):
        evaluator = SingleCellLatentSpaceEvaluator(data_type="trajectory", verbose=False)
        result = evaluator.comprehensive_evaluation(self.latent)
        assert result["data_type"] == "trajectory"

    def test_data_type_steady_state(self):
        evaluator = SingleCellLatentSpaceEvaluator(data_type="steady_state", verbose=False)
        result = evaluator.comprehensive_evaluation(self.latent)
        assert result["data_type"] == "steady_state"

    def test_interpretation_present(self):
        evaluator = SingleCellLatentSpaceEvaluator(verbose=False)
        result = evaluator.comprehensive_evaluation(self.latent)
        assert "interpretation" in result
        interp = result["interpretation"]
        assert "quality_level" in interp
        assert "strengths" in interp
        assert "weaknesses" in interp

    def test_compare_methods(self):
        evaluator = SingleCellLatentSpaceEvaluator(verbose=False)
        latent_a = np.random.randn(self.n_cells, self.latent_dim).astype(np.float32)
        latent_b = np.random.randn(self.n_cells, self.latent_dim).astype(np.float32)
        method_dict = {
            "MethodA": latent_a,
            "MethodB": latent_b,
        }
        df = evaluator.compare_methods(method_dict)
        assert len(df) == 2
        assert "Overall_Quality" in df.columns

    def test_convenience_function(self):
        from mocoo.evaluation import evaluate_single_cell_latent_space
        result = evaluate_single_cell_latent_space(
            self.latent, data_type="trajectory", verbose=False
        )
        assert isinstance(result, dict)
        assert "overall_quality" in result


class TestClustering:
    """Test compute_clustering_metrics."""

    def setup_method(self):
        np.random.seed(42)
        self.n_cells = 200
        self.latent_dim = 32
        self.latent = np.random.randn(self.n_cells, self.latent_dim).astype(np.float32)
        self.labels = np.repeat(np.arange(5), 40)

    def test_returns_dict(self):
        result = compute_clustering_metrics(self.latent, self.labels)
        assert isinstance(result, dict)

    def test_expected_keys(self):
        result = compute_clustering_metrics(self.latent, self.labels)
        for key in ("NMI", "ARI", "ASW", "DAV", "CAL", "COR"):
            assert key in result, f"Missing key: {key}"

    def test_values_are_numeric(self):
        result = compute_clustering_metrics(self.latent, self.labels)
        for key, value in result.items():
            assert isinstance(value, (int, float, np.floating, np.integer)), (
                f"{key} is not numeric: {type(value)}"
            )

    def test_ari_nmi_range(self):
        result = compute_clustering_metrics(self.latent, self.labels)
        assert -0.5 <= result["ARI"] <= 1.0
        assert 0.0 <= result["NMI"] <= 1.0


class TestBenchDRE:
    """Test benchmark-optimized DRE metrics."""

    def setup_method(self):
        np.random.seed(42)
        self.latent = np.random.randn(200, 32).astype(np.float32)
        self.proj_2d = np.random.randn(200, 2).astype(np.float32)

    def test_returns_dict(self):
        result = compute_dre_metrics(self.latent, self.proj_2d)
        assert isinstance(result, dict)

    def test_expected_keys(self):
        result = compute_dre_metrics(self.latent, self.proj_2d, prefix="DRE_umap")
        for key in (
            "DRE_umap_distance_correlation",
            "DRE_umap_Q_local",
            "DRE_umap_Q_global",
            "DRE_umap_overall_quality",
        ):
            assert key in result, f"Missing key: {key}"

    def test_custom_prefix(self):
        result = compute_dre_metrics(self.latent, self.proj_2d, prefix="DRE_tsne")
        assert "DRE_tsne_overall_quality" in result


class TestBenchLSE:
    """Test benchmark-optimized LSE metrics."""

    def setup_method(self):
        np.random.seed(42)
        self.latent = np.random.randn(200, 32).astype(np.float32)

    def test_returns_dict(self):
        result = compute_lse_metrics(self.latent)
        assert isinstance(result, dict)

    def test_expected_keys(self):
        result = compute_lse_metrics(self.latent)
        for key in (
            "LSE_manifold_dimensionality",
            "LSE_spectral_decay_rate",
            "LSE_participation_ratio",
            "LSE_anisotropy_score",
            "LSE_noise_resilience",
            "LSE_core_quality",
            "LSE_overall_quality",
        ):
            assert key in result, f"Missing key: {key}"


class TestDREX:
    """Test compute_drex_metrics."""

    def setup_method(self):
        np.random.seed(42)
        self.latent = np.random.randn(200, 32).astype(np.float32)
        self.proj_2d = np.random.randn(200, 2).astype(np.float32)

    def test_returns_dict(self):
        result = compute_drex_metrics(self.latent, self.proj_2d)
        assert isinstance(result, dict)

    def test_expected_keys(self):
        result = compute_drex_metrics(self.latent, self.proj_2d)
        for key in (
            "DREX_trustworthiness",
            "DREX_continuity",
            "DREX_distance_spearman",
            "DREX_distance_pearson",
            "DREX_local_scale_quality",
            "DREX_neighborhood_symmetry",
            "DREX_knn_rank_correlation",
            "DREX_overall_quality",
        ):
            assert key in result, f"Missing key: {key}"

    def test_trustworthiness_range(self):
        result = compute_drex_metrics(self.latent, self.proj_2d)
        assert 0.0 <= result["DREX_trustworthiness"] <= 1.0


class TestLSEX:
    """Test compute_lsex_metrics."""

    def setup_method(self):
        np.random.seed(42)
        self.latent = np.random.randn(200, 32).astype(np.float32)
        self.labels = np.repeat(np.arange(5), 40)

    def test_returns_dict(self):
        result = compute_lsex_metrics(self.latent)
        assert isinstance(result, dict)

    def test_returns_dict_with_labels(self):
        result = compute_lsex_metrics(self.latent, labels=self.labels)
        assert isinstance(result, dict)

    def test_expected_keys(self):
        result = compute_lsex_metrics(self.latent)
        for key in (
            "LSEX_two_hop_connectivity",
            "LSEX_radial_concentration",
            "LSEX_local_curvature",
            "LSEX_cluster_compactness",
            "LSEX_neighbor_purity",
            "LSEX_sampling_stability",
            "LSEX_inter_cluster_gap",
            "LSEX_overall_quality",
        ):
            assert key in result, f"Missing key: {key}"

    def test_no_entropy_stability(self):
        result = compute_lsex_metrics(self.latent)
        assert "LSEX_entropy_stability" not in result


class TestDiagnostics:
    """Test compute_latent_diagnostics."""

    def setup_method(self):
        np.random.seed(42)
        self.latent = np.random.randn(200, 32).astype(np.float32)

    def test_returns_dict(self):
        result = compute_latent_diagnostics(self.latent)
        assert isinstance(result, dict)

    def test_expected_keys(self):
        result = compute_latent_diagnostics(self.latent)
        for key in (
            "diag_mean_norm",
            "diag_std_mean",
            "diag_std_min",
            "diag_std_max",
            "diag_var_mean",
            "diag_near_zero_dims",
            "diag_pairwise_dist_mean",
            "diag_pairwise_dist_std",
        ):
            assert key in result, f"Missing key: {key}"

    def test_near_zero_dims_type(self):
        result = compute_latent_diagnostics(self.latent)
        assert isinstance(result["diag_near_zero_dims"], int)


class TestMetadata:
    """Test metric display metadata."""

    def test_core_metrics_structure(self):
        assert isinstance(CORE_METRICS, list)
        for item in CORE_METRICS:
            assert len(item) == 3
            key, label, higher_is_better = item
            assert isinstance(key, str)
            assert isinstance(label, str)
            assert isinstance(higher_is_better, bool)

    def test_all_metric_groups_structure(self):
        assert isinstance(ALL_METRIC_GROUPS, list)
        for group_name, metrics in ALL_METRIC_GROUPS:
            assert isinstance(group_name, str)
            assert isinstance(metrics, list)
