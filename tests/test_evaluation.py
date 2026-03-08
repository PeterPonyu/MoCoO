"""Tests for mocoo.evaluation subpackage."""
import numpy as np
import pytest
from mocoo.evaluation import (
    DimensionalityReductionEvaluator,
    SingleCellLatentSpaceEvaluator,
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
