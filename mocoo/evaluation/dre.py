"""
Dimensionality Reduction Evaluator (DRE).

Provides the ``DimensionalityReductionEvaluator`` class for quantifying how
well a dimensionality-reduction method preserves the structure of the original
high-dimensional data.  Three core metrics are reported:

* **distance_correlation** -- Spearman correlation between pairwise distances
  in the high- and low-dimensional spaces (global structure preservation).
* **Q_global** -- Global quality index derived from the co-ranking matrix.
* **Q_local** -- Local quality index derived from the co-ranking matrix.

The module also exposes two convenience functions --
``evaluate_dimensionality_reduction`` and
``compare_dimensionality_reduction_methods`` -- for quick one-liner usage.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr
import warnings
from typing import Dict, Tuple


class DimensionalityReductionEvaluator:
    """Streamlined dimensionality-reduction quality evaluator.

    Focuses on three core metrics:

    * ``distance_correlation`` -- distance correlation (global structure
      preservation).
    * ``Q_global`` -- global quality index.
    * ``Q_local`` -- local quality index.

    Features:

    * Efficient vectorised computation.
    * Concentrates on the most informative evaluation metrics.
    * Complements the single-cell latent-space evaluation framework.
    """

    def __init__(self, verbose: bool = True):
        """Initialise the evaluator.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print progress messages (default ``True``).
        """
        self.verbose = verbose

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _validate_inputs(self, X_high: np.ndarray, X_low: np.ndarray, k: int) -> None:
        """Validate input parameters."""
        if not isinstance(X_high, np.ndarray) or not isinstance(X_low, np.ndarray):
            raise TypeError("Input data must be numpy arrays")

        if X_high.shape[0] != X_low.shape[0]:
            raise ValueError(
                f"High- and low-dimensional data must have the same number of "
                f"samples: {X_high.shape[0]} vs {X_low.shape[0]}"
            )

        if k >= X_high.shape[0]:
            raise ValueError(
                f"k ({k}) must be less than the number of samples ({X_high.shape[0]})"
            )

        if X_high.ndim != 2 or X_low.ndim != 2:
            raise ValueError("Input data must be 2-dimensional arrays")

    # ==================== 1. Distance Correlation ====================

    def distance_correlation_score(self, X_high: np.ndarray, X_low: np.ndarray) -> float:
        """Compute distance correlation (Spearman).

        Evaluates the monotonic relationship between pairwise distances in the
        high-dimensional and low-dimensional spaces.

        Parameters
        ----------
        X_high : np.ndarray
            High-dimensional data, shape ``(n_samples, n_features_high)``.
        X_low : np.ndarray
            Low-dimensional data, shape ``(n_samples, n_features_low)``.

        Returns
        -------
        float
            Distance correlation score.  A value close to 1 indicates good
            global structure preservation.
        """
        try:
            self._log("Computing distance matrices...")

            D_high = pairwise_distances(X_high)
            D_low = pairwise_distances(X_low)

            distance_corr, _ = spearmanr(D_high.flatten(), D_low.flatten())

            return distance_corr if not np.isnan(distance_corr) else 0.0

        except Exception as e:
            warnings.warn(f"Error computing distance correlation: {e}")
            return 0.0

    # ==================== 2. Ranking Matrix ====================

    def get_ranking_matrix(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Compute the ranking matrix (optimised version).

        Parameters
        ----------
        distance_matrix : np.ndarray
            Pairwise distance matrix, shape ``(n, n)``.

        Returns
        -------
        np.ndarray
            Ranking matrix of shape ``(n, n)`` where entry ``(i, j)`` gives
            the rank of sample *j* among the neighbours of sample *i*.
        """
        try:
            n = len(distance_matrix)

            # Use argsort to obtain ranks directly, avoiding explicit loops
            sorted_indices = np.argsort(distance_matrix, axis=1)

            ranking_matrix = np.zeros((n, n), dtype=np.int32)

            # Vectorised operation: assign ranks for each row
            for i in range(n):
                ranking_matrix[i, sorted_indices[i]] = np.arange(n)

            # Exclude self (set diagonal to 0, shift remaining ranks by -1)
            mask = np.eye(n, dtype=bool)
            ranking_matrix[~mask] = ranking_matrix[~mask] - 1
            ranking_matrix[mask] = 0

            return ranking_matrix

        except Exception as e:
            warnings.warn(f"Error computing ranking matrix: {e}")
            return np.zeros((len(distance_matrix), len(distance_matrix)), dtype=np.int32)

    # ==================== 3. Co-ranking Matrix ====================

    def get_coranking_matrix(self, rank_high: np.ndarray, rank_low: np.ndarray) -> np.ndarray:
        """Compute the co-ranking matrix (optimised version).

        Parameters
        ----------
        rank_high : np.ndarray
            Ranking matrix in the high-dimensional space.
        rank_low : np.ndarray
            Ranking matrix in the low-dimensional space.

        Returns
        -------
        np.ndarray
            Co-ranking matrix of shape ``(n-1, n-1)``.
        """
        try:
            n = len(rank_high)
            corank = np.zeros((n - 1, n - 1), dtype=np.int32)

            # Vectorised operation using advanced indexing
            mask = (rank_high > 0) & (rank_low > 0)
            valid_high = rank_high[mask] - 1  # convert to 0-based indices
            valid_low = rank_low[mask] - 1

            # Ensure indices are within bounds
            valid_mask = (valid_high < n - 1) & (valid_low < n - 1)
            valid_high = valid_high[valid_mask]
            valid_low = valid_low[valid_mask]

            # Accumulate counts
            np.add.at(corank, (valid_high, valid_low), 1)

            return corank

        except Exception as e:
            warnings.warn(f"Error computing co-ranking matrix: {e}")
            n = len(rank_high)
            return np.zeros((n - 1, n - 1), dtype=np.int32)

    # ==================== 4. Q Metric Computation ====================

    def compute_qnx_series(self, corank: np.ndarray) -> np.ndarray:
        """Compute the Q_NX series.

        Parameters
        ----------
        corank : np.ndarray
            Co-ranking matrix.

        Returns
        -------
        np.ndarray
            Array of Q_NX values for each neighbourhood size.
        """
        try:
            n = corank.shape[0] + 1
            qnx_values = []

            Qnx_cum = 0

            for K in range(1, n - 1):
                if K - 1 < corank.shape[0]:
                    intrusions = np.sum(corank[:K, K - 1]) if K - 1 < corank.shape[1] else 0
                    extrusions = np.sum(corank[K - 1, :K]) if K - 1 < corank.shape[0] else 0
                    diagonal = corank[K - 1, K - 1] if K - 1 < min(corank.shape) else 0

                    Qnx_increment = intrusions + extrusions - diagonal
                    Qnx_cum += Qnx_increment

                    # Normalise
                    qnx_normalized = Qnx_cum / (K * n)
                    qnx_values.append(qnx_normalized)

            return np.array(qnx_values)

        except Exception as e:
            warnings.warn(f"Error computing Q_NX series: {e}")
            return np.array([0.0])

    def get_q_local_global(self, qnx_values: np.ndarray) -> Tuple[float, float, int]:
        """Compute local and global quality scalars.

        Parameters
        ----------
        qnx_values : np.ndarray
            Array of Q_NX values.

        Returns
        -------
        tuple of (float, float, int)
            ``(Q_local, Q_global, K_max)`` where *K_max* is the boundary
            between the local and global regimes determined via the Local
            Continuity Meta-Criterion (LCMC).
        """
        try:
            if len(qnx_values) == 0:
                return 0.0, 0.0, 1

            # Compute LCMC (Local Continuity Meta-Criterion)
            lcmc = np.copy(qnx_values)
            N = len(qnx_values)

            for j in range(N):
                lcmc[j] = lcmc[j] - j / N

            K_max = np.argmax(lcmc) + 1

            # Compute Q_local and Q_global
            if K_max > 0:
                Q_local = np.mean(qnx_values[:K_max])
            else:
                Q_local = qnx_values[0] if len(qnx_values) > 0 else 0.0

            if K_max < len(qnx_values):
                Q_global = np.mean(qnx_values[K_max:])
            else:
                Q_global = qnx_values[-1] if len(qnx_values) > 0 else 0.0

            return Q_local, Q_global, K_max

        except Exception as e:
            warnings.warn(f"Error computing Q metrics: {e}")
            return 0.0, 0.0, 1

    # ==================== 5. Comprehensive Evaluation ====================

    def comprehensive_evaluation(self, X_high: np.ndarray, X_low: np.ndarray, k: int = 10) -> Dict:
        """Run a comprehensive dimensionality-reduction quality evaluation.

        Parameters
        ----------
        X_high : np.ndarray
            High-dimensional data, shape ``(n_samples, n_features_high)``.
        X_low : np.ndarray
            Low-dimensional data, shape ``(n_samples, n_features_low)``.
        k : int, optional
            Number of neighbours to consider (default 10).

        Returns
        -------
        dict
            Dictionary containing the core evaluation metrics.
        """
        self._validate_inputs(X_high, X_low, k)

        self._log(f"Starting dimensionality-reduction quality evaluation "
                  f"(n_samples={X_high.shape[0]}, k={k})...")

        results = {}

        # 1. Distance correlation
        self._log("Computing distance correlation...")
        results['distance_correlation'] = self.distance_correlation_score(X_high, X_low)

        # 2. Ranking matrices
        self._log("Computing ranking matrices...")
        D_high = pairwise_distances(X_high)
        D_low = pairwise_distances(X_low)

        rank_high = self.get_ranking_matrix(D_high)
        rank_low = self.get_ranking_matrix(D_low)

        # 3. Co-ranking matrix
        self._log("Computing co-ranking matrix...")
        corank = self.get_coranking_matrix(rank_high, rank_low)

        # 4. Q metrics
        self._log("Computing Q metrics...")
        qnx_values = self.compute_qnx_series(corank)
        Q_local, Q_global, K_max = self.get_q_local_global(qnx_values)

        results['Q_local'] = Q_local
        results['Q_global'] = Q_global
        results['K_max'] = K_max

        # Overall quality
        overall_quality = np.mean([
            results['distance_correlation'],
            results['Q_local'],
            results['Q_global']
        ])
        results['overall_quality'] = overall_quality

        if self.verbose:
            self._print_results(results)

        return results

    def _print_results(self, results: Dict) -> None:
        """Print evaluation results."""

        print("\n" + "=" * 60)
        print("        Dimensionality Reduction Quality Report")
        print("=" * 60)

        print(f"\n[Core Quality Metrics]")
        print(f"  Distance Correlation: {results['distance_correlation']:.4f}")
        print(f"    -> Close to 1 indicates good global structure preservation")

        print(f"\n  Local Quality (Q_local): {results['Q_local']:.4f}")
        print(f"    -> Close to 1 indicates good local structure preservation")

        print(f"\n  Global Quality (Q_global): {results['Q_global']:.4f}")
        print(f"    -> Close to 1 indicates good global structure preservation")

        print(f"\n[Auxiliary Information]")
        print(f"  Local-global boundary (K_max): {results['K_max']}")

        overall_quality = results['overall_quality']

        print(f"\n[Overall Assessment]")
        print(f"  Mean quality score: {overall_quality:.4f}")

        if overall_quality >= 0.8:
            quality_level = "Excellent"
        elif overall_quality >= 0.6:
            quality_level = "Good"
        elif overall_quality >= 0.4:
            quality_level = "Fair"
        else:
            quality_level = "Needs improvement"

        print(f"  Quality level: {quality_level}")

        print("=" * 60)

    def compare_methods(self, method_results_dict: Dict, k: int = 10) -> pd.DataFrame:
        """Compare the effectiveness of different dimensionality-reduction methods.

        Parameters
        ----------
        method_results_dict : dict
            Mapping of ``{method_name: (X_high, X_low)}``.
        k : int, optional
            Number of neighbours to consider (default 10).

        Returns
        -------
        pd.DataFrame
            Comparison table sorted by overall quality (descending).
        """

        comparison_results = []

        for method_name, (X_high, X_low) in method_results_dict.items():
            self._log(f"\nEvaluating method: {method_name}")

            # Temporarily suppress verbose output
            original_verbose = self.verbose
            self.verbose = False

            results = self.comprehensive_evaluation(X_high, X_low, k)

            self.verbose = original_verbose

            overall_quality = np.mean([
                results['distance_correlation'],
                results['Q_local'],
                results['Q_global']
            ])

            comparison_results.append({
                'Method': method_name,
                'Distance_Correlation': results['distance_correlation'],
                'Q_Local': results['Q_local'],
                'Q_Global': results['Q_global'],
                'Overall_Quality': overall_quality,
            })

        df = pd.DataFrame(comparison_results)
        df = df.sort_values('Overall_Quality', ascending=False)

        if self.verbose:
            self._print_comparison_table(df)

        return df

    def _print_comparison_table(self, df: pd.DataFrame) -> None:
        """Print the method-comparison table."""

        print(f"\n{'=' * 90}")
        print(f"              Dimensionality Reduction Method Comparison")
        print('=' * 90)

        pd.set_option('display.float_format', '{:.4f}'.format)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)

        print(df.to_string(index=False))

        print(f"\nBest method: {df.iloc[0]['Method']} "
              f"(overall score: {df.iloc[0]['Overall_Quality']:.4f})")

        print('=' * 90)


# ==================== Convenience Functions ====================

def evaluate_dimensionality_reduction(
    X_high: np.ndarray,
    X_low: np.ndarray,
    k: int = 10,
    verbose: bool = True,
) -> Dict:
    """Convenience function: evaluate dimensionality-reduction quality.

    Parameters
    ----------
    X_high : np.ndarray
        High-dimensional data.
    X_low : np.ndarray
        Low-dimensional data.
    k : int, optional
        Number of neighbours to consider (default 10).
    verbose : bool, optional
        Whether to print detailed output (default ``True``).

    Returns
    -------
    dict
        Evaluation results.
    """
    evaluator = DimensionalityReductionEvaluator(verbose=verbose)
    return evaluator.comprehensive_evaluation(X_high, X_low, k)


def compare_dimensionality_reduction_methods(
    method_results_dict: Dict,
    k: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    """Convenience function: compare different dimensionality-reduction methods.

    Parameters
    ----------
    method_results_dict : dict
        Mapping of ``{method_name: (X_high, X_low)}``.
    k : int, optional
        Number of neighbours to consider (default 10).
    verbose : bool, optional
        Whether to print detailed output (default ``True``).

    Returns
    -------
    pd.DataFrame
        Comparison results sorted by overall quality.
    """
    evaluator = DimensionalityReductionEvaluator(verbose=verbose)
    return evaluator.compare_methods(method_results_dict, k)
