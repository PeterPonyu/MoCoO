"""Benchmark DRE and LSE metrics using the gold-standard evaluators.

Delegates to :class:`DimensionalityReductionEvaluator` (co-ranking matrix)
and :class:`SingleCellLatentSpaceEvaluator` (PCA / SVD) for accurate
computation, then re-keys outputs with the canonical ``DRE_*`` / ``LSE_*``
prefix scheme expected by the pipeline.
"""

import numpy as np

from .dre import DimensionalityReductionEvaluator
from .lse import SingleCellLatentSpaceEvaluator


def compute_dre_metrics(
    latent: np.ndarray,
    projection_2d: np.ndarray,
    k: int = 15,
    prefix: str = "DRE_umap",
) -> dict:
    """Compute DRE metrics via the co-ranking-matrix evaluator.

    Parameters
    ----------
    latent : np.ndarray, shape (n_cells, latent_dim)
    projection_2d : np.ndarray, shape (n_cells, 2)
    k : int
        Number of neighbors for local quality.
    prefix : str
        Key prefix (e.g. "DRE_umap", "DRE_tsne").

    Returns
    -------
    dict
        Keys: {prefix}_distance_correlation, {prefix}_Q_local,
        {prefix}_Q_global, {prefix}_overall_quality.
    """
    latent = np.asarray(latent, dtype=float)
    projection_2d = np.asarray(projection_2d, dtype=float)
    m = {}
    try:
        evaluator = DimensionalityReductionEvaluator(verbose=False)
        results = evaluator.comprehensive_evaluation(latent, projection_2d, k=k)
        m[f"{prefix}_distance_correlation"] = results["distance_correlation"]
        m[f"{prefix}_Q_local"] = results["Q_local"]
        m[f"{prefix}_Q_global"] = results["Q_global"]
        m[f"{prefix}_overall_quality"] = results["overall_quality"]
    except Exception:
        for key in ("distance_correlation", "Q_local", "Q_global", "overall_quality"):
            m[f"{prefix}_{key}"] = np.nan
    return m


def compute_lse_metrics(latent: np.ndarray) -> dict:
    """Compute LSE metrics via the gold-standard evaluator.

    Parameters
    ----------
    latent : np.ndarray, shape (n_cells, latent_dim)

    Returns
    -------
    dict
        Keys: LSE_manifold_dimensionality, LSE_spectral_decay_rate,
        LSE_participation_ratio, LSE_anisotropy_score, LSE_noise_resilience,
        LSE_core_quality, LSE_overall_quality.
    """
    latent = np.asarray(latent, dtype=float)
    m = {}
    try:
        evaluator = SingleCellLatentSpaceEvaluator(
            data_type="trajectory", verbose=False,
        )
        results = evaluator.comprehensive_evaluation(latent)
        m["LSE_manifold_dimensionality"] = results["manifold_dimensionality"]
        m["LSE_spectral_decay_rate"] = results["spectral_decay_rate"]
        m["LSE_participation_ratio"] = results["participation_ratio"]
        m["LSE_anisotropy_score"] = results["anisotropy_score"]
        m["LSE_noise_resilience"] = results["noise_resilience"]
        m["LSE_core_quality"] = results["core_quality"]
        m["LSE_overall_quality"] = results["overall_quality"]
    except Exception:
        for key in (
            "manifold_dimensionality",
            "spectral_decay_rate",
            "participation_ratio",
            "anisotropy_score",
            "noise_resilience",
            "core_quality",
            "overall_quality",
        ):
            m[f"LSE_{key}"] = np.nan
    return m
