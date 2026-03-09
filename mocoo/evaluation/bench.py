"""Benchmark-optimized DRE and LSE metrics.

These are kNN-based approximations suitable for rapid benchmarking.
For publication-quality co-ranking matrix analysis, use
:class:`mocoo.evaluation.DimensionalityReductionEvaluator` instead.
"""

import numpy as np
from scipy.spatial.distance import pdist

from ._neighbors import knn_indices


def _q_local(knn_source: np.ndarray, knn_target: np.ndarray, k: int) -> float:
    """Fraction of k-NN preserved between source and target spaces."""
    n = knn_source.shape[0]
    overlap = 0.0
    for i in range(n):
        s = set(knn_source[i, :k])
        t = set(knn_target[i, :k])
        overlap += len(s & t) / k
    return overlap / n


def _distance_correlation(
    X_source: np.ndarray, X_target: np.ndarray, max_samples: int = 2000
) -> float:
    """Pearson correlation between pairwise distances in two spaces."""
    X_source = np.asarray(X_source, dtype=float)
    X_target = np.asarray(X_target, dtype=float)
    n = X_source.shape[0]
    if n > max_samples:
        idx = np.random.RandomState(42).choice(n, max_samples, replace=False)
        X_source = X_source[idx]
        X_target = X_target[idx]
    d_s = pdist(X_source)
    d_t = pdist(X_target)
    if d_s.std() < 1e-10 or d_t.std() < 1e-10:
        return 0.0
    return float(np.corrcoef(d_s, d_t)[0, 1])


def compute_dre_metrics(
    latent: np.ndarray,
    projection_2d: np.ndarray,
    k: int = 15,
    prefix: str = "DRE_umap",
) -> dict:
    """Compute benchmark DRE metrics: distance_correlation, Q_local, Q_global, overall.

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
        knn_src = knn_indices(latent, max(k, 50))
        knn_tgt = knn_indices(projection_2d, max(k, 50))

        m[f"{prefix}_distance_correlation"] = _distance_correlation(
            latent, projection_2d
        )
        m[f"{prefix}_Q_local"] = _q_local(knn_src, knn_tgt, k)
        m[f"{prefix}_Q_global"] = _q_local(
            knn_src, knn_tgt, min(50, knn_src.shape[1])
        )
        m[f"{prefix}_overall_quality"] = np.mean(
            [
                m[f"{prefix}_distance_correlation"],
                m[f"{prefix}_Q_local"],
                m[f"{prefix}_Q_global"],
            ]
        )
    except Exception:
        for key in ("distance_correlation", "Q_local", "Q_global", "overall_quality"):
            m[f"{prefix}_{key}"] = np.nan
    return m


def compute_lse_metrics(latent: np.ndarray) -> dict:
    """Compute benchmark LSE metrics from singular value decomposition.

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
        z = latent - latent.mean(axis=0)
        _, s, _ = np.linalg.svd(z, full_matrices=False)
        s = np.maximum(s, 0)

        # Manifold dimensionality: participation ratio
        p = s**2 / (s**2).sum()
        participation_ratio = 1.0 / np.sum(p**2) if np.sum(p**2) > 0 else 0
        m["LSE_manifold_dimensionality"] = participation_ratio / latent.shape[1]

        # Spectral decay rate
        log_s = np.log(s[s > 1e-10] + 1e-10)
        if len(log_s) > 1:
            x = np.arange(len(log_s))
            slope = np.polyfit(x, log_s, 1)[0]
            m["LSE_spectral_decay_rate"] = max(0, -slope)
        else:
            m["LSE_spectral_decay_rate"] = 0.0

        m["LSE_participation_ratio"] = participation_ratio

        # Anisotropy: ratio of largest to average SV
        m["LSE_anisotropy_score"] = float(s[0] / (s.mean() + 1e-10))

        # Noise resilience: fraction of variance in top-80% SVs
        cumvar = np.cumsum(s**2) / (s**2).sum()
        n_sig = np.searchsorted(cumvar, 0.8) + 1
        m["LSE_noise_resilience"] = n_sig / len(s)

        # Core quality: geometric mean of normalized PR and noise resilience
        norm_pr = min(1.0, participation_ratio / latent.shape[1])
        m["LSE_core_quality"] = np.sqrt(norm_pr * m["LSE_noise_resilience"])

        m["LSE_overall_quality"] = np.mean(
            [
                m["LSE_manifold_dimensionality"],
                min(1.0, m["LSE_spectral_decay_rate"]),
                m["LSE_noise_resilience"],
                m["LSE_core_quality"],
            ]
        )
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
