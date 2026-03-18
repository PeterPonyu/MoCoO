"""Trajectory branching detection via velocity divergence."""

import numpy as np
from sklearn.cluster import DBSCAN
from typing import Dict, Optional


def detect_branch_points(
    divergence: np.ndarray,
    latent: np.ndarray,
    threshold_quantile: float = 0.9,
    eps: float = 1.0,
    min_samples: int = 5,
) -> Dict:
    """Identify trajectory branch points as high-divergence regions.

    High |div v| indicates cells at bifurcation points where the velocity
    field is expanding (positive) or converging (negative).

    Parameters
    ----------
    divergence : np.ndarray, shape (N,)
        Per-cell velocity divergence: trace(df/dz).
    latent : np.ndarray, shape (N, D)
        Latent coordinates.
    threshold_quantile : float
        Quantile of |divergence| above which cells are branch candidates.
    eps : float
        DBSCAN epsilon for clustering branch candidates in latent space.
    min_samples : int
        DBSCAN min_samples.

    Returns
    -------
    dict
        - ``divergence``: (N,) raw divergence values.
        - ``is_branch_point``: (N,) boolean mask.
        - ``branch_clusters``: (N,) int labels (-1 = not a branch point).
        - ``n_branches``: number of distinct branch regions found.
    """
    threshold = np.quantile(np.abs(divergence), threshold_quantile)
    is_branch = np.abs(divergence) > threshold

    branch_clusters = np.full(len(divergence), -1, dtype=int)
    if is_branch.sum() >= min_samples:
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(latent[is_branch])
        branch_clusters[is_branch] = db.labels_

    unique_labels = set(branch_clusters[is_branch])
    n_branches = len(unique_labels - {-1})

    return {
        'divergence': divergence,
        'is_branch_point': is_branch,
        'branch_clusters': branch_clusters,
        'n_branches': n_branches,
    }
