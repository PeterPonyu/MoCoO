"""Extended Dimensionality Reduction metrics (DREX).

Aligned with PanODE-LAB eval_lib reference implementation.
Metrics: trustworthiness, continuity, distance correlations (Spearman, Pearson),
local scale quality, neighborhood symmetry.
"""

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, pearsonr
from sklearn.neighbors import NearestNeighbors

from ._neighbors import knn_indices


def _trustworthiness(X_high: np.ndarray, X_low: np.ndarray, k: int = 15) -> float:
    """How well the low-D embedding preserves neighborhood structure."""
    X_high = np.asarray(X_high, dtype=float)
    X_low = np.asarray(X_low, dtype=float)
    from sklearn.manifold import trustworthiness as _tw

    return _tw(X_high, X_low, n_neighbors=k)


def _continuity(X_high: np.ndarray, X_low: np.ndarray, k: int = 15) -> float:
    """How well the high-D neighborhoods are continued in low-D."""
    X_high = np.asarray(X_high, dtype=float)
    X_low = np.asarray(X_low, dtype=float)
    n = X_high.shape[0]
    nn_high = knn_indices(X_high, k)
    nn_low = knn_indices(X_low, k)

    cont = 0.0
    for i in range(n):
        low_set = set(nn_low[i])
        for j in nn_high[i]:
            if j not in low_set:
                cont += 1
    max_cont = n * k * (2 * n - 3 * k - 1)
    if max_cont == 0:
        return 1.0
    return 1.0 - (2.0 / max_cont) * cont


def compute_drex_metrics(
    latent: np.ndarray,
    projection_2d: np.ndarray,
    k: int = 15,
) -> dict:
    """Compute DREX metrics (eval_lib aligned).

    Parameters
    ----------
    latent : np.ndarray, shape (n_cells, latent_dim)
    projection_2d : np.ndarray, shape (n_cells, 2)
    k : int
        Number of neighbors.

    Returns
    -------
    dict
        Keys: DREX_trustworthiness, DREX_continuity, DREX_distance_spearman,
        DREX_distance_pearson, DREX_local_scale_quality,
        DREX_neighborhood_symmetry, DREX_overall_quality.
    """
    latent = np.asarray(latent, dtype=float)
    projection_2d = np.asarray(projection_2d, dtype=float)
    m = {}
    try:
        m["DREX_trustworthiness"] = _trustworthiness(latent, projection_2d, k)
        m["DREX_continuity"] = _continuity(latent, projection_2d, k)

        # Distance correlations (Spearman and Pearson)
        n = min(latent.shape[0], 2000)
        if latent.shape[0] > n:
            idx = np.random.RandomState(42).choice(
                latent.shape[0], n, replace=False
            )
            d_h = pdist(latent[idx])
            d_l = pdist(projection_2d[idx])
        else:
            d_h = pdist(latent)
            d_l = pdist(projection_2d)

        m["DREX_distance_spearman"] = float(spearmanr(d_h, d_l).correlation)
        m["DREX_distance_pearson"] = float(pearsonr(d_h, d_l)[0])

        # Local scale quality: correlation of k-NN distances
        nn_h = NearestNeighbors(n_neighbors=k + 1).fit(latent)
        dists_h = nn_h.kneighbors(latent, return_distance=True)[0][:, 1:]
        nn_l = NearestNeighbors(n_neighbors=k + 1).fit(projection_2d)
        dists_l = nn_l.kneighbors(projection_2d, return_distance=True)[0][:, 1:]
        local_h = dists_h.mean(axis=1)
        local_l = dists_l.mean(axis=1)
        m["DREX_local_scale_quality"] = float(spearmanr(local_h, local_l).correlation)

        # Neighborhood symmetry: overlap of kNN in both directions
        knn_h = knn_indices(latent, k)
        knn_l = knn_indices(projection_2d, k)
        sym = 0.0
        for i in range(latent.shape[0]):
            s_h = set(knn_h[i])
            s_l = set(knn_l[i])
            sym += len(s_h & s_l) / k
        m["DREX_neighborhood_symmetry"] = sym / latent.shape[0]

        # Unweighted mean overall (eval_lib aligned)
        m["DREX_overall_quality"] = float(np.mean([
            m["DREX_trustworthiness"],
            m["DREX_continuity"],
            max(0, m["DREX_distance_spearman"]),
            max(0, m["DREX_distance_pearson"]),
            max(0, m["DREX_local_scale_quality"]),
            m["DREX_neighborhood_symmetry"],
        ]))
    except Exception:
        for key in (
            "trustworthiness",
            "continuity",
            "distance_spearman",
            "distance_pearson",
            "local_scale_quality",
            "neighborhood_symmetry",
            "overall_quality",
        ):
            m[f"DREX_{key}"] = np.nan
    return m
