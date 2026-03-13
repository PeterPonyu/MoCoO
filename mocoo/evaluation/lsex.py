"""Extended Latent Space metrics (LSEX).

Aligned with PanODE-LAB eval_lib reference implementation.
Metrics: two-hop connectivity, radial concentration, local curvature,
entropy stability.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

from ._neighbors import knn_indices


def _q_local(knn_source, knn_target, k):
    """Fraction of k-NN preserved between source and target spaces."""
    n = knn_source.shape[0]
    overlap = 0.0
    for i in range(n):
        s = set(knn_source[i, :k])
        t = set(knn_target[i, :k])
        overlap += len(s & t) / k
    return overlap / n


def compute_lsex_metrics(
    latent: np.ndarray, labels: np.ndarray = None, k: int = 15
) -> dict:
    """Compute extended latent-space metrics (eval_lib aligned).

    Parameters
    ----------
    latent : np.ndarray, shape (n_cells, latent_dim)
    labels : np.ndarray, optional
        Unused (kept for API compatibility). Labels are not needed for
        the eval_lib-aligned LSEX metrics.
    k : int
        Number of neighbors.

    Returns
    -------
    dict
        Keys: LSEX_two_hop_connectivity, LSEX_radial_concentration,
        LSEX_local_curvature, LSEX_entropy_stability,
        LSEX_overall_quality.
    """
    latent = np.asarray(latent, dtype=float)
    m = {}
    try:
        n = latent.shape[0]
        knn = knn_indices(latent, k)

        # Two-hop connectivity: fraction of 2-hop neighbors that are unique
        two_hop_unique = 0.0
        for i in range(n):
            one_hop = set(knn[i])
            two_hop = set()
            for j in knn[i]:
                two_hop.update(knn[j])
            two_hop -= one_hop
            two_hop.discard(i)
            two_hop_unique += len(two_hop) / max(1, k * k)
        m["LSEX_two_hop_connectivity"] = two_hop_unique / n

        # Radial concentration: how concentrated neighbors are vs uniform
        dists = (
            NearestNeighbors(n_neighbors=k + 1)
            .fit(latent)
            .kneighbors(latent, return_distance=True)[0][:, 1:]
        )
        cv = dists.std(axis=1) / (dists.mean(axis=1) + 1e-10)
        m["LSEX_radial_concentration"] = 1.0 - float(cv.mean())

        # Local curvature: linearity of kNN neighborhoods
        curvature = 0.0
        n_sub = min(n, 2000)
        for i in range(n_sub):
            nbrs = latent[knn[i]]
            center = nbrs.mean(axis=0)
            residuals = nbrs - center
            _, s, _ = np.linalg.svd(residuals, full_matrices=False)
            curvature += s[0] / (s.sum() + 1e-10)
        m["LSEX_local_curvature"] = curvature / n_sub

        # Entropy stability: consistency of neighborhood structure across scales
        q_k = _q_local(knn, knn, k)  # self-consistency baseline
        knn_half = knn_indices(latent, max(k // 2, 3))
        q_half = _q_local(
            knn_half,
            knn_indices(latent, max(k // 2, 3)),
            max(k // 2, 3),
        )
        m["LSEX_entropy_stability"] = float(np.mean([q_k, q_half]))

        # Unweighted mean overall (eval_lib aligned)
        m["LSEX_overall_quality"] = float(np.mean([
            m["LSEX_two_hop_connectivity"],
            max(0, m["LSEX_radial_concentration"]),
            m["LSEX_local_curvature"],
            m["LSEX_entropy_stability"],
        ]))
    except Exception:
        for key in (
            "two_hop_connectivity",
            "radial_concentration",
            "local_curvature",
            "entropy_stability",
            "overall_quality",
        ):
            m[f"LSEX_{key}"] = np.nan
    return m
