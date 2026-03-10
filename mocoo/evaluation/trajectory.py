"""Trajectory evaluation metrics for pseudotime and velocity analysis.

Addresses reviewer concerns about pseudotime circularity (Concern 7)
and biological validation depth (Concern 6).
"""

import numpy as np
from scipy import stats


def pseudotime_concordance(
    predicted_pseudotime: np.ndarray,
    reference_ordering: np.ndarray,
    method: str = "spearman",
) -> dict:
    """Rank correlation between predicted pseudotime and a reference ordering.

    Parameters
    ----------
    predicted_pseudotime : np.ndarray, shape (n_cells,)
        Model-predicted pseudotime values.
    reference_ordering : np.ndarray, shape (n_cells,)
        Reference developmental ordering (e.g., known differentiation stage,
        capture time, or external pseudotime from another method).
    method : str
        Correlation method: 'spearman' or 'kendall'.

    Returns
    -------
    dict
        Keys: pseudotime_concordance, pseudotime_concordance_pvalue.
    """
    predicted_pseudotime = np.asarray(predicted_pseudotime, dtype=float)
    reference_ordering = np.asarray(reference_ordering, dtype=float)

    # Remove NaNs
    valid = ~(np.isnan(predicted_pseudotime) | np.isnan(reference_ordering))
    pt = predicted_pseudotime[valid]
    ref = reference_ordering[valid]

    if len(pt) < 5:
        return {"pseudotime_concordance": np.nan, "pseudotime_concordance_pvalue": np.nan}

    if method == "kendall":
        corr, pval = stats.kendalltau(pt, ref)
    else:
        corr, pval = stats.spearmanr(pt, ref)

    return {
        "pseudotime_concordance": float(corr),
        "pseudotime_concordance_pvalue": float(pval),
    }


def velocity_consistency_score(
    latent: np.ndarray,
    pseudotime: np.ndarray,
    n_neighbors: int = 15,
) -> dict:
    """Quantify how consistent the latent displacement field is with pseudotime.

    For each cell, checks whether its nearest neighbors in later pseudotime
    are displaced in a consistent direction. A high score means the latent
    geometry is coherent with the temporal ordering.

    Parameters
    ----------
    latent : np.ndarray, shape (n_cells, latent_dim)
    pseudotime : np.ndarray, shape (n_cells,)
    n_neighbors : int

    Returns
    -------
    dict
        Keys: velocity_consistency, velocity_directionality.
    """
    from sklearn.neighbors import NearestNeighbors

    latent = np.asarray(latent, dtype=float)
    pseudotime = np.asarray(pseudotime, dtype=float)
    n = len(pseudotime)

    if n < n_neighbors + 1:
        return {"velocity_consistency": np.nan, "velocity_directionality": np.nan}

    nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(latent)
    _, indices = nn.kneighbors(latent)
    indices = indices[:, 1:]

    # For each cell, compute displacement to neighbors with higher pseudotime
    consistency_scores = []
    directionality_scores = []

    for i in range(n):
        nbr_idx = indices[i]
        dt = pseudotime[nbr_idx] - pseudotime[i]
        dz = latent[nbr_idx] - latent[i]

        # Forward neighbors (later in pseudotime)
        fwd = dt > 0
        if fwd.sum() < 2:
            continue

        # Consistency: do forward neighbors share a common displacement direction?
        fwd_dz = dz[fwd]
        mean_dir = fwd_dz.mean(axis=0)
        mean_dir_norm = np.linalg.norm(mean_dir)
        if mean_dir_norm < 1e-10:
            continue

        mean_dir = mean_dir / mean_dir_norm
        cosines = (fwd_dz @ mean_dir) / (np.linalg.norm(fwd_dz, axis=1) + 1e-10)
        consistency_scores.append(float(cosines.mean()))

        # Directionality: fraction of neighbors with correct pseudotime ordering
        directionality_scores.append(float(fwd.mean()))

    return {
        "velocity_consistency": float(np.mean(consistency_scores)) if consistency_scores else np.nan,
        "velocity_directionality": float(np.mean(directionality_scores)) if directionality_scores else np.nan,
    }


def pseudotime_smoothness(
    latent: np.ndarray,
    pseudotime: np.ndarray,
    n_neighbors: int = 15,
) -> dict:
    """Measure how smoothly pseudotime varies across the kNN graph.

    A smooth pseudotime field means neighboring cells in latent space
    have similar pseudotime values. Evaluates trajectory coherence
    independently of the velocity field.

    Parameters
    ----------
    latent : np.ndarray, shape (n_cells, latent_dim)
    pseudotime : np.ndarray, shape (n_cells,)
    n_neighbors : int

    Returns
    -------
    dict
        Keys: pseudotime_smoothness (1 - normalised variance of pseudotime
        differences in kNN).
    """
    from sklearn.neighbors import NearestNeighbors

    latent = np.asarray(latent, dtype=float)
    pseudotime = np.asarray(pseudotime, dtype=float)

    if len(pseudotime) < n_neighbors + 1:
        return {"pseudotime_smoothness": np.nan}

    nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(latent)
    _, indices = nn.kneighbors(latent)
    indices = indices[:, 1:]

    # Variance of pseudotime differences among neighbors
    pt_range = pseudotime.max() - pseudotime.min()
    if pt_range < 1e-10:
        return {"pseudotime_smoothness": np.nan}

    local_vars = []
    for i in range(len(pseudotime)):
        nbr_pt = pseudotime[indices[i]]
        local_var = np.var(nbr_pt - pseudotime[i]) / (pt_range ** 2)
        local_vars.append(local_var)

    # Smoothness = 1 - mean normalised local variance
    smoothness = 1.0 - float(np.mean(local_vars))
    return {"pseudotime_smoothness": max(0.0, smoothness)}
