"""Clustering quality metrics: NMI, ARI, ASW, DAV, CAL, COR."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)


def compute_clustering_metrics(
    latent: np.ndarray,
    labels: np.ndarray,
    random_state: int = 42,
) -> dict:
    """Compute clustering quality metrics.

    Runs KMeans with n_clusters matching the number of true labels,
    then evaluates the predicted clusters against ground truth.

    Parameters
    ----------
    latent : np.ndarray, shape (n_cells, latent_dim)
    labels : array-like, shape (n_cells,)
        Ground-truth integer labels.
    random_state : int
        Random seed for KMeans.

    Returns
    -------
    dict
        Keys: NMI, ARI, ASW, DAV, CAL, COR. Values are float or NaN.
    """
    latent = np.asarray(latent, dtype=float)
    labels = np.asarray(labels)
    n_clusters = len(np.unique(labels))
    pred = KMeans(
        n_clusters=n_clusters, n_init=10, random_state=random_state
    ).fit_predict(latent)

    m = {
        "NMI": normalized_mutual_info_score(labels, pred),
        "ARI": adjusted_rand_score(labels, pred),
    }

    try:
        m["ASW"] = (
            silhouette_score(latent, pred) if len(np.unique(pred)) > 1 else np.nan
        )
    except Exception:
        m["ASW"] = np.nan
    try:
        m["DAV"] = davies_bouldin_score(latent, pred)
    except Exception:
        m["DAV"] = np.nan
    try:
        m["CAL"] = calinski_harabasz_score(latent, pred)
    except Exception:
        m["CAL"] = np.nan
    try:
        acorr = np.abs(np.corrcoef(latent.T))
        m["COR"] = float(acorr.sum(axis=1).mean() - 1)
    except Exception:
        m["COR"] = np.nan

    return m
