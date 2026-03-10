"""Clustering quality metrics: NMI, ARI, ASW, DAV, CAL, COR.

Also provides Leiden-based and kNN-based (reclustering-free) evaluation.
"""

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


def compute_leiden_metrics(
    latent: np.ndarray,
    labels: np.ndarray,
    resolutions: list[float] | None = None,
) -> dict:
    """Compute ARI and NMI using Leiden clustering instead of KMeans.

    Addresses reviewer concern about KMeans bias for trajectory-shaped manifolds.
    Requires ``scanpy`` and ``leidenalg``.

    Parameters
    ----------
    latent : np.ndarray, shape (n_cells, latent_dim)
    labels : array-like, shape (n_cells,)
        Ground-truth labels.
    resolutions : list of float, optional
        Leiden resolution parameters to sweep. Default: [0.5, 1.0, 2.0].

    Returns
    -------
    dict
        Keys: Leiden_ARI_{res}, Leiden_NMI_{res} for each resolution,
        plus Leiden_ARI_best and Leiden_NMI_best (best across resolutions).
    """
    if resolutions is None:
        resolutions = [0.5, 1.0, 2.0]

    latent = np.asarray(latent, dtype=float)
    labels = np.asarray(labels)
    m = {}

    try:
        import scanpy as sc
        import anndata

        adata = anndata.AnnData(X=latent)
        sc.pp.neighbors(adata, use_rep="X", n_neighbors=15)

        best_ari, best_nmi = -1.0, -1.0
        for res in resolutions:
            sc.tl.leiden(adata, resolution=res, key_added=f"leiden_{res}")
            pred = adata.obs[f"leiden_{res}"].astype(int).values

            ari = adjusted_rand_score(labels, pred)
            nmi = normalized_mutual_info_score(labels, pred)
            m[f"Leiden_ARI_{res}"] = ari
            m[f"Leiden_NMI_{res}"] = nmi

            if ari > best_ari:
                best_ari = ari
            if nmi > best_nmi:
                best_nmi = nmi

        m["Leiden_ARI_best"] = best_ari
        m["Leiden_NMI_best"] = best_nmi
    except ImportError:
        for res in resolutions:
            m[f"Leiden_ARI_{res}"] = np.nan
            m[f"Leiden_NMI_{res}"] = np.nan
        m["Leiden_ARI_best"] = np.nan
        m["Leiden_NMI_best"] = np.nan
    except Exception:
        for res in resolutions:
            m[f"Leiden_ARI_{res}"] = np.nan
            m[f"Leiden_NMI_{res}"] = np.nan
        m["Leiden_ARI_best"] = np.nan
        m["Leiden_NMI_best"] = np.nan

    return m


def compute_neighborhood_metrics(
    latent: np.ndarray,
    labels: np.ndarray,
    k: int = 15,
) -> dict:
    """Compute reclustering-free neighborhood metrics.

    Evaluates latent quality using kNN purity and label-aware silhouette
    without any KMeans reclustering step. Addresses reviewer concern
    about evaluation bias from KMeans.

    Parameters
    ----------
    latent : np.ndarray, shape (n_cells, latent_dim)
    labels : array-like, shape (n_cells,)
        Ground-truth labels.
    k : int
        Number of nearest neighbors.

    Returns
    -------
    dict
        Keys: kNN_purity, label_ASW.
    """
    from sklearn.neighbors import NearestNeighbors

    latent = np.asarray(latent, dtype=float)
    labels = np.asarray(labels)
    m = {}

    try:
        nn = NearestNeighbors(n_neighbors=k + 1).fit(latent)
        _, indices = nn.kneighbors(latent)
        indices = indices[:, 1:]  # exclude self

        # kNN purity: fraction of neighbors sharing the same ground-truth label
        same_label = np.array([
            np.mean(labels[indices[i]] == labels[i]) for i in range(len(labels))
        ])
        m["kNN_purity"] = float(same_label.mean())
    except Exception:
        m["kNN_purity"] = np.nan

    try:
        # Label-aware silhouette: silhouette using ground-truth labels, not KMeans
        if len(np.unique(labels)) > 1:
            m["label_ASW"] = float(silhouette_score(latent, labels))
        else:
            m["label_ASW"] = np.nan
    except Exception:
        m["label_ASW"] = np.nan

    return m
