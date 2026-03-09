"""Extended Latent Space metrics (LSEX).

Metrics: two-hop connectivity, radial concentration, local curvature,
cluster compactness, neighbor purity, sampling stability, inter-cluster gap.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from ._neighbors import knn_indices


def compute_lsex_metrics(
    latent: np.ndarray, labels: np.ndarray = None, k: int = 15
) -> dict:
    """Compute extended latent-space metrics.

    Parameters
    ----------
    latent : np.ndarray, shape (n_cells, latent_dim)
    labels : np.ndarray, optional
        Integer ground-truth labels. If None, KMeans labels are inferred.
    k : int
        Number of neighbors.

    Returns
    -------
    dict
        Keys: LSEX_two_hop_connectivity, LSEX_radial_concentration,
        LSEX_local_curvature, LSEX_cluster_compactness,
        LSEX_neighbor_purity, LSEX_sampling_stability,
        LSEX_inter_cluster_gap, LSEX_overall_quality.
    """
    latent = np.asarray(latent, dtype=float)
    m = {}
    try:
        n = latent.shape[0]
        knn = knn_indices(latent, k)

        # --- Infer labels if not provided ---
        if labels is None:
            n_clusters = min(10, max(2, int(np.sqrt(n / 2))))
            labels_use = KMeans(
                n_clusters=n_clusters, random_state=42, n_init=10
            ).fit_predict(latent)
        else:
            labels_use = np.asarray(labels, dtype=int)

        unique_labels = np.unique(labels_use)
        n_clusters = len(unique_labels)

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

        # Cluster compactness: ratio of intra-cluster to inter-cluster distance
        centroids = np.array(
            [latent[labels_use == c].mean(axis=0) for c in unique_labels]
        )
        intra_dists = []
        for c in unique_labels:
            members = latent[labels_use == c]
            if len(members) > 1:
                d = np.linalg.norm(members - centroids[c == unique_labels][0], axis=1)
                intra_dists.append(d.mean())
        avg_intra = np.mean(intra_dists) if intra_dists else 1e-10

        if n_clusters > 1:
            from scipy.spatial.distance import pdist

            centroid_dists = pdist(centroids)
            avg_inter = centroid_dists.mean() if len(centroid_dists) > 0 else 1e-10
        else:
            avg_inter = 1e-10

        ratio = avg_intra / (avg_inter + 1e-10)
        m["LSEX_cluster_compactness"] = float(max(0.0, 1.0 - ratio))

        # Neighbor purity: fraction of k-NN sharing the same label
        purity = 0.0
        for i in range(n):
            same = sum(1 for j in knn[i] if labels_use[j] == labels_use[i])
            purity += same / k
        m["LSEX_neighbor_purity"] = purity / n

        # Sampling stability: kNN overlap between two 80% subsamples
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(123)
        n_sub = int(0.8 * n)
        idx1 = rng1.choice(n, n_sub, replace=False)
        idx2 = rng2.choice(n, n_sub, replace=False)
        common = np.intersect1d(idx1, idx2)

        if len(common) > 10:
            # Build position maps for common indices within each subsample
            pos1 = {v: i for i, v in enumerate(idx1)}
            pos2 = {v: i for i, v in enumerate(idx2)}

            knn1 = knn_indices(latent[idx1], k)
            knn2 = knn_indices(latent[idx2], k)

            overlap_sum = 0.0
            for orig_idx in common:
                # Neighbors of this point in subsample 1 (original indices)
                nbrs1 = set(idx1[j] for j in knn1[pos1[orig_idx]])
                # Neighbors of this point in subsample 2 (original indices)
                nbrs2 = set(idx2[j] for j in knn2[pos2[orig_idx]])
                overlap_sum += len(nbrs1 & nbrs2) / k
            m["LSEX_sampling_stability"] = overlap_sum / len(common)
        else:
            m["LSEX_sampling_stability"] = 0.5  # fallback

        # Inter-cluster gap: min centroid gap / average intra-cluster spread
        if n_clusters > 1:
            min_gap = centroid_dists.min()
            gap_ratio = min_gap / (avg_intra + 1e-10)
            m["LSEX_inter_cluster_gap"] = float(
                min(1.0, gap_ratio / (gap_ratio + 1.0))
            )
        else:
            m["LSEX_inter_cluster_gap"] = 0.0

        m["LSEX_overall_quality"] = float(
            np.mean(
                [
                    m["LSEX_two_hop_connectivity"],
                    max(0, m["LSEX_radial_concentration"]),
                    m["LSEX_local_curvature"],
                    m["LSEX_cluster_compactness"],
                    m["LSEX_neighbor_purity"],
                    m["LSEX_sampling_stability"],
                    m["LSEX_inter_cluster_gap"],
                ]
            )
        )
    except Exception:
        for key in (
            "two_hop_connectivity",
            "radial_concentration",
            "local_curvature",
            "cluster_compactness",
            "neighbor_purity",
            "sampling_stability",
            "inter_cluster_gap",
            "overall_quality",
        ):
            m[f"LSEX_{key}"] = np.nan
    return m
