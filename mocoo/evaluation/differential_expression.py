"""Decoder-based differential expression analysis."""

import numpy as np
from scipy.stats import ranksums
from typing import Dict, List, Optional


def decoder_de(
    decoded_centroids: Dict[int, np.ndarray],
    decoded_all: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    top_n: int = 20,
    gene_names: Optional[List[str]] = None,
) -> Dict:
    """Differential expression via decoded cluster centroids.

    For each cluster, computes log2 fold-change of decoded centroid vs.
    the global mean. Optionally performs Wilcoxon rank-sum testing on
    per-cell decoded values.

    Parameters
    ----------
    decoded_centroids : dict of {label: np.ndarray (G,)}
        Decoded gene proportions per cluster centroid.
    decoded_all : np.ndarray (N, G), optional
        Per-cell decoded values for statistical testing.
    labels : np.ndarray (N,), optional
        Cluster label per cell (needed if decoded_all is provided).
    top_n : int
        Number of top DE genes per cluster.
    gene_names : list of str, optional

    Returns
    -------
    dict of {label: {'top_genes', 'log2fc', 'pvalues'}}
    """
    centroids = np.stack(list(decoded_centroids.values()))
    global_mean = centroids.mean(axis=0)
    G = global_mean.shape[0]

    if gene_names is None:
        gene_names = [f"gene_{i}" for i in range(G)]

    results = {}
    for lab, centroid in decoded_centroids.items():
        log2fc = np.log2(centroid + 1e-8) - np.log2(global_mean + 1e-8)
        pvals = np.ones(G)

        if decoded_all is not None and labels is not None:
            mask = labels == lab
            if mask.sum() > 1 and (~mask).sum() > 1:
                for g in range(G):
                    _, p = ranksums(decoded_all[mask, g], decoded_all[~mask, g])
                    pvals[g] = p

        abs_fc = np.abs(log2fc)
        top_idx = np.argsort(abs_fc)[::-1][:top_n]

        results[lab] = {
            'top_genes': [gene_names[i] for i in top_idx],
            'top_indices': top_idx.tolist(),
            'log2fc': log2fc,
            'pvalues': pvals,
        }

    return results
