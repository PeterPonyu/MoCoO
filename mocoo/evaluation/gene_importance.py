"""Gene importance scoring from decoder Jacobians."""

import numpy as np
from typing import Optional, Dict, List


def rank_genes_by_jacobian(
    jacobian: np.ndarray,
    gene_names: Optional[List[str]] = None,
    top_n: int = 20,
) -> Dict:
    """Rank genes by sensitivity to latent dimensions via Jacobian norms.

    Parameters
    ----------
    jacobian : np.ndarray, shape (N, G, D)
        Decoder Jacobian per cell: J[i, g, d] = dmu_g/dz_d at cell i.
    gene_names : list of str, optional
        Gene names; uses integer indices if not provided.
    top_n : int
        Number of top genes to report per dimension.

    Returns
    -------
    dict
        - ``importance``: (G,) overall gene importance (L2 norm across dims).
        - ``per_dim``: (G, D) mean |J| per gene per latent dimension.
        - ``top_genes_per_dim``: {dim_idx: [(gene_name, score), ...]}.
        - ``ranked_genes``: gene names sorted by descending overall importance.
    """
    N, G, D = jacobian.shape
    if gene_names is None:
        gene_names = [str(i) for i in range(G)]

    mean_abs = np.abs(jacobian).mean(axis=0)           # (G, D)
    mean_abs = np.nan_to_num(mean_abs, nan=0.0, posinf=0.0, neginf=0.0)
    importance = np.linalg.norm(mean_abs, axis=1)       # (G,)

    top_per_dim = {}
    for d in range(D):
        scores = mean_abs[:, d]
        order = np.argsort(scores)[::-1][:top_n]
        top_per_dim[d] = [(gene_names[i], float(scores[i])) for i in order]

    ranked = [gene_names[i] for i in np.argsort(importance)[::-1]]

    return {
        'importance': importance,
        'per_dim': mean_abs,
        'top_genes_per_dim': top_per_dim,
        'ranked_genes': ranked,
    }
