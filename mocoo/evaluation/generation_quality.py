"""Quality metrics for FM-generated cells."""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from typing import Dict


def generation_quality_metrics(
    real_latent: np.ndarray,
    generated_latent: np.ndarray,
    k: int = 10,
) -> Dict[str, float]:
    """Assess quality of generated cell latents relative to real cells.

    Parameters
    ----------
    real_latent : np.ndarray, shape (N, D)
    generated_latent : np.ndarray, shape (M, D)
    k : int
        Number of neighbors for NND computation.

    Returns
    -------
    dict
        - ``nnd_mean``: mean nearest-neighbor distance generated→real.
        - ``nnd_std``: std of NND.
        - ``coverage``: fraction of real cells with a generated neighbor
          within median real-to-real NND.
        - ``authenticity``: fraction of generated cells closer to real
          than to other generated cells.
        - ``diversity``: mean pairwise distance among generated cells.
    """
    k_eff = min(k, len(real_latent) - 1)
    nn_real = NearestNeighbors(n_neighbors=k_eff + 1).fit(real_latent)

    # Generated → real NND
    dists_gen_real, _ = nn_real.kneighbors(generated_latent)
    nnd = dists_gen_real[:, 0]

    # Real-to-real baseline (skip self at index 0)
    dists_real_real, _ = nn_real.kneighbors(real_latent)
    real_nnd_median = float(np.median(dists_real_real[:, 1]))

    # Coverage: real cells with a close generated neighbor
    nn_gen = NearestNeighbors(n_neighbors=1).fit(generated_latent)
    dists_real_gen, _ = nn_gen.kneighbors(real_latent)
    coverage = float((dists_real_gen[:, 0] < real_nnd_median).mean())

    # Authenticity: generated closer to real than to other generated
    if len(generated_latent) > 1:
        nn_gen2 = NearestNeighbors(n_neighbors=2).fit(generated_latent)
        dists_gen_gen, _ = nn_gen2.kneighbors(generated_latent)
        gen_self_nnd = dists_gen_gen[:, 1]  # skip self
        authenticity = float((nnd < gen_self_nnd).mean())
    else:
        authenticity = 1.0

    # Diversity (subsample for speed)
    n_sub = min(500, len(generated_latent))
    idx = np.random.choice(len(generated_latent), n_sub, replace=False)
    pdist = pairwise_distances(generated_latent[idx])
    diversity = float(pdist[np.triu_indices(n_sub, k=1)].mean())

    return {
        'nnd_mean': float(nnd.mean()),
        'nnd_std': float(nnd.std()),
        'coverage': coverage,
        'authenticity': authenticity,
        'diversity': diversity,
    }
