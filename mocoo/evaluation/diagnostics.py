"""Latent collapse and redundancy diagnostics."""

import numpy as np
from scipy.spatial.distance import pdist


def compute_latent_diagnostics(
    latent: np.ndarray, max_samples: int = 2000
) -> dict:
    """Compute latent-space health diagnostics.

    Parameters
    ----------
    latent : np.ndarray, shape (n_cells, latent_dim)
    max_samples : int
        Max samples for pairwise distance computation.

    Returns
    -------
    dict
        Keys: diag_mean_norm, diag_std_mean, diag_std_min, diag_std_max,
        diag_var_mean, diag_near_zero_dims, diag_pairwise_dist_mean,
        diag_pairwise_dist_std.
    """
    z = np.asarray(latent, dtype=float)
    std = z.std(axis=0)
    var = z.var(axis=0)

    n = z.shape[0]
    if n > max_samples:
        idx = np.random.choice(n, max_samples, replace=False)
        z_sub = z[idx]
    else:
        z_sub = z

    try:
        dists = pdist(z_sub)
        dist_mean = float(np.mean(dists))
        dist_std = float(np.std(dists))
    except Exception:
        dist_mean = dist_std = np.nan

    return {
        "diag_mean_norm": float(np.linalg.norm(z.mean(axis=0))),
        "diag_std_mean": float(std.mean()),
        "diag_std_min": float(std.min()),
        "diag_std_max": float(std.max()),
        "diag_var_mean": float(var.mean()),
        "diag_near_zero_dims": int((std < 1e-3).sum()),
        "diag_pairwise_dist_mean": dist_mean,
        "diag_pairwise_dist_std": dist_std,
    }
