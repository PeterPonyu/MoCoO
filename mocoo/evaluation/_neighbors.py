"""Shared nearest-neighbor utilities for evaluation metrics."""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def knn_indices(X: np.ndarray, k: int) -> np.ndarray:
    """Return k-NN index arrays for rows of X (excluding self).

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    k : int
        Number of neighbors.

    Returns
    -------
    np.ndarray, shape (n_samples, k)
        Indices of k nearest neighbors for each sample.
    """
    X = np.asarray(X, dtype=float)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn.fit(X)
    return nn.kneighbors(X, return_distance=False)[:, 1:]
