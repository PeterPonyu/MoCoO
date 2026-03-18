"""Cell type annotation transfer via prototypes or kNN."""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, Optional


def annotate_by_prototype(
    latent: np.ndarray,
    prototypes: np.ndarray,
    reference_labels: Optional[np.ndarray] = None,
) -> Dict:
    """Assign cell types by nearest learned prototype.

    Parameters
    ----------
    latent : np.ndarray, shape (N, D)
    prototypes : np.ndarray, shape (P, D)
    reference_labels : np.ndarray (P,), optional
        Labels for each prototype. Uses prototype index if None.

    Returns
    -------
    dict with 'labels', 'distances', 'confidence'
    """
    dists = np.linalg.norm(
        latent[:, None, :] - prototypes[None, :, :], axis=2,
    )  # (N, P)
    assignments = dists.argmin(axis=1)
    min_dists = dists.min(axis=1)

    neg_dists = -dists
    shifted = neg_dists - neg_dists.max(axis=1, keepdims=True)
    exp_d = np.exp(shifted)
    softmax = exp_d / exp_d.sum(axis=1, keepdims=True)
    confidence = softmax.max(axis=1)

    labels = reference_labels[assignments] if reference_labels is not None else assignments

    return {'labels': labels, 'distances': min_dists, 'confidence': confidence}


def annotate_by_knn(
    query_latent: np.ndarray,
    reference_latent: np.ndarray,
    reference_labels: np.ndarray,
    k: int = 15,
) -> Dict:
    """kNN-based annotation transfer from reference to query cells.

    Parameters
    ----------
    query_latent : np.ndarray (N_query, D)
    reference_latent : np.ndarray (N_ref, D)
    reference_labels : np.ndarray (N_ref,)
    k : int

    Returns
    -------
    dict with 'labels', 'confidence', 'probabilities'
    """
    k_eff = min(k, len(reference_latent))
    knn = KNeighborsClassifier(n_neighbors=k_eff)
    knn.fit(reference_latent, reference_labels)

    labels = knn.predict(query_latent)
    probs = knn.predict_proba(query_latent)
    confidence = probs.max(axis=1)

    return {'labels': labels, 'confidence': confidence, 'probabilities': probs}


def evaluate_annotation(predicted: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
    """Evaluate annotation quality.

    Returns
    -------
    dict with 'accuracy', 'f1_macro', 'f1_weighted'
    """
    return {
        'accuracy': float(accuracy_score(ground_truth, predicted)),
        'f1_macro': float(f1_score(ground_truth, predicted, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(ground_truth, predicted, average='weighted', zero_division=0)),
    }
