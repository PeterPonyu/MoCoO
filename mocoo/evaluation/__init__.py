"""
mocoo.evaluation -- Unified evaluation API for MoCoO.

Primary entry point
-------------------
.. autofunction:: compute_all_metrics

Gold-standard evaluators (co-ranking matrix / full SVD)
-------------------------------------------------------
* :class:`DimensionalityReductionEvaluator` -- co-ranking matrix DRE
* :class:`SingleCellLatentSpaceEvaluator`  -- SVD-based latent quality

Benchmark-optimized metric groups
----------------------------------
* :func:`compute_clustering_metrics` -- NMI, ARI, ASW, DAV, CAL, COR
* :func:`compute_dre_metrics`        -- kNN-based DRE (fast)
* :func:`compute_lse_metrics`        -- SVD-based LSE (fast)
* :func:`compute_drex_metrics`       -- Extended DR metrics
* :func:`compute_lsex_metrics`       -- Extended latent metrics
* :func:`compute_latent_diagnostics` -- Collapse / redundancy stats
* :func:`compute_batch_integration`  -- scIB batch integration (lazy import)

Display metadata
-----------------
* :data:`CORE_METRICS`, :data:`ALL_METRIC_GROUPS`
"""

import numpy as np

# ── Gold-standard evaluators ────────────────────────────────────────────────
from .dre import (
    DimensionalityReductionEvaluator,
    evaluate_dimensionality_reduction,
    compare_dimensionality_reduction_methods,
)
from .lse import (
    SingleCellLatentSpaceEvaluator,
    evaluate_single_cell_latent_space,
    compare_single_cell_methods,
)

# ── Benchmark-optimized metric groups ───────────────────────────────────────
from .clustering import compute_clustering_metrics
from .bench import compute_dre_metrics, compute_lse_metrics
from .drex import compute_drex_metrics
from .lsex import compute_lsex_metrics
from .diagnostics import compute_latent_diagnostics

# ── Display metadata ───────────────────────────────────────────────────────
from .metadata import (
    CORE_METRICS,
    ALL_METRIC_GROUPS,
    EXT_METRICS_CLUSTERING,
    EXT_METRICS_DRE,
    EXT_METRICS_LSE,
    EXT_METRICS_DREX,
    EXT_METRICS_LSEX,
)

# ── Internal helpers ────────────────────────────────────────────────────────
from . import _projections


def compute_all_metrics(
    latent: np.ndarray,
    labels: np.ndarray,
    dre_k: int = 15,
    include_batch: bool = False,
    cell_type_labels: np.ndarray = None,
    batch_labels: np.ndarray = None,
) -> dict:
    """Compute the full metric battery. Single source of truth.

    Parameters
    ----------
    latent : np.ndarray, shape (n_cells, latent_dim)
        Latent embeddings.
    labels : array-like, shape (n_cells,)
        Ground-truth integer labels for clustering evaluation.
    dre_k : int
        Number of neighbors for DRE / DREX evaluations.
    include_batch : bool
        Whether to compute batch integration metrics (requires ``scib``).
    cell_type_labels : array-like, optional
        String cell-type labels for batch integration.
    batch_labels : array-like, optional
        String batch labels for batch integration.

    Returns
    -------
    dict
        All metric values (NaN for any that fail).
        Private keys ``_umap_2d`` and ``_tsne_2d`` hold projection arrays
        for downstream visualization.
    """
    latent = np.asarray(latent, dtype=float)
    labels = np.asarray(labels, dtype=int)
    metrics = {}

    # 1. Clustering
    metrics.update(compute_clustering_metrics(latent, labels))

    # 2. 2D projections
    umap_2d, tsne_2d = _projections.compute_2d_projections(latent)

    # 3. DRE (UMAP)
    if umap_2d is not None:
        metrics.update(compute_dre_metrics(latent, umap_2d, dre_k, "DRE_umap"))
    else:
        for k in ("distance_correlation", "Q_local", "Q_global", "overall_quality"):
            metrics[f"DRE_umap_{k}"] = np.nan

    # 4. DRE (tSNE)
    if tsne_2d is not None:
        metrics.update(compute_dre_metrics(latent, tsne_2d, dre_k, "DRE_tsne"))
    else:
        for k in ("distance_correlation", "Q_local", "Q_global", "overall_quality"):
            metrics[f"DRE_tsne_{k}"] = np.nan

    # 5. LSE
    metrics.update(compute_lse_metrics(latent))

    # 6. DREX (using UMAP)
    if umap_2d is not None:
        metrics.update(compute_drex_metrics(latent, umap_2d, dre_k))
    else:
        for k in (
            "trustworthiness",
            "continuity",
            "distance_spearman",
            "distance_pearson",
            "local_scale_quality",
            "neighborhood_symmetry",
            "overall_quality",
        ):
            metrics[f"DREX_{k}"] = np.nan

    # 7. LSEX
    metrics.update(compute_lsex_metrics(latent, dre_k))

    # 8. Latent diagnostics
    metrics.update(compute_latent_diagnostics(latent))

    # 9. Batch integration (optional, lazy import)
    if include_batch and batch_labels is not None:
        from .batch import compute_batch_integration

        metrics.update(
            compute_batch_integration(latent, cell_type_labels, batch_labels)
        )

    # 10. Store projections for visualization
    metrics["_umap_2d"] = umap_2d
    metrics["_tsne_2d"] = tsne_2d

    return metrics


__all__ = [
    # Gold-standard evaluators
    "DimensionalityReductionEvaluator",
    "evaluate_dimensionality_reduction",
    "compare_dimensionality_reduction_methods",
    "SingleCellLatentSpaceEvaluator",
    "evaluate_single_cell_latent_space",
    "compare_single_cell_methods",
    # Unified benchmark API
    "compute_all_metrics",
    # Individual metric groups
    "compute_clustering_metrics",
    "compute_dre_metrics",
    "compute_lse_metrics",
    "compute_drex_metrics",
    "compute_lsex_metrics",
    "compute_latent_diagnostics",
    # Display metadata
    "CORE_METRICS",
    "ALL_METRIC_GROUPS",
    "EXT_METRICS_CLUSTERING",
    "EXT_METRICS_DRE",
    "EXT_METRICS_LSE",
    "EXT_METRICS_DREX",
    "EXT_METRICS_LSEX",
]
