"""Expanded metrics computation for MoCoO benchmark.

DEPRECATED: Use mocoo.evaluation.compute_all_metrics() directly.
This module is retained for backward compatibility and re-exports
the canonical implementations from the mocoo.evaluation package.
"""

# Re-export the unified API from the package
from mocoo.evaluation import compute_all_metrics
from mocoo.evaluation.clustering import compute_clustering_metrics
from mocoo.evaluation.diagnostics import compute_latent_diagnostics
from mocoo.evaluation.metadata import (
    CORE_METRICS,
    ALL_METRIC_GROUPS,
    EXT_METRICS_CLUSTERING,
    EXT_METRICS_DRE,
    EXT_METRICS_LSE,
    EXT_METRICS_DREX,
    EXT_METRICS_LSEX,
)
