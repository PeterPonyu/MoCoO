"""
mocoo.evaluation -- Dimensionality-reduction and latent-space evaluation tools.

This subpackage provides two evaluator classes:

* :class:`DimensionalityReductionEvaluator` -- assesses how well a
  dimensionality-reduction method preserves high-dimensional structure
  (distance correlation, Q_local, Q_global).
* :class:`SingleCellLatentSpaceEvaluator` -- evaluates latent-space quality
  specifically for single-cell data (trajectory and steady-state).

Convenience functions are also re-exported for quick usage::

    from mocoo.evaluation import evaluate_dimensionality_reduction
    from mocoo.evaluation import evaluate_single_cell_latent_space
"""

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

__all__ = [
    "DimensionalityReductionEvaluator",
    "evaluate_dimensionality_reduction",
    "compare_dimensionality_reduction_methods",
    "SingleCellLatentSpaceEvaluator",
    "evaluate_single_cell_latent_space",
    "compare_single_cell_methods",
]
