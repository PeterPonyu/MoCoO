"""DatasetRegistry — standardised data loading for MoCoO benchmarks.

Provides a single entry point for loading, preprocessing, and querying
metadata for all evaluated datasets.  Each registered dataset declares
its h5ad path, cell-type column, optional batch column, and suggested
preprocessing parameters.

Usage:
    from benchmarks.scripts.pipeline.dataset_registry import DatasetRegistry

    reg = DatasetRegistry()
    adata, meta = reg.load("IRALL", max_cells=3000, hvg=3000)
    print(meta)  # {'cell_type_col': ..., 'batch_col': ..., ...}
"""

from __future__ import annotations

import scanpy as sc
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from scipy.sparse import issparse
from sklearn.preprocessing import LabelEncoder


@dataclass
class DatasetSpec:
    """Specification for a registered dataset."""

    name: str
    path: str
    cell_type_col: str
    batch_col: Optional[str] = None
    description: str = ""
    organism: str = "unknown"
    tissue: str = "unknown"
    expected_cells: int = 0
    expected_types: int = 0
    default_max_cells: int = 3000
    default_hvg: int = 3000


# ═══════════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════════

_DATASETS_DIR = Path("/home/zeyufu/Desktop/datasets")

_REGISTRY: dict[str, DatasetSpec] = {
    "IRALL": DatasetSpec(
        name="IRALL",
        path="/home/zeyufu/LAB/scRL/IRALL.h5ad",
        cell_type_col="cell_type",
        batch_col="batch",
        description="Mouse haematopoiesis time-series (d0-d30)",
        organism="Mus musculus",
        tissue="bone marrow",
        expected_cells=41252,
        expected_types=12,
    ),
    "dentate": DatasetSpec(
        name="dentate",
        path="/home/zeyufu/vGAE_LAB/data/dentate.h5ad",
        cell_type_col="Clusters",
        batch_col=None,
        description="Mouse dentate gyrus neurogenesis",
        organism="Mus musculus",
        tissue="dentate gyrus",
        expected_cells=18213,
        expected_types=14,
    ),
    "endo": DatasetSpec(
        name="endo",
        path="/home/zeyufu/vGAE_LAB/data/endo.h5ad",
        cell_type_col="clusters_fine",
        batch_col="day",
        description="Mouse endocrine pancreas development",
        organism="Mus musculus",
        tissue="pancreas",
        expected_cells=2531,
        expected_types=13,
        default_max_cells=2500,  # small dataset
    ),
    "paul": DatasetSpec(
        name="paul",
        path="/home/zeyufu/LAB/data/paul.h5ad",
        cell_type_col="paul15_clusters",
        batch_col=None,
        description="Mouse myeloid/erythroid progenitor differentiation (Paul et al. 2015)",
        organism="Mus musculus",
        tissue="bone marrow",
        expected_cells=2730,
        expected_types=19,
        default_max_cells=2700,  # small dataset
    ),
    "spinoids": DatasetSpec(
        name="spinoids",
        path="/home/zeyufu/LAB/data/spinoids.h5ad",
        cell_type_col="annotation",
        batch_col=None,
        description="Human spinal cord organoid development",
        organism="Homo sapiens",
        tissue="spinal cord organoid",
        expected_cells=9619,
        expected_types=8,
    ),
}


class DatasetRegistry:
    """Registry of available datasets for MoCoO benchmarks."""

    def __init__(self):
        self._registry = dict(_REGISTRY)

    # ── Query ──────────────────────────────────────────────────────────────

    def list(self) -> list[str]:
        """Return registered dataset names."""
        return list(self._registry.keys())

    def info(self, name: str) -> DatasetSpec:
        """Return the DatasetSpec for a registered dataset."""
        if name not in self._registry:
            raise KeyError(
                f"Unknown dataset '{name}'. Available: {self.list()}"
            )
        return self._registry[name]

    def has_batch(self, name: str) -> bool:
        """Whether the dataset has batch labels."""
        return self.info(name).batch_col is not None

    # ── Registration ───────────────────────────────────────────────────────

    def register(self, spec: DatasetSpec):
        """Register a new dataset."""
        self._registry[spec.name] = spec

    # ── Loading ────────────────────────────────────────────────────────────

    def load(
        self,
        name: str,
        max_cells: Optional[int] = None,
        hvg: Optional[int] = None,
        seed: int = 42,
        normalize: bool = True,
    ) -> tuple:
        """Load and preprocess a dataset.

        Returns
        -------
        adata : AnnData
            Preprocessed AnnData object with:
            - Standardised ``obs['cell_type']`` column (always present)
            - Standardised ``obs['batch']`` column (if available)
            - ``layers['counts']`` preserved for NB/ZINB losses
            - HVG-filtered if *hvg* is set
        meta : dict
            Metadata about the loaded dataset.
        """
        spec = self.info(name)
        max_cells = max_cells or spec.default_max_cells
        hvg = hvg or spec.default_hvg

        adata = sc.read_h5ad(spec.path)
        adata.var_names_make_unique()

        # ── Standardise obs columns ──
        if spec.cell_type_col != "cell_type":
            adata.obs["cell_type"] = adata.obs[spec.cell_type_col].values
        if spec.batch_col and spec.batch_col != "batch":
            adata.obs["batch"] = adata.obs[spec.batch_col].values

        # Integer-encode cell types
        le = LabelEncoder()
        labels_int = le.fit_transform(adata.obs["cell_type"].values.astype(str))

        # ── Subsample ──
        original_n = adata.n_obs
        if adata.n_obs > max_cells:
            sc.pp.subsample(adata, n_obs=max_cells, random_state=seed)

        # ── Gene filtering ──
        sc.pp.filter_genes(adata, min_cells=10)

        # ── Ensure counts layer ──
        if "counts" not in adata.layers:
            X = adata.X.toarray() if issparse(adata.X) else adata.X.copy()
            adata.layers["counts"] = X

        # ── HVG selection ──
        if normalize and adata.n_vars > hvg:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            try:
                sc.pp.highly_variable_genes(
                    adata, n_top_genes=hvg, flavor="seurat_v3",
                    layer="counts",
                )
            except Exception:
                sc.pp.highly_variable_genes(adata, n_top_genes=hvg)
            adata = adata[:, adata.var["highly_variable"]].copy()

        # ── Metadata ──
        cell_types = adata.obs["cell_type"].astype(str).values
        batch_labels = (
            adata.obs["batch"].astype(str).values
            if "batch" in adata.obs.columns
            else None
        )

        meta = {
            "name": name,
            "spec": spec,
            "original_cells": original_n,
            "cells": adata.n_obs,
            "genes": adata.n_vars,
            "n_cell_types": len(np.unique(cell_types)),
            "n_batches": len(np.unique(batch_labels)) if batch_labels is not None else 0,
            "cell_type_col": "cell_type",
            "batch_col": "batch" if batch_labels is not None else None,
            "has_batch": batch_labels is not None,
        }

        return adata, meta

    def load_all(
        self,
        max_cells: Optional[int] = None,
        hvg: Optional[int] = None,
        seed: int = 42,
    ) -> list[tuple]:
        """Load all registered datasets.

        Returns list of (adata, meta) tuples.
        """
        results = []
        for name in self.list():
            try:
                adata, meta = self.load(name, max_cells, hvg, seed)
                results.append((adata, meta))
                print(f"  ✓ {name}: {meta['cells']} cells, "
                      f"{meta['n_cell_types']} types, "
                      f"{meta['n_batches']} batches")
            except Exception as e:
                print(f"  ✗ {name}: {e}")
        return results


# ── Convenience singleton ──────────────────────────────────────────────────
_default_registry: Optional[DatasetRegistry] = None


def get_registry() -> DatasetRegistry:
    """Return the default DatasetRegistry singleton."""
    global _default_registry
    if _default_registry is None:
        _default_registry = DatasetRegistry()
    return _default_registry
