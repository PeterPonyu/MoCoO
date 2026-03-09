"""2D projection helpers (UMAP, tSNE) for DRE/DREX evaluation."""

import numpy as np


def compute_2d_projections(latent: np.ndarray):
    """Compute UMAP and tSNE embeddings using scanpy.

    Parameters
    ----------
    latent : np.ndarray, shape (n_cells, latent_dim)

    Returns
    -------
    tuple[np.ndarray | None, np.ndarray | None]
        (umap_2d, tsne_2d), either may be None on failure.
    """
    latent = np.asarray(latent, dtype=float)
    import scanpy as sc

    umap_2d = None
    tsne_2d = None
    try:
        adata = sc.AnnData(latent.astype(np.float32))
        sc.pp.neighbors(adata, use_rep="X", n_neighbors=15)
        sc.tl.umap(adata)
        umap_2d = adata.obsm["X_umap"]
        sc.tl.tsne(adata, use_rep="X")
        tsne_2d = adata.obsm["X_tsne"]
    except Exception as e:
        print(f"  Warning: 2D projection failed: {e}")
    return umap_2d, tsne_2d
