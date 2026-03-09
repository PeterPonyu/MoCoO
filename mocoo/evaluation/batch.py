"""Batch integration metrics using scIB.

Metrics: iLISI, bASW, cLISI, graph_conn, iso_label_ASW,
bio_conservation, batch_correction, overall_score.

The ``scib`` library is imported lazily so the package does not
hard-depend on it.
"""

import numpy as np


def compute_batch_integration(
    latent: np.ndarray,
    cell_type_labels: np.ndarray,
    batch_labels: np.ndarray,
) -> dict:
    """Compute batch integration metrics from latent embeddings.

    Parameters
    ----------
    latent : np.ndarray, shape (n_cells, latent_dim)
    cell_type_labels : array-like, shape (n_cells,)
        Cell-type labels (string or categorical).
    batch_labels : array-like, shape (n_cells,)
        Batch labels (string or categorical).

    Returns
    -------
    dict
        Keys: iLISI, bASW, cLISI, graph_conn, iso_label_ASW,
        bio_conservation, batch_correction, overall_score.
    """
    import scib
    import scanpy as sc
    import pandas as pd

    # Build an AnnData with the embedding + metadata
    adata = sc.AnnData(
        X=latent.astype(np.float32),
        obs=pd.DataFrame(
            {
                "cell_type": pd.Categorical(cell_type_labels),
                "batch": pd.Categorical(batch_labels),
            }
        ),
    )
    adata.obsm["X_emb"] = latent.astype(np.float32)
    sc.pp.neighbors(adata, use_rep="X_emb", n_neighbors=15)

    metrics = {}

    # iLISI -- integration LISI (batch mixing quality)
    try:
        ilisi = scib.metrics.ilisi_graph(
            adata,
            batch_key="batch",
            type_="embed",
            use_rep="X_emb",
            n_cores=1,
        )
        metrics["iLISI"] = round(float(ilisi), 6)
    except Exception as e:
        print(f"    iLISI failed: {e}")
        metrics["iLISI"] = float("nan")

    # bASW -- batch-aware silhouette width
    try:
        basw = scib.metrics.silhouette_batch(
            adata,
            batch_key="batch",
            group_key="cell_type",
            embed="X_emb",
        )
        metrics["bASW"] = round(float(basw), 6)
    except Exception as e:
        print(f"    bASW failed: {e}")
        metrics["bASW"] = float("nan")

    # cLISI -- cell-type LISI (biological conservation)
    try:
        clisi = scib.metrics.clisi_graph(
            adata,
            label_key="cell_type",
            type_="embed",
            use_rep="X_emb",
            n_cores=1,
        )
        metrics["cLISI"] = round(float(clisi), 6)
    except Exception as e:
        print(f"    cLISI failed: {e}")
        metrics["cLISI"] = float("nan")

    # Graph connectivity (bio conservation)
    try:
        gc = scib.metrics.graph_connectivity(
            adata,
            label_key="cell_type",
        )
        metrics["graph_conn"] = round(float(gc), 6)
    except Exception as e:
        print(f"    graph_conn failed: {e}")
        metrics["graph_conn"] = float("nan")

    # Isolated label silhouette (bio conservation)
    try:
        iso_asw = scib.metrics.isolated_labels_asw(
            adata,
            label_key="cell_type",
            batch_key="batch",
            embed="X_emb",
        )
        metrics["iso_label_ASW"] = round(float(iso_asw), 6)
    except Exception as e:
        print(f"    iso_label_ASW failed: {e}")
        metrics["iso_label_ASW"] = float("nan")

    # Overall: 0.4 * bio_conservation + 0.6 * batch_correction (scIB convention)
    bio = np.nanmean(
        [
            metrics.get("cLISI", np.nan),
            metrics.get("graph_conn", np.nan),
            metrics.get("iso_label_ASW", np.nan),
        ]
    )
    batch = np.nanmean(
        [
            metrics.get("iLISI", np.nan),
            metrics.get("bASW", np.nan),
        ]
    )
    metrics["bio_conservation"] = round(float(bio), 6)
    metrics["batch_correction"] = round(float(batch), 6)
    metrics["overall_score"] = round(0.4 * bio + 0.6 * batch, 6)

    return metrics
