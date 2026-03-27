#!/usr/bin/env python3
"""Run missing dataset×method combos one at a time with per-method output."""
import gc, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.sparse import issparse
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                              davies_bouldin_score, normalized_mutual_info_score,
                              silhouette_score)

warnings.filterwarnings("ignore")

SKILL_PARENT = str(Path.home() / ".copilot" / "skills")
sys.path.insert(0, SKILL_PARENT)

BASE = Path.home()
DATASETS = {
    "dentate": (str(BASE / "vGAE_LAB/data/dentate.h5ad"), "Clusters"),
    "hemato": (str(BASE / "Downloads/DevelopmentDatasets/hemato.h5ad"), "Cell type"),
    "lung": (str(BASE / "Downloads/DevelopmentDatasets/lung.h5ad"), "clusters"),
    "spinoids": (str(BASE / "LAB/data/spinoids.h5ad"), "seurat_clusters"),
}

UNIFIED = {
    "CellBLAST": ("create_cellblast_model", {"latent_dim": 10, "hidden_dims": [256, 128]}),
    "CLEAR": ("create_clear_model", {"latent_dim": 128, "hidden_dims": [512, 256]}),
    "GM-VAE": ("create_gmvae_model", {"latent_dim": 10, "hidden_dims": [512, 256], "distribution": "euclidean"}),
    "SCALEX": ("create_scalex_model", {"latent_dim": 10, "hidden_dims": [512, 256]}),
    "scDAC": ("create_scdac_model", {"latent_dim": 32}),
    "scDeepCluster": ("create_scdeepcluster_model", {"latent_dim": 32, "n_clusters": 10}),
    "scSMD": ("create_scsmd_model", {"latent_dim": 10, "n_clusters": 10}),
    "siVAE": ("create_sivae_model", {"latent_dim": 50, "hidden_dims": [512, 256]}),
}
_TUPLE = {"CLEAR"}

OUT = Path(__file__).resolve().parent.parent.parent / "results/baselines/gap_fill.csv"


def load(ds):
    path, col = DATASETS[ds]
    a = sc.read_h5ad(path)
    a.var_names_make_unique()
    if a.n_obs > 3000:
        sc.pp.subsample(a, n_obs=3000, random_state=0)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=min(3000, a.n_vars))
    a = a[:, a.var["highly_variable"]].copy()
    X = a.X.toarray() if issparse(a.X) else np.array(a.X)
    X = np.nan_to_num(X, nan=0.0)
    labels = a.obs[col].astype(str).values
    return X, labels


def metrics(lat, lab):
    km = KMeans(n_clusters=len(np.unique(lab)), random_state=0, n_init=10)
    pred = km.fit_predict(lat)
    return {"ARI": round(adjusted_rand_score(lab, pred), 4),
            "NMI": round(normalized_mutual_info_score(lab, pred), 4),
            "ASW": round(silhouette_score(lat, lab), 4),
            "CH": round(calinski_harabasz_score(lat, lab), 2),
            "DB": round(davies_bouldin_score(lat, lab), 4)}


def run_ml(name, X):
    from sklearn.decomposition import PCA, FastICA, NMF, FactorAnalysis, TruncatedSVD
    c = min(50, X.shape[1])
    m = {"PCA": PCA(n_components=c),
         "ICA": FastICA(n_components=c, random_state=0, max_iter=500),
         "NMF": NMF(n_components=c, random_state=0, max_iter=500),
         "FA": FactorAnalysis(n_components=c, random_state=0),
         "TruncatedSVD": TruncatedSVD(n_components=c, random_state=0)}
    X_in = np.clip(X, 0, None) if name == "NMF" else X
    return m[name].fit_transform(X_in)


class _TL:
    def __init__(self, d, bs=128, shuf=True, tup=False):
        self.d, self.bs, self.shuf, self.tup = d, bs, shuf, tup
    def __iter__(self):
        n = self.d.shape[0]
        idx = torch.randperm(n) if self.shuf else torch.arange(n)
        for i in range(0, n, self.bs):
            b = self.d[idx[i:i+self.bs]]
            yield (b, b) if self.tup else b
    def __len__(self):
        return (self.d.shape[0] + self.bs - 1) // self.bs


def run_dl(name, X, n_clusters):
    import importlib
    um = importlib.import_module("external-benchmarker.unified_models")
    fn, kw = UNIFIED[name]
    kw = {**kw, "input_dim": X.shape[1]}
    if "n_clusters" in kw:
        kw["n_clusters"] = n_clusters
    torch.manual_seed(0); np.random.seed(0)
    model = getattr(um, fn)(**kw)
    Xt = torch.FloatTensor(X)
    tup = name in _TUPLE
    model.fit(_TL(Xt, tup=tup), epochs=50, lr=1e-3, device="cpu", verbose=False)
    lat = model.encode(_TL(Xt, bs=256, shuf=False, tup=tup), device="cpu")
    if isinstance(lat, torch.Tensor):
        lat = lat.detach().cpu().numpy()
    return lat


def main():
    rows = []
    if OUT.exists():
        rows = pd.read_csv(OUT).to_dict("records")
        print(f"Resuming with {len(rows)} existing rows")

    done = {(r["dataset"], r["method"]) for r in rows}
    all_methods = list(["PCA", "ICA", "NMF", "FA", "TruncatedSVD"] + list(UNIFIED.keys()))

    for ds in DATASETS:
        print(f"\n{'='*50}\n{ds}\n{'='*50}")
        X, lab = load(ds)
        nc = len(np.unique(lab))
        print(f"  {X.shape[0]} cells × {X.shape[1]} genes, {nc} clusters")

        for meth in all_methods:
            if (ds, meth) in done:
                print(f"  {meth}: already done, skipping")
                continue
            t0 = time.time()
            try:
                if meth in UNIFIED:
                    lat = run_dl(meth, X, nc)
                else:
                    lat = run_ml(meth, X)
                m = metrics(lat, lab)
                m.update({"method": meth, "dataset": ds, "seed": 0,
                          "train_time_s": round(time.time()-t0, 1)})
                rows.append(m)
                print(f"  {meth}: ARI={m['ARI']:.4f} NMI={m['NMI']:.4f} t={time.time()-t0:.0f}s")
            except Exception as e:
                print(f"  {meth}: ERROR - {e}")
            gc.collect()

        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"  [saved {len(rows)} rows]")

    print(f"\nDone. {len(rows)} rows in {OUT}")


if __name__ == "__main__":
    main()
