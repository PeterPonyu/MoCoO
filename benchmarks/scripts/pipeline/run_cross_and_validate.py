#!/usr/bin/env python3
"""Cross-dataset benchmark + pseudotime-marker validation.

Runs the 6-config ablation on dentate and endo datasets (IRALL already done),
then validates pseudotime-marker gene correlations on all ODE configs,
and computes latent smoothness metrics.

Usage:
    python benchmarks/scripts/pipeline/run_cross_and_validate.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from mocoo import MoCoO
from benchmarks.scripts.pipeline.dataset_registry import get_registry
from mocoo.evaluation import SingleCellLatentSpaceEvaluator, DimensionalityReductionEvaluator

# ── Corrected config definitions (matching run_benchmark.py) ──────────────
SHARED = dict(
    latent_dim=32,
    hidden_dim=128,
    i_dim=4,
    lr=1e-4,
    batch_size=128,
    beta=1.0,
    recon=1.0,
    loss_mode="nb",
    random_seed=42,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
)

CONFIGS = {
    "VAE": dict(use_ode=False, use_moco=False, use_prototype=False),
    "VAE+ODE": dict(
        use_ode=True, use_moco=False, use_prototype=False,
        vae_reg=0.6, ode_reg=0.4,
    ),
    "VAE+MoCo": dict(
        use_ode=False, use_moco=True, use_prototype=False,
        moco_weight=0.5, moco_T=0.2, moco_K=4096,
    ),
    "VAE+MoCo+Proto": dict(
        use_ode=False, use_moco=True, use_prototype=True,
        n_prototypes=12, moco_weight=0.5, moco_T=0.2, moco_K=4096,
        proto_weight=0.05,
    ),
    "VAE+ODE+MoCo": dict(
        use_ode=True, use_moco=True, use_prototype=False,
        vae_reg=0.6, ode_reg=0.4,
        moco_weight=0.3, moco_T=0.2, moco_K=4096,
    ),
    "Full": dict(
        use_ode=True, use_moco=True, use_prototype=True,
        n_prototypes=12,
        vae_reg=0.6, ode_reg=0.4,
        moco_weight=0.3, moco_T=0.2, moco_K=4096,
        proto_weight=0.05,
    ),
}

# ── Canonical marker genes per dataset ────────────────────────────────────
MARKER_GENES = {
    "IRALL": [
        # Hematopoiesis markers
        "Cd34", "Kit", "Sca1",           # HSC/progenitor
        "Hba-a1", "Hba-a2", "Hbb-bs",   # Erythroid
        "Elane", "Mpo", "Ctsg",          # Granulocyte
        "Cd19", "Pax5", "Ebf1",          # B-cell
        "Cd3e", "Cd3d", "Cd4", "Cd8a",   # T-cell
        "Klrb1c", "Ncr1",               # NK
        "Itgam", "Csf1r",               # Monocyte/Macrophage
        "Pf4", "Ppbp",                   # Megakaryocyte
    ],
    "dentate": [
        # Dentate gyrus neurogenesis markers
        "Sox2", "Pax6", "Nes",           # Radial glia / NSC
        "Eomes", "Tbr2",                 # Intermediate progenitor
        "Neurod1", "Dcx", "Tbr1",        # Immature neuron
        "Prox1", "Calb1",               # Granule cell
        "Gad1", "Gad2",                 # Interneuron (GABAergic)
        "Gfap", "Aldh1l1",              # Astrocyte
        "Olig1", "Olig2",               # Oligodendrocyte
        "Slc1a3", "Fabp7",              # Astrocyte/RGC
        "Stmn1", "Mki67",               # Proliferating
    ],
    "endo": [
        # Endocrine pancreas differentiation markers  
        "Neurog3", "Neurod1",            # Endocrine progenitor
        "Ins1", "Ins2",                  # Beta cell
        "Gcg",                           # Alpha cell
        "Sst",                           # Delta cell
        "Ppy", "Ghrl",                   # PP / Epsilon cell
        "Sox9", "Hnf1b",                # Ductal
        "Krt19",                         # Ductal marker
        "Mki67", "Top2a",               # Proliferating
        "Chga", "Chgb",                 # Pan-endocrine
        "Pdx1", "Nkx6-1",              # Beta/progenitor TF
    ],
    "paul": [
        # Myeloid/erythroid progenitor markers (Paul et al. 2015)
        "Gata1", "Gata2", "Klf1",       # Erythroid TFs
        "Hba-a1", "Hba-a2", "Hbb-b1",   # Erythroid maturation
        "Cebpa", "Spi1",                # Myeloid TFs (PU.1)
        "Mpo", "Elane", "Ctsg",          # Granulocyte
        "Irf8", "Csf1r",               # Monocyte/DC
        "Cd34", "Kit",                   # HSC/progenitor
        "Itga2b", "Pf4",               # Megakaryocyte
        "Fcgr3", "Ly6c2",              # Neutrophil/Monocyte
        "Epor",                          # Erythroid receptor
    ],
    "spinoids": [
        # Spinal cord organoid development markers
        "SOX2", "PAX6", "NES",           # Neural progenitors
        "NEUROG1", "NEUROG2",            # Neurogenesis TFs
        "TUBB3", "MAP2", "DCX",          # Neurons
        "OLIG2", "NKX6-1",             # Motor neuron progenitors
        "ISL1", "MNX1",                 # Motor neurons
        "PAX3", "PAX7",                 # Dorsal progenitors
        "TBX6", "MEOX1",               # Somites
        "TBXT", "CDX2",                 # Axial progenitors
        "MKI67", "TOP2A",               # Proliferating
        "SOX10", "FOXD3",              # Neural crest
    ],
}


def run_single(name, adata, config, epochs, patience, val_every,
               track_metrics: bool = False):
    """Train one configuration, return result dict + model."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import (
        adjusted_rand_score, normalized_mutual_info_score,
        silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    )
    from sklearn.manifold import TSNE
    import umap as umap_lib

    params = {**SHARED, **config}
    model = MoCoO(adata, **params)
    model.fit(epochs=epochs, patience=patience, val_every=val_every,
              track_metrics=track_metrics)

    latent = model.get_latent()
    labels_all = model.labels
    n_clusters = len(np.unique(labels_all))

    pred = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(latent)
    result = {
        "config": name,
        "ARI": round(adjusted_rand_score(labels_all, pred), 4),
        "NMI": round(normalized_mutual_info_score(labels_all, pred), 4),
        "ASW": round(silhouette_score(latent, pred), 4),
        "CH": round(float(calinski_harabasz_score(latent, pred)), 2),
        "DB": round(davies_bouldin_score(latent, pred), 4),
        "actual_epochs": model.actual_epochs,
        "train_time_s": round(model.train_time, 1),
    }

    # ── LSE: Latent Structure Evaluation ──
    try:
        lse = SingleCellLatentSpaceEvaluator(data_type="trajectory", verbose=False)
        lse_res = lse.comprehensive_evaluation(latent)
        result["LSE_overall"] = round(lse_res["overall_quality"], 4)
        result["LSE_manifold_dim"] = round(lse_res["manifold_dimensionality"], 4)
        result["LSE_spectral_decay"] = round(lse_res["spectral_decay_rate"], 4)
        result["LSE_participation"] = round(lse_res["participation_ratio"], 4)
        result["LSE_anisotropy"] = round(lse_res["anisotropy_score"], 4)
        result["LSE_directionality"] = round(lse_res["trajectory_directionality"], 4)
        result["LSE_noise_resilience"] = round(lse_res["noise_resilience"], 4)
        print(f"    LSE overall={lse_res['overall_quality']:.4f}")
    except Exception as e:
        print(f"    LSE failed: {e}")

    # ── DRE: Dimensionality Reduction Evaluation (latent vs UMAP/t-SNE) ──
    try:
        dre = DimensionalityReductionEvaluator(verbose=False)
        # UMAP embedding
        umap_emb = umap_lib.UMAP(n_components=2, random_state=42).fit_transform(latent)
        dre_umap = dre.comprehensive_evaluation(latent, umap_emb, k=10)
        result["DRE_UMAP_dist_corr"] = round(dre_umap["distance_correlation"], 4)
        result["DRE_UMAP_Q_local"] = round(dre_umap["Q_local"], 4)
        result["DRE_UMAP_Q_global"] = round(dre_umap["Q_global"], 4)
        result["DRE_UMAP_overall"] = round(dre_umap["overall_quality"], 4)
        print(f"    DRE-UMAP overall={dre_umap['overall_quality']:.4f}")

        # t-SNE embedding
        tsne_emb = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(latent)
        dre_tsne = dre.comprehensive_evaluation(latent, tsne_emb, k=10)
        result["DRE_tSNE_dist_corr"] = round(dre_tsne["distance_correlation"], 4)
        result["DRE_tSNE_Q_local"] = round(dre_tsne["Q_local"], 4)
        result["DRE_tSNE_Q_global"] = round(dre_tsne["Q_global"], 4)
        result["DRE_tSNE_overall"] = round(dre_tsne["overall_quality"], 4)
        print(f"    DRE-tSNE overall={dre_tsne['overall_quality']:.4f}")
    except Exception as e:
        print(f"    DRE failed: {e}")

    return result, model


def validate_pseudotime(model, adata, dataset_name, config_name, outdir):
    """Validate pseudotime-marker correlations for ODE configs."""
    markers = MARKER_GENES.get(dataset_name, [])
    if not markers:
        return {}

    # Filter to genes present in adata
    available = [g for g in markers if g in adata.var_names]
    print(f"    Marker genes: {len(available)}/{len(markers)} found in adata")

    if not available:
        # Fall back to top correlated genes
        print(f"    Using top-20 auto-detected correlated genes instead")
        corr = model.pseudotime_marker_correlation(adata, top_n=20)
    else:
        corr = model.pseudotime_marker_correlation(adata, marker_genes=available)

    # Print top correlations
    sorted_genes = sorted(corr.items(), key=lambda x: abs(x[1]['spearman_r']), reverse=True)
    print(f"    Top pseudotime-gene correlations ({config_name}):")
    for gene, vals in sorted_genes[:10]:
        r = vals['spearman_r']
        p = vals['spearman_p']
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"      {gene:>12s}: ρ={r:+.3f} (p={p:.2e}) {sig}")

    # Save
    corr_path = outdir / f"pseudotime_markers_{config_name.replace('+', '_')}.json"
    with open(corr_path, "w") as f:
        json.dump(corr, f, indent=2)

    return corr


def compute_smoothness(model, config_name):
    """Compute latent space smoothness for a trained model."""
    sm = model.get_latent_smoothness()
    print(f"    Smoothness ({config_name}): "
          f"kNN-H={sm['knn_entropy']:.3f}  "
          f"dist={sm['pairwise_dist_mean']:.2f}±{sm['pairwise_dist_std']:.2f}  "
          f"eff_dim={sm['effective_dim']:.1f}")
    return sm


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=["dentate", "endo"])
    parser.add_argument("--configs", nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--max-cells", type=int, default=3000)
    parser.add_argument("--hvg", type=int, default=3000)
    args = parser.parse_args()

    reg = get_registry()
    configs_to_run = args.configs if args.configs else list(CONFIGS.keys())
    outdir = Path(__file__).resolve().parent.parent.parent / "results"
    outdir.mkdir(parents=True, exist_ok=True)

    all_results = []
    all_smoothness = {}

    for ds_name in args.datasets:
        print(f"\n{'═'*70}")
        print(f"  DATASET: {ds_name}")
        print(f"{'═'*70}")

        adata, meta = reg.load(ds_name, max_cells=args.max_cells, hvg=args.hvg)
        ds_outdir = outdir / ds_name
        ds_outdir.mkdir(parents=True, exist_ok=True)

        ds_results = []
        ds_smoothness = {}

        for cfg_name in configs_to_run:
            if cfg_name not in CONFIGS:
                continue

            print(f"\n  ── {cfg_name} ──")
            try:
                r, model = run_single(
                    cfg_name, adata, CONFIGS[cfg_name],
                    args.epochs, args.patience, args.val_every,
                )
                r["dataset"] = ds_name

                # Smoothness analysis
                sm = compute_smoothness(model, cfg_name)
                r["knn_entropy"] = sm["knn_entropy"]
                r["effective_dim"] = sm["effective_dim"]
                ds_smoothness[cfg_name] = sm

                # Pseudotime-marker correlation (ODE configs only)
                if CONFIGS[cfg_name].get("use_ode", False):
                    corr = validate_pseudotime(model, adata, ds_name, cfg_name, ds_outdir)
                    if corr:
                        top_genes = sorted(corr.items(),
                                           key=lambda x: abs(x[1]['spearman_r']),
                                           reverse=True)
                        r["top_marker_corr"] = round(abs(top_genes[0][1]['spearman_r']), 4)
                        r["n_sig_markers"] = sum(
                            1 for _, v in corr.items() if v['spearman_p'] < 0.05
                        )

                ds_results.append(r)

                # Save per-config JSON (convert numpy scalars to Python floats)
                saveable = {}
                for k, v in r.items():
                    if isinstance(v, (np.ndarray, list)):
                        continue
                    elif isinstance(v, (np.floating, np.integer)):
                        saveable[k] = float(v)
                    else:
                        saveable[k] = v
                with open(ds_outdir / f"{cfg_name.replace('+', '_')}.json", "w") as f:
                    json.dump(saveable, f, indent=2)

                print(f"    ARI={r['ARI']:.3f} NMI={r['NMI']:.3f} "
                      f"ASW={r['ASW']:.3f} CH={r['CH']:.1f} DB={r['DB']:.3f}")

            except Exception as e:
                print(f"    FAILED: {e}")
                import traceback; traceback.print_exc()

        # Dataset summary CSV
        if ds_results:
            fields = [k for k in ds_results[0]
                      if not isinstance(ds_results[0][k], (np.ndarray, list))]
            with open(ds_outdir / "summary.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                for r in ds_results:
                    w.writerow({k: v for k, v in r.items()
                                if not isinstance(v, (np.ndarray, list))})

        all_results.extend(ds_results)
        all_smoothness[ds_name] = ds_smoothness

    # ── Cross-dataset summary ──
    if all_results:
        print(f"\n{'═'*70}")
        print("  CROSS-DATASET SUMMARY")
        print(f"{'═'*70}")
        print(f"{'Config':<20} {'Dataset':<10} {'ARI':>6} {'NMI':>6} {'ASW':>6} "
              f"{'kNN-H':>6} {'EffDim':>6}")
        print(f"{'─'*70}")
        for r in all_results:
            print(f"{r['config']:<20} {r['dataset']:<10} "
                  f"{r['ARI']:>6.3f} {r['NMI']:>6.3f} {r['ASW']:>6.3f} "
                  f"{r.get('knn_entropy', 0):>6.3f} "
                  f"{r.get('effective_dim', 0):>6.1f}")

        # Aggregate by config
        by_config = defaultdict(list)
        for r in all_results:
            by_config[r["config"]].append(r)

        print(f"\n{'Config':<20} {'n':>3} {'ARI (mean±std)':>16} "
              f"{'NMI (mean±std)':>16} {'ASW (mean±std)':>16}")
        print(f"{'─'*70}")
        for cfg, runs in by_config.items():
            aris = [r["ARI"] for r in runs]
            nmis = [r["NMI"] for r in runs]
            asws = [r["ASW"] for r in runs]
            print(f"{cfg:<20} {len(runs):>3} "
                  f"{np.mean(aris):>6.3f}±{np.std(aris):.3f}  "
                  f"{np.mean(nmis):>6.3f}±{np.std(nmis):.3f}  "
                  f"{np.mean(asws):>6.3f}±{np.std(asws):.3f}")

        # Save meta analysis
        meta_path = outdir / "meta_analysis.csv"
        fields = sorted({k for r in all_results for k in r
                         if not isinstance(r.get(k), (np.ndarray, list))})
        with open(meta_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in all_results:
                w.writerow({k: v for k, v in r.items()
                            if not isinstance(v, (np.ndarray, list))})
        print(f"\nMeta-analysis saved to {meta_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
