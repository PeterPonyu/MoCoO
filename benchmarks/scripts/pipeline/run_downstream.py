"""
Downstream Analysis Runner — All 7 modules on one or all datasets.

Trains the Full MoCoO configuration (ODE + MoCo + Prototypes) plus FM,
then runs all implemented downstream analysis modules:

  A1  Gene importance via decoder Jacobian
  B1  Gene-space RNA velocity via chain rule
  B2  Trajectory branching detection via divergence
  A2  Differential expression via decoder perturbation
  C1  In-silico cell generation quality
  D1  Cell type annotation transfer (prototype + kNN)
  F1  Posterior sampling uncertainty

Output structure:
  benchmarks/results/<dataset>/downstream/
    gene_importance.json
    gene_velocity.npy
    branching.json
    differential_expression.json
    generation_quality.json
    annotation_transfer.json
    uncertainty.json
    summary.json
    model.pt                      (if --save-model)

Usage:
  python benchmarks/scripts/pipeline/run_downstream.py --dataset IRALL
  python benchmarks/scripts/pipeline/run_downstream.py --dataset all
  python benchmarks/scripts/pipeline/run_downstream.py --dataset IRALL --module A1
  python benchmarks/scripts/pipeline/run_downstream.py --dataset IRALL --save-model
  python benchmarks/scripts/pipeline/run_downstream.py --dataset IRALL --load-model path.pt
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import scanpy as sc
import torch

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from mocoo import MoCoO
from mocoo.configs import (
    load_config,
    get_shared_params,
    get_model_configs,
    get_training_params,
)

# ── Lazy imports for downstream modules ─────────────────────────────────────
# Imported at call site to keep top-level imports minimal.


# ═════════════════════════════════════════════════════════════════════════════
# Dataset registry (matches fig1_training_pipeline.py)
# ═════════════════════════════════════════════════════════════════════════════

DATASET_SPECS = {
    "IRALL": {
        "path": "LAB/scRL/IRALL.h5ad",
        "max_cells": 3000, "hvg": 3000,
        "cell_type_col": "cell_type",
        "known_markers": ["Gata1", "Gata2", "Spi1", "Cebpa", "Mpo"],
    },
    "dentate": {
        "path": "vGAE_LAB/data/dentate.h5ad",
        "max_cells": 3000, "hvg": 3000,
        "cell_type_col": "Clusters",
        "known_markers": ["Dcx", "Prox1", "Sox2", "Neurod1", "Gfap"],
    },
    "endo": {
        "path": "vGAE_LAB/data/endo.h5ad",
        "max_cells": 2500, "hvg": 3000,
        "cell_type_col": "clusters_fine",
        "known_markers": ["Ins1", "Ins2", "Gcg", "Sst", "Ppy"],
    },
    "paul": {
        "path": "LAB/data/paul.h5ad",
        "max_cells": 2700, "hvg": 3000,
        "cell_type_col": "paul15_clusters",
        "known_markers": ["Elane", "Hba-a2", "Irf8", "Gfi1", "Mpo"],
    },
    "spinoids": {
        "path": "LAB/data/spinoids.h5ad",
        "max_cells": 3000, "hvg": 3000,
        "cell_type_col": "annotation",
        "known_markers": ["SOX2", "PAX6", "NEUROD1", "OLIG2", "MKI67"],
    },
}


# ═════════════════════════════════════════════════════════════════════════════
# Data loading (reuses fig1_training_pipeline pattern)
# ═════════════════════════════════════════════════════════════════════════════

def load_dataset(path: str, max_cells: int, hvg: int, seed: int = 42):
    """Load, subsample, HVG-filter, ensure counts layer."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    adata = sc.read_h5ad(path)
    print(f"  Loaded: {adata.shape[0]} cells × {adata.shape[1]} genes")

    if adata.n_obs > max_cells:
        sc.pp.subsample(adata, n_obs=max_cells, random_state=seed)
        print(f"  Subsampled → {adata.n_obs} cells")

    sc.pp.filter_genes(adata, min_cells=10)

    if "counts" not in adata.layers:
        from scipy.sparse import issparse
        X = adata.X.toarray() if issparse(adata.X) else adata.X.copy()
        adata.layers["counts"] = X

    if adata.n_vars > hvg:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        try:
            sc.pp.highly_variable_genes(
                adata, n_top_genes=hvg, flavor="seurat_v3", layer="counts"
            )
        except (ImportError, Exception):
            sc.pp.highly_variable_genes(adata, n_top_genes=hvg)
        adata = adata[:, adata.var["highly_variable"]].copy()
        print(f"  HVG filtered → {adata.n_vars} genes")

    return adata


# ═════════════════════════════════════════════════════════════════════════════
# Training
# ═════════════════════════════════════════════════════════════════════════════

def train_full_model(adata, shared, training, fm_steps=2000, fm_lr=1e-3):
    """Train Full config (ODE+MoCo+Proto) and FM."""
    cfg = load_config("default")
    full_cfg = get_model_configs(cfg)["Full"]
    params = {**shared, **full_cfg}

    print("\n  ── Training Full model ──")
    model = MoCoO(adata, **params)
    model.fit(
        epochs=training["epochs"],
        patience=training["patience"],
        val_every=training["val_every"],
    )
    res = model.get_resource_metrics()
    print(f"    Epochs: {int(res['actual_epochs'])}, Time: {res['train_time']:.1f}s")

    print("  ── Training Flow Matching ──")
    model.train_fm(epochs=fm_steps, lr=fm_lr)
    print(f"    FM training complete")

    return model


# ═════════════════════════════════════════════════════════════════════════════
# Downstream analysis modules
# ═════════════════════════════════════════════════════════════════════════════

def run_A1_gene_importance(model, gene_names, outdir, known_markers=None):
    """A1: Gene importance via decoder Jacobian."""
    from mocoo.evaluation.gene_importance import rank_genes_by_jacobian

    print("    [A1] Gene importance via Jacobian...")
    t0 = time.time()
    jacobian = model.get_gene_jacobian(batch_size=128)
    result = rank_genes_by_jacobian(jacobian, gene_names, top_n=50)
    elapsed = time.time() - t0

    # Validate
    assert jacobian.shape[1] == len(gene_names), "Jacobian gene dim mismatch"
    assert np.all(np.isfinite(result["importance"])), "Non-finite importance"

    # Check known markers
    markers_found = []
    if known_markers:
        top50 = set(result["ranked_genes"][:50])
        for m in known_markers:
            # Case-insensitive search
            found = any(m.lower() == g.lower() for g in top50)
            markers_found.append({"marker": m, "in_top50": found})

    out = {
        "ranked_genes_top50": result["ranked_genes"][:50],
        "top_genes_per_dim": {
            str(k): v for k, v in result["top_genes_per_dim"].items()
        },
        "markers_found": markers_found,
        "elapsed_s": round(elapsed, 1),
    }

    with open(outdir / "gene_importance.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"         {elapsed:.1f}s — top gene: {result['ranked_genes'][0]}")
    return jacobian  # cached for B1


def run_B1_gene_velocity(model, outdir):
    """B1: Gene-space RNA velocity."""
    print("    [B1] Gene-space velocity...")
    t0 = time.time()
    velocity = model.get_gene_velocity(batch_size=128)
    elapsed = time.time() - t0

    assert velocity.ndim == 2, f"Expected 2D, got {velocity.ndim}D"
    assert np.all(np.isfinite(velocity)), "Non-finite velocity values"

    np.save(outdir / "gene_velocity.npy", velocity)
    print(f"         {elapsed:.1f}s — shape {velocity.shape}")


def run_B2_branching(model, outdir):
    """B2: Trajectory branching detection."""
    from mocoo.evaluation.branching import detect_branch_points

    print("    [B2] Branching detection...")
    t0 = time.time()
    divergence = model.get_divergence(batch_size=128)
    latent = model.get_latent()
    result = detect_branch_points(
        divergence, latent, threshold_quantile=0.9, eps=1.5, min_samples=5,
    )
    elapsed = time.time() - t0

    assert result["divergence"].shape[0] == latent.shape[0]
    assert np.all(np.isfinite(result["divergence"]))

    out = {
        "n_branches": int(result["n_branches"]),
        "n_branch_cells": int(result["is_branch_point"].sum()),
        "divergence_stats": {
            "mean": float(np.mean(result["divergence"])),
            "std": float(np.std(result["divergence"])),
            "q90": float(np.quantile(np.abs(result["divergence"]), 0.9)),
        },
        "elapsed_s": round(elapsed, 1),
    }

    with open(outdir / "branching.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"         {elapsed:.1f}s — {out['n_branches']} branches, "
          f"{out['n_branch_cells']} branch cells")


def run_A2_differential_expression(model, gene_names, outdir):
    """A2: Differential expression via decoder perturbation."""
    from mocoo.evaluation.differential_expression import decoder_de

    print("    [A2] Differential expression...")
    t0 = time.time()

    latent = model.get_latent_qm()
    labels = model.labels  # KMeans pseudo-labels
    decoded_all = model.get_decoded()  # uses q_m after fix

    # Compute per-cluster centroids and decode them
    decoded_centroids = {}
    for lab in np.unique(labels):
        z_mean = latent[labels == lab].mean(axis=0)
        decoded_centroids[lab] = model.get_decoded(z_mean[None, :])[0]

    result = decoder_de(
        decoded_centroids, decoded_all=decoded_all, labels=labels,
        top_n=50, gene_names=gene_names,
    )
    elapsed = time.time() - t0

    # Serialise
    out = {}
    for lab, data in result.items():
        out[str(lab)] = {
            "top_genes": data["top_genes"],
            "top_log2fc": [float(data["log2fc"][i]) for i in
                           np.argsort(-np.abs(data["log2fc"]))[:50]],
        }
        if "pvalues" in data and data["pvalues"] is not None:
            sig = float((data["pvalues"] < 0.05).mean())
            out[str(lab)]["frac_sig_005"] = sig

    out["_meta"] = {
        "n_clusters": len(result),
        "elapsed_s": round(elapsed, 1),
    }

    with open(outdir / "differential_expression.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"         {elapsed:.1f}s — {len(result)} clusters")


def run_C1_generation_quality(model, outdir):
    """C1: In-silico cell generation quality."""
    from mocoo.evaluation.generation_quality import generation_quality_metrics

    print("    [C1] Generation quality...")
    t0 = time.time()

    real_latent = model.get_latent()
    gen_latent = model.generate_cells(n=500, steps=100, decode=False)
    metrics = generation_quality_metrics(real_latent, gen_latent, k=10)
    elapsed = time.time() - t0

    assert 0.0 <= metrics["coverage"] <= 1.0
    assert 0.0 <= metrics["authenticity"] <= 1.0

    out = {k: float(v) for k, v in metrics.items()}
    out["n_generated"] = 500
    out["elapsed_s"] = round(elapsed, 1)

    with open(outdir / "generation_quality.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"         {elapsed:.1f}s — coverage={metrics['coverage']:.3f}, "
          f"authenticity={metrics['authenticity']:.3f}")


def run_D1_annotation_transfer(model, adata, spec, outdir):
    """D1: Annotation transfer (prototype + kNN)."""
    from mocoo.evaluation.annotation_transfer import evaluate_annotation
    from sklearn.preprocessing import LabelEncoder

    print("    [D1] Annotation transfer...")
    t0 = time.time()
    out = {}

    cell_type_col = spec["cell_type_col"]
    if cell_type_col not in adata.obs.columns:
        # Fallback: try common alternatives
        for alt in ["cell_type", "Clusters", "ClusterName", "clusters_fine",
                     "paul15_clusters", "annotation"]:
            if alt in adata.obs.columns:
                cell_type_col = alt
                break
        else:
            print(f"         ⚠ No cell type column found, skipping D1")
            out["error"] = f"Column '{spec['cell_type_col']}' not found"
            with open(outdir / "annotation_transfer.json", "w") as f:
                json.dump(out, f, indent=2)
            return

    le = LabelEncoder()
    gt_all = le.fit_transform(adata.obs[cell_type_col].values)

    # --- Prototype annotation ---
    try:
        proto_result = model.annotate_cells(method='prototype')
        out["prototype"] = {
            "n_prototypes": int(np.unique(proto_result["labels"]).shape[0]),
            "mean_confidence": float(proto_result["confidence"].mean()),
        }
    except RuntimeError as e:
        out["prototype"] = {"error": str(e)}

    # --- kNN annotation (train→test) ---
    gt_train = gt_all[model.train_idx]
    gt_test = gt_all[model.test_idx]

    ref_latent = model.take_latent(model.X_train, use_qm=True)
    q_latent = model.take_latent(model.X_test, use_qm=True)

    knn_result = model.annotate_cells(
        query_data=model.X_test,
        reference_data=model.X_train,
        reference_labels=gt_train,
        method='knn', k=15,
    )

    eval_scores = evaluate_annotation(knn_result['labels'], gt_test)
    out["knn"] = {
        "accuracy": float(eval_scores["accuracy"]),
        "f1_macro": float(eval_scores["f1_macro"]),
        "f1_weighted": float(eval_scores["f1_weighted"]),
        "mean_confidence": float(knn_result["confidence"].mean()),
        "n_classes": int(len(le.classes_)),
        "class_names": le.classes_.tolist(),
    }

    elapsed = time.time() - t0
    out["elapsed_s"] = round(elapsed, 1)

    with open(outdir / "annotation_transfer.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"         {elapsed:.1f}s — kNN acc={eval_scores['accuracy']:.3f}, "
          f"F1={eval_scores['f1_macro']:.3f}")


def run_F1_uncertainty(model, outdir):
    """F1: Posterior sampling uncertainty."""
    print("    [F1] Uncertainty quantification...")
    t0 = time.time()
    unc = model.get_uncertainty(n_samples=50)
    elapsed = time.time() - t0

    u = unc["uncertainty"]
    assert np.all(np.isfinite(u)), "Non-finite uncertainty"
    assert np.all(u >= 0), "Negative uncertainty"

    out = {
        "mean": float(np.mean(u)),
        "median": float(np.median(u)),
        "std": float(np.std(u)),
        "q5": float(np.quantile(u, 0.05)),
        "q95": float(np.quantile(u, 0.95)),
        "n_cells": int(len(u)),
        "n_samples": 50,
        "elapsed_s": round(elapsed, 1),
    }

    with open(outdir / "uncertainty.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"         {elapsed:.1f}s — mean={u.mean():.4f}, "
          f"q5-q95=[{np.quantile(u, 0.05):.4f}, {np.quantile(u, 0.95):.4f}]")


# ═════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═════════════════════════════════════════════════════════════════════════════

ALL_MODULES = ["A1", "B1", "B2", "A2", "C1", "D1", "F1"]


def run_dataset(dataset_name, data_dir, outdir_base, shared, training, args):
    """Run all downstream modules on one dataset."""
    spec = DATASET_SPECS[dataset_name]
    data_path = os.path.join(data_dir, spec["path"])

    print(f"\n{'=' * 70}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Path: {data_path}")
    print(f"{'=' * 70}")

    adata = load_dataset(data_path, spec["max_cells"], spec["hvg"])
    gene_names = adata.var_names.tolist()

    # Output directory
    outdir = Path(outdir_base) / dataset_name / "downstream"
    outdir.mkdir(parents=True, exist_ok=True)

    # Select modules
    modules = args.modules if args.modules else ALL_MODULES

    # Train or load model
    if args.load_model:
        print(f"\n  Loading model from {args.load_model}")
        cfg = load_config("default")
        full_cfg = get_model_configs(cfg)["Full"]
        params = {**shared, **full_cfg}
        model = MoCoO(adata, **params)
        model.fit(epochs=1, patience=1, val_every=1)  # minimal init to set up splits
        model.load_model(args.load_model)
        # Re-train FM if needed for C1
        if "C1" in modules and not hasattr(model, 'fm_net'):
            print("  ── Training Flow Matching ──")
            model.train_fm(epochs=2000, lr=1e-3)
    else:
        model = train_full_model(adata, shared, training)

    if args.save_model:
        model_path = outdir / "model.pt"
        model.save_model(str(model_path))
        print(f"  Model saved to {model_path}")

    results_summary = {"dataset": dataset_name, "n_cells": adata.n_obs,
                       "n_genes": adata.n_vars, "modules": {}}

    print(f"\n  Running modules: {', '.join(modules)}")
    print(f"  {'─' * 50}")

    # A1 — Gene Importance
    if "A1" in modules:
        jacobian = run_A1_gene_importance(
            model, gene_names, outdir, spec.get("known_markers"),
        )
        results_summary["modules"]["A1"] = "gene_importance.json"

    # B1 — Gene Velocity
    if "B1" in modules:
        run_B1_gene_velocity(model, outdir)
        results_summary["modules"]["B1"] = "gene_velocity.npy"

    # B2 — Branching
    if "B2" in modules:
        run_B2_branching(model, outdir)
        results_summary["modules"]["B2"] = "branching.json"

    # A2 — Differential Expression
    if "A2" in modules:
        run_A2_differential_expression(model, gene_names, outdir)
        results_summary["modules"]["A2"] = "differential_expression.json"

    # C1 — Generation Quality
    if "C1" in modules:
        run_C1_generation_quality(model, outdir)
        results_summary["modules"]["C1"] = "generation_quality.json"

    # D1 — Annotation Transfer
    if "D1" in modules:
        run_D1_annotation_transfer(model, adata, spec, outdir)
        results_summary["modules"]["D1"] = "annotation_transfer.json"

    # F1 — Uncertainty
    if "F1" in modules:
        run_F1_uncertainty(model, outdir)
        results_summary["modules"]["F1"] = "uncertainty.json"

    # Save summary
    results_summary["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(outdir / "summary.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n  {'=' * 50}")
    print(f"  ✓ All modules complete → {outdir}")
    return results_summary


def main():
    parser = argparse.ArgumentParser(
        description="Downstream Analysis Runner — 7 modules on one or all datasets"
    )
    parser.add_argument(
        "--dataset", default="IRALL",
        help="Dataset name or 'all' (default: IRALL)",
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("MOCOO_DATA_DIR", os.path.expanduser("~")),
        help="Base data directory (default: MOCOO_DATA_DIR or ~)",
    )
    parser.add_argument(
        "--outdir", default=None,
        help="Output base directory (default: benchmarks/results)",
    )
    parser.add_argument(
        "--module", nargs="*", dest="modules", default=None,
        help="Subset of modules to run (e.g. A1 B1). Default: all",
    )
    parser.add_argument(
        "--save-model", action="store_true",
        help="Save trained model checkpoint",
    )
    parser.add_argument(
        "--load-model", default=None,
        help="Load model from checkpoint (skip training)",
    )
    args = parser.parse_args()

    # Validate modules
    if args.modules:
        for m in args.modules:
            if m not in ALL_MODULES:
                parser.error(f"Unknown module '{m}'. Choose from: {ALL_MODULES}")

    cfg = load_config("default")
    shared = get_shared_params(cfg)
    training = get_training_params(cfg)

    outdir_base = args.outdir or str(
        Path(__file__).resolve().parent.parent.parent / "results"
    )

    datasets = list(DATASET_SPECS.keys()) if args.dataset == "all" else [args.dataset]

    print(f"\n{'#' * 70}")
    print(f"  MoCoO Downstream Analysis Runner")
    print(f"  Datasets: {datasets}")
    print(f"  Modules: {args.modules or ALL_MODULES}")
    print(f"{'#' * 70}")

    all_results = []
    for ds in datasets:
        if ds not in DATASET_SPECS:
            print(f"  ⚠ Unknown dataset '{ds}', skipping")
            continue
        result = run_dataset(ds, args.data_dir, outdir_base, shared, training, args)
        all_results.append(result)

    # Final summary
    print(f"\n{'#' * 70}")
    print(f"  DONE — {len(all_results)} dataset(s) processed")
    for r in all_results:
        mods = ", ".join(r["modules"].keys())
        print(f"    {r['dataset']}: {r['n_cells']} cells — [{mods}]")
    print(f"{'#' * 70}")


if __name__ == "__main__":
    main()
