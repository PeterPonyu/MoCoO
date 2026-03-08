# MoCoO Pipeline — Reproducibility Guide

Step-by-step commands to reproduce article results and generate figures.

---

## Prerequisites

- Python 3.8+
- `pip install -e ".[benchmark]"` (or `.[dev]`)
- Datasets: IRALL, dentate, endo, paul, spinoids (h5ad format)

### Environment Variables (Optional)

| Variable | Purpose |
|----------|---------|
| `MOCOO_DATA_DIR` | Base directory for dataset paths (default: `/home/zeyufu`) |

---

## 1. Source Data

Datasets are loaded via `DatasetRegistry` in `benchmarks/scripts/pipeline/dataset_registry.py`.

To use custom paths, set:
```bash
export MOCOO_DATA_DIR=/path/to/your/data
```

Expected layout under `MOCOO_DATA_DIR`:
- `LAB/scRL/IRALL.h5ad`
- `vGAE_LAB/data/dentate.h5ad`, `endo.h5ad`
- `LAB/data/paul.h5ad`, `spinoids.h5ad`

---

## 2. Run Benchmarks

### Option A: Single-dataset ablation (IRALL, beta sweep)

Produces `benchmark_data.npz` + per-config JSON (required for plotting scripts):

```bash
# Beta = 1.0
python benchmarks/scripts/pipeline/run_benchmark.py \
  --data /path/to/IRALL.h5ad \
  --max-cells 1000 --hvg 2000 \
  --epochs 50 --patience 15 \
  --beta 1.0 \
  --outdir benchmarks/results/beta1.0

# Repeat with --beta 0.1 and --beta 0.01
```

### Option B: Cross-dataset + LSE/DRE + pseudotime validation

Does **not** produce `benchmark_data.npz`; outputs per-dataset JSON + summary CSV:

```bash
python benchmarks/scripts/pipeline/run_cross_and_validate.py \
  --datasets IRALL dentate endo paul spinoids \
  --epochs 150 --max-cells 3000
```

### Option C: Multi-seed + expanded metrics (LSE, DRE, DREX, LSEX)

```bash
python benchmarks/scripts/pipeline/run_multiseed.py \
  --datasets IRALL --seeds 42 123 456 \
  --epochs 150
```

---

## 3. Generate Figures

All plotting scripts use project-relative default paths. Override with CLI arguments:

```bash
python benchmarks/scripts/plotting/plot_ablation_summary.py \
  --resultsdir benchmarks/results/beta1.0 \
  --outdir benchmarks/figures
```

Other plot scripts (see `benchmarks/FIGURES.md` for details):
- `plot_composed.py` — composed benchmark figure
- `plot_biological_validation.py` — biovalidation panels
- `plot_training_dynamics.py` — training curves
- `plot_quant_comparison.py` — quantitative comparison
- `plot_ode_trajectory.py` — ODE pseudotime analysis
- `plot_batch_integration.py` — batch integration

---

## 4. Article–Code Mapping

| Article Section | Source |
|-----------------|--------|
| Tables I–III (beta sweep) | `run_benchmark.py` with `--beta 1.0`, `0.1`, `0.01` |
| Tables IV–VII (component effects, synergy) | Derived from Tables I–III |
| Tables VIII–XIII (pseudotime–marker) | `run_cross_and_validate.py` pseudotime validation |
| Fig. 1 (composed benchmark) | `plot_composed.py` |

---

## 5. Config Consistency Note

The article states `α_vae=0.8`, `α_ode=0.2`. Ensure benchmark scripts use matching values:
- `run_benchmark.py`: `vae_reg=0.8`, `ode_reg=0.2` (matches article)
- `run_cross_and_validate.py`: currently uses `vae_reg=0.6`, `ode_reg=0.4` — consider aligning for article reproducibility
