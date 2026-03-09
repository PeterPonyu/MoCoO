# ═══════════════════════════════════════════════════════════════════════════
# MoCoO — Top-Level Makefile
# Orchestrates the full pipeline: install → test → benchmark → figures → paper
# ═══════════════════════════════════════════════════════════════════════════
#
# Usage examples:
#   make help                     # show all targets
#   make all                      # full pipeline
#   make benchmark EPOCHS=200     # override training epochs
#   make figures                  # regenerate all figures
#   make paper                    # build LaTeX paper
#
# Override any variable on the command line:
#   make benchmark EPOCHS=300 PATIENCE=50 MAX_CELLS=5000
# ═══════════════════════════════════════════════════════════════════════════

SHELL := /bin/bash

# ── Source optional path configuration ────────────────────────────────────
-include mocoo/configs/paths.env

# ── Overridable variables ─────────────────────────────────────────────────
PYTHON      ?= python
EPOCHS      ?= 150
PATIENCE    ?= 30
MAX_CELLS   ?= 3000
HVG         ?= 3000
DATA_DIR    ?= $(MOCOO_DATA_DIR)
BETA        ?= 1.0
N_SEEDS     ?= 5
DATASETS    ?= IRALL dentate endo paul spinoids

# ── Derived paths (project-relative) ─────────────────────────────────────
RESULTS_DIR ?= $(or $(MOCOO_RESULTS_DIR),benchmarks/results)
FIGURES_DIR ?= $(or $(MOCOO_FIGURES_DIR),benchmarks/figures)
PAPER_DIR   ?= $(or $(MOCOO_PAPER_DIR),paper)

# ── Script locations ──────────────────────────────────────────────────────
PIPELINE     := benchmarks/scripts/pipeline
PLOTTING     := benchmarks/scripts/plotting
EVALUATION   := benchmarks/scripts/evaluation

# ── Key output files (used for dependency tracking) ───────────────────────
BENCHMARK_NPZ   := $(RESULTS_DIR)/dataset_default/benchmark_data.npz
METRICS_CSV     := $(RESULTS_DIR)/dataset_default/summary_expanded.csv
CROSS_META_CSV  := $(RESULTS_DIR)/meta_analysis.csv
MULTISEED_CSV   := $(RESULTS_DIR)/multiseed/multiseed_IRALL.csv

# ═══════════════════════════════════════════════════════════════════════════
# PHONY declarations
# ═══════════════════════════════════════════════════════════════════════════
.PHONY: all help install test lint clean \
        benchmark cross-dataset beta-sweep multiseed baseline \
        metrics significance \
        figures fig-ablation fig-comparison fig-composed fig-dynamics \
        fig-batch fig-trajectory fig-biovalidation \
        paper paper-clean

# ═══════════════════════════════════════════════════════════════════════════
# DEFAULT / HELP
# ═══════════════════════════════════════════════════════════════════════════
.DEFAULT_GOAL := help

help: ## Show all available targets
	@echo ""
	@echo "══════════════════════════════════════════════════════════════"
	@echo "  MoCoO Pipeline Makefile"
	@echo "══════════════════════════════════════════════════════════════"
	@echo ""
	@echo "  CORE PIPELINE"
	@echo "  ─────────────────────────────────────────────────────────"
	@echo "  make install          Install MoCoO in editable mode"
	@echo "  make test             Run pytest suite"
	@echo "  make lint             Run black + isort + flake8"
	@echo "  make benchmark        Single-dataset ablation (IRALL, $(EPOCHS) epochs)"
	@echo "  make cross-dataset    5-dataset cross-dataset benchmark"
	@echo "  make beta-sweep       Beta sensitivity (50 epochs, beta=0.01,0.1,1.0)"
	@echo "  make multiseed        Multi-seed evaluation ($(N_SEEDS) seeds)"
	@echo "  make baseline         PCA+KMeans baseline comparison"
	@echo "  make metrics          Recompute expanded metrics from saved latents"
	@echo "  make significance     Run significance tests on multi-seed results"
	@echo ""
	@echo "  FIGURES"
	@echo "  ─────────────────────────────────────────────────────────"
	@echo "  make figures          Generate all figures"
	@echo "  make fig-ablation     Fig 3: ablation summary"
	@echo "  make fig-comparison   Fig 2: quantitative comparison"
	@echo "  make fig-composed     Fig 5: composed multi-panel"
	@echo "  make fig-dynamics     Fig 4: training dynamics"
	@echo "  make fig-batch        Supplemental: batch integration"
	@echo "  make fig-trajectory   Supplemental: ODE trajectory"
	@echo "  make fig-biovalidation Supplemental: biological validation"
	@echo ""
	@echo "  PAPER"
	@echo "  ─────────────────────────────────────────────────────────"
	@echo "  make paper            Build the LaTeX paper (paper/main.tex)"
	@echo "  make paper-clean      Clean paper build artifacts"
	@echo ""
	@echo "  UTILITY"
	@echo "  ─────────────────────────────────────────────────────────"
	@echo "  make clean            Clean all generated artifacts"
	@echo "  make all              Full pipeline: test -> benchmark -> figures -> paper"
	@echo ""
	@echo "  OVERRIDABLE VARIABLES (current values)"
	@echo "  ─────────────────────────────────────────────────────────"
	@echo "  EPOCHS      = $(EPOCHS)"
	@echo "  PATIENCE    = $(PATIENCE)"
	@echo "  MAX_CELLS   = $(MAX_CELLS)"
	@echo "  DATA_DIR    = $(DATA_DIR)"
	@echo "  HVG         = $(HVG)"
	@echo "  N_SEEDS     = $(N_SEEDS)"
	@echo "  RESULTS_DIR = $(RESULTS_DIR)"
	@echo "  FIGURES_DIR = $(FIGURES_DIR)"
	@echo ""

# ═══════════════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
all: test benchmark metrics figures paper ## Full pipeline: test -> benchmark -> figures -> paper
	@echo ""
	@echo "══════════════════════════════════════════════════════════════"
	@echo "  Pipeline complete."
	@echo "══════════════════════════════════════════════════════════════"

# ═══════════════════════════════════════════════════════════════════════════
# UTILITY TARGETS
# ═══════════════════════════════════════════════════════════════════════════
install: ## Install MoCoO in editable mode with dev+benchmark extras
	@echo ""
	@echo "══════════════════════════════════════════════════════════════"
	@echo "  Installing MoCoO [editable, dev+benchmark]"
	@echo "══════════════════════════════════════════════════════════════"
	$(PYTHON) -m pip install -e ".[dev,benchmark]"

test: ## Run pytest suite
	@echo ""
	@echo "══════════════════════════════════════════════════════════════"
	@echo "  Running tests"
	@echo "══════════════════════════════════════════════════════════════"
	$(PYTHON) -m pytest tests/ -v

lint: ## Run black + isort + flake8
	@echo ""
	@echo "══════════════════════════════════════════════════════════════"
	@echo "  Running linters (black, isort, flake8)"
	@echo "══════════════════════════════════════════════════════════════"
	$(PYTHON) -m black --check mocoo/ benchmarks/ tests/
	$(PYTHON) -m isort --check-only mocoo/ benchmarks/ tests/
	$(PYTHON) -m flake8 mocoo/ benchmarks/ tests/ --max-line-length 120 --ignore E501,W503,E203

clean: paper-clean ## Clean all generated artifacts
	@echo ""
	@echo "══════════════════════════════════════════════════════════════"
	@echo "  Cleaning generated artifacts"
	@echo "══════════════════════════════════════════════════════════════"
	rm -rf $(RESULTS_DIR)/dataset_default
	rm -rf $(RESULTS_DIR)/multiseed
	rm -rf $(RESULTS_DIR)/beta*
	rm -f  $(RESULTS_DIR)/meta_analysis.csv
	rm -f  $(RESULTS_DIR)/pca_kmeans_baseline.csv
	rm -rf $(FIGURES_DIR)/*.png $(FIGURES_DIR)/*.pdf
	rm -rf __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "  Done."

# ═══════════════════════════════════════════════════════════════════════════
# CORE PIPELINE TARGETS
# ═══════════════════════════════════════════════════════════════════════════

# ── benchmark: single-dataset ablation (IRALL) ───────────────────────────
# Produces benchmark_data.npz + per-config JSON + summary CSV.
benchmark: ## Run single-dataset ablation (IRALL, configurable epochs)
	@echo ""
	@echo "══════════════════════════════════════════════════════════════"
	@echo "  Benchmark: IRALL ablation ($(EPOCHS) epochs, beta=$(BETA))"
	@echo "══════════════════════════════════════════════════════════════"
	@mkdir -p $(RESULTS_DIR)/dataset_default
	MOCOO_DATA_DIR=$(DATA_DIR) $(PYTHON) $(PIPELINE)/run_benchmark.py \
		--data $(DATA_DIR)/LAB/scRL/IRALL.h5ad \
		--epochs $(EPOCHS) \
		--patience $(PATIENCE) \
		--max-cells $(MAX_CELLS) \
		--hvg $(HVG) \
		--beta $(BETA) \
		--outdir $(RESULTS_DIR)/dataset_default

# ── cross-dataset: run across all 5 datasets ─────────────────────────────
cross-dataset: ## Run 5-dataset cross-dataset benchmark
	@echo ""
	@echo "══════════════════════════════════════════════════════════════"
	@echo "  Cross-dataset benchmark ($(DATASETS))"
	@echo "  Epochs=$(EPOCHS), MaxCells=$(MAX_CELLS)"
	@echo "══════════════════════════════════════════════════════════════"
	MOCOO_DATA_DIR=$(DATA_DIR) $(PYTHON) $(PIPELINE)/run_cross_dataset.py \
		--datasets $(DATASETS) \
		--epochs $(EPOCHS) \
		--patience $(PATIENCE) \
		--max-cells $(MAX_CELLS) \
		--hvg $(HVG) \
		--outdir $(RESULTS_DIR)

# ── beta-sweep: sensitivity analysis across beta values ───────────────────
beta-sweep: ## Run beta sensitivity (50 epochs, beta=0.01,0.1,1.0)
	@echo ""
	@echo "══════════════════════════════════════════════════════════════"
	@echo "  Beta sweep: beta = 0.01, 0.1, 1.0 (50 epochs each)"
	@echo "══════════════════════════════════════════════════════════════"
	@for BETA_VAL in 0.01 0.1 1.0; do \
		echo ""; \
		echo "  ── beta = $$BETA_VAL ──"; \
		mkdir -p $(RESULTS_DIR)/beta$$BETA_VAL; \
		MOCOO_DATA_DIR=$(DATA_DIR) $(PYTHON) $(PIPELINE)/run_benchmark.py \
			--data $(DATA_DIR)/LAB/scRL/IRALL.h5ad \
			--epochs 50 \
			--patience $(PATIENCE) \
			--max-cells $(MAX_CELLS) \
			--hvg $(HVG) \
			--beta $$BETA_VAL \
			--outdir $(RESULTS_DIR)/beta$$BETA_VAL; \
	done

# ── multiseed: multi-seed evaluation for statistical robustness ──────────
multiseed: ## Run multi-seed evaluation (N_SEEDS seeds)
	@echo ""
	@echo "══════════════════════════════════════════════════════════════"
	@echo "  Multi-seed evaluation ($(N_SEEDS) seeds)"
	@echo "══════════════════════════════════════════════════════════════"
	@mkdir -p $(RESULTS_DIR)/multiseed
	MOCOO_DATA_DIR=$(DATA_DIR) $(PYTHON) $(PIPELINE)/run_multiseed.py \
		--seeds $(N_SEEDS) \
		--datasets IRALL \
		--epochs $(EPOCHS) \
		--max_cells $(MAX_CELLS) \
		--hvg $(HVG) \
		--patience $(PATIENCE) \
		--output_dir $(RESULTS_DIR)/multiseed

# ── baseline: PCA + KMeans comparison ─────────────────────────────────────
baseline: ## Run PCA+KMeans baseline comparison
	@echo ""
	@echo "══════════════════════════════════════════════════════════════"
	@echo "  PCA + KMeans baseline"
	@echo "══════════════════════════════════════════════════════════════"
	$(PYTHON) $(EVALUATION)/pca_kmeans_baseline.py \
		--datasets IRALL dentate endo \
		--n_seeds $(N_SEEDS) \
		--output $(RESULTS_DIR)/pca_kmeans_baseline.csv

# ── metrics: recompute expanded metrics from saved latents ────────────────
# Depends on benchmark having produced benchmark_data.npz.
metrics: benchmark ## Recompute expanded metrics from saved latents
	@echo ""
	@echo "══════════════════════════════════════════════════════════════"
	@echo "  Recomputing expanded metrics"
	@echo "══════════════════════════════════════════════════════════════"
	$(PYTHON) $(EVALUATION)/recompute_metrics.py \
		--resultsdir $(RESULTS_DIR)/dataset_default

# ── significance: statistical tests on multi-seed results ─────────────────
# Depends on multiseed having produced the CSV.
significance: multiseed ## Run significance tests on multi-seed results
	@echo ""
	@echo "══════════════════════════════════════════════════════════════"
	@echo "  Significance tests"
	@echo "══════════════════════════════════════════════════════════════"
	$(PYTHON) $(EVALUATION)/significance_tests.py \
		--input $(RESULTS_DIR)/multiseed/multiseed_IRALL.csv \
		--baseline VAE \
		--output_dir $(RESULTS_DIR)/multiseed

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE TARGETS
# ═══════════════════════════════════════════════════════════════════════════

# All figures depend on benchmark results existing.
figures: fig-ablation fig-comparison fig-composed fig-dynamics fig-batch fig-trajectory fig-biovalidation ## Generate all figures
	@echo ""
	@echo "══════════════════════════════════════════════════════════════"
	@echo "  All figures generated in $(FIGURES_DIR)/"
	@echo "══════════════════════════════════════════════════════════════"

fig-ablation: ## Fig 3: ablation summary (synergy, waterfall, heatmap)
	@echo ""
	@echo "══════════════════════════════════════════════════════════════"
	@echo "  Figure: Ablation summary"
	@echo "══════════════════════════════════════════════════════════════"
	@mkdir -p $(FIGURES_DIR)
	$(PYTHON) $(PLOTTING)/plot_ablation_summary.py \
		--resultsdir $(RESULTS_DIR)/dataset_default \
		--outdir $(FIGURES_DIR)

fig-comparison: ## Fig 2: quantitative latent space comparison
	@echo ""
	@echo "══════════════════════════════════════════════════════════════"
	@echo "  Figure: Quantitative comparison"
	@echo "══════════════════════════════════════════════════════════════"
	@mkdir -p $(FIGURES_DIR)
	$(PYTHON) $(PLOTTING)/plot_quant_comparison.py \
		--resultsdir $(RESULTS_DIR)/dataset_default \
		--outdir $(FIGURES_DIR)

fig-composed: ## Fig 5: composed multi-panel benchmark figure
	@echo ""
	@echo "══════════════════════════════════════════════════════════════"
	@echo "  Figure: Composed benchmark"
	@echo "══════════════════════════════════════════════════════════════"
	@mkdir -p $(FIGURES_DIR)
	$(PYTHON) $(PLOTTING)/plot_composed.py \
		--resultsdir $(RESULTS_DIR)/dataset_default \
		--outdir $(FIGURES_DIR)

fig-dynamics: ## Fig 4: training dynamics and convergence
	@echo ""
	@echo "══════════════════════════════════════════════════════════════"
	@echo "  Figure: Training dynamics"
	@echo "══════════════════════════════════════════════════════════════"
	@mkdir -p $(FIGURES_DIR)
	$(PYTHON) $(PLOTTING)/plot_training_dynamics.py \
		--resultsdir $(RESULTS_DIR)/dataset_default \
		--outdir $(FIGURES_DIR)

fig-batch: ## Supplemental: batch integration and cross-dataset generalization
	@echo ""
	@echo "══════════════════════════════════════════════════════════════"
	@echo "  Figure: Batch integration"
	@echo "══════════════════════════════════════════════════════════════"
	@mkdir -p $(FIGURES_DIR)
	$(PYTHON) $(PLOTTING)/plot_batch_integration.py \
		--resultsdir $(RESULTS_DIR) \
		--outdir $(FIGURES_DIR)

fig-trajectory: ## Supplemental: ODE pseudotime trajectory analysis
	@echo ""
	@echo "══════════════════════════════════════════════════════════════"
	@echo "  Figure: ODE trajectory"
	@echo "══════════════════════════════════════════════════════════════"
	@mkdir -p $(FIGURES_DIR)
	MOCOO_DATA_DIR=$(DATA_DIR) $(PYTHON) $(PLOTTING)/plot_ode_trajectory.py \
		--resultsdir $(RESULTS_DIR)/dataset_default \
		--outdir $(FIGURES_DIR) \
		--data $(DATA_DIR)/LAB/scRL/IRALL.h5ad

fig-biovalidation: ## Supplemental: biological validation (perturbation, gene expression)
	@echo ""
	@echo "══════════════════════════════════════════════════════════════"
	@echo "  Figure: Biological validation"
	@echo "══════════════════════════════════════════════════════════════"
	@mkdir -p $(FIGURES_DIR)
	MOCOO_DATA_DIR=$(DATA_DIR) $(PYTHON) $(PLOTTING)/plot_biological_validation.py \
		--resultsdir $(RESULTS_DIR)/dataset_default \
		--outdir $(FIGURES_DIR) \
		--data $(DATA_DIR)/LAB/scRL/IRALL.h5ad

# ═══════════════════════════════════════════════════════════════════════════
# PAPER TARGETS
# ═══════════════════════════════════════════════════════════════════════════

paper: ## Build the LaTeX paper (requires paper/main.tex)
	@echo ""
	@echo "══════════════════════════════════════════════════════════════"
	@echo "  Building paper"
	@echo "══════════════════════════════════════════════════════════════"
	@if [ -f $(PAPER_DIR)/main.tex ]; then \
		cd $(PAPER_DIR) && \
		if command -v latexmk >/dev/null 2>&1; then \
			latexmk -pdf -interaction=nonstopmode main.tex; \
		elif command -v pdflatex >/dev/null 2>&1; then \
			pdflatex -interaction=nonstopmode main.tex && \
			bibtex main 2>/dev/null || true && \
			pdflatex -interaction=nonstopmode main.tex && \
			pdflatex -interaction=nonstopmode main.tex; \
		else \
			echo "  ERROR: Neither latexmk nor pdflatex found."; \
			echo "  Install a TeX distribution (e.g., texlive-full)."; \
			exit 1; \
		fi; \
		echo "  Paper built: $(PAPER_DIR)/main.pdf"; \
	else \
		echo "  No $(PAPER_DIR)/main.tex found -- skipping paper build."; \
		echo "  Place your LaTeX source in $(PAPER_DIR)/main.tex to enable this target."; \
	fi

paper-clean: ## Clean paper build artifacts
	@echo ""
	@echo "══════════════════════════════════════════════════════════════"
	@echo "  Cleaning paper build artifacts"
	@echo "══════════════════════════════════════════════════════════════"
	@if [ -d $(PAPER_DIR) ]; then \
		cd $(PAPER_DIR) && \
		rm -f *.aux *.bbl *.blg *.fdb_latexmk *.fls *.log *.out \
		      *.synctex.gz *.toc *.lof *.lot *.nav *.snm *.vrb \
		      main.pdf 2>/dev/null; \
		echo "  Done."; \
	fi
