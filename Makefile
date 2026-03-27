# ═══════════════════════════════════════════════════════════════════════════
# MoCoO — Top-Level Makefile
# Orchestrates the full pipeline: install → test → benchmark → figures
# Optional local manuscript builds are supported when a paper/ tree exists.
# ═══════════════════════════════════════════════════════════════════════════
#
# Experiment Series:
#   make series1          Beta ablation study (Tables I-V)
#   make series2          Cross-dataset generalization (Tables VIII-XIII)
#   make series3          Multi-seed robustness + significance tests
#   make series4          External baselines (PCA+KMeans)
#
# Usage examples:
#   make help                     # show all targets
#   make all                      # full pipeline
#   make benchmark EPOCHS=200     # override training epochs
#   make figures                  # regenerate all figures
#   make paper                    # build local LaTeX manuscript if present
#
# Override any variable on the command line:
#   make benchmark EPOCHS=300 PATIENCE=50 MAX_CELLS=5000
# ═══════════════════════════════════════════════════════════════════════════

SHELL := /bin/bash

# ── Source optional path configuration ────────────────────────────────────
-include mocoo/configs/paths.env

# ── Overridable variables ─────────────────────────────────────────────────
PYTHON        ?= python
EPOCHS        ?= 400
PATIENCE      ?= 60
MAX_CELLS     ?= 3000
HVG           ?= 3000
DATA_DIR      ?= $(MOCOO_DATA_DIR)
BETA          ?= 1.0
N_SEEDS       ?= 5
DATASETS      ?= IRALL dentate endo paul spinoids

# Beta ablation specific (Series 1)
BETA_EPOCHS   ?= 400
BETA_PATIENCE ?= 60
BETA_VALUES   ?= 0.01 0.1 1.0

# ── Derived paths (project-relative) ─────────────────────────────────────
RESULTS_DIR   ?= $(or $(MOCOO_RESULTS_DIR),benchmarks/results)
FIGURES_DIR   ?= $(or $(MOCOO_FIGURES_DIR),benchmarks/figures)
# Optional local manuscript workspace (not tracked in git)
PAPER_DIR     ?= $(or $(MOCOO_PAPER_DIR),paper)

# ── Series-specific output directories ───────────────────────────────────
SINGLE_DIR    := $(RESULTS_DIR)
BETA_DIR      := $(RESULTS_DIR)/beta_ablation
CROSS_DIR     := $(RESULTS_DIR)/cross_dataset
MULTI_DIR     := $(RESULTS_DIR)/multiseed
BASE_DIR      := $(RESULTS_DIR)/baselines

# ── Script locations ──────────────────────────────────────────────────────
PIPELINE     := benchmarks/scripts/pipeline
PLOTTING     := benchmarks/scripts/plotting
EVALUATION   := benchmarks/scripts/evaluation

# ── Key output files (used for dependency tracking) ───────────────────────
BENCHMARK_NPZ   := $(SINGLE_DIR)/benchmark_data.npz
METRICS_CSV     := $(SINGLE_DIR)/summary_expanded.csv
CROSS_META_CSV  := $(CROSS_DIR)/meta_analysis.csv
MULTISEED_CSV   := $(MULTI_DIR)/multiseed_IRALL.csv

# ═══════════════════════════════════════════════════════════════════════════
# PHONY declarations
# ═══════════════════════════════════════════════════════════════════════════
.PHONY: all help install test lint clean \
        benchmark cross-dataset beta-ablation beta-sweep multiseed baseline \
        series1 series2 series3 series4 \
        metrics significance \
        figures fig2 fig3 fig4 \
        tables paper paper-clean paper-mdpi paper-elsevier paper-all

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
	@echo "  EXPERIMENT SERIES"
	@echo "  ─────────────────────────────────────────────────────────"
	@echo "  make series1          Series 1: Beta ablation (Tables I-V)"
	@echo "                        $(BETA_EPOCHS) epochs, beta=$(BETA_VALUES)"
	@echo "  make series2          Series 2: Cross-dataset (Tables VIII-XIII)"
	@echo "  make series3          Series 3: Multi-seed + significance tests"
	@echo "  make series4          Series 4: External baselines (PCA+KMeans)"
	@echo ""
	@echo "  CORE PIPELINE"
	@echo "  ─────────────────────────────────────────────────────────"
	@echo "  make install          Install MoCoO in editable mode"
	@echo "  make test             Run pytest suite"
	@echo "  make lint             Run black + isort + flake8"
	@echo "  make benchmark        Single-dataset ablation (IRALL, $(EPOCHS) epochs)"
	@echo "  make cross-dataset    5-dataset cross-dataset benchmark"
	@echo "  make beta-ablation    Beta ablation study ($(BETA_EPOCHS) epochs)"
	@echo "  make multiseed        Multi-seed evaluation ($(N_SEEDS) seeds)"
	@echo "  make baseline         PCA+KMeans baseline comparison"
	@echo "  make metrics          Recompute expanded metrics from saved latents"
	@echo "  make significance     Run significance tests on multi-seed results"
	@echo ""
	@echo "  FIGURES"
	@echo "  ─────────────────────────────────────────────────────────"
	@echo "  make figures          Generate all figures"
	@echo "  make fig2             Fig 2: ablation hero (performance + FM)"
	@echo "  make fig3             Fig 3: external benchmarking (19 methods, 5 metrics)"
	@echo "  make fig4             Fig 4: biological validation"
	@echo ""
	@echo "  LOCAL MANUSCRIPT"
	@echo "  ─────────────────────────────────────────────────────────"
	@echo "  make paper            Build a local manuscript if $(PAPER_DIR)/main.tex exists"
	@echo "  make paper-clean      Clean local manuscript build artifacts"
	@echo ""
	@echo "  UTILITY"
	@echo "  ─────────────────────────────────────────────────────────"
	@echo "  make clean            Clean all generated artifacts"
	@echo "  make all              Full pipeline: test -> all series -> figures -> paper"
	@echo ""
	@echo "  OVERRIDABLE VARIABLES (current values)"
	@echo "  ─────────────────────────────────────────────────────────"
	@echo "  EPOCHS        = $(EPOCHS)"
	@echo "  PATIENCE      = $(PATIENCE)"
	@echo "  BETA_EPOCHS   = $(BETA_EPOCHS)"
	@echo "  BETA_PATIENCE = $(BETA_PATIENCE)"
	@echo "  BETA_VALUES   = $(BETA_VALUES)"
	@echo "  MAX_CELLS     = $(MAX_CELLS)"
	@echo "  DATA_DIR      = $(DATA_DIR)"
	@echo "  HVG           = $(HVG)"
	@echo "  N_SEEDS       = $(N_SEEDS)"
	@echo "  RESULTS_DIR   = $(RESULTS_DIR)"
	@echo "  FIGURES_DIR   = $(FIGURES_DIR)"
	@echo ""

# ═══════════════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
all: test benchmark series1 series2 series3 series4 metrics figures paper ## Full pipeline (paper step is optional)
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
	rm -rf $(SINGLE_DIR)
	rm -rf $(MULTI_DIR)
	rm -rf $(BETA_DIR)
	rm -rf $(CROSS_DIR)
	rm -rf $(BASE_DIR)
	rm -rf $(FIGURES_DIR)/*.png $(FIGURES_DIR)/*.pdf
	rm -rf __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "  Done."

# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT SERIES
# ═══════════════════════════════════════════════════════════════════════════

# ── Series 1: Beta Ablation (Paper Tables I-V) ───────────────────────────
series1: beta-ablation ## Series 1: Beta ablation study (Tables I-V)

beta-ablation: ## Beta ablation: all 6 configs x 3 betas (200 epochs)
	@echo ""
	@echo "══════════════════════════════════════════════════════════════"
	@echo "  Series 1: Beta Ablation ($(BETA_EPOCHS) epochs)"
	@echo "  Betas: $(BETA_VALUES)"
	@echo "══════════════════════════════════════════════════════════════"
	@mkdir -p $(BETA_DIR)
	MOCOO_DATA_DIR=$(DATA_DIR) $(PYTHON) $(PIPELINE)/run_beta_ablation.py \
		--data $(DATA_DIR)/LAB/scRL/IRALL.h5ad \
		--epochs $(BETA_EPOCHS) \
		--patience $(BETA_PATIENCE) \
		--max-cells $(MAX_CELLS) \
		--hvg $(HVG) \
		--betas $(BETA_VALUES) \
		--outdir $(BETA_DIR)

# ── Series 2: Cross-Dataset (Paper Tables VIII-XIII) ─────────────────────
series2: cross-dataset ## Series 2: Cross-dataset generalization

cross-dataset: ## Run 5-dataset cross-dataset benchmark
	@echo ""
	@echo "══════════════════════════════════════════════════════════════"
	@echo "  Series 2: Cross-dataset benchmark ($(DATASETS))"
	@echo "  Epochs=$(EPOCHS), MaxCells=$(MAX_CELLS)"
	@echo "══════════════════════════════════════════════════════════════"
	MOCOO_DATA_DIR=$(DATA_DIR) $(PYTHON) $(PIPELINE)/run_cross_dataset.py \
		--datasets $(DATASETS) \
		--epochs $(EPOCHS) \
		--patience $(PATIENCE) \
		--max-cells $(MAX_CELLS) \
		--hvg $(HVG) \
		--outdir $(CROSS_DIR)

# ── Series 3: Multi-Seed Robustness ──────────────────────────────────────
series3: multiseed significance ## Series 3: Statistical robustness

multiseed: ## Run multi-seed evaluation (N_SEEDS seeds)
	@echo ""
	@echo "══════════════════════════════════════════════════════════════"
	@echo "  Series 3a: Multi-seed evaluation ($(N_SEEDS) seeds)"
	@echo "══════════════════════════════════════════════════════════════"
	@mkdir -p $(MULTI_DIR)
	MOCOO_DATA_DIR=$(DATA_DIR) $(PYTHON) $(PIPELINE)/run_multiseed.py \
		--seeds $(N_SEEDS) \
		--datasets IRALL \
		--epochs $(EPOCHS) \
		--max_cells $(MAX_CELLS) \
		--hvg $(HVG) \
		--patience $(PATIENCE) \
		--output_dir $(MULTI_DIR)

significance: multiseed ## Run significance tests on multi-seed results
	@echo ""
	@echo "══════════════════════════════════════════════════════════════"
	@echo "  Series 3b: Significance tests"
	@echo "══════════════════════════════════════════════════════════════"
	$(PYTHON) $(EVALUATION)/significance_tests.py \
		--input $(MULTI_DIR)/multiseed_IRALL.csv \
		--baseline VAE \
		--output_dir $(MULTI_DIR)/significance

# ── Series 4: External Baselines ──────────────────────────────────────────
series4: baseline ## Series 4: External baselines

baseline: ## Run PCA+KMeans baseline comparison
	@echo ""
	@echo "══════════════════════════════════════════════════════════════"
	@echo "  Series 4: PCA + KMeans baseline"
	@echo "══════════════════════════════════════════════════════════════"
	@mkdir -p $(BASE_DIR)
	$(PYTHON) $(EVALUATION)/pca_kmeans_baseline.py \
		--datasets IRALL dentate endo \
		--n_seeds $(N_SEEDS) \
		--output $(BASE_DIR)/pca_kmeans.csv

# ═══════════════════════════════════════════════════════════════════════════
# CORE PIPELINE TARGETS
# ═══════════════════════════════════════════════════════════════════════════

# ── benchmark: single-dataset ablation (IRALL) ───────────────────────────
benchmark: ## Single-dataset ablation (IRALL, configurable epochs)
	@echo ""
	@echo "══════════════════════════════════════════════════════════════"
	@echo "  Benchmark: IRALL ablation ($(EPOCHS) epochs)"
	@echo "══════════════════════════════════════════════════════════════"
	@mkdir -p $(SINGLE_DIR)
	MOCOO_DATA_DIR=$(DATA_DIR) $(PYTHON) $(PIPELINE)/run_benchmark.py \
		--data $(DATA_DIR)/LAB/scRL/IRALL.h5ad \
		--epochs $(EPOCHS) \
		--patience $(PATIENCE) \
		--max-cells $(MAX_CELLS) \
		--hvg $(HVG) \
		--outdir $(SINGLE_DIR)

# ── metrics: recompute expanded metrics from saved latents ────────────────
metrics: benchmark ## Recompute expanded metrics from saved latents
	@echo ""
	@echo "══════════════════════════════════════════════════════════════"
	@echo "  Recomputing expanded metrics"
	@echo "══════════════════════════════════════════════════════════════"
	$(PYTHON) $(EVALUATION)/recompute_metrics.py \
		--resultsdir $(SINGLE_DIR)

# ── Legacy alias (backward compatibility) ─────────────────────────────────
beta-sweep: beta-ablation ## DEPRECATED: use beta-ablation

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE TARGETS
# ═══════════════════════════════════════════════════════════════════════════

# Figure set: fig2 (ablation hero), fig3 (external bench), fig4 (biovalidation)
figures: fig2 fig3 fig4 ## Generate all figures
	@echo ""
	@echo "══════════════════════════════════════════════════════════════"
	@echo "  All figures generated in $(FIGURES_DIR)/"
	@echo "══════════════════════════════════════════════════════════════"

# ── Main Figure Scripts ─────────────────────────────────────────────────
# An optional local Fig 1 manuscript asset can be maintained under $(PAPER_DIR).

fig2: ## Fig 2: ablation hero (absolute performance + FM indicators)
	@echo ""
	@echo "══════════════════════════════════════════════════════════════"
	@echo "  Fig 2: Ablation Hero (absolute performance + FM improvement)"
	@echo "══════════════════════════════════════════════════════════════"
	@mkdir -p $(FIGURES_DIR)
	$(PYTHON) $(PLOTTING)/fig2_ablation_hero.py \
		--resultsdir $(SINGLE_DIR) \
		--outdir $(FIGURES_DIR)

fig3: ## Fig 3: external benchmarking (19 methods, 5 metrics)
	@echo ""
	@echo "══════════════════════════════════════════════════════════════"
	@echo "  Fig 3: External Benchmarking (19 methods, 5 metrics)"
	@echo "══════════════════════════════════════════════════════════════"
	@mkdir -p $(FIGURES_DIR)
	$(PYTHON) $(PLOTTING)/fig3_external_benchmarking.py \
		--resultsdir $(SINGLE_DIR) \
		--outdir $(FIGURES_DIR)

fig4: ## Fig 4: biological validation (annotation transfer + pseudotime + generation)
	@echo ""
	@echo "══════════════════════════════════════════════════════════════"
	@echo "  Fig 4: Biological Validation"
	@echo "══════════════════════════════════════════════════════════════"
	@mkdir -p $(FIGURES_DIR)
	$(PYTHON) $(PLOTTING)/fig4_biovalidation.py \
		--resultsdir $(SINGLE_DIR) \
		--outdir $(FIGURES_DIR)

# ═══════════════════════════════════════════════════════════════════════════
# PAPER TARGETS
# ═══════════════════════════════════════════════════════════════════════════

tables: ## Generate LaTeX tables (tables_fm.tex + tables_perdataset.tex)
	@echo ""
	@echo "══════════════════════════════════════════════════════════════"
	@echo "  Generating LaTeX tables"
	@echo "══════════════════════════════════════════════════════════════"
	@mkdir -p $(PAPER_DIR)
	$(PYTHON) $(PLOTTING)/generate_latex_tables.py --outdir $(PAPER_DIR)

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

paper-mdpi: ## Build MDPI version
	@echo ""
	@echo "══════════════════════════════════════════════════════════════"
	@echo "  Building MDPI paper"
	@echo "══════════════════════════════════════════════════════════════"
	@if [ -f $(PAPER_DIR)/mdpi/main.tex ]; then \
		cd $(PAPER_DIR)/mdpi && latexmk -pdf -interaction=nonstopmode main.tex; \
		echo "  MDPI paper built: $(PAPER_DIR)/mdpi/main.pdf"; \
	else \
		echo "  No $(PAPER_DIR)/mdpi/main.tex found -- skipping MDPI build."; \
	fi

paper-elsevier: ## Build Elsevier version
	@echo ""
	@echo "══════════════════════════════════════════════════════════════"
	@echo "  Building Elsevier paper"
	@echo "══════════════════════════════════════════════════════════════"
	@if [ -f $(PAPER_DIR)/elsevier/main.tex ]; then \
		cd $(PAPER_DIR)/elsevier && latexmk -pdf -interaction=nonstopmode main.tex; \
		echo "  Elsevier paper built: $(PAPER_DIR)/elsevier/main.pdf"; \
	else \
		echo "  No $(PAPER_DIR)/elsevier/main.tex found -- skipping Elsevier build."; \
	fi

paper-all: paper paper-mdpi paper-elsevier ## Build all paper formats
