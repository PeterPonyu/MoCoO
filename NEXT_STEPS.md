# MoCoO — Next Steps for Publication Readiness

## Priority 1: Figure Regeneration (Immediate)
All plotting scripts have been updated (bold removal, whitespace reduction, absent-panel hiding).
Regenerate every figure to reflect these changes:
```bash
cd /home/zeyufu/Desktop/MoCoO
python benchmarks/scripts/plotting/plot_quant_comparison.py       # Fig 2
python benchmarks/scripts/plotting/plot_ablation_summary.py       # Fig 3
python benchmarks/scripts/plotting/plot_training_dynamics.py      # Fig 4
python benchmarks/scripts/plotting/plot_composed.py               # Fig 5a
python benchmarks/scripts/plotting/plot_subcategory_heatmap.py    # Fig 5b
python -m benchmarks.scripts.plotting.plot_beta_sensitivity       # Fig 6a
python benchmarks/scripts/plotting/plot_ode_trajectory.py         # Fig 6b (supp)
python -m benchmarks.scripts.plotting.plot_generalization          # Fig 7a
python benchmarks/scripts/plotting/plot_batch_integration.py      # Fig 7b (supp, panels hidden)
python -m benchmarks.scripts.plotting.plot_biological_validation   # Supp bio
```
Then recompile architecture diagram and paper:
```bash
cd paper && pdflatex fig_architecture.tex
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Priority 2: Compute Missing scIB Batch Integration Metrics
**Status**: No batch integration data exists — panels are currently hidden.
**Action**:
1. Run scIB (or scib-metrics) on IRALL with batch key (requires multi-batch dataset or synthetic batch split)
2. Compute: iLISI, bASW, cLISI, graph connectivity, isolated label ASW
3. Store results in `benchmarks/results/cross_dataset/{dataset}/summary_batch.csv`
4. Once data exists, batch integration panels will auto-populate
5. Remove the placeholder text in paper Results §4.7 and replace with real numbers
6. Remove limitation item 5 in Discussion

## Priority 3: Cross-Dataset Evaluation
**Status**: Only IRALL evaluated. Paper references 5 datasets (IRALL, dentate, endo, paul, spinoids) but only for pseudotime markers.
**Action**:
1. Run all 6 configs × 3 betas on dentate, endo, paul, spinoids (same preprocessing: 3000 HVG, normalize)
2. Store in `benchmarks/results/cross_dataset/{dataset}/{config}.json`
3. Create cross-dataset summary table + generalization figure
4. Strengthens external validity claims significantly

## Priority 4: Multi-Seed Robustness Evaluation
**Status**: All results from single seed. No error bars or confidence intervals.
**Action**:
1. Run 5 seeds per config × beta on IRALL
2. Report mean ± std for all metrics in Tables II–IV
3. Add error bars to Fig 2 bar charts and Fig 6 line plots
4. Essential for statistical credibility at a top venue

## Priority 5: External Baseline Comparison
**Status**: Only internal ablation (6 MoCoO configs). No comparison with published methods.
**Action**:
1. Run scVI, scVelo, Harmony, PHATE, DCA, Seurat v5 on IRALL with same preprocessing
2. Add PCA+KMeans baseline (script exists: `benchmarks/scripts/evaluation/pca_kmeans_baseline.py`)
3. Add rows to Tables II–IV or a new table
4. Critical for publication — reviewers will expect external baselines

## Priority 6: Table VII Redundancy Resolution
**Status**: Table VII (`tab:beta_full`) is fully recoverable from Tables II–IV (Full row).
**Decision**: If space is tight, cut Table VII and reference the Full rows in Tables II–IV directly. Saves ~½ column.

## Priority 7: Supplementary Material Organization
**Action**:
- Move per-dataset marker tables (IX–XIII) to supplementary if space is tight
- Organize supplementary into: A (architecture details), B (full metric tables), C (biological validation), D (batch integration), E (training dynamics)
- Ensure all supplementary figures are referenced in main text

## Priority 8: Final Proofreading
- Run `chktex main.tex` for LaTeX warnings
- Verify all cross-references resolve (`\ref`, `\cite`)
- Check figure numbering matches text references
- Verify consistent notation (β vs $\beta$, scRNA-seq hyphenation, etc.)
- Grammar/spell check with Grammarly or LanguageTool
