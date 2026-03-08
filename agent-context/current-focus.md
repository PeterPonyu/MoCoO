# Current Focus

**Last updated:** 2026-03-08

## Recently Completed

- **Parallel codebase and manuscript review** (8 Cursor agents, synthesized and applied 2026-03-08):
  - Manuscript table numbering fixed: Tables IX–XIV renumbered to VIII–XIII (gap closed)
  - README citation author corrected: `Ponyu, Peter` → `Fu, Zeyu`
  - `benchmarks/FIGURES.md` proto_weight corrected: `0.05` → `0.1` (matches code and manuscript)
  - `benchmarks/FIGURES.md` per-component ablation table: malformed single-line fixed to proper markdown
  - `tests/test_mocoo.py` real-data path: hardcode → `MOCOO_TEST_DATA` env var with fallback

## Top Next Steps

1. **Add Acknowledgments** in manuscript (current placeholder: *(To be added)*).
2. **Complete references [16]–[19]**: replace "et al." placeholders with full citations (scAGCL, scGPCL, scDiff, VeloVAE).
3. **Multi-seed evaluation**: `run_multiseed.py` exists — run and add std devs to manuscript tables.
4. **External baselines**: add scVI / Harmony comparison (noted in manuscript Limitations).
5. **Document DRE/LSE aggregation**: clarify in Methods how UMAP vs tSNE variants are combined.

## Latest Review

See `agent-context/tasks/2026-03-08-codebase-manuscript-review/REVIEW-NOTE.md` for the full assessment.
