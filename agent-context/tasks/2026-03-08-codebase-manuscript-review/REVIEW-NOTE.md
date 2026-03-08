# MoCoO Codebase and Manuscript Review

**Date:** 2026-03-08
**Scope:** Codebase, manuscript (`MoCoO_JBHI_Article.md`), benchmarks, tests, docs
**Agents involved:** 8 parallel Cursor agents (euv, kzd, mvj, ngx, uiw, vlc, wrm, zyr); synthesized and applied by integrating agent

---

## Executive Summary

The MoCoO package, manuscript, and benchmark pipeline are well-aligned. Code implements the described architecture; benchmark results match manuscript tables; tests cover all configurations. Four high-confidence fixes were applied (see §Changes Applied). Several improvements remain for pre-submission.

---

## 1. Codebase Assessment

### Strengths

- **Modular design**: Clear separation of VAE, ODE, MoCo, and prototype components.
- **Comprehensive tests**: 20 architecture configs (use_ode × use_moco × loss_mode), ODE-specific outputs, MoCo losses, real-data paths.
- **Benchmark pipeline**: 6-config ablation, beta sweep (1.0, 0.1, 0.01), LSE/DRE/DREX/LSEX metrics, pseudotime–marker correlation.
- **Result alignment**: `benchmarks/results/beta*/summary.csv` values match manuscript Tables I–III (rounded).

### Code Improvements

| Priority | Improvement | Location | Status |
|----------|-------------|----------|--------|
| **High** | Portable real-data test path | `tests/test_mocoo.py:51` | **APPLIED** — `MOCOO_TEST_DATA` env var with fallback |
| **High** | Citation author mismatch | `README.md:226` | **APPLIED** — `Fu, Zeyu` |
| **Medium** | Add `get_test_latent` to README API | `README.md` | Pending — method exists but undocumented in README |
| **Low** | `promptsforcalude.md` at root | Root | Pending — consider archiving if obsolete |

### Implementation–Manuscript Consistency

- **Loss formulation**: Matches `model.py` `total_loss` composition (recon, irecon, qz_div, vel_loss, kl_div, dip/tc/mmd, moco, proto). ✓
- **ODE stop-gradient**: Code uses `q_z.detach()` in `qz_div` and `vel_loss` — matches manuscript. ✓
- **Hyperparameters**: `run_benchmark.py` CONFIGS match manuscript §IV.B (vae_reg=0.8, ode_reg=0.2, moco_weight=0.6, proto_weight=0.1, n_prototypes=12). ✓

---

## 2. Manuscript Assessment

### Strengths

- Clear structure: Introduction, Related Work, Method, Experiments, Results, Discussion, Conclusion.
- Ablation design well-motivated; beta sweep and synergy analysis rigorous.
- Pseudotime–marker tables (VIII–XIII) are biologically interpretable and consistent with FIGURES.md.

### Manuscript Improvements

| Priority | Improvement | Location | Status |
|----------|-------------|----------|--------|
| **High** | Table numbering gap (VII → IX) | §V tables | **APPLIED** — Tables IX–XIV renumbered to VIII–XIII |
| **Medium** | References [16]–[19] completeness | References | Pending — "et al." placeholders; fill full citations |
| **Medium** | PanODE ref [9] | References | Pending — "manuscript in preparation"; verify consistency with MoCoO positioning |
| **Low** | Abstract ρ rounding | Abstract | Pending — *Hbb-bs* ρ=0.28 in abstract vs 0.275 in table; standardize |
| **Low** | Acknowledgments placeholder | §VII | Pending — fill before submission |

### Result–Claim Alignment

- Tables I–III (beta sweep): match `benchmarks/results/beta*/summary*.csv`. ✓
- Tables IV–V (component effects, synergy): correctly derived from Tables I–III. ✓
- Table VI (Full model across beta): consistent. ✓
- Tables VIII–XIII (pseudotime–marker): align with `benchmarks/FIGURES.md` biovalidation section. ✓

---

## 3. Benchmark and Figure Consistency

| Item | Status |
|------|--------|
| FIGURES.md per-component ablation table (malformed `\n`) | **APPLIED** — fixed to proper markdown |
| FIGURES.md `proto_weight 0.05` vs code/manuscript `0.1` | **APPLIED** — corrected to `0.1` |
| FIGURES.md 150 epochs / 3,000 cells vs manuscript beta sweep 50 epochs / 1,000 cells | OK — different experiments (full-scale vs fast-iteration); both valid |
| DRE/LSE aggregation (UMAP vs tSNE variants) | Pending — document in Methods how subscores are combined |

---

## 4. Changes Applied This Session

1. `MoCoO_JBHI_Article.md` — Table renumbering: TABLE IX→VIII, X→IX, XI→X, XII→XI, XIII→XII, XIV→XIII
2. `README.md` — Citation author: `Ponyu, Peter` → `Fu, Zeyu`
3. `benchmarks/FIGURES.md` — proto_weight: `12 / 0.05` → `12 / 0.1`; per-component ablation table: single-line with `\n` → proper markdown
4. `tests/test_mocoo.py` — real_adata fixture path: hardcode → `os.environ.get('MOCOO_TEST_DATA', ...)`

**Rejected (with reason):**
- Inserting fabricated TABLE VIII content (uiw worktree): The specific marker-gene table proposed by one agent cannot be verified from available raw data. Renumbering is the correct and sufficient fix.

---

## 5. Next Steps (Prioritized)

### Immediate (Pre-Submission)

1. Add **Acknowledgments** — replace `*(To be added)*` in §VII.
2. Complete **references [16]–[19]** (scAGCL, scGPCL, scDiff, VeloVAE) with full author lists.
3. Verify **PanODE ref [9]** relationship and update if manuscript has been published.

### Short-Term (Post-Review)

4. **Multi-seed evaluation** — `run_multiseed.py` exists; run and add std devs to manuscript tables.
5. **External baselines** — add scVI / Harmony comparison (flagged in Limitations).
6. **Document DRE/LSE aggregation** — clarify UMAP vs tSNE variant selection in Methods.

### Medium-Term (Package Maturity)

7. **scIB batch integration** — ensure pipeline computes iLISI, bASW, cLISI.
8. **Adaptive prototypes** — current P=12 vs K=19 on paul dataset; consider per-dataset tuning.

---

## 6. Verification

- Tests: `pytest tests/test_mocoo.py` → 87 passed, 6 skipped (real-data path absent) — confirmed by euv worktree. No code logic changed.
- Manuscript table renumbering: grep confirms no in-text forward references to "Table IX"–"Table XIV" requiring updates.
- README citation: verified against manuscript author field.
- FIGURES.md proto_weight: verified against `benchmarks/scripts/pipeline/run_multiseed.py` (proto_weight=0.1 on lines 67, 78).
