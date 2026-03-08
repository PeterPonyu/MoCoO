# Validation

## Tests

- `pytest tests/test_mocoo.py` → 87 passed, 6 skipped (real_adata path not set)
  - Verified by euv worktree prior to any code changes.
  - Only `tests/test_mocoo.py` modified (fixture path portability); no logic changed.

## Manuscript

- Table renumbering verified: grep confirms TABLE VIII through TABLE XIII in sequence with no gap.
- No in-text references to "Table IX"–"Table XIV" found that would require updating.

## Code–Manuscript Consistency

- `proto_weight=0.1` confirmed in `benchmarks/scripts/pipeline/run_multiseed.py` lines 67 and 78.
- FIGURES.md now consistent with code and manuscript §IV.B.

## README

- Citation block now shows `author={Fu, Zeyu}` — consistent with manuscript.
