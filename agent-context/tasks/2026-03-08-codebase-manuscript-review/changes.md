# Changes Applied

## Files Modified

| File | Change |
|------|--------|
| `MoCoO_JBHI_Article.md` | Table renumbering: TABLE IX→VIII, X→IX, XI→X, XII→XI, XIII→XII, XIV→XIII |
| `README.md` | Citation author: `Ponyu, Peter` → `Fu, Zeyu` |
| `benchmarks/FIGURES.md` | proto_weight `12 / 0.05` → `12 / 0.1`; per-component ablation table fixed (malformed `\n` → proper markdown) |
| `pyproject.toml` | `authors` and `maintainers`: `Peter Ponyu` → `Fu, Zeyu` (**independent review — missed by all 8 agents**) |
| `tests/test_mocoo.py` | real_adata fixture: hardcoded path → `os.environ.get('MOCOO_TEST_DATA', ...)` |

## Files Created

- `agent-context/current-focus.md`
- `agent-context/tasks/2026-03-08-codebase-manuscript-review/target.md`
- `agent-context/tasks/2026-03-08-codebase-manuscript-review/REVIEW-NOTE.md`
- `agent-context/tasks/2026-03-08-codebase-manuscript-review/changes.md`
- `agent-context/tasks/2026-03-08-codebase-manuscript-review/validation.md`
- `agent-context/tasks/2026-03-08-codebase-manuscript-review/handoff.md`

## Rejected Changes

- Inserting new TABLE VIII with specific marker-gene content (uiw worktree proposal): content cannot be verified from raw benchmark data; renumbering is correct and sufficient.
