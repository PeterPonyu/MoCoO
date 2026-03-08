---
description: "Audit, consolidate, and apply outputs from Cursor parallel agent worktrees into the current workspace. Use when Cursor spawned multiple agent worktrees with code edits, article drafts, documentation, plans, configs, analyses, figures, or audit reports that need to be reviewed, validated, implemented, and cleaned up."
name: "Apply Cursor Agent Outputs"
argument-hint: "Optional: focus area (e.g. 'only paper edits', 'code and docs only', 'skip figures')"
agent: "agent"
---

# Audit, Consolidate, and Apply Cursor Agent Outputs

Cursor spawns parallel agents that each work in an isolated git worktree under `~/.cursor/worktrees/<repo>/`. Each worktree may contain code patches, article/manuscript revisions, audit reports, implementation plans, checklists, configs, notebooks, figures, datasets metadata, or new documentation. Your job is to:

1. **Discover** every worktree and read all their outputs
2. **Consolidate** overlapping plans and deduplicate redundant content
3. **Audit** each proposed change for correctness, safety, and reasonableness before applying it
4. **Perform** the validated edits directly in the workspace
5. **Verify** the result using checks appropriate to the kind of work that was applied
6. **Raise independent concerns** that Cursor agents missed or understated
7. **Remove** the processed worktrees
8. **Repeat follow-up review/refinement loops** until no further high-confidence issues remain

Use the todo list tool throughout. Run all commands yourself — do not leave anything for the user to run.

---

## Phase 1 — Discover Worktrees

Run:
```
git worktree list
```

Collect every path under `~/.cursor/worktrees/`. Record each 3-letter worktree ID.

---

## Phase 2 — Parallel Inventory (launch all subagents at once)

For **every** worktree simultaneously, launch a parallel `Explore` subagent with this task:

> "In the worktree at `<path>`, list all files (committed, staged, unstaged, untracked). Read the full content of every new or changed file. Classify each artifact (code, article/manuscript, documentation, config, notebook, figure, data-analysis output, checklist, audit). Return: (1) worktree ID, (2) list of files with their purpose, (3) all concrete recommendations or diffs — quoted verbatim, (4) which source files or deliverables in the main repo each recommendation targets, and (5) what kind of verification each recommendation would require."

Do NOT read worktrees sequentially. All `Explore` subagents must be launched in the same parallel batch.

---

## Phase 3 — Consolidate and Deduplicate

After all subagents return:

1. **Group outputs by target deliverable** in the main repo — all recommendations touching a given source file, article section, figure set, config, or document together.
2. **Deduplicate** — if multiple agents recommend the same change, keep the most specific/complete version. Note the contributing worktree IDs.
3. **Rank priority by impact**, not by file type. Example priorities:
	- correctness or safety fixes
	- scientific/result integrity fixes
	- broken builds/tests/pipelines
	- article/manuscript factual consistency
	- documentation/config improvements
	- stylistic or structural cleanups
4. Build a **consolidated action plan** as a todo list: one item per distinct change, ordered by priority.

---

## Phase 4 — Audit Each Proposed Change

Before applying **any** change, evaluate it against these criteria:

| Criterion | Questions to ask |
|-----------|-----------------|
| **Correctness** | Does the proposed code actually do what the agent claims? Are there logic errors, off-by-one mistakes, wrong API calls? |
| **Substance / Fidelity** | For articles, docs, figures, or analysis outputs: are claims, captions, references, metric names, and interpretations faithful to the underlying source material and results? |
| **Safety / Security** | Does it introduce any OWASP Top 10 vulnerabilities (injection, broken access, insecure defaults)? |
| **Compatibility** | Does it break existing interfaces, function signatures, or expected behaviour used elsewhere in the codebase? Read the callers before applying. |
| **Consistency** | Does it stay consistent with the rest of the repo: terminology, notation, variable names, reported results, section structure, and existing conventions? |
| **Scope creep** | Is this change actually necessary, or is the agent over-engineering? Only apply what improves the codebase — skip speculative or purely cosmetic changes. |
| **Conflicts** | Does this change conflict with another agent's recommendation for the same file? Resolve by picking the most correct version or synthesising both. |

For each proposed change, record your verdict: **Apply as-is / Apply with corrections / Skip (reason)**. Only proceed to Phase 5 for approved changes.

---

## Phase 5 — Perform the Edits by Artifact Type

Apply every approved change directly to the workspace files using file-editing tools. For each edit:

1. Read the current state of the target file first
2. Apply the minimal correct diff — do not reformat unrelated code
3. If the agent's proposed patch no longer applies cleanly due to drift, re-derive the correct edit from first principles
4. Preserve the style, structure, and genre conventions of that artifact type

Apply artifact-specific handling:

- **Code / scripts / configs**: apply minimal diffs; keep behaviour and interfaces coherent; update related tests or configs if required.
- **Articles / manuscripts / papers**: preserve argument structure, citations, terminology, section flow, figure/table references, and result claims. Never introduce unsupported scientific claims.
- **Documentation / READMEs / checklists**: merge overlapping guidance into one authoritative version per topic.
- **Figures / tables / analysis outputs**: ensure labels, units, legends, metric names, and references match the underlying data and manuscript text.
- **Notebooks / experiment assets**: preserve reproducibility, execution order assumptions, and dataset/config references.

For audit reports or checklists, consolidate the best content from all agents into a single file per topic. Place reference material in `docs/` when appropriate; keep top-level project documents in the repo root when that matches the existing structure.

---

## Phase 6 — Verify with Artifact-Appropriate Checks

After all edits are applied:

1. Determine which verification modes fit the changed artifacts. Use all that apply:
	- **Code**: tests, linting, type checks, build, import smoke tests, targeted execution
	- **Docs / markdown**: link/reference checks, heading consistency, command/path sanity, example accuracy
	- **Articles / manuscripts**: citation consistency, section cross-reference correctness, terminology consistency, figure/table numbering, result-to-claim alignment, and if LaTeX exists, compile when practical
	- **Configs / pipelines**: schema validation, dry-run, parser load, or command validation when available
	- **Notebooks / analyses**: cell execution or equivalent reproducibility checks when practical
2. If a relevant verification path exists, run it yourself.
3. If a relevant verification path fails, diagnose and fix rather than silently accepting the failure.
4. If no practical automated verification exists for a given artifact, perform a manual consistency review and state that explicitly in the final summary.

---

## Phase 7 — Independent Review: What Else?

**After** seeing all Cursor agent outputs and applying their changes, step back and conduct your own independent review. Cursor agents are autonomous but bounded — they may miss issues that emerge from reading multiple files together, from broader engineering perspective, or from manuscript/research-quality review. Look across:

### 7a — Correctness gaps Cursor may have missed
- **Cross-file semantic bugs**: Does function A produce output that function B consumes with wrong assumptions? (e.g. dtype mismatch, shape expectation, off-by-one at boundaries)
- **Untested code paths**: Scan for `if`/`elif` branches, exception handlers, or conditional features that have no test coverage whatsoever
- **Silent failures**: Bare `except:` blocks, functions that return `None` on error instead of raising, or fallback paths that swallow important state

### 7b — Architecture concerns
- **Circular imports or tight coupling** that will cause problems as the codebase grows
- **God classes / god methods**: single methods or classes doing too many unrelated things that Cursor agents flagged but you can now verify against the actual code
- **Dependency direction violations**: utility code importing from higher-level modules

### 7c — Reproducibility & science integrity
- **Unseeded randomness**: any call to `random`, `np.random`, `torch` sampling, or data shuffling that is not covered by the global seed
- **Data leakage**: train/val/test split happening after feature scaling or label-derived transforms
- **Metric correctness**: verify that every metric function computes what it claims (e.g. ARI vs AMI vs NMI — check the import, not just the variable name)

### 7d — Writing, article, and evidence quality
- **Claim-to-evidence mismatches**: statements in abstract, results, conclusions, captions, or README not supported by actual experiments or source data
- **Internal inconsistencies**: conflicting terminology, inconsistent notation, mismatched section names, stale figure references, or contradictory parameter values across paper/docs/code
- **Overstatement**: language that implies proof, generality, or significance not justified by the reported evidence

### 7e — Dependency, tooling, and packaging health
- **Undeclared imports**: cross-check top-level imports against declared dependencies for the languages and tools used in the repo
- **Version pins that are too tight or too loose**: check for constraints that could pull in breaking versions or hide incompatibilities
- **Optional dependencies used unconditionally**: optional packages or tools imported/invoked without guards

### 7f — Security (beyond what Cursor found)
- **Any user-supplied string passed to `eval`, `exec`, `os.system`, `subprocess` without sanitisation**
- **File paths constructed from external input without validation** (path traversal)
- **Credentials or tokens hardcoded** anywhere (emails, API keys, example passwords)

### Output of Phase 7
Present findings as a prioritised list separate from the Cursor-derived changes:
- **P0 (apply now)**: genuine bugs or security issues you found independently
- **P1 (apply if confident)**: correctness concerns worth fixing in this session
- **P2 (flag for later)**: design issues the team should know about but that are not urgent

For P0 and P1 items: apply fixes immediately using the same audit-then-apply discipline as Phases 4–5. Re-run relevant verification after.
For P2 items: list them clearly so the user can act on them in a future session.

### Phase 7 Loop — Do Not Stop After One Review Pass

After applying any P0/P1 fixes from Phase 7, ask the next question explicitly in your own reasoning:

> "Did these fixes, rereads, or verification results reveal any additional P0/P1 issues or follow-up refinements that are now obvious and high-confidence?"

If the answer is **yes**, do not stop. Repeat this loop:

1. Re-read the affected files and adjacent files impacted by the latest fixes
2. Re-run the most relevant verification for those artifacts
3. Re-classify new findings into P0 / P1 / P2
4. Apply any new P0/P1 fixes immediately
5. Repeat until no further high-confidence P0/P1 issues remain

Stop the loop only when one of these is true:

- no further P0/P1 issues remain after the latest verification
- remaining concerns are only P2 / speculative / preference-level
- further edits would require user input, external data, or unsupported assumptions

In the final summary, report how many independent review / refinement cycles were completed.

---

## Phase 8 — Final Summary and Worktree Cleanup

### 8a — Summary report
Print a structured summary:
- Worktrees processed (count + IDs)
- Changes from Cursor agents: applied / corrected / skipped (with reasons for skips)
- Changes from independent review (Phase 7): applied / flagged
- Number of follow-up review / refinement cycles completed after the first independent review pass
- Final verification results by artifact type (tests, builds, compile checks, manual review, manuscript consistency review, etc.)
- `git diff --stat` of the net change to the workspace

### 8b — Worktree removal
For each processed Cursor worktree, run:
```
git worktree remove --force <path>
```

Remove them one at a time and confirm each succeeds before moving to the next. If a worktree removal fails (e.g. locked or contains uncommitted work that was NOT yet integrated), report the error and skip that worktree — do not force-remove anything that still has unintegrated content.

After all removals, run `git worktree list` once more to confirm the workspace is clean.

### 8c — Commit prompt
Show the user the full `git diff --stat` and suggest a commit message summarising what was applied. Do NOT run `git commit` — the user decides when to commit.

---

## Strict Rules

- **Parallel subagents** — Phase 2 subagents must all launch simultaneously, never one-by-one
- **Audit before apply** — never apply a change that fails the Phase 4 audit criteria
- **No blind copy-paste** — always read the current file before applying a patch; re-derive if the file drifted
- **No artifact-type assumptions** — do not assume the repo is Python-only, code-only, or software-only; choose validation based on what changed
- **Converge before stopping** — if your own review surfaces more high-confidence follow-up work, continue the review/fix/verify loop until only lower-priority issues remain
- **No user commands** — run every shell command yourself
- **No auto-commit** — show diff and suggest message; let the user decide
- **No silent skips** — if a change is skipped at any phase, explain why in the Phase 8 summary
- **Worktree removal is mandatory** — do not end the session without attempting Phase 8b
