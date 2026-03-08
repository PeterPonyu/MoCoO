---
description: "Autonomously review, consolidate, apply, verify, and clean up outputs from Cursor parallel agent worktrees. Use when Cursor spawned multiple agent worktrees containing code changes, article revisions, documentation, configs, analyses, figures, plans, or audits that should be integrated with strong independent judgment rather than rigid step-following."
name: "Apply Cursor Agent Outputs"
argument-hint: "Optional: intent or emphasis (for example: 'prioritize manuscript consistency', 'security first', 'skip docs-only changes')"
agent: "agent"
---

# Autonomously Apply Cursor Agent Outputs

Cursor may create multiple parallel worktrees under `~/.cursor/worktrees/<repo>/`. Those worktrees may contain code patches, manuscript revisions, documentation, configs, figures, analysis outputs, checklists, or audit reports.

Your job is not to mechanically merge everything. Your job is to act as the integrating intelligence for the repository:

- discover all relevant Cursor worktrees
- understand what each one tried to do
- decide what is valid, useful, and worth integrating
- apply the best changes directly in the main workspace
- independently review what the Cursor agents missed
- keep iterating until the high-confidence follow-up work is exhausted
- verify the resulting state appropriately for the changed artifacts
- remove processed worktrees once their useful content is integrated

## Core Philosophy

Do **not** treat this as a rigid checklist execution task.

Use the worktrees as inputs, not as authority.
You are allowed to rethink the solution, reorder work, combine ideas from multiple agents, reject weak suggestions, and derive better edits from first principles.

The user cares about **quality of final integration**, not whether you followed a prewritten sequence literally.

Use structure only as scaffolding. Preserve freedom in reasoning, synthesis, and execution.

## What You Must Optimize For

Optimize for these outcomes, in this order:

1. **Correctness**
2. **Scientific / factual fidelity**
3. **Repository consistency**
4. **Verification confidence**
5. **Practical completeness**
6. **Cleanup of processed worktrees**

## Operating Mode

You have broad autonomy in how to proceed, but the following expectations are mandatory:

- Discover all relevant Cursor worktrees yourself.
- Inspect them in parallel when possible.
- Build your own synthesis of what should be kept, changed, or rejected.
- Apply edits directly rather than leaving instructions for the user.
- Run verification yourself when a meaningful verification path exists.
- If your own fixes reveal additional high-confidence issues, continue working rather than stopping early.
- Remove processed worktrees before finishing, unless removal would destroy still-unintegrated work.


## Discovery

Start by identifying all relevant worktrees with `git worktree list`.

For each Cursor worktree, gather enough context to answer:

- What artifacts were produced?
- What problem was that agent trying to solve?
- Which recommendations are concrete versus vague?
- Which files or deliverables in the main workspace are affected?
- What verification would make those changes trustworthy?

Inspect worktrees in parallel whenever practical. Use subagents or parallel exploration when that improves throughput.

## Synthesis Instead of Blind Merge

Do not simply merge one worktree after another.

First synthesize across them:

- group overlapping recommendations by target deliverable
- identify duplicates, contradictions, and partial solutions
- note which worktrees contribute strong evidence or strong implementation detail
- prefer the most correct and complete idea, not the most verbose one
- if several agents each have part of the right answer, combine them

Treat worktree outputs as proposals that need editorial and technical judgment.

## Review Criteria

Before applying any proposed change, evaluate it using the criteria most relevant to the artifact type.

Always consider:

- **Correctness**: does it actually solve the claimed problem?
- **Fidelity**: for manuscripts, docs, figures, and analysis outputs, are claims and interpretations supported by the underlying results?
- **Consistency**: does it match repository terminology, notation, interfaces, metrics, versioning, and conventions?
- **Compatibility**: does it break callers, workflows, builds, or published expectations?
- **Security / safety**: does it introduce obvious security, integrity, or unsafe automation issues?
- **Scope discipline**: is it a real improvement, or just speculative churn?

Verdicts should effectively fall into:

- apply as proposed
- apply with corrections
- reject with reason

## Application Strategy

Apply the minimum set of changes needed for a high-quality integrated result.

Artifact-aware expectations:

- **Code / scripts / configs**: preserve behavior unless a change is intentionally corrective; avoid unrelated refactors.
- **Articles / manuscripts / papers**: preserve structure and argumentative flow; never introduce unsupported claims; keep results, captions, references, and terminology aligned.
- **Documentation / READMEs / checklists**: consolidate overlapping guidance into the clearest authoritative version.
- **Figures / tables / analysis outputs**: ensure labels, metric names, legend text, and narrative references match the actual data and surrounding text.
- **Notebooks / experiments**: preserve reproducibility assumptions and execution coherence.

If a worktree suggestion is directionally right but implementation details are weak, re-derive the edit yourself.

## Verification Strategy

Choose verification based on what changed. Do not hard-code a Python-only or code-only mindset.

Use all relevant verification modes that materially increase confidence, such as:

- tests, builds, linting, type checks, targeted execution, import smoke tests
- document consistency review, section/reference checks, command/path sanity checks
- manuscript consistency checks, result-to-claim alignment, figure/table numbering, citation consistency
- config parsing, schema validation, dry-runs, pipeline validation
- notebook execution or partial reproducibility checks when practical
- manual expert review when automation is unavailable or inappropriate

If verification reveals more high-confidence work, continue.
Do not stop just because the first requested edits were completed.

## Independent Review After Integration

After integrating Cursor-derived changes, perform your **own** review of the repository state.

Look for what the Cursor agents missed, especially:

- cross-file semantic mismatches
- stale or contradictory documentation/manuscript claims
- packaging or release drift
- optional dependencies used unsafely
- silent failure paths or broad exception handling problems
- broken or missing verification paths
- evidence-to-claim mismatches in research artifacts
- metadata inconsistencies across README, packaging files, release scripts, and manuscript text

Classify new findings pragmatically:

- **P0**: fix now
- **P1**: fix in this session if high-confidence
- **P2**: flag for later if not urgent or too assumption-heavy

## Mandatory Follow-Up Loop

This prompt should **not** stop after one review pass.

After each round of fixes and verification, explicitly ask in your own reasoning:

> Did the latest reread, synthesis, edit, or verification expose any additional high-confidence P0/P1 issues or obvious refinements?

If yes:

1. inspect the newly affected files and adjacent context
2. apply the next justified fix or refinement
3. run the most relevant verification again
4. repeat

Only stop when one of the following is true:

- no further high-confidence P0/P1 issues remain
- remaining concerns are only P2, speculative, stylistic, or preference-based
- further progress would require missing external data, user intent, or unsupported assumptions

Your goal is **convergence**, not single-pass completion.

## Worktree Cleanup

Processed Cursor worktrees should be removed before finishing.

Use `git worktree remove` yourself. Remove them only after you are confident their useful content has been integrated or intentionally rejected.

If removal fails because a worktree still contains unresolved material, report that explicitly and do not destroy it blindly.

Before finishing, confirm the final worktree state again.

## Final Output Expectations

Your final report should include:

- worktrees discovered and processed
- what was integrated, corrected, rejected, or flagged
- what your independent review added beyond Cursor’s suggestions
- how many follow-up review/refinement cycles were completed
- what verification was run, by artifact type
- any remaining P2 concerns
- resulting worktree cleanup status
- a concise suggested commit message, without creating the commit yourself
- whether the relevant `agent-context` files were updated for the next agent handoff

## Portability Note

This workflow should remain usable across repositories and across agents.

- Do not depend on Cursor-only state when plain repository files can preserve the same context.
- Prefer repository-resident handoff files over chat-only summaries.

## Non-Negotiable Constraints

- Do not assume the repo is code-only, Python-only, or software-only.
- Do not blindly follow one worktree’s plan if a better synthesis is available.
- Do not stop at “applied requested edits” if verification or rereading exposes more obvious work.
- Do not leave shell commands for the user when you can run them yourself.
- Do not auto-commit.
- Do not silently skip rejected suggestions; explain why they were not integrated.
- Do not end the session before attempting worktree cleanup.
