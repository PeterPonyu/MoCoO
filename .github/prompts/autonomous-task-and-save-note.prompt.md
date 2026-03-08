---
description: "Handle an open-ended repository task and save one clear local markdown note with the result. Use when the task should be completed directly and left in a form that is easy to resume later."
name: "Autonomous Task And Save Note"
argument-hint: "Describe the goal, target deliverable, and any priorities or boundaries"
agent: "agent"
---

# Autonomous Task And Save Note

You are handling an open-ended repository task.

The user may ask for code changes, debugging, documentation work, manuscript revisions, analysis refinement, review, audit, or mixed tasks spanning several artifact types.

Complete the task directly when the path is clear.

## Approach

Treat the user request as the deliverable plus any limits around it.

Work this way:

- infer the real objective behind the request
- gather only the context needed to act confidently
- choose your own execution strategy
- apply changes directly when the path is clear
- verify results in the most relevant way for the artifact type
- save one concise local markdown note so the work can be resumed later

Do not ask unnecessary questions if you can make progress safely.
Do not stop at partial completion if more obvious, high-confidence work remains.

## Save One Local Note

Before finishing, create or update a single local markdown file for the task.

Use a direct path and name, for example:

- `<timestamp>-<short-target-slug>.md`
- `notes/<timestamp>-<short-target-slug>.md`
- `task-notes/<timestamp>-<short-target-slug>.md`

Do not create a multi-file task folder unless the user explicitly asks for that structure.

## What The Note Should Contain

Keep the note short, factual, and easy for another agent or human to scan.

Use simple sections such as:

- task
- context checked
- actions taken
- key decisions
- validation
- final result
- open issues or next steps

If the task is primarily a review, use `findings` as the first section and order items by severity or importance.

## Notes Rule

Do not dump hidden private chain-of-thought.
Instead, record concise summaries of what you checked, what you decided, and why.

## Context Gathering

Before acting, gather enough context to answer:

- what is the actual deliverable?
- which files or artifacts are involved?
- what existing conventions matter?
- what are the likely failure modes?
- what kind of verification is meaningful here?

Search and read in parallel when useful, but avoid noisy or redundant exploration.
Stop gathering context once you have enough confidence to act.

## Execution

Choose your editing and validation strategy based on what the task actually touches.

- **Code / scripts / configs**: preserve intended behavior unless fixing it; keep diffs minimal; avoid unrelated refactors.
- **Documentation / READMEs / guides**: optimize for clarity, consistency, and accuracy; keep examples runnable or obviously correct.
- **Articles / manuscripts / reports**: preserve argument structure, terminology, references, figure/table coherence, and evidence-to-claim alignment.
- **Figures / tables / result summaries**: ensure labels, units, metric names, legends, and narrative references match the underlying results.
- **Notebooks / experiments / analyses**: preserve reproducibility assumptions, execution coherence, and dataset/config references.

If the task spans multiple artifact types, integrate them coherently rather than treating them as separate unrelated edits.

## Validation

Pick verification methods that fit the artifact type. Use all relevant ones that materially increase confidence.

Examples:

- tests, builds, targeted execution, linting, type checks, import smoke tests
- config parsing, dry-runs, workflow validation
- document consistency checks, link/path sanity checks, example verification
- manuscript consistency review, figure/table numbering checks, citation/reference consistency, result-to-claim alignment
- manual expert review when automation is unavailable or not appropriate

If verification fails, fix the issue if it is within scope and high-confidence.
If no practical automated verification exists, perform a careful manual review and say so explicitly.

## Final Output

By the end, normally provide:

- what you changed or decided not to change
- why those choices were correct
- what verification you ran
- any remaining lower-priority concerns

In chat, give the result in the format that best fits the task.
Also save the same outcome in the single local markdown note.