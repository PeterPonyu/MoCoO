## Current Strategy

Add a repo-wide continuity contract, update the existing agent and prompt to consume that contract, and provide a deployment script and documentation for cross-repo reuse.

## Alternatives Considered

- Leave portability as undocumented manual copy-paste.
- Depend only on VS Code prompt files without a tool-agnostic `AGENTS.md` layer.

## Decision Summary

- Chose `AGENTS.md` because it is a repository-resident, model-agnostic entry point that Claude can follow.
- Chose a deployment script because it makes cross-project reuse repeatable and safer than ad hoc copying.
- Kept deployment conservative by default and only overwriting files with `--force`.

## Implementation Steps

1. Add `AGENTS.md` with continuity and handoff rules.
2. Update the custom agent and apply-Cursor prompt to reference `agent-context/`.
3. Add a deployment script for other repositories.
4. Add a short guide covering deployment and Cursor-to-Claude switching.
5. Validate syntax and file placement.

## Validation Plan

- Run `bash -n` on the deployment script.
- Run the script with `--help`.
- Review repository status for the expected new files.

## Replan Notes

- No replan required so far.
