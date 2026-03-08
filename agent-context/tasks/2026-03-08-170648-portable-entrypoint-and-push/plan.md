## Current Strategy

Make `AGENTS.md` the explicit source of truth, add a Copilot bridge file, update the portable bundle installer and docs, validate the bundle, then stage only the workflow files for commit and push.

## Alternatives Considered

- Add several editor-specific entry files with duplicated logic.
- Leave the entrypoint implied instead of explicit.

## Decision Summary

- Chose a single canonical entrypoint to avoid drift across editor-specific files.
- Added one lightweight Copilot bridge file because it improves automatic discovery without fragmenting the workflow contract.
- Will use selective staging to avoid including unrelated worktree edits in the commit.

## Implementation Steps

1. Refine `AGENTS.md` around the canonical entrypoint rule.
2. Add `.github/copilot-instructions.md`.
3. Update the agent, prompt, docs, and installer to reference the entrypoint.
4. Validate the bundle.
5. Stage only the workflow bundle files, commit, and push to `bp`.

## Validation Plan

- Run `bash -n scripts/install_agent_bundle.sh`.
- Run `bash scripts/install_agent_bundle.sh --help`.
- Inspect `git status --short` before and after staging.

## Replan Notes

- None yet.
