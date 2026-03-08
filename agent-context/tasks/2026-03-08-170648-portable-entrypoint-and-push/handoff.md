## Task Summary

Refined the portable workflow so all agent-related configuration converges on `AGENTS.md` as the canonical repository entrypoint, with editor-specific bridge behavior layered on top.

## Current State

- DONE: canonical entrypoint wording added
- DONE: Copilot bridge file added
- DONE: installer and docs updated
- IN PROGRESS: commit and push

## Completed

- Updated `AGENTS.md` to be the explicit source of truth.
- Added `.github/copilot-instructions.md`.
- Updated the agent, prompt, docs, and installer around that entrypoint model.

## Remaining

- Commit and push to bp.

## Important Files

- `AGENTS.md`
- `.github/copilot-instructions.md`
- `.github/agents/context-preserving-task.agent.md`
- `.github/prompts/general-autonomous-task.prompt.md`
- `.github/prompts/apply-cursor-agent-edits.prompt.md`
- `docs/AGENT_DEPLOYMENT.md`
- `scripts/install_agent_bundle.sh`

## Risks / Blockers

- The repository contains unrelated modified files, so staging must stay surgical.
- Push may fail if remote auth or policy blocks it.

## Recommended Next Actions

1. Validate the bundle again after refinement.
2. Stage only the workflow bundle files.
3. Commit and push to bp.

## Suggested Prompt for Next Model

Inspect `AGENTS.md` and the latest task folder, then verify the workflow bundle is staged cleanly without unrelated repo edits before pushing.