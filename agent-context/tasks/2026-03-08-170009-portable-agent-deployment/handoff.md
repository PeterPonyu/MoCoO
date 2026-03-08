## Task Summary

Made the context-preserving agent workflow portable across repositories and explicit for Cursor-to-Claude continuation.

## Current State

- DONE: repo-wide continuity contract added
- DONE: custom agent updated for tool-agnostic handoff
- DONE: apply-Cursor prompt updated to consume `agent-context/`
- DONE: deployment script and guide added
- DONE: validation record updated with script and status checks

## Completed

- Added `AGENTS.md`.
- Added `scripts/install_agent_bundle.sh`.
- Added `docs/AGENT_DEPLOYMENT.md`.
- Updated the custom agent and apply-Cursor prompt.

## Remaining

- Optional: test installation into a second repository before standardizing the bundle.

## Important Files

- `AGENTS.md`
- `.github/agents/context-preserving-task.agent.md`
- `.github/prompts/apply-cursor-agent-edits.prompt.md`
- `scripts/install_agent_bundle.sh`
- `docs/AGENT_DEPLOYMENT.md`

## Risks / Blockers

- No functional blocker.
- The main remaining risk is unvalidated script syntax or usage text.

## Recommended Next Actions

1. Push the new workflow files.
2. Test installation into a second repository.
3. If desired, narrow the agent tool scope after one or two real tasks.

## Suggested Prompt for Next Model

Review `AGENTS.md`, `agent-context/current-focus.md`, and the latest task handoff, then validate whether the portable bundle is sufficient for a Cursor-to-Claude continuation without chat history.