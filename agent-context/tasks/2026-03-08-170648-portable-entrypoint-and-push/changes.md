## Files Created

- `.github/copilot-instructions.md`
- `agent-context/tasks/2026-03-08-170648-portable-entrypoint-and-push/target.md`
- `agent-context/tasks/2026-03-08-170648-portable-entrypoint-and-push/plan.md`
- `agent-context/tasks/2026-03-08-170648-portable-entrypoint-and-push/worklog.md`
- `agent-context/tasks/2026-03-08-170648-portable-entrypoint-and-push/changes.md`
- `agent-context/tasks/2026-03-08-170648-portable-entrypoint-and-push/validation.md`
- `agent-context/tasks/2026-03-08-170648-portable-entrypoint-and-push/handoff.md`

## Files Modified

- `AGENTS.md`
- `.github/agents/context-preserving-task.agent.md`
- `.github/prompts/general-autonomous-task.prompt.md`
- `docs/AGENT_DEPLOYMENT.md`
- `scripts/install_agent_bundle.sh`
- `agent-context/current-focus.md`
- `agent-context/project-map.md`
- `agent-context/claude-brief.md`

## Files Removed

- None.

## Behavioral Changes

- `AGENTS.md` is now explicitly the canonical repository entrypoint for the workflow.
- Copilot has a repository-local automatic bridge to the same entrypoint.
- The installer now deploys that bridge file to target repositories.

## Interface Changes

- None beyond the new `.github/copilot-instructions.md` bridge file.

## Config/Env Changes

- None.

## Migration Notes

- Keep editor-specific helper files thin and pointing to `AGENTS.md` rather than re-implementing the workflow independently.
