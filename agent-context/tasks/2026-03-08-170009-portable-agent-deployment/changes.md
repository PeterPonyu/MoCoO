## Files Created

- `AGENTS.md`
- `scripts/install_agent_bundle.sh`
- `docs/AGENT_DEPLOYMENT.md`
- `agent-context/tasks/2026-03-08-170009-portable-agent-deployment/target.md`
- `agent-context/tasks/2026-03-08-170009-portable-agent-deployment/plan.md`
- `agent-context/tasks/2026-03-08-170009-portable-agent-deployment/worklog.md`
- `agent-context/tasks/2026-03-08-170009-portable-agent-deployment/changes.md`
- `agent-context/tasks/2026-03-08-170009-portable-agent-deployment/validation.md`
- `agent-context/tasks/2026-03-08-170009-portable-agent-deployment/handoff.md`

## Files Modified

- `.github/agents/context-preserving-task.agent.md`
- `.github/prompts/apply-cursor-agent-edits.prompt.md`
- `agent-context/current-focus.md`
- `agent-context/project-map.md`
- `agent-context/claude-brief.md`

## Files Removed

- None.

## Behavioral Changes

- The repository now has a tool-agnostic continuity contract via `AGENTS.md`.
- The apply-Cursor workflow now explicitly resumes from `agent-context/` before inspecting worktrees.
- The workflow can now be installed into another repository with a single script invocation.

## Interface Changes

- Added a reusable shell entry point: `bash scripts/install_agent_bundle.sh <target-repo>`.

## Config/Env Changes

- None.

## Migration Notes

- For cross-agent continuation, treat `agent-context/` as the source of truth and use the prompt files as workflow references.
