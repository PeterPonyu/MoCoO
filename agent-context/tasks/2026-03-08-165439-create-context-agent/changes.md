## Files Created

- `.github/agents/context-preserving-task.agent.md`
- `agent-context/current-focus.md`
- `agent-context/project-map.md`
- `agent-context/claude-brief.md`
- `agent-context/tasks/2026-03-08-165439-create-context-agent/target.md`
- `agent-context/tasks/2026-03-08-165439-create-context-agent/plan.md`
- `agent-context/tasks/2026-03-08-165439-create-context-agent/worklog.md`
- `agent-context/tasks/2026-03-08-165439-create-context-agent/changes.md`
- `agent-context/tasks/2026-03-08-165439-create-context-agent/validation.md`
- `agent-context/tasks/2026-03-08-165439-create-context-agent/handoff.md`

## Files Modified

- None.

## Files Removed

- None.

## Behavioral Changes

- The repository now exposes a dedicated workspace custom agent for context-preserving task execution.
- Future runs can store durable task context in `agent-context/` rather than relying only on chat history.

## Interface Changes

- Added a new selectable custom agent in the VS Code agent picker.

## Config/Env Changes

- None.

## Migration Notes

- Existing prompt files remain usable; the new agent provides a stronger reusable entry point for the same workflow.
