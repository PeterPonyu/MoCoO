## Task Summary

Created a workspace custom agent that operationalizes the repository's context-preserving Cursor-style workflow and added the corresponding `agent-context` memory trail for this setup task.

## Current State

- DONE: agent file created
- DONE: repository-level `agent-context` files initialized
- DONE: task-specific handoff files written

## Completed

- Extracted the specialized role from the conversation and `prompts.md`.
- Chose workspace scope and a minimal useful tool set.
- Added `.github/agents/context-preserving-task.agent.md`.
- Recorded project knowledge, current focus, and a Claude-oriented brief.

## Remaining

- Optional: rename the agent if the team wants stronger branding.
- Optional: adjust the tool set after real-world usage.

## Important Files

- `.github/agents/context-preserving-task.agent.md`
- `prompts.md`
- `agent-context/current-focus.md`
- `agent-context/project-map.md`
- `agent-context/claude-brief.md`

## Risks / Blockers

- No functional blocker.
- The main open design question is whether terminal access should remain enabled by default.

## Recommended Next Actions

1. Use the new agent on a small real task.
2. Review whether its `agent-context` output is concise and useful.
3. Narrow or expand tool access if actual usage shows drift.

## Suggested Prompt for Next Model

Use the `Context-Preserving Task Agent` to complete this repository task, keep `agent-context` updated throughout, and leave a concise `handoff.md` for the next model.