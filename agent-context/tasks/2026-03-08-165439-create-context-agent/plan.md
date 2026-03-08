## Current Strategy

Translate `prompts.md` into a focused workspace custom agent with minimal necessary tools, then create the supporting `agent-context` files required by that workflow.

## Alternatives Considered

- Keep the behavior only as a prompt file.
- Convert the behavior into workspace instructions instead of a custom agent.

## Decision Summary

- Chose a custom agent because the user explicitly asked for `.agent.md` creation and the workflow is specialized enough to merit explicit agent selection.
- Kept the tool set to `read`, `search`, `edit`, `execute`, and `todo` because the workflow needs repository edits, shell validation, and task tracking.

## Implementation Steps

1. Read the create-agent instructions and the agent-customization reference.
2. Extract the specialized role, tool preferences, and job scope from the conversation and `prompts.md`.
3. Create `.github/agents/context-preserving-task.agent.md`.
4. Create the repository `agent-context` files for this task.
5. Validate file placement and frontmatter structure.

## Validation Plan

- Confirm the agent file is under `.github/agents/`.
- Review the frontmatter for required fields.
- Inspect git status to ensure the expected files were created.

## Replan Notes

- No replan required so far.
