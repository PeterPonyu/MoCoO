## Task

Create a workspace custom agent that captures the behavior described in `prompts.md` and is suitable for future Claude or Copilot runs.

## User Request

Follow the create-agent prompt instructions and turn the repository's context-preserving Cursor-style workflow into a reusable `.agent.md` customization.

## Interpreted Goal

Add a workspace agent that can be selected for meaningful repository tasks where the agent should both execute the work and persist a durable task trail under `agent-context/`.

## Success Criteria

- A `.github/agents/*.agent.md` file exists with valid frontmatter and a clear specialized role.
- The agent description is keyword-rich enough for discovery and picker use.
- The behavior from `prompts.md` is translated into concise agent instructions.
- Repository-resident `agent-context` files document this setup task for future model continuity.

## Constraints

- Use workspace scope, not user-profile scope.
- Keep the agent focused and tool usage minimal but sufficient.
- Persist decision-oriented summaries rather than chain-of-thought.

## Relevant Files

- `prompts.md`
- `.github/prompts/general-autonomous-task.prompt.md`
- `.github/prompts/apply-cursor-agent-edits.prompt.md`
- `.github/agents/context-preserving-task.agent.md`

## Risks

- Over-scoping the agent into a generic Swiss-army persona.
- Under-specifying the persistence requirements and losing the intended handoff behavior.

## Open Questions

- Should future revisions narrow terminal access or keep the current default tool set?
- Does the team prefer a more branded agent name?
