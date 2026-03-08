## Task

Make the new context-preserving agent workflow easy to push to the repository, easy to deploy into other repositories, and explicit about Cursor-to-Claude continuation.

## User Request

Update the newly created agent so it can be pushed to the repository, deployed across projects easily, and support switching from Cursor-based task execution to Claude-based continuation using the saved process and the apply-Cursor workflow.

## Interpreted Goal

Turn the current repo-specific customization into a portable bundle with clear repository-resident continuity rules and a practical deployment path for reuse in other repos.

## Success Criteria

- The workflow is documented in a repo-wide, tool-agnostic file that Claude can follow.
- The custom agent and apply-Cursor prompt explicitly reference repository-resident continuity files.
- There is an easy way to install the workflow into another repository.
- The continuation path from Cursor to Claude is documented clearly.

## Constraints

- Keep the continuity state in plain Markdown inside the repository.
- Avoid relying on editor-only state for handoff.
- Preserve existing unrelated repository changes.

## Relevant Files

- `AGENTS.md`
- `.github/agents/context-preserving-task.agent.md`
- `.github/prompts/apply-cursor-agent-edits.prompt.md`
- `scripts/install_agent_bundle.sh`
- `docs/AGENT_DEPLOYMENT.md`

## Risks

- Confusing prompt files with durable task state.
- Overwriting useful customization files in target repositories during deployment.

## Open Questions

- Whether the deployment script should remain conservative by default or become more opinionated later.
