## Task

Refine the portable workflow so all agent-related configuration has a single canonical repository entrypoint and prepare a clean push of those reusable files.

## User Request

Make prompts, custom agents, skills, and instructions converge on one entry file that tells the current model how to find the configuration, keep the setup portable across editors and UIs, and push the reusable workflow updates to the remote GitHub repository.

## Interpreted Goal

Use `AGENTS.md` as the canonical source of truth, add any missing bridge files needed for editor support, refine portability docs and installer behavior, then commit and push only the workflow bundle changes.

## Success Criteria

- `AGENTS.md` is clearly documented as the canonical entrypoint.
- Copilot and other tooling have a repository-local bridge to that entrypoint.
- The portable installer copies the full reusable bundle.
- The commit excludes unrelated repository modifications.
- The workflow changes are pushed to `bp/main` if authentication permits.

## Constraints

- Do not include unrelated modified files in the commit.
- Keep the workflow portable across editors and not dependent on one UI.
- Preserve existing unrelated worktree changes.

## Relevant Files

- `AGENTS.md`
- `.github/copilot-instructions.md`
- `.github/agents/context-preserving-task.agent.md`
- `.github/prompts/general-autonomous-task.prompt.md`
- `.github/prompts/apply-cursor-agent-edits.prompt.md`
- `docs/AGENT_DEPLOYMENT.md`
- `scripts/install_agent_bundle.sh`

## Risks

- Accidentally staging unrelated modified files.
- Push failure due to remote authentication or branch protection.

## Open Questions

- None blocking; proceed with a clean staged commit of only the workflow bundle.
