## Current Objective

Make `AGENTS.md` the single canonical entrypoint for the portable workflow and prepare a clean push of only the workflow bundle files.

## Latest Important Context

- The repository now has a workspace custom agent at `.github/agents/context-preserving-task.agent.md`.
- Root `AGENTS.md` defines the cross-agent continuity contract.
- `.github/copilot-instructions.md` now bridges Copilot back to `AGENTS.md` as the source of truth.
- `.github/prompts/apply-cursor-agent-edits.prompt.md` now explicitly consumes `agent-context/` before reviewing Cursor worktrees.
- `scripts/install_agent_bundle.sh` can copy the portable bundle into another repository.

## Relevant Architecture

- Library code is in `mocoo/`.
- Tests are in `tests/`.
- Benchmark and plotting workflows live under `benchmarks/scripts/`.
- Package and tooling metadata are in `pyproject.toml` and `setup.cfg`.

## Recent Changes

- Added `AGENTS.md` for tool-agnostic handoff behavior.
- Added `.github/copilot-instructions.md` so Copilot reads the same repository entrypoint.
- Updated the custom agent to require plain Markdown, editor-independent handoff artifacts.
- Added a deployment guide and a reusable install script.

## Known Constraints

- Persist concise engineering summaries, not hidden chain-of-thought.
- Keep the workflow repo-resident and portable rather than depending on VS Code state.

## Unresolved Questions

- Whether the deployment script should eventually install a more opinionated task template set.
- Whether the custom agent tool scope should be narrowed after real-world usage.

## Suggested Next Prompt

Stage only the portable workflow files, commit them without unrelated repo edits, and push to bp.


