# Portable Agent Deployment

This repository now includes a portable agent workflow bundle designed for reuse across repositories and across coding agents.

## Canonical Entrypoint

`AGENTS.md` is the canonical repository entrypoint.

Any editor-specific configuration should either read it directly or point to it. The goal is that a cloned repository still contains the full operational contract even if the next editor UI ignores custom prompt metadata.

This is why `AGENTS.md` stays at repository root instead of moving into `.github/prompts/`.
Prompt files are useful for editor integration, but they are not a sufficiently portable source of truth on their own.

## Included Files

- `AGENTS.md`
- `.github/copilot-instructions.md`
- `.github/agents/context-preserving-task.agent.md`
- `.github/prompts/apply-cursor-agent-edits.prompt.md`
- `.github/prompts/general-autonomous-task.prompt.md`
- `.github/prompts/review-and-persist.prompt.md`
- `scripts/install_agent_bundle.sh`

## Deploy To Another Repository

From this repository, run:

```bash
bash scripts/install_agent_bundle.sh /path/to/target-repo
```

Use `--force` if you intentionally want to overwrite existing copies:

```bash
bash scripts/install_agent_bundle.sh --force /path/to/target-repo
```

The script will:

- copy the portable workflow files
- create the `agent-context/` runtime directory and its README if they do not exist
- preserve existing target-repo files unless `--force` is supplied

## Best Practice For Cross-Project Use

- Keep the workflow contract in `AGENTS.md` so non-VS-Code agents can follow it.
- Keep live task state in `agent-context/` rather than only in chat.
- Reuse the same task folder when the work is a continuation of the same task.
- Use plain Markdown summaries so the next agent can resume without editor-specific state.
- Do not treat `agent-context/` as a portable config folder; it is repository-specific task memory.

## Switching From Cursor To Claude

Yes, this can work well, but only if the important state is stored in the repository.

Recommended resume order for Claude:

1. read `AGENTS.md`
2. read `agent-context/current-focus.md`
3. read `agent-context/claude-brief.md`
4. read the latest relevant task `handoff.md`
5. inspect the actual changed files
6. if Cursor created worktrees, use `.github/prompts/apply-cursor-agent-edits.prompt.md` as the operating reference while reviewing those worktrees

Important nuance:

- Claude will not automatically execute VS Code prompt files.
- The prompt file is still useful because it documents the integration workflow.
- The real continuity comes from `agent-context/` and the repository files themselves.

## Source Repo Versus Target Repo

- The source bundle repository should carry the reusable configuration files and a short `agent-context/README.md`.
- A target project repository should accumulate real task-state files in `agent-context/` as work proceeds.

Review tasks are included in this rule: a meaningful review should create a real task folder and save `review-report.md` plus the standard task files.

## Recommended Pattern

- Start the task with the `Context-Preserving Task Agent` in VS Code.
- Let automatic repository instructions point Copilot to `AGENTS.md`.
- Let it keep `agent-context/` current as it works.
- If you switch to Claude, point Claude at `AGENTS.md` and the latest task folder.
- If Cursor created separate worktrees, use the apply-Cursor prompt as the checklist for integrating them.
