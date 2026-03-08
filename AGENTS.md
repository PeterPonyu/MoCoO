# Agent Continuity Guide

This file is the canonical repository entrypoint for agent configuration and task continuity.

Any model or editor that opens this repository should treat `AGENTS.md` as the first file to read for workflow expectations, handoff rules, and portability guidance.

This repository uses a portable, tool-agnostic task continuity workflow.

## Primary Goal

For meaningful tasks, agents should both:

1. complete the task itself
2. leave behind a reusable task trail in `agent-context/`

The task trail is intended to be readable by Cursor, GitHub Copilot, Claude, and other strong coding agents.

## Entrypoint Rule

Use `AGENTS.md` as the source of truth for how agents should operate in this repository.

- Editor-specific prompt, agent, instruction, or skill files should point back to `AGENTS.md` rather than redefining the workflow independently.
- Durable state belongs in `agent-context/`, not in editor-specific configuration.
- If a tool supports automatic repository instructions, those instructions should direct the model to `AGENTS.md` first.

## Required Continuity Files

For each meaningful task, create or update a timestamped task folder under:

`agent-context/tasks/<timestamp>-<short-target-slug>/`

Maintain:

- `target.md`
- `plan.md`
- `worklog.md`
- `changes.md`
- `validation.md`
- `handoff.md`

Maintain repository-level continuity files when useful:

- `agent-context/current-focus.md`
- `agent-context/project-map.md`
- `agent-context/claude-brief.md`

## Cross-Agent Resume Order

When a new agent takes over work, it should orient itself in this order:

1. read `agent-context/current-focus.md`
2. read `agent-context/claude-brief.md`
3. open the most recent relevant task folder under `agent-context/tasks/`
4. read that task's `handoff.md`
5. read that task's `validation.md` if verification status matters
6. inspect the actual changed files before editing further

If the agent supports automatic repository instructions, it should follow this order without waiting for user restatement.

## Cursor To Claude Handoff

If work started in Cursor and continues in Claude:

- use the `agent-context` files as the main continuity source
- treat `.github/prompts/apply-cursor-agent-edits.prompt.md` as a workflow reference, not as magical state
- if Cursor created worktrees, inspect them directly and reconcile them with the `agent-context` task files
- keep future updates in the same task folder when continuing the same task

Claude will not automatically execute VS Code prompt files, but it can follow this workflow reliably because all critical state is stored in plain Markdown inside the repo.

## Persistence Rules

Persist concise engineering summaries, not hidden chain-of-thought.

Record:

- user request and interpreted goal
- constraints and assumptions
- decision summaries
- files inspected and changed
- validation commands and outcomes
- unresolved issues and next actions

Do not record secrets, tokens, or irrelevant terminal noise.

## Portability

This workflow is designed to be copied into other repositories.

Portable files:

- `AGENTS.md`
- `.github/copilot-instructions.md`
- `.github/agents/context-preserving-task.agent.md`
- `.github/prompts/apply-cursor-agent-edits.prompt.md`
- `.github/prompts/general-autonomous-task.prompt.md`

Use `scripts/install_agent_bundle.sh <target-repo>` to install this bundle into another repository.

After installation, adjust only the repository-specific details. Keep `AGENTS.md` as the canonical entrypoint.

