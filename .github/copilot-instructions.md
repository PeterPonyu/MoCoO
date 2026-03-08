# Copilot Repository Entrypoint

Start with `AGENTS.md`.

Use `AGENTS.md` as the canonical source for:

- how task continuity is stored
- how to resume prior work
- what must be written into `agent-context/`
- how to keep the workflow portable across editors and models

If this repository contains editor-specific prompts, custom agents, or other customization files, treat them as helpers layered on top of `AGENTS.md`, not as competing sources of truth.

For meaningful tasks:

1. read `AGENTS.md`
2. read `agent-context/current-focus.md` if present
3. read `agent-context/claude-brief.md` if present
4. continue in the relevant task folder under `agent-context/tasks/` when the work is a continuation

Persist concise engineering summaries in plain Markdown so another model can continue from the repository alone.
