#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [--force] <target-repo>"
  echo
  echo "Install the portable agent workflow bundle into another repository."
  echo
  echo "Examples:"
  echo "  $0 ../other-repo"
  echo "  $0 --force /path/to/repo"
}

force=0

if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

if [[ ${1:-} == "--help" ]]; then
  usage
  exit 0
fi

if [[ ${1:-} == "--force" ]]; then
  force=1
  shift
fi

if [[ $# -ne 1 ]]; then
  usage
  exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_repo="$(cd "$script_dir/.." && pwd)"
target_repo="${1%/}"

if [[ ! -d "$target_repo" ]]; then
  echo "Target repository does not exist: $target_repo" >&2
  exit 1
fi

copy_file() {
  local source_path="$1"
  local target_path="$2"

  if [[ -e "$target_path" && $force -eq 0 ]]; then
    echo "Skipping existing file: $target_path"
    return
  fi

  mkdir -p "$(dirname "$target_path")"
  install -m 0644 "$source_path" "$target_path"
  echo "Installed: $target_path"
}

write_if_missing() {
  local target_path="$1"
  local content="$2"

  if [[ -e "$target_path" && $force -eq 0 ]]; then
    echo "Skipping existing file: $target_path"
    return
  fi

  mkdir -p "$(dirname "$target_path")"
  printf '%s' "$content" > "$target_path"
  echo "Initialized: $target_path"
}

copy_file "$source_repo/AGENTS.md" "$target_repo/AGENTS.md"
copy_file "$source_repo/.github/copilot-instructions.md" "$target_repo/.github/copilot-instructions.md"
copy_file "$source_repo/.github/agents/context-preserving-task.agent.md" "$target_repo/.github/agents/context-preserving-task.agent.md"
copy_file "$source_repo/.github/prompts/apply-cursor-agent-edits.prompt.md" "$target_repo/.github/prompts/apply-cursor-agent-edits.prompt.md"
copy_file "$source_repo/.github/prompts/general-autonomous-task.prompt.md" "$target_repo/.github/prompts/general-autonomous-task.prompt.md"

write_if_missing "$target_repo/agent-context/current-focus.md" "## Active Task\n\n- Task: TODO\n- Status: TODO\n- Branch: TODO\n\n## Top Next Steps\n\n- TODO\n\n## Highest-Risk Issue\n\n- TODO\n"
write_if_missing "$target_repo/agent-context/project-map.md" "## Repository Map\n\n- TODO\n"
write_if_missing "$target_repo/agent-context/claude-brief.md" "## Current Objective\n\nTODO\n"

mkdir -p "$target_repo/agent-context/tasks"

echo
echo "Portable agent bundle installed into: $target_repo"
echo "Next steps:"
echo "1. Review AGENTS.md and the custom agent text for repo-specific adjustments."
echo "2. In VS Code, select the Context-Preserving Task Agent for meaningful tasks."
echo "3. If handing work to Claude or another editor, start from AGENTS.md and agent-context/."

