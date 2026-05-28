#!/usr/bin/env bash
# Generate GitHub release notes for PREV_TAG..NEW_TAG using the Cursor agent CLI.
#
# Usage:
#   scripts/generate_release_notes.sh v0.47.14 v0.47.15
#   scripts/generate_release_notes.sh v0.47.14 v0.47.15 > /tmp/notes.md
#
# Requires: git, and `agent` or `cursor agent` logged in (cursor agent login).
# Env:
#   CURSOR_AGENT_MODEL — optional model slug passed to --model
#
set -euo pipefail

PREV_TAG="${1:?usage: generate_release_notes.sh PREV_TAG NEW_TAG}"
NEW_TAG="${2:?usage: generate_release_notes.sh PREV_TAG NEW_TAG}"

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "error: run from inside the dronmakr git repository." >&2
  exit 1
}
cd "$REPO_ROOT"

if ! git rev-parse "${PREV_TAG}^{commit}" >/dev/null 2>&1; then
  echo "error: previous tag \"${PREV_TAG}\" not found." >&2
  exit 1
fi
if ! git rev-parse "${NEW_TAG}^{commit}" >/dev/null 2>&1; then
  echo "error: new tag \"${NEW_TAG}\" not found." >&2
  exit 1
fi

GIT_LOG="$(git log "${PREV_TAG}..${NEW_TAG}" --pretty=format:'- %s (%h)' --no-merges 2>/dev/null || true)"
if [[ -z "$GIT_LOG" ]]; then
  echo "error: no commits between ${PREV_TAG} and ${NEW_TAG}." >&2
  exit 1
fi

DIFF_STAT="$(git diff "${PREV_TAG}..${NEW_TAG}" --stat 2>/dev/null | tail -n 30 || true)"

run_cursor_agent() {
  local prompt=$1
  local -a agent_args=(
    --print
    --mode ask
    --trust
    --output-format text
    --workspace "$REPO_ROOT"
  )
  if [[ -n "${CURSOR_AGENT_MODEL:-}" ]]; then
    agent_args+=(--model "$CURSOR_AGENT_MODEL")
  fi

  if command -v agent >/dev/null 2>&1; then
    agent "${agent_args[@]}" "$prompt"
  elif command -v cursor >/dev/null 2>&1; then
    cursor agent "${agent_args[@]}" "$prompt"
  else
    echo "error: Cursor agent CLI not found (install Cursor CLI or ensure \`agent\` is on PATH)." >&2
    return 127
  fi
}

PROMPT="$(cat <<EOF
Write GitHub release notes for dronmakr ${NEW_TAG}.

Changes since ${PREV_TAG}:

Commits:
${GIT_LOG}

Files changed (summary):
${DIFF_STAT:- (no file diff available)}

Requirements:
- Output ONLY markdown release notes (no preamble, no wrapping code fences)
- Start with one short introductory sentence for the release
- Follow with a concise bullet list of user-facing changes
- Group related changes when helpful
- Omit version-bump-only commits (e.g. "Bump version to v…")
- Keep each bullet to one line when possible
- Use clear past-tense verbs (Fixed, Added, Updated, Improved)
EOF
)"

run_cursor_agent "$PROMPT"
