#!/usr/bin/env bash
# Create a GitHub Release for the latest git tag using gh CLI (--generate-notes).
# Publishing triggers .github/workflows/release-desktop.yml when the workflow is enabled on the repo.
# To bump version, push the new tag, and create that release in one step, use scripts/bump_and_release.sh.
#
# Prerequisites:
#   - gh authenticated (gh auth login), git, tags pushed to GitHub (--verify-tag)
#
# Usage:
#   scripts/gh_release_latest_tag.sh
#   scripts/gh_release_latest_tag.sh --no-fetch
#   scripts/gh_release_latest_tag.sh --dry-run
#   scripts/gh_release_latest_tag.sh -- --draft --notes "Smoke test build"
#
# Env:
#   FETCH_TAGS=0 — skip "git fetch origin --tags"

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "error: run this from inside the dronmakr git repository." >&2
  exit 1
}
cd "$REPO_ROOT"

FETCH_TAGS="${FETCH_TAGS:-1}"
NO_FETCH=false
DRY_RUN=false
PASS_THROUGH=()

while (($#)); do
  case "$1" in
    --no-fetch) NO_FETCH=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    --)
      shift
      PASS_THROUGH+=("$@")
      break
      ;;
    *)
      PASS_THROUGH+=("$1")
      shift
      ;;
  esac
done

if ! command -v gh >/dev/null 2>&1; then
  echo "error: gh (GitHub CLI) is not installed or not on PATH." >&2
  exit 1
fi

if [[ "$NO_FETCH" == false && "$FETCH_TAGS" != "0" ]]; then
  echo "Fetching tags from origin…"
  git fetch origin --tags
fi

TAG="$(git tag -l --sort=-version:refname | head -n1)" || true
if [[ -z "${TAG:-}" ]]; then
  echo "error: no tags found locally." >&2
  exit 1
fi

if gh release view "$TAG" >/dev/null 2>&1; then
  echo "error: GitHub release for tag \"$TAG\" already exists." >&2
  exit 1
fi

echo "Latest tag: $TAG"

if [[ "$DRY_RUN" == true ]]; then
  CMD=(gh release create "$TAG" --verify-tag --generate-notes --latest)
  if ((${#PASS_THROUGH[@]} > 0)); then
    for _arg in "${PASS_THROUGH[@]}"; do
      CMD+=("$_arg")
    done
  fi
  printf "Would run:"
  printf " %q" "${CMD[@]}"
  printf "\n"
  exit 0
fi

CMD=(gh release create "$TAG" --verify-tag --generate-notes --latest)
if ((${#PASS_THROUGH[@]} > 0)); then
  for _arg in "${PASS_THROUGH[@]}"; do
    CMD+=("$_arg")
  done
fi

exec "${CMD[@]}"
