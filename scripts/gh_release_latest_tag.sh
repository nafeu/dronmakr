#!/usr/bin/env bash
# Create a GitHub Release for the latest git tag using gh CLI (templated release notes).
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
#   scripts/gh_release_latest_tag.sh -- --draft --title "Smoke test build"
#
# Env:
#   FETCH_TAGS=0 — skip "git fetch origin --tags"

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "error: run this from inside the dronmakr git repository." >&2
  exit 1
}
cd "$REPO_ROOT"

write_release_notes() {
  local tag=$1
  local outfile
  outfile=$(mktemp "${TMPDIR:-/tmp}/dronmakr-release-notes.XXXXXX")
  if ! python3 "$REPO_ROOT/scripts/generate_release_notes.py" --tag "$tag" -o "$outfile" 2>/dev/null; then
    rm -f "$outfile"
    return 1
  fi
  echo "$outfile"
}

filter_gh_release_args() {
  local filtered=()
  local arg
  if ((${#PASS_THROUGH[@]} > 0)); then
    for arg in "${PASS_THROUGH[@]}"; do
      [[ "$arg" == "--generate-notes" ]] && continue
      filtered+=("$arg")
    done
  fi
  if ((${#filtered[@]} > 0)); then
    PASS_THROUGH=("${filtered[@]}")
  else
    PASS_THROUGH=()
  fi
}

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
  NOTES_FILE=$(write_release_notes "$TAG")
  echo "[Dry run] Release notes (${NOTES_FILE}):"
  cat "$NOTES_FILE"
  rm -f "$NOTES_FILE"
  filter_gh_release_args
  CMD=(gh release create "$TAG" --verify-tag --notes-file /tmp/dronmakr-release-notes.XXXXXX --latest)
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

filter_gh_release_args
NOTES_FILE=$(write_release_notes "$TAG")
trap 'rm -f "$NOTES_FILE"' EXIT

CMD=(gh release create "$TAG" --verify-tag --notes-file "$NOTES_FILE" --latest)
if ((${#PASS_THROUGH[@]} > 0)); then
  for _arg in "${PASS_THROUGH[@]}"; do
    CMD+=("$_arg")
  done
fi

exec "${CMD[@]}"
