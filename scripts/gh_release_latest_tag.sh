#!/usr/bin/env bash
# Create a GitHub Release for the latest git tag (semver-sorted) using gh CLI.
# Publishing triggers .github/workflows/release-desktop.yml when the workflow is enabled on the repo.
#
# Prerequisites: gh authenticated (gh auth login), git, tags pushed to GitHub (--verify-tag).
#
# Usage:
#   scripts/gh_release_latest_tag.sh
#   scripts/gh_release_latest_tag.sh --no-fetch
#   scripts/gh_release_latest_tag.sh --dry-run
#   scripts/gh_release_latest_tag.sh -- --draft --notes "Smoke test build"
#
# Env:
#   FETCH_TAGS=0  — skip "git fetch origin --tags"

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

# Newest tag by version refname (--sort=-version:refname is Git-aware semver-ish ordering).
TAG="$(git tag -l --sort=-version:refname | head -n1)" || true
if [[ -z "${TAG:-}" ]]; then
  echo "error: no tags found locally. Create and push a tag first, e.g. git tag v0.46.0 && git push origin v0.46.0" >&2
  exit 1
fi

CURRENT="$(git symbolic-ref --short HEAD 2>/dev/null || true)"
HASH_TAG="$(git rev-parse "$TAG^{}" 2>/dev/null || true)"
HASH_HEAD="$(git rev-parse HEAD)"

if gh release view "$TAG" >/dev/null 2>&1; then
  echo "error: GitHub release for tag \"$TAG\" already exists. View: gh release view $TAG" >&2
  exit 1
fi

echo "Latest tag by version sort: $TAG"
if [[ -n "${CURRENT:-}" && "$HASH_HEAD" != "$HASH_TAG" ]]; then
  echo "note: current branch HEAD ($CURRENT) differs from tag $TAG; the release metadata still points at the tag commit." >&2
fi

CMD=(gh release create "$TAG" --verify-tag --generate-notes --latest)
# Bash 3.2 + set -u: "${empty[@]}" is an unbound-variable error — only append when non-empty.
if ((${#PASS_THROUGH[@]} > 0)); then
  for _arg in "${PASS_THROUGH[@]}"; do
    CMD+=("$_arg")
  done
fi

if [[ "$DRY_RUN" == true ]]; then
  printf "Would run:"
  printf " %q" "${CMD[@]}"
  printf "\n"
  exit 0
fi

exec "${CMD[@]}"
