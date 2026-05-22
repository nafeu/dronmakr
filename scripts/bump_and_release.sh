#!/usr/bin/env bash
# Bump semver in version.py (commit + tag + push), then create GitHub Release for that tag with gh CLI.
#
# Prerequisites: gh authenticated (`gh auth login`), same assumptions as bump_version.sh + gh_release_latest_tag.sh.
#
# Usage:
#   ./scripts/bump_and_release.sh patch
#   ./scripts/bump_and_release.sh minor --dry-run
#   ./scripts/bump_and_release.sh patch -- --draft --title "Smoke test"
#
# Extra args after -- are forwarded to `gh release create` (same as gh_release_latest_tag.sh).
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VERSION_FILE="version.py"
BUMP_TYPE=""
DRY_RUN=false
PASS_THROUGH=()

usage() {
  echo "Usage: $(basename "$0") [major|minor|patch] [--dry-run] [-- <gh-release-args>]"
  echo "Combines scripts/bump_version.sh and gh release create for the new tag."
}

while (($#)); do
  case "$1" in
    major | minor | patch)
      if [[ -n "$BUMP_TYPE" ]]; then
        echo "error: bump type already set to \"$BUMP_TYPE\"." >&2
        usage >&2
        exit 1
      fi
      BUMP_TYPE=$1
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --)
      shift
      PASS_THROUGH+=("$@")
      break
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "error: Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$BUMP_TYPE" ]]; then
  usage >&2
  exit 1
fi

if [[ ! -f "$VERSION_FILE" ]]; then
  echo "error: ${VERSION_FILE} not found under ${ROOT_DIR}" >&2
  exit 1
fi

current_version=$(sed -nE 's/^__version__ *= *"([0-9]+\.[0-9]+\.[0-9]+)"/\1/p' "$VERSION_FILE")
IFS='.' read -r major minor patch <<< "$current_version"

case "$BUMP_TYPE" in
  major)
    ((major++))
    minor=0
    patch=0
    ;;
  minor)
    ((minor++))
    patch=0
    ;;
  patch)
    ((patch++))
    ;;
esac

new_version="${major}.${minor}.${patch}"
NEW_TAG="v${new_version}"

echo "Bump type: ${BUMP_TYPE}"
echo "Current version: ${current_version}"
echo "Next version:   ${new_version}  (release tag ${NEW_TAG})"

if [[ "$DRY_RUN" == true ]]; then
  echo "[Dry run] Would run: scripts/bump_version.sh ${BUMP_TYPE}"
  CMD_PREVIEW=(gh release create "$NEW_TAG" --verify-tag --generate-notes --latest)
  if ((${#PASS_THROUGH[@]} > 0)); then
    for _arg in "${PASS_THROUGH[@]}"; do
      CMD_PREVIEW+=("$_arg")
    done
  fi
  printf "[Dry run] Would run:"
  printf " %q" "${CMD_PREVIEW[@]}"
  printf "\n"
  exit 0
fi

"$ROOT_DIR/scripts/bump_version.sh" "$BUMP_TYPE"

read_version="$(sed -nE 's/^__version__ *= *"([0-9]+\.[0-9]+\.[0-9]+)"/\1/p' "$VERSION_FILE")"
if [[ "$read_version" != "$new_version" ]]; then
  echo "error: expected version.py to be ${new_version} after bump; got \"${read_version}\"." >&2
  exit 1
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "error: Bump and push succeeded (${NEW_TAG}); gh is missing. Install GitHub CLI, then either:" >&2
  echo "  gh release create ${NEW_TAG} --verify-tag --generate-notes --latest" >&2
  echo "  or: scripts/gh_release_latest_tag.sh" >&2
  exit 1
fi

git fetch origin --tags

CURRENT="$(git symbolic-ref --short HEAD 2>/dev/null || true)"
HASH_HEAD="$(git rev-parse HEAD)"
HASH_TAG="$(git rev-parse "${NEW_TAG}^{}" 2>/dev/null || true)"
if [[ -z "$HASH_TAG" ]]; then
  echo "error: tag \"${NEW_TAG}\" not found after bump_version.sh." >&2
  exit 1
fi

if gh release view "$NEW_TAG" >/dev/null 2>&1; then
  echo "error: GitHub release for \"$NEW_TAG\" already exists (after bump)." >&2
  echo "  gh release view $NEW_TAG" >&2
  exit 1
fi

if [[ -n "${CURRENT:-}" && "$HASH_HEAD" != "$HASH_TAG" ]]; then
  echo "note: current branch (${CURRENT}) HEAD differs from ${NEW_TAG} commit; gh still targets the tagged commit." >&2
fi

CMD=(gh release create "$NEW_TAG" --verify-tag --generate-notes --latest)

if ((${#PASS_THROUGH[@]} > 0)); then
  for _arg in "${PASS_THROUGH[@]}"; do
    CMD+=("$_arg")
  done
fi

"${CMD[@]}"
echo "Done: released ${NEW_TAG} on GitHub."
