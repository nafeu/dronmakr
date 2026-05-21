#!/usr/bin/env bash
# Bump __version__ in version.py (semver), git commit + annotated tag + push origin.
#
# Examples:
#   ./scripts/bump_version.sh patch
#   ./scripts/bump_version.sh minor --dry-run
#
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VERSION_FILE="version.py"
DRY_RUN=false

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    major|minor|patch)
      BUMP_TYPE=$1
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $(basename "$0") [major|minor|patch] [--dry-run]"
      exit 1
      ;;
  esac
done

if [[ -z "${BUMP_TYPE:-}" ]]; then
  echo "Usage: $(basename "$0") [major|minor|patch] [--dry-run]"
  exit 1
fi

if [[ ! -f "$VERSION_FILE" ]]; then
  echo "error: ${VERSION_FILE} not found under ${ROOT_DIR}"
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

echo "Current version: ${current_version}"
echo "New version:     ${new_version}"

if [ "$DRY_RUN" = true ]; then
  echo "[Dry run] Would update ${VERSION_FILE}, commit, tag v${new_version}, and push to origin."
  exit 0
fi

sed -i.bak -E "s/^(__version__ *= *\")[0-9]+\.[0-9]+\.[0-9]+(\".*)/\1${new_version}\2/" "$VERSION_FILE"
rm "${VERSION_FILE}.bak"

git add "$VERSION_FILE"
git commit -m "Bump version to v${new_version}"
git tag -a "v${new_version}" -m "Version ${new_version}"

git push origin HEAD
git push origin "v${new_version}"

echo "Done: bumped to v${new_version}, committed, tagged, and pushed."
