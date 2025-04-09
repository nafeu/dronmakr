#!/usr/bin/env bash

set -e

VERSION_FILE="version.py"
DRY_RUN=false

# --- Parse options ---
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
      exit 1
      ;;
  esac
done

if [[ -z "$BUMP_TYPE" ]]; then
  echo "Usage: $0 [major|minor|patch] [--dry-run]"
  exit 1
fi

if [[ ! -f "$VERSION_FILE" ]]; then
  echo "Error: $VERSION_FILE not found."
  exit 1
fi

# --- Extract and bump version ---
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
  echo "[Dry Run] Skipping file update, commit, and tagging."
  exit 0
fi

# --- Update version.py ---
sed -i.bak -E "s/__version__\s*=\s*\"[0-9]+\.[0-9]+\.[0-9]+\"/__version__ = \"${new_version}\"/" "$VERSION_FILE"
rm "${VERSION_FILE}.bak"

# --- Git commit and tag ---
git add "$VERSION_FILE"
git commit -m "Bump version to v${new_version}"
git tag -a "v${new_version}" -m "Version ${new_version}"

# --- Push commit and tag ---
git push origin HEAD
git push origin "v${new_version}"

echo "✅ Bumped to v${new_version}, committed, tagged, and pushed!"
