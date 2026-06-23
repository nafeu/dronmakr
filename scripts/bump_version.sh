#!/usr/bin/env bash
# Bump semver across backend/dronmakr/version.py, package.json, and Tauri metadata
# (src-tauri/tauri.conf.json, src-tauri/Cargo.toml), then commit + annotated tag + push.
# To bump and immediately create the GitHub Release for the new tag, use scripts/bump_and_release.sh.
#
# Examples:
#   ./scripts/bump_version.sh patch
#   ./scripts/bump_version.sh minor --dry-run
#
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VERSION_FILE="backend/dronmakr/version.py"
VERSION_FILES=(
  "$VERSION_FILE"
  "package.json"
  "src-tauri/tauri.conf.json"
  "src-tauri/Cargo.toml"
)
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

for file in "${VERSION_FILES[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "error: ${file} not found under ${ROOT_DIR}" >&2
    exit 1
  fi
done

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
  echo "[Dry run] Would update:"
  for file in "${VERSION_FILES[@]}"; do
    echo "  - ${file}"
  done
  echo "[Dry run] Then commit, tag v${new_version}, and push to origin."
  exit 0
fi

sync_semver_in_file() {
  local file=$1
  case "$file" in
    backend/dronmakr/version.py)
      sed -i.bak -E "s/^(__version__ *= *\")[0-9]+\.[0-9]+\.[0-9]+(\".*)/\1${new_version}\2/" "$file"
      ;;
    package.json|src-tauri/tauri.conf.json)
      sed -i.bak -E "s/(\"version\"[[:space:]]*:[[:space:]]*\")[0-9]+\.[0-9]+\.[0-9]+(\")/\1${new_version}\2/" "$file"
      ;;
    src-tauri/Cargo.toml)
      sed -i.bak -E "s/^(version = \")[0-9]+\.[0-9]+\.[0-9]+(\")/\1${new_version}\2/" "$file"
      ;;
    *)
      echo "error: no semver sync rule for ${file}" >&2
      return 1
      ;;
  esac
  rm -f "${file}.bak"
}

for file in "${VERSION_FILES[@]}"; do
  sync_semver_in_file "$file"
done

git add "${VERSION_FILES[@]}"
git commit -m "Bump version to v${new_version}"
git tag -a "v${new_version}" -m "Version ${new_version}"

git push origin HEAD
git push origin "v${new_version}"

echo "Done: bumped to v${new_version}, committed, tagged, and pushed."
