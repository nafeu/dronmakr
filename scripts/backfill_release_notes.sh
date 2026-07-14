#!/usr/bin/env bash
# Retroactively set GitHub Release bodies from scripts/release_notes_template.md.
#
# Usage:
#   ./scripts/backfill_release_notes.sh              # default: v0.58.* releases only
#   ./scripts/backfill_release_notes.sh 0.58.        # explicit semver prefix (no leading v)
#   ./scripts/backfill_release_notes.sh 0.58. --dry-run
#
# Only updates releases that already exist on GitHub (does not create new releases).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PREFIX="0.58."
DRY_RUN=false

usage() {
  echo "Usage: $(basename "$0") [semver-prefix] [--dry-run]"
  echo "Default prefix: 0.58. (skips 0.4x and other lines)"
}

while (($#)); do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      PREFIX="${1%/}."
      shift
      ;;
  esac
done

if ! command -v gh >/dev/null 2>&1; then
  echo "error: gh (GitHub CLI) is not installed or not on PATH." >&2
  exit 1
fi

git fetch origin --tags

RELEASE_TAGS=()
while IFS= read -r tag; do
  [[ -n "$tag" ]] && RELEASE_TAGS+=("$tag")
done < <(
  gh release list --limit 500 --json tagName --jq '.[].tagName' \
    | grep -E "^v${PREFIX//./\\.}[0-9]+$" \
    | sort -V
)

if ((${#RELEASE_TAGS[@]} == 0)); then
  echo "No GitHub releases found matching v${PREFIX}*"
  exit 0
fi

echo "Updating ${#RELEASE_TAGS[@]} release(s) matching v${PREFIX}*:"
printf '  %s\n' "${RELEASE_TAGS[@]}"

for tag in "${RELEASE_TAGS[@]}"; do
  notes_file=$(mktemp "${TMPDIR:-/tmp}/dronmakr-release-notes.XXXXXX")
  if ! python3 "$ROOT_DIR/scripts/generate_release_notes.py" --tag "$tag" -o "$notes_file" 2>/dev/null; then
    rm -f "$notes_file"
    echo "error: failed to generate notes for ${tag}" >&2
    exit 1
  fi

  if [[ "$DRY_RUN" == true ]]; then
    echo ""
    echo "=== ${tag} (dry run) ==="
    cat "$notes_file"
    rm -f "$notes_file"
    continue
  fi

  gh release edit "$tag" --notes-file "$notes_file"
  rm -f "$notes_file"
  echo "Updated ${tag}"
done

echo "Done."
