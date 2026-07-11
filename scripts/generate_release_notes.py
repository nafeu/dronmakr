#!/usr/bin/env python3
"""Fill scripts/release_notes_template.md for a GitHub Release."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = Path(__file__).resolve().parent / "release_notes_template.md"
BUMP_COMMIT_RE = re.compile(r"^Bump version to v\d+\.\d+\.\d+$")


def run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def list_version_tags() -> list[str]:
    out = run_git("tag", "-l", "v*", "--sort=-version:refname")
    return [line.strip() for line in out.splitlines() if line.strip()]


def previous_tag(tag: str, tags: list[str]) -> str | None:
    try:
        index = tags.index(tag)
    except ValueError:
        return None
    if index + 1 >= len(tags):
        return None
    return tags[index + 1]


def strip_version(tag: str) -> str:
    return tag[1:] if tag.startswith("v") else tag


def changelog_entries(previous: str | None, tag: str, *, head: bool = False) -> list[str]:
    if head and previous:
        rev_range = f"{previous}..HEAD"
    elif previous:
        rev_range = f"{previous}..{tag}"
    else:
        rev_range = tag if not head else "HEAD"
    try:
        raw = run_git(
            "log",
            rev_range,
            "--pretty=format:%h|%s",
            "--no-merges",
            "--reverse",
        )
    except subprocess.CalledProcessError:
        return []

    entries: list[str] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        commit_hash, subject = line.split("|", 1)
        if BUMP_COMMIT_RE.match(subject.strip()):
            continue
        entries.append(f"- {subject.strip()} (`{commit_hash}`)")
    return entries


def render_changelog(previous: str | None, tag: str, *, head: bool = False) -> str:
    entries = changelog_entries(previous, tag, head=head)
    if not entries:
        if previous:
            return "_No user-facing changes since {previous}._".format(previous=previous)
        return "_Initial release._"
    header = ""
    if previous:
        header = f"Changes since **{previous}**:\n\n"
    return header + "\n".join(entries)


def render_notes(
    tag: str,
    *,
    previous: str | None = None,
    allow_missing_tag: bool = False,
) -> str:
    if not TEMPLATE.is_file():
        raise FileNotFoundError(f"Missing release notes template: {TEMPLATE}")

    tags = list_version_tags()
    tag_exists = tag in tags
    if not tag_exists and not allow_missing_tag:
        raise ValueError(
            f"Tag {tag!r} not found locally. Fetch tags or pass --allow-missing-tag for previews."
        )

    prev = previous if previous is not None else (previous_tag(tag, tags) if tag_exists else None)
    if not tag_exists and prev is None and tags:
        prev = tags[0]

    version = strip_version(tag)
    changelog = render_changelog(prev, tag, head=not tag_exists)

    text = TEMPLATE.read_text(encoding="utf-8")
    return (
        text.replace("{{VERSION}}", version)
        .replace("{{CHANGELOG}}", changelog)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="Release tag (e.g. v1.2.3)")
    parser.add_argument(
        "--previous-tag",
        help="Previous release tag for the changelog range (default: prior semver tag)",
    )
    parser.add_argument(
        "--allow-missing-tag",
        action="store_true",
        help="Preview notes for a tag not created yet (changelog uses previous-tag..HEAD)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write notes to this file (default: stdout)",
    )
    args = parser.parse_args()

    try:
        notes = render_notes(
            args.tag,
            previous=args.previous_tag,
            allow_missing_tag=args.allow_missing_tag,
        )
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.output:
        args.output.write_text(notes, encoding="utf-8")
        print(f"Wrote release notes to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(notes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
