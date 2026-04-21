import json
import re

VALID_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def _collect_root_key_line_numbers(file_path: str) -> dict[str, int]:
    """Best-effort map of root-level JSON keys to source line numbers."""
    key_lines: dict[str, int] = {}
    with open(file_path, "r") as f:
        text = f.read()

    line = 1
    i = 0
    n = len(text)
    depth = 0

    while i < n:
        ch = text[i]

        if ch == "\n":
            line += 1
            i += 1
            continue

        if ch.isspace():
            i += 1
            continue

        if ch == "{":
            depth += 1
            i += 1
            continue

        if ch == "}":
            depth = max(0, depth - 1)
            i += 1
            continue

        if ch == '"' and depth == 1:
            key_line = line
            i += 1
            key_chars = []
            while i < n:
                c = text[i]
                if c == "\\" and i + 1 < n:
                    key_chars.append(text[i : i + 2])
                    i += 2
                    continue
                if c == '"':
                    i += 1
                    break
                if c == "\n":
                    line += 1
                key_chars.append(c)
                i += 1

            key_text = "".join(key_chars)
            while i < n and text[i].isspace():
                if text[i] == "\n":
                    line += 1
                i += 1
            if i < n and text[i] == ":":
                try:
                    decoded_key = json.loads(f'"{key_text}"')
                except Exception:
                    decoded_key = key_text
                key_lines[decoded_key] = key_line
            continue

        i += 1

    return key_lines


def _validate_root_object_names(file_path: str, label: str) -> list[str]:
    with open(file_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        return [
            f"- {label}: root JSON value must be an object with named entries.",
        ]

    line_map = _collect_root_key_line_numbers(file_path)
    errors: list[str] = []
    for key in data.keys():
        if not isinstance(key, str) or not VALID_NAME_RE.fullmatch(key):
            line_info = line_map.get(key)
            where = f"line {line_info}" if line_info else "unknown line"
            errors.append(
                f"- {label}: invalid name '{key}' at {where} (must be lowercase kebab-case: letters/numbers with '-' only)."
            )
    return errors


def validate_server_config_names() -> None:
    """
    Validate root-level names in config files used by the web UI.
    Raises ValueError with actionable, line-specific messages on failure.
    """
    validation_targets = [
        ("config/beat-patterns.json", "config/beat-patterns.json"),
        ("config/drum-kits.json", "config/drum-kits.json"),
    ]
    errors: list[str] = []

    for path, label in validation_targets:
        try:
            errors.extend(_validate_root_object_names(path, label))
        except FileNotFoundError:
            # These files may be created during startup by ensure_* helpers.
            continue
        except json.JSONDecodeError as e:
            errors.append(
                f"- {label}: invalid JSON at line {e.lineno}, column {e.colno}: {e.msg}"
            )

    if errors:
        raise ValueError(
            "Configuration validation failed. Fix the following entries manually before launching the server again:\n"
            + "\n".join(errors)
        )
