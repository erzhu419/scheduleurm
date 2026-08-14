"""Launch command normalization helpers."""

from __future__ import annotations

import re


_PYTHON_TOKEN_RE = re.compile(
    r'(^|[\s;&|`(])'
    r'((?:[A-Za-z0-9_./~-]+/)?'
    r'(?:python|python3)(?:\d+\.\d+)?)'
    r'(\s+)'
)


def inject_python_u(cmd: str) -> str:
    """Insert ``-u`` after python invocation tokens that do not already have it."""
    if not cmd:
        return cmd

    def _has_u_already(rest: str) -> bool:
        first = rest.lstrip().split(None, 1)[0] if rest.strip() else ""
        return first == "-u" or (first.startswith("-u") and not first.startswith("--"))

    out_parts = []
    last_end = 0
    for match in _PYTHON_TOKEN_RE.finditer(cmd):
        _start, end = match.span()
        if _has_u_already(cmd[end:]):
            continue
        out_parts.append(cmd[last_end:end])
        out_parts.append("-u ")
        last_end = end
    if not out_parts:
        return cmd
    out_parts.append(cmd[last_end:])
    return "".join(out_parts)
