"""Shared rsync execution helpers for staging and result sync paths."""

from __future__ import annotations

import subprocess
from typing import Any, Callable


def _stream_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _trim_stream(value: Any, limit: int) -> str:
    return _stream_text(value).strip()[:limit]


def run_local_rsync(
    args: list[str],
    *,
    operation: str,
    run_subprocess: Callable[..., Any] = subprocess.run,
    timeout_s: int = 600,
    timeout_human: str | None = None,
    rc_operation: str | None = None,
    timeout_operation: str | None = None,
    exception_operation: str | None = None,
    failure_word: str = "",
    stderr_limit: int = 200,
    exception_limit: int = 200,
) -> tuple[bool, str]:
    """Run local rsync and format the legacy scheduler error message shape."""
    rc_name = rc_operation or operation
    timeout_name = timeout_operation or operation
    exception_name = exception_operation or operation
    timeout_text = timeout_human or f">{timeout_s}s"
    try:
        result = run_subprocess(args, capture_output=True, text=True, timeout=timeout_s)
        if result.returncode != 0:
            return (
                False,
                f"{rc_name}{failure_word} rc={result.returncode}: "
                f"{_trim_stream(getattr(result, 'stderr', ''), stderr_limit)}",
            )
        return True, ""
    except subprocess.TimeoutExpired:
        return False, f"{timeout_name} timeout ({timeout_text})"
    except Exception as exc:
        return False, f"{exception_name} exception: {str(exc)[:exception_limit]}"


def run_remote_rsync_command(
    *,
    node: str,
    command: str,
    run_on: Callable[..., tuple[int, str, str]],
    operation: str,
    timeout_s: int = 600,
    stderr_limit: int = 200,
    exception_limit: int = 200,
) -> tuple[bool, str]:
    """Run an rsync shell command on a relay/remote node with consistent errors."""
    try:
        rc, out, err = run_on(node, command, timeout=timeout_s, check=False)
        if rc != 0:
            return False, f"{operation} rc={rc}: {_trim_stream(err or out, stderr_limit)}"
        return True, ""
    except Exception as exc:
        return False, f"{operation} exception: {str(exc)[:exception_limit]}"
