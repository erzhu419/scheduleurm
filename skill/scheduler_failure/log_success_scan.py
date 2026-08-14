"""Full-log success marker scans for terminal task diagnosis."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class FullLogSuccessScanDeps:
    node_configs: dict
    node_is_windows: Callable[[str], bool]
    scan_windows_log_for_patterns: Callable[[dict, list], list]
    success_patterns_for_task: Callable[[dict], list]
    run_on: Callable[..., tuple]


def scan_full_log_for_success(task: dict, *, deps: FullLogSuccessScanDeps) -> list:
    """Scan the entire task log for success markers."""
    log_path = task.get("log_path")
    if not log_path or task.get("auto_adopted"):
        return []
    node = task.get("node")
    try:
        if node and deps.node_configs.get(node, {}).get("host") is None:
            log_file = Path(log_path)
            if not log_file.exists():
                return []
            patterns = deps.success_patterns_for_task(task)
            seen = set()
            with open(log_file, "rb") as fh:
                buf = b""
                while True:
                    chunk = fh.read(262144)
                    if not chunk:
                        break
                    buf = buf[-256:] + chunk
                    text = buf.decode("utf-8", errors="replace")
                    for pattern in patterns:
                        if pattern not in seen and pattern in text:
                            seen.add(pattern)
                    if len(seen) == len(patterns):
                        break
            return list(seen)
        if node and deps.node_is_windows(node):
            return deps.scan_windows_log_for_patterns(task, deps.success_patterns_for_task(task))
        if node:
            patterns = deps.success_patterns_for_task(task)
            grep_es = " ".join(f"-e {shlex.quote(pattern)}" for pattern in patterns)
            rc, out, _ = deps.run_on(
                node,
                f"grep -F -m 1 {grep_es} {shlex.quote(log_path)} 2>/dev/null || true",
                timeout=15,
                check=False,
            )
            if rc != 0 or not out:
                return []
            matched_text = out.strip()
            if not matched_text:
                return []
            return [pattern for pattern in patterns if pattern in matched_text]
    except Exception:
        return []
    return []
