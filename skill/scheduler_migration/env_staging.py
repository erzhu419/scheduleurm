"""Python environment checks for ETA-based task migration."""

from __future__ import annotations

import re
import shlex

from .types import MigrationStagingDeps


def verify_python_env(task: dict, target_node: str, *, deps: MigrationStagingDeps) -> tuple[bool, str, bool]:
    cmd_str = deps.apply_node_cmd_rewrites(target_node, task.get("cmd") or "")
    py_match = re.search(r'(/[\w./-]+/python\d*(?:\.\d+)?)\b', cmd_str)
    py_path = py_match.group(1) if py_match else None
    if not py_path:
        return True, "", False
    try:
        rc, _, _ = deps.run_on(
            target_node,
            f"test -x {shlex.quote(py_path)}",
            timeout=5,
            check=False,
        )
        if rc != 0:
            return (
                False,
                f"python at {py_path} not executable on {target_node}; "
                f"deploy the conda env first (env_spec=conda:... if available)",
                True,
            )
    except Exception as exc:
        return False, f"env probe failed: {exc}", True
    return True, "", True
