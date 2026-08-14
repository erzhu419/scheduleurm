from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class EnvSmokeDeps:
    load_state: Callable[[], dict]
    notify: Callable[..., object]
    node_is_windows: Callable[[str], bool]
    apply_node_cmd_rewrites: Callable[[str, str], str]
    run_on: Callable[..., tuple]
    environ: dict = None


def smoke_test_envs(*, deps: EnvSmokeDeps) -> None:
    try:
        state = deps.load_state()
    except Exception as exc:
        deps.notify(
            "env_smoke_test_error",
            {"error": f"could not load state: {str(exc)[:100]}"},
            feishu_enabled=False,
        )
        return
    seen = set()
    failures = []
    for task in state.get("tasks", []):
        if task.get("status") not in ("queued", "launching", "running"):
            continue
        if task.get("auto_adopted"):
            continue
        spec = (task.get("env_spec") or "none").lower()
        if spec.startswith("docker"):
            continue
        raw_cmd = task.get("cmd", "") or ""
        probes: list[tuple[str, str]] = []
        target = task.get("node") or task.get("require_node") or task.get("preferred_node")
        if not target:
            continue
        if deps.node_is_windows(target):
            continue
        cmd = deps.apply_node_cmd_rewrites(target, raw_cmd)
        match_abs = re.search(r"(/[\w/.\-]+/python[\d.]*)\b", cmd)
        if match_abs:
            python_path = match_abs.group(1)
            probes.append((f"{shlex.quote(python_path)} -c 'print(\"ok\")'", python_path))
        match_conda = re.search(r"\bconda\s+run\s+(?:--no-capture-output\s+)?-n\s+(\S+)", cmd)
        if match_conda:
            envname = match_conda.group(1)
            probes.append((
                f"conda run -n {shlex.quote(envname)} python -c 'print(\"ok\")' 2>&1",
                f"conda:{envname}",
            ))
        if not probes:
            continue
        for probe_cmd, label in probes:
            key = (target, label)
            if key in seen:
                continue
            seen.add(key)
            try:
                rc, out, err = deps.run_on(target, probe_cmd, timeout=15, check=False)
            except Exception as exc:
                rc, out, err = 1, "", str(exc)
            if rc != 0:
                failures.append({"node": target, "env": label, "err": (err or out or "?")[:200]})
    if failures:
        deps.notify("env_smoke_test_failed", {"count": len(failures), "details": failures})
    else:
        deps.notify("env_smoke_test_passed", {"checked": len(seen)}, feishu_enabled=False)


def smoke_test_envs_enabled(environ=None) -> bool:
    env = os.environ if environ is None else environ
    return str(env.get("SCHEDULEURM_ENV_SMOKE_TEST", "")).lower() in (
        "1", "true", "yes", "on"
    )
