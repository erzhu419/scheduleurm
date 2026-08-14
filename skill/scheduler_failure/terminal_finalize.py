"""Finalize local terminal tasks using wrapper and redirected logs."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class TerminalFinalizeDeps:
    state_dir: Any
    node_is_windows: Callable[[str], bool]
    is_local_node: Callable[[str], bool]
    run_on: Callable[..., tuple[int, str, str]]
    diagnose_terminal: Callable[[dict], dict]
    requeue_after_crash: Callable[[dict, dict], str | None]
    record_result_artifacts: Callable[[dict], Any]
    claim_enabled_for: Callable[[str], bool]
    release_task_claims_and_intents: Callable[..., Any]
    set_current_usage: Callable[[dict, int, int, float], Any]
    now: Callable[[], float]


def terminal_log_path(task_id: str, node: str, deps: TerminalFinalizeDeps) -> str:
    if deps.is_local_node(node):
        return f"{deps.state_dir}/logs/{task_id}.log"
    return f"/tmp/sched_{task_id}.log"


def _probe_size(node: str, path: str, deps: TerminalFinalizeDeps) -> int:
    """Return file size on node, or 0 on missing file / probe failure."""
    try:
        if deps.is_local_node(node):
            local_path = Path(path)
            return local_path.stat().st_size if local_path.exists() else 0
        rc, out, _ = deps.run_on(
            node,
            f"wc -c < {shlex.quote(path)} 2>/dev/null",
            timeout=5,
            check=False,
        )
        if rc != 0:
            return 0
        try:
            return int((out or "0").strip())
        except ValueError:
            return 0
    except Exception:
        return 0


def _redirect_target(cmd: str) -> str | None:
    cmd_has_own_redirect = bool(
        re.search(r"(?<!\d)(?:&>|>>|2>&1|>&|>)\s*[^\s|;&)]+", cmd)
    ) or "2>&1" in cmd
    if not cmd_has_own_redirect:
        return None
    match = re.search(r"(?:^|\s)(?:&>|>)\s*([^\s|;&)<>]+)", cmd)
    if not match:
        return None
    return match.group(1)


def try_finalize_terminal_local_task(
    task: dict,
    node: str,
    state: dict,
    *,
    deps: TerminalFinalizeDeps,
) -> bool:
    """Finalize a local launching task when only terminal log evidence remains."""
    if deps.node_is_windows(node):
        return False
    task_id = task.get("id") or ""
    if not task_id:
        return False

    log_path = terminal_log_path(task_id, node, deps)
    log_size = _probe_size(node, log_path, deps)
    if log_size <= 0:
        real_log = _redirect_target(task.get("cmd") or "")
        if not real_log:
            return False
        real_size = _probe_size(node, real_log, deps)
        if real_size <= 0:
            return False

    task["log_path"] = log_path
    task["finished_at"] = deps.now()
    task["started_at"] = task.get("launching_started_at") or task["finished_at"]
    task.pop("launching_started_at", None)
    task.pop("launch_token", None)
    task["remote_pids"] = []
    task["alive_pids"] = []
    task["peak_vram_mb"] = 0
    task["peak_ram_mb"] = 0
    deps.set_current_usage(task, 0, 0, 0.0)

    diag = deps.diagnose_terminal(task)
    task["_diagnosis"] = diag
    if diag.get("is_crash"):
        task["status"] = "failed"
        task["last_block_reason"] = (
            f"WAL recovery: terminal orphan local task on {node}; diagnosed "
            f"crash: {(diag.get('reason') or '')[:120]}"
        )
        new_id = deps.requeue_after_crash(task, state)
        if new_id:
            task["requeued_as"] = new_id
    else:
        task["status"] = "done"
        task["last_block_reason"] = (
            f"WAL recovery: terminal orphan local task on {node}; diagnosed "
            f"clean exit ({(diag.get('reason') or 'no signal')[:120]})"
        )

    try:
        deps.record_result_artifacts(task)
    except Exception:
        pass

    if deps.claim_enabled_for(node):
        try:
            deps.release_task_claims_and_intents(task, extra_nodes=[node])
        except Exception:
            pass
    return True
