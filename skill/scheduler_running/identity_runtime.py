"""Runtime wrappers for running task identity helpers."""

from __future__ import annotations

import os
import socket
import time
from typing import Any, Mapping


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_running_identity_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    def _task_pids(task):
        pids = task.get("remote_pids")
        if pids:
            return list(pids)
        pid = task.get("remote_pid")
        return [pid] if pid else []

    def _task_process_groups(task):
        groups = set()
        pg = task.get("process_group")
        try:
            if pg and int(pg) > 1:
                groups.add(int(pg))
        except (TypeError, ValueError):
            pass
        if not task.get("auto_adopted"):
            for pid in _ns(namespace, "_task_pids")(task):
                try:
                    if int(pid) > 1:
                        groups.add(int(pid))
                except (TypeError, ValueError):
                    continue
        return sorted(groups)

    def _set_current_usage(task, vram_mb=0, ram_mb=0, pcpu=0.0):
        task["current_vram_mb"] = int(vram_mb or 0)
        task["current_ram_mb"] = int(ram_mb or 0)
        task["current_pcpu"] = float(pcpu or 0.0)
        if int(vram_mb or 0) > 0:
            task.pop("vram_estimation_source", None)
            task.pop("vram_estimation_note", None)

    def _format_mem_gb(mb, *, approx: bool = False) -> str:
        return _ns(namespace, "_format_mem_gb_impl")(mb, approx=approx)

    def _format_node_ram_summary(node: dict) -> str:
        return _ns(namespace, "_format_node_ram_summary_impl")(node)

    def _local_user() -> str:
        try:
            import getpass
            return getpass.getuser()
        except Exception:
            return os.environ.get("USER") or "unknown"

    def _local_host_short() -> str:
        try:
            return (socket.gethostname() or "host").split(".", 1)[0]
        except Exception:
            return "host"

    def _actor_info(action: str = "", reason: str = "") -> dict:
        try:
            scheduler_id = _ns(namespace, "_ClaimManager").scheduler_id()
        except Exception:
            scheduler_id = ""
        user = _ns(namespace, "_local_user")()
        host = _ns(namespace, "_local_host_short")()
        return {
            "user": user,
            "host": host,
            "label": f"{user}@{host}",
            "scheduler_id": scheduler_id,
            "action": action,
            "reason": str(reason or "")[:500],
            "ts": time.time(),
        }

    def _record_task_kill_actor(task: dict, action: str, reason: str) -> dict:
        actor = _ns(namespace, "_actor_info")(action, reason)
        task["last_killed_at"] = actor["ts"]
        task["last_killed_by"] = actor["label"]
        task["last_kill_actor"] = actor
        task["last_kill_action"] = action
        task["last_kill_reason"] = str(reason or "")[:500]
        return actor

    def _format_task_owner(task: dict) -> str:
        user = (
            task.get("submitted_by")
            or task.get("process_owner")
            or task.get("owner")
            or _ns(namespace, "_local_user")()
        )
        origin = task.get("origin") or ""
        sid = task.get("scheduler_id") or task.get("submitted_scheduler_id") or ""
        try:
            own_sid = _ns(namespace, "_ClaimManager").scheduler_id()
        except Exception:
            own_sid = ""
        if task.get("auto_adopted") or origin in ("external", "manual-adopt"):
            tag = "external?" if origin != "manual-adopt" else "manual"
        elif sid and own_sid and sid != own_sid:
            tag = "other-sched?"
        else:
            tag = "this"
        return f"{user}:{tag}"

    def _kill_task_processes(task, timeout=15):
        return _ns(namespace, "_BACKEND").kill(task, timeout=timeout)

    def _kill_task_processes_batch(tasks, timeout=15):
        return _ns(namespace, "_BACKEND").kill_many(tasks, timeout=timeout)

    def check_running(task):
        fake_state = {"tasks": [task]}
        res = _ns(namespace, "_BACKEND").batch_probe(fake_state).get(task["id"])
        return _ns(namespace, "_check_running_probe_result_impl")(
            task,
            res,
            set_current_usage=_ns(namespace, "_set_current_usage"),
        )

    return {
        "_task_pids": _task_pids,
        "_task_process_groups": _task_process_groups,
        "_set_current_usage": _set_current_usage,
        "_format_mem_gb": _format_mem_gb,
        "_format_node_ram_summary": _format_node_ram_summary,
        "_local_user": _local_user,
        "_local_host_short": _local_host_short,
        "_actor_info": _actor_info,
        "_record_task_kill_actor": _record_task_kill_actor,
        "_format_task_owner": _format_task_owner,
        "_kill_task_processes": _kill_task_processes,
        "_kill_task_processes_batch": _kill_task_processes_batch,
        "check_running": check_running,
    }
