from __future__ import annotations

import math
import shlex
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .local_launch import runtime_exit_status_path


@dataclass(frozen=True)
class LocalOrphanRecoveryDeps:
    state_dir: Any
    node_configs: dict
    node_is_windows: Callable[[str], bool]
    run_on: Callable[..., tuple[int, str, str]]
    remember_last_placement: Callable[[dict], Any]
    set_current_usage: Callable[[dict, int, int, float], Any]
    claim_enabled_for: Callable[[str], bool]
    update_claim_pid: Callable[[str, str, int], Any]
    requires_local_capacity_check: Callable[[str, Optional[dict]], bool]
    now: Callable[[], float]
    recover_remote_nodes: bool = True


def _scheduleurm_process_inventory_command() -> str:
    script = r'''
import os
import sys

def decode(value):
    try:
        return value.decode("utf-8", "replace")
    except AttributeError:
        return value

for name in os.listdir("/proc"):
    if not name.isdigit():
        continue
    pid = int(name)
    base = "/proc/{}".format(pid)
    try:
        with open("{}/environ".format(base), "rb") as handle:
            fields = handle.read().split(b"\0")
    except (IOError, OSError):
        continue
    values = {}
    for field in fields:
        if b"=" not in field:
            continue
        key, value = field.split(b"=", 1)
        if key in (b"SCHEDULEURM_TASK_ID", b"SCHEDULEURM_LAUNCH_TOKEN"):
            values[key] = decode(value)
    task_id = values.get(b"SCHEDULEURM_TASK_ID", "")
    if not task_id:
        continue
    try:
        sid = os.getsid(pid)
        pgid = os.getpgid(pid)
        with open("{}/status".format(base), "rb") as handle:
            status_lines = handle.readlines()
        with open("{}/stat".format(base), "rb") as handle:
            stat_tail = handle.read().rsplit(b")", 1)[1].split()
    except (IOError, OSError, IndexError):
        continue
    state = ""
    rss_mb = 0
    for line in status_lines:
        if line.startswith(b"State:"):
            parts = line.split()
            state = decode(parts[1]) if len(parts) > 1 else ""
        elif line.startswith(b"VmRSS:"):
            parts = line.split()
            rss_mb = int(parts[1]) // 1024 if len(parts) > 1 else 0
    start_ticks = decode(stat_tail[19]) if len(stat_tail) > 19 else "0"
    launch_token = values.get(b"SCHEDULEURM_LAUNCH_TOKEN", "")
    clean = lambda value: str(value).replace("\t", " ").replace("\n", " ")
    sys.stdout.write(
        "P\t{}\t{}\t{}\t{}\t{}\t{}\t0\t{}\t{}\n".format(
            clean(task_id), pid, sid, pgid, clean(state), rss_mb,
            clean(start_ticks), clean(launch_token),
        )
    )
'''
    return "python -c " + shlex.quote(script)


def adopt_live_local_pid_rows(
    task: dict,
    node: str,
    rows: list,
    *,
    deps: LocalOrphanRecoveryDeps,
    prior_status: Optional[str] = None,
) -> bool:
    """Adopt live LocalBackend PIDs carrying this task's SCHEDULEURM_TASK_ID marker."""
    candidates = []
    for row in rows:
        try:
            pid = int(row.get("pid"))
            sid = int(row.get("sid") or 0)
            pgid = int(row.get("pgid") or 0)
        except Exception:
            continue
        state = str(row.get("state") or "")
        if state and state[0] in ("Z", "X"):
            continue
        try:
            rss_mb = max(0, int(float(row.get("rss_mb") or 0)))
        except Exception:
            rss_mb = 0
        try:
            pcpu = max(0.0, float(row.get("pcpu") or 0.0))
        except Exception:
            pcpu = 0.0
        try:
            start_ticks = max(0, int(row.get("start_ticks") or 0))
        except Exception:
            start_ticks = 0
        candidates.append({
            "pid": pid,
            "pgid": pgid,
            "is_leader": pid == sid,
            "rss_mb": rss_mb,
            "pcpu": pcpu,
            "start_ticks": start_ticks,
            "launch_token": str(row.get("launch_token") or ""),
        })
    if not candidates:
        return False

    expected_token = str(task.get("launch_token") or task.get("exit_status_token") or "")
    observed_tokens = {item["launch_token"] for item in candidates if item["launch_token"]}
    if expected_token and observed_tokens and expected_token not in observed_tokens:
        return False
    if not expected_token:
        if len(observed_tokens) > 1:
            return False
        if observed_tokens:
            expected_token = next(iter(observed_tokens))
    if expected_token and observed_tokens:
        candidates = [item for item in candidates if item["launch_token"] == expected_token]
        if not candidates:
            return False

    candidates.sort(key=lambda item: (0 if item["is_leader"] else 1, item["pid"]))
    representative_pid = candidates[0]["pid"]
    representative_pgid = candidates[0]["pgid"]
    pids = sorted({item["pid"] for item in candidates})
    pgids = {item["pgid"] for item in candidates if item["pgid"] > 1}
    total_ram = sum(item["rss_mb"] for item in candidates)
    total_pcpu = sum(item["pcpu"] for item in candidates)
    start_ticks = {
        str(item["pid"]): item["start_ticks"]
        for item in candidates
        if item["start_ticks"] > 0
    }

    old_status = prior_status or task.get("status")
    task["status"] = "running"
    task["node"] = node
    task["remote_pids"] = pids
    task["alive_pids"] = pids
    task["process_group"] = next(iter(pgids)) if len(pgids) == 1 else None
    if start_ticks:
        task["remote_pid_start_ticks"] = start_ticks
    else:
        task.pop("remote_pid_start_ticks", None)
    if expected_token:
        task["exit_status_token"] = expected_token
        task["exit_status_path"] = runtime_exit_status_path(task.get("id"), expected_token)
    else:
        task.pop("exit_status_token", None)
        task.pop("exit_status_path", None)
    task["started_at"] = task.get("launching_started_at") or task.get("started_at") or deps.now()
    task["finished_at"] = None
    deps.remember_last_placement(task)
    task["peak_vram_mb"] = max(int(task.get("peak_vram_mb") or 0), 0)
    task["peak_ram_mb"] = max(int(task.get("peak_ram_mb") or 0), total_ram)
    deps.set_current_usage(task, 0, total_ram, total_pcpu)
    if total_ram > 0:
        task["ram_mb"] = max(int(task.get("ram_mb") or 0), total_ram)
    if total_pcpu > 0:
        task["cpu_cores"] = max(int(task.get("cpu_cores") or 0), max(1, math.ceil(total_pcpu / 100.0)))
    task.pop("launching_started_at", None)
    task.pop("launch_token", None)
    task["orphan_recovered_at"] = deps.now()
    task["orphan_recovered_from_status"] = old_status

    tid = task.get("id") or ""
    if deps.claim_enabled_for(node):
        try:
            deps.update_claim_pid(node, tid, representative_pid)
        except Exception:
            pass
    if deps.node_configs.get(node, {}).get("host") is None:
        task["log_path"] = f"{deps.state_dir}/logs/{tid}.log"
    else:
        task["log_path"] = f"/tmp/sched_{tid}.log"

    task["last_block_reason"] = (
        f"WAL recovery: adopted {len(pids)} live local PID(s) on {node} "
        f"matching SCHEDULEURM_TASK_ID={tid}; rss={total_ram}MB pcpu={total_pcpu:.1f}; "
        "avoids double-launch"
    )
    task["_orphan_representative_pid"] = representative_pid
    task["_orphan_representative_pgid"] = representative_pgid
    return True


def try_recover_orphan_local_task(
    task: dict,
    node: str,
    *,
    deps: LocalOrphanRecoveryDeps,
) -> bool:
    if deps.node_is_windows(node):
        return False
    tid = task.get("id") or ""
    if not tid:
        return False

    task_marker = shlex.quote(f"SCHEDULEURM_TASK_ID={tid}")
    cmd = (
        "for p in $(ps -eo pid= 2>/dev/null); do "
        "  env=$(tr '\\0' '\\n' < /proc/$p/environ 2>/dev/null) || continue; "
        f"  if printf '%s\\n' \"$env\" | grep -Fqx -- {task_marker}; then "
        "    sid=$(ps -o sid= -p $p 2>/dev/null | tr -d ' '); "
        "    pgid=$(ps -o pgid= -p $p 2>/dev/null | tr -d ' '); "
        "    state=$(awk '/^State:/ {print $2}' /proc/$p/status 2>/dev/null); "
        "    rss=$(awk '/VmRSS:/ {print int($2/1024)}' /proc/$p/status 2>/dev/null); "
        "    pcpu=$(ps -o pcpu= -p $p 2>/dev/null | tr -d ' '); "
        "    start_ticks=$(awk '{print $22; exit}' /proc/$p/stat 2>/dev/null); "
        "    launch_token=$(printf '%s\\n' \"$env\" | "
        "      sed -n 's/^SCHEDULEURM_LAUNCH_TOKEN=//p' | head -n 1); "
        "    echo \"$p|$sid|$pgid|$state|${rss:-0}|${pcpu:-0}|${start_ticks:-0}|${launch_token:-}\"; "
        "  fi; "
        "done"
    )
    try:
        rc, out, _ = deps.run_on(node, cmd, timeout=20, check=False)
        if rc != 0:
            return False
    except Exception:
        return False

    rows = []
    for line in (out or "").splitlines():
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 4:
            continue
        row = {
            "pid": parts[0],
            "sid": parts[1],
            "pgid": parts[2],
            "state": parts[3],
            "rss_mb": parts[4] if len(parts) > 4 else 0,
            "pcpu": parts[5] if len(parts) > 5 else 0.0,
        }
        if len(parts) > 6:
            row["start_ticks"] = parts[6]
        if len(parts) > 7:
            row["launch_token"] = parts[7]
        rows.append(row)
    if not adopt_live_local_pid_rows(task, node, rows, deps=deps, prior_status=task.get("status")):
        return False

    spec = (task.get("env_spec") or "").lower()
    looks_docker = spec.startswith("docker") or (spec == "auto" and task.get("image"))
    if looks_docker:
        cname = f"sched-{tid}"
        try:
            rc_d, out_d, _ = deps.run_on(
                node,
                f"docker inspect --format '{{{{.State.Pid}}}}' "
                f"{shlex.quote(cname)} 2>/dev/null",
                timeout=5,
                check=False,
            )
            if rc_d == 0:
                value = (out_d or "").strip()
                if value.isdigit() and int(value) > 0:
                    cpid = int(value)
                    task["container_name"] = cname
                    task["container_main_pid"] = cpid
                    task["remote_pids"] = [cpid]
                    task["alive_pids"] = [cpid]
                    task.pop("remote_pid_start_ticks", None)
                    if deps.claim_enabled_for(node):
                        try:
                            deps.update_claim_pid(node, tid, cpid)
                        except Exception:
                            pass
        except Exception:
            pass
    return True


def scan_scheduleurm_env_pids_on_node(node: str, *, deps: LocalOrphanRecoveryDeps) -> dict:
    """Return {task_id: [pid rows]} for live processes carrying SCHEDULEURM_TASK_ID."""
    if deps.node_is_windows(node):
        return {}
    cmd = _scheduleurm_process_inventory_command()
    try:
        rc, out, _ = deps.run_on(node, cmd, timeout=25, check=False)
        if rc != 0:
            return {}
    except Exception:
        return {}
    rows_by_tid = {}
    for line in (out or "").splitlines():
        if line.startswith("P\t"):
            tagged = [part.strip() for part in line.split("\t")]
            parts = tagged[1:] if len(tagged) == 10 else []
        else:
            parts = [part.strip() for part in line.split("|")]
        if len(parts) < 5:
            continue
        task_id = parts[0]
        if not task_id:
            continue
        row = {
            "pid": parts[1],
            "sid": parts[2],
            "pgid": parts[3],
            "state": parts[4],
            "rss_mb": parts[5] if len(parts) > 5 else 0,
            "pcpu": parts[6] if len(parts) > 6 else 0.0,
        }
        if len(parts) > 7:
            row["start_ticks"] = parts[7]
        if len(parts) > 8:
            row["launch_token"] = parts[8]
        rows_by_tid.setdefault(task_id, []).append(row)
    return rows_by_tid


def scan_launch_recovery_evidence_on_node(
    node: str,
    tasks: list[dict],
    *,
    deps: LocalOrphanRecoveryDeps,
) -> dict:
    """Scan one node once for all stale launch PIDs and exit sentinels."""
    if deps.node_is_windows(node):
        return {
            "ok": True,
            "supported": False,
            "rows_by_task": {},
            "exit_statuses": {},
        }

    status_checks = []
    for task in tasks:
        task_id = str(task.get("id") or "")
        token = str(task.get("launch_token") or task.get("exit_status_token") or "")
        if not task_id or not token:
            continue
        status_path = str(task.get("exit_status_path") or "")
        if not status_path:
            status_path = runtime_exit_status_path(task_id, token)
        status_checks.append(
            f"if IFS=\"$(printf '\\t')\" read -r rc ft tok < "
            f"{shlex.quote(status_path)} 2>/dev/null; then "
            f"printf 'X\\t%s\\t%s\\t%s\\t%s\\n' {shlex.quote(task_id)} "
            '"$rc" "$ft" "$tok"; fi; '
        )

    status_cmd = "".join(status_checks) + "true"
    try:
        status_rc, status_out, status_err = deps.run_on(
            node,
            status_cmd,
            timeout=15,
            check=False,
        )
    except Exception as exc:
        return {
            "ok": False,
            "supported": True,
            "rows_by_task": {},
            "exit_statuses": {},
            "error": f"sentinel scan {type(exc).__name__}: {str(exc)[:200]}",
        }
    if status_rc != 0:
        detail = " ".join(((status_err or status_out or "").strip()).split())
        return {
            "ok": False,
            "supported": True,
            "rows_by_task": {},
            "exit_statuses": {},
            "error": f"sentinel scan rc={status_rc}" + (f": {detail[:200]}" if detail else ""),
        }

    exit_statuses: dict[str, dict] = {}
    for line in (status_out or "").splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 5 or parts[0] != "X":
            continue
        try:
            exit_statuses[parts[1].strip()] = {
                "exit_code": int(parts[2]),
                "finished_at": float(parts[3]),
                "token": parts[4].strip(),
            }
        except (TypeError, ValueError):
            continue

    unresolved_ids = {
        str(task.get("id") or "")
        for task in tasks
        if str(task.get("id") or "") not in exit_statuses
    }
    if not unresolved_ids:
        return {
            "ok": True,
            "supported": True,
            "rows_by_task": {},
            "exit_statuses": exit_statuses,
        }

    process_cmd = _scheduleurm_process_inventory_command()
    try:
        rc, out, err = deps.run_on(node, process_cmd, timeout=30, check=False)
    except Exception as exc:
        return {
            "ok": True,
            "supported": True,
            "absence_proven": False,
            "rows_by_task": {},
            "exit_statuses": exit_statuses,
            "error": f"process scan {type(exc).__name__}: {str(exc)[:200]}",
        }
    if rc != 0:
        detail = " ".join(((err or out or "").strip()).split())
        return {
            "ok": True,
            "supported": True,
            "absence_proven": False,
            "rows_by_task": {},
            "exit_statuses": exit_statuses,
            "error": f"process scan rc={rc}" + (f": {detail[:200]}" if detail else ""),
        }

    rows_by_task: dict[str, list[dict]] = {}
    for line in (out or "").splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) == 10 and parts[0] == "P":
            task_id = parts[1].strip()
            if not task_id:
                continue
            rows_by_task.setdefault(task_id, []).append({
                "pid": parts[2].strip(),
                "sid": parts[3].strip(),
                "pgid": parts[4].strip(),
                "state": parts[5].strip(),
                "rss_mb": parts[6].strip(),
                "pcpu": parts[7].strip(),
                "start_ticks": parts[8].strip(),
                "launch_token": parts[9].strip(),
            })
    return {
        "ok": True,
        "supported": True,
        "rows_by_task": rows_by_task,
        "exit_statuses": exit_statuses,
    }


def recover_queued_live_local_tasks(state: dict, *, deps: LocalOrphanRecoveryDeps) -> int:
    """Recover queued records whose LocalBackend children are still alive remotely."""
    queued_by_node = {}
    for task in state.get("tasks", []):
        if task.get("status") != "queued":
            continue
        task_id = task.get("id")
        if not task_id:
            continue
        node = task.get("node") or task.get("require_node") or task.get("preferred_node")
        if not node or node not in deps.node_configs:
            continue
        if deps.node_is_windows(node):
            continue
        if (
            not deps.recover_remote_nodes
            and deps.node_configs.get(node, {}).get("host") is not None
        ):
            continue
        if not deps.requires_local_capacity_check(node, task):
            continue
        queued_by_node.setdefault(node, {})[task_id] = task
    recovered = 0
    for node, tasks_by_id in queued_by_node.items():
        rows_by_id = scan_scheduleurm_env_pids_on_node(node, deps=deps)
        if not rows_by_id:
            continue
        for task_id, task in tasks_by_id.items():
            rows = rows_by_id.get(task_id)
            if not rows:
                continue
            if adopt_live_local_pid_rows(task, node, rows, deps=deps, prior_status="queued"):
                recovered += 1
    return recovered
