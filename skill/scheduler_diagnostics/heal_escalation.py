from __future__ import annotations

import json
import fcntl
import os
import time as time_module
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class HealSessionDeps:
    state_dir: Path
    log_dir: Path
    heal_fire_lock: Path
    heal_debounce_s: int
    claude_bin: str
    node_bin: str
    claude_cli_js: str
    heal_fire_log: Path
    home: Callable[[], Path]
    env_get: Callable[[str, str], str]
    now: Callable[[], float]
    strftime: Callable[[str], str]
    popen: Callable[..., object]
    devnull: object


@dataclass(frozen=True)
class EscalationDeps:
    escalations_file: Path
    fire_heal_session: Callable[[], bool]
    now: Callable[[], float]


def _escalation_lock_path(path: Path) -> Path:
    return path.with_name(path.name + ".lock")


def _latest_escalations_locked(path: Path) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    if not path.exists():
        return latest
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
        except Exception:
            continue
        task_id = record.get("task_id")
        if task_id:
            latest[str(task_id)] = record
    return latest


def latest_escalations(path: Path) -> dict[str, dict]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _escalation_lock_path(path)
    with open(lock_path, "a+") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_SH)
        try:
            return _latest_escalations_locked(path)
        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)


def _append_escalation_records(path: Path, records: list[dict]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _escalation_lock_path(path)
    with open(lock_path, "a+") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        try:
            with open(path, "a") as output:
                for record in records:
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                os.fsync(output.fileno())
        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)


def _resolved_by_code_staging(record: dict) -> bool:
    if record.get("category") != "ENV_MISSING":
        return False
    text = f"{record.get('reason') or ''}\n{record.get('tail') or ''}".lower()
    return "can't open file" in text and ".py" in text


def resolve_staged_cwd_escalations(
    cwd: str,
    nodes: list[str] | tuple[str, ...] | set[str],
    *,
    path: Path,
    now: Callable[[], float] = time_module.time,
) -> list[str]:
    """Resolve pending missing-script escalations after cwd staging succeeds."""
    if not cwd or not nodes:
        return []
    node_set = {str(node) for node in nodes}
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _escalation_lock_path(path)
    resolved: list[str] = []
    with open(lock_path, "a+") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        try:
            latest = _latest_escalations_locked(path)
            records = []
            for task_id, record in latest.items():
                if record.get("status") != "pending":
                    continue
                if record.get("cwd") != cwd or record.get("node") not in node_set:
                    continue
                if not _resolved_by_code_staging(record):
                    continue
                resolved_record = dict(record)
                resolved_record.update(
                    {
                        "ts": now(),
                        "status": "resolved",
                        "resolution": "staging_success",
                        "resolution_reason": "project cwd staged successfully after missing script",
                    }
                )
                records.append(resolved_record)
                resolved.append(task_id)
            if records:
                with open(path, "a") as output:
                    for record in records:
                        output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()
                    os.fsync(output.fileno())
        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)
    return resolved


def recover_resolved_staging_failures(
    state: dict,
    latest: dict[str, dict],
    *,
    requeue_after_crash: Callable[[dict, dict], str | None],
) -> int:
    """Requeue hard failures once their missing project script was restaged."""
    tasks_by_id = {task.get("id"): task for task in state.get("tasks", [])}
    recovered = 0
    for task_id, record in latest.items():
        if record.get("status") != "resolved" or record.get("resolution") != "staging_success":
            continue
        task = tasks_by_id.get(task_id)
        if not task or task.get("status") != "failed" or task.get("requeued_as"):
            continue
        task["resolved_environment_retry"] = {
            "ts": record.get("ts"),
            "node": record.get("node"),
            "resolution": record.get("resolution"),
        }
        new_id = requeue_after_crash(task, state)
        if new_id:
            task["requeued_as"] = new_id
            recovered += 1
    return recovered


def fire_heal_session(*, deps: HealSessionDeps) -> bool:
    """Spawn a debounced headless scheduler-heal session."""
    try:
        deps.state_dir.mkdir(parents=True, exist_ok=True)
        deps.log_dir.mkdir(parents=True, exist_ok=True)
        now = deps.now()
        if deps.heal_fire_lock.exists():
            try:
                last = deps.heal_fire_lock.stat().st_mtime
                if now - last < deps.heal_debounce_s:
                    return False
            except Exception:
                pass
        deps.heal_fire_lock.touch()
        os.utime(deps.heal_fire_lock, (now, now))
        log_fh = open(deps.heal_fire_log, "ab")
        log_fh.write(f"\n=== fire @ {deps.strftime('%Y-%m-%d %H:%M:%S')} ===\n".encode())
        log_fh.flush()
        nvm_bin = os.path.dirname(deps.claude_bin)
        clean_env = {
            "HOME": str(deps.home()),
            "USER": deps.env_get("USER", "erzhu419"),
            "PATH": f"{nvm_bin}:/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "CLAUDE_HEAL_FIRE": "1",
        }
        deps.popen(
            [deps.node_bin, deps.claude_cli_js, "-p", "/scheduler-heal", "--dangerously-skip-permissions"],
            stdin=deps.devnull,
            stdout=log_fh,
            stderr=log_fh,
            start_new_session=True,
            cwd=str(deps.home()),
            env=clean_env,
        )
        log_fh.close()
        return True
    except Exception as exc:
        try:
            with open(deps.heal_fire_log, "a") as fh:
                fh.write(f"\n[{deps.strftime('%Y-%m-%d %H:%M:%S')}] _fire_heal_session FAILED: {exc}\n")
        except Exception:
            pass
        return False


def escalation_record(task: dict, category: str, diag: dict, *, now: Callable[[], float]) -> dict:
    return {
        "ts": now(),
        "task_id": task["id"],
        "signature": task.get("signature", ""),
        "project": task.get("project", ""),
        "category": category,
        "node": task.get("node"),
        "gpu_idx": task.get("gpu_idx"),
        "reason": diag.get("reason", ""),
        "tail": (diag.get("tail", "") or "")[-500:],
        "log_path": diag.get("log_path"),
        "retry_count": task.get("retry_count", 0),
        "cmd": (task.get("cmd", "") or "")[:200],
        "cwd": task.get("cwd", ""),
        "status": "pending",
    }


def write_escalation(task: dict, category: str, diag: dict, *, deps: EscalationDeps) -> dict:
    """Append a scheduler-heal escalation record and trigger a debounced heal session."""
    record = escalation_record(task, category, diag, now=deps.now)
    _append_escalation_records(deps.escalations_file, [record])
    deps.fire_heal_session()
    return record
