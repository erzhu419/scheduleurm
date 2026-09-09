"""Running-task lifecycle transitions driven by backend probe results."""

from __future__ import annotations

import copy
import math
import time as _time
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class RunningLifecycleDeps:
    set_current_usage: Callable[[dict, int, int, float], None]
    mark_probe_unknown: Callable[[dict, Optional[dict]], None]
    clear_probe_unknown: Callable[[dict], Optional[dict]]
    annotate_diag_after_unknown: Callable[[dict, Optional[dict]], dict]
    cached_task_success_marker: Callable[[dict], str | None]
    diagnose_terminal: Callable[[dict], dict]
    release_task_claims_and_intents: Callable[[dict], Any]
    record_result_artifacts: Callable[[dict], Any]
    requeue_after_crash: Callable[[dict, dict], str | None]
    local_launch_transport_alive: Callable[[dict], bool]
    terminal_diagnostic_matches: Callable[[dict, Optional[dict]], bool]
    scan_completed_log_for_crash: Callable[[dict], tuple[bool, str]]
    mark_user_cancelled: Callable[[dict, str], None]
    history_get: Callable[[str], dict]
    untrusted_startup_oom_sample: Callable[[dict, int], bool]
    history_record: Callable[..., Any]
    runtime_history_record: Callable[..., Any]
    apply_discovered_result_artifacts: Callable[[dict, list], Any]
    remember_last_placement: Callable[[dict], None]
    declared_cpu_slot_floor: Callable[[dict], int]
    default_ram_mb: int
    default_cpu_cores: int
    node_down_requeue_s: int
    early_death_seconds: int = 120
    now: Callable[[], float] = _time.time


def check_running_probe_result(
    task: dict,
    res: Optional[dict],
    *,
    set_current_usage: Callable[[dict, int, int, float], None],
) -> str:
    """Interpret a single backend probe result for the public check_running helper."""
    if not res:
        return "dead"
    if res["state"] == "unknown":
        return "unknown"
    if res["state"] == "dead":
        set_current_usage(task, 0, 0, 0.0)
        return "dead"
    set_current_usage(
        task,
        res.get("vram_mb", 0),
        res.get("ram_mb", 0),
        res.get("pcpu", 0.0),
    )
    if res["vram_mb"] > 0:
        task["peak_vram_mb"] = max(task.get("peak_vram_mb", 0), res["vram_mb"])
    if res["ram_mb"] > 0:
        task["peak_ram_mb"] = max(task.get("peak_ram_mb", 0), res["ram_mb"])
    task["alive_pids"] = res["alive_pids"]
    return "alive"


def _clear_unknown_terminal_diagnosis_deferred(task: dict) -> None:
    task.pop("unknown_terminal_diagnosis_deferred", None)
    task.pop("unknown_terminal_diagnosis_deferred_at", None)


def handle_unknown_probe_result(
    task: dict,
    state: dict,
    res: Optional[dict],
    terminal_diagnostics: Optional[dict] = None,
    *,
    deps: RunningLifecycleDeps,
) -> None:
    """Apply the conservative unknown-probe path for one running task."""
    deps.mark_probe_unknown(task, res)
    if not task.get("reroute_on_node_down"):
        return
    try:
        unknown_for = deps.now() - float(task.get("probe_unknown_since") or deps.now())
    except Exception:
        unknown_for = 0
    threshold = int(task.get("node_down_requeue_s") or deps.node_down_requeue_s)
    if unknown_for < threshold:
        return
    cached_success = deps.cached_task_success_marker(task)
    if cached_success:
        now_done = deps.now()
        _clear_unknown_terminal_diagnosis_deferred(task)
        deps.set_current_usage(task, 0, 0, 0.0)
        task["status"] = "done"
        task["finished_at"] = now_done
        task["last_block_reason"] = (
            f"probe unknown for {int(unknown_for)}s, but cached success "
            f"marker {cached_success!r} was already observed; suppressing reroute"
        )
        task["_diagnosis"] = {
            "is_crash": False,
            "reason": task["last_block_reason"],
            "tail": task.get("last_progress_line") or "(cached success marker)",
            "lifetime_s": int(max(0, now_done - (task.get("started_at") or now_done))),
            "log_path": task.get("log_path"),
            "success_marker": cached_success,
        }
        try:
            deps.release_task_claims_and_intents(task)
        except Exception:
            pass
        try:
            deps.record_result_artifacts(task)
        except Exception:
            pass
        return
    launch_state = str((res or {}).get("launch_safety_state") or "unknown")
    launch_reason = str(
        (res or {}).get("launch_safety_reason")
        or (res or {}).get("error")
        or "batch backend probe was inconclusive"
    )
    if launch_state in ("alive", "unknown"):
        _clear_unknown_terminal_diagnosis_deferred(task)
        task["last_block_reason"] = (
            f"node/probe unknown for {int(unknown_for)}s on {task.get('node')}; "
            f"not rerouting because recorded launch artifact is {launch_state} "
            f"({launch_reason})"
        )
        return
    precomputed_terminal = (terminal_diagnostics or {}).get(
        str(task.get("id") or "")
    )
    if not deps.terminal_diagnostic_matches(task, precomputed_terminal):
        precomputed_terminal = None
    if terminal_diagnostics is not None and (
        precomputed_terminal is None
        or not isinstance(precomputed_terminal.get("diagnosis"), dict)
    ):
        task["unknown_terminal_diagnosis_deferred"] = True
        task["unknown_terminal_diagnosis_deferred_at"] = deps.now()
        task["last_block_reason"] = (
            f"node/probe unknown for {int(unknown_for)}s on {task.get('node')}; "
            "terminal log/result diagnosis pending outside state lock"
        )
        return
    terminal_diag = (
        copy.deepcopy(precomputed_terminal["diagnosis"])
        if precomputed_terminal is not None
        else deps.diagnose_terminal(task)
    )
    terminal_success = terminal_diag.get("success_marker") if terminal_diag else None
    if terminal_success:
        now_done = deps.now()
        _clear_unknown_terminal_diagnosis_deferred(task)
        deps.set_current_usage(task, 0, 0, 0.0)
        task["status"] = "done"
        task["finished_at"] = now_done
        terminal_diag = dict(terminal_diag)
        reason = terminal_diag.get("reason") or "normal exit (success marker found)"
        terminal_diag["reason"] = (
            f"{reason}; backend probe was unknown for {int(unknown_for)}s, "
            "but terminal log contains a success marker; suppressing reroute"
        )
        task["_diagnosis"] = terminal_diag
        task["last_block_reason"] = terminal_diag["reason"]
        try:
            deps.release_task_claims_and_intents(task)
        except Exception:
            pass
        try:
            if precomputed_terminal is not None:
                deps.apply_discovered_result_artifacts(
                    task,
                    list(precomputed_terminal.get("result_artifacts") or []),
                )
            else:
                deps.record_result_artifacts(task)
        except Exception:
            pass
        return
    reason = (
        f"node/probe unknown for {int(unknown_for)}s on {task.get('node')}; "
        "rerouting opt-in CPU batch shard"
    )
    _clear_unknown_terminal_diagnosis_deferred(task)
    deps.set_current_usage(task, 0, 0, 0.0)
    task["status"] = "failed"
    task["finished_at"] = deps.now()
    task["last_block_reason"] = reason
    task["_diagnosis"] = {
        "is_crash": True,
        "reason": reason,
        "tail": task.get("last_probe_unknown_reason") or "backend probe unknown",
        "lifetime_s": int(max(0, task["finished_at"] - (task.get("started_at") or task["finished_at"]))),
        "log_path": task.get("log_path"),
    }
    try:
        deps.release_task_claims_and_intents(task)
    except Exception:
        pass
    new_id = deps.requeue_after_crash(task, state)
    if new_id:
        task["requeued_as"] = new_id


def _lightweight_deferred_terminal_diagnosis(
    task: dict,
    res: dict,
    *,
    deps: RunningLifecycleDeps,
) -> dict:
    """Classify local PID-dead transitions without remote log I/O under the state lock."""
    finished = task.get("finished_at") or deps.now()
    started = task.get("started_at") or finished
    lifetime = max(0, finished - started)
    log_path = task.get("log_path")
    terminal_reason = (
        res.get("terminal_reason")
        or "local backend pid probe found no live tracked process"
    )
    try:
        success_marker = deps.cached_task_success_marker(task) or None
    except Exception:
        success_marker = None

    if success_marker:
        is_crash = False
        reason = (
            f"normal exit (cached success marker {success_marker!r}); "
            "terminal log diagnosis deferred outside watch main lock"
        )
    elif task.get("auto_adopted") or not log_path:
        is_crash = False
        reason = (
            f"{terminal_reason}; no scheduler log available; terminal log diagnosis "
            "deferred outside watch main lock"
        )
    elif 0 < lifetime < max(1, int(deps.early_death_seconds or 120)):
        is_crash = True
        reason = (
            f"{terminal_reason} after only {lifetime:.0f}s with no cached success "
            "marker; terminal log diagnosis deferred outside watch main lock"
        )
    else:
        is_crash = False
        reason = (
            f"{terminal_reason}; terminal log diagnosis deferred outside watch main "
            "lock; ambiguous; assumed normal unless runtime history marks incomplete"
        )

    return {
        "is_crash": is_crash,
        "reason": reason,
        "tail": "(terminal log diagnosis deferred)",
        "lifetime_s": int(lifetime),
        "log_size": 0,
        "log_path": log_path,
        "success_marker": success_marker,
        "terminal_diagnosis_deferred": True,
    }


def handle_dead_probe_result(task: dict, state: dict, res: dict,
                             reconnect_sync: Optional[dict],
                             terminal_diagnostics: Optional[dict], *,
                             deps: RunningLifecycleDeps) -> None:
    """Apply terminal transition policy for a backend-dead running task."""
    if deps.local_launch_transport_alive(task):
        deps.mark_probe_unknown(
            task,
            {"error": "backend reported dead but local launch transport is still alive"},
        )
        return
    cancel_request_token = str(task.get("cancel_request_token") or "")
    if cancel_request_token:
        request_actor = copy.deepcopy(task.get("cancel_request_actor") or {})
        reason = str(task.get("cancel_request_reason") or "user batch force-cancel")
        deps.set_current_usage(task, 0, 0, 0.0)
        try:
            deps.release_task_claims_and_intents(task)
        except Exception:
            pass
        deps.mark_user_cancelled(task, reason)
        if request_actor:
            task["cancel_actor"] = request_actor
            task["cancelled_by"] = request_actor.get("label") or task.get("cancelled_by")
            task["last_kill_actor"] = request_actor
            task["last_killed_by"] = request_actor.get("label") or task.get("last_killed_by")
        task["last_cancel_request_token"] = cancel_request_token
        task.pop("terminal_transition_deferred", None)
        task.pop("terminal_transition_deferred_at", None)
        for key in (
            "cancel_request_token",
            "cancel_requested_at",
            "cancel_request_actor",
            "cancel_request_reason",
        ):
            task.pop(key, None)
        return
    terminal_ok = res.get("terminal_ok")
    terminal_snapshot_supplied = terminal_diagnostics is not None
    precomputed_terminal = (terminal_diagnostics or {}).get(str(task.get("id") or ""))
    if not deps.terminal_diagnostic_matches(task, precomputed_terminal):
        precomputed_terminal = None
    if res.get("exit_code") is not None:
        task["exit_code"] = res.get("exit_code")
    if res.get("backend_finished_at") is not None:
        task["backend_finished_at"] = res.get("backend_finished_at")
    if (
        terminal_snapshot_supplied
        and precomputed_terminal is None
        and not res.get("terminal_cancelled")
    ):
        task["terminal_transition_deferred"] = True
        task["terminal_transition_deferred_at"] = deps.now()
        task["last_block_reason"] = (
            "process exited; terminal log/result diagnosis pending outside state lock"
        )
        return
    cached_success = None
    if (
        res.get("terminal_diagnosis_deferred")
        and terminal_ok is None
        and precomputed_terminal is None
    ):
        try:
            cached_success = deps.cached_task_success_marker(task) or None
        except Exception:
            cached_success = None
        if not cached_success:
            task["terminal_transition_deferred"] = True
            task["terminal_transition_deferred_at"] = deps.now()
            task["last_block_reason"] = (
                "process exited; terminal log/result diagnosis pending outside state lock"
            )
            return

    task.pop("terminal_transition_deferred", None)
    task.pop("terminal_transition_deferred_at", None)
    # A successful terminal commit supersedes queued/probe/diagnostic block
    # messages. Keep failure reasons below, where a crash transition rewrites
    # this field with the final diagnosis.
    task.pop("last_block_reason", None)
    deps.set_current_usage(task, 0, 0, 0.0)
    task["status"] = "done"
    task["finished_at"] = deps.now()
    try:
        deps.release_task_claims_and_intents(task)
    except Exception:
        pass

    backend_state = res.get("backend_state")
    terminal_reason = res.get("terminal_reason") or (
        f"legacy external scheduler terminal state {backend_state}" if backend_state else ""
    )
    if res.get("terminal_cancelled"):
        deps.mark_user_cancelled(
            task,
            terminal_reason or "legacy external scheduler terminal state CANCELLED; treated as user/admin cancel",
        )
        diag = {
            "is_crash": False,
            "reason": terminal_reason or "legacy external scheduler terminal state CANCELLED",
            "tail": "(legacy external scheduler reported CANCELLED; not auto-requeued)",
            "lifetime_s": int(max(0, task["finished_at"] - (task.get("started_at") or task["finished_at"]))),
            "log_size": 0,
            "log_path": task.get("log_path"),
            "success_marker": "LEGACY_EXTERNAL_CANCELLED",
        }
    elif terminal_ok is True:
        precomputed_crash = (
            precomputed_terminal.get("completed_log_crash")
            if precomputed_terminal else None
        )
        if isinstance(precomputed_crash, (list, tuple)) and len(precomputed_crash) >= 2:
            crash_matched, crash_reason = bool(precomputed_crash[0]), str(precomputed_crash[1] or "")
        else:
            crash_matched, crash_reason = deps.scan_completed_log_for_crash(task)
        if crash_matched:
            diag = {
                "is_crash": True,
                "reason": crash_reason,
                "tail": "(see log_path; pattern detected in tail)",
                "lifetime_s": int(max(0, task["finished_at"] - (task.get("started_at") or task["finished_at"]))),
                "log_size": 0,
                "log_path": task.get("log_path"),
                "success_marker": None,
            }
        else:
            diag = {
                "is_crash": False,
                "reason": terminal_reason or "legacy external scheduler terminal state COMPLETED",
                "tail": "(legacy external scheduler reported COMPLETED; log not required for success)",
                "lifetime_s": int(max(0, task["finished_at"] - (task.get("started_at") or task["finished_at"]))),
                "log_size": 0,
                "log_path": task.get("log_path"),
                "success_marker": f"LEGACY_EXTERNAL_{backend_state or 'COMPLETED'}",
            }
    else:
        if cached_success:
            diag = {
                "is_crash": False,
                "reason": (
                    f"normal exit (cached success marker {cached_success!r}); "
                    "terminal log diagnosis skipped"
                ),
                "tail": task.get("last_progress_line") or "(cached success marker)",
                "lifetime_s": int(max(
                    0,
                    task["finished_at"]
                    - (task.get("started_at") or task["finished_at"]),
                )),
                "log_size": 0,
                "log_path": task.get("log_path"),
                "success_marker": cached_success,
            }
        else:
            diag = (
                copy.deepcopy(precomputed_terminal.get("diagnosis"))
                if precomputed_terminal and isinstance(precomputed_terminal.get("diagnosis"), dict)
                else (
                    _lightweight_deferred_terminal_diagnosis(task, res, deps=deps)
                    if res.get("terminal_diagnosis_deferred") and terminal_ok is None
                    else deps.diagnose_terminal(task)
                )
            )
        if terminal_ok is False:
            diag = dict(diag)
            diag["is_crash"] = True
            reason = terminal_reason or "legacy external scheduler terminal state indicates failure"
            if backend_state == "OUT_OF_MEMORY":
                reason += " (out of memory)"
            prior = diag.get("reason") or ""
            if prior and prior != "ambiguous; assumed normal":
                reason = f"{reason}; {prior}"
            diag["reason"] = reason
    diag = deps.annotate_diag_after_unknown(diag, reconnect_sync)
    task["_diagnosis"] = diag
    if diag["is_crash"] and not task.get("auto_adopted"):
        task["status"] = "failed"
        task["last_block_reason"] = diag["reason"]
        new_id = deps.requeue_after_crash(task, state)
        if new_id:
            task["requeued_as"] = new_id
    elif (not diag.get("is_crash")
          and terminal_ok is not True
          and not diag.get("success_marker")
          and not diag.get("terminal_diagnosis_deferred")
          and task.get("status") == "done"
          and task.get("started_at") and task.get("finished_at")):
        sig = task.get("signature") or ""
        h = deps.history_get(sig) or {}
        expected = int(h.get("dur_s_ewma", 0))
        runs = int(h.get("dur_s_runs", 0))
        lifetime = max(0, task["finished_at"] - task["started_at"])
        if expected > 0 and runs >= 2 and lifetime < 0.5 * expected:
            task["status"] = "failed"
            task["last_block_reason"] = (
                f"incomplete: ran {int(lifetime)}s "
                f"({int(100*lifetime/expected)}% of EWMA {expected}s); "
                f"likely killed/crashed (not user-cancelled)"
            )
            cmd_real = bool(task.get("cmd")) and not task["cmd"].startswith("(auto-adopted")
            if cmd_real:
                new_id = deps.requeue_after_crash(task, state)
                if new_id:
                    task["requeued_as"] = new_id
    try:
        if precomputed_terminal and isinstance(precomputed_terminal.get("result_artifacts"), list):
            deps.apply_discovered_result_artifacts(task, precomputed_terminal.get("result_artifacts") or [])
        elif terminal_snapshot_supplied or (
            res.get("terminal_diagnosis_deferred")
            and terminal_ok is None
            and not bool(task.get("result_dir") or task.get("result_dirs"))
        ):
            task["result_artifacts_pending"] = True
        else:
            deps.record_result_artifacts(task)
    except Exception:
        pass
    duration_s = 0
    if task.get("started_at") and task.get("finished_at"):
        duration_start = float(
            task.get("actual_started_at") or task["started_at"])
        duration_end = float(
            task.get("backend_finished_at") or task["finished_at"])
        duration_s = max(0, int(duration_end - duration_start))
    peak_vram_for_history = task.get("peak_vram_mb", 0)
    peak_ram_for_history = task.get("peak_ram_mb", 0)
    duration_for_history = duration_s if task.get("status") == "done" else 0
    if deps.untrusted_startup_oom_sample(task, duration_s):
        task["history_skipped_reason"] = (
            "startup OOM before progress; peak likely JAX/CUDA preallocation, "
            "not steady-state resource need"
        )
        peak_vram_for_history = 0
        peak_ram_for_history = 0
    deps.history_record(
        task.get("signature"),
        peak_vram_mb=peak_vram_for_history,
        peak_ram_mb=peak_ram_for_history,
        cpu_cores=task.get("cpu_cores", 0),
        duration_s=duration_for_history,
        task=task,
    )
    if task.get("status") == "done":
        deps.runtime_history_record(task, duration_s=duration_s)
    if reconnect_sync:
        task["last_status_sync_status"] = task.get("status")
        task["last_status_sync_reason"] = (
            f"node probe reconnected and synced terminal status={task.get('status')}"
        )


def apply_alive_probe_result(task: dict, res: dict, reconnect_sync: Optional[dict], *,
                             deps: RunningLifecycleDeps) -> None:
    """Fold alive probe telemetry into task state and resource estimates."""
    try:
        cancel_request_age = deps.now() - float(task.get("cancel_requested_at") or 0)
    except (TypeError, ValueError):
        cancel_request_age = 0
    if task.get("cancel_request_token") and cancel_request_age > 3600:
        task["expired_cancel_request_at"] = deps.now()
        for key in (
            "cancel_request_token",
            "cancel_requested_at",
            "cancel_request_actor",
            "cancel_request_reason",
        ):
            task.pop(key, None)
    deps.remember_last_placement(task)
    _clear_unknown_terminal_diagnosis_deferred(task)
    if reconnect_sync:
        task["last_status_sync_status"] = "running"
        task["last_status_sync_reason"] = (
            f"node probe reconnected and confirmed running after "
            f"{int(reconnect_sync.get('duration_s') or 0)}s unknown"
        )
    task["alive_pids"] = res["alive_pids"]
    total_vram = res["vram_mb"]
    deps.set_current_usage(task, total_vram, res.get("ram_mb", 0), res.get("pcpu", 0.0))
    if total_vram > 0:
        task["peak_vram_mb"] = max(task.get("peak_vram_mb", 0), total_vram)
    total_ram = res["ram_mb"]
    if total_ram > 0:
        task["peak_ram_mb"] = max(task.get("peak_ram_mb", 0), total_ram)
        declared_ram = task.get("ram_mb", deps.default_ram_mb)
        if total_ram > declared_ram * 1.1:
            task["ram_mb"] = total_ram
    total_pcpu = res["pcpu"]
    if total_pcpu > 0:
        new_cpu = max(1, math.ceil(total_pcpu / 100.0))
        cur_cpu = task.get("cpu_cores", deps.default_cpu_cores)
        if task.get("auto_adopted"):
            if new_cpu < cur_cpu:
                task["cpu_cores"] = new_cpu
        else:
            declared_floor = deps.declared_cpu_slot_floor(task)
            try:
                explicit_declared_cpu = int(task.get("cpu_declared_cores") or 0)
            except (TypeError, ValueError):
                explicit_declared_cpu = 0
            if task.get("cpu_cores_explicit") and explicit_declared_cpu > 0:
                # Explicit CPU counts describe the placement contract. Parent and
                # coordinator overhead may add fractional aggregate CPU without
                # consuming another schedulable worker slot.
                cur_cpu = explicit_declared_cpu
                task["cpu_cores"] = explicit_declared_cpu
            elif declared_floor and cur_cpu < declared_floor:
                cur_cpu = declared_floor
                task["cpu_cores"] = declared_floor
            if not task.get("cpu_cores_explicit") and new_cpu > cur_cpu:
                task["cpu_cores"] = new_cpu
            elif cur_cpu > new_cpu * 2.5 and not (
                    task.get("cpu_cores_explicit") or declared_floor):
                task["cpu_cores"] = new_cpu
