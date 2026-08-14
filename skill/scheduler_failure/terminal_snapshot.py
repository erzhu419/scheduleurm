"""Terminal diagnostic snapshot helpers used outside the scheduler state lock."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Optional


def _no_terminal_evidence(_tasks: list[dict], **_kwargs) -> dict:
    return {}


@dataclass(frozen=True)
class TerminalDiagnosticsSnapshotDeps:
    state_lock: Callable[..., Any]
    load_state: Callable[[], dict]
    scan_completed_log_for_crash: Callable[[dict], tuple[bool, str]]
    diagnose_terminal: Callable[[dict], dict]
    discover_result_artifacts: Callable[..., list]
    notify: Callable[..., Any]
    now: Callable[[], float]
    max_tasks_per_cycle: int = -1
    max_wall_s: float = 0.0
    include_log_result_artifacts: bool = True
    collect_terminal_evidence: Callable[..., dict] = _no_terminal_evidence
    node_down_requeue_s: int = 300


def terminal_diagnostic_fingerprint(task: dict) -> tuple:
    return (
        str(task.get("id") or ""),
        str(task.get("node") or ""),
        str(task.get("log_path") or ""),
        str(task.get("started_at") or ""),
        str(task.get("slurm_job_id") or ""),
        str(task.get("cmd") or ""),
    )


def terminal_diagnostic_matches(task: dict, payload: Optional[dict]) -> bool:
    if not isinstance(payload, dict):
        return False
    return tuple(payload.get("fingerprint") or ()) == terminal_diagnostic_fingerprint(task)


def terminal_diagnostics_snapshot_outside_lock(
    probe_results: Optional[dict],
    probed_task_ids: Optional[set],
    purpose: str = "terminal-diagnostics",
    *,
    deps: TerminalDiagnosticsSnapshotDeps,
) -> dict:
    """Precompute slow terminal log diagnostics without holding the writer lock."""
    if not probe_results:
        return {}
    probed_ids = {str(x) for x in (probed_task_ids or set()) if str(x)}
    candidate_ids = {
        str(tid)
        for tid, res in (probe_results or {}).items()
        if isinstance(res, dict)
        and res.get("state") in ("dead", "unknown")
        and (not probed_ids or str(tid) in probed_ids)
    }
    if not candidate_ids:
        return {}
    try:
        with deps.state_lock(shared=True, purpose=f"{purpose}:snapshot"):
            snapshot = deps.load_state()
            tasks = [
                copy.deepcopy(t)
                for t in snapshot.get("tasks", [])
                if t.get("status") == "running"
                and str(t.get("id") or "") in candidate_ids
            ]
    except Exception as e:
        deps.notify(
            "terminal_diagnostics_snapshot_error",
            {"purpose": purpose, "error": str(e)[:200]},
            feishu_enabled=False,
        )
        return {}

    eligible_tasks = []
    for task in tasks:
        tid = str(task.get("id") or "")
        res = probe_results.get(tid) or {}
        if res.get("state") != "unknown":
            eligible_tasks.append(task)
            continue
        # The batch probe already attempted this backend. Re-probing every
        # unknown task here turns one failed node route into N serial SSH calls
        # and defeats both limits below. Only consume a definitive safety
        # result if the batch probe explicitly supplied one.
        if str(res.get("launch_safety_state") or "") != "dead":
            continue
        if not task.get("reroute_on_node_down"):
            continue
        try:
            unknown_for = deps.now() - float(task.get("probe_unknown_since") or deps.now())
        except Exception:
            unknown_for = 0.0
        threshold = int(
            task.get("node_down_requeue_s") or deps.node_down_requeue_s
        )
        if unknown_for < threshold:
            continue
        eligible_tasks.append(task)
    tasks = eligible_tasks
    if not tasks:
        return {}

    try:
        max_tasks = int(deps.max_tasks_per_cycle)
    except Exception:
        max_tasks = -1
    tasks.sort(
        key=lambda task: (
            0
            if (
                task.get("terminal_transition_deferred")
                or task.get("unknown_terminal_diagnosis_deferred")
            )
            else 1,
            float(
                task.get("terminal_transition_deferred_at")
                or task.get("unknown_terminal_diagnosis_deferred_at")
                or 0
            ),
            str(task.get("id") or ""),
        )
    )
    if max_tasks >= 0:
        selected = []
        unknown_selected = 0
        for task in tasks:
            tid = str(task.get("id") or "")
            if (probe_results.get(tid) or {}).get("state") == "unknown":
                if unknown_selected >= max_tasks:
                    continue
                unknown_selected += 1
            selected.append(task)
        tasks = selected
    out = {}
    started_at = deps.now()
    evidence_timeout_s = 0.0
    if deps.max_wall_s > 0:
        evidence_timeout_s = max(
            0.0,
            float(deps.max_wall_s) - (deps.now() - started_at),
        )
    try:
        terminal_evidence = (
            deps.collect_terminal_evidence(
                tasks,
                timeout_s=evidence_timeout_s,
            )
            if deps.max_wall_s <= 0 or evidence_timeout_s > 0
            else {}
        ) or {}
    except Exception as e:
        deps.notify(
            "terminal_evidence_batch_error",
            {"purpose": purpose, "error": str(e)[:200]},
            feishu_enabled=False,
        )
        terminal_evidence = {}

    slow_tasks = []
    for task in tasks:
        tid = str(task.get("id") or "")
        res = probe_results.get(tid) or {}
        evidence = terminal_evidence.get(tid)
        if (
            not isinstance(evidence, dict)
            or res.get("terminal_cancelled")
        ):
            slow_tasks.append(task)
            continue
        task.setdefault("finished_at", deps.now())
        task["_terminal_evidence"] = evidence
        task["_terminal_batch_probe_only"] = True
        try:
            diagnosis = deps.diagnose_terminal(task)
        except Exception:
            diagnosis = {"needs_deep_terminal_diagnosis": True}
        if diagnosis.get("needs_deep_terminal_diagnosis"):
            task["_terminal_batch_probe_only"] = False
            slow_tasks.append(task)
            continue
        payload = {
            "fingerprint": terminal_diagnostic_fingerprint(task),
            "computed_at": deps.now(),
            "diagnosis": diagnosis,
        }
        if res.get("terminal_ok") is True:
            payload["completed_log_crash"] = (
                bool(diagnosis.get("is_crash")),
                str(diagnosis.get("reason") or "")
                if diagnosis.get("is_crash") else "",
            )
        payload["result_artifacts"] = list(
            evidence.get("result_artifacts") or []
        )
        out[tid] = payload

    if max_tasks >= 0:
        slow_tasks = slow_tasks[:max_tasks]

    for task in slow_tasks:
        if (
            deps.max_wall_s > 0
            and deps.now() - started_at >= deps.max_wall_s
        ):
            break
        tid = str(task.get("id") or "")
        res = probe_results.get(tid) or {}
        task.setdefault("finished_at", deps.now())
        payload = {
            "fingerprint": terminal_diagnostic_fingerprint(task),
            "computed_at": deps.now(),
        }
        try:
            if res.get("terminal_cancelled"):
                pass
            elif res.get("terminal_ok") is True:
                payload["completed_log_crash"] = deps.scan_completed_log_for_crash(task)
            else:
                payload["diagnosis"] = deps.diagnose_terminal(task)
            include_log = (
                bool(deps.include_log_result_artifacts)
                and not bool(task.get("result_dir") or task.get("result_dirs"))
                and not bool(task.get("_terminal_evidence"))
            )
            payload["result_artifacts"] = deps.discover_result_artifacts(
                task,
                include_log=include_log,
            )
        except Exception as e:
            payload["error"] = str(e)[:200]
        out[tid] = payload
    return out
