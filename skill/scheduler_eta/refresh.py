"""ETA refresh from scheduler log tails."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class EtaRefreshDeps:
    load_eta_tracker_module: Callable[[], Any]
    load_runtime_history: Callable[[], dict]
    runtime_history_closest_index: Callable[[dict], Any]
    eta_tail_targets: Callable[[dict], tuple[dict, list]]
    history_get: Callable[[str], dict | None]
    runtime_total_history_s: Callable[..., int]
    effective_elapsed_s: Callable[[dict], float]
    history_fallback_eta_seconds: Callable[[int, float], tuple[int, bool]]
    eta_tail_outputs_by_node: Callable[[dict], dict]
    last_progress_line: Callable[[str], str | None]
    bapr_batch_projection: Callable[[dict, str, float], dict | None]
    apply_runtime_projection: Callable[[dict, dict | None], Any]
    eta_confidence_for_source: Callable[[str], str]
    now: Callable[[], float]


def _runtime_total(task: dict, runtime_history: dict, closest_index: Any,
                   deps: EtaRefreshDeps) -> int:
    return deps.runtime_total_history_s(
        task,
        runtime_history=runtime_history,
        closest_index=closest_index,
    )


def _history_eta(task: dict, runtime_history: dict, closest_index: Any,
                 deps: EtaRefreshDeps) -> tuple[int, bool, int, float, int]:
    sig = task.get("signature") or ""
    history = deps.history_get(sig) or {}
    runtime_total = _runtime_total(task, runtime_history, closest_index, deps)
    ewma = runtime_total or int(history.get("dur_s_ewma", 0))
    elapsed = deps.effective_elapsed_s(task)
    eta_s, overrun = deps.history_fallback_eta_seconds(ewma, elapsed)
    return eta_s, overrun, runtime_total, elapsed, ewma


def _set_fallback_eta(task: dict, eta_s: int, overrun: bool, runtime_total: int,
                      *, no_log_path: bool) -> None:
    task["eta_seconds"] = eta_s
    if not task.get("eta_seconds"):
        task.pop("eta_source", None)
        task.pop("eta_confidence", None)
        task["eta_detail"] = (
            "no scheduler log_path and no runtime history fallback"
            if no_log_path
            else "no progress marker in log tail and no runtime history fallback"
        )
        return
    if task.get("eta_seconds"):
        if overrun:
            task["eta_source"] = (
                "runtime_history_overrun" if runtime_total else "duration_ewma_overrun"
            )
        else:
            task["eta_source"] = (
                "runtime_history_fallback" if runtime_total else "duration_ewma_fallback"
            )
        task["eta_confidence"] = "low"
    if overrun:
        task["eta_detail"] = (
            "no scheduler log_path; historical ETA overrun, using bounded unknown-remaining floor"
            if no_log_path
            else "no progress marker in log tail; historical ETA overrun, using bounded unknown-remaining floor"
        )
    else:
        if no_log_path:
            task["eta_detail"] = (
                "no scheduler log_path; ETA from runtime history fallback"
                if runtime_total
                else "no scheduler log_path; ETA from duration EWMA fallback"
            )
        else:
            task["eta_detail"] = (
                "no progress marker in log tail; ETA from runtime history fallback"
                if runtime_total
                else "no progress marker in log tail; ETA from duration EWMA fallback"
            )


def _refresh_pure_ewma_tasks(
    tasks: list,
    probed_task_ids: set | None,
    runtime_history: dict,
    closest_index: Any,
    deps: EtaRefreshDeps,
) -> None:
    for task in tasks:
        if probed_task_ids is not None and str(task.get("id") or "") not in probed_task_ids:
            continue
        eta_s, overrun, runtime_total, _elapsed, _ewma = _history_eta(
            task, runtime_history, closest_index, deps)
        task["eta_updated_at"] = int(deps.now())
        task["eta_log_bytes"] = 0
        _set_fallback_eta(task, eta_s, overrun, runtime_total, no_log_path=True)
        task.pop("eta_probe_error", None)


def _split_tail_output(output: str) -> dict:
    chunks = re.compile(r"===ETA_LOG_(\S+?)===").split(output)
    if len(chunks) < 3:
        return {}
    it = iter(chunks[1:])
    return dict(zip(it, it))


def _apply_probe_failure_fallback(
    task: dict,
    runtime_history: dict,
    closest_index: Any,
    deps: EtaRefreshDeps,
) -> None:
    task["eta_probe_error"] = "log tail probe failed; kept previous ETA or used fallback"
    task["eta_updated_at"] = int(deps.now())
    if task.get("eta_seconds") is not None:
        return
    eta_s, overrun, runtime_total, _elapsed, _ewma = _history_eta(
        task, runtime_history, closest_index, deps)
    _set_fallback_eta(task, eta_s, overrun, runtime_total, no_log_path=False)


def _apply_log_eta(
    task: dict,
    tail_text: str,
    eta_tracker: Any,
    runtime_history: dict,
    closest_index: Any,
    deps: EtaRefreshDeps,
) -> None:
    fallback_eta, fallback_overrun, runtime_total, elapsed, ewma = _history_eta(
        task, runtime_history, closest_index, deps)
    task["eta_seconds"] = eta_tracker.compute_eta_seconds(
        tail_text,
        elapsed_s=elapsed,
        fallback_ewma_s=ewma,
        cmd=task.get("cmd"),
    )
    task["eta_updated_at"] = int(deps.now())
    task["eta_log_bytes"] = len(tail_text.encode("utf-8", errors="replace"))
    task.pop("eta_probe_error", None)

    progress_line = deps.last_progress_line(tail_text)
    if progress_line:
        task["last_progress_line"] = progress_line
    projection = eta_tracker.runtime_projection(
        tail_text, elapsed_s=elapsed, cmd=task.get("cmd"))
    bapr_projection = deps.bapr_batch_projection(task, tail_text, elapsed)
    if bapr_projection:
        projection = bapr_projection
        task["eta_seconds"] = int(bapr_projection.get("eta_s") or 0)
    deps.apply_runtime_projection(task, projection)
    if projection:
        source = projection.get("source") or "progress"
        task["eta_source"] = source
        task["eta_confidence"] = deps.eta_confidence_for_source(source)
        if source == "bapr_fork_protocol":
            stage = projection.get("protocol_stage") or "unknown"
            current = projection.get("current")
            total = projection.get("total_units")
            future = projection.get("future_stages")
            task["eta_detail"] = (
                f"parsed BAPR fork protocol stage={stage}; "
                f"protocol_units={current}/{total}; future_stages={future}"
            )
        elif source == "bapr_independent_specialist":
            current = projection.get("current")
            total = projection.get("total_units")
            task["eta_detail"] = (
                "parsed BAPR independent-specialist progress; "
                f"iterations={current}/{total}; "
                f"window={float(projection.get('unit_s') or 0):.1f}s/iter; "
                f"terminal={float(projection.get('terminal_gap_s') or 0):.0f}s"
            )
        elif source == "bapr_seed_batch":
            done = projection.get("completed_seeds")
            total = projection.get("seed_count")
            suffix = (
                f"; completed_seeds={done}/{total}"
                if done is not None and total is not None
                else ""
            )
            task["eta_detail"] = (
                f"parsed BAPR seed batch progress from scheduler log tail{suffix}"
            )
        elif source == "kg_inner_progress":
            task["eta_detail"] = (
                "parsed KG inner stage timings; "
                f"stage={projection.get('stage')}/{projection.get('total_stages')}; "
                f"base={float(projection.get('base_step_s') or 0):.1f}s; "
                f"future_evals={projection.get('future_eval_count')}; "
                f"terminal={float(projection.get('terminal_gap_s') or 0):.1f}s"
            )
        elif source == "saas_native_eta":
            task["eta_detail"] = (
                "parsed task-native canonical SAAS projection; "
                f"iteration={projection.get('current')}/"
                f"{projection.get('total_units')}; "
                f"model={projection.get('eta_model')}; "
                f"robust_alternate={int(projection.get('robust_eta_s') or 0)}s"
            )
        else:
            task["eta_detail"] = f"parsed {source} from scheduler log tail"
    elif task.get("eta_seconds") or (fallback_overrun and fallback_eta > 0):
        if fallback_overrun:
            task["eta_seconds"] = fallback_eta
        _set_fallback_eta(
            task,
            int(task.get("eta_seconds") or fallback_eta),
            fallback_overrun,
            runtime_total,
            no_log_path=False,
        )
    else:
        task["eta_detail"] = "no parseable progress marker and no runtime history fallback"


def refresh_eta_from_logs(
    state: dict,
    *,
    tail_outputs: Optional[dict] = None,
    probed_task_ids: Optional[set] = None,
    deps: EtaRefreshDeps,
) -> None:
    """Parse log-tail ETA signals and update running task ETA fields."""
    eta_tracker = deps.load_eta_tracker_module()
    if eta_tracker is None:
        return

    runtime_history = deps.load_runtime_history()
    closest_index = deps.runtime_history_closest_index(runtime_history)
    by_node, pure_ewma = deps.eta_tail_targets(state)
    if probed_task_ids is not None:
        probed_task_ids = {str(value) for value in probed_task_ids if str(value)}

    _refresh_pure_ewma_tasks(
        pure_ewma, probed_task_ids, runtime_history, closest_index, deps)
    if not by_node:
        return

    outputs = deps.eta_tail_outputs_by_node(by_node) if tail_outputs is None else (tail_outputs or {})
    for node, entries in by_node.items():
        output = outputs.get(node)
        if output is None:
            for task, _log_path in entries:
                if probed_task_ids is not None and str(task.get("id") or "") not in probed_task_ids:
                    continue
                _apply_probe_failure_fallback(task, runtime_history, closest_index, deps)
            continue
        log_by_tid = _split_tail_output(output)
        for task, _log_path in entries:
            if probed_task_ids is not None and str(task.get("id") or "") not in probed_task_ids:
                continue
            tail_text = log_by_tid.get(task["id"], "")
            _apply_log_eta(task, tail_text, eta_tracker, runtime_history, closest_index, deps)
