"""Runtime history recording helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class RuntimeHistoryRecordDeps:
    load_runtime_history: Callable[[], dict]
    save_runtime_history: Callable[[dict], Any]
    task_runtime_keys: Callable[[dict], list]
    runtime_device_kind_for_task: Callable[[dict, Any], str]
    runtime_node_bucket_key: Callable[[str, str], str]
    percentile: Callable[[list, int], int]
    max_entries: int
    samples_per_key: int
    percentile_value: int
    min_walltime_s: int
    walltime_mult: float
    now: Callable[[], float] = time.time


def _append_percentile_sample(
    rec: dict,
    *,
    samples_field: str,
    value_field: str,
    value: int | float,
    samples_per_key: int,
    percentile_value: int,
    percentile: Callable[[list, int], int],
    scale_ms: bool = False,
) -> None:
    samples = rec.get(samples_field) or []
    samples.append(value)
    samples = samples[-samples_per_key:]
    rec[samples_field] = samples
    if scale_ms:
        rec[value_field] = float(
            percentile([int(float(sample) * 1000) for sample in samples], percentile_value)
        ) / 1000.0
    else:
        rec[value_field] = percentile(samples, percentile_value)


def _record_node_runtime(
    cur: dict,
    task: dict,
    *,
    total_s: int,
    unit_s: float,
    now: int,
    deps: RuntimeHistoryRecordDeps,
) -> None:
    node = task.get("node") or task.get("last_node") or ""
    if not node:
        return
    device = deps.runtime_device_kind_for_task(task, task.get("gpu_idx"))
    bucket = deps.runtime_node_bucket_key(node, device)
    node_runtime = cur.get("node_runtime")
    if not isinstance(node_runtime, dict):
        node_runtime = {}
    node_rec = node_runtime.get(bucket)
    if not isinstance(node_rec, dict):
        node_rec = {}
    _append_percentile_sample(
        node_rec,
        samples_field="total_s_samples",
        value_field="total_s",
        value=int(total_s),
        samples_per_key=deps.samples_per_key,
        percentile_value=deps.percentile_value,
        percentile=deps.percentile,
    )
    if unit_s > 0:
        _append_percentile_sample(
            node_rec,
            samples_field="unit_s_samples",
            value_field="unit_s",
            value=float(unit_s),
            samples_per_key=deps.samples_per_key,
            percentile_value=deps.percentile_value,
            percentile=deps.percentile,
            scale_ms=True,
        )
    node_rec["runs"] = int(node_rec.get("runs") or 0) + 1
    node_rec["last_seen"] = now
    node_rec["device_kind"] = device
    node_runtime[bucket] = node_rec
    cur["node_runtime"] = node_runtime


def runtime_history_record(task: dict, duration_s: int = 0, *, deps: RuntimeHistoryRecordDeps) -> None:
    """Fold exact-parameter runtime samples into runtime_history.json."""
    if not task:
        return
    actual_duration_s = max(0, int(duration_s or 0))
    actual_full_run = bool(
        actual_duration_s > 0
        and int(task.get("retry_count") or 0) == 0
        and not task.get("resume_from")
        and not task.get("resume_managed_by_cmd")
    )
    if actual_full_run:
        total_s = actual_duration_s
        source = "duration"
    else:
        total_s = int(task.get("runtime_total_s_est") or 0)
        source = task.get("runtime_est_source") or ""
        if total_s <= 0 and actual_duration_s > 0:
            total_s = actual_duration_s
            source = "duration"
    if total_s <= 0:
        return

    total_units = int(task.get("runtime_total_units") or 0)
    if actual_full_run and total_units > 0:
        unit_s = float(actual_duration_s) / float(total_units)
    else:
        unit_s = float(task.get("runtime_unit_s_est") or 0.0)
    history = deps.load_runtime_history()
    now = int(deps.now())

    for key, kind, payload in deps.task_runtime_keys(task):
        cur = history.get(key)
        if not isinstance(cur, dict):
            cur = {}
        if actual_full_run and cur.get("source") not in ("duration", "actual_duration"):
            # Replace preflight/live projections once a complete non-resumed
            # production run provides authoritative wall time. Keeping the
            # projected sample in an upper-percentile history would let one
            # bad ETA poison every future closest-match estimate.
            for field in (
                "total_s_samples", "total_s", "unit_s_samples", "unit_s",
                "walltime_s", "node_runtime",
            ):
                cur.pop(field, None)
            cur["runs"] = 0
        _append_percentile_sample(
            cur,
            samples_field="total_s_samples",
            value_field="total_s",
            value=int(total_s),
            samples_per_key=deps.samples_per_key,
            percentile_value=deps.percentile_value,
            percentile=deps.percentile,
        )
        cur["walltime_s"] = int(max(deps.min_walltime_s, cur["total_s"] * deps.walltime_mult))
        if unit_s > 0:
            _append_percentile_sample(
                cur,
                samples_field="unit_s_samples",
                value_field="unit_s",
                value=float(unit_s),
                samples_per_key=deps.samples_per_key,
                percentile_value=deps.percentile_value,
                percentile=deps.percentile,
                scale_ms=True,
            )
        if total_units > 0:
            cur["total_units"] = total_units
        _record_node_runtime(cur, task, total_s=total_s, unit_s=unit_s, now=now, deps=deps)
        cur["source"] = source or cur.get("source") or "duration"
        cur["kind"] = kind
        cur["signature"] = payload.get("signature") or ""
        cur["project"] = payload.get("project") or ""
        cur["description"] = payload.get("description") or ""
        cur["cwd"] = payload.get("cwd") or ""
        cur["env_spec"] = payload.get("env_spec") or "none"
        # Long launcher environments can exceed 500 characters before the
        # script name and algorithm flags appear.  Truncating at the old limit
        # made semantically identical reruns invisible to closest-history ETA.
        # Keep enough of the command to retain both runtime identity and the
        # experiment arguments without allowing unbounded state growth.
        command = payload.get("cmd") or ""
        if len(command) > 8192:
            command = command[:2048] + " ... " + command[-6139:]
        cur["cmd"] = command
        cur["runs"] = int(cur.get("runs") or 0) + 1
        cur["last_seen"] = now
        history[key] = cur

    if len(history) > deps.max_entries:
        kept = sorted(
            history.items(),
            key=lambda kv: -(kv[1].get("last_seen", 0) if isinstance(kv[1], dict) else 0),
        )
        history = dict(kept[:deps.max_entries])
    deps.save_runtime_history(history)
