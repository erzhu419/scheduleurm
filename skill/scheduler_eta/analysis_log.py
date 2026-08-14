"""Append-only ETA observations and terminal accuracy summaries."""

from __future__ import annotations

import fcntl
import json
import math
import re
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
TERMINAL_STATUSES = frozenset({"done", "failed", "cancelled"})
_SAMPLE_EVENTS = frozenset({"initial", "sample"})


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 3)


def _safe_task_name(task_id: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id).strip("._")
    return value or "task"


def _effective_started_at(task: dict) -> tuple[float | None, str]:
    actual = _finite_float(task.get("actual_started_at"))
    if actual is not None and actual > 0:
        return actual, "actual_started_at"
    started = _finite_float(task.get("started_at"))
    if started is not None and started > 0:
        return started, "started_at"
    return None, ""


def _attempt_id(task_id: str, started_at: float) -> str:
    return f"{task_id}:{int(round(started_at * 1000.0))}"


def _read_records(handle, attempt_id: str) -> list[dict]:
    handle.seek(0)
    records = []
    for raw_line in handle:
        try:
            row = json.loads(raw_line)
        except (TypeError, ValueError):
            continue
        if isinstance(row, dict) and row.get("attempt_id") == attempt_id:
            records.append(row)
    return records


def _append_record(handle, record: dict) -> None:
    handle.seek(0, 2)
    handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")
    handle.flush()


def _eta_fields(task: dict, observed_at: float, elapsed_s: float) -> dict:
    raw_eta = _finite_float(task.get("eta_seconds"))
    eta_available = raw_eta is not None and raw_eta > 0
    eta_s = max(0.0, raw_eta) if eta_available else None
    updated_at = _finite_float(task.get("eta_updated_at"))
    return {
        "eta_available": eta_available,
        "eta_seconds": _round(eta_s),
        "predicted_total_s": _round(elapsed_s + eta_s) if eta_s is not None else None,
        "predicted_finish_at": _round(observed_at + eta_s) if eta_s is not None else None,
        "eta_source": str(task.get("eta_source") or ""),
        "eta_confidence": str(task.get("eta_confidence") or ""),
        "eta_updated_at": _round(updated_at),
        "eta_age_s": (
            _round(max(0.0, observed_at - updated_at))
            if updated_at is not None else None
        ),
        "eta_detail": str(task.get("eta_detail") or "")[:500],
        "eta_probe_error": str(task.get("eta_probe_error") or "")[:300],
    }


def _sample_record(
    task: dict,
    *,
    event: str,
    attempt_id: str,
    started_at: float,
    started_at_source: str,
    observed_at: float,
    interval_s: int,
    captured_late: bool = False,
    previous_minute_index: int | None = None,
    fixed_interval_trigger: bool = False,
) -> dict:
    elapsed_s = max(0.0, observed_at - started_at)
    minute_index = int(elapsed_s // interval_s)
    missed_intervals = 0
    if previous_minute_index is not None:
        missed_intervals = max(0, minute_index - previous_minute_index - 1)
    record = {
        "schema_version": SCHEMA_VERSION,
        "event": event,
        "task_id": str(task.get("id") or ""),
        "attempt_id": attempt_id,
        "status": str(task.get("status") or ""),
        "project": str(task.get("project") or ""),
        "signature": str(task.get("signature") or ""),
        "description": str(task.get("description") or "")[:300],
        "node": task.get("node"),
        "gpu_idx": task.get("gpu_idx"),
        "observed_at": _round(observed_at),
        "started_at": _round(started_at),
        "started_at_source": started_at_source,
        "elapsed_s": _round(elapsed_s),
        "minute_index": minute_index,
        "missed_intervals": missed_intervals,
        "captured_late": bool(captured_late),
        "fixed_interval_trigger": bool(fixed_interval_trigger),
        "last_progress_line": str(task.get("last_progress_line") or "")[-500:],
    }
    record.update(_eta_fields(task, observed_at, elapsed_s))
    return record


def _actual_finish(task: dict, started_at: float) -> tuple[float, str, float]:
    scheduler_finished = _finite_float(task.get("finished_at"))
    backend_finished = _finite_float(task.get("backend_finished_at"))
    # Local status sentinels use epoch seconds. Permit their sub-second rounding
    # to fall just before the Python launch timestamp, then clamp to zero runtime.
    if backend_finished is not None and backend_finished >= started_at - 1.5:
        actual_finished = max(started_at, backend_finished)
        source = "backend_finished_at"
    elif scheduler_finished is not None and scheduler_finished >= started_at:
        actual_finished = scheduler_finished
        source = "finished_at"
    else:
        actual_finished = max(started_at, scheduler_finished or started_at)
        source = "started_at_fallback"
    detection_lag = (
        max(0.0, scheduler_finished - actual_finished)
        if scheduler_finished is not None else 0.0
    )
    return actual_finished, source, detection_lag


def _prediction_rows(records: list[dict], actual_duration_s: float) -> list[dict]:
    predictions = []
    for row in records:
        if row.get("event") not in _SAMPLE_EVENTS or not row.get("eta_available"):
            continue
        predicted_total = _finite_float(row.get("predicted_total_s"))
        elapsed = _finite_float(row.get("elapsed_s"))
        eta_s = _finite_float(row.get("eta_seconds"))
        if predicted_total is None or elapsed is None or eta_s is None:
            continue
        error = predicted_total - actual_duration_s
        predictions.append({
            "event": row.get("event"),
            "eta_source": str(row.get("eta_source") or ""),
            "elapsed_s": elapsed,
            "eta_seconds": eta_s,
            "predicted_total_s": predicted_total,
            "error_s": error,
            "abs_error_s": abs(error),
            "remaining_actual_s": max(0.0, actual_duration_s - elapsed),
        })
    return predictions


def _terminal_record(
    task: dict,
    *,
    attempt_id: str,
    started_at: float,
    started_at_source: str,
    observed_at: float,
    records: list[dict],
) -> dict:
    actual_finished, finished_source, detection_lag = _actual_finish(task, started_at)
    actual_duration = max(0.0, actual_finished - started_at)
    predictions = _prediction_rows(records, actual_duration)
    errors = [row["error_s"] for row in predictions]
    abs_errors = [abs(value) for value in errors]
    initial = next((row for row in predictions if row["event"] == "initial"), None)
    final = predictions[-1] if predictions else None
    source_counts = Counter(row["eta_source"] or "unknown" for row in predictions)
    missing_eta_samples = sum(
        1 for row in records
        if row.get("event") in _SAMPLE_EVENTS and not row.get("eta_available")
    )

    record = {
        "schema_version": SCHEMA_VERSION,
        "event": "terminal",
        "task_id": str(task.get("id") or ""),
        "attempt_id": attempt_id,
        "status": str(task.get("status") or ""),
        "project": str(task.get("project") or ""),
        "signature": str(task.get("signature") or ""),
        "description": str(task.get("description") or "")[:300],
        "node": task.get("node"),
        "gpu_idx": task.get("gpu_idx"),
        "observed_at": _round(observed_at),
        "started_at": _round(started_at),
        "started_at_source": started_at_source,
        "actual_finished_at": _round(actual_finished),
        "actual_finished_at_source": finished_source,
        "actual_duration_s": _round(actual_duration),
        "scheduler_finished_at": _round(_finite_float(task.get("finished_at"))),
        "watcher_detection_lag_s": _round(detection_lag),
        "exit_code": task.get("exit_code"),
        "prediction_count": len(predictions),
        "missing_eta_samples": missing_eta_samples,
        "eta_source_counts": dict(sorted(source_counts.items())),
        "initial_eta_seconds": _round(initial["eta_seconds"]) if initial else None,
        "initial_predicted_total_s": _round(initial["predicted_total_s"]) if initial else None,
        "initial_error_s": _round(initial["error_s"]) if initial else None,
        "initial_abs_error_s": _round(initial["abs_error_s"]) if initial else None,
        "initial_error_pct": (
            _round(100.0 * initial["error_s"] / actual_duration)
            if initial and actual_duration > 0 else None
        ),
        "eta_bias_s": _round(statistics.fmean(errors)) if errors else None,
        "eta_mae_s": _round(statistics.fmean(abs_errors)) if abs_errors else None,
        "eta_median_abs_error_s": _round(statistics.median(abs_errors)) if abs_errors else None,
        "eta_max_abs_error_s": _round(max(abs_errors)) if abs_errors else None,
        "final_sample_elapsed_s": _round(final["elapsed_s"]) if final else None,
        "final_eta_seconds": _round(final["eta_seconds"]) if final else None,
        "final_error_s": _round(final["error_s"]) if final else None,
        "diagnosis_reason": str((task.get("_diagnosis") or {}).get("reason") or "")[:500],
    }
    record.update({
        f"terminal_{key}": value
        for key, value in _eta_fields(task, observed_at, actual_duration).items()
    })
    return record


def _append_summary(summary_path: Path, record: dict) -> None:
    with summary_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            _append_record(handle, record)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _record_task(
    task: dict,
    *,
    log_dir: Path,
    interval_s: int,
    include_periodic: bool,
    force_periodic: bool,
    observed_at: float,
) -> tuple[int, int, Path | None]:
    task_id = str(task.get("id") or "")
    status = str(task.get("status") or "")
    started_at, started_at_source = _effective_started_at(task)
    if not task_id or started_at is None:
        return 0, 0, None
    if status != "running" and status not in TERMINAL_STATUSES:
        return 0, 0, None

    attempt_id = _attempt_id(task_id, started_at)
    path = log_dir / f"{_safe_task_name(task_id)}.jsonl"
    written = 0
    terminal_written = 0
    terminal_record = None
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            records = _read_records(handle, attempt_id)
            if any(row.get("event") == "terminal" for row in records):
                return 0, 0, path

            sample_records = [row for row in records if row.get("event") in _SAMPLE_EVENTS]
            if not sample_records:
                initial = _sample_record(
                    task,
                    event="initial",
                    attempt_id=attempt_id,
                    started_at=started_at,
                    started_at_source=started_at_source,
                    observed_at=observed_at,
                    interval_s=interval_s,
                    captured_late=(
                        status != "running" or observed_at - started_at >= interval_s
                    ),
                )
                _append_record(handle, initial)
                records.append(initial)
                sample_records.append(initial)
                written += 1

            if status == "running" and include_periodic:
                last = sample_records[-1]
                last_index = int(last.get("minute_index") or 0)
                current_index = int(max(0.0, observed_at - started_at) // interval_s)
                last_observed_at = _finite_float(last.get("observed_at")) or 0.0
                forced_due = (
                    force_periodic
                    and observed_at - last_observed_at >= max(1.0, interval_s * 0.5)
                )
                if current_index > last_index or forced_due:
                    sample = _sample_record(
                        task,
                        event="sample",
                        attempt_id=attempt_id,
                        started_at=started_at,
                        started_at_source=started_at_source,
                        observed_at=observed_at,
                        interval_s=interval_s,
                        previous_minute_index=last_index,
                        fixed_interval_trigger=force_periodic,
                    )
                    _append_record(handle, sample)
                    records.append(sample)
                    written += 1

            if status in TERMINAL_STATUSES:
                terminal_record = _terminal_record(
                    task,
                    attempt_id=attempt_id,
                    started_at=started_at,
                    started_at_source=started_at_source,
                    observed_at=observed_at,
                    records=records,
                )
                _append_record(handle, terminal_record)
                written += 1
                terminal_written = 1
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    if terminal_record is not None:
        _append_summary(log_dir / "summary.jsonl", terminal_record)
    return written, terminal_written, path


def record_eta_analysis_tasks(
    tasks: Iterable[dict],
    *,
    log_dir: str | Path,
    interval_s: int = 60,
    include_periodic: bool = False,
    force_periodic: bool = False,
    now: float | None = None,
) -> dict:
    """Record initial, minute, and terminal ETA observations outside state lock."""
    observed_at = time.time() if now is None else float(now)
    interval_s = max(1, int(interval_s))
    output_dir = Path(log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "tasks": 0,
        "records": 0,
        "terminals": 0,
        "errors": [],
        "log_dir": str(output_dir),
    }
    seen = set()
    for task in tasks:
        task_id = str((task or {}).get("id") or "")
        started_at, _source = _effective_started_at(task or {})
        dedupe_key = (task_id, started_at)
        if not task_id or dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        try:
            written, terminals, path = _record_task(
                task,
                log_dir=output_dir,
                interval_s=interval_s,
                include_periodic=include_periodic,
                force_periodic=force_periodic,
                observed_at=observed_at,
            )
            if path is not None:
                result["tasks"] += 1
            result["records"] += written
            result["terminals"] += terminals
        except Exception as exc:
            result["errors"].append({
                "task_id": task_id,
                "error": f"{type(exc).__name__}: {str(exc)[:300]}",
            })
    return result
