"""Live A/B validation for the Scheduleurm sweetspot placement hook.

This runner does not change scheduler semantics.  It drives the existing
CLI, records raw command output, and uses the existing per-task edit switch to
construct an intentionally over-packed legacy baseline.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

from .trace_export import init_run


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _queue_path() -> Path:
    return Path.home() / ".claude" / "scheduler" / "queue.json"


def _now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(data, sort_keys=True) + "\n")


def _load_queue_tasks() -> list[dict[str, Any]]:
    path = _queue_path()
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return list(data.get("tasks") or [])


def _tasks_by_ids(ids: Iterable[str]) -> list[dict[str, Any]]:
    wanted = set(ids)
    return [t for t in _load_queue_tasks() if str(t.get("id")) in wanted]


def _run_cmd(
    args: list[str],
    *,
    cwd: Path,
    raw_dir: Path,
    label: str,
    env: dict[str, str] | None = None,
    timeout_s: int = 600,
) -> subprocess.CompletedProcess[str]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    full_env = os.environ.copy()
    if env:
        full_env.update({str(k): str(v) for k, v in env.items()})
    proc = subprocess.run(
        args,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        env=full_env,
        timeout=timeout_s,
    )
    (raw_dir / f"{label}.stdout").write_text(proc.stdout or "")
    (raw_dir / f"{label}.stderr").write_text(proc.stderr or "")
    meta = {
        "args": args,
        "cwd": str(cwd),
        "env_override": dict(env or {}),
        "returncode": proc.returncode,
        "timestamp": time.time(),
    }
    _write_json(raw_dir / f"{label}.meta.json", meta)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{label} failed rc={proc.returncode}: "
            f"{(proc.stderr or proc.stdout or '').strip()[:500]}"
        )
    return proc


def _watcher_status() -> dict[str, Any]:
    proc = subprocess.run(
        ["systemctl", "--user", "is-active", "scheduler"],
        text=True,
        capture_output=True,
    )
    status = (proc.stdout or proc.stderr or "").strip()
    return {
        "returncode": proc.returncode,
        "status": status or "unknown",
        "active": proc.returncode == 0 and status == "active",
    }


def _scheduler_cmd(
    scheduler_args: list[str],
    *,
    raw_dir: Path,
    label: str,
    env: dict[str, str] | None = None,
    timeout_s: int = 600,
) -> subprocess.CompletedProcess[str]:
    return _run_cmd(
        ["python3", "skill/scheduler.py", *scheduler_args],
        cwd=_repo_root(),
        raw_dir=raw_dir,
        label=label,
        env=env,
        timeout_s=timeout_s,
    )


def _parse_submit_id(stdout: str) -> str:
    m = re.search(r"\bsubmitted\s+(t\d+)\b", stdout or "")
    if not m:
        raise RuntimeError(f"could not parse submitted task id from: {stdout[:300]!r}")
    return m.group(1)


def _parse_rate(line: str | None) -> float:
    if not line:
        return 0.0
    m = re.search(r"\brate=([0-9]+(?:\.[0-9]+)?)\s*step/s\b", line)
    if not m:
        return 0.0
    return float(m.group(1))


def _task_row(task: dict[str, Any], *, phase: str, stage: str, sample_idx: int | None = None) -> dict[str, Any]:
    cfg = task.get("placement_algorithm_config") or {}
    audit = task.get("placement_algorithm_audit") or {}
    score = audit.get("score") or {}
    return {
        "ts": time.time(),
        "phase": phase,
        "stage": stage,
        "sample_idx": sample_idx,
        "id": task.get("id"),
        "status": task.get("status"),
        "node": task.get("node"),
        "gpu_idx": task.get("gpu_idx"),
        "placement_algorithm": task.get("placement_algorithm"),
        "hard_rule_override": cfg.get("hard_rule_override"),
        "candidate_bucket": audit.get("candidate_bucket"),
        "score": score.get("score"),
        "score_components": score.get("components"),
        "rate_step_s": _parse_rate(task.get("last_progress_line")),
        "runtime_current_unit": task.get("runtime_current_unit"),
        "runtime_total_units": task.get("runtime_total_units"),
        "eta_seconds": task.get("eta_seconds"),
        "eta_source": task.get("eta_source"),
        "progress_ratio": task.get("progress_ratio"),
        "last_progress_line": task.get("last_progress_line"),
        "started_at": task.get("started_at"),
        "current_vram_mb": task.get("current_vram_mb"),
        "peak_vram_mb": task.get("peak_vram_mb"),
        "current_ram_mb": task.get("current_ram_mb"),
        "peak_ram_mb": task.get("peak_ram_mb"),
        "last_block_reason": task.get("last_block_reason"),
        "last_eviction_kind": task.get("last_eviction_kind"),
        "last_resource_eviction": task.get("last_resource_eviction"),
        "last_kill_action": task.get("last_kill_action"),
        "last_kill_reason": task.get("last_kill_reason"),
        "launch_error": task.get("launch_error"),
    }


def _has_progress(task: dict[str, Any]) -> bool:
    if int(task.get("runtime_current_unit") or 0) > 0:
        return True
    return _parse_rate(task.get("last_progress_line")) > 0


def _snapshot(raw_dir: Path, label: str, ids: list[str]) -> list[dict[str, Any]]:
    tasks = _tasks_by_ids(ids)
    _write_json(raw_dir / f"{label}.json", tasks)
    return tasks


def _record_event(run_dir: Path, kind: str, **payload: Any) -> None:
    _append_jsonl(
        run_dir / "reports" / "events.jsonl",
        {"ts": time.time(), "kind": kind, **payload},
    )


def _status_refresh(raw_dir: Path, label: str) -> None:
    _scheduler_cmd(
        ["status", "--json"],
        raw_dir=raw_dir,
        label=label,
        env={
            "SCHEDULEURM_RUNNING_PROBE_MIN_INTERVAL_S": "0",
            "SCHEDULEURM_ETA_REFRESH_MIN_INTERVAL_S": "0",
        },
        timeout_s=900,
    )


def _submit_phase(
    *,
    phase: str,
    run_id: str,
    raw_dir: Path,
    node: str,
    cwd: str,
    remote_script: str,
    python_bin: str,
    task_count: int,
    steps: int,
    size: int,
    vram_mb: int,
    ram_mb: int,
    cpu: int,
    mem_fraction: float,
    allow_legacy_over_one_third: bool,
) -> list[str]:
    ids: list[str] = []
    for i in range(task_count):
        label = f"{run_id}-{phase}-{i}"
        cmd = (
            f"OMP_NUM_THREADS=1 "
            f"XLA_PYTHON_CLIENT_PREALLOCATE=false "
            f"XLA_PYTHON_CLIENT_MEM_FRACTION={mem_fraction:.3f} "
            f"{python_bin} -u {remote_script} "
            f"--steps {steps} --size {size} --label {label}"
        )
        proc = _scheduler_cmd(
            [
                "submit",
                "--description",
                f"ScheduleurmBench module2 {phase} task {i}",
                "--cmd",
                cmd,
                "--cwd",
                cwd,
                "--signature",
                f"ScheduleurmBench/module2/{run_id}/{phase}/{i}",
                "--vram",
                str(vram_mb),
                "--ram-mb",
                str(ram_mb),
                "--cpu",
                str(cpu),
                "--priority",
                "high",
                "--project",
                "ScheduleurmBench",
                "--require-node",
                node,
                "--allow-no-ckpt",
                "--allow-no-resume",
                "--allow-duplicate",
            ],
            raw_dir=raw_dir,
            label=f"{phase}_submit_{i}",
            timeout_s=600,
        )
        tid = _parse_submit_id(proc.stdout)
        ids.append(tid)
        if phase == "legacy" and allow_legacy_over_one_third:
            _scheduler_cmd(
                ["edit", tid, "--allow-gpu-over-one-third"],
                raw_dir=raw_dir,
                label=f"{phase}_edit_allow_over_one_third_{i}",
                timeout_s=120,
            )
    _snapshot(raw_dir, f"{phase}_post_submit_tasks", ids)
    return ids


def _dispatch_phase(raw_dir: Path, phase: str, algorithm: str, hard_rule_mode: str) -> None:
    env: dict[str, str] = {}
    if algorithm != "legacy":
        env.update(
            {
                "SCHEDULEURM_ALGO_GPU_SWEET_SPOT_TASKS": "2",
                "SCHEDULEURM_ALGO_MAX_TASKS_PER_GPU": "2",
            }
        )
    _scheduler_cmd(
        ["dispatch", "--algorithm", algorithm, "--hard-rule-mode", hard_rule_mode],
        raw_dir=raw_dir,
        label=f"{phase}_dispatch_{algorithm}",
        env=env,
        timeout_s=1200,
    )


def _wait_for_progress(
    *,
    phase: str,
    ids: list[str],
    run_dir: Path,
    raw_dir: Path,
    min_progress_tasks: int,
    timeout_s: int,
    poll_s: int,
) -> list[dict[str, Any]]:
    deadline = time.time() + timeout_s
    last_tasks: list[dict[str, Any]] = []
    attempt = 0
    while True:
        attempt += 1
        _status_refresh(raw_dir, f"{phase}_warmup_status_{attempt:03d}")
        last_tasks, summary = _record_observation(
            run_dir=run_dir,
            raw_dir=raw_dir,
            phase=phase,
            stage="warmup",
            ids=ids,
            sample_idx=attempt,
        )
        progressed = sum(1 for t in last_tasks if t.get("status") == "running" and _has_progress(t))
        _record_event(
            run_dir,
            "warmup_sample",
            phase=phase,
            attempt=attempt,
            progressed=progressed,
            required=min_progress_tasks,
            summary=summary,
        )
        if progressed >= min_progress_tasks:
            _record_event(
                run_dir,
                "warmup_complete",
                phase=phase,
                attempt=attempt,
                progressed=progressed,
                required=min_progress_tasks,
            )
            return last_tasks
        if time.time() >= deadline:
            _record_event(
                run_dir,
                "warmup_timeout",
                phase=phase,
                attempt=attempt,
                progressed=progressed,
                required=min_progress_tasks,
                timeout_s=timeout_s,
            )
            return last_tasks
        time.sleep(max(1, poll_s))


def _measure_window(
    *,
    phase: str,
    ids: list[str],
    run_dir: Path,
    raw_dir: Path,
    measure_s: int,
    poll_s: int,
) -> list[dict[str, Any]]:
    end = time.time() + max(0, int(measure_s))
    sample_idx = 0
    last_tasks: list[dict[str, Any]] = []
    while True:
        sample_idx += 1
        _status_refresh(raw_dir, f"{phase}_measure_status_{sample_idx:03d}")
        last_tasks, summary = _record_observation(
            run_dir=run_dir,
            raw_dir=raw_dir,
            phase=phase,
            stage="measure",
            ids=ids,
            sample_idx=sample_idx,
        )
        _record_event(
            run_dir,
            "measure_sample",
            phase=phase,
            sample_idx=sample_idx,
            remaining_s=max(0.0, end - time.time()),
            summary=summary,
        )
        if time.time() >= end:
            return last_tasks
        time.sleep(min(max(1, poll_s), max(0.0, end - time.time())))


def _summarize_phase(phase: str, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    running = [t for t in tasks if t.get("status") == "running"]
    queued = [t for t in tasks if t.get("status") == "queued"]
    rates = []
    per_gpu: dict[str, int] = {}
    progress_units = []
    evictions = []
    blocked = []
    for t in running:
        gpu = str(t.get("gpu_idx"))
        per_gpu[gpu] = per_gpu.get(gpu, 0) + 1
        rate = _parse_rate(t.get("last_progress_line"))
        if rate > 0:
            rates.append(rate)
        if int(t.get("runtime_current_unit") or 0) > 0:
            progress_units.append(int(t.get("runtime_current_unit") or 0))
    for t in tasks:
        if t.get("last_resource_eviction") or t.get("last_eviction_kind") or t.get("last_kill_action"):
            evictions.append({
                "id": t.get("id"),
                "status": t.get("status"),
                "last_eviction_kind": t.get("last_eviction_kind"),
                "last_kill_action": t.get("last_kill_action"),
                "last_kill_reason": t.get("last_kill_reason"),
                "last_resource_eviction": t.get("last_resource_eviction"),
            })
        if t.get("last_block_reason"):
            blocked.append({
                "id": t.get("id"),
                "status": t.get("status"),
                "last_block_reason": t.get("last_block_reason"),
            })
    counts: dict[str, int] = {}
    for t in tasks:
        status = str(t.get("status"))
        counts[status] = counts.get(status, 0) + 1
    return {
        "phase": phase,
        "ids": [t.get("id") for t in tasks],
        "status_counts": counts,
        "per_gpu_running": dict(sorted(per_gpu.items())),
        "queued_count": len(queued),
        "running_count": len(running),
        "running_with_rate_count": len(rates),
        "rates_step_s": rates,
        "aggregate_active_rate_step_s": sum(rates),
        "mean_active_rate_step_s": statistics.fmean(rates) if rates else 0.0,
        "min_active_rate_step_s": min(rates) if rates else 0.0,
        "max_active_rate_step_s": max(rates) if rates else 0.0,
        "runtime_current_units": progress_units,
        "eviction_count": len(evictions),
        "evictions": evictions,
        "blocked_count": len(blocked),
        "blocked": blocked,
    }


def _record_observation(
    *,
    run_dir: Path,
    raw_dir: Path,
    phase: str,
    stage: str,
    ids: list[str],
    sample_idx: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    label = f"{phase}_{stage}"
    if sample_idx is not None:
        label = f"{label}_{sample_idx:03d}"
    tasks = _snapshot(raw_dir, f"{label}_tasks", ids)
    summary = _summarize_phase(phase, tasks)
    event = {
        "ts": time.time(),
        "phase": phase,
        "stage": stage,
        "sample_idx": sample_idx,
        "summary": summary,
    }
    _append_jsonl(run_dir / "reports" / "timeline.jsonl", event)
    for task in tasks:
        _append_jsonl(
            run_dir / "reports" / "per_task_timeline.jsonl",
            _task_row(task, phase=phase, stage=stage, sample_idx=sample_idx),
        )
    return tasks, summary


def _cancel_all(raw_dir: Path, phase: str, ids: list[str]) -> None:
    for tid in ids:
        try:
            _scheduler_cmd(
                ["cancel", "--force", tid],
                raw_dir=raw_dir,
                label=f"{phase}_cancel_{tid}",
                timeout_s=300,
            )
        except Exception as e:
            (raw_dir / f"{phase}_cancel_{tid}.error.txt").write_text(str(e) + "\n")


def _run_phase(
    *,
    phase: str,
    algorithm: str,
    run_id: str,
    run_dir: Path,
    args: argparse.Namespace,
    allow_legacy_over_one_third: bool,
) -> tuple[list[str], dict[str, Any]]:
    raw_dir = run_dir / "raw"
    _record_event(
        run_dir,
        "phase_start",
        phase=phase,
        algorithm=algorithm,
        hard_rule_mode=args.hard_rule_mode,
        task_count=args.task_count,
        size=args.size,
        steps=args.steps,
    )
    ids = _submit_phase(
        phase=phase,
        run_id=run_id,
        raw_dir=raw_dir,
        node=args.node,
        cwd=args.cwd,
        remote_script=args.remote_script,
        python_bin=args.python_bin,
        task_count=args.task_count,
        steps=args.steps,
        size=args.size,
        vram_mb=args.vram_mb,
        ram_mb=args.ram_mb,
        cpu=args.cpu,
        mem_fraction=args.mem_fraction,
        allow_legacy_over_one_third=allow_legacy_over_one_third,
    )
    _record_event(run_dir, "phase_submitted", phase=phase, ids=ids)
    _record_observation(
        run_dir=run_dir,
        raw_dir=raw_dir,
        phase=phase,
        stage="post_submit",
        ids=ids,
    )
    _record_event(run_dir, "dispatch_start", phase=phase, algorithm=algorithm)
    _dispatch_phase(raw_dir, phase, algorithm, args.hard_rule_mode)
    _record_event(run_dir, "dispatch_done", phase=phase, algorithm=algorithm)
    _status_refresh(raw_dir, f"{phase}_post_dispatch_status")
    _record_observation(
        run_dir=run_dir,
        raw_dir=raw_dir,
        phase=phase,
        stage="post_dispatch",
        ids=ids,
    )
    _wait_for_progress(
        phase=phase,
        ids=ids,
        run_dir=run_dir,
        raw_dir=raw_dir,
        min_progress_tasks=args.min_progress_tasks,
        timeout_s=args.warmup_timeout_s,
        poll_s=args.poll_s,
    )
    _record_event(run_dir, "measure_start", phase=phase, measure_s=args.measure_s)
    measured = _measure_window(
        phase=phase,
        ids=ids,
        run_dir=run_dir,
        raw_dir=raw_dir,
        measure_s=args.measure_s,
        poll_s=args.poll_s,
    )
    _status_refresh(raw_dir, f"{phase}_post_measure_status")
    measured, summary = _record_observation(
        run_dir=run_dir,
        raw_dir=raw_dir,
        phase=phase,
        stage="post_measure",
        ids=ids,
    )
    summary = _summarize_phase(phase, measured)
    _write_json(run_dir / "reports" / f"{phase}_summary.json", summary)
    _record_event(run_dir, "phase_summary", phase=phase, summary=summary)
    _cancel_all(raw_dir, phase, ids)
    _status_refresh(raw_dir, f"{phase}_post_cancel_status")
    _record_observation(
        run_dir=run_dir,
        raw_dir=raw_dir,
        phase=phase,
        stage="post_cancel",
        ids=ids,
    )
    _record_event(run_dir, "phase_cancelled", phase=phase, ids=ids)
    return ids, summary


def _verdict(run_id: str, args: argparse.Namespace, legacy: dict[str, Any], sweet: dict[str, Any]) -> dict[str, Any]:
    legacy_max_gpu = max([int(v) for v in legacy.get("per_gpu_running", {}).values()] or [0])
    sweet_max_gpu = max([int(v) for v in sweet.get("per_gpu_running", {}).values()] or [0])
    legacy_agg = float(legacy.get("aggregate_active_rate_step_s") or 0.0)
    sweet_agg = float(sweet.get("aggregate_active_rate_step_s") or 0.0)
    legacy_mean = float(legacy.get("mean_active_rate_step_s") or 0.0)
    sweet_mean = float(sweet.get("mean_active_rate_step_s") or 0.0)
    placement_pass = legacy_max_gpu >= 3 and sweet_max_gpu <= 2 and int(sweet.get("queued_count") or 0) >= 1
    measurement_pass = (
        int(legacy.get("running_with_rate_count") or 0) >= args.min_progress_tasks
        and int(sweet.get("running_with_rate_count") or 0) >= args.min_progress_tasks
    )
    aggregate_ratio = sweet_agg / legacy_agg if legacy_agg > 0 else 0.0
    mean_ratio = sweet_mean / legacy_mean if legacy_mean > 0 else 0.0
    legacy_flow_s = _mean_completion_time_proxy_s(
        task_count=args.task_count,
        steps=args.steps,
        active_slots=int(legacy.get("running_count") or 0),
        mean_rate=legacy_mean,
    )
    sweet_flow_s = _mean_completion_time_proxy_s(
        task_count=args.task_count,
        steps=args.steps,
        active_slots=int(sweet.get("running_count") or 0),
        mean_rate=sweet_mean,
    )
    flow_ratio = legacy_flow_s / sweet_flow_s if sweet_flow_s > 0 else 0.0
    aggregate_pass = measurement_pass and aggregate_ratio >= args.min_aggregate_improvement
    flow_pass = measurement_pass and flow_ratio >= args.min_flow_time_improvement
    if args.performance_objective == "aggregate":
        performance_pass = aggregate_pass
    elif args.performance_objective == "flow_time":
        performance_pass = flow_pass
    else:
        performance_pass = aggregate_pass and flow_pass
    failure_reasons = []
    if not placement_pass:
        failure_reasons.append(
            f"placement failed: legacy max/GPU={legacy_max_gpu}, "
            f"sweetspot max/GPU={sweet_max_gpu}, sweetspot queued={int(sweet.get('queued_count') or 0)}"
        )
    if not measurement_pass:
        failure_reasons.append(
            f"measurement failed: legacy rates={int(legacy.get('running_with_rate_count') or 0)}, "
            f"sweetspot rates={int(sweet.get('running_with_rate_count') or 0)}, "
            f"required={args.min_progress_tasks}"
        )
    if args.performance_objective in ("aggregate", "both") and not aggregate_pass:
        failure_reasons.append(
            f"aggregate objective failed: ratio={aggregate_ratio:.6f} "
            f"< {args.min_aggregate_improvement:.6f}"
        )
    if args.performance_objective in ("flow_time", "both") and not flow_pass:
        failure_reasons.append(
            f"flow-time objective failed: ratio={flow_ratio:.6f} "
            f"< {args.min_flow_time_improvement:.6f}"
        )
    return {
        "run_id": run_id,
        "node": args.node,
        "task_count": args.task_count,
        "size": args.size,
        "steps": args.steps,
        "measure_s": args.measure_s,
        "legacy_allow_over_one_third": True,
        "sweetspot_algorithm": "sweetspot_v1",
        "sweetspot_max_tasks_per_gpu": 2,
        "placement_pass": placement_pass,
        "measurement_pass": measurement_pass,
        "performance_pass": performance_pass,
        "pass": bool(placement_pass and performance_pass),
        "performance_objective": args.performance_objective,
        "aggregate_active_rate_improvement_ratio": aggregate_ratio,
        "aggregate_objective_pass": aggregate_pass,
        "mean_active_rate_improvement_ratio": mean_ratio,
        "legacy_mean_completion_time_proxy_s": legacy_flow_s,
        "sweetspot_mean_completion_time_proxy_s": sweet_flow_s,
        "flow_time_improvement_ratio": flow_ratio,
        "flow_time_objective_pass": flow_pass,
        "min_aggregate_improvement": args.min_aggregate_improvement,
        "min_flow_time_improvement": args.min_flow_time_improvement,
        "failure_reasons": failure_reasons,
        "legacy": legacy,
        "sweetspot": sweet,
    }


def _mean_completion_time_proxy_s(
    *,
    task_count: int,
    steps: int,
    active_slots: int,
    mean_rate: float,
) -> float:
    """Fast-forward identical-job mean flow time from a stable service rate.

    The runner intentionally cancels jobs after a stable measurement window
    instead of waiting for every benchmark to finish.  This proxy computes the
    completion time that the observed mean service rate would imply under a
    nonpreemptive wave schedule with `active_slots` concurrent jobs.
    """
    n = max(0, int(task_count))
    slots = max(0, int(active_slots))
    rate = float(mean_rate or 0.0)
    if n <= 0 or slots <= 0 or rate <= 0:
        return 0.0
    per_wave_s = float(max(1, int(steps))) / rate
    total = 0.0
    remaining = n
    wave = 1
    while remaining > 0:
        batch = min(slots, remaining)
        total += batch * wave * per_wave_s
        remaining -= batch
        wave += 1
    return total / n


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="")
    parser.add_argument("--node", default="jtl110gpu2")
    parser.add_argument("--cwd", default=str(Path.home() / "scheduleurm_bench_cwd"))
    parser.add_argument("--remote-script", default="/tmp/scheduleurm_gpu_progress_benchmark.py")
    parser.add_argument("--python-bin", default="/home/erzhu419/.venvs/resac-jax-gpu1-0438/bin/python")
    parser.add_argument("--task-count", type=int, default=6)
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--size", type=int, default=12288)
    parser.add_argument("--vram-mb", type=int, default=800)
    parser.add_argument("--ram-mb", type=int, default=4096)
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--mem-fraction", type=float, default=0.20)
    parser.add_argument("--warmup-timeout-s", type=int, default=240)
    parser.add_argument("--measure-s", type=int, default=120)
    parser.add_argument("--poll-s", type=int, default=30)
    parser.add_argument("--min-progress-tasks", type=int, default=4)
    parser.add_argument("--min-aggregate-improvement", type=float, default=1.05)
    parser.add_argument("--min-flow-time-improvement", type=float, default=1.05)
    parser.add_argument("--hard-rule-mode", default="clean_bench")
    parser.add_argument(
        "--allow-active-watcher",
        action="store_true",
        help="Allow running even if the systemd scheduler watcher is active.",
    )
    parser.add_argument(
        "--performance-objective",
        choices=("aggregate", "flow_time", "both"),
        default="aggregate",
    )
    args = parser.parse_args()

    run_id = args.run_id or f"module2_sweetspot_ab_{_now_stamp()}"
    manifest_path = init_run(run_id)
    run_dir = manifest_path.parent
    raw_dir = run_dir / "raw"
    _write_json(
        run_dir / "reports" / "runner_config.json",
        {k: getattr(args, k) for k in sorted(vars(args))} | {"run_id": run_id},
    )
    watcher = _watcher_status()
    _record_event(run_dir, "preflight_watcher_status", watcher=watcher)
    if watcher["active"] and not args.allow_active_watcher:
        _write_json(
            run_dir / "reports" / "verdict.json",
            {
                "run_id": run_id,
                "pass": False,
                "failure_reasons": [
                    "systemd scheduler watcher is active; stop it or pass --allow-active-watcher"
                ],
                "watcher": watcher,
            },
        )
        raise SystemExit(
            "scheduler watcher is active; stop it before clean A/B "
            "or pass --allow-active-watcher"
        )
    _run_cmd(["mkdir", "-p", args.cwd], cwd=_repo_root(), raw_dir=raw_dir, label="local_mkdir_cwd")
    _run_cmd(
        ["ssh", args.node, "mkdir", "-p", args.cwd],
        cwd=_repo_root(),
        raw_dir=raw_dir,
        label="remote_mkdir_cwd",
    )
    _run_cmd(
        [
            "rsync",
            "-az",
            str(_repo_root() / "algorithm" / "experiments" / "gpu_progress_benchmark.py"),
            f"{args.node}:{args.remote_script}",
        ],
        cwd=_repo_root(),
        raw_dir=raw_dir,
        label="rsync_benchmark_script",
    )

    all_ids: list[str] = []
    try:
        legacy_ids, legacy_summary = _run_phase(
            phase="legacy",
            algorithm="legacy",
            run_id=run_id,
            run_dir=run_dir,
            args=args,
            allow_legacy_over_one_third=True,
        )
        all_ids.extend(legacy_ids)
        sweet_ids, sweet_summary = _run_phase(
            phase="sweetspot",
            algorithm="sweetspot_v1",
            run_id=run_id,
            run_dir=run_dir,
            args=args,
            allow_legacy_over_one_third=False,
        )
        all_ids.extend(sweet_ids)
        verdict = _verdict(run_id, args, legacy_summary, sweet_summary)
        _write_json(run_dir / "reports" / "verdict.json", verdict)
        _record_event(run_dir, "verdict", verdict=verdict)
        print(json.dumps(verdict, indent=2, sort_keys=True))
        return 0 if verdict["pass"] else 2
    finally:
        active = [
            t.get("id")
            for t in _tasks_by_ids(all_ids)
            if t.get("status") in ("queued", "launching", "running")
        ]
        if active:
            _record_event(run_dir, "final_cleanup_start", active_ids=active)
            _cancel_all(raw_dir, "final_cleanup", [str(t) for t in active])
            _record_event(run_dir, "final_cleanup_done", active_ids=active)


if __name__ == "__main__":
    raise SystemExit(main())
