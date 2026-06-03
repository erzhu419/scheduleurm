"""Live fixed-profile service-curve validation.

This runner measures progress service under pinned co-location profiles.  It is
not a new scheduler policy: it submits short benchmark jobs, pins them to a
small set of GPUs, records progress rates, and cancels the jobs after a stable
measurement window.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .sweetspot_ab_validation import (
    _cancel_all,
    _mean_completion_time_proxy_s,
    _now_stamp,
    _parse_submit_id,
    _record_event,
    _record_observation,
    _repo_root,
    _run_cmd,
    _scheduler_cmd,
    _status_refresh,
    _summarize_phase,
    _tasks_by_ids,
    _watcher_status,
    _write_json,
)
from .trace_export import init_run


class ServiceCurveCapacityBoundary(RuntimeError):
    def __init__(self, phase: str, reasons: list[str]):
        self.phase = phase
        self.reasons = list(reasons)
        joined = "; ".join(self.reasons) if self.reasons else "capacity boundary"
        super().__init__(f"{phase} capacity boundary: {joined}")


def _submit_profile(
    *,
    run_id: str,
    phase: str,
    run_dir: Path,
    node: str,
    gpu_plan: list[int],
    cwd: str,
    remote_script: str,
    python_bin: str,
    steps: int,
    size: int,
    vram_mb: int,
    ram_mb: int,
    cpu: int,
    mem_fraction: float,
) -> list[str]:
    raw_dir = run_dir / "raw"
    ids: list[str] = []
    for i, gpu_idx in enumerate(gpu_plan):
        label = f"{run_id}-{phase}-gpu{gpu_idx}-{i}"
        cmd = (
            "OMP_NUM_THREADS=1 "
            "XLA_PYTHON_CLIENT_PREALLOCATE=false "
            f"XLA_PYTHON_CLIENT_MEM_FRACTION={mem_fraction:.3f} "
            f"{python_bin} -u {remote_script} "
            f"--steps {steps} --size {size} --label {label}"
        )
        proc = _scheduler_cmd(
            [
                "submit",
                "--description",
                f"ScheduleurmBench module5 {phase} task {i}",
                "--cmd",
                cmd,
                "--cwd",
                cwd,
                "--signature",
                f"ScheduleurmBench/module5/{run_id}/{phase}/{i}",
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
        _scheduler_cmd(
            [
                "edit",
                tid,
                "--require-gpu",
                str(gpu_idx),
                "--allow-gpu-over-one-third",
            ],
            raw_dir=raw_dir,
            label=f"{phase}_pin_gpu_{gpu_idx}_{i}",
            timeout_s=120,
        )
    _record_event(run_dir, "profile_submitted", phase=phase, ids=ids, gpu_plan=gpu_plan)
    return ids


def _wait_for_profile_progress(
    *,
    phase: str,
    run_dir: Path,
    ids: list[str],
    required: int,
    timeout_s: int,
    poll_s: int,
    min_runtime_unit: int = 1,
    boundary_reason_fn: Any | None = None,
) -> None:
    raw_dir = run_dir / "raw"
    deadline = time.time() + max(1, int(timeout_s))
    attempt = 0
    while True:
        attempt += 1
        _status_refresh(raw_dir, f"{phase}_warmup_status_{attempt:03d}", ids)
        tasks, summary = _record_observation(
            run_dir=run_dir,
            raw_dir=raw_dir,
            phase=phase,
            stage="warmup",
            ids=ids,
            sample_idx=attempt,
        )
        progressed = profile_progress_count(summary, min_runtime_unit=min_runtime_unit)
        boundary_reasons = list(boundary_reason_fn(summary) if boundary_reason_fn else [])
        _record_event(
            run_dir,
            "service_curve_warmup",
            phase=phase,
            attempt=attempt,
            progressed=progressed,
            required=required,
            min_runtime_unit=max(1, int(min_runtime_unit)),
            boundary_reasons=boundary_reasons,
            summary=summary,
        )
        if boundary_reasons:
            _record_event(
                run_dir,
                "service_curve_warmup_capacity_boundary",
                phase=phase,
                attempt=attempt,
                reasons=boundary_reasons,
                summary=summary,
            )
            raise ServiceCurveCapacityBoundary(phase, boundary_reasons)
        if progressed >= required:
            return
        if time.time() >= deadline:
            statuses = {str(t.get("id")): t.get("status") for t in tasks}
            raise RuntimeError(
                f"{phase} warmup timed out: progressed={progressed}, required={required}, "
                f"min_runtime_unit={max(1, int(min_runtime_unit))}, statuses={statuses}"
            )
        time.sleep(max(1, int(poll_s)))


def profile_progress_count(summary: dict[str, Any], *, min_runtime_unit: int = 1) -> int:
    min_unit = max(1, int(min_runtime_unit))
    with_rate = int(summary.get("running_with_rate_count") or 0)
    units = [
        int(x)
        for x in (summary.get("runtime_current_units") or [])
        if int(x) >= min_unit
    ]
    if min_unit <= 1:
        return with_rate
    return min(with_rate, len(units))


def _aggregate_rate(summary: dict[str, Any]) -> float:
    for key in ("aggregate_active_rate_unit_s", "aggregate_active_rate_step_s"):
        if summary.get(key) is not None:
            return float(summary.get(key) or 0.0)
    return 0.0


def _mean_rate(summary: dict[str, Any]) -> float:
    for key in ("mean_active_rate_unit_s", "mean_active_rate_step_s"):
        if summary.get(key) is not None:
            return float(summary.get(key) or 0.0)
    return 0.0


def _measurement_candidates(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    full = [
        dict(summary)
        for summary in samples
        if int(summary.get("running_count") or 0) > 0
        and int(summary.get("running_with_rate_count") or 0) >= int(summary.get("running_count") or 0)
        and _aggregate_rate(summary) > 0.0
    ]
    if full:
        return full
    return [
        dict(summary)
        for summary in samples
        if int(summary.get("running_with_rate_count") or 0) > 0
        and _aggregate_rate(summary) > 0.0
    ]


def select_measurement_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {}
    candidates = _measurement_candidates(samples)
    if not candidates:
        selected = dict(samples[-1])
        selected["summary_selection"] = "last_sample_no_running_rate"
        selected["measurement_sample_count"] = len(samples)
        selected["measurement_valid_sample_count"] = 0
        selected["terminal_status_counts"] = samples[-1].get("status_counts") or {}
        return selected

    rates = sorted(_aggregate_rate(summary) for summary in candidates)
    mid = len(rates) // 2
    median_rate = rates[mid] if len(rates) % 2 else (rates[mid - 1] + rates[mid]) / 2.0
    selected = min(candidates, key=lambda summary: (abs(_aggregate_rate(summary) - median_rate), -_aggregate_rate(summary)))
    last = samples[-1]
    selected["summary_selection"] = "median_running_rate_sample"
    selected["measurement_sample_count"] = len(samples)
    selected["measurement_valid_sample_count"] = len(candidates)
    selected["measurement_aggregate_rate_unit_s_min"] = min(rates)
    selected["measurement_aggregate_rate_unit_s_max"] = max(rates)
    selected["measurement_aggregate_rate_unit_s_median"] = median_rate
    selected["measurement_raw_last_aggregate_rate_unit_s"] = _aggregate_rate(last)
    selected["measurement_raw_last_mean_rate_unit_s"] = _mean_rate(last)
    selected["measurement_raw_last_running_count"] = int(last.get("running_count") or 0)
    selected["measurement_raw_last_running_with_rate_count"] = int(last.get("running_with_rate_count") or 0)
    selected["terminal_status_counts"] = last.get("status_counts") or {}
    return selected


def _measure_profile(
    *,
    phase: str,
    run_dir: Path,
    ids: list[str],
    measure_s: int,
    poll_s: int,
) -> dict[str, Any]:
    raw_dir = run_dir / "raw"
    end = time.time() + max(0, int(measure_s))
    sample_idx = 0
    sample_summaries: list[dict[str, Any]] = []
    while True:
        sample_idx += 1
        _status_refresh(raw_dir, f"{phase}_measure_status_{sample_idx:03d}", ids)
        tasks, last_summary = _record_observation(
            run_dir=run_dir,
            raw_dir=raw_dir,
            phase=phase,
            stage="measure",
            ids=ids,
            sample_idx=sample_idx,
        )
        sample_summary = dict(last_summary)
        sample_summary["measurement_sample_idx"] = sample_idx
        sample_summaries.append(sample_summary)
        _record_event(
            run_dir,
            "service_curve_measure",
            phase=phase,
            sample_idx=sample_idx,
            remaining_s=max(0.0, end - time.time()),
            summary=last_summary,
        )
        if time.time() >= end:
            final_summary = select_measurement_summary(sample_summaries) or _summarize_phase(phase, tasks)
            _write_json(run_dir / "reports" / f"{phase}_summary.json", final_summary)
            return final_summary
        time.sleep(min(max(1, int(poll_s)), max(0.0, end - time.time())))


def _profile_phase_name(count: int) -> str:
    return f"profile_{count}_per_gpu"


def _gpu_plan(gpus: list[int], count_per_gpu: int) -> list[int]:
    out: list[int] = []
    for gpu in gpus:
        out.extend([gpu] * max(0, int(count_per_gpu)))
    return out


def expected_per_gpu_from_plan(gpu_plan: list[int]) -> dict[str, int]:
    expected: dict[str, int] = {}
    for gpu in gpu_plan:
        key = str(int(gpu))
        expected[key] = expected.get(key, 0) + 1
    return dict(sorted(expected.items()))


def annotate_expected_placement(summary: dict[str, Any], gpu_plan: list[int]) -> dict[str, Any]:
    """Attach the intended pinned GPU placement and whether the profile obeyed it."""

    expected = expected_per_gpu_from_plan(gpu_plan)
    actual = {
        str(k): int(v)
        for k, v in (summary.get("per_gpu_running") or {}).items()
        if int(v) > 0
    }
    actual = dict(sorted(actual.items()))
    issues: list[str] = []
    if actual != expected:
        issues.append(f"actual per-GPU running {actual} != expected {expected}")
    out = dict(summary)
    out["expected_per_gpu_running"] = expected
    out["placement_valid"] = not issues
    out["placement_issues"] = issues
    return out


def _makespan_proxy_s(*, task_count: int, steps: int, active_slots: int, mean_rate: float) -> float:
    n = max(0, int(task_count))
    slots = max(0, int(active_slots))
    rate = float(mean_rate or 0.0)
    if n <= 0 or slots <= 0 or rate <= 0:
        return 0.0
    waves = (n + slots - 1) // slots
    return waves * float(max(1, int(steps))) / rate


def _proxy_rows_for_job_count(
    *,
    rows: list[dict[str, Any]],
    steps: int,
    total_jobs: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        mean_rate = float(row.get("mean_active_rate_step_s") or 0.0)
        active_slots = max(1, int(row.get("running_count") or 0))
        enriched = dict(row)
        enriched["proxy_job_count"] = int(total_jobs)
        enriched["mean_flow_proxy_s"] = _mean_completion_time_proxy_s(
            task_count=total_jobs,
            steps=steps,
            active_slots=active_slots,
            mean_rate=mean_rate,
        )
        enriched["makespan_proxy_s"] = _makespan_proxy_s(
            task_count=total_jobs,
            steps=steps,
            active_slots=active_slots,
            mean_rate=mean_rate,
        )
        out.append(enriched)
    return out


def _best_count(rows: list[dict[str, Any]], key: str) -> int | None:
    finite = [
        row for row in rows
        if row.get("placement_valid", True) is not False
        and row.get("capacity_boundary", False) is not True
        and float(row.get(key) or 0.0) > 0.0
    ]
    if not finite:
        return None
    best = min(finite, key=lambda row: row[key])
    return int(best["count_per_gpu"])


def build_service_curve_verdict(
    *,
    run_id: str,
    node: str,
    steps: int,
    total_jobs_for_proxy: int,
    proxy_job_counts: list[int] | None = None,
    expected_sweetspot_count: int,
    min_two_vs_three_gain: float,
    summaries: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    rows = []
    failure_reasons: list[str] = []
    for count, summary in sorted(summaries.items()):
        running = int(summary.get("running_count") or 0)
        with_rate = int(summary.get("running_with_rate_count") or 0)
        expected_per_gpu = {
            str(k): int(v)
            for k, v in (summary.get("expected_per_gpu_running") or {}).items()
        }
        actual_per_gpu = {
            str(k): int(v)
            for k, v in (summary.get("per_gpu_running") or {}).items()
        }
        expected_running = sum(int(v) for v in (expected_per_gpu or actual_per_gpu).values())
        placement_issues = list(summary.get("placement_issues") or [])
        placement_valid = bool(summary.get("placement_valid", True))
        capacity_boundary = bool(summary.get("capacity_boundary", False))
        boundary_reasons = list(summary.get("boundary_reasons") or [])
        if expected_per_gpu and actual_per_gpu != expected_per_gpu:
            placement_valid = False
            if not placement_issues:
                placement_issues.append(
                    f"actual per-GPU running {actual_per_gpu} != expected {expected_per_gpu}"
                )
        mean_rate = float(
            summary.get("mean_active_rate_unit_s")
            if summary.get("mean_active_rate_unit_s") is not None
            else summary.get("mean_active_rate_step_s") or 0.0
        )
        aggregate_rate = float(
            summary.get("aggregate_active_rate_unit_s")
            if summary.get("aggregate_active_rate_unit_s") is not None
            else summary.get("aggregate_active_rate_step_s") or 0.0
        )
        rate_units = sorted(str(x) for x in (summary.get("rate_units") or []) if str(x))
        active_slots = max(1, running)
        flow_proxy = _mean_completion_time_proxy_s(
            task_count=total_jobs_for_proxy,
            steps=steps,
            active_slots=active_slots,
            mean_rate=mean_rate,
        )
        makespan_proxy = _makespan_proxy_s(
            task_count=total_jobs_for_proxy,
            steps=steps,
            active_slots=active_slots,
            mean_rate=mean_rate,
        )
        rows.append({
            "count_per_gpu": count,
            "running_count": running,
            "running_with_rate_count": with_rate,
            "expected_running_from_per_gpu": expected_running,
            "per_gpu_running": summary.get("per_gpu_running") or {},
            "expected_per_gpu_running": expected_per_gpu,
            "placement_valid": placement_valid,
            "placement_issues": placement_issues,
            "capacity_boundary": capacity_boundary,
            "boundary_reasons": boundary_reasons,
            "rate_units": rate_units,
            "mean_active_rate_unit_s": mean_rate,
            "aggregate_active_rate_unit_s": aggregate_rate,
            "mean_active_rate_step_s": mean_rate,
            "aggregate_active_rate_step_s": aggregate_rate,
            "legacy_six_job_flow_time_proxy_s": flow_proxy,
            "legacy_six_job_makespan_proxy_s": makespan_proxy,
            "eviction_count": int(summary.get("eviction_count") or 0),
            "blocked_count": int(summary.get("blocked_count") or 0),
        })
        if capacity_boundary:
            pass
        elif running <= 0 or with_rate < running:
            failure_reasons.append(
                f"profile {count}/GPU did not measure every running task: "
                f"running={running}, with_rate={with_rate}"
            )
        if not placement_valid and not capacity_boundary:
            failure_reasons.append(
                f"profile {count}/GPU placement invalid: {'; '.join(placement_issues)}"
            )
        if with_rate > 0 and len(rate_units) != 1:
            failure_reasons.append(
                f"profile {count}/GPU has ambiguous progress-rate units: {rate_units}"
            )
        if int(summary.get("eviction_count") or 0) > 0 and not capacity_boundary:
            failure_reasons.append(f"profile {count}/GPU had evictions")
        if int(summary.get("blocked_count") or 0) > 0 and not capacity_boundary:
            failure_reasons.append(f"profile {count}/GPU had placement blocks")

    by_count = {row["count_per_gpu"]: row for row in rows}
    gain_2_vs_3 = 0.0
    if 2 in by_count and 3 in by_count:
        r2 = float(by_count[2]["mean_active_rate_step_s"] or 0.0)
        r3 = float(by_count[3]["mean_active_rate_step_s"] or 0.0)
        gain_2_vs_3 = r2 / r3 if r3 > 0 else 0.0
        if gain_2_vs_3 < min_two_vs_three_gain:
            failure_reasons.append(
                f"2/GPU per-task service gain over 3/GPU {gain_2_vs_3:.6f} "
                f"< {min_two_vs_three_gain:.6f}"
            )
    elif float(min_two_vs_three_gain or 0.0) > 0.0:
        failure_reasons.append("profiles 2/GPU and 3/GPU are both required")

    counts = [int(x) for x in (proxy_job_counts or []) if int(x) > 0]
    if not counts:
        counts = [int(total_jobs_for_proxy)]
    if int(total_jobs_for_proxy) not in counts:
        counts.insert(0, int(total_jobs_for_proxy))
    proxy_tables = []
    for n_jobs in counts:
        proxy_rows = _proxy_rows_for_job_count(rows=rows, steps=steps, total_jobs=n_jobs)
        proxy_tables.append({
            "job_count": n_jobs,
            "best_mean_flow_count_per_gpu": _best_count(proxy_rows, "mean_flow_proxy_s"),
            "best_makespan_count_per_gpu": _best_count(proxy_rows, "makespan_proxy_s"),
            "rows": proxy_rows,
        })
    primary_proxy = proxy_tables[0] if proxy_tables else {"rows": []}
    best_count = primary_proxy.get("best_mean_flow_count_per_gpu")
    best_makespan_count = primary_proxy.get("best_makespan_count_per_gpu")
    expectation_checked = int(expected_sweetspot_count) > 0
    if expectation_checked and best_count != int(expected_sweetspot_count):
        failure_reasons.append(
            f"expected flow-time sweetspot {expected_sweetspot_count}/GPU, got {best_count}/GPU"
        )

    return {
        "run_id": run_id,
        "node": node,
        "steps": steps,
        "total_jobs_for_proxy": total_jobs_for_proxy,
        "proxy_job_counts": counts,
        "expected_sweetspot_count": expected_sweetspot_count,
        "expected_sweetspot_checked": expectation_checked,
        "best_flow_time_count_per_gpu": best_count,
        "best_makespan_count_per_gpu": best_makespan_count,
        "two_vs_three_mean_rate_gain": gain_2_vs_3,
        "min_two_vs_three_gain": min_two_vs_three_gain,
        "rows": rows,
        "proxy_tables": proxy_tables,
        "capacity_boundaries": [
            {
                "count_per_gpu": row["count_per_gpu"],
                "per_gpu_running": row.get("per_gpu_running") or {},
                "expected_per_gpu_running": row.get("expected_per_gpu_running") or {},
                "boundary_reasons": row.get("boundary_reasons") or [],
                "placement_issues": row.get("placement_issues") or [],
            }
            for row in rows
            if row.get("capacity_boundary")
        ],
        "failure_reasons": failure_reasons,
        "pass": not failure_reasons,
    }


def _write_curve_markdown(path: Path, verdict: dict[str, Any]) -> None:
    lines = [
        "# Service Curve Validation",
        "",
        f"Run id: `{verdict['run_id']}`",
        f"Node: `{verdict['node']}`",
        f"Pass: `{verdict['pass']}`",
        "",
        "| Count/GPU | Running | With rate | Actual GPUs | Placement | Boundary | Unit | Mean unit/s | Aggregate unit/s | Evictions | Blocks |",
        "|---:|---:|---:|:---|:---|:---|:---|---:|---:|---:|---:|",
    ]
    for row in verdict.get("rows") or []:
        unit = ",".join(row.get("rate_units") or []) or "unknown"
        actual = json.dumps(row.get("per_gpu_running") or {}, sort_keys=True)
        placement = "ok" if row.get("placement_valid", True) is not False else "invalid"
        boundary = "yes" if row.get("capacity_boundary") else "no"
        lines.append(
            f"| {row['count_per_gpu']} | {row['running_count']} | "
            f"{row['running_with_rate_count']} | `{actual}` | {placement} | "
            f"{boundary} | {unit} | "
            f"{row['mean_active_rate_unit_s']:.6f} | "
            f"{row['aggregate_active_rate_unit_s']:.6f} | "
            f"{row['eviction_count']} | {row['blocked_count']} |"
        )
    for table in verdict.get("proxy_tables") or []:
        lines.extend([
            "",
            f"## Completion Proxy: n={table['job_count']}",
            "",
            "| Count/GPU | Mean flow proxy s | Makespan proxy s |",
            "|---:|---:|---:|",
        ])
        for row in table.get("rows") or []:
            lines.append(
                "| {count_per_gpu} | {mean_flow_proxy_s:.3f} | "
                "{makespan_proxy_s:.3f} |".format(**row)
            )
    if verdict.get("failure_reasons"):
        lines.extend(["", "## Failure Reasons", ""])
        lines.extend(f"- {reason}" for reason in verdict["failure_reasons"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="")
    parser.add_argument("--node", default="jtl110gpu2")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--profiles", default="1,2,3")
    parser.add_argument("--cwd", default=str(Path.home() / "scheduleurm_bench_cwd"))
    parser.add_argument("--remote-script", default="/tmp/scheduleurm_gpu_progress_benchmark.py")
    parser.add_argument("--python-bin", default="/home/erzhu419/.venvs/resac-jax-gpu1-0438/bin/python")
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--size", type=int, default=12288)
    parser.add_argument("--vram-mb", type=int, default=800)
    parser.add_argument("--ram-mb", type=int, default=4096)
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--mem-fraction", type=float, default=0.20)
    parser.add_argument("--warmup-timeout-s", type=int, default=240)
    parser.add_argument("--measure-s", type=int, default=90)
    parser.add_argument("--poll-s", type=int, default=30)
    parser.add_argument("--hard-rule-mode", default="clean_bench")
    parser.add_argument(
        "--expected-sweetspot-count",
        type=int,
        default=0,
        help="Optional fixed expectation. 0 means report the measured best profile without failing on mismatch.",
    )
    parser.add_argument("--total-jobs-for-proxy", type=int, default=6)
    parser.add_argument(
        "--proxy-job-counts",
        default="6,12,24",
        help="Comma-separated total-task counts for completion fast-forward tables.",
    )
    parser.add_argument("--min-two-vs-three-gain", type=float, default=1.05)
    parser.add_argument("--allow-active-watcher", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id or f"module5_service_curve_{_now_stamp()}"
    manifest_path = init_run(run_id)
    run_dir = manifest_path.parent
    raw_dir = run_dir / "raw"
    gpus = [int(x) for x in str(args.gpus).split(",") if x.strip()]
    profiles = [int(x) for x in str(args.profiles).split(",") if x.strip()]
    config = {k: getattr(args, k) for k in sorted(vars(args))} | {
        "run_id": run_id,
        "gpus_list": gpus,
        "profiles_list": profiles,
        "proxy_job_counts_list": [int(x) for x in str(args.proxy_job_counts).split(",") if x.strip()],
    }
    _write_json(run_dir / "reports" / "runner_config.json", config)

    watcher = _watcher_status()
    _record_event(run_dir, "preflight_watcher_status", watcher=watcher)
    if watcher["active"] and not args.allow_active_watcher:
        raise SystemExit("scheduler watcher is active; stop it before clean service-curve validation")

    _run_cmd(["mkdir", "-p", args.cwd], cwd=_repo_root(), raw_dir=raw_dir, label="local_mkdir_cwd")
    _run_cmd(["ssh", args.node, "mkdir", "-p", args.cwd], cwd=_repo_root(), raw_dir=raw_dir, label="remote_mkdir_cwd")
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

    summaries: dict[int, dict[str, Any]] = {}
    all_ids: list[str] = []
    try:
        for count in profiles:
            phase = _profile_phase_name(count)
            plan = _gpu_plan(gpus, count)
            _record_event(run_dir, "profile_start", phase=phase, count_per_gpu=count, gpu_plan=plan)
            ids = _submit_profile(
                run_id=run_id,
                phase=phase,
                run_dir=run_dir,
                node=args.node,
                gpu_plan=plan,
                cwd=args.cwd,
                remote_script=args.remote_script,
                python_bin=args.python_bin,
                steps=args.steps,
                size=args.size,
                vram_mb=args.vram_mb,
                ram_mb=args.ram_mb,
                cpu=args.cpu,
                mem_fraction=args.mem_fraction,
            )
            all_ids.extend(ids)
            _record_observation(run_dir=run_dir, raw_dir=raw_dir, phase=phase, stage="post_submit", ids=ids)
            _scheduler_cmd(
                ["dispatch", "--algorithm", "legacy", "--hard-rule-mode", args.hard_rule_mode],
                raw_dir=raw_dir,
                label=f"{phase}_dispatch",
                timeout_s=1200,
            )
            _record_observation(run_dir=run_dir, raw_dir=raw_dir, phase=phase, stage="post_dispatch", ids=ids)
            _wait_for_profile_progress(
                phase=phase,
                run_dir=run_dir,
                ids=ids,
                required=len(ids),
                timeout_s=args.warmup_timeout_s,
                poll_s=args.poll_s,
            )
            summary = _measure_profile(
                phase=phase,
                run_dir=run_dir,
                ids=ids,
                measure_s=args.measure_s,
                poll_s=args.poll_s,
            )
            summary = annotate_expected_placement(summary, plan)
            _write_json(run_dir / "reports" / f"{phase}_summary.json", summary)
            summaries[count] = summary
            _cancel_all(raw_dir, phase, ids)
            _status_refresh(raw_dir, f"{phase}_post_cancel_status", ids)
            _record_observation(run_dir=run_dir, raw_dir=raw_dir, phase=phase, stage="post_cancel", ids=ids)
            _record_event(run_dir, "profile_done", phase=phase, count_per_gpu=count, summary=summary)

        verdict = build_service_curve_verdict(
            run_id=run_id,
            node=args.node,
            steps=args.steps,
            total_jobs_for_proxy=args.total_jobs_for_proxy,
            proxy_job_counts=[int(x) for x in str(args.proxy_job_counts).split(",") if x.strip()],
            expected_sweetspot_count=args.expected_sweetspot_count,
            min_two_vs_three_gain=args.min_two_vs_three_gain,
            summaries=summaries,
        )
        _write_json(run_dir / "reports" / "service_curve_verdict.json", verdict)
        _write_curve_markdown(run_dir / "reports" / "service_curve.md", verdict)
        print(json.dumps(verdict, indent=2, sort_keys=True))
        return 0 if verdict["pass"] else 2
    finally:
        active = [
            str(t.get("id"))
            for t in _tasks_by_ids(all_ids)
            if t.get("status") in ("queued", "launching", "running")
        ]
        if active:
            _record_event(run_dir, "final_cleanup_start", active_ids=active)
            _cancel_all(raw_dir, "final_cleanup", active)
            _record_event(run_dir, "final_cleanup_done", active_ids=active)


if __name__ == "__main__":
    raise SystemExit(main())
