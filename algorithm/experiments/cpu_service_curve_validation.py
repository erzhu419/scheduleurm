"""CPU/light fixed-profile service-curve validation."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

from .service_curve_validation import _measure_profile, _wait_for_profile_progress
from .sweetspot_ab_validation import (
    _cancel_all,
    _now_stamp,
    _parse_submit_id,
    _record_event,
    _record_observation,
    _repo_root,
    _run_cmd,
    _scheduler_cmd,
    _status_refresh,
    _tasks_by_ids,
    _watcher_status,
    _write_json,
)
from .trace_export import init_run


def _phase_name(count: int) -> str:
    return f"profile_{int(count)}_per_resource"


def _deploy_script(node: str, remote_script: str, raw_dir: Path) -> None:
    src = _repo_root() / "algorithm" / "experiments" / "cpu_progress_benchmark.py"
    if node == "local":
        dst = Path(remote_script)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        _record_event(raw_dir.parent, "cpu_benchmark_deployed", node=node, remote_script=remote_script)
        return
    _run_cmd(
        ["rsync", "-az", str(src), f"{node}:{remote_script}"],
        cwd=_repo_root(),
        raw_dir=raw_dir,
        label="rsync_cpu_benchmark_script",
    )


def _mkdir_target(node: str, cwd: str, raw_dir: Path) -> None:
    if node == "local":
        _run_cmd(["mkdir", "-p", cwd], cwd=_repo_root(), raw_dir=raw_dir, label="local_mkdir_cwd")
    else:
        _run_cmd(["ssh", node, "mkdir", "-p", cwd], cwd=_repo_root(), raw_dir=raw_dir, label="remote_mkdir_cwd")


def _submit_profile(
    *,
    run_id: str,
    phase: str,
    run_dir: Path,
    node: str,
    count: int,
    cwd: str,
    remote_script: str,
    python_bin: str,
    steps: int,
    mode: str,
    work_items: int,
    sleep_s: float,
    ram_mb: int,
    cpu: int,
    project: str,
    signature_prefix: str,
) -> list[str]:
    raw_dir = run_dir / "raw"
    ids: list[str] = []
    for i in range(max(1, int(count))):
        label = f"{run_id}-{phase}-{i}"
        cmd = (
            f"{python_bin} -u {remote_script} "
            f"--steps {int(steps)} --mode {mode} --work-items {int(work_items)} "
            f"--sleep-s {float(sleep_s):.6f} --label {label}"
        )
        proc = _scheduler_cmd(
            [
                "submit",
                "--description",
                f"Scheduleurm CPU curve {phase} task {i}",
                "--cmd",
                cmd,
                "--cwd",
                cwd,
                "--signature",
                f"{signature_prefix}/{run_id}/{phase}/{i}",
                "--vram",
                "0",
                "--ram-mb",
                str(ram_mb),
                "--cpu",
                str(cpu),
                "--priority",
                "high",
                "--project",
                project,
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
        ids.append(_parse_submit_id(proc.stdout))
    _record_event(run_dir, "cpu_profile_submitted", phase=phase, ids=ids, count=count)
    return ids


def _annotate_summary(summary: dict[str, Any], *, expected: int) -> dict[str, Any]:
    out = dict(summary)
    out["expected_parallel_running"] = int(expected)
    out["placement_valid"] = int(out.get("running_count") or 0) == int(expected)
    out["placement_issues"] = [] if out["placement_valid"] else [
        f"running_count {int(out.get('running_count') or 0)} != expected {int(expected)}"
    ]
    return out


def _build_verdict(
    *,
    run_id: str,
    node: str,
    steps: int,
    mode: str,
    summaries: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    rows = []
    failures: list[str] = []
    for count, summary in sorted(summaries.items()):
        boundary = bool(summary.get("capacity_boundary"))
        running = int(summary.get("running_count") or 0)
        with_rate = int(summary.get("running_with_rate_count") or 0)
        aggregate = float(
            summary.get("measurement_aggregate_rate_unit_s_median")
            or summary.get("aggregate_active_rate_unit_s")
            or 0.0
        )
        row = {
            "count_per_resource": int(count),
            "running_count": running,
            "running_with_rate_count": with_rate,
            "aggregate_active_rate_unit_s": aggregate,
            "mean_active_rate_unit_s": aggregate / float(running) if running > 0 else 0.0,
            "rate_units": sorted(str(x) for x in (summary.get("rate_units") or []) if str(x)),
            "blocked_count": int(summary.get("blocked_count") or 0),
            "eviction_count": int(summary.get("eviction_count") or 0),
            "capacity_boundary": boundary,
            "boundary_reasons": list(summary.get("boundary_reasons") or []),
            "placement_valid": bool(summary.get("placement_valid", True)),
            "placement_issues": list(summary.get("placement_issues") or []),
        }
        rows.append(row)
        if boundary:
            continue
        if running != int(count) or with_rate < running:
            failures.append(f"profile {count} incomplete: running={running}, with_rate={with_rate}")
        if row["blocked_count"] > 0:
            failures.append(f"profile {count} had blocked tasks")
        if row["eviction_count"] > 0:
            failures.append(f"profile {count} had evictions")
        if not row["placement_valid"]:
            failures.append(f"profile {count} placement invalid: {'; '.join(row['placement_issues'])}")
        if len(row["rate_units"]) != 1:
            failures.append(f"profile {count} has ambiguous units: {row['rate_units']}")
    best = None
    valid = [
        row for row in rows
        if row["aggregate_active_rate_unit_s"] > 0
        and row["placement_valid"]
        and not row["capacity_boundary"]
    ]
    if valid:
        best = max(valid, key=lambda row: row["aggregate_active_rate_unit_s"])["count_per_resource"]
    return {
        "run_id": run_id,
        "node": node,
        "mode": mode,
        "steps": int(steps),
        "rows": rows,
        "capacity_boundaries": [
            {
                "count_per_resource": row["count_per_resource"],
                "running_count": row["running_count"],
                "running_with_rate_count": row["running_with_rate_count"],
                "blocked_count": row["blocked_count"],
                "boundary_reasons": row["boundary_reasons"],
                "placement_issues": row["placement_issues"],
            }
            for row in rows
            if row.get("capacity_boundary")
        ],
        "best_aggregate_count_per_resource": best,
        "failure_reasons": failures,
        "pass": not failures,
    }


def _write_markdown(path: Path, verdict: dict[str, Any]) -> None:
    lines = [
        "# CPU Service Curve Validation",
        "",
        f"Run id: `{verdict['run_id']}`",
        f"Node: `{verdict['node']}`",
        f"Mode: `{verdict['mode']}`",
        f"Pass: `{verdict['pass']}`",
        "",
        "| Count | Running | With rate | Boundary | Unit | Aggregate unit/s | Mean unit/s | Blocks | Evictions |",
        "|---:|---:|---:|:---|:---|---:|---:|---:|---:|",
    ]
    for row in verdict.get("rows") or []:
        unit = ",".join(row.get("rate_units") or []) or "unknown"
        lines.append(
            f"| {row['count_per_resource']} | {row['running_count']} | "
            f"{row['running_with_rate_count']} | "
            f"{'yes' if row.get('capacity_boundary') else 'no'} | {unit} | "
            f"{row['aggregate_active_rate_unit_s']:.6f} | "
            f"{row['mean_active_rate_unit_s']:.6f} | "
            f"{row['blocked_count']} | {row['eviction_count']} |"
        )
    if verdict.get("failure_reasons"):
        lines.extend(["", "## Failure Reasons", ""])
        lines.extend(f"- {x}" for x in verdict["failure_reasons"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="")
    parser.add_argument("--node", default="local")
    parser.add_argument("--profiles", default="1,2,4,8,16")
    parser.add_argument("--cwd", default=str(Path.home() / "scheduleurm_bench_cwd"))
    parser.add_argument("--remote-script", default="/tmp/scheduleurm_cpu_progress_benchmark.py")
    parser.add_argument("--python-bin", default="python3")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--mode", choices=("light", "cpu"), default="light")
    parser.add_argument("--work-items", type=int, default=10000)
    parser.add_argument("--sleep-s", type=float, default=0.0)
    parser.add_argument("--ram-mb", type=int, default=512)
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--warmup-timeout-s", type=int, default=180)
    parser.add_argument("--measure-s", type=int, default=60)
    parser.add_argument("--poll-s", type=int, default=15)
    parser.add_argument("--hard-rule-mode", default="clean_bench")
    parser.add_argument("--project", default="ScheduleurmBench")
    parser.add_argument("--signature-prefix", default="ScheduleurmBench/cpu_service_curve")
    parser.add_argument("--allow-active-watcher", action="store_true")
    parser.add_argument("--stop-on-capacity-boundary", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id or f"module_cpu_curve_{_now_stamp()}"
    manifest_path = init_run(run_id)
    run_dir = manifest_path.parent
    raw_dir = run_dir / "raw"
    profiles = [int(x) for x in str(args.profiles).split(",") if x.strip()]
    if not profiles:
        raise SystemExit("--profiles must be nonempty")

    _write_json(
        run_dir / "reports" / "runner_config.json",
        {k: getattr(args, k) for k in sorted(vars(args))} | {
            "run_id": run_id,
            "profiles_list": profiles,
        },
    )
    watcher = _watcher_status()
    _record_event(run_dir, "preflight_watcher_status", watcher=watcher)
    if watcher["active"] and not args.allow_active_watcher:
        raise SystemExit("scheduler watcher is active; stop it before clean CPU service-curve validation")

    _mkdir_target(args.node, args.cwd, raw_dir)
    _deploy_script(args.node, args.remote_script, raw_dir)

    summaries: dict[int, dict[str, Any]] = {}
    all_ids: list[str] = []
    try:
        for count in profiles:
            phase = _phase_name(count)
            _record_event(run_dir, "cpu_profile_start", phase=phase, count=count)
            ids = _submit_profile(
                run_id=run_id,
                phase=phase,
                run_dir=run_dir,
                node=args.node,
                count=count,
                cwd=args.cwd,
                remote_script=args.remote_script,
                python_bin=args.python_bin,
                steps=args.steps,
                mode=args.mode,
                work_items=args.work_items,
                sleep_s=args.sleep_s,
                ram_mb=args.ram_mb,
                cpu=args.cpu,
                project=args.project,
                signature_prefix=args.signature_prefix,
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
            try:
                _wait_for_profile_progress(
                    phase=phase,
                    run_dir=run_dir,
                    ids=ids,
                    required=len(ids),
                    timeout_s=args.warmup_timeout_s,
                    poll_s=args.poll_s,
                )
            except RuntimeError as exc:
                if not args.stop_on_capacity_boundary:
                    raise
                _status_refresh(raw_dir, f"{phase}_capacity_boundary_status", ids)
                _tasks, summary = _record_observation(
                    run_dir=run_dir,
                    raw_dir=raw_dir,
                    phase=phase,
                    stage="capacity_boundary",
                    ids=ids,
                )
                summary = _annotate_summary(summary, expected=len(ids))
                summary["capacity_boundary"] = True
                summary["boundary_reasons"] = [str(exc)]
                _write_json(run_dir / "reports" / f"{phase}_summary.json", summary)
                summaries[count] = summary
                _record_event(
                    run_dir,
                    "cpu_profile_capacity_boundary",
                    phase=phase,
                    count=count,
                    summary=summary,
                    reason=str(exc),
                )
                _cancel_all(raw_dir, phase, ids)
                _status_refresh(raw_dir, f"{phase}_post_cancel_status", ids)
                _record_observation(run_dir=run_dir, raw_dir=raw_dir, phase=phase, stage="post_cancel", ids=ids)
                break
            summary = _measure_profile(
                phase=phase,
                run_dir=run_dir,
                ids=ids,
                measure_s=args.measure_s,
                poll_s=args.poll_s,
            )
            summary = _annotate_summary(summary, expected=len(ids))
            _write_json(run_dir / "reports" / f"{phase}_summary.json", summary)
            summaries[count] = summary
            _cancel_all(raw_dir, phase, ids)
            _status_refresh(raw_dir, f"{phase}_post_cancel_status", ids)
            _record_observation(run_dir=run_dir, raw_dir=raw_dir, phase=phase, stage="post_cancel", ids=ids)
            _record_event(run_dir, "cpu_profile_done", phase=phase, count=count, summary=summary)

        verdict = _build_verdict(run_id=run_id, node=args.node, steps=args.steps, mode=args.mode, summaries=summaries)
        _write_json(run_dir / "reports" / "service_curve_verdict.json", verdict)
        _write_markdown(run_dir / "reports" / "service_curve.md", verdict)
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
        time.sleep(0.1)


if __name__ == "__main__":
    raise SystemExit(main())
