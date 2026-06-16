"""Build paired Gavel/Scheduleurm service-unit calibration windows.

The output is consumed by ``gavel_service_unit_calibration_gate``.  It compares
Gavel's native simulator makespan on bounded q01/q11 trace windows with the
Scheduleurm measured-service replay makespan for the same task count and unit
definition.  This is a service-unit calibration artifact, not a direct
full-stack scheduler superiority experiment.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from simulation.defaults import build_default_cache
from simulation.service_cache import deterministic_makespan_s
from simulation.tasksets import taskset_by_name

from .gavel_direct_native_smoke import (
    DEFAULT_DEP_TARGET,
    PROTO_FILES,
    REFERENCE_ROOT,
    REPO_ROOT,
    _check_dependency_imports,
    _copy_repo,
    _gavel_env,
    _generate_protobuf_stubs,
)
from .gavel_native_performance_microbaseline import _parse_gavel_log
from .sota_adapters.scheduleurm_to_gavel_trace_converter import build_native_trace_lines


ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_RUN_ROOT = Path("/tmp/scheduleurm_gavel_service_unit_paired_holdout")
DEFAULT_WINDOWS = {
    "q01_gpu_bound_compute": (1, 2, 3, 4, 6, 8),
    "q11_cpu_gpu_coupled": (2, 4, 6, 8, 12, 16),
}


def build_gavel_service_unit_paired_holdout(
    *,
    reference_root: str | Path = REFERENCE_ROOT,
    dependency_target: str | Path = DEFAULT_DEP_TARGET,
    run_root: str | Path = DEFAULT_RUN_ROOT,
    policy: str = "fifo",
    cluster_spec: str = "8:0:0",
    num_gpus_per_server: str = "8:1:1",
    timeout_seconds: int = 45,
    keep_run_dir: bool = True,
) -> dict[str, Any]:
    reference_root = Path(reference_root).expanduser()
    dependency_target = Path(dependency_target).expanduser()
    run_root = Path(run_root).expanduser()
    copied_repo = run_root / "gavel"
    scheduler_dir = copied_repo / "scheduler"

    import_result = _check_dependency_imports(dependency_target)
    copy_result = _copy_repo(reference_root / "gavel", copied_repo)
    stub_result = (
        _generate_protobuf_stubs(scheduler_dir, dependency_target)
        if copy_result.get("pass") else
        {"pass": False, "commands": [], "blocker": "copy failed"}
    )
    rows: list[dict[str, Any]] = []
    if import_result.get("pass") and stub_result.get("pass"):
        cache = build_default_cache()
        for taskset, windows in DEFAULT_WINDOWS.items():
            for idx, job_count in enumerate(windows):
                split = "calibration" if idx < len(windows) // 2 else "holdout"
                rows.append(
                    _paired_window_row(
                        scheduler_dir=scheduler_dir,
                        dependency_target=dependency_target,
                        cache=cache,
                        taskset_name=taskset,
                        job_count=int(job_count),
                        split=split,
                        policy=policy,
                        cluster_spec=cluster_spec,
                        num_gpus_per_server=num_gpus_per_server,
                        timeout_seconds=timeout_seconds,
                    )
                )
    report = {
        "gate": "gavel_service_unit_paired_holdout",
        "reference_root": str(reference_root),
        "run_root": str(run_root),
        "dependency_target": str(dependency_target),
        "dependency_imports": import_result,
        "copy": copy_result,
        "protobuf_stubs": stub_result,
        "proto_files": list(PROTO_FILES),
        "policy": policy,
        "rows": rows,
        "row_count": len(rows),
        "usable_row_count": sum(1 for row in rows if row.get("usable_paired_row")),
        "tasksets": sorted(DEFAULT_WINDOWS),
        "pass": bool(rows) and all(row.get("usable_paired_row") for row in rows),
        "scope": (
            "Paired bounded-window service-unit calibration between Gavel native "
            "simulator makespan and Scheduleurm measured-service replay makespan. "
            "This artifact is not a direct full-stack SOTA superiority result."
        ),
    }
    if not keep_run_dir:
        shutil.rmtree(run_root, ignore_errors=True)
        report["run_root_removed"] = True
    else:
        report["run_root_removed"] = False
    return report


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Gavel Service-Unit Paired Holdout",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `row_count` | {report.get('row_count', 0)} |",
        f"| `usable_row_count` | {report.get('usable_row_count', 0)} |",
        f"| `policy` | `{report.get('policy')}` |",
        "",
        "| Taskset | Split | Jobs | Gavel s | Scheduleurm s | Scale | Usable |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("rows") or []:
        lines.append(
            "| `{taskset}` | `{split}` | {jobs} | {gavel:.6g} | {sched:.6g} | {scale:.6g} | {usable} |".format(
                taskset=row.get("taskset"),
                split=row.get("split"),
                jobs=int(row.get("job_count") or 0),
                gavel=float(row.get("gavel_makespan_s") or 0.0),
                sched=float(row.get("scheduleurm_makespan_s") or 0.0),
                scale=float(row.get("service_unit_scale_scheduleurm_over_gavel") or 0.0),
                usable=str(bool(row.get("usable_paired_row"))).lower(),
            )
        )
    lines.extend(["", "## Scope", "", str(report.get("scope") or ""), ""])
    return "\n".join(lines)


def _paired_window_row(
    *,
    scheduler_dir: Path,
    dependency_target: Path,
    cache: Any,
    taskset_name: str,
    job_count: int,
    split: str,
    policy: str,
    cluster_spec: str,
    num_gpus_per_server: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    taskset = taskset_by_name(taskset_name)
    member = taskset.members[0]
    trace_file = _write_window_trace(scheduler_dir, taskset_name, job_count)
    output_file = scheduler_dir / f"gavel_paired_{taskset_name}_{job_count}_{policy}.log"
    command = [
        sys.executable,
        "scripts/drivers/simulate_scheduler_with_trace.py",
        "-t",
        str(trace_file),
        "-p",
        policy,
        "--throughputs_file",
        "simulation_throughputs.json",
        "-c",
        cluster_spec,
        "--num_gpus_per_server",
        num_gpus_per_server,
        "--seed",
        "0",
        "--time_per_iteration",
        "60",
        "-s",
        "0",
        "-e",
        str(int(job_count)),
    ]
    run = _run_to_file(
        command,
        cwd=scheduler_dir,
        env=_gavel_env(scheduler_dir, dependency_target),
        timeout_seconds=timeout_seconds,
        output_file=output_file,
    )
    parsed = _parse_gavel_log(output_file)
    scheduleurm = _scheduleurm_best_makespan(cache, member, int(job_count))
    gavel_s = float(parsed.get("makespan_s") or 0.0)
    scheduleurm_s = float(scheduleurm.get("scheduleurm_makespan_s") or 0.0)
    usable = (
        run.get("returncode") == 0
        and int(parsed.get("completed_count") or 0) >= int(job_count)
        and gavel_s > 0.0
        and scheduleurm_s > 0.0
    )
    return {
        "taskset": taskset_name,
        "workload_key": member.workload_key,
        "split": split,
        "window_id": f"{taskset_name}:jobs{int(job_count)}:{split}",
        "job_count": int(job_count),
        "policy": policy,
        "cluster_spec": cluster_spec,
        "num_gpus_per_server": num_gpus_per_server,
        "trace_file": str(trace_file),
        "output_file": str(output_file),
        "command": command,
        "returncode": run.get("returncode"),
        "timeout": run.get("timeout"),
        "completed_count": int(parsed.get("completed_count") or 0),
        "gavel_makespan_s": gavel_s,
        "gavel_average_jct_s": float(parsed.get("average_jct_s") or 0.0),
        "scheduleurm_makespan_s": scheduleurm_s,
        "scheduleurm_observed_s": scheduleurm_s,
        "gavel_observed_s": gavel_s,
        "service_unit_scale_scheduleurm_over_gavel": (
            scheduleurm_s / gavel_s if gavel_s > 0.0 else 0.0
        ),
        "usable_paired_row": usable,
        "source": str(output_file),
        **scheduleurm,
    }


def _write_window_trace(scheduler_dir: Path, taskset_name: str, job_count: int) -> Path:
    lines = build_native_trace_lines(taskset_name)[: int(job_count)]
    path = scheduler_dir / f"scheduleurm_paired_{taskset_name}_{int(job_count)}.trace"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _scheduleurm_best_makespan(cache: Any, member: Any, job_count: int) -> dict[str, Any]:
    candidates = []
    for record in cache.profiles(member.workload_key):
        if record.capacity_boundary or float(record.aggregate_rate) <= 0.0:
            continue
        makespan = deterministic_makespan_s(
            task_count=int(job_count),
            total_units=float(member.total_units),
            resource_count=int(member.resource_count),
            profile=int(record.profile),
            aggregate_rate=float(record.aggregate_rate),
        )
        candidates.append((makespan, int(record.profile), record.source))
    if not candidates:
        return {
            "scheduleurm_best_profile": None,
            "scheduleurm_makespan_s": 0.0,
            "scheduleurm_profile_source": "",
        }
    makespan, profile, source = min(candidates, key=lambda row: (row[0], row[1]))
    return {
        "scheduleurm_best_profile": profile,
        "scheduleurm_makespan_s": float(makespan),
        "scheduleurm_profile_source": source,
    }


def _run_to_file(
    command: list[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout_seconds: int,
    output_file: Path,
) -> dict[str, Any]:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", errors="replace") as fh:
        try:
            proc = subprocess.run(
                command,
                cwd=cwd,
                env=dict(env),
                stdout=fh,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=int(timeout_seconds),
                check=False,
            )
            return {"returncode": proc.returncode, "timeout": False}
        except subprocess.TimeoutExpired:
            return {"returncode": None, "timeout": True}


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_gavel_service_unit_paired_holdout(
        reference_root=args.reference_root,
        dependency_target=args.dependency_target,
        run_root=args.run_root,
        policy=args.policy,
        timeout_seconds=args.timeout_seconds,
        keep_run_dir=not args.clean,
    )
    _write_json(args.output, report)
    if args.markdown_output:
        Path(args.markdown_output).expanduser().write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.gavel_service_unit_paired_holdout")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build paired q01/q11 Gavel/Scheduleurm holdout rows")
    build.add_argument("--reference-root", default=str(REFERENCE_ROOT))
    build.add_argument("--dependency-target", default=str(DEFAULT_DEP_TARGET))
    build.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    build.add_argument("--policy", default="fifo")
    build.add_argument("--timeout-seconds", type=int, default=45)
    build.add_argument("--clean", action="store_true")
    build.add_argument("--output", default=str(ARTIFACT_ROOT / "gavel_service_unit_paired_holdout_20260612.json"))
    build.add_argument("--markdown-output", default=str(REPO_ROOT / "md" / "gavel_service_unit_paired_holdout_20260612.md"))
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
