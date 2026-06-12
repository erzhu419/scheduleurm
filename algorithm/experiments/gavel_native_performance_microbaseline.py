"""Native Gavel simulator same-trace microbaseline audit.

This is stronger than an entrypoint smoke test but still narrower than a
direct full-stack Scheduleurm-vs-Gavel comparison.  It runs Gavel's native
trace simulator in an isolated temporary copy on bounded Scheduleurm-exported
trace windows, parses completion-time metrics, and records whether the native
runner produced a clean finite-sample performance row.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

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
from .sota_adapters.scheduleurm_to_gavel_trace_converter import build_native_trace_lines


ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_RUN_ROOT = Path("/tmp/scheduleurm_gavel_native_performance_microbaseline")
DEFAULT_CASES = (
    ("q01_gpu_bound_compute", 4, "8:0:0", "8:1:1"),
    ("q11_cpu_gpu_coupled", 8, "8:0:0", "8:1:1"),
)
DEFAULT_POLICIES = ("fifo",)


@dataclass(frozen=True)
class GavelMicroCase:
    taskset: str
    job_count: int
    cluster_spec: str
    num_gpus_per_server: str


def build_gavel_native_performance_microbaseline(
    *,
    reference_root: str | Path = REFERENCE_ROOT,
    dependency_target: str | Path = DEFAULT_DEP_TARGET,
    run_root: str | Path = DEFAULT_RUN_ROOT,
    policies: tuple[str, ...] = DEFAULT_POLICIES,
    timeout_seconds: int = 25,
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

    cases = [
        GavelMicroCase(
            taskset=name,
            job_count=int(count),
            cluster_spec=cluster_spec,
            num_gpus_per_server=num_gpus_per_server,
        )
        for name, count, cluster_spec, num_gpus_per_server in DEFAULT_CASES
    ]
    rows = []
    if import_result.get("pass") and stub_result.get("pass"):
        for case in cases:
            trace_file = _write_short_trace(scheduler_dir, case)
            for policy in policies:
                rows.append(_run_case(
                    scheduler_dir=scheduler_dir,
                    dependency_target=dependency_target,
                    case=case,
                    policy=policy,
                    trace_file=trace_file,
                    timeout_seconds=timeout_seconds,
                ))
    else:
        for case in cases:
            rows.append({
                "taskset": case.taskset,
                "job_count": case.job_count,
                "policy": "",
                "status": "NOT_RUN",
                "usable_native_performance_sample": False,
                "reason": "dependency_or_stub_preparation_failed",
            })

    usable_rows = [row for row in rows if row.get("usable_native_performance_sample")]
    report = {
        "gate": "gavel_native_performance_microbaseline",
        "reference_root": str(reference_root),
        "run_root": str(run_root),
        "copied_repo": str(copied_repo),
        "dependency_target": str(dependency_target),
        "dependency_imports": import_result,
        "copy": copy_result,
        "protobuf_stubs": stub_result,
        "proto_files": list(PROTO_FILES),
        "policies": list(policies),
        "rows": rows,
        "usable_native_performance_sample_count": len(usable_rows),
        "native_gavel_simulator_microbaseline_ready": len(usable_rows) == len(rows) and bool(rows),
        "same_workload_trace_schema_ready": bool(usable_rows),
        "service_unit_equivalence_ready": False,
        "direct_full_stack_performance_ready": False,
        "pass": bool(usable_rows),
        "blockers": _blockers(rows),
        "scope": (
            "Runs Gavel's native trace simulator on bounded Scheduleurm-exported trace "
            "windows in an isolated copy.  A pass gives native simulator completion/JCT "
            "evidence for those finite windows only.  It does not validate Scheduleurm "
            "measured-service units against Gavel's throughput tables, and it is not a "
            "direct full-stack production-system comparison."
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
        "# Gavel Native Performance Microbaseline",
        "",
        "## Summary",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `usable_native_performance_sample_count` | {report.get('usable_native_performance_sample_count', 0)} |",
        f"| `native_gavel_simulator_microbaseline_ready` | {str(bool(report.get('native_gavel_simulator_microbaseline_ready'))).lower()} |",
        f"| `same_workload_trace_schema_ready` | {str(bool(report.get('same_workload_trace_schema_ready'))).lower()} |",
        f"| `service_unit_equivalence_ready` | {str(bool(report.get('service_unit_equivalence_ready'))).lower()} |",
        f"| `direct_full_stack_performance_ready` | {str(bool(report.get('direct_full_stack_performance_ready'))).lower()} |",
        "",
        "## Native Rows",
        "",
        "| Taskset | Policy | Jobs | Status | Completed | Avg JCT s | Makespan s | Utilization | Usable |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("rows") or []:
        lines.append(
            "| `{taskset}` | `{policy}` | {jobs} | `{status}` | {completed} | {avg:.6g} | {makespan:.6g} | {util:.6g} | {usable} |".format(
                taskset=row.get("taskset"),
                policy=row.get("policy"),
                jobs=int(row.get("job_count") or 0),
                status=row.get("status"),
                completed=int(row.get("completed_count") or 0),
                avg=float(row.get("average_jct_s") or 0.0),
                makespan=float(row.get("makespan_s") or 0.0),
                util=float(row.get("cluster_utilization") or 0.0),
                usable=str(bool(row.get("usable_native_performance_sample"))).lower(),
            )
        )
    lines.extend(["", "## Blockers", ""])
    for item in report.get("blockers") or []:
        lines.append(f"- {item}")
    lines.extend(["", "## Scope", "", str(report.get("scope") or ""), ""])
    return "\n".join(lines)


def _write_short_trace(scheduler_dir: Path, case: GavelMicroCase) -> Path:
    lines = build_native_trace_lines(case.taskset)[:case.job_count]
    path = scheduler_dir / f"scheduleurm_{case.taskset}_{case.job_count}.trace"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _run_case(
    *,
    scheduler_dir: Path,
    dependency_target: Path,
    case: GavelMicroCase,
    policy: str,
    trace_file: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    out_file = scheduler_dir / f"gavel_micro_{case.taskset}_{case.job_count}_{policy}.log"
    cmd = [
        sys.executable,
        "scripts/drivers/simulate_scheduler_with_trace.py",
        "-t",
        str(trace_file),
        "-p",
        policy,
        "--throughputs_file",
        "simulation_throughputs.json",
        "-c",
        case.cluster_spec,
        "--num_gpus_per_server",
        case.num_gpus_per_server,
        "--seed",
        "0",
        "--time_per_iteration",
        "60",
        "-s",
        "0",
        "-e",
        str(case.job_count),
    ]
    result = _run_to_file(
        cmd,
        cwd=scheduler_dir,
        env=_gavel_env(scheduler_dir, dependency_target),
        timeout_seconds=timeout_seconds,
        output_file=out_file,
    )
    parsed = _parse_gavel_log(out_file)
    status = _status(result, parsed, expected=case.job_count)
    usable = (
        status == "PASS"
        and parsed.get("jct_count") == case.job_count
        and parsed.get("completed_count", 0) >= case.job_count
    )
    return {
        "taskset": case.taskset,
        "policy": policy,
        "job_count": case.job_count,
        "cluster_spec": case.cluster_spec,
        "num_gpus_per_server": case.num_gpus_per_server,
        "trace_file": str(trace_file),
        "output_file": str(out_file),
        "command": cmd,
        **result,
        **parsed,
        "status": status,
        "usable_native_performance_sample": usable,
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
                env=dict(env or os.environ),
                stdout=fh,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            return {"returncode": proc.returncode, "timeout": False, "timeout_seconds": timeout_seconds}
        except subprocess.TimeoutExpired:
            return {"returncode": None, "timeout": True, "timeout_seconds": timeout_seconds}
        except OSError as exc:
            fh.write(f"\nOSERROR: {exc}\n")
            return {"returncode": None, "timeout": False, "timeout_seconds": timeout_seconds, "oserror": str(exc)}


def _parse_gavel_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    completed_values = [int(x) for x in re.findall(r"Number of completed jobs: (\d+)", text)]
    jcts = [float(x) for x in re.findall(r"^Job \d+: ([0-9.]+)$", text, flags=re.MULTILINE)]
    avg_match = re.findall(r"Average job completion time: ([0-9.]+) seconds", text)
    total_match = re.findall(r"Total duration: ([0-9.]+) seconds", text)
    util_match = re.findall(r"Cluster utilization: ([0-9.]+)", text)
    return {
        "log_bytes": len(text.encode("utf-8", errors="replace")),
        "completed_count": max(completed_values) if completed_values else 0,
        "completed_count_tail": completed_values[-5:],
        "jct_count": len(jcts),
        "average_jct_s": float(avg_match[-1]) if avg_match else (sum(jcts) / len(jcts) if jcts else 0.0),
        "makespan_s": float(total_match[-1]) if total_match else (max(jcts) if jcts else 0.0),
        "cluster_utilization": float(util_match[-1]) if util_match else 0.0,
        "jct_tail": jcts[-8:],
    }


def _status(result: Mapping[str, Any], parsed: Mapping[str, Any], *, expected: int) -> str:
    if result.get("returncode") == 0 and parsed.get("completed_count", 0) >= expected:
        return "PASS"
    if result.get("timeout") and parsed.get("completed_count", 0) >= expected:
        return "COMPLETION_OBSERVED_BUT_TIMEOUT"
    if result.get("timeout") and parsed.get("completed_count", 0) == 0:
        return "TIMEOUT_NON_PROGRESS"
    if result.get("timeout"):
        return "TIMEOUT_PARTIAL_PROGRESS"
    return "FAIL"


def _blockers(rows: list[Mapping[str, Any]]) -> list[str]:
    blockers = []
    for row in rows:
        if not row.get("usable_native_performance_sample"):
            blockers.append(
                f"{row.get('taskset')} policy {row.get('policy')} is not a usable native performance row: {row.get('status')}"
            )
    blockers.append(
        "Gavel native simulator performance rows are not yet Scheduleurm measured-service-unit equivalence."
    )
    blockers.append(
        "This gate does not run Pollux, Sia, IADeep, or Salus full production stacks."
    )
    return blockers


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    policies = tuple(x.strip() for x in args.policies.split(",") if x.strip())
    report = build_gavel_native_performance_microbaseline(
        reference_root=args.reference_root,
        dependency_target=args.dependency_target,
        run_root=args.run_root,
        policies=policies,
        timeout_seconds=args.timeout_seconds,
        keep_run_dir=not args.clean,
    )
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.gavel_native_performance_microbaseline")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Run bounded native Gavel same-trace microbaseline")
    build.add_argument("--reference-root", default=str(REFERENCE_ROOT))
    build.add_argument("--dependency-target", default=str(DEFAULT_DEP_TARGET))
    build.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    build.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    build.add_argument("--timeout-seconds", type=int, default=25)
    build.add_argument("--clean", action="store_true")
    build.add_argument("--output", default=str(ARTIFACT_ROOT / "gavel_native_performance_microbaseline_20260612.json"))
    build.add_argument("--markdown-output", default=str(REPO_ROOT / "md" / "gavel_native_performance_microbaseline_20260612.md"))
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
