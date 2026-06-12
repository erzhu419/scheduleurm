"""Isolated native-smoke audit for the Gavel external baseline.

This script deliberately runs only in a temporary copy of the cloned Gavel
repository.  It never edits ``reference/repos/gavel`` and never touches the
live Scheduleurm scheduler.  The audit separates four facts:

* Python dependencies are importable from an isolated target directory;
* protobuf RPC stubs can be generated in the copied Gavel tree;
* a native Gavel entrypoint can start far enough to print help;
* the native generated-jobs simulator can complete a bounded run;
* a bounded native trace simulation can make progress.

Only the last item would move us toward a direct full-stack SOTA baseline.  A
dependency/stub/help pass is useful engineering evidence, but it is not a
performance comparison and must not be reported as one.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from .sota_adapters.scheduleurm_to_gavel_trace_converter import build_native_trace_lines


REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = REPO_ROOT / "reference" / "repos"
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_DEP_TARGET = Path("/tmp/scheduleurm_gavel_pydeps")
DEFAULT_RUN_ROOT = Path("/tmp/scheduleurm_gavel_direct_native_smoke")

PIP_PACKAGES = (
    "grpcio",
    "grpcio-tools",
    "func_timeout",
    "cvxpy",
    "matrix_completion",
    "scipy",
)

IMPORT_MODULES = (
    "grpc",
    "grpc_tools",
    "func_timeout",
    "cvxpy",
    "matrix_completion",
    "scipy",
)

PROTO_FILES = (
    "iterator_to_scheduler.proto",
    "scheduler_to_worker.proto",
    "worker_to_scheduler.proto",
    "common.proto",
    "enums.proto",
)


def build_gavel_direct_native_smoke(
    *,
    reference_root: str | Path = REFERENCE_ROOT,
    dependency_target: str | Path = DEFAULT_DEP_TARGET,
    run_root: str | Path = DEFAULT_RUN_ROOT,
    install_deps: bool = False,
    run_native_trace: bool = True,
    timeout_seconds: int = 20,
    keep_run_dir: bool = True,
) -> dict[str, Any]:
    reference_root = Path(reference_root).expanduser()
    dependency_target = Path(dependency_target).expanduser()
    run_root = Path(run_root).expanduser()
    source_repo = reference_root / "gavel"
    copied_repo = run_root / "gavel"
    scheduler_dir = copied_repo / "scheduler"

    install_result = None
    if install_deps:
        install_result = _install_dependencies(dependency_target, timeout_seconds=max(60, timeout_seconds * 6))

    import_result = _check_dependency_imports(dependency_target)
    copy_result = _copy_repo(source_repo, copied_repo)
    stub_result = _generate_protobuf_stubs(scheduler_dir, dependency_target) if copy_result["pass"] else {
        "pass": False,
        "commands": [],
        "blocker": "Gavel repository copy failed",
    }
    help_result = _entrypoint_help_smoke(scheduler_dir, dependency_target, timeout_seconds) if stub_result["pass"] else {
        "pass": False,
        "status": "NOT_RUN",
        "blocker": "protobuf stub generation failed",
    }
    generated_result = _generated_jobs_smoke(scheduler_dir, dependency_target, timeout_seconds) if stub_result["pass"] else {
        "pass": False,
        "status": "NOT_RUN",
        "blocker": "protobuf stub generation failed",
    }
    trace_result = (
        _native_trace_smoke(scheduler_dir, dependency_target, timeout_seconds)
        if stub_result["pass"] and run_native_trace else
        {
            "pass": False,
            "status": "NOT_RUN",
            "blocker": "native trace smoke disabled" if not run_native_trace else "protobuf stub generation failed",
        }
    )
    scheduleurm_trace_result = (
        _scheduleurm_native_trace_seed_smoke(scheduler_dir, dependency_target, timeout_seconds)
        if stub_result["pass"] and run_native_trace else
        {
            "pass": False,
            "status": "NOT_RUN",
            "blocker": "native trace smoke disabled" if not run_native_trace else "protobuf stub generation failed",
        }
    )

    dependency_import_pass = bool(import_result["pass"])
    protobuf_stub_generation_pass = bool(stub_result["pass"])
    entrypoint_help_pass = bool(help_result["pass"])
    native_generated_jobs_pass = bool(generated_result["pass"])
    direct_native_trace_pass = bool(trace_result["pass"])
    scheduleurm_native_trace_seed_pass = bool(scheduleurm_trace_result["pass"])
    same_workload_direct_baseline_ready = False
    blockers: list[str] = []
    if not dependency_import_pass:
        blockers.append("isolated Gavel dependency imports are incomplete")
    if not protobuf_stub_generation_pass:
        blockers.append("Gavel protobuf stubs cannot be generated in the isolated copy")
    if not entrypoint_help_pass:
        blockers.append("Gavel native entrypoint help smoke does not pass")
    if not native_generated_jobs_pass:
        blockers.append("Gavel generated-jobs native simulator smoke does not pass")
    if not direct_native_trace_pass:
        blockers.append("Gavel bounded native trace smoke is not a usable direct baseline")
    if not scheduleurm_native_trace_seed_pass:
        blockers.append("Scheduleurm-to-Gavel native trace seed smoke does not pass")
    blockers.append(
        "Scheduleurm-to-Gavel native trace seed is a compatibility mapping; measured service-unit and policy-semantics equivalence still require validation before direct SOTA comparison"
    )

    report = {
        "gate": "gavel_direct_native_smoke",
        "reference_root": str(reference_root),
        "source_repo": str(source_repo),
        "run_root": str(run_root),
        "copied_repo": str(copied_repo),
        "dependency_target": str(dependency_target),
        "install_deps": bool(install_deps),
        "install_result": install_result,
        "dependency_imports": import_result,
        "copy": copy_result,
        "protobuf_stubs": stub_result,
        "entrypoint_help": help_result,
        "native_generated_jobs_smoke": generated_result,
        "native_trace_smoke": trace_result,
        "scheduleurm_native_trace_seed_smoke": scheduleurm_trace_result,
        "dependency_import_pass": dependency_import_pass,
        "protobuf_stub_generation_pass": protobuf_stub_generation_pass,
        "entrypoint_help_pass": entrypoint_help_pass,
        "native_generated_jobs_pass": native_generated_jobs_pass,
        "direct_native_trace_pass": direct_native_trace_pass,
        "scheduleurm_native_trace_seed_pass": scheduleurm_native_trace_seed_pass,
        "same_workload_direct_baseline_ready": same_workload_direct_baseline_ready,
        "pass": (
            dependency_import_pass
            and protobuf_stub_generation_pass
            and entrypoint_help_pass
            and native_generated_jobs_pass
        ),
        "blockers": blockers,
        "scope": (
            "Isolated Gavel native-stack smoke audit.  A pass means the local clone can be "
            "prepared and its entrypoint can start in a temporary copy.  It does not mean "
            "Gavel has been run as a same-workload full-stack Scheduleurm baseline."
        ),
    }

    if not keep_run_dir:
        shutil.rmtree(run_root, ignore_errors=True)
        report["run_root_removed"] = True
    else:
        report["run_root_removed"] = False
    return report


def markdown_report(report: Mapping[str, Any]) -> str:
    trace = report.get("native_trace_smoke") or {}
    generated = report.get("native_generated_jobs_smoke") or {}
    scheduleurm_trace = report.get("scheduleurm_native_trace_seed_smoke") or {}
    imports = report.get("dependency_imports") or {}
    lines = [
        "# Gavel Direct Native Smoke",
        "",
        "## Summary",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `dependency_import_pass` | {str(bool(report.get('dependency_import_pass'))).lower()} |",
        f"| `protobuf_stub_generation_pass` | {str(bool(report.get('protobuf_stub_generation_pass'))).lower()} |",
        f"| `entrypoint_help_pass` | {str(bool(report.get('entrypoint_help_pass'))).lower()} |",
        f"| `native_generated_jobs_pass` | {str(bool(report.get('native_generated_jobs_pass'))).lower()} |",
        f"| `direct_native_trace_pass` | {str(bool(report.get('direct_native_trace_pass'))).lower()} |",
        f"| `scheduleurm_native_trace_seed_pass` | {str(bool(report.get('scheduleurm_native_trace_seed_pass'))).lower()} |",
        f"| `same_workload_direct_baseline_ready` | {str(bool(report.get('same_workload_direct_baseline_ready'))).lower()} |",
        "",
        "## Dependency Imports",
        "",
        "| Module | Importable | Detail |",
        "|---|---:|---|",
    ]
    for row in imports.get("modules") or []:
        lines.append(
            f"| `{row.get('module')}` | {str(bool(row.get('ok'))).lower()} | `{_md_cell(row.get('detail') or '')}` |"
        )
    lines.extend([
        "",
        "## Generated-Jobs Native Smoke",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| status | `{generated.get('status')}` |",
        f"| returncode | `{generated.get('returncode')}` |",
        f"| command | `{_md_cell(' '.join(generated.get('command') or []))}` |",
        "",
        "## Scheduleurm Native Trace Seed Smoke",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| status | `{scheduleurm_trace.get('status')}` |",
        f"| returncode | `{scheduleurm_trace.get('returncode')}` |",
        f"| command | `{_md_cell(' '.join(scheduleurm_trace.get('command') or []))}` |",
        f"| trace file | `{_md_cell(scheduleurm_trace.get('trace_file') or '')}` |",
        "",
        "## Native Trace Smoke",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| status | `{trace.get('status')}` |",
        f"| returncode | `{trace.get('returncode')}` |",
        f"| command | `{_md_cell(' '.join(trace.get('command') or []))}` |",
        f"| classifier | `{_md_cell(trace.get('classifier') or '')}` |",
        "",
        "## Blockers",
        "",
    ])
    for blocker in report.get("blockers") or []:
        lines.append(f"- {blocker}")
    lines.extend([
        "",
        "## Scope",
        "",
        str(report.get("scope") or ""),
        "",
    ])
    return "\n".join(lines)


def _install_dependencies(dependency_target: Path, *, timeout_seconds: int) -> dict[str, Any]:
    dependency_target.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--target",
        str(dependency_target),
        *PIP_PACKAGES,
    ]
    return _run(cmd, cwd=REPO_ROOT, timeout_seconds=timeout_seconds)


def _check_dependency_imports(dependency_target: Path) -> dict[str, Any]:
    env = _python_env(dependency_target)
    rows = []
    for module in IMPORT_MODULES:
        proc = subprocess.run(
            [sys.executable, "-c", f"import {module}; print('OK')"],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
            check=False,
        )
        rows.append({
            "module": module,
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "detail": (proc.stdout or proc.stderr).strip()[-1000:],
        })
    return {
        "dependency_target": str(dependency_target),
        "modules": rows,
        "pass": all(row["ok"] for row in rows),
    }


def _copy_repo(source_repo: Path, copied_repo: Path) -> dict[str, Any]:
    if not source_repo.exists():
        return {"pass": False, "source_repo": str(source_repo), "blocker": "source Gavel repo missing"}
    shutil.rmtree(copied_repo.parent, ignore_errors=True)
    copied_repo.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source_repo,
        copied_repo,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
    )
    return {
        "pass": copied_repo.exists(),
        "source_repo": str(source_repo),
        "copied_repo": str(copied_repo),
    }


def _generate_protobuf_stubs(scheduler_dir: Path, dependency_target: Path) -> dict[str, Any]:
    rpc_stubs = scheduler_dir / "runtime" / "rpc_stubs"
    rpc_stubs.mkdir(parents=True, exist_ok=True)
    commands = []
    ok = True
    for proto in PROTO_FILES:
        cmd = [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            "-Iruntime/protobuf",
            "--python_out=runtime/rpc_stubs",
            "--grpc_python_out=runtime/rpc_stubs",
            f"runtime/protobuf/{proto}",
        ]
        result = _run(cmd, cwd=scheduler_dir, timeout_seconds=20, env=_python_env(dependency_target))
        commands.append({"proto": proto, **result})
        ok = ok and result["returncode"] == 0
    return {
        "pass": ok,
        "rpc_stubs_dir": str(rpc_stubs),
        "commands": commands,
    }


def _entrypoint_help_smoke(
    scheduler_dir: Path,
    dependency_target: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    cmd = [sys.executable, "scripts/sweeps/run_sweep_static.py", "-h"]
    result = _run(
        cmd,
        cwd=scheduler_dir,
        timeout_seconds=timeout_seconds,
        env=_gavel_env(scheduler_dir, dependency_target),
    )
    return {
        **result,
        "status": "PASS" if result["returncode"] == 0 else "FAIL",
        "pass": result["returncode"] == 0,
    }


def _native_trace_smoke(
    scheduler_dir: Path,
    dependency_target: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/drivers/simulate_scheduler_with_trace.py",
        "-t",
        "traces/physical_cluster/small_test.trace",
        "-p",
        "fifo",
        "--throughputs_file",
        "simulation_throughputs.json",
        "-c",
        "2:0:0",
        "--num_gpus_per_server",
        "2:1:1",
        "--seed",
        "0",
        "--time_per_iteration",
        "60",
        "-s",
        "0",
        "-e",
        "2",
    ]
    result = _run(
        cmd,
        cwd=scheduler_dir,
        timeout_seconds=timeout_seconds,
        env=_gavel_env(scheduler_dir, dependency_target),
    )
    classifier = _classify_trace_result(result)
    return {
        **result,
        "status": "PASS" if result["returncode"] == 0 else classifier,
        "classifier": classifier,
        "pass": result["returncode"] == 0,
        "trace_scope": (
            "Native Gavel small_test.trace runner smoke only; not a Scheduleurm same-workload comparison."
        ),
    }


def _scheduleurm_native_trace_seed_smoke(
    scheduler_dir: Path,
    dependency_target: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    trace_file = scheduler_dir / "scheduleurm_q01_native_trace_seed.trace"
    trace_file.write_text(
        "\n".join(build_native_trace_lines("q01_gpu_bound_compute")) + "\n",
        encoding="utf-8",
    )
    cmd = [
        sys.executable,
        "scripts/drivers/simulate_scheduler_with_trace.py",
        "-t",
        str(trace_file),
        "-p",
        "fifo",
        "--throughputs_file",
        "simulation_throughputs.json",
        "-c",
        "2:0:0",
        "--num_gpus_per_server",
        "2:1:1",
        "--seed",
        "0",
        "--time_per_iteration",
        "60",
        "-s",
        "0",
        "-e",
        "2",
    ]
    result = _run(
        cmd,
        cwd=scheduler_dir,
        timeout_seconds=timeout_seconds,
        env=_gavel_env(scheduler_dir, dependency_target),
    )
    classifier = _classify_trace_result(result)
    return {
        **result,
        "status": "PASS" if result["returncode"] == 0 else classifier,
        "classifier": classifier,
        "pass": result["returncode"] == 0,
        "trace_file": str(trace_file),
        "trace_scope": (
            "Scheduleurm q01-to-Gavel native trace compatibility smoke.  This validates native schema consumption, not service-unit equivalence."
        ),
    }


def _generated_jobs_smoke(
    scheduler_dir: Path,
    dependency_target: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/drivers/simulate_scheduler_with_generated_jobs.py",
        "-c",
        "1:0:0",
        "--num_gpus_per_server",
        "1:1:1",
        "-s",
        "0",
        "-e",
        "2",
        "-p",
        "fifo",
        "--seed",
        "0",
        "-i",
        "60",
        "-l",
        "0",
        "-f",
        "60",
        "--throughputs_file",
        "simulation_throughputs.json",
        "-v",
    ]
    result = _run(
        cmd,
        cwd=scheduler_dir,
        timeout_seconds=timeout_seconds,
        env=_gavel_env(scheduler_dir, dependency_target),
    )
    text = f"{result.get('stdout_tail') or ''}\n{result.get('stderr_tail') or ''}"
    completed = "Number of completed jobs: 2" in text or "Job 1:" in text
    status = "PASS" if result["returncode"] == 0 and completed else _classify_trace_result(result)
    return {
        **result,
        "status": status,
        "completed_two_jobs": completed,
        "pass": status == "PASS",
        "trace_scope": (
            "Native Gavel generated-jobs simulator smoke only; not a Scheduleurm same-workload comparison."
        ),
    }


def _classify_trace_result(result: Mapping[str, Any]) -> str:
    text = f"{result.get('stdout_tail') or ''}\n{result.get('stderr_tail') or ''}"
    if result.get("timeout"):
        if "Number of completed jobs: 0" in text or "left unused" in text:
            return "TIMEOUT_NON_PROGRESS"
        return "TIMEOUT"
    if result.get("returncode") == 0:
        return "PASS"
    if "Number of completed jobs: 0" in text or "left unused" in text:
        return "FAIL_NON_PROGRESS"
    return "FAIL"


def _run(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: int,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            env=dict(env or os.environ),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
        return {
            "command": command,
            "cwd": str(cwd),
            "timeout_seconds": timeout_seconds,
            "timeout": False,
            "returncode": proc.returncode,
            "stdout_tail": (proc.stdout or "")[-4000:],
            "stderr_tail": (proc.stderr or "")[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "cwd": str(cwd),
            "timeout_seconds": timeout_seconds,
            "timeout": True,
            "returncode": None,
            "stdout_tail": _tail_from_timeout(exc.stdout),
            "stderr_tail": _tail_from_timeout(exc.stderr),
        }
    except OSError as exc:
        return {
            "command": command,
            "cwd": str(cwd),
            "timeout_seconds": timeout_seconds,
            "timeout": False,
            "returncode": None,
            "stdout_tail": "",
            "stderr_tail": str(exc),
        }


def _tail_from_timeout(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value[-4000:]


def _python_env(dependency_target: Path) -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    paths = [str(dependency_target)]
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def _gavel_env(scheduler_dir: Path, dependency_target: Path) -> dict[str, str]:
    env = _python_env(dependency_target)
    paths = [
        str(dependency_target),
        str(scheduler_dir / "runtime" / "rpc_stubs"),
        str(scheduler_dir),
    ]
    existing = os.environ.get("PYTHONPATH", "")
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def _md_cell(value: str) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")[:1000]


def _write_json(path: str | Path, report: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_gavel_direct_native_smoke(
        reference_root=args.reference_root,
        dependency_target=args.dependency_target,
        run_root=args.run_root,
        install_deps=args.install_deps,
        run_native_trace=not args.no_native_trace,
        timeout_seconds=args.timeout_seconds,
        keep_run_dir=not args.remove_run_dir,
    )
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.gavel_direct_native_smoke")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Run isolated native-smoke audit for Gavel")
    build.add_argument("--reference-root", default=str(REFERENCE_ROOT))
    build.add_argument("--dependency-target", default=str(DEFAULT_DEP_TARGET))
    build.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    build.add_argument("--install-deps", action="store_true")
    build.add_argument("--no-native-trace", action="store_true")
    build.add_argument("--timeout-seconds", type=int, default=20)
    build.add_argument("--remove-run-dir", action="store_true")
    build.add_argument(
        "--output",
        default=str(ARTIFACT_ROOT / "gavel_direct_native_smoke_20260612.json"),
    )
    build.add_argument(
        "--markdown-output",
        default=str(REPO_ROOT / "md" / "gavel_direct_native_smoke_20260612.md"),
    )
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
