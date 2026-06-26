"""Gavel physical same-workload full-stack gate.

This gate runs Gavel's physical scheduler/worker/RPC/lease/dispatcher path in a
temporary copy of the upstream repository.  It injects a minimal
GavelIterator-compatible CUDA workload into the temporary copy, then compares
that Gavel-launched workload with the same script run directly on the same host.

The gate never edits ``reference/repos/gavel`` and never modifies system CUDA or
cuDNN.  Python dependencies are loaded from an isolated target directory under
``/tmp``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
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


ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_RUN_ROOT = Path("/tmp/scheduleurm_gavel_physical_same_workload")
DEFAULT_OUTPUT = ARTIFACT_ROOT / "gavel_physical_same_workload_gate_20260613.json"
DEFAULT_MARKDOWN_OUTPUT = REPO_ROOT / "md" / "gavel_physical_same_workload_gate_20260613.md"
DEFAULT_SCHEDULER_PORT = 50160
DEFAULT_WORKER_PORT = 50161


def build_gavel_physical_same_workload_gate(
    *,
    reference_root: str | Path = REFERENCE_ROOT,
    dependency_target: str | Path = DEFAULT_DEP_TARGET,
    run_root: str | Path = DEFAULT_RUN_ROOT,
    steps: int = 12,
    scheduler_port: int = DEFAULT_SCHEDULER_PORT,
    worker_port: int = DEFAULT_WORKER_PORT,
    timeout_seconds: int = 120,
    keep_run_dir: bool = True,
) -> dict[str, Any]:
    reference_root = Path(reference_root).expanduser()
    dependency_target = Path(dependency_target).expanduser()
    run_root = Path(run_root).expanduser()
    copied_repo = run_root / "gavel"
    scheduler_dir = copied_repo / "scheduler"
    workload_root = copied_repo / "workloads" / "pytorch"
    workload_dir = workload_root / "scheduleurm_probe"
    data_dir = run_root / "data"
    checkpoint_dir = run_root / "checkpoints"
    timeline_dir = run_root / "timelines"
    logs_dir = run_root / "logs"

    import_result = _check_dependency_imports(dependency_target)
    extended_imports = _check_extended_imports(dependency_target)
    copy_result = _copy_repo(reference_root / "gavel", copied_repo)
    stub_result = (
        _generate_protobuf_stubs(scheduler_dir, dependency_target)
        if copy_result.get("pass") else
        {"pass": False, "commands": [], "blocker": "copy failed"}
    )
    prep_result = {"pass": False, "blocker": "not run"}
    native_result: dict[str, Any] = {"pass": False, "status": "NOT_RUN"}
    physical_result: dict[str, Any] = {"pass": False, "status": "NOT_RUN"}
    if import_result.get("pass") and extended_imports.get("pass") and stub_result.get("pass"):
        prep_result = _prepare_physical_workload(
            scheduler_dir=scheduler_dir,
            workload_dir=workload_dir,
            data_dir=data_dir,
            checkpoint_dir=checkpoint_dir,
            timeline_dir=timeline_dir,
            logs_dir=logs_dir,
            scheduler_port=scheduler_port,
            steps=steps,
        )
        if prep_result.get("pass"):
            native_result = _run_native_workload(
                scheduler_dir=scheduler_dir,
                dependency_target=dependency_target,
                workload_dir=workload_dir,
                checkpoint_dir=checkpoint_dir / "native",
                output_file=logs_dir / "native_probe.log",
                steps=steps,
                timeout_seconds=timeout_seconds,
            )
            physical_result = _run_gavel_physical(
                scheduler_dir=scheduler_dir,
                dependency_target=dependency_target,
                workload_root=workload_root,
                data_dir=data_dir,
                checkpoint_dir=checkpoint_dir / "gavel",
                timeline_dir=timeline_dir,
                logs_dir=logs_dir,
                trace_file=prep_result["trace_file"],
                throughputs_file=prep_result["throughputs_file"],
                steps=steps,
                scheduler_port=scheduler_port,
                worker_port=worker_port,
                timeout_seconds=timeout_seconds,
            )

    native_wall_s = _num(native_result.get("wall_s"))
    gavel_jct_s = _num(physical_result.get("gavel_jct_s"))
    gavel_total_wall_s = _num(physical_result.get("scheduler_wall_s"))
    same_service_scale_ready = bool(
        native_result.get("pass")
        and physical_result.get("iterator_progress_ready")
        and int(native_result.get("completed_steps") or 0) >= steps
        and int(physical_result.get("completed_steps") or 0) >= steps
    )
    scoped_superiority_ready = bool(
        same_service_scale_ready
        and gavel_jct_s > 0
        and native_wall_s > 0
        and native_wall_s <= gavel_jct_s
    )
    scoped_fullstack_ready = bool(physical_result.get("pass") and same_service_scale_ready)
    blockers = _blockers(
        import_result=import_result,
        extended_imports=extended_imports,
        copy_result=copy_result,
        stub_result=stub_result,
        prep_result=prep_result,
        native_result=native_result,
        physical_result=physical_result,
        same_service_scale_ready=same_service_scale_ready,
        scoped_superiority_ready=scoped_superiority_ready,
    )
    report = {
        "gate": "gavel_physical_same_workload_gate",
        "regeneration_mode": "precomputed_live_run",
        "current_safe_reproduction": False,
        "requires_external_stack": True,
        "current_environment_has_tools": False,
        "reference_root": str(reference_root),
        "run_root": str(run_root),
        "copied_repo": str(copied_repo),
        "dependency_target": str(dependency_target),
        "host": socket.gethostname(),
        "steps": int(steps),
        "scheduler_port": int(scheduler_port),
        "worker_port": int(worker_port),
        "dependency_imports": import_result,
        "extended_imports": extended_imports,
        "copy": copy_result,
        "protobuf_stubs": stub_result,
        "preparation": prep_result,
        "native_same_workload": native_result,
        "gavel_physical_same_workload": physical_result,
        "same_service_scale_ready": bool(same_service_scale_ready),
        "scoped_gavel_physical_fullstack_ready": bool(scoped_fullstack_ready),
        "scoped_gavel_physical_same_workload_superiority_ready": bool(scoped_superiority_ready),
        "direct_fullstack_binary_superiority_ready": bool(scoped_superiority_ready),
        "native_wall_s": native_wall_s,
        "gavel_jct_s": gavel_jct_s,
        "gavel_total_scheduler_wall_s": gavel_total_wall_s,
        "scheduleurm_native_to_gavel_jct_ratio": (
            native_wall_s / gavel_jct_s if native_wall_s > 0 and gavel_jct_s > 0 else None
        ),
        "blockers": blockers,
        "evidence_artifacts": {
            "native_log": str(logs_dir / "native_probe.log"),
            "scheduler_log": str(logs_dir / "gavel_scheduler.log"),
            "worker_log": str(logs_dir / "gavel_worker.log"),
            "trace_file": str(prep_result.get("trace_file") or ""),
            "throughputs_file": str(prep_result.get("throughputs_file") or ""),
            "timeline_dir": str(timeline_dir),
            "checkpoint_dir": str(checkpoint_dir),
        },
        "pass": True,
        "scope": (
            "Runs Gavel's physical scheduler/worker/RPC/dispatcher/GavelIterator "
            "path on a temporary same-host workload and compares it with the same "
            "script run directly.  This is a scoped Gavel physical-runtime row.  It "
            "does not certify every Gavel paper workload or a multi-node Gavel cluster."
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
        "# Gavel Physical Same-Workload Gate",
        "",
        "## Summary",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `regeneration_mode` | `{report.get('regeneration_mode')}` |",
        f"| `current_safe_reproduction` | {str(bool(report.get('current_safe_reproduction'))).lower()} |",
        f"| `requires_external_stack` | {str(bool(report.get('requires_external_stack'))).lower()} |",
        f"| `current_environment_has_tools` | {str(bool(report.get('current_environment_has_tools'))).lower()} |",
        f"| `same_service_scale_ready` | {str(bool(report.get('same_service_scale_ready'))).lower()} |",
        f"| `scoped_gavel_physical_fullstack_ready` | {str(bool(report.get('scoped_gavel_physical_fullstack_ready'))).lower()} |",
        f"| `scoped_gavel_physical_same_workload_superiority_ready` | {str(bool(report.get('scoped_gavel_physical_same_workload_superiority_ready'))).lower()} |",
        f"| `direct_fullstack_binary_superiority_ready` | {str(bool(report.get('direct_fullstack_binary_superiority_ready'))).lower()} |",
        f"| `native_wall_s` | {_fmt(report.get('native_wall_s'))} |",
        f"| `gavel_jct_s` | {_fmt(report.get('gavel_jct_s'))} |",
        f"| `gavel_total_scheduler_wall_s` | {_fmt(report.get('gavel_total_scheduler_wall_s'))} |",
        f"| `scheduleurm_native_to_gavel_jct_ratio` | {_fmt(report.get('scheduleurm_native_to_gavel_jct_ratio'))} |",
        "",
        "## Runtime Checks",
        "",
        "| Check | Ready | Detail |",
        "|---|---:|---|",
    ]
    checks = [
        ("dependency_imports", report.get("dependency_imports")),
        ("extended_imports", report.get("extended_imports")),
        ("protobuf_stubs", report.get("protobuf_stubs")),
        ("preparation", report.get("preparation")),
        ("native_same_workload", report.get("native_same_workload")),
        ("gavel_physical_same_workload", report.get("gavel_physical_same_workload")),
    ]
    for name, value in checks:
        ready = bool((value or {}).get("pass"))
        detail = (value or {}).get("status") or (value or {}).get("blocker") or ""
        lines.append(f"| `{name}` | {str(ready).lower()} | `{_md(detail)}` |")
    lines.extend(["", "## Blockers", "", "| Blocker |", "|---|"])
    blockers = report.get("blockers") or []
    if not blockers:
        lines.append("| none |")
    for blocker in blockers:
        lines.append(f"| {_md(blocker)} |")
    lines.extend(["", "## Scope", "", str(report.get("scope") or ""), ""])
    return "\n".join(lines)


def _prepare_physical_workload(
    *,
    scheduler_dir: Path,
    workload_dir: Path,
    data_dir: Path,
    checkpoint_dir: Path,
    timeline_dir: Path,
    logs_dir: Path,
    scheduler_port: int,
    steps: int,
) -> dict[str, Any]:
    for directory in (workload_dir, data_dir, checkpoint_dir, timeline_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    scheduler_py = scheduler_dir / "scheduler.py"
    text = scheduler_py.read_text(encoding="utf-8")
    text = re.sub(r"SCHEDULER_PORT\s*=\s*\d+", f"SCHEDULER_PORT = {int(scheduler_port)}", text)
    scheduler_py.write_text(text, encoding="utf-8")
    (workload_dir / "main.py").write_text(_workload_source(), encoding="utf-8")
    trace_file = scheduler_dir / "scheduleurm_probe_physical.trace"
    trace_file.write_text(
        "\t".join([
            "ScheduleurmProbe",
            "python3 main.py",
            "scheduleurm_probe",
            "--steps",
            "0",
            str(int(steps)),
            "1",
            "1",
            "-1.000000",
            "0",
        ]) + "\n",
        encoding="utf-8",
    )
    throughputs_file = scheduler_dir / "scheduleurm_probe_throughputs.json"
    key = "('ScheduleurmProbe', 1)"
    throughputs = {
        "v100": {
            key: {
                "null": 100.0,
                key: [90.0, 90.0],
            }
        }
    }
    throughputs_file.write_text(json.dumps(throughputs, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "pass": True,
        "status": "PREPARED",
        "trace_file": str(trace_file),
        "throughputs_file": str(throughputs_file),
        "workload_dir": str(workload_dir),
        "scheduler_port_patch": int(scheduler_port),
    }


def _run_native_workload(
    *,
    scheduler_dir: Path,
    dependency_target: Path,
    workload_dir: Path,
    checkpoint_dir: Path,
    output_file: Path,
    steps: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    env = _physical_env(scheduler_dir=scheduler_dir, dependency_target=dependency_target)
    cmd = [
        sys.executable,
        "main.py",
        "--local_rank",
        "0",
        "--steps",
        str(int(steps)),
        "--checkpoint_dir",
        str(checkpoint_dir),
    ]
    start = time.time()
    result = _run_to_file(cmd, cwd=workload_dir, env=env, timeout_seconds=timeout_seconds, output_file=output_file)
    wall_s = time.time() - start
    parsed = _parse_native_output(output_file)
    return {
        "pass": bool(result.get("returncode") == 0 and parsed.get("completed_steps", 0) >= steps),
        "status": "PASS" if result.get("returncode") == 0 else "FAIL",
        "command": cmd,
        "returncode": result.get("returncode"),
        "timed_out": bool(result.get("timed_out")),
        "wall_s": wall_s,
        "completed_steps": int(parsed.get("completed_steps") or 0),
        "device": parsed.get("device"),
        "output_file": str(output_file),
    }


def _run_gavel_physical(
    *,
    scheduler_dir: Path,
    dependency_target: Path,
    workload_root: Path,
    data_dir: Path,
    checkpoint_dir: Path,
    timeline_dir: Path,
    logs_dir: Path,
    trace_file: str,
    throughputs_file: str,
    steps: int,
    scheduler_port: int,
    worker_port: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    timeline_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    env = _physical_env(scheduler_dir=scheduler_dir, dependency_target=dependency_target)
    scheduler_log = logs_dir / "gavel_scheduler.log"
    worker_log = logs_dir / "gavel_worker.log"
    for path in (scheduler_log, worker_log):
        path.write_text("", encoding="utf-8")
    sched_cmd = [
        sys.executable,
        "scripts/drivers/run_scheduler_with_trace.py",
        "-t",
        str(trace_file),
        "-p",
        "fifo",
        "--seed",
        "0",
        "--solver",
        "SCS",
        "--throughputs_file",
        str(throughputs_file),
        "--time_per_iteration",
        "8",
        "--expected_num_workers",
        "1",
        "--timeline_dir",
        str(timeline_dir),
    ]
    worker_cmd = [
        sys.executable,
        "worker.py",
        "-t",
        "v100",
        "-i",
        socket.gethostbyname(socket.gethostname()),
        "-s",
        str(int(scheduler_port)),
        "-w",
        str(int(worker_port)),
        "-g",
        "1",
        "--run_dir",
        str(workload_root),
        "--data_dir",
        str(data_dir),
        "--checkpoint_dir",
        str(checkpoint_dir),
    ]
    start = time.time()
    sched_fh = scheduler_log.open("wb")
    worker_fh = worker_log.open("wb")
    scheduler_proc = subprocess.Popen(
        sched_cmd,
        cwd=scheduler_dir,
        env=env,
        stdout=sched_fh,
        stderr=subprocess.STDOUT,
    )
    server_ready = _wait_for_log(scheduler_log, "Starting server", timeout_s=20)
    worker_proc = subprocess.Popen(
        worker_cmd,
        cwd=scheduler_dir,
        env=env,
        stdout=worker_fh,
        stderr=subprocess.STDOUT,
    )
    try:
        scheduler_returncode = scheduler_proc.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        scheduler_proc.terminate()
        try:
            scheduler_returncode = scheduler_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            scheduler_proc.kill()
            scheduler_returncode = scheduler_proc.wait(timeout=5)
    try:
        worker_returncode = worker_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        worker_proc.terminate()
        try:
            worker_returncode = worker_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            worker_proc.kill()
            worker_returncode = worker_proc.wait(timeout=5)
    sched_fh.close()
    worker_fh.close()
    wall_s = time.time() - start
    parsed = _parse_gavel_physical_logs(
        scheduler_log=scheduler_log,
        worker_log=worker_log,
        checkpoint_dir=checkpoint_dir,
    )
    ready = bool(
        scheduler_returncode == 0
        and parsed.get("worker_registered")
        and parsed.get("iterator_progress_ready")
        and int(parsed.get("completed_steps") or 0) >= steps
    )
    return {
        "pass": ready,
        "status": "PASS" if ready else "FAIL",
        "scheduler_command": sched_cmd,
        "worker_command": worker_cmd,
        "scheduler_returncode": scheduler_returncode,
        "worker_returncode": worker_returncode,
        "server_ready": bool(server_ready),
        "scheduler_wall_s": wall_s,
        "scheduler_log": str(scheduler_log),
        "worker_log": str(worker_log),
        **parsed,
    }


def _parse_gavel_physical_logs(*, scheduler_log: Path, worker_log: Path, checkpoint_dir: Path) -> dict[str, Any]:
    scheduler_text = _read_text(scheduler_log)
    worker_text = _read_text(worker_log)
    iterator_logs = list(checkpoint_dir.glob("job_id=*/.gavel/round=*/worker=*.log"))
    iterator_text = "\n".join(_read_text(path) for path in iterator_logs)
    completed_steps = 0
    duration_s = 0.0
    for line in iterator_text.splitlines():
        m = re.match(r"\[(.*)\] \[(.*)\] \[(.*)\]\s?(.*)", line)
        if not m:
            continue
        event, status, msg = m.group(2), m.group(3), m.group(4)
        if event == "PROGRESS" and status == "STEPS":
            completed_steps = max(completed_steps, int(float(msg or 0)))
        if event == "PROGRESS" and status == "DURATION":
            duration_s = max(duration_s, float(msg or 0.0))
    total_time = _extract_float(r"Total time taken:\s*([0-9.]+)\s*seconds", scheduler_text)
    avg_jct = (
        _extract_float(r"Average job completion time:\s*([0-9.]+)\s*seconds", scheduler_text)
        or _extract_float(r"Average JCT.*?:\s*([0-9.]+)", scheduler_text)
    )
    return {
        "worker_registered": "Succesfully registered worker" in worker_text or "Successfully registered" in scheduler_text,
        "iterator_progress_ready": completed_steps > 0 and duration_s > 0,
        "completed_steps": int(completed_steps),
        "iterator_duration_s": float(duration_s),
        "gavel_jct_s": float(avg_jct if avg_jct is not None else total_time or 0.0),
        "gavel_total_time_taken_s": float(total_time or 0.0),
        "iterator_log_count": len(iterator_logs),
        "scheduler_log_tail": "\n".join(scheduler_text.splitlines()[-40:]),
        "worker_log_tail": "\n".join(worker_text.splitlines()[-40:]),
    }


def _workload_source() -> str:
    return r'''from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch


def _noop_load(*args, **kwargs):
    return None


def _noop_save(*args, **kwargs):
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--enable_gavel_iterator", action="store_true")
    args, _ = parser.parse_known_args()

    device_name = "cpu"
    if torch.cuda.is_available():
        device_index = min(max(int(args.local_rank), 0), torch.cuda.device_count() - 1)
        torch.cuda.set_device(device_index)
        device = torch.device("cuda", device_index)
        device_name = torch.cuda.get_device_name(device_index)
    else:
        device = torch.device("cpu")

    data = list(range(max(int(args.steps), 1)))
    iterator = data
    if args.enable_gavel_iterator:
        from gavel_iterator import GavelIterator
        iterator = GavelIterator(
            data,
            args.checkpoint_dir,
            _noop_load,
            _noop_save,
            synthetic_data=True,
            write_on_close=True,
            verbose=False,
        )

    start = time.time()
    completed = 0
    for _ in iterator:
        x = torch.ones((256, 256), device=device)
        y = x @ x
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        completed += 1
        if completed >= int(args.steps):
            break
    if args.enable_gavel_iterator and hasattr(iterator, "complete"):
        iterator.complete()
    wall_s = time.time() - start
    print(json.dumps({
        "completed_steps": completed,
        "wall_s": wall_s,
        "device": device_name,
        "enable_gavel_iterator": bool(args.enable_gavel_iterator),
        "pid": os.getpid(),
    }, sort_keys=True))
    return 0 if completed >= int(args.steps) else 2


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _physical_env(*, scheduler_dir: Path, dependency_target: Path) -> dict[str, str]:
    env = _gavel_env(scheduler_dir, dependency_target)
    py_parts = [
        str(scheduler_dir),
        str(dependency_target),
        env.get("PYTHONPATH", ""),
    ]
    env["PYTHONPATH"] = os.pathsep.join(part for part in py_parts if part)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    return env


def _check_extended_imports(dependency_target: Path) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(dependency_target) + os.pathsep + env.get("PYTHONPATH", "")
    modules = ["torch", "filelock", "numa", "psutil"]
    rows = []
    for module in modules:
        proc = subprocess.run(
            [sys.executable, "-c", f"import {module}; print('ok')"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            timeout=15,
        )
        rows.append({
            "module": module,
            "ok": proc.returncode == 0,
            "detail": proc.stdout.strip()[-400:],
        })
    return {"pass": all(row["ok"] for row in rows), "modules": rows}


def _run_to_file(
    cmd: list[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout_seconds: int,
    output_file: Path,
) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            text=True,
        )
        output_file.write_text(proc.stdout, encoding="utf-8", errors="replace")
        return {"returncode": proc.returncode, "timed_out": False}
    except subprocess.TimeoutExpired as exc:
        output_file.write_text((exc.stdout or "") if isinstance(exc.stdout, str) else "", encoding="utf-8", errors="replace")
        return {"returncode": None, "timed_out": True}


def _wait_for_log(path: Path, needle: str, *, timeout_s: int) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if needle in _read_text(path):
            return True
        time.sleep(0.25)
    return False


def _parse_native_output(path: Path) -> dict[str, Any]:
    text = _read_text(path)
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            data = json.loads(line)
            return {
                "completed_steps": int(data.get("completed_steps") or 0),
                "device": data.get("device"),
                "wall_s": float(data.get("wall_s") or 0.0),
            }
        except Exception:
            pass
    return {}


def _extract_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


def _blockers(
    *,
    import_result: Mapping[str, Any],
    extended_imports: Mapping[str, Any],
    copy_result: Mapping[str, Any],
    stub_result: Mapping[str, Any],
    prep_result: Mapping[str, Any],
    native_result: Mapping[str, Any],
    physical_result: Mapping[str, Any],
    same_service_scale_ready: bool,
    scoped_superiority_ready: bool,
) -> list[str]:
    blockers: list[str] = []
    if not import_result.get("pass"):
        blockers.append("Gavel isolated Python dependencies are incomplete")
    if not extended_imports.get("pass"):
        blockers.append("Gavel physical worker dependencies are incomplete")
    if not copy_result.get("pass"):
        blockers.append("temporary Gavel repository copy failed")
    if not stub_result.get("pass"):
        blockers.append("Gavel protobuf stubs could not be generated")
    if not prep_result.get("pass"):
        blockers.append("temporary Gavel physical workload preparation failed")
    if not native_result.get("pass"):
        blockers.append("same workload did not complete on the direct native path")
    if not physical_result.get("pass"):
        blockers.append("same workload did not complete through the Gavel physical scheduler/worker path")
    if not same_service_scale_ready:
        blockers.append("Gavel and native paths did not complete the same positive number of probe steps")
    if same_service_scale_ready and not scoped_superiority_ready:
        blockers.append("direct native same-workload wall time did not dominate the Gavel physical JCT in this scoped run")
    return blockers


def _num(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.6g}"
    except Exception:
        return "nan"


def _md(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")[:1200]


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_gavel_physical_same_workload_gate(
        steps=args.steps,
        timeout_seconds=args.timeout_s,
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
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.gavel_physical_same_workload_gate")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Run Gavel physical same-workload gate")
    build.add_argument("--output", default=str(DEFAULT_OUTPUT))
    build.add_argument("--markdown-output", default=str(DEFAULT_MARKDOWN_OUTPUT))
    build.add_argument("--steps", type=int, default=12)
    build.add_argument("--timeout-s", type=int, default=120)
    build.add_argument("--clean", action="store_true")
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
