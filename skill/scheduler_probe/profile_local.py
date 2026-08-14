"""Local profiling command for resource and runtime history."""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ProfileLocalDeps:
    log_dir: Any
    parse_env: Callable[[Any], dict]
    project_from_path: Callable[[Any], str]
    inject_python_u: Callable[[str], str]
    local_backend_factory: Callable[[], Any]
    runtime_profile_from_log: Callable[..., dict | None]
    apply_runtime_projection: Callable[[dict, dict | None], None]
    history_record: Callable[..., Any]
    runtime_history_record: Callable[..., Any]
    now: Callable[[], float] = time.time
    sleep: Callable[[float], None] = time.sleep


def _profile_local_log_path(args: Any, *, deps: ProfileLocalDeps) -> str:
    log_path = os.path.expanduser(args.log_path or "")
    if log_path:
        return log_path
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.signature).strip("_") or "profile"
    return str(deps.log_dir / f"profile_{safe}_{int(deps.now())}.log")


def _profile_local_project(args: Any, cwd: str, *, deps: ProfileLocalDeps) -> str:
    return args.project or deps.project_from_path(cwd) or (
        args.signature.split("/", 1)[0] if "/" in args.signature else args.signature)


def _terminate_profile_process(proc: subprocess.Popen, *, sleep: Callable[[float], None]) -> int:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        sleep(2)
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        pass
    return proc.wait()


def _sample_profile_process(
    proc: subprocess.Popen,
    task: dict,
    *,
    timeout_s: float,
    sample_interval: float,
    started_at: float,
    deps: ProfileLocalDeps,
) -> tuple[int, int, int, int]:
    peak_vram = 0
    peak_ram = 0
    peak_cpu = 1
    backend = deps.local_backend_factory()
    interval = max(1.0, float(sample_interval))
    try:
        while True:
            res = backend.batch_probe({"tasks": [task]}).get("profile-local")
            if res and res.get("state") == "alive":
                peak_vram = max(peak_vram, int(res.get("vram_mb") or 0))
                peak_ram = max(peak_ram, int(res.get("ram_mb") or 0))
                pcpu = float(res.get("pcpu") or 0.0)
                peak_cpu = max(peak_cpu, max(1, int((pcpu + 99.0) // 100.0)))
            rc = proc.poll()
            if rc is not None:
                break
            if timeout_s and deps.now() - started_at > timeout_s:
                rc = _terminate_profile_process(proc, sleep=deps.sleep)
                break
            deps.sleep(interval)
    finally:
        rc = proc.poll()
        if rc is None:
            rc = proc.wait()
    return int(rc), peak_vram, peak_ram, peak_cpu


def cmd_profile_local(args: Any, *, deps: ProfileLocalDeps) -> None:
    """Run a local preflight command outside the scheduler queue and record profile history."""
    cwd = os.path.abspath(os.path.expanduser(args.cwd))
    if not os.path.isdir(cwd):
        sys.exit(f"cwd does not exist: {cwd}")
    project = _profile_local_project(args, cwd, deps=deps)
    log_path = _profile_local_log_path(args, deps=deps)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    extra_env = deps.parse_env(args.env)
    env = os.environ.copy()
    env.update(extra_env)
    inner = deps.inject_python_u(args.cmd)
    shell_cmd = f"cd {shlex.quote(cwd)} && {inner}"

    start = deps.now()
    with open(log_path, "ab", buffering=0) as log_f:
        header = (
            f"\n=== scheduleurm profile-local start {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
            f"signature: {args.signature}\n"
            f"cwd: {cwd}\n"
            f"cmd: {args.cmd}\n"
        )
        log_f.write(header.encode("utf-8", errors="replace"))
        proc = subprocess.Popen(
            ["bash", "-lc", shell_cmd],
            cwd=cwd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid,
        )
        task = {
            "id": "profile-local",
            "status": "running",
            "node": "local",
            "remote_pids": [proc.pid],
            "process_group": proc.pid,
            "cmd": args.cmd,
            "cwd": cwd,
            "signature": args.signature,
            "project": project,
            "description": args.description or "local preflight profile",
            "extra_env": extra_env,
            "env_spec": "none",
            "image": "",
            "started_at": start,
        }
        rc, peak_vram, peak_ram, peak_cpu = _sample_profile_process(
            proc,
            task,
            timeout_s=args.timeout,
            sample_interval=args.sample_interval,
            started_at=start,
            deps=deps,
        )
        footer = (
            f"\n=== scheduleurm profile-local end rc={rc} "
            f"duration_s={int(deps.now() - start)} peak_vram_mb={peak_vram} "
            f"peak_ram_mb={peak_ram} peak_cpu={peak_cpu} ===\n"
        )
        log_f.write(footer.encode("utf-8", errors="replace"))

    duration_s = int(max(0, deps.now() - start))
    profile_task = {
        "id": "profile-local",
        "signature": args.signature,
        "project": project,
        "description": args.description or "local preflight profile",
        "cmd": args.cmd,
        "cwd": cwd,
        "env_spec": "none",
        "image": "",
        "extra_env": extra_env,
    }
    profile = deps.runtime_profile_from_log(log_path, args.cmd, observed_duration_s=duration_s)
    if profile:
        profile = dict(profile)
        profile["source"] = (profile.get("source") or "profile-local") + ":" + log_path
        deps.apply_runtime_projection(profile_task, profile)
    deps.history_record(
        args.signature,
        peak_vram_mb=peak_vram,
        peak_ram_mb=peak_ram,
        cpu_cores=peak_cpu,
        duration_s=duration_s,
        task=profile_task,
    )
    deps.runtime_history_record(profile_task, duration_s=duration_s)

    out = {
        "ok": rc == 0,
        "exit_code": rc,
        "log_path": log_path,
        "duration_s": duration_s,
        "peak_vram_mb": peak_vram,
        "peak_ram_mb": peak_ram,
        "peak_cpu_cores": peak_cpu,
        "runtime_profile": {
            "source": profile_task.get("runtime_est_source") or ("duration" if duration_s else ""),
            "total_s": profile_task.get("runtime_total_s_est") or duration_s,
            "eta_s": profile_task.get("runtime_eta_s_est"),
            "unit_s": profile_task.get("runtime_unit_s_est"),
            "total_units": profile_task.get("runtime_total_units"),
        },
    }
    if args.json:
        print(json.dumps(out, indent=2))
        return
    print(f"profile-local: rc={rc} log={log_path}")
    print(f"  duration={duration_s}s peak_vram={peak_vram}MB peak_ram={peak_ram}MB peak_cpu={peak_cpu}")
    runtime_profile = out["runtime_profile"]
    if runtime_profile.get("total_s"):
        print(f"  runtime_total={runtime_profile.get('total_s')}s source={runtime_profile.get('source')}")
