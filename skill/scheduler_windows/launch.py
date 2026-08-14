from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class WindowsLaunchDeps:
    node_configs: dict
    log_dir: Any
    wrapper_resource_log_interval_s: float
    node_cpu_fallback_block_reason: Callable[[dict, str, dict], Optional[str]]
    windows_env_spec_error: Callable[[dict], Optional[str]]
    ensure_launcher: Callable[[str], tuple[bool, str]]
    ensure_detacher: Callable[[str], tuple[bool, str]]
    sched_dir: Callable[[str], str]
    log_path: Callable[[dict], str]
    windows_path_for_node: Callable[[str, str], str]
    ps_quote: Callable[[str], str]
    run_windows_ps: Callable[..., tuple[int, str, str]]
    apply_cpu_parallel_plan_to_task: Callable[[dict, Optional[dict]], Any]
    cpu_parallel_env: Callable[[dict], dict]
    safe_extra_env_items: Callable[[dict], list[tuple[str, str]]]
    launch_extra_env: Callable[[dict], dict]
    windows_prepare_command: Callable[[dict], dict]
    remember_last_placement: Callable[[dict], Any]
    set_current_usage: Callable[[dict, int, int, float], Any]
    now: Callable[[], float]
    time_ns: Callable[[], int]
    getpid: Callable[[], int]


def windows_backend_launch(
    task: dict,
    *,
    node_state: Optional[dict],
    deps: WindowsLaunchDeps,
) -> tuple[bool, str]:
    node = task["node"]
    node_info = deps.node_configs.get(node, {}) or {}
    if int(task.get("est_vram_mb") or 0) > 0:
        block = deps.node_cpu_fallback_block_reason(task, node, node_info)
        if block:
            return False, f"WindowsBackend is CPU-only; refusing GPU task: {block}"
        if not task.get("cpu_fallback_selected"):
            return False, "WindowsBackend is CPU-only; refusing GPU task"

    env_err = deps.windows_env_spec_error(task)
    if env_err:
        return False, env_err

    ok, launcher = deps.ensure_launcher(node)
    if not ok:
        return False, f"windows launcher deploy failed: {launcher}"
    ok, detacher = deps.ensure_detacher(node)
    if not ok:
        return False, f"windows detacher deploy failed: {detacher}"

    cwd = deps.windows_path_for_node(node, task.get("cwd") or "")
    log_path = deps.log_path(task)
    py = node_info.get("windows_python") or "python"
    ps_cwd_check = f"if (Test-Path -LiteralPath {deps.ps_quote(cwd)}) {{ 'OK' }} else {{ 'MISSING' }}"
    try:
        rc_cwd, out_cwd, err_cwd = deps.run_windows_ps(node, ps_cwd_check, timeout=10, check=False)
    except Exception as exc:
        return False, f"cwd check failed on {node}: {str(exc)[:160]}"
    if rc_cwd != 0 or "OK" not in out_cwd:
        return False, f"cwd missing on {node}: {cwd}"

    cpu_plan = deps.apply_cpu_parallel_plan_to_task(task, node_state)
    env = {"CUDA_VISIBLE_DEVICES": "", "SCHEDULEURM_TASK_ID": task.get("id") or ""}
    env.update({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "SCHEDULEURM_WINDOWS_AUTO_PIN": "1",
        "SCHEDULEURM_PIN_QUIET": "1",
    })
    cpu_env = deps.cpu_parallel_env(task) if cpu_plan else {}
    if cpu_env:
        env.update(cpu_env)
    for key, value in deps.safe_extra_env_items(deps.launch_extra_env(task)):
        if key == "CUDA_VISIBLE_DEVICES":
            continue
        env[key] = value

    cmd_payload = deps.windows_prepare_command(task)
    env.update(cmd_payload.pop("env", {}) or {})
    payload = {
        "task_id": task.get("id"),
        "cwd": cwd,
        "log_path": log_path,
        "env": env,
        "auto_pin": bool(node_info.get("windows_auto_pin", True)),
        "skip_ht_pair": bool(node_info.get("windows_skip_ht_pair", True)),
        "pin_base": int(task.get("windows_pin_base") or 0),
        "pin_cores": int(task.get("windows_pin_cores") or task.get("cpu_cores") or 1),
        "pin_interval_s": 1.0,
        "resource_log_interval_s": deps.wrapper_resource_log_interval_s,
        "cpu_plan": cpu_env,
    }
    payload.update(cmd_payload)

    sched_dir = deps.sched_dir(node)
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task.get("id") or "task"))
    launch_nonce = f"{deps.time_ns()}_{deps.getpid()}"
    # A fixed per-task payload path can remain open after a timed-out upload.
    # Give every launch attempt its own file so retries cannot collide with a
    # stale PowerShell process or zero-byte artifact from an earlier attempt.
    payload_path = (
        sched_dir.rstrip("\\/")
        + rf"\payloads\{safe_id}_{launch_nonce}.b64"
    )
    pid_path = sched_dir.rstrip("\\/") + rf"\pids\{safe_id}_{launch_nonce}.pid"
    boot_out_path = log_path + f".{launch_nonce}.boot.out"
    boot_err_path = log_path + f".{launch_nonce}.boot.err"
    payload["wrapper_pid_path"] = pid_path
    payload_b64 = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")

    write_payload_ps = rf'''
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$payloadPath={deps.ps_quote(payload_path)}
[IO.Directory]::CreateDirectory((Split-Path -Parent $payloadPath)) | Out-Null
$fs = [IO.File]::Open($payloadPath, [IO.FileMode]::Create, [IO.FileAccess]::Write, [IO.FileShare]::None)
try {{
  [Console]::OpenStandardInput().CopyTo($fs)
}} finally {{
  $fs.Close()
}}
Write-Output 'WROTE'
'''
    try:
        rc_payload, out_payload, err_payload = deps.run_windows_ps(
            node,
            write_payload_ps,
            timeout=20,
            check=False,
            input_data=payload_b64.encode("ascii"),
        )
    except Exception as exc:
        return False, f"windows payload upload failed: {str(exc)[:180]}"
    if rc_payload != 0 or "WROTE" not in out_payload:
        msg = (err_payload or out_payload or f"rc={rc_payload}").strip()
        return False, f"windows payload upload failed: {msg[:200]}"

    ps = rf'''
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$py={deps.ps_quote(py)}
$launcher={deps.ps_quote(launcher)}
$detacher={deps.ps_quote(detacher)}
$payloadPath={deps.ps_quote(payload_path)}
$logPath={deps.ps_quote(log_path)}
$pidPath={deps.ps_quote(pid_path)}
$bootOut={deps.ps_quote(boot_out_path)}
$bootErr={deps.ps_quote(boot_err_path)}
[IO.Directory]::CreateDirectory((Split-Path -Parent $logPath)) | Out-Null
[IO.Directory]::CreateDirectory((Split-Path -Parent $pidPath)) | Out-Null
Remove-Item -LiteralPath $bootOut,$bootErr -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
$detOut = & $py $detacher $py $launcher $payloadPath $bootOut $bootErr $pidPath
if ($LASTEXITCODE -ne 0) {{
  throw ('windows detacher failed: ' + (($detOut | Out-String).Trim()))
}}
Start-Sleep -Milliseconds 1800
$pidText = ''
if (Test-Path -LiteralPath $pidPath) {{
  $pidText = ([IO.File]::ReadAllText($pidPath)).Trim()
}}
if (-not $pidText) {{
  $pidText = (($detOut | Select-String -Pattern 'PID=(\d+)' | Select-Object -First 1).Matches.Groups[1].Value)
}}
$alive = $false
if ($pidText -match '^\d+$') {{
  $alive = [bool](Get-Process -Id ([int]$pidText) -ErrorAction SilentlyContinue)
}}
$hasLog = Test-Path -LiteralPath $logPath
if (-not $alive -and -not $hasLog) {{
  $boot = ''
  if (Test-Path -LiteralPath $bootErr) {{
    $boot = ((Get-Content -LiteralPath $bootErr -Tail 20 -ErrorAction SilentlyContinue) -join "`n")
  }}
  throw ('windows wrapper exited before writing task log; pid=' + $pidText + '; boot_err=' + $boot)
}}
Write-Output ('PID=' + $pidText)
exit 0
'''
    try:
        deps.log_dir.mkdir(parents=True, exist_ok=True)
        local_ssh_log = deps.log_dir / f"{task.get('id')}.winssh.log"
        rc_launch, out_launch, err_launch = deps.run_windows_ps(node, ps, timeout=20, check=False)
        with open(local_ssh_log, "ab", buffering=0) as log_file:
            if out_launch:
                log_file.write(out_launch.encode("utf-8", "replace"))
            if err_launch:
                log_file.write(err_launch.encode("utf-8", "replace"))
        if rc_launch != 0:
            msg = ((err_launch or "") + (out_launch or ""))
            return False, f"windows detached launch ssh rc={rc_launch}: {msg.strip()[:200]}"

        pid = None
        match = re.search(r"PID=(\d+)", out_launch or "")
        if match:
            pid = int(match.group(1))
        poll_ps = rf'''
$p={deps.ps_quote(pid_path)}
if (Test-Path -LiteralPath $p) {{ [IO.File]::ReadAllText($p).Trim() }}
'''
        if not pid:
            try:
                rc_pid, out_pid, _ = deps.run_windows_ps(node, poll_ps, timeout=5, check=False)
                text = (out_pid or "").strip()
                if rc_pid == 0 and text.isdigit():
                    pid = int(text)
            except Exception:
                pass
        if not pid:
            return False, f"windows detached launch did not produce wrapper pid; local_ssh_log={local_ssh_log}"

        task["remote_pids"] = [pid]
        task["process_group"] = pid
        task["log_path"] = log_path
        task["bootstrap_stdout_path"] = boot_out_path
        task["bootstrap_stderr_path"] = boot_err_path
        task.pop("local_ssh_pid", None)
        task["local_ssh_log_path"] = str(local_ssh_log)
        task["wrapper_pid_path"] = pid_path
        task["status"] = "running"
        task["started_at"] = deps.now()
        deps.remember_last_placement(task)
        task["peak_vram_mb"] = 0
        task["peak_ram_mb"] = 0
        deps.set_current_usage(task, 0, 0, 0.0)
        return True, f"win_pid={pid}"
    except Exception as exc:
        return False, f"windows launch exception: {str(exc)[:200]}"
