"""Windows backend probe helpers.

These helpers keep the Windows process/log probing protocol out of the backend
class body.  They are intentionally pure except for the PowerShell script
strings they return; scheduler.py still owns transport, timeouts, and node
configuration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

try:
    from .task_probe import (
        collect_windows_probe_targets,
        normalize_windows_json_rows,
        windows_log_exit_probe_script,
        windows_log_exit_probe_specs,
        windows_log_exit_results_from_rows,
        windows_process_probe_script,
        windows_process_probe_specs,
        windows_process_results_from_rows,
        windows_queue_accounting_fallback_result,
        windows_terminal_log_result,
    )
except ModuleNotFoundError:
    from .task_probe import (
        collect_windows_probe_targets,
        normalize_windows_json_rows,
        windows_log_exit_probe_script,
        windows_log_exit_probe_specs,
        windows_log_exit_results_from_rows,
        windows_process_probe_script,
        windows_process_probe_specs,
        windows_process_results_from_rows,
        windows_queue_accounting_fallback_result,
        windows_terminal_log_result,
    )


@dataclass(frozen=True)
class WindowsNodeProbeDeps:
    node_configs: dict
    queue_tasks: Callable[[], list[dict]]
    run_windows_ps: Callable[..., tuple[int, str, str]]
    run_subprocess: Callable[..., Any]
    ssh_base_args: Callable[[str], list[str]]
    windows_probe_error_hint: Callable[[str, str], str]
    env_cpu_load_source: str | None = None


WINDOWS_CHEAP_CAPACITY_PS = r'''
$logical = [int][System.Environment]::ProcessorCount
try {
  $envLogical = [int]$env:NUMBER_OF_PROCESSORS
  if ($envLogical -gt $logical) { $logical = $envLogical }
} catch {}
try {
  $regCount = (Get-ChildItem 'HKLM:\HARDWARE\DESCRIPTION\System\CentralProcessor' -ErrorAction Stop | Measure-Object).Count
  if ($regCount -gt $logical) { $logical = [int]$regCount }
} catch {}
try {
  Add-Type -AssemblyName Microsoft.VisualBasic | Out-Null
  $ci = New-Object Microsoft.VisualBasic.Devices.ComputerInfo
  $free = [int][math]::Round($ci.AvailablePhysicalMemory / 1MB)
  $total = [int][math]::Round($ci.TotalPhysicalMemory / 1MB)
} catch {
  $free = 0
  $total = 0
}
Write-Output "$free|$total|$logical"
'''


WINDOWS_FULL_NODE_PROBE_PS = r'''
$ErrorActionPreference = "Continue"
try {
  Add-Type -AssemblyName Microsoft.VisualBasic | Out-Null
  $ci = New-Object Microsoft.VisualBasic.Devices.ComputerInfo
  $free = [int][math]::Round($ci.AvailablePhysicalMemory / 1MB)
  $total = [int][math]::Round($ci.TotalPhysicalMemory / 1MB)
  $logical = [int][System.Environment]::ProcessorCount
  try {
    $envLogical = [int]$env:NUMBER_OF_PROCESSORS
    if ($envLogical -gt $logical) { $logical = $envLogical }
  } catch {}
  try {
    $regCount = (Get-ChildItem 'HKLM:\HARDWARE\DESCRIPTION\System\CentralProcessor' -ErrorAction Stop | Measure-Object).Count
    if ($regCount -gt $logical) { $logical = [int]$regCount }
  } catch {}
  try {
    $cs = Get-CimInstance Win32_ComputerSystem -ErrorAction SilentlyContinue
    if ($cs.NumberOfLogicalProcessors -and [int]$cs.NumberOfLogicalProcessors -gt $logical) { $logical = [int]$cs.NumberOfLogicalProcessors }
  } catch {}
  $loadPct = 0
  try {
    $cpu = Get-CimInstance Win32_Processor -ErrorAction SilentlyContinue | Measure-Object -Property LoadPercentage -Average
    if ($cpu.Average -ne $null) { $loadPct = [int][math]::Round($cpu.Average) }
  } catch { $loadPct = 0 }
  if ($loadPct -le 0) {
    try {
      $sample = (Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 1 -MaxSamples 1).CounterSamples[0].CookedValue
      $loadPct = [int][math]::Round($sample)
    } catch { $loadPct = 0 }
  }
} catch {
  $ErrorActionPreference = "Stop"
  $os = Get-CimInstance Win32_OperatingSystem
  $cs = Get-CimInstance Win32_ComputerSystem
  $cpu = Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average
  $free = [int][math]::Round($os.FreePhysicalMemory / 1024)
  $total = [int][math]::Round($os.TotalVisibleMemorySize / 1024)
  if ($cs.NumberOfLogicalProcessors -and [int]$cs.NumberOfLogicalProcessors -gt $logical) { $logical = [int]$cs.NumberOfLogicalProcessors }
  $loadPct = 0
  if ($cpu.Average -ne $null) { $loadPct = [int][math]::Round($cpu.Average) }
}
Write-Output "$free|$total|$logical|$loadPct"
'''


def schedulable_windows_cpu_from_logical(logical_cores: int, info: dict) -> int:
    try:
        declared = int(info.get("cpu_cores") or 0)
    except Exception:
        declared = 0
    try:
        logical = int(logical_cores or 0)
    except Exception:
        logical = 0
    observed = 0
    if logical > 0:
        observed = logical
        if info.get("windows_skip_ht_pair", True) and logical > 1:
            observed = max(1, logical // 2)
    if declared > 0 and observed > 0:
        return max(1, min(declared, observed))
    return max(1, declared or observed or 1)


def _parse_cheap_capacity_output(raw: str) -> dict:
    for line in reversed((raw or "").splitlines()):
        line = line.strip()
        if re.match(r"^\d+\|\d+\|\d+$", line):
            free_s, total_s, logical_s = line.split("|", 2)
            return {
                "free_ram_mb": int(free_s),
                "total_ram_mb": int(total_s),
                "logical_cpu": int(logical_s),
            }
    return {}


def cheap_windows_host_capacity(
    name: str,
    info: dict,
    *,
    deps: WindowsNodeProbeDeps,
    timeout_s: float = 8.0,
) -> dict:
    """Return cheap Windows RAM/core counters without WMI/CIM."""
    try:
        rc, out, _ = deps.run_windows_ps(
            name,
            WINDOWS_CHEAP_CAPACITY_PS,
            timeout=timeout_s,
            check=False,
        )
        if rc == 0:
            return _parse_cheap_capacity_output(out)
    except Exception:
        pass
    return {}


def _cheap_observed_cpu_capacity(
    name: str,
    info: dict,
    *,
    deps: WindowsNodeProbeDeps,
) -> tuple[int, int]:
    cap = cheap_windows_host_capacity(name, info, deps=deps)
    if cap.get("logical_cpu"):
        logical = int(cap["logical_cpu"])
        return schedulable_windows_cpu_from_logical(logical, info), logical
    total = schedulable_windows_cpu_from_logical(0, info)
    return total, 0


def windows_queue_usage(name: str, *, deps: WindowsNodeProbeDeps) -> tuple[int, int]:
    used_cpu = 0
    used_ram = 0
    try:
        for task in deps.queue_tasks():
            if task.get("status") != "running" or task.get("node") != name:
                continue
            used_cpu += max(0, int(task.get("cpu_cores") or 0))
            used_ram += max(0, int(task.get("current_ram_mb") or task.get("ram_mb") or 0))
    except Exception:
        pass
    return used_cpu, used_ram


def _declared_or_probed_total_ram(info: dict, probed_total_ram: int, fallback_free_ram: int = 0) -> int:
    declared_ram = int(info.get("ram_mb") or 0)
    if declared_ram and probed_total_ram:
        return min(declared_ram, probed_total_ram)
    return declared_ram or probed_total_ram or max(1, fallback_free_ram)


def _queue_accounting_fallback(name: str, reason: str, *, deps: WindowsNodeProbeDeps) -> dict:
    """Fallback when live Windows counters time out under heavy CPU load."""
    try:
        ping = deps.run_subprocess(
            deps.ssh_base_args(name) + ["cmd", "/c", "echo", "OK"],
            capture_output=True,
            timeout=6,
        )
        if ping.returncode != 0 or b"OK" not in (ping.stdout or b""):
            return {
                "name": name,
                "alive": False,
                "error": deps.windows_probe_error_hint(name, reason),
            }
    except Exception:
        return {
            "name": name,
            "alive": False,
            "error": deps.windows_probe_error_hint(name, reason),
        }

    info = deps.node_configs.get(name, {})
    total_cpu, observed_logical = _cheap_observed_cpu_capacity(name, info, deps=deps)
    cap = cheap_windows_host_capacity(name, info, deps=deps)
    probed_total_ram = int(cap.get("total_ram_mb") or 0)
    actual_free_ram = int(cap.get("free_ram_mb") or 0)
    declared_ram = int(info.get("ram_mb") or 0)
    if declared_ram and probed_total_ram:
        total_ram = min(declared_ram, probed_total_ram)
    else:
        total_ram = declared_ram or probed_total_ram
    used_cpu, used_ram = windows_queue_usage(name, deps=deps)
    queue_free_ram = max(0, total_ram - used_ram) if total_ram else 0
    sched_free_ram = (
        min(actual_free_ram, total_ram) if actual_free_ram and total_ram
        else queue_free_ram
    )
    return {
        "name": name,
        "alive": True,
        "gpus": [],
        "free_ram_mb": max(0, sched_free_ram),
        "actual_free_ram_mb": actual_free_ram,
        "total_ram_mb": total_ram,
        "free_cpu": max(0, total_cpu - used_cpu),
        "total_cpu": total_cpu,
        "loadavg": float(min(total_cpu, used_cpu)),
        "cpu_load_pct": int(round(100.0 * min(total_cpu, used_cpu) / max(1, total_cpu))),
        "cores": observed_logical,
        "logical_cpu": observed_logical,
        "physical_cpu": total_cpu,
        "os": "windows",
        "probe_fallback": "queue_accounting",
        "probe_fallback_reason": str(reason)[:200],
    }


def _parse_full_probe_output(raw: str) -> tuple[int, int, int, float]:
    lines = [line.strip() for line in (raw or "").splitlines()]
    line = next(
        (
            line for line in reversed(lines)
            if re.match(r"^\d+(\.\d+)?\|\d+(\.\d+)?\|\d+(\.\d+)?\|\d+(\.\d+)?$", line)
        ),
        "",
    )
    if not line:
        raise ValueError((raw or "no probe output").strip()[:120])
    free_s, total_s, logical_s, load_s = line.split("|")[:4]
    return (
        int(float(free_s)),
        int(float(total_s)),
        int(float(logical_s)),
        max(0.0, min(100.0, float(load_s))),
    )


def probe_windows_node(name: str, *, deps: WindowsNodeProbeDeps) -> dict:
    """Probe a Windows CPU-only node."""
    info = deps.node_configs.get(name, {})
    cpu_load_source = str(
        info.get("windows_cpu_load_source")
        or deps.env_cpu_load_source
        or "queue"
    ).strip().lower()

    if cpu_load_source in ("queue", "accounting", "queue_accounting"):
        cap = cheap_windows_host_capacity(name, info, deps=deps)
        if cap:
            observed_logical = int(cap.get("logical_cpu") or 0)
            total_cpu = schedulable_windows_cpu_from_logical(observed_logical, info)
            probed_total_ram = int(cap.get("total_ram_mb") or 0)
            actual_free_ram = int(cap.get("free_ram_mb") or 0)
            total_ram = _declared_or_probed_total_ram(info, probed_total_ram, actual_free_ram)
            used_cpu, _used_ram = windows_queue_usage(name, deps=deps)
            free_cpu = max(0, total_cpu - used_cpu)
            return {
                "name": name,
                "alive": True,
                "gpus": [],
                "free_ram_mb": min(actual_free_ram, total_ram) if total_ram else actual_free_ram,
                "actual_free_ram_mb": actual_free_ram,
                "total_ram_mb": total_ram,
                "free_cpu": free_cpu,
                "total_cpu": total_cpu,
                "loadavg": float(min(total_cpu, used_cpu)),
                "cpu_load_pct": int(round(100.0 * min(total_cpu, used_cpu) / max(1, total_cpu))),
                "cores": observed_logical,
                "logical_cpu": observed_logical,
                "physical_cpu": total_cpu,
                "os": "windows",
                "probe_fallback": "queue_cpu_live_ram",
            }

    try:
        rc, out, err = deps.run_windows_ps(
            name,
            WINDOWS_FULL_NODE_PROBE_PS,
            timeout=6,
            check=False,
        )
        if rc != 0:
            return {
                "name": name,
                "alive": False,
                "error": deps.windows_probe_error_hint(
                    name,
                    err or out or "powershell probe failed",
                ),
            }
        free_ram, probed_total_ram, logical_cores, load_pct = _parse_full_probe_output(out)
    except Exception as exc:
        return _queue_accounting_fallback(name, str(exc), deps=deps)

    if free_ram <= 0 or probed_total_ram <= 0 or logical_cores <= 0:
        cap = cheap_windows_host_capacity(name, info, deps=deps)
        free_ram = free_ram or int(cap.get("free_ram_mb") or 0)
        probed_total_ram = probed_total_ram or int(cap.get("total_ram_mb") or 0)
        logical_cores = logical_cores or int(cap.get("logical_cpu") or 0)

    total_ram = _declared_or_probed_total_ram(info, probed_total_ram, free_ram)
    sched_free_ram = min(free_ram, total_ram)
    total_cpu = schedulable_windows_cpu_from_logical(logical_cores, info)
    used_cpu = int(round(total_cpu * load_pct / 100.0))
    free_cpu = max(0, total_cpu - used_cpu)
    return {
        "name": name,
        "alive": True,
        "gpus": [],
        "free_ram_mb": sched_free_ram,
        "actual_free_ram_mb": free_ram,
        "total_ram_mb": total_ram,
        "free_cpu": free_cpu,
        "total_cpu": total_cpu,
        "loadavg": float(used_cpu),
        "cpu_load_pct": int(load_pct),
        "cores": logical_cores,
        "logical_cpu": logical_cores,
        "physical_cpu": total_cpu,
        "os": "windows",
    }
