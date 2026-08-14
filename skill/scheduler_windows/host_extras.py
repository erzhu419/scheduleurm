from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class WindowsHostExtrasDeps:
    path_exists: Callable[[str], bool]
    run_subprocess: Callable[..., Any]


WINDOWS_POWERSHELL_PATHS = (
    "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
    "/mnt/c/WINDOWS/system32/WindowsPowerShell/v1.0/powershell.exe",
)


def _powershell_path(deps: WindowsHostExtrasDeps) -> str:
    for path in WINDOWS_POWERSHELL_PATHS:
        if deps.path_exists(path):
            return path
    return ""


def probe_windows_host_extras(timeout_s: float = 4.0, *, deps: WindowsHostExtrasDeps) -> dict:
    """Best-effort WSL -> Windows host metrics used by local node probing."""
    out = {}
    pwsh = _powershell_path(deps)
    if not pwsh:
        return out

    script = (
        "try { "
        "  Add-Type -AssemblyName Microsoft.VisualBasic -ErrorAction Stop; "
        "  $ci = New-Object Microsoft.VisualBasic.Devices.ComputerInfo; "
        "  $free = [math]::Round($ci.AvailablePhysicalMemory / 1MB); "
        "  $tot  = [math]::Round($ci.TotalPhysicalMemory / 1MB); "
        "} catch { "
        "  $os = Get-CimInstance Win32_OperatingSystem; "
        "  $free = [math]::Round($os.FreePhysicalMemory / 1024); "
        "  $tot  = [math]::Round($os.TotalVisibleMemorySize / 1024); "
        "} "
        "$cpu = Get-CimInstance Win32_Processor -ErrorAction SilentlyContinue "
        "| Measure-Object -Property LoadPercentage -Average; "
        "$cpu_pct = if ($cpu.Average -ne $null) { [math]::Round($cpu.Average) } else { 0 }; "
        "Write-Output \"$free|$tot|$cpu_pct\""
    )
    try:
        result = deps.run_subprocess(
            [pwsh, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if result.returncode != 0:
            return out
        line = (result.stdout or "").strip().splitlines()[-1] if result.stdout else ""
        bits = line.split("|")
        if len(bits) >= 2:
            try:
                out["host_free_ram_mb"] = int(bits[0])
                out["host_total_ram_mb"] = int(bits[1])
                if len(bits) >= 3:
                    out["host_cpu_load_pct"] = int(bits[2])
            except ValueError:
                pass
    except Exception:
        pass

    gpu_script = (
        "$gpu = (Get-Counter '\\GPU Engine(*engtype_Compute*)\\Utilization "
        "Percentage' -ErrorAction SilentlyContinue).CounterSamples "
        "| Measure-Object -Maximum CookedValue; "
        "$gpu_pct = if ($gpu.Maximum) { [math]::Round($gpu.Maximum) } else { 0 }; "
        "Write-Output \"$gpu_pct\""
    )
    try:
        result = deps.run_subprocess(
            [pwsh, "-NoProfile", "-NonInteractive", "-Command", gpu_script],
            capture_output=True,
            text=True,
            timeout=max(1.0, min(3.0, timeout_s)),
        )
        if result.returncode == 0:
            line = (result.stdout or "").strip().splitlines()[-1] if result.stdout else ""
            if line:
                out["gpu_compute_util_pct"] = int(float(line))
    except Exception:
        pass
    return out
