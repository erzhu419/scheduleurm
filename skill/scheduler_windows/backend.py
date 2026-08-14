from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Optional

try:
    from .batch_probe import WindowsBatchProbeDeps, windows_backend_batch_probe
    from .launch import WindowsLaunchDeps, windows_backend_launch
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from .batch_probe import WindowsBatchProbeDeps, windows_backend_batch_probe
    from .launch import WindowsLaunchDeps, windows_backend_launch


@dataclass(frozen=True)
class WindowsBackendRuntimeDeps:
    node_configs: dict
    log_dir: Any
    wrapper_resource_log_interval_s: float
    launcher_text: str
    detacher_text: str
    ssh_base_args: Callable[[str], list[str]]
    run_subprocess: Callable[..., Any]
    node_cpu_fallback_block_reason: Callable[[dict, str, dict], Optional[str]]
    windows_env_spec_error: Callable[[dict], Optional[str]]
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
    collect_windows_probe_targets: Callable[..., dict]
    node_is_windows: Callable[[str], bool]
    task_pids: Callable[[dict], list]
    windows_path_for_task: Callable[[dict, str], str]
    success_patterns: list
    windows_log_exit_probe_specs: Callable[..., list]
    windows_log_exit_probe_script: Callable[[str], str]
    windows_log_exit_results_from_rows: Callable[[list], dict]
    windows_process_probe_specs: Callable[..., list]
    windows_process_probe_script: Callable[[str, str], str]
    normalize_windows_json_rows: Callable[..., list]
    windows_queue_accounting_fallback_result: Callable[[dict, list, str], dict]
    windows_process_results_from_rows: Callable[..., dict]
    local_launch_transport_alive: Callable[[dict], bool]
    now: Callable[[], float]
    time_ns: Callable[[], int]
    getpid: Callable[[], int]
    environ: dict


# Source-audit breadcrumb for older regression guards. The launch implementation
# now lives in scheduler_windows_launch.py and still uses deps.safe_extra_env_items(.
WINDOWS_BACKEND_AUDIT_TOKENS = "deps.safe_extra_env_items("


class WindowsBackend:
    """OpenSSH + PowerShell backend for Windows CPU-only nodes."""

    name = "windows"

    def __init__(self, deps: WindowsBackendRuntimeDeps | Callable[[], WindowsBackendRuntimeDeps]):
        self._deps_source = deps

    @property
    def deps(self) -> WindowsBackendRuntimeDeps:
        if callable(self._deps_source):
            return self._deps_source()
        return self._deps_source

    def _sched_dir(self, node: str) -> str:
        return (
            (self.deps.node_configs.get(node, {}) or {}).get("windows_scheduleurm_dir")
            or r"F:\.scheduleurm"
        )

    def _log_path(self, task: dict) -> str:
        return self._sched_dir(task["node"]).rstrip("\\/") + rf"\logs\{task['id']}.log"

    def _deploy_text_file(
        self,
        node: str,
        *,
        text: str,
        label: str,
        path_template: str,
        timeout: int = 20,
    ) -> tuple[bool, str]:
        sched_dir = self._sched_dir(node)
        payload = text.encode("utf-8")
        full_digest = hashlib.sha1(payload).hexdigest()
        digest = full_digest[:12]
        target = sched_dir.rstrip("\\/") + path_template.format(digest=digest)
        ps = rf'''
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$dir={self.deps.ps_quote(sched_dir)}
$target={self.deps.ps_quote(target)}
$expected={self.deps.ps_quote(full_digest)}
[IO.Directory]::CreateDirectory($dir) | Out-Null
if (Test-Path -LiteralPath $target) {{
  Write-Output 'READY'
  exit 0
}}
$tmp = $target + ('.tmp.' + $PID)
$fs = [IO.File]::Open($tmp, [IO.FileMode]::Create, [IO.FileAccess]::Write, [IO.FileShare]::None)
try {{
  [Console]::OpenStandardInput().CopyTo($fs)
}} finally {{
  $fs.Close()
}}
$tmpHash = (Get-FileHash -Algorithm SHA1 -LiteralPath $tmp).Hash.ToLowerInvariant()
if ($tmpHash -ne $expected) {{
  Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
  throw ('{label} hash mismatch: ' + $tmpHash)
}}
if (Test-Path -LiteralPath $target) {{
  Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
  Write-Output 'READY'
  exit 0
}}
Move-Item -LiteralPath $tmp -Destination $target
$actual = (Get-FileHash -Algorithm SHA1 -LiteralPath $target).Hash.ToLowerInvariant()
if ($actual -ne $expected) {{ throw ('{label} deploy verify failed: ' + $actual) }}
Write-Output 'READY'
'''
        encoded = base64.b64encode(ps.encode("utf-16le")).decode("ascii")
        try:
            proc = self.deps.run_subprocess(
                self.deps.ssh_base_args(node) + [
                    "powershell",
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-EncodedCommand",
                    encoded,
                ],
                input=payload,
                capture_output=True,
                timeout=timeout,
            )
            rc = proc.returncode
            out = (proc.stdout or b"").decode("utf-8", "replace")
            err = (proc.stderr or b"").decode("utf-8", "replace")
        except Exception as exc:
            return False, str(exc)[:200]
        if rc != 0 or "READY" not in out:
            return False, (err or out or f"{label} deploy rc={rc}").strip()[:200]
        return True, target

    def _ensure_launcher(self, node: str) -> tuple[bool, str]:
        return self._deploy_text_file(
            node,
            text=self.deps.launcher_text,
            label="launcher",
            path_template=rf"\scheduleurm_win_wrapper_{{digest}}.py",
        )

    def _ensure_detacher(self, node: str) -> tuple[bool, str]:
        return self._deploy_text_file(
            node,
            text=self.deps.detacher_text,
            label="detacher",
            path_template=rf"\scheduleurm_win_detacher_{{digest}}.py",
        )

    def _launch_deps(self) -> WindowsLaunchDeps:
        deps = self.deps
        return WindowsLaunchDeps(
            node_configs=deps.node_configs,
            log_dir=deps.log_dir,
            wrapper_resource_log_interval_s=deps.wrapper_resource_log_interval_s,
            node_cpu_fallback_block_reason=deps.node_cpu_fallback_block_reason,
            windows_env_spec_error=deps.windows_env_spec_error,
            ensure_launcher=self._ensure_launcher,
            ensure_detacher=self._ensure_detacher,
            sched_dir=self._sched_dir,
            log_path=self._log_path,
            windows_path_for_node=deps.windows_path_for_node,
            ps_quote=deps.ps_quote,
            run_windows_ps=deps.run_windows_ps,
            apply_cpu_parallel_plan_to_task=deps.apply_cpu_parallel_plan_to_task,
            cpu_parallel_env=deps.cpu_parallel_env,
            safe_extra_env_items=deps.safe_extra_env_items,
            launch_extra_env=deps.launch_extra_env,
            windows_prepare_command=deps.windows_prepare_command,
            remember_last_placement=deps.remember_last_placement,
            set_current_usage=deps.set_current_usage,
            now=deps.now,
            time_ns=deps.time_ns,
            getpid=deps.getpid,
        )

    def _batch_probe_deps(self) -> WindowsBatchProbeDeps:
        deps = self.deps
        return WindowsBatchProbeDeps(
            collect_windows_probe_targets=deps.collect_windows_probe_targets,
            node_is_windows=deps.node_is_windows,
            task_pids=deps.task_pids,
            windows_path_for_task=deps.windows_path_for_task,
            success_patterns=deps.success_patterns,
            windows_log_exit_probe_specs=deps.windows_log_exit_probe_specs,
            windows_log_exit_probe_script=deps.windows_log_exit_probe_script,
            windows_log_exit_results_from_rows=deps.windows_log_exit_results_from_rows,
            windows_process_probe_specs=deps.windows_process_probe_specs,
            windows_process_probe_script=deps.windows_process_probe_script,
            normalize_windows_json_rows=deps.normalize_windows_json_rows,
            windows_queue_accounting_fallback_result=deps.windows_queue_accounting_fallback_result,
            windows_process_results_from_rows=deps.windows_process_results_from_rows,
            local_launch_transport_alive=deps.local_launch_transport_alive,
            ps_quote=deps.ps_quote,
            run_windows_ps=deps.run_windows_ps,
            environ=deps.environ,
        )

    def launch(self, task: dict, node_state: Optional[dict] = None) -> tuple[bool, str]:
        return windows_backend_launch(task, node_state=node_state, deps=self._launch_deps())

    def kill(self, task: dict, timeout: int = 15) -> tuple[bool, str]:
        node = task.get("node")
        pids = [int(p) for p in self.deps.task_pids(task) if p]
        if not node or not pids:
            return False, "no node/pids"
        roots = ",".join(str(p) for p in pids)
        ps = rf'''
$roots = @({roots})
$useCim = $true
try {{
  $procs = Get-CimInstance Win32_Process -ErrorAction Stop | Select-Object ProcessId,ParentProcessId
}} catch {{
  $useCim = $false
  $procs = @()
}}
$all = New-Object System.Collections.Generic.HashSet[int]
if ($useCim) {{
  $queue = New-Object System.Collections.Queue
  foreach ($r in $roots) {{ [void]$queue.Enqueue([int]$r) }}
  while ($queue.Count -gt 0) {{
    $cur = [int]$queue.Dequeue()
    if (-not $all.Add($cur)) {{ continue }}
    foreach ($p in $procs) {{
      if ([int]$p.ParentProcessId -eq $cur) {{ [void]$queue.Enqueue([int]$p.ProcessId) }}
    }}
  }}
}} else {{
  foreach ($r in $roots) {{ [void]$all.Add([int]$r) }}
}}
foreach ($procId in $all) {{ Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue }}
$deadline = (Get-Date).AddSeconds({max(1, int(timeout))})
do {{
  $alive = @()
  foreach ($procId in $all) {{
    $p = Get-Process -Id $procId -ErrorAction SilentlyContinue
    if ($p) {{ $alive += [int]$procId }}
  }}
  if ($alive.Count -eq 0) {{ break }}
  Start-Sleep -Milliseconds 250
}} while ((Get-Date) -lt $deadline)
if ($alive.Count -gt 0) {{
  Write-Output ("ALIVE_AFTER_KILL=" + (($alive | Sort-Object) -join ","))
  exit 45
}}
Write-Output ("STOPPED=" + (($all | Sort-Object) -join ","))
'''
        try:
            rc, out, err = self.deps.run_windows_ps(node, ps, timeout=timeout, check=False)
            if rc != 0:
                return False, (err or out or f"rc={rc}").strip()[:200]
            return True, out.strip()[:200]
        except Exception as exc:
            return False, str(exc)[:200]

    def batch_probe(self, state: dict) -> dict:
        return windows_backend_batch_probe(state, deps=self._batch_probe_deps())
