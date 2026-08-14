"""Windows running-task log/process probe protocol helpers."""

from __future__ import annotations

import json
from typing import Callable


def _int_list(values) -> list[int]:
    out: list[int] = []
    for value in values or []:
        try:
            out.append(int(value))
        except Exception:
            continue
    return out


def windows_terminal_log_result(rc_int):
    return {
        "state": "dead",
        "alive_pids": [],
        "vram_mb": 0,
        "ram_mb": 0,
        "pcpu": 0.0,
        "terminal_ok": (rc_int == 0) if rc_int is not None else None,
        "backend_state": (
            f"WINDOWS_LOG_RC_{rc_int}" if rc_int is not None else "WINDOWS_LOG_EXIT"
        ),
        "terminal_reason": (
            f"windows wrapper child exit rc={rc_int}"
            if rc_int is not None else
            "windows wrapper child exit observed in log"
        ),
        "exit_code": rc_int,
    }


def collect_windows_probe_targets(
    state: dict,
    *,
    node_is_windows: Callable[[str], bool],
    task_pids: Callable[[dict], list],
) -> dict:
    by_node: dict = {}
    for task in state.get("tasks", []):
        if task.get("status") != "running":
            continue
        node = task.get("node")
        if not node or not node_is_windows(node):
            continue
        pids = task_pids(task)
        if not pids:
            continue
        by_node.setdefault(node, []).append((task, pids))
    return by_node


def windows_log_exit_probe_specs(
    tasks_for_node: list,
    *,
    windows_path_for_task: Callable[[dict, str], str],
    success_patterns=(),
) -> list[dict]:
    specs = []
    scan_all = True
    for task, pids in tasks_for_node:
        log_path = task.get("log_path") or ""
        if not log_path:
            continue
        known = _int_list(task.get("alive_pids") or [])
        roots = _int_list(pids or [])
        last_line = task.get("last_progress_line") or ""
        try:
            progress_ratio = float(task.get("progress_ratio") or 0.0)
        except Exception:
            progress_ratio = 0.0
        if progress_ratio <= 0:
            try:
                cur = float(task.get("runtime_current_unit") or 0.0)
                total = float(task.get("runtime_total_units") or 0.0)
                if total > 0:
                    progress_ratio = cur / total
            except Exception:
                pass
        try:
            eta_seconds = float(task.get("eta_seconds") or 0.0)
        except Exception:
            eta_seconds = 0.0
        if (not scan_all
                and "scheduleurm child exit rc=" not in last_line
                and not any(p in last_line for p in success_patterns)
                and progress_ratio < 0.999
                and not (eta_seconds > 0 and eta_seconds <= 600 and progress_ratio >= 0.98)):
            continue
        specs.append({
            "id": task.get("id"),
            "log": windows_path_for_task(task, log_path),
            "roots": sorted(set(roots)),
            "known": sorted(set(known)),
        })
    return specs


def windows_process_probe_specs(
    tasks_for_node: list,
    *,
    windows_path_for_task: Callable[[dict, str], str],
) -> list[dict]:
    specs = []
    for task, pids in tasks_for_node:
        known = _int_list(task.get("alive_pids") or [])
        for pid in pids:
            specs.append({
                "root": int(pid),
                "log": windows_path_for_task(task, task.get("log_path") or ""),
                "known": sorted(set(known)),
                "started_at": float(task.get("started_at") or 0),
                "finished_at": float(task.get("finished_at") or 0),
            })
    return specs


def normalize_windows_json_rows(raw: str, *, require_nonempty: bool) -> list:
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty windows process probe output")
    data = json.loads(text)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("windows process probe returned non-list rows")
    if require_nonempty and not data:
        raise ValueError("windows process probe returned no rows")
    return data


def windows_log_exit_results_from_rows(rows: list) -> dict:
    ret = {}
    for row in rows:
        tid = str(row.get("id") or "")
        if not tid:
            continue
        rc_val = row.get("exit_code")
        try:
            rc_int = int(rc_val)
        except Exception:
            rc_int = None
        if row.get("log_exit"):
            ret[tid] = windows_terminal_log_result(rc_int)
            continue
        alive = row.get("alive") or []
        if isinstance(alive, int):
            alive = [alive]
        alive = sorted(set(int(x) for x in alive))
        if alive:
            ret[tid] = {
                "state": "alive",
                "alive_pids": alive,
                "vram_mb": 0,
                "ram_mb": int(row.get("ram_mb") or 0),
                "pcpu": 0.0,
                "probe_fallback": "windows_log_pid_liveness",
            }
            continue
        if int(row.get("candidate_count") or 0) > 0:
            ret[tid] = {
                "state": "dead",
                "alive_pids": [],
                "vram_mb": 0,
                "ram_mb": 0,
                "pcpu": 0.0,
                "probe_fallback": "windows_log_pid_liveness",
            }
    return ret


def windows_queue_accounting_fallback_result(task: dict, pids: list, reason: str) -> dict:
    known = _int_list(task.get("alive_pids") or [])
    known.extend(_int_list(pid for pid in (pids or []) if pid))
    ram = int(task.get("current_ram_mb") or task.get("peak_ram_mb") or task.get("ram_mb") or 0)
    pcpu = float(max(1, int(task.get("cpu_cores") or 1)) * 100)
    return {
        "state": "alive",
        "alive_pids": sorted(set(known)),
        "vram_mb": 0,
        "ram_mb": ram,
        "pcpu": pcpu,
        "probe_fallback": reason,
    }


def windows_process_results_from_rows(
    tasks_for_node: list,
    rows: list,
    *,
    local_launch_transport_alive: Callable[[dict], bool],
) -> dict:
    results = {}
    by_root = {int(row.get("root")): row for row in rows if row.get("root") is not None}
    for task, pids in tasks_for_node:
        alive = []
        ram = 0
        pcpu = 0.0
        for pid in pids:
            rec = by_root.get(int(pid))
            if not rec:
                continue
            if rec.get("log_exit"):
                rc_val = rec.get("exit_code")
                try:
                    rc_int = int(rc_val)
                except Exception:
                    rc_int = None
                results[task["id"]] = windows_terminal_log_result(rc_int)
                break
            row_alive = rec.get("alive") or []
            if isinstance(row_alive, int):
                row_alive = [row_alive]
            alive.extend(int(x) for x in row_alive)
            ram += int(rec.get("ram_mb") or 0)
            try:
                pcpu += float(rec.get("pcpu") or 0.0)
            except Exception:
                pass
        if task["id"] in results:
            continue
        if not alive:
            if local_launch_transport_alive(task):
                results[task["id"]] = {
                    "state": "unknown",
                    "alive_pids": [],
                    "vram_mb": 0,
                    "ram_mb": 0,
                    "pcpu": 0.0,
                    "error": "windows root missing but local launch transport is still alive",
                }
                continue
            results[task["id"]] = {
                "state": "dead",
                "alive_pids": [],
                "vram_mb": 0,
                "ram_mb": 0,
                "pcpu": 0.0,
            }
        else:
            if pcpu <= 0:
                pcpu = float(max(1, int(task.get("cpu_cores") or 1)) * 100)
            results[task["id"]] = {
                "state": "alive",
                "alive_pids": sorted(set(alive)),
                "vram_mb": 0,
                "ram_mb": ram,
                "pcpu": pcpu,
            }
    return results


def windows_log_exit_probe_script(quoted_spec_json: str) -> str:
    return rf'''
$specs = @(({quoted_spec_json} | ConvertFrom-Json))
$rows = @()
	foreach ($spec in $specs) {{
	  $id = [string]$spec.id
	  $log = [string]$spec.log
	  $all = New-Object System.Collections.Generic.HashSet[int]
	  foreach ($pid0 in @($spec.roots)) {{
	    try {{ [void]$all.Add([int]$pid0) }} catch {{ }}
	  }}
	  foreach ($pid0 in @($spec.known)) {{
	    try {{ [void]$all.Add([int]$pid0) }} catch {{ }}
	  }}
	  $logExit = $false
	  $logRc = $null
	  $logMtime = 0
  if ($log -and (Test-Path -LiteralPath $log)) {{
    try {{
      $it = Get-Item -LiteralPath $log -ErrorAction SilentlyContinue
      if ($it) {{ $logMtime = [int64]([DateTimeOffset]::new($it.LastWriteTimeUtc).ToUnixTimeSeconds()) }}
      $fs = [IO.File]::Open($log, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
      try {{
        $len = [int64]$fs.Length
        $start = [Math]::Max([int64]0, $len - 4096)
        [void]$fs.Seek($start, [IO.SeekOrigin]::Begin)
        $buf = New-Object byte[] ([int]($len - $start))
        $read = $fs.Read($buf, 0, $buf.Length)
        $text = [Text.Encoding]::UTF8.GetString($buf, 0, $read)
        foreach ($line in ($text -split "`r?`n")) {{
          if ($line -match 'scheduleurm windows wrapper start' -or
              $line -match 'scheduleurm child pid=' -or
              $line -match 'scheduleurm resource-progress') {{ $logExit = $false; $logRc = $null }}
	          if ($line -match 'scheduleurm child exit rc=([+-]?\d+)') {{
	            $logExit = $true
	            $logRc = [int]$Matches[1]
	          }}
	          if ($line -match 'scheduleurm child pid=(\d+)') {{ [void]$all.Add([int]$Matches[1]) }}
	          if ($line -match 'pid=(\d+)') {{ [void]$all.Add([int]$Matches[1]) }}
	          if ($line -match '"root_pid"\s*:\s*(\d+)') {{ [void]$all.Add([int]$Matches[1]) }}
	        }}
	      }} finally {{
	        $fs.Close()
	      }}
	    }} catch {{ }}
	  }}
	  $alive = @()
	  $rss = 0
	  foreach ($procId in $all) {{
	    try {{
	      $p = Get-Process -Id ([int]$procId) -ErrorAction SilentlyContinue
	      if (-not $p) {{ continue }}
	      $alive += [int]$procId
	      $rss += [int64]$p.WorkingSet64
	    }} catch {{ }}
	  }}
	  $rows += [pscustomobject]@{{
	    id=$id
	    alive=$alive
	    ram_mb=[int][math]::Round($rss / 1MB)
	    candidate_count=[int]$all.Count
	    log_exit=[bool]$logExit
	    exit_code=$logRc
	    log_mtime=[int64]$logMtime
  }}
}}
$rows | ConvertTo-Json -Compress -Depth 4
'''


def windows_process_probe_script(root_list: str, quoted_spec_json: str) -> str:
    return rf'''
$roots = @({root_list})
$specs = @(({quoted_spec_json} | ConvertFrom-Json))
$rows = @()
foreach ($spec in $specs) {{
  $r = [int]$spec.root
  $startedAt = [double]($spec.started_at)
  $finishedAt = [double]($spec.finished_at)
  $all = New-Object System.Collections.Generic.HashSet[int]
  [void]$all.Add([int]$r)
  foreach ($knownPid in @($spec.known)) {{
    try {{ [void]$all.Add([int]$knownPid) }} catch {{ }}
  }}
  $log = [string]$spec.log
  $logExit = $false
  $logRc = $null
  $logMtime = 0
  $seedCountBeforeLog = $all.Count
  if ($log -and (Test-Path -LiteralPath $log)) {{
    try {{
      $it = Get-Item -LiteralPath $log -ErrorAction SilentlyContinue
      if ($it) {{ $logMtime = [int64]([DateTimeOffset]::new($it.LastWriteTimeUtc).ToUnixTimeSeconds()) }}
    }} catch {{ }}
    $lines = Get-Content -LiteralPath $log -Tail 30 -ErrorAction SilentlyContinue
    foreach ($line in $lines) {{
      if ($line -match 'scheduleurm windows wrapper start' -or
          $line -match 'scheduleurm child pid=' -or
          $line -match 'scheduleurm resource-progress') {{ $logExit = $false; $logRc = $null }}
      if ($line -match 'scheduleurm child exit rc=([+-]?\d+)') {{ $logExit = $true; $logRc = [int]$Matches[1] }}
      if ($line -match 'scheduleurm child pid=(\d+)') {{ [void]$all.Add([int]$Matches[1]) }}
      if ($line -match 'pid=(\d+)') {{ [void]$all.Add([int]$Matches[1]) }}
      if ($line -match '"root_pid"\s*:\s*(\d+)') {{ [void]$all.Add([int]$Matches[1]) }}
    }}
  }}
  $alive = @()
  $rss = 0
  foreach ($procId in $all) {{
    try {{
      $p = Get-Process -Id ([int]$procId) -ErrorAction SilentlyContinue
      if (-not $p) {{ continue }}
      $keep = $true
      try {{
        if ($p.StartTime) {{
          $unix = [int64](([DateTimeOffset]$p.StartTime).ToUnixTimeSeconds())
          if ($startedAt -gt 0 -and $unix -lt ($startedAt - 120)) {{ $keep = $false }}
          if ($finishedAt -gt 0 -and $unix -gt ($finishedAt + 120)) {{ $keep = $false }}
        }}
      }} catch {{ }}
      if (-not $keep) {{ continue }}
      $alive += [int]$procId
      $rss += [int64]$p.WorkingSet64
    }} catch {{ }}
  }}
  $rows += [pscustomobject]@{{
    root=[int]$r
    alive=$alive
    ram_mb=[int][math]::Round($rss / 1MB)
    log_exit=[bool]$logExit
    exit_code=$logRc
    log_mtime=[int64]$logMtime
    log_seeded=[bool]($all.Count -gt $seedCountBeforeLog)
  }}
}}
$rows | ConvertTo-Json -Compress -Depth 4
'''
