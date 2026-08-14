from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class WindowsTaskPathDeps:
    node_is_windows: Callable[[str], bool]
    windows_path_for_node: Callable[[str, str], str]
    home: Callable[[], Path]


@dataclass(frozen=True)
class WindowsLogDeps:
    windows_path_for_task: Callable[[dict, str], str]
    windows_tail_ps: Callable[..., str]
    ps_quote: Callable[[str], str]
    run_windows_ps: Callable[..., tuple[int, str, str]]


def windows_tail_ps(path: str, max_bytes: int = 4096, *, ps_quote: Callable[[str], str]) -> str:
    """PowerShell snippet that prints the last max_bytes of a text log."""
    max_bytes = max(1, int(max_bytes))
    return rf'''
$p = {ps_quote(path)}
if (Test-Path -LiteralPath $p) {{
  $fs = [IO.File]::Open($p, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
  try {{
    $len = [int64]$fs.Length
    $start = [Math]::Max([int64]0, $len - {max_bytes})
    [void]$fs.Seek($start, [IO.SeekOrigin]::Begin)
    $buf = New-Object byte[] ([int]($len - $start))
    $read = $fs.Read($buf, 0, $buf.Length)
    [Console]::OutputEncoding = [Text.Encoding]::UTF8
    [Text.Encoding]::UTF8.GetString($buf, 0, $read)
  }} finally {{
    $fs.Close()
  }}
}}
'''


def is_windows_native_path(path: str) -> bool:
    text = str(path or "").strip().strip('"').strip("'")
    return bool(re.match(r"^[A-Za-z]:[\\/]", text) or text.startswith("\\\\"))


def windows_path_for_task(task: dict, path: str, *, deps: WindowsTaskPathDeps) -> str:
    """Map an absolute or cwd-relative task path onto the Windows node layout."""
    node = task.get("node")
    if not node or not deps.node_is_windows(node) or not path:
        return path
    text = str(path).strip().strip('"').strip("'")
    if is_windows_native_path(text):
        return text.replace("/", "\\")
    if text.startswith("/home/") or text.startswith(str(deps.home())):
        return deps.windows_path_for_node(node, text)
    cwd = task.get("cwd") or ""
    if cwd:
        base = deps.windows_path_for_node(node, cwd).rstrip("\\/")
        return base + "\\" + text.replace("/", "\\")
    return text.replace("/", "\\")


def fetch_windows_text_tail(
    task: dict,
    path: str,
    max_bytes: int = 4096,
    *,
    deps: WindowsLogDeps,
) -> tuple[str, int]:
    node = task.get("node")
    if not node:
        return ("", 0)
    win_path = deps.windows_path_for_task(task, path)
    ps = deps.windows_tail_ps(win_path, max_bytes=max_bytes) + rf'''
Write-Output '___SZ___'
if (Test-Path -LiteralPath {deps.ps_quote(win_path)}) {{
  Write-Output ([int64](Get-Item -LiteralPath {deps.ps_quote(win_path)}).Length)
}} else {{
  Write-Output 0
}}
'''
    rc, out, _ = deps.run_windows_ps(node, ps, timeout=10, check=False)
    if rc != 0 or not out:
        return ("", 0)
    if "___SZ___" in out:
        body, _, size_text = out.rpartition("___SZ___")
        try:
            return body, int((size_text or "0").strip().splitlines()[-1])
        except Exception:
            return body, 0
    return out, 0


def scan_windows_log_for_patterns(
    task: dict,
    patterns: list[str],
    *,
    deps: WindowsLogDeps,
) -> list[str]:
    node = task.get("node")
    log_path = task.get("log_path")
    if not node or not log_path:
        return []
    win_path = deps.windows_path_for_task(task, log_path)
    patterns_b64 = base64.b64encode(json.dumps(patterns).encode("utf-8")).decode("ascii")
    ps = rf'''
$path = {deps.ps_quote(win_path)}
if (-not (Test-Path -LiteralPath $path)) {{ exit 0 }}
$patterns = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String({deps.ps_quote(patterns_b64)})) | ConvertFrom-Json
$hits = @()
foreach ($pat in $patterns) {{
  if (Select-String -LiteralPath $path -SimpleMatch -CaseSensitive -Pattern $pat -Quiet -ErrorAction SilentlyContinue) {{
    $hits += [string]$pat
  }}
}}
$hits | ConvertTo-Json -Compress
'''
    rc, out, _ = deps.run_windows_ps(node, ps, timeout=15, check=False)
    if rc != 0 or not (out or "").strip():
        return []
    try:
        data = json.loads(out.strip().splitlines()[-1])
        if isinstance(data, str):
            return [data]
        if isinstance(data, list):
            return [str(item) for item in data]
    except Exception:
        return []
    return []
