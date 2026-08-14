"""Final-model artifact success detection for terminal task diagnosis."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class FinalModelSuccessDeps:
    node_configs: dict
    final_model_file_groups_from_cmd: Callable[[dict], list]
    mtimes_match_task_window: Callable[[dict, list], bool]
    node_is_windows: Callable[[str], bool]
    windows_path_for_task: Callable[[dict, str], str]
    ps_quote: Callable[[str], str]
    run_windows_ps: Callable[..., tuple]
    bash_path_arg: Callable[[str], str]
    run_on: Callable[..., tuple]
    expanduser: Callable[[str], str] = os.path.expanduser


def terminal_final_model_success(task: dict, *, deps: FinalModelSuccessDeps) -> str:
    """Treat complete final model file groups as a success marker."""
    groups = deps.final_model_file_groups_from_cmd(task)
    if not groups:
        return ""
    node = task.get("node") or "local"
    node_info = deps.node_configs.get(node) or {}
    for _model_dir, files in groups:
        try:
            if node_info.get("host") is None:
                mtimes = []
                for path in files:
                    candidate = Path(deps.expanduser(path))
                    if not candidate.is_file():
                        break
                    mtimes.append(int(candidate.stat().st_mtime))
                if deps.mtimes_match_task_window(task, mtimes):
                    return "final_model_files"
            elif deps.node_is_windows(node):
                win_files = [deps.windows_path_for_task(task, path) for path in files]
                arr = "@(" + ",".join(deps.ps_quote(path) for path in win_files) + ")"
                ps = rf'''
$paths = {arr}
$mtimes = @()
foreach ($p in $paths) {{
  if (-not (Test-Path -LiteralPath $p -PathType Leaf)) {{ exit 1 }}
  $mtime = (Get-Item -LiteralPath $p).LastWriteTimeUtc
  $mtimes += [int64]([DateTimeOffset]::new($mtime).ToUnixTimeSeconds())
}}
$mtimes | ConvertTo-Json -Compress
'''
                rc, out, _ = deps.run_windows_ps(node, ps, timeout=10, check=False)
                if rc != 0 or not (out or "").strip():
                    continue
                data = json.loads(out.strip().splitlines()[-1])
                mtimes = data if isinstance(data, list) else [data]
                if deps.mtimes_match_task_window(task, mtimes):
                    return "final_model_files"
            else:
                quoted = [deps.bash_path_arg(path) for path in files]
                tests = " && ".join(f"test -f {path}" for path in quoted)
                cmd = f"{tests} && stat -c %Y {' '.join(quoted)}"
                rc, out, _ = deps.run_on(node, cmd, timeout=10, check=False)
                if rc != 0 or not (out or "").strip():
                    continue
                mtimes = [int(value) for value in out.split() if value.strip().isdigit()]
                if deps.mtimes_match_task_window(task, mtimes):
                    return "final_model_files"
        except Exception:
            continue
    return ""
