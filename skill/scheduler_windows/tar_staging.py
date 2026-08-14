from __future__ import annotations

import base64
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class WindowsTarStagingDeps:
    node_is_windows: Callable[[str], bool]
    windows_path_for_node: Callable[[str, str], str]
    ps_quote: Callable[[str], str]
    ssh_base_args: Callable[[str], list[str]]
    run_windows_ps: Callable[..., tuple[int, str, str]]
    popen: Callable[..., Any] = subprocess.Popen
    run_subprocess: Callable[..., Any] = subprocess.run


def tar_exclude_args(extra_excludes: list | None = None) -> list[str]:
    patterns = [
        ".git", "__pycache__", "*.pyc",
        "results", "results_*",
        "logs", "logs_*",
        "experiment_output",
        "archive*", "*.tar.gz",
    ]
    for exclude in extra_excludes or []:
        clean = str(exclude or "").strip().strip("/")
        if not clean:
            continue
        patterns.append(clean)
        patterns.append(f"./{clean}")
    return [f"--exclude={pattern}" for pattern in patterns]


def stage_local_dir_to_windows(
    local_dir: str,
    target_node: str,
    target_dir: str,
    *,
    extra_excludes: list | None = None,
    timeout_s: int = 600,
    deps: WindowsTarStagingDeps,
) -> tuple[bool, str]:
    """Stream a local directory to a Windows node using tar over SSH."""
    if not deps.node_is_windows(target_node):
        return False, f"target {target_node} is not a Windows node"
    src = Path(local_dir)
    if not src.is_dir():
        return False, f"local directory missing for Windows staging: {local_dir}"
    win_dest = deps.windows_path_for_node(target_node, target_dir)
    ps = (
        f"$dest={deps.ps_quote(win_dest)}; "
        "[IO.Directory]::CreateDirectory($dest) | Out-Null; "
        "tar -xf - -C $dest; "
        "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; "
        "Write-Output 'READY'"
    )
    encoded = base64.b64encode(ps.encode("utf-16le")).decode("ascii")
    tar_args = ["tar", "-h", "-C", str(src)] + tar_exclude_args(extra_excludes) + ["-cf", "-", "."]
    ssh_args = deps.ssh_base_args(target_node) + [
        "powershell",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy", "Bypass",
        "-EncodedCommand", encoded,
    ]
    tar_proc = None
    try:
        tar_proc = deps.popen(
            tar_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        ssh_proc = deps.run_subprocess(
            ssh_args,
            stdin=tar_proc.stdout,
            capture_output=True,
            timeout=timeout_s,
        )
        if tar_proc.stdout:
            tar_proc.stdout.close()
        _, tar_err = tar_proc.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        if tar_proc:
            try:
                tar_proc.kill()
            except Exception:
                pass
        return False, f"Windows tar staging timeout (>{timeout_s}s)"
    except Exception as exc:
        if tar_proc:
            try:
                tar_proc.kill()
            except Exception:
                pass
        return False, f"Windows tar staging exception: {str(exc)[:200]}"

    tar_rc = tar_proc.returncode if tar_proc else 1
    tar_err_s = (tar_err or b"").decode("utf-8", "replace").strip()
    out_s = (ssh_proc.stdout or b"").decode("utf-8", "replace")
    err_s = (ssh_proc.stderr or b"").decode("utf-8", "replace")
    if ssh_proc.returncode != 0 or "READY" not in out_s:
        return False, (
            f"Windows extract failed rc={ssh_proc.returncode}: "
            f"{(err_s or out_s).strip()[:200]}"
        )
    if tar_rc != 0:
        return False, f"local tar failed rc={tar_rc}: {tar_err_s[:200]}"
    rc, out, err = deps.run_windows_ps(
        target_node,
        f"if (Test-Path -LiteralPath {deps.ps_quote(win_dest)}) {{ 'OK' }} else {{ 'MISSING' }}",
        timeout=10,
        check=False,
    )
    if rc != 0 or "OK" not in out:
        return False, f"Windows staged directory not verified: {win_dest}; {(err or out).strip()[:160]}"
    return True, f"synced to Windows {win_dest}"
