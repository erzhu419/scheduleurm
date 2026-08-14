from __future__ import annotations

import base64
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

try:
    from scheduler_data_motion.rsync import run_local_rsync, run_remote_rsync_command
except ModuleNotFoundError:
    from ..scheduler_data_motion.rsync import run_local_rsync, run_remote_rsync_command


@dataclass(frozen=True)
class ResultSyncExecutorDeps:
    node_is_windows: Callable[[str], bool]
    sync_windows_result: Callable[[dict], tuple[bool, str]]
    remote_path_for_node: Callable[[str, str], str]
    relay_node_for_node: Callable[[str], Optional[str]]
    relay_path_for_node: Callable[[str, str], str]
    relay_ssh_target_for_node: Callable[[str], str]
    run_on: Callable[..., tuple[int, str, str]]
    ssh_rsync_shell_for_node: Callable[[str], str]
    rsync_path_for_node: Callable[[str, str], str]
    ssh_target_for_node: Callable[[str], str]
    result_sync_timeout_s: int
    subprocess_run: Callable[..., Any] = subprocess.run


def _mkdir_local_result_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class WindowsResultSyncDeps:
    windows_path_for_node: Callable[[str, str], str]
    ps_quote: Callable[[str], str]
    ssh_base_args: Callable[[str], list[str]]
    result_sync_timeout_s: int
    make_local_dir: Callable[[str], Any] = _mkdir_local_result_dir
    popen: Callable[..., Any] = subprocess.Popen
    subprocess_run: Callable[..., Any] = subprocess.run
    pipe: Any = subprocess.PIPE


def _stream_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _sync_result_dirs(candidate: dict, deps: ResultSyncExecutorDeps) -> tuple[bool, str]:
    result_dirs = list(candidate.get("result_dirs") or [])
    local_dirs = list(candidate.get("local_result_dirs") or [])
    errors = []
    synced = 0
    for idx, result_dir in enumerate(result_dirs):
        child = dict(candidate)
        child.pop("result_dirs", None)
        child.pop("local_result_dirs", None)
        child.pop("local_result_dir_base", None)
        child["result_dir"] = result_dir
        if idx < len(local_dirs) and local_dirs[idx]:
            child["local_result_dir"] = local_dirs[idx]
        elif candidate.get("local_result_dir_base"):
            child["local_result_dir"] = os.path.join(
                candidate["local_result_dir_base"],
                os.path.basename(str(result_dir).rstrip("/")),
            )
        else:
            child["local_result_dir"] = result_dir
        ok, msg = sync_one_result(child, deps=deps)
        if ok:
            synced += 1
        else:
            errors.append(f"{result_dir}: {msg}")
    if errors:
        return (False, f"synced {synced}/{len(result_dirs)} dirs; " + "; ".join(errors)[:220])
    return (True, f"synced {synced} dirs")


def _ensure_local_result_dir(path: str) -> tuple[bool, str]:
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return True, ""
    except Exception as exc:
        return False, f"mkdir local target failed: {str(exc)[:120]}"


def _sync_via_relay(
    node: str,
    remote_dir: str,
    dst: str,
    deps: ResultSyncExecutorDeps,
) -> tuple[bool, str]:
    relay_node = deps.relay_node_for_node(node)
    if not relay_node:
        return (False, "no relay node")

    relay_dir = deps.relay_path_for_node(node, remote_dir)
    target = deps.relay_ssh_target_for_node(node)
    pull_to_relay = [
        "rsync", "-az", "--partial",
        f"{target}:{remote_dir.rstrip('/')}/",
        relay_dir.rstrip("/") + "/",
    ]
    pull_cmd = " ".join(shlex.quote(arg) for arg in pull_to_relay)
    try:
        deps.run_on(relay_node, f"mkdir -p {shlex.quote(relay_dir)}", timeout=10, check=False)
    except Exception as exc:
        return (False, f"relay result pull exception: {str(exc)[:200]}")

    ok, msg = run_remote_rsync_command(
        node=relay_node,
        command=pull_cmd,
        run_on=deps.run_on,
        operation="relay result pull",
        timeout_s=deps.result_sync_timeout_s,
    )
    if not ok:
        return False, msg

    ok, msg = run_local_rsync(
        [
            "rsync", "-az", "--partial", "-e", deps.ssh_rsync_shell_for_node(relay_node),
            deps.rsync_path_for_node(relay_node, relay_dir.rstrip("/") + "/"),
            dst,
        ],
        operation="relay->local rsync",
        run_subprocess=deps.subprocess_run,
        timeout_s=deps.result_sync_timeout_s,
        timeout_human=f">{deps.result_sync_timeout_s}s",
    )
    if not ok:
        return False, msg
    return True, "ok"


def _sync_direct_rsync(
    node: str,
    remote_dir: str,
    dst: str,
    deps: ResultSyncExecutorDeps,
) -> tuple[bool, str]:
    src = f"{deps.ssh_target_for_node(node)}:{remote_dir.rstrip('/')}/"
    ok, msg = run_local_rsync(
        ["rsync", "-az", "--partial", "-e", deps.ssh_rsync_shell_for_node(node), src, dst],
        operation="rsync",
        run_subprocess=deps.subprocess_run,
        timeout_s=deps.result_sync_timeout_s,
        timeout_human=f">{deps.result_sync_timeout_s}s",
    )
    if not ok:
        return False, msg
    return True, "ok"


def sync_windows_result(candidate: dict, *, deps: WindowsResultSyncDeps) -> tuple[bool, str]:
    """Pull a Windows result directory to local via SSH+tar."""
    node = candidate["node"]
    remote_dir = deps.windows_path_for_node(node, candidate["result_dir"].rstrip("/"))
    dst = candidate["local_result_dir"].rstrip("/") + "/"
    try:
        deps.make_local_dir(dst)
    except Exception as exc:
        return (False, f"mkdir local target failed: {str(exc)[:120]}")

    ps = (
        f"$src={deps.ps_quote(remote_dir)}; "
        "if (!(Test-Path -LiteralPath $src)) { Write-Error \"missing result_dir $src\"; exit 44 }; "
        "tar -cf - -C $src .; "
        "exit $LASTEXITCODE"
    )
    encoded = base64.b64encode(ps.encode("utf-16le")).decode("ascii")
    ssh_args = deps.ssh_base_args(node) + [
        "powershell",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy", "Bypass",
        "-EncodedCommand", encoded,
    ]
    ssh_proc = None
    try:
        ssh_proc = deps.popen(ssh_args, stdout=deps.pipe, stderr=deps.pipe)
        tar_proc = deps.subprocess_run(
            ["tar", "-xf", "-", "-C", dst],
            stdin=ssh_proc.stdout,
            capture_output=True,
            text=True,
            timeout=deps.result_sync_timeout_s,
        )
        if ssh_proc.stdout:
            ssh_proc.stdout.close()
        _, ssh_err = ssh_proc.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        if ssh_proc:
            try:
                ssh_proc.kill()
            except Exception:
                pass
        return (False, f"Windows result tar sync timeout (>{deps.result_sync_timeout_s}s)")
    except Exception as exc:
        if ssh_proc:
            try:
                ssh_proc.kill()
            except Exception:
                pass
        return (False, f"Windows result tar sync exception: {str(exc)[:200]}")

    ssh_err_s = _stream_text(ssh_err).strip()
    if ssh_proc.returncode != 0:
        return (False, f"Windows result tar ssh rc={ssh_proc.returncode}: {ssh_err_s[:200]}")
    if tar_proc.returncode != 0:
        tar_err_s = _stream_text(getattr(tar_proc, "stderr", "")).strip()
        return (False, f"local result tar extract rc={tar_proc.returncode}: {tar_err_s[:200]}")
    return (True, "ok")


def sync_one_result(candidate: dict, *, deps: ResultSyncExecutorDeps) -> tuple[bool, str]:
    """Pull remote result_dir to local_result_dir without touching scheduler state."""
    node = candidate.get("node") or ""
    if candidate.get("result_dirs"):
        return _sync_result_dirs(candidate, deps)

    if deps.node_is_windows(node):
        return deps.sync_windows_result(candidate)

    remote_dir = deps.remote_path_for_node(node, candidate["result_dir"].rstrip("/"))
    dst = candidate["local_result_dir"].rstrip("/") + "/"
    ok, msg = _ensure_local_result_dir(dst)
    if not ok:
        return False, msg

    if deps.relay_node_for_node(node):
        return _sync_via_relay(node, remote_dir, dst, deps)
    return _sync_direct_rsync(node, remote_dir, dst, deps)
