"""Resume checkpoint staging for first launches."""

from __future__ import annotations

import shlex
import hashlib
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Callable

try:
    from scheduler_data_motion.rsync import run_local_rsync, run_remote_rsync_command
except ModuleNotFoundError:
    from ..scheduler_data_motion.rsync import run_local_rsync, run_remote_rsync_command


@dataclass(frozen=True)
class ResumeCkptStagingDeps:
    node_configs: dict
    task_requires_resume_scan: Callable[[dict], bool]
    resume_checkpoint_stage_check: Callable[..., tuple[str, dict | None, str]]
    resume_ckpt_stage_key: Callable[[str, str, str], tuple]
    staging_cache_hit: Callable[[tuple], bool]
    mark_stage_success: Callable[[tuple], None]
    mark_cap_exceeded: Callable[[tuple], None]
    node_is_windows: Callable[[str], bool]
    remote_path_for_node: Callable[[str, str], str]
    run_on: Callable[..., tuple[int, str, str]]
    stage_local_dir_to_windows: Callable[..., tuple[bool, str]]
    windows_path_for_node: Callable[[str, str], str]
    run_windows_ps: Callable[..., tuple[int, str, str]]
    ps_quote: Callable[[str], str]
    relay_node_for_node: Callable[[str], str | None]
    relay_path_for_node: Callable[[str, str], str]
    rsync_path_for_node: Callable[[str, str], str]
    ssh_rsync_shell_for_node: Callable[[str], str]
    relay_ssh_target_for_node: Callable[[str], str]
    rsync_shell_for_pair: Callable[[str, str], str | None]
    run_subprocess: Callable[..., Any] = subprocess.run
    relay_spool_root: str = os.path.join(
        tempfile.gettempdir(), "scheduleurm-ckpt-relay")
    relay_spool_stale_s: int = 24 * 60 * 60
    now: Callable[[], float] = time.time


def _run_rsync(args: list[str], *, label: str, deps: ResumeCkptStagingDeps) -> tuple[bool, str]:
    if label == "failed":
        return run_local_rsync(
            args,
            operation="rsync ckpt",
            rc_operation="rsync ckpt failed",
            timeout_operation="rsync ckpt",
            exception_operation="rsync ckpt",
            run_subprocess=deps.run_subprocess,
            timeout_s=600,
            timeout_human=">600s",
        )
    return run_local_rsync(
        args,
        operation=f"rsync ckpt {label}",
        run_subprocess=deps.run_subprocess,
        timeout_s=600,
        timeout_human=">600s",
    )


def _probe_source_ckpt(
    source_node: str,
    source_ckpt_dir: str,
    ckpt_dir: str,
    *,
    deps: ResumeCkptStagingDeps,
) -> tuple[bool, int, str]:
    try:
        rc_test, _, _ = deps.run_on(
            source_node,
            f"test -d {shlex.quote(source_ckpt_dir)}",
            timeout=5,
            check=False,
        )
    except Exception as exc:
        return False, -1, f"ckpt_dir reachability check on {source_node} failed: {str(exc)[:120]}"
    if rc_test != 0:
        return False, -1, f"ckpt_dir {source_ckpt_dir} missing on resume source {source_node}"

    size_mb = -1
    try:
        rc, out, _ = deps.run_on(
            source_node,
            f"du -sm {shlex.quote(source_ckpt_dir)} 2>/dev/null | awk '{{print $1}}'",
            timeout=15,
            check=False,
        )
        if rc == 0 and out.strip().isdigit():
            size_mb = int(out.strip())
    except Exception:
        pass
    if size_mb < 0:
        return False, -1, f"ckpt_dir {ckpt_dir} exists on {source_node} but size probe failed"
    return True, size_mb, ""


def _verify_target_ckpt(
    task: dict,
    target_node: str,
    target_ckpt_dir: str,
    source_loc: dict,
    *,
    context: str,
    deps: ResumeCkptStagingDeps,
) -> tuple[bool, str]:
    resume_path = source_loc.get("path")
    try:
        if resume_path:
            target_resume = deps.remote_path_for_node(target_node, resume_path)
            rc, _, _ = deps.run_on(
                target_node,
                f"test -e {shlex.quote(target_resume)}",
                timeout=5,
                check=False,
            )
        else:
            rc, _, _ = deps.run_on(
                target_node,
                f"ls -1 {shlex.quote(target_ckpt_dir)} 2>/dev/null | head -1",
                timeout=5,
                check=False,
            )
        if rc != 0:
            return False, f"ckpt_dir {target_ckpt_dir} not verified on {target_node} after {context}"
    except Exception as exc:
        return False, f"post-{context} ckpt check failed: {str(exc)[:120]}"
    return True, ""


def _ensure_target_dir(
    target_node: str,
    target_dir: str,
    *,
    context: str,
    deps: ResumeCkptStagingDeps,
) -> tuple[bool, str]:
    try:
        rc, out, err = deps.run_on(
            target_node,
            f"mkdir -p {shlex.quote(target_dir)}",
            timeout=10,
            check=False,
        )
    except Exception as exc:
        return False, f"{context} mkdir failed: {str(exc)[:160]}"
    if rc != 0:
        detail = str(err or out or "").strip()[:200]
        return False, f"{context} mkdir failed rc={rc}: {detail}"
    return True, ""


def _stage_to_windows(
    task: dict,
    target_node: str,
    source_node: str,
    source_loc: dict,
    ckpt_dir: str,
    src_host: str | None,
    key: tuple,
    size_mb: int,
    *,
    deps: ResumeCkptStagingDeps,
) -> tuple[bool, str]:
    if source_node != "local" or src_host:
        return False, (
            f"ckpt_dir {ckpt_dir} needs Windows staging, but only "
            f"local→Windows is supported (source={source_node})"
        )
    ok, msg = deps.stage_local_dir_to_windows(
        ckpt_dir,
        target_node,
        ckpt_dir,
        timeout_s=600,
    )
    if not ok:
        return False, msg
    resume_path = source_loc.get("path")
    try:
        if resume_path:
            win_resume = deps.windows_path_for_node(target_node, resume_path)
            rc, out, _ = deps.run_windows_ps(
                target_node,
                f"if (Test-Path -LiteralPath {deps.ps_quote(win_resume)}) {{ 'OK' }} else {{ 'MISSING' }}",
                timeout=10,
                check=False,
            )
        else:
            win_dir = deps.windows_path_for_node(target_node, ckpt_dir)
            rc, out, _ = deps.run_windows_ps(
                target_node,
                f"$x=Get-ChildItem -LiteralPath {deps.ps_quote(win_dir)} -File -ErrorAction SilentlyContinue | Select-Object -First 1; if ($x) {{ 'OK' }} else {{ 'MISSING' }}",
                timeout=10,
                check=False,
            )
        if rc != 0 or "OK" not in out:
            return False, f"ckpt_dir {ckpt_dir} not verified on {target_node} after Windows staging"
    except Exception as exc:
        return False, f"post-Windows ckpt check failed: {str(exc)[:120]}"
    deps.mark_stage_success(key)
    return True, f"synced ckpt_dir to Windows ({size_mb}MB): {msg}"


def _stage_via_relay(
    source_node: str,
    target_node: str,
    source_ckpt_dir: str,
    target_ckpt_dir: str,
    source_loc: dict,
    key: tuple,
    size_mb: int,
    *,
    deps: ResumeCkptStagingDeps,
) -> tuple[bool, str]:
    relay_node = deps.relay_node_for_node(target_node)
    if not relay_node:
        return False, "no relay node"
    relay_ckpt_dir = deps.relay_path_for_node(target_node, target_ckpt_dir)
    ok, msg = _ensure_target_dir(
        relay_node,
        relay_ckpt_dir,
        context="relay checkpoint target",
        deps=deps,
    )
    if not ok:
        return False, msg
    rsync_args = ["rsync", "-az", "--partial"]
    relay_shell = deps.ssh_rsync_shell_for_node(relay_node)
    if relay_shell:
        rsync_args.extend(["-e", relay_shell])
    rsync_args.extend([
        source_ckpt_dir.rstrip("/") + "/",
        deps.rsync_path_for_node(relay_node, relay_ckpt_dir.rstrip("/") + "/"),
    ])
    ok, msg = _run_rsync(rsync_args, label="local→relay", deps=deps)
    if not ok:
        return False, msg

    ok, msg = _ensure_target_dir(
        target_node,
        target_ckpt_dir,
        context="checkpoint target",
        deps=deps,
    )
    if not ok:
        return False, msg
    target = deps.relay_ssh_target_for_node(target_node)
    relay_cmd = " ".join(shlex.quote(arg) for arg in [
        "rsync",
        "-az",
        "--partial",
        relay_ckpt_dir.rstrip("/") + "/",
        f"{target}:{target_ckpt_dir.rstrip('/')}/",
    ])
    ok, msg = run_remote_rsync_command(
        node=relay_node,
        command=relay_cmd,
        run_on=deps.run_on,
        operation="rsync ckpt relay→target",
        timeout_s=600,
    )
    if not ok:
        return False, msg

    ok, msg = _verify_target_ckpt(
        {},
        target_node,
        target_ckpt_dir,
        source_loc,
        context="relay rsync",
        deps=deps,
    )
    if not ok:
        return False, msg
    deps.mark_stage_success(key)
    return True, f"synced ckpt_dir via {relay_node} ({size_mb}MB)"


def _stage_direct(
    source_node: str,
    target_node: str,
    source_ckpt_dir: str,
    target_ckpt_dir: str,
    source_loc: dict,
    key: tuple,
    size_mb: int,
    *,
    deps: ResumeCkptStagingDeps,
) -> tuple[bool, str]:
    ok, msg = _ensure_target_dir(
        target_node,
        target_ckpt_dir,
        context="checkpoint target",
        deps=deps,
    )
    if not ok:
        return False, msg
    rsync_args = ["rsync", "-az", "--partial"]
    rsync_shell = deps.rsync_shell_for_pair(source_node, target_node)
    if rsync_shell:
        rsync_args.extend(["-e", rsync_shell])
    rsync_args.extend([
        deps.rsync_path_for_node(source_node, source_ckpt_dir.rstrip("/") + "/"),
        deps.rsync_path_for_node(target_node, target_ckpt_dir.rstrip("/") + "/"),
    ])
    ok, msg = _run_rsync(rsync_args, label="failed", deps=deps)
    if not ok:
        return False, msg.replace("rsync ckpt failed rc=", "rsync ckpt failed rc=")

    ok, msg = _verify_target_ckpt(
        {},
        target_node,
        target_ckpt_dir,
        source_loc,
        context="rsync",
        deps=deps,
    )
    if not ok:
        return False, msg
    deps.mark_stage_success(key)
    return True, f"synced ckpt_dir ({size_mb}MB)"


def _stage_remote_to_remote_via_local(
    source_node: str,
    target_node: str,
    source_ckpt_dir: str,
    target_ckpt_dir: str,
    source_loc: dict,
    key: tuple,
    size_mb: int,
    *,
    deps: ResumeCkptStagingDeps,
) -> tuple[bool, str]:
    """Relay a remote checkpoint through a resumable control-host spool."""
    ok, msg = _ensure_target_dir(
        target_node,
        target_ckpt_dir,
        context="checkpoint target",
        deps=deps,
    )
    if not ok:
        return False, msg

    spool_root = os.path.abspath(deps.relay_spool_root)
    os.makedirs(spool_root, mode=0o700, exist_ok=True)
    try:
        os.chmod(spool_root, 0o700)
    except OSError:
        pass
    digest = hashlib.sha256(repr(key).encode("utf-8")).hexdigest()
    spool_dir = os.path.join(spool_root, digest)
    now = deps.now()
    try:
        for entry in os.scandir(spool_root):
            if not entry.is_dir(follow_symlinks=False) or entry.path == spool_dir:
                continue
            if now - entry.stat(follow_symlinks=False).st_mtime > deps.relay_spool_stale_s:
                shutil.rmtree(entry.path, ignore_errors=True)
    except OSError:
        pass
    os.makedirs(spool_dir, mode=0o700, exist_ok=True)

    pull_args = ["rsync", "-az", "--partial"]
    source_shell = deps.ssh_rsync_shell_for_node(source_node)
    if source_shell:
        pull_args.extend(["-e", source_shell])
    pull_args.extend([
        deps.rsync_path_for_node(
            source_node, source_ckpt_dir.rstrip("/") + "/"),
        spool_dir.rstrip("/") + "/",
    ])
    ok, msg = _run_rsync(
        pull_args, label="remote-to-local", deps=deps)
    if not ok:
        return False, msg

    push_args = ["rsync", "-az", "--partial"]
    target_shell = deps.ssh_rsync_shell_for_node(target_node)
    if target_shell:
        push_args.extend(["-e", target_shell])
    push_args.extend([
        spool_dir.rstrip("/") + "/",
        deps.rsync_path_for_node(
            target_node, target_ckpt_dir.rstrip("/") + "/"),
    ])
    ok, msg = _run_rsync(
        push_args, label="local-to-remote", deps=deps)
    if not ok:
        return False, msg

    ok, msg = _verify_target_ckpt(
        {},
        target_node,
        target_ckpt_dir,
        source_loc,
        context="remote relay rsync",
        deps=deps,
    )
    if not ok:
        return False, msg
    deps.mark_stage_success(key)
    shutil.rmtree(spool_dir, ignore_errors=True)
    return True, f"synced ckpt_dir remote-to-remote via local ({size_mb}MB)"


def stage_resume_ckpt_for_launch(
    task: dict,
    target_node: str,
    *,
    source_loc: dict | None = None,
    max_ckpt_mb: int,
    deps: ResumeCkptStagingDeps,
) -> tuple[bool, str]:
    """Stage a resume checkpoint directory to a first-launch target."""
    if not deps.task_requires_resume_scan(task):
        return True, "not a resume-scanned task"
    ckpt_dir = task.get("ckpt_dir")
    if not ckpt_dir:
        return True, "no ckpt_dir"
    if not target_node:
        return False, "no target node"

    target_host = deps.node_configs.get(target_node, {}).get("host")
    if source_loc is None:
        state, source_loc, msg = deps.resume_checkpoint_stage_check(task, target_node)
        if state == "ready":
            return True, msg
        if not source_loc:
            return False, msg

    source_node = source_loc.get("node")
    if not source_node:
        return False, "resume source has no node"
    if source_node == target_node:
        return True, "source==target; checkpoint already local"
    src_host = deps.node_configs.get(source_node, {}).get("host")
    if deps.node_is_windows(source_node):
        return False, (
            f"ckpt_dir {ckpt_dir} staging from Windows source {source_node} "
            "is not supported yet; keep resume task on that Windows node or "
            "copy the checkpoint to local first"
        )
    key = deps.resume_ckpt_stage_key(
        source_node, target_node, ckpt_dir, source_loc)
    if deps.staging_cache_hit(key):
        return True, "cache hit (checkpoint already staged)"

    source_ckpt_dir = deps.remote_path_for_node(source_node, ckpt_dir)
    target_ckpt_dir = deps.remote_path_for_node(target_node, ckpt_dir)
    ok, size_mb, msg = _probe_source_ckpt(
        source_node,
        source_ckpt_dir,
        ckpt_dir,
        deps=deps,
    )
    if not ok:
        return False, msg
    if size_mb > max_ckpt_mb:
        deps.mark_cap_exceeded(key)
        return False, (
            f"CAP_EXCEEDED: ckpt_dir {ckpt_dir} is {size_mb}MB > "
            f"{max_ckpt_mb}MB cap; keep task on checkpoint-local node"
        )

    if deps.node_is_windows(target_node):
        return _stage_to_windows(
            task,
            target_node,
            source_node,
            source_loc,
            ckpt_dir,
            src_host,
            key,
            size_mb,
            deps=deps,
        )

    if src_host and target_host:
        return _stage_remote_to_remote_via_local(
            source_node,
            target_node,
            source_ckpt_dir,
            target_ckpt_dir,
            source_loc,
            key,
            size_mb,
            deps=deps,
        )

    relay_node = deps.relay_node_for_node(target_node)
    if relay_node and not src_host:
        return _stage_via_relay(
            source_node,
            target_node,
            source_ckpt_dir,
            target_ckpt_dir,
            source_loc,
            key,
            size_mb,
            deps=deps,
        )

    return _stage_direct(
        source_node,
        target_node,
        source_ckpt_dir,
        target_ckpt_dir,
        source_loc,
        key,
        size_mb,
        deps=deps,
    )
