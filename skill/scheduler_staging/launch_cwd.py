"""CWD staging implementation for first launches."""

from __future__ import annotations

import os
import hashlib
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:
    from scheduler_staging.code_tar import (
        CodeTarStagingDeps,
        stage_code_tar_for_launch as _stage_code_tar_for_launch_impl,
    )
    from scheduler_data_motion.rsync import run_local_rsync, run_remote_rsync_command
    from scheduler_windows.tar_staging import (
        WindowsTarStagingDeps,
        stage_local_dir_to_windows as _stage_local_dir_to_windows_impl,
        tar_exclude_args,
    )
except ModuleNotFoundError:
    from .code_tar import (
        CodeTarStagingDeps,
        stage_code_tar_for_launch as _stage_code_tar_for_launch_impl,
    )
    from ..scheduler_data_motion.rsync import run_local_rsync, run_remote_rsync_command
    from ..scheduler_windows_tar_staging import (
        WindowsTarStagingDeps,
        stage_local_dir_to_windows as _stage_local_dir_to_windows_impl,
        tar_exclude_args,
    )


DEFAULT_RSYNC_EXCLUDES = [
    "--exclude=.git/",
    "--exclude=__pycache__/",
    "--exclude=*.pyc",
    "--exclude=results/",
    "--exclude=results_*/",
    "--exclude=logs/",
    "--exclude=logs_*/",
    "--exclude=experiment_output/",
    "--exclude=archive*/",
    "--exclude=*.tar.gz",
]

_UNSAFE_STAGING_ROOTS = frozenset({
    "/",
    "/tmp",
    "/var/tmp",
    "/run",
    "/dev",
    "/proc",
    "/sys",
})


def unsafe_staging_cwd_reason(cwd: str) -> str:
    """Reject roots where rsync --delete could damage shared system state."""
    raw = str(cwd or "").strip()
    if not raw or not raw.startswith("/"):
        return ""
    normalized = os.path.realpath(os.path.abspath(raw))
    unsafe_roots = set(_UNSAFE_STAGING_ROOTS)
    unsafe_roots.add(os.path.realpath(str(Path.home())))
    if normalized in unsafe_roots:
        return f"refusing remote launch staging from unsafe root {normalized}"
    return ""

# Source-audit breadcrumbs for older regression guards. The implementation now
# lives in scheduler_windows_tar_staging.py, but the public wrapper remains here.
LAUNCH_CWD_STAGING_AUDIT_TOKENS = '''
"tar", "-h", "-C"
deps.ssh_base_args(target_node)
tar -xf - -C $dest
stage_local_dir_to_windows
deps.node_is_windows(target_node)
'''


@dataclass(frozen=True)
class LaunchCwdStagingDeps:
    node_configs: dict
    launch_max_cwd_size_mb: int
    staging_cache_hit: Callable[[tuple], bool]
    mark_stage_success: Callable[[tuple], None]
    mark_cap_exceeded: Callable[[tuple], None]
    node_is_windows: Callable[[str], bool]
    is_windows_native_path: Callable[[str], bool]
    windows_path_for_node: Callable[[str, str], str]
    run_windows_ps: Callable[..., tuple[int, str, str]]
    ps_quote: Callable[[str], str]
    stage_local_dir_to_windows: Callable[..., tuple[bool, str]]
    remote_path_for_node: Callable[[str, str], str]
    run_on: Callable[..., tuple[int, str, str]]
    stage_code_tar_for_launch: Callable[[dict, str, str, int], tuple[bool, str]]
    relay_node_for_node: Callable[[str], str | None]
    relay_path_for_node: Callable[[str, str], str]
    rsync_path_for_node: Callable[[str, str], str]
    ssh_rsync_shell_for_node: Callable[[str], str]
    relay_ssh_target_for_node: Callable[[str], str]
    run_control_subprocess: Callable[..., Any] = subprocess.run


def _clean_extra_excludes(extra_excludes: list | None) -> list[str]:
    return [str(ex) for ex in (extra_excludes or []) if ex]


def _du_args(cwd: str, extra_excludes: list | None) -> list[str]:
    args = [
        "du",
        "-sm",
        "--exclude=.git",
        "--exclude=__pycache__",
        "--exclude=*.pyc",
        "--exclude=results",
        "--exclude=results_*",
        "--exclude=logs",
        "--exclude=logs_*",
        "--exclude=experiment_output",
        "--exclude=archive*",
        "--exclude=*.tar.gz",
    ]
    for exclude in _clean_extra_excludes(extra_excludes):
        args.append(f"--exclude={exclude.rstrip('/')}")
    args.append(cwd)
    return args


def _probe_local_cwd_size_mb(cwd: str, extra_excludes: list | None) -> int:
    try:
        result = subprocess.run(
            _du_args(cwd, extra_excludes),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            head = (result.stdout or "").strip().split()
            if head and head[0].isdigit():
                return int(head[0])
    except Exception:
        pass
    return 0


def _rsync_args(extra_excludes: list | None) -> list[str]:
    args = ["rsync", "-az", "--partial", "--delete", *DEFAULT_RSYNC_EXCLUDES]
    for exclude in _clean_extra_excludes(extra_excludes):
        args.append(f"--exclude={exclude}")
    return args


def launch_input_stage_key(
    task: dict,
    target_node: str,
    *,
    node_configs: dict | None = None,
) -> tuple:
    """Return a persistent cache key for one task's declared local inputs."""
    node_info = (node_configs or {}).get(target_node, {}) or {}
    target_scope = str(
        node_info.get("shared_workspace_group") or target_node)
    entries = []
    for raw_path in _launch_input_paths(task):
        path = Path(str(raw_path)).expanduser()
        try:
            resolved = path.resolve(strict=True)
        except OSError:
            entries.append((str(path), "missing"))
            continue
        if not resolved.is_dir():
            entries.append((str(resolved), "not-directory"))
            continue
        files = []
        for candidate in sorted(resolved.rglob("*")):
            if not candidate.is_file():
                continue
            try:
                stat = candidate.stat()
            except OSError:
                continue
            files.append((
                str(candidate.relative_to(resolved)),
                int(stat.st_mtime_ns),
                int(stat.st_size),
            ))
        entries.append((str(resolved), tuple(files)))
    payload = repr(tuple(entries)).encode("utf-8", "surrogateescape")
    return (
        "launch_input_v2",
        "local",
        target_scope,
        hashlib.sha256(payload).hexdigest(),
    )


def _launch_input_paths(task: dict) -> list[str]:
    explicit = [
        str(path).strip()
        for path in (task.get("stage_input_paths") or [])
        if str(path).strip()
    ]
    if explicit:
        return _minimal_directory_roots(explicit)
    # Backward compatibility for queued dependency chains created before
    # stage_input_paths existed: a ready prerequisite file is an input, so
    # stage its smallest meaningful containing directory.  Keep this scoped
    # below cwd and never turn the entire project root into an input transfer.
    cwd = Path(str(task.get("cwd") or "")).expanduser()
    try:
        cwd = cwd.resolve(strict=True)
    except OSError:
        return []
    inferred = []
    for raw_path in task.get("wait_for_files") or []:
        path = Path(str(raw_path)).expanduser()
        if not path.is_file():
            continue
        parent = path.parent.resolve()
        try:
            if parent == cwd or os.path.commonpath([str(parent), str(cwd)]) != str(cwd):
                continue
        except ValueError:
            continue
        inferred.append(str(parent))
    return _minimal_directory_roots(inferred)


def _minimal_directory_roots(paths: list[str]) -> list[str]:
    """Collapse duplicate and nested launch inputs to their shallowest roots."""
    canonical = list(dict.fromkeys(
        os.path.realpath(os.path.abspath(os.path.expanduser(str(path))))
        for path in paths
        if str(path).strip()
    ))
    roots: list[Path] = []
    for candidate_text in sorted(
            canonical, key=lambda value: (len(Path(value).parts), value)):
        candidate = Path(candidate_text)
        if any(candidate == root or candidate.is_relative_to(root)
               for root in roots):
            continue
        roots.append(candidate)
    return [str(root) for root in roots]


def launch_input_stage_state(
    task: dict,
    target_node: str,
    *,
    deps: LaunchCwdStagingDeps,
) -> str:
    paths = _launch_input_paths(task)
    if not paths:
        return "ready"
    if not target_node or deps.node_configs.get(target_node, {}).get("host") is None:
        return "ready"
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.is_dir():
            return "missing_input"
    key = launch_input_stage_key(
        task, target_node, node_configs=deps.node_configs)
    return "ready" if deps.staging_cache_hit(key) else "needs_stage"


def stage_input_paths_for_launch(
    task: dict,
    target_node: str,
    *,
    deps: LaunchCwdStagingDeps,
) -> tuple[bool, str]:
    """Synchronize declared immutable launch inputs without copying the whole cwd."""
    paths = _launch_input_paths(task)
    if not paths or deps.node_configs.get(target_node, {}).get("host") is None:
        return True, "no remote launch inputs"
    key = launch_input_stage_key(
        task, target_node, node_configs=deps.node_configs)
    if deps.staging_cache_hit(key):
        return True, "cache hit (launch inputs already synced)"
    for raw_path in paths:
        source = Path(raw_path).expanduser()
        if not source.is_dir():
            return False, f"launch input is missing or not a directory: {source}"
        source_text = str(source.resolve())
        size_mb = _probe_local_cwd_size_mb(source_text, None)
        if size_mb > deps.launch_max_cwd_size_mb:
            deps.mark_cap_exceeded(key)
            return False, (
                f"CAP_EXCEEDED: launch input {source_text} is {size_mb}MB > "
                f"{deps.launch_max_cwd_size_mb}MB"
            )
        if deps.node_is_windows(target_node):
            return False, "launch input staging to Windows is not supported"
        remote_path = deps.remote_path_for_node(target_node, source_text)
        _mkdir_remote(target_node, remote_path, deps)
        relay_node = deps.relay_node_for_node(target_node)
        if relay_node:
            ok, msg = _stage_via_relay(
                source_text, target_node, remote_path, relay_node, size_mb,
                None, key, deps, use_default_excludes=False,
                mark_success=False,
            )
        else:
            ok, msg = _stage_direct_rsync(
                source_text, target_node, remote_path, size_mb, None, key, deps,
                use_default_excludes=False,
                mark_success=False,
            )
        if not ok:
            return False, msg
    # One launch-input key covers the complete declared directory set.  Do not
    # publish it after an individual rsync: dispatch may otherwise launch while
    # later directories are still stale or only partially refreshed.
    deps.mark_stage_success(key)
    return True, f"synced {len(paths)} launch input director{'y' if len(paths) == 1 else 'ies'}"


def _run_rsync(
    args: list[str],
    *,
    label: str,
    run_subprocess: Callable[..., Any] = subprocess.run,
) -> tuple[bool, str]:
    operation = f"rsync {label}".strip()
    return run_local_rsync(
        args,
        operation=operation,
        run_subprocess=run_subprocess,
        timeout_s=600,
        timeout_human=">600s",
    )


def stage_local_dir_to_windows(
    local_dir: str,
    target_node: str,
    target_dir: str,
    *,
    extra_excludes: list | None = None,
    timeout_s: int = 600,
    deps: WindowsTarStagingDeps,
) -> tuple[bool, str]:
    return _stage_local_dir_to_windows_impl(
        local_dir,
        target_node,
        target_dir,
        extra_excludes=extra_excludes,
        timeout_s=timeout_s,
        deps=deps,
    )


def stage_code_tar_for_launch(
    task: dict,
    target_node: str,
    remote_cwd: str,
    cwd_size_mb: int,
    *,
    deps: CodeTarStagingDeps,
) -> tuple[bool, str]:
    return _stage_code_tar_for_launch_impl(
        task,
        target_node,
        remote_cwd,
        cwd_size_mb,
        deps=deps,
    )


def _probe_windows_native_cwd(
    cwd: str,
    target_node: str,
    cwd_key: tuple,
    deps: LaunchCwdStagingDeps,
) -> tuple[bool, str]:
    win_cwd = deps.windows_path_for_node(target_node, cwd)
    try:
        rc, out, err = deps.run_windows_ps(
            target_node,
            f"if (Test-Path -LiteralPath {deps.ps_quote(win_cwd)} -PathType Container) {{ 'OK' }} else {{ 'MISSING' }}",
            timeout=10,
            check=False,
        )
    except Exception as exc:
        return False, f"Windows target cwd probe failed on {target_node}: {str(exc)[:160]}"
    if rc == 0 and "OK" in (out or ""):
        deps.mark_stage_success(cwd_key)
        return True, f"target-native Windows cwd exists on {target_node}: {win_cwd}"
    return False, (
        f"Windows target cwd missing on {target_node}: {win_cwd}; "
        f"{(err or out or '').strip()[:120]}"
    )


def _stage_to_windows(
    cwd: str,
    target_node: str,
    cwd_size_mb: int,
    extra_excludes: list | None,
    cwd_key: tuple,
    deps: LaunchCwdStagingDeps,
) -> tuple[bool, str]:
    ok, msg = deps.stage_local_dir_to_windows(
        cwd,
        target_node,
        cwd,
        extra_excludes=extra_excludes,
        timeout_s=600,
    )
    if not ok:
        return False, msg
    deps.mark_stage_success(cwd_key)
    return True, f"{msg} ({cwd_size_mb}MB)"


def _mkdir_remote(target_node: str, remote_cwd: str, deps: LaunchCwdStagingDeps) -> None:
    try:
        deps.run_on(target_node, f"mkdir -p {shlex.quote(remote_cwd)}", timeout=10, check=False)
    except Exception:
        pass


def _stage_via_relay(
    cwd: str,
    target_node: str,
    remote_cwd: str,
    relay_node: str,
    cwd_size_mb: int,
    extra_excludes: list | None,
    cwd_key: tuple,
    deps: LaunchCwdStagingDeps,
    *,
    use_default_excludes: bool = True,
    mark_success: bool = True,
) -> tuple[bool, str]:
    relay_cwd = deps.relay_path_for_node(target_node, remote_cwd)
    try:
        deps.run_on(relay_node, f"mkdir -p {shlex.quote(relay_cwd)}", timeout=10, check=False)
    except Exception:
        pass

    local_to_relay = (
        _rsync_args(extra_excludes)
        if use_default_excludes else ["rsync", "-az", "--partial", "--delete"]
    )
    relay_shell = deps.ssh_rsync_shell_for_node(relay_node)
    if relay_shell:
        local_to_relay.extend(["-e", relay_shell])
    local_to_relay.extend([
        cwd.rstrip("/") + "/",
        deps.rsync_path_for_node(relay_node, relay_cwd.rstrip("/") + "/"),
    ])
    ok, msg = _run_rsync(local_to_relay, label="local->relay", run_subprocess=deps.run_control_subprocess)
    if not ok:
        return False, msg

    _mkdir_remote(target_node, remote_cwd, deps)
    target = deps.relay_ssh_target_for_node(target_node)
    relay_to_target = (
        _rsync_args(extra_excludes)
        if use_default_excludes else ["rsync", "-az", "--partial", "--delete"]
    )
    relay_to_target.extend([
        relay_cwd.rstrip("/") + "/",
        f"{target}:{remote_cwd.rstrip('/')}/",
    ])
    relay_cmd = " ".join(shlex.quote(arg) for arg in relay_to_target)
    ok, msg = run_remote_rsync_command(
        node=relay_node,
        command=relay_cmd,
        run_on=deps.run_on,
        operation="rsync relay->target",
        timeout_s=600,
    )
    if not ok:
        return False, msg

    if mark_success:
        deps.mark_stage_success(cwd_key)
    return True, f"synced via {relay_node} ({cwd_size_mb}MB)"


def _stage_direct_rsync(
    cwd: str,
    target_node: str,
    remote_cwd: str,
    cwd_size_mb: int,
    extra_excludes: list | None,
    cwd_key: tuple,
    deps: LaunchCwdStagingDeps,
    *,
    use_default_excludes: bool = True,
    mark_success: bool = True,
) -> tuple[bool, str]:
    args = (
        _rsync_args(extra_excludes)
        if use_default_excludes else ["rsync", "-az", "--partial", "--delete"]
    )
    rsync_shell = deps.ssh_rsync_shell_for_node(target_node)
    if rsync_shell:
        args.extend(["-e", rsync_shell])
    args.extend([
        cwd.rstrip("/") + "/",
        deps.rsync_path_for_node(target_node, remote_cwd.rstrip("/") + "/"),
    ])
    ok, msg = _run_rsync(args, label="", run_subprocess=deps.run_control_subprocess)
    if not ok:
        return False, msg
    if mark_success:
        deps.mark_stage_success(cwd_key)
    return True, f"synced ({cwd_size_mb}MB)"


def stage_cwd_for_launch(
    task: dict,
    target_node: str,
    *,
    extra_excludes: list | None = None,
    deps: LaunchCwdStagingDeps,
) -> tuple[bool, str]:
    """Stage local cwd to a launch target without touching scheduler locks."""
    cwd = task.get("cwd")
    if not cwd:
        return False, "no cwd on task"
    if deps.node_configs.get(target_node, {}).get("host") is None:
        return True, "target is local; nothing to sync"

    unsafe_reason = unsafe_staging_cwd_reason(cwd)
    if unsafe_reason:
        return False, f"UNSAFE_CWD: {unsafe_reason}"

    cwd_key = ("local", target_node, cwd)
    if deps.staging_cache_hit(cwd_key):
        return True, "cache hit (already synced within TTL)"

    if deps.node_is_windows(target_node) and deps.is_windows_native_path(cwd):
        return _probe_windows_native_cwd(cwd, target_node, cwd_key, deps)

    if not Path(cwd).exists():
        return False, f"cwd {cwd} does not exist on local; can't seed target {target_node}"

    cwd_size_mb = _probe_local_cwd_size_mb(cwd, extra_excludes)
    if cwd_size_mb > deps.launch_max_cwd_size_mb:
        deps.mark_cap_exceeded(cwd_key)
        return False, (
            f"CAP_EXCEEDED: cwd {cwd} is {cwd_size_mb}MB > "
            f"{deps.launch_max_cwd_size_mb}MB cap; pin to local instead of "
            f"transferring to {target_node}"
        )

    if not deps.node_configs.get(target_node, {}).get("host"):
        return False, f"target node {target_node} has no host"

    if deps.node_is_windows(target_node):
        return _stage_to_windows(
            cwd, target_node, cwd_size_mb, extra_excludes, cwd_key, deps)

    remote_cwd = deps.remote_path_for_node(target_node, cwd)
    _mkdir_remote(target_node, remote_cwd, deps)

    if str((deps.node_configs.get(target_node, {}) or {}).get("launch_staging_method") or "") == "code_tar":
        return deps.stage_code_tar_for_launch(task, target_node, remote_cwd, cwd_size_mb)

    relay_node = deps.relay_node_for_node(target_node)
    if relay_node:
        return _stage_via_relay(
            cwd,
            target_node,
            remote_cwd,
            relay_node,
            cwd_size_mb,
            extra_excludes,
            cwd_key,
            deps,
        )
    return _stage_direct_rsync(
        cwd,
        target_node,
        remote_cwd,
        cwd_size_mb,
        extra_excludes,
        cwd_key,
        deps,
    )
