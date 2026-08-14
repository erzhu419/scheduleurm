"""CWD staging for ETA-based task migration."""

from __future__ import annotations

import shlex

from .types import MigrationStagingDeps, node_host, run_migration_rsync

try:
    from scheduler_staging.launch_cwd import unsafe_staging_cwd_reason
except ImportError:  # pragma: no cover - package import fallback
    from ..scheduler_staging.launch_cwd import unsafe_staging_cwd_reason


def stage_cwd(
    source_node: str,
    target_node: str,
    cwd: str,
    *,
    deps: MigrationStagingDeps,
) -> tuple[bool, str]:
    unsafe_reason = unsafe_staging_cwd_reason(cwd)
    if unsafe_reason:
        return False, f"UNSAFE_CWD: {unsafe_reason}"
    cwd_key = (source_node, target_node, cwd)
    if deps.staging_cache_hit(cwd_key):
        return True, ""

    src_host = node_host(deps, source_node)
    tgt_host = node_host(deps, target_node)
    source_cwd = deps.remote_path_for_node(source_node, cwd)
    target_cwd = deps.remote_path_for_node(target_node, cwd)
    src_path = deps.rsync_path_for_node(source_node, source_cwd.rstrip("/") + "/")
    dst_path = deps.rsync_path_for_node(target_node, target_cwd.rstrip("/") + "/")

    try:
        deps.run_on(target_node, f"mkdir -p {shlex.quote(target_cwd)}", timeout=10, check=False)
    except Exception:
        pass

    if src_host and tgt_host:
        return False, (
            f"cwd rsync remote→remote not yet supported "
            f"(src={src_host}, tgt={tgt_host}); user must sync code "
            f"manually OR via shared NFS"
        )

    cwd_size_mb = 0
    if source_node:
        try:
            rc_du, out_du, _ = deps.run_on(
                source_node,
                f"du -sm --exclude=.git --exclude=__pycache__ "
                f"--exclude='*.pyc' {shlex.quote(source_cwd)} 2>/dev/null | "
                f"awk '{{print $1}}'",
                timeout=15,
                check=False,
            )
            if rc_du == 0 and out_du.strip().isdigit():
                cwd_size_mb = int(out_du.strip())
        except Exception:
            cwd_size_mb = 0
    if cwd_size_mb > deps.migration_max_cwd_size_mb:
        return False, (
            f"cwd {cwd} is {cwd_size_mb}MB > max "
            f"{deps.migration_max_cwd_size_mb}MB; migration aborted "
            f"(rsync would risk hitting the 600s timeout)"
        )

    rsync_args = [
        "rsync",
        "-az",
        "--partial",
        "--exclude=.git",
        "--exclude=__pycache__",
        "--exclude=*.pyc",
    ]
    rsync_shell = deps.rsync_shell_for_pair(source_node, target_node)
    if rsync_shell:
        rsync_args.extend(["-e", rsync_shell])
    rsync_args.extend([src_path, dst_path])
    ok, msg = run_migration_rsync(rsync_args, kind="cwd", deps=deps)
    if not ok:
        return False, msg

    try:
        rc2, _, _ = deps.run_on(
            target_node,
            f"test -d {shlex.quote(target_cwd)}",
            timeout=5,
            check=False,
        )
        if rc2 != 0:
            return False, "cwd still missing after rsync"
    except Exception as exc:
        return False, f"post-rsync ssh check failed: {exc}"
    deps.mark_staging_success(cwd_key)
    return True, ""
