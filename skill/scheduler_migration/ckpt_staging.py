"""Checkpoint staging for ETA-based task migration."""

from __future__ import annotations

import shlex

from .types import MigrationStagingDeps, node_host, run_migration_rsync


def source_ckpt_presence(
    source_node: str,
    ckpt_dir: str,
    *,
    deps: MigrationStagingDeps,
) -> tuple[bool, str | None, str]:
    source_ckpt_dir = deps.remote_path_for_node(source_node, ckpt_dir)
    try:
        rc_test, _, _ = deps.run_on(
            source_node,
            f"test -d {shlex.quote(source_ckpt_dir)}",
            timeout=5,
            check=False,
        )
        return rc_test == 0, source_ckpt_dir, ""
    except Exception as exc:
        msg = (
            f"ckpt_dir {ckpt_dir} reachability check on {source_node} "
            f"failed: {str(exc)[:120]}; fail-closed (would otherwise "
            f"migrate without rsyncing the ckpt → silent step-0 restart)"
        )
        return False, None, msg


def probe_ckpt_size(
    source_node: str,
    source_ckpt_dir: str,
    ckpt_dir: str,
    *,
    deps: MigrationStagingDeps,
) -> tuple[bool, int, str]:
    size_mb = -1
    try:
        rc, out, _ = deps.run_on(
            source_node,
            f"du -sm {shlex.quote(source_ckpt_dir)} 2>/dev/null | "
            f"awk '{{print $1}}'",
            timeout=15,
            check=False,
        )
        if rc == 0 and out.strip().isdigit():
            size_mb = int(out.strip())
    except Exception:
        pass
    if size_mb < 0:
        return (
            False,
            -1,
            f"ckpt_dir {ckpt_dir} exists on {source_node} but size "
            f"probe failed; fail-closed (would otherwise migrate "
            f"without rsyncing the ckpt → silent step-0 restart)",
        )
    return True, size_mb, ""


def stage_ckpt(
    source_node: str,
    target_node: str,
    ckpt_dir: str,
    size_mb: int,
    *,
    max_ckpt_mb: int,
    deps: MigrationStagingDeps,
) -> tuple[bool, str, bool]:
    if size_mb > max_ckpt_mb:
        return (
            False,
            f"ckpt_dir {ckpt_dir} is {size_mb}MB > max {max_ckpt_mb}MB; "
            f"migration aborted (rsync would take too long)",
            True,
        )

    ckpt_key = (source_node, target_node, ckpt_dir)
    if size_mb <= 0 or deps.staging_cache_hit(ckpt_key):
        return True, "", True

    src_host = node_host(deps, source_node)
    tgt_host = node_host(deps, target_node)
    if src_host and tgt_host:
        return (
            False,
            f"ckpt_dir {ckpt_dir} ({size_mb}MB) needs rsync but "
            f"remote→remote not supported (src={src_host}, tgt={tgt_host}); "
            f"task would lose its checkpoint on migration. User must "
            f"sync ckpt manually OR via shared NFS",
            True,
        )

    source_ckpt_dir = deps.remote_path_for_node(source_node, ckpt_dir)
    target_ckpt_dir = deps.remote_path_for_node(target_node, ckpt_dir)
    src_path = deps.rsync_path_for_node(source_node, source_ckpt_dir.rstrip("/") + "/")
    dst_path = deps.rsync_path_for_node(target_node, target_ckpt_dir.rstrip("/") + "/")
    try:
        deps.run_on(target_node, f"mkdir -p {shlex.quote(target_ckpt_dir)}", timeout=10, check=False)
    except Exception:
        pass

    rsync_args = ["rsync", "-az", "--partial"]
    rsync_shell = deps.rsync_shell_for_pair(source_node, target_node)
    if rsync_shell:
        rsync_args.extend(["-e", rsync_shell])
    rsync_args.extend([src_path, dst_path])
    ok, msg = run_migration_rsync(rsync_args, kind="ckpt", deps=deps)
    if not ok:
        return False, msg, True

    try:
        rc2, out2, _ = deps.run_on(
            target_node,
            f"ls -1 {shlex.quote(target_ckpt_dir)} 2>/dev/null | head -1",
            timeout=5,
            check=False,
        )
        if rc2 != 0 or not (out2 or "").strip():
            return (
                False,
                f"ckpt_dir {target_ckpt_dir} appears empty on {target_node} "
                f"after rsync (rc={rc2}); migration aborted",
                True,
            )
    except Exception as exc:
        return False, f"post-rsync ckpt check failed: {exc}", True

    deps.mark_staging_success(ckpt_key)
    return True, "", True


def stage_checkpoint_if_needed(
    task: dict,
    source_node: str,
    target_node: str,
    *,
    max_ckpt_mb: int,
    deps: MigrationStagingDeps,
) -> tuple[bool, str, bool]:
    ckpt_dir = task.get("ckpt_dir")
    if not ckpt_dir:
        return True, "", False

    src_present = False
    source_ckpt_dir = None
    if source_node:
        src_present, source_ckpt_dir, error_msg = source_ckpt_presence(
            source_node,
            ckpt_dir,
            deps=deps,
        )
        if error_msg:
            return False, error_msg, True

    if not src_present:
        return True, "", False

    ok, size_mb, msg = probe_ckpt_size(
        source_node,
        source_ckpt_dir,
        ckpt_dir,
        deps=deps,
    )
    if not ok:
        return False, msg, True
    return stage_ckpt(
        source_node,
        target_node,
        ckpt_dir,
        size_mb,
        max_ckpt_mb=max_ckpt_mb,
        deps=deps,
    )
