from __future__ import annotations

import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class CodeTarStagingDeps:
    node_configs: dict
    ssh_base_args: Callable[[str], list[str]]
    run_on: Callable[..., tuple[int, str, str]]
    mark_stage_success: Callable[[tuple], Any]
    getpid: Callable[[], int] = os.getpid
    now: Callable[[], float] = time.time
    run_subprocess: Callable[..., Any] = subprocess.run
    tmp_dir: str = "/tmp"


def stage_code_tar_for_launch(
    task: dict,
    target_node: str,
    remote_cwd: str,
    cwd_size_mb: int,
    *,
    deps: CodeTarStagingDeps,
) -> tuple[bool, str]:
    """Stage a compact source-code tarball for nodes with flaky rsync streams."""
    cwd = task.get("cwd")
    if not cwd:
        return False, "no cwd on task"
    info = deps.node_configs.get(target_node, {}) or {}
    include_paths = [
        str(item).strip()
        for item in (info.get("launch_stage_include_paths") or [])
        if str(item).strip()
    ]
    if not include_paths:
        return False, f"{target_node} launch_staging_method=code_tar but no include paths configured"

    existing = []
    for rel in include_paths:
        if rel.startswith("/") or ".." in Path(rel).parts:
            return False, f"unsafe code_tar include path for {target_node}: {rel}"
        if (Path(cwd) / rel).exists():
            existing.append(rel)
    if not existing:
        return False, f"no configured code_tar include paths exist under {cwd}"

    stage_excludes = []
    for item in task.get("stage_excludes") or []:
        rel = str(item).strip().rstrip("/")
        if not rel:
            continue
        if rel.startswith("/") or ".." in Path(rel).parts:
            return False, f"unsafe code_tar exclude path for {target_node}: {rel}"
        stage_excludes.append(rel)

    task_token = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(task.get("id") or "task"),
    ).strip("_") or "task"
    stamp = (
        f"{deps.getpid()}_{int(float(deps.now()) * 1_000_000_000)}_"
        f"{task_token}"
    )
    local_tar = f"{deps.tmp_dir.rstrip('/')}/scheduleurm_{target_node}_code_{stamp}.tar"
    remote_tar = f"{remote_cwd.rstrip('/')}/.scheduleurm_code_stage_{stamp}.tar"
    tar_args = [
        "tar", "-cf", local_tar,
        "--exclude=__pycache__",
        "--exclude=*.pyc",
        "--exclude=.pytest_cache",
    ]
    for rel in stage_excludes:
        tar_args.extend((f"--exclude={rel}", f"--exclude={rel}/**"))
    tar_args += existing
    try:
        result = deps.run_subprocess(tar_args, cwd=cwd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            return False, f"code_tar create rc={result.returncode}: {(result.stderr or '').strip()[:200]}"

        base = deps.ssh_base_args(target_node)
        if not base or base[0] != "ssh" or len(base) < 2:
            return False, f"cannot build scp transport for {target_node}"
        scp_args = ["scp"] + base[1:-1] + [local_tar, f"{base[-1]}:{remote_tar}"]
        result = deps.run_subprocess(scp_args, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return False, f"code_tar scp rc={result.returncode}: {(result.stderr or '').strip()[:200]}"

        # Overlay code only after the full archive has reached the target.
        # Never remove an include root first: those trees can contain active
        # checkpoints/results that are intentionally absent from the tarball.
        cmd = (
            f"cd {shlex.quote(remote_cwd)} && "
            f"tar -xf {shlex.quote(remote_tar)} && "
            f"rm -f {shlex.quote(remote_tar)}"
        )
        rc, out, err = deps.run_on(target_node, cmd, timeout=120, check=False)
        if rc != 0:
            diagnostic = "\n".join(
                line for line in (err or out).splitlines()
                if "setlocale:" not in line
            ).strip()
            return False, f"code_tar unpack rc={rc}: {diagnostic[-400:]}"
    except subprocess.TimeoutExpired:
        return False, "code_tar staging timeout"
    except Exception as exc:
        return False, f"code_tar staging exception: {str(exc)[:200]}"
    finally:
        try:
            Path(local_tar).unlink(missing_ok=True)
        except Exception:
            pass

    deps.mark_stage_success(("local", target_node, cwd))
    return True, f"code_tar staged {len(existing)} paths to {target_node} ({cwd_size_mb}MB cwd)"
