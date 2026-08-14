from __future__ import annotations

import hashlib
import os
import re
import secrets
import shlex
import subprocess
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class LocalLaunchDeps:
    state_dir: Any
    node_configs: dict
    remote_path_for_node: Callable[[str, str], str]
    apply_node_cmd_rewrites: Callable[[str, str], str]
    rewrite_command_paths_for_node: Callable[[str, str], str]
    inject_python_u: Callable[[str], str]
    maybe_wrap_docker: Callable[[dict, str, str], tuple[str, Optional[str]]]
    run_on: Callable[..., tuple[int, str, str]]
    claim_enabled_for: Callable[[str], bool]
    claim: Callable[..., tuple[bool, Any, str]]
    update_claim_pid: Callable[[str, str, int], bool]
    release_task_claims_and_intents: Callable[..., Any]
    task_required_gpu_idx: Callable[[dict], Any]
    gpu_fits: Callable[[dict, dict, dict], bool]
    node_launch_extra_env: Callable[[dict], dict]
    safe_extra_env_items: Callable[[dict], list[tuple[str, str]]]
    remember_last_placement: Callable[[dict], Any]
    set_current_usage: Callable[[dict, int, int, float], Any]
    now: Callable[[], float]
    sleep: Callable[[float], Any]


def _launch_log_path(task: dict, deps: LocalLaunchDeps) -> str:
    node = task["node"]
    if deps.node_configs[node]["host"] is None:
        return f"{deps.state_dir}/logs/{task['id']}.log"
    return f"/tmp/sched_{task['id']}.log"


def _prepare_inner_command(task: dict, remote_cwd: str, deps: LocalLaunchDeps) -> tuple[str, Optional[str]]:
    node = task.get("node")
    inner = deps.apply_node_cmd_rewrites(node, task["cmd"])
    inner = deps.rewrite_command_paths_for_node(node, inner)
    inner = deps.inject_python_u(inner)
    resume_path = task.get("resume_from")
    resume_flag = task.get("resume_flag") or ""
    if resume_path and resume_flag:
        inner = f"{inner} {resume_flag} {shlex.quote(deps.remote_path_for_node(task['node'], resume_path))}"
    return deps.maybe_wrap_docker(task, inner, remote_cwd)


def _probe_cwd(task: dict, remote_cwd: str, deps: LocalLaunchDeps) -> tuple[bool, str]:
    details = []
    rc_cwd = 1
    err_cwd = ""
    for attempt, timeout_s in enumerate((10, 20), start=1):
        try:
            rc_cwd, _, err_cwd = deps.run_on(
                task["node"],
                f"test -d {shlex.quote(remote_cwd)}",
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            rc_cwd, err_cwd = 124, f"timeout after {timeout_s}s"
        except Exception as exc:
            rc_cwd, err_cwd = 255, str(exc)
        if rc_cwd == 0:
            return True, ""
        detail = f"attempt {attempt}: rc={rc_cwd}"
        if err_cwd:
            detail += f" err={str(err_cwd).strip()[:180]}"
        details.append(detail)
        if attempt == 1:
            deps.sleep(0.5)

    detail_text = "; ".join(details)
    if rc_cwd in (124, 125, 126, 127, 255) or err_cwd:
        return False, f"cwd probe failed on {task['node']}: {remote_cwd}; {detail_text}"
    return False, f"cwd missing on {task['node']}: {remote_cwd}; {detail_text}"


def _claim_launch_resources(task: dict, node_state: Optional[dict], deps: LocalLaunchDeps):
    if not deps.claim_enabled_for(task["node"]):
        return None, None

    original_gpu = task.get("gpu_idx")
    gpu_attempts = [original_gpu]
    require_gpu_idx = deps.task_required_gpu_idx(task)
    if original_gpu is not None and node_state and require_gpu_idx is None:
        try:
            node_info = deps.node_configs[task["node"]]
            for gpu in node_state.get("gpus") or []:
                gpu_idx = gpu.get("idx")
                if gpu_idx == original_gpu or gpu_idx in gpu_attempts:
                    continue
                if deps.gpu_fits(task, gpu, node_info):
                    gpu_attempts.append(gpu_idx)
        except Exception:
            pass

    conflicts = []
    for gpu_idx in gpu_attempts:
        task["gpu_idx"] = gpu_idx
        ok_claim, info, kind = deps.claim(task["node"], task, gpu_idx, node_state)
        if ok_claim:
            return info, None
        if kind != "conflict":
            task["gpu_idx"] = original_gpu
            return None, f"CLAIM_ERROR: {info}"
        conflicts.append(f"gpu{gpu_idx}: {info}" if gpu_idx is not None else str(info))

    task["gpu_idx"] = original_gpu
    return None, f"CLAIM_RACE: {'; '.join(conflicts) or 'claim conflict'}"


def _env_prefix(task: dict, deps: LocalLaunchDeps, *, launch_token: str) -> str:
    gpu_idx = task.get("gpu_idx")
    if gpu_idx is None:
        prefix = 'export CUDA_VISIBLE_DEVICES=""; '
    else:
        prefix = f"export CUDA_VISIBLE_DEVICES={gpu_idx}; "
    prefix += f"export SCHEDULEURM_TASK_ID={shlex.quote(task.get('id') or '')}; "
    prefix += f"export SCHEDULEURM_LAUNCH_TOKEN={shlex.quote(launch_token)}; "
    for key, value in deps.safe_extra_env_items(deps.node_launch_extra_env(task)):
        prefix += f"export {key}={shlex.quote(value)}; "
    return prefix


def _parse_launch_pid(output: str) -> Optional[int]:
    for line in output.splitlines():
        line = line.strip()
        if not line.startswith("PID="):
            continue
        try:
            return int(line[4:])
        except ValueError:
            pass
    return None


def _parse_launch_start_ticks(output: str) -> Optional[int]:
    for line in output.splitlines():
        line = line.strip()
        if not line.startswith("START_TICKS="):
            continue
        try:
            value = int(line.split("=", 1)[1])
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def runtime_exit_status_path(task_id: Any, launch_token: Any) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task_id or "task"))
    token = str(launch_token or "")
    digest = hashlib.sha256(token.encode("utf-8", errors="replace")).hexdigest()[:24]
    return f"/tmp/scheduleurm_exit_{safe_id}_{digest}.status"


def _runtime_exit_status(task: dict) -> tuple[str, str]:
    token = str(task.get("launch_token") or secrets.token_hex(16))
    return token, runtime_exit_status_path(task.get("id"), token)


def _completion_wrapper(inner: str, status_path: str, status_token: str) -> str:
    status_arg = shlex.quote(status_path)
    token_arg = shlex.quote(status_token)
    return (
        "set +e; "
        f"bash -c {shlex.quote(inner)}; rc=$?; "
        f"tmp={status_arg}.tmp.$$; umask 077; "
        f"printf '%s\\t%s\\t%s\\n' \"$rc\" \"$(date +%s)\" {token_arg} > \"$tmp\"; "
        f"mv -f -- \"$tmp\" {status_arg}; exit \"$rc\""
    )


def _resolve_container_main_pid(task: dict, deps: LocalLaunchDeps) -> Optional[int]:
    container_name = task.get("container_name")
    if not container_name:
        return None
    for _ in range(6):
        try:
            rc, out, _ = deps.run_on(
                task["node"],
                f"docker inspect --format '{{{{.State.Pid}}}}' {shlex.quote(container_name)} 2>/dev/null",
                timeout=4,
                check=False,
            )
        except Exception:
            rc, out = 1, ""
        text = out.strip() if rc == 0 else ""
        if text.isdigit() and int(text) > 0:
            return int(text)
        deps.sleep(0.5)
    return None


def local_backend_launch(
    task: dict,
    *,
    node_state: Optional[dict],
    deps: LocalLaunchDeps,
) -> tuple[bool, str]:
    log_path = _launch_log_path(task, deps)
    cwd = task["cwd"]
    remote_cwd = deps.remote_path_for_node(task["node"], cwd)
    inner, docker_err = _prepare_inner_command(task, remote_cwd, deps)
    if docker_err:
        return False, docker_err

    ok_cwd, cwd_msg = _probe_cwd(task, remote_cwd, deps)
    if not ok_cwd:
        return False, cwd_msg

    claim_record, claim_error = _claim_launch_resources(task, node_state, deps)
    if claim_error:
        return False, claim_error

    def release_and_fail(message: str) -> tuple[bool, str]:
        if claim_record:
            try:
                deps.release_task_claims_and_intents(task)
            except Exception:
                pass
        return False, message

    status_token, status_path = _runtime_exit_status(task)
    env_prefix = _env_prefix(task, deps, launch_token=status_token)
    wrapped_inner = _completion_wrapper(inner, status_path, status_token)
    full_cmd = (
        f"mkdir -p {shlex.quote(os.path.dirname(log_path))}; "
        f"rm -f -- {shlex.quote(status_path)}; "
        f"cd {shlex.quote(remote_cwd)} && {env_prefix} "
        f"setsid bash -c {shlex.quote(wrapped_inner)} > {shlex.quote(log_path)} 2>&1 < /dev/null & "
        "pid=$!; echo PID=$pid; "
        "awk '{print \"START_TICKS=\" $22}' /proc/$pid/stat 2>/dev/null || true"
    )
    try:
        rc, out, err = deps.run_on(task["node"], full_cmd, timeout=20, check=False)
        if rc != 0:
            return release_and_fail(f"launch rc={rc}: {err.strip()[:200]}")
        pid = _parse_launch_pid(out)
        if not pid:
            return release_and_fail(f"could not parse PID from launch output: {out[:200]}")

        task["remote_pids"] = [pid]
        task["process_group"] = pid
        task["exit_status_path"] = status_path
        task["exit_status_token"] = status_token
        start_ticks = _parse_launch_start_ticks(out)
        if start_ticks:
            task["remote_pid_start_ticks"] = {str(pid): start_ticks}
        else:
            task.pop("remote_pid_start_ticks", None)
        task["log_path"] = log_path
        task["status"] = "running"
        task["started_at"] = deps.now()
        deps.remember_last_placement(task)
        task["peak_vram_mb"] = 0
        task["peak_ram_mb"] = 0
        deps.set_current_usage(task, 0, 0, 0.0)

        if claim_record:
            try:
                deps.update_claim_pid(task["node"], task["id"], pid)
            except Exception:
                pass

        container_name = task.get("container_name")
        if container_name:
            container_pid = _resolve_container_main_pid(task, deps)
            if container_pid:
                task["remote_pids"] = [container_pid]
                task["container_main_pid"] = container_pid
                task.pop("remote_pid_start_ticks", None)
                if claim_record:
                    try:
                        deps.update_claim_pid(task["node"], task["id"], container_pid)
                    except Exception:
                        pass
        return True, f"pid={pid}" + (
            f" container={container_name}@{task.get('container_main_pid')}" if container_name else ""
        )
    except Exception as exc:
        return release_and_fail(f"launch exception: {exc}")
