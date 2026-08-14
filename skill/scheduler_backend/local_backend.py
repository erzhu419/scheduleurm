from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .local_launch import LocalLaunchDeps, local_backend_launch
from .local_probe import LocalBatchProbeDeps, local_backend_batch_probe


MAX_BATCH_KILL_TASKS_PER_COMMAND = 256
KILL_VERIFICATION_FAILED_RC = 75


def _kill_command(
    *,
    containers: list[str],
    pgids: list[int],
    pids: list[int],
) -> str:
    """Terminate tracked Unix launch identities and fail unless all are gone."""
    parts = []
    quoted_containers = " ".join(shlex.quote(name) for name in containers)
    if quoted_containers:
        parts.append(f"docker stop -t 5 {quoted_containers} 2>/dev/null || true")
    parts += [f"kill -- -{pgid} 2>/dev/null || true" for pgid in pgids]
    parts += [f"kill {pid} 2>/dev/null || true" for pid in pids]
    parts.append("sleep 1")
    if quoted_containers:
        parts.append(f"docker kill {quoted_containers} 2>/dev/null || true")
    parts += [f"kill -9 -- -{pgid} 2>/dev/null || true" for pgid in pgids]
    parts += [f"kill -9 {pid} 2>/dev/null || true" for pid in pids]
    parts.append("sleep 1")

    parts.append("_scheduleurm_kill_alive=0")
    for container in containers:
        quoted = shlex.quote(container)
        parts.append(
            "if [ \"$(docker inspect --format '{{.State.Running}}' "
            f"{quoted} 2>/dev/null || true)\" = true ]; then "
            f"echo SCHEDULEURM_KILL_SURVIVOR=container:{quoted}; "
            "_scheduleurm_kill_alive=1; fi"
        )
    for pgid in pgids:
        parts.append(
            f"if kill -0 -- -{pgid} 2>/dev/null && "
            "ps -eo pgid=,stat= | "
            f"awk -v want={pgid} '$1 == want && substr($2,1,1) != \"Z\" "
            "{ alive=1 } END { exit alive ? 0 : 1 }'; then "
            f"echo SCHEDULEURM_KILL_SURVIVOR=pgid:{pgid}; "
            "_scheduleurm_kill_alive=1; fi"
        )
    for pid in pids:
        parts.append(
            f"if kill -0 {pid} 2>/dev/null && "
            f"ps -o stat= -p {pid} 2>/dev/null | "
            "awk 'NF && substr($1,1,1) != \"Z\" { alive=1 } "
            "END { exit alive ? 0 : 1 }'; then "
            f"echo SCHEDULEURM_KILL_SURVIVOR=pid:{pid}; "
            "_scheduleurm_kill_alive=1; fi"
        )
    parts.append(
        f"if [ \"$_scheduleurm_kill_alive\" -ne 0 ]; then "
        f"exit {KILL_VERIFICATION_FAILED_RC}; fi"
    )
    return "; ".join(parts)


def _remote_result(result: Any) -> tuple[int, str, str]:
    if isinstance(result, tuple) and len(result) >= 3:
        return int(result[0]), str(result[1] or ""), str(result[2] or "")
    return 0, str(result or ""), ""


def _kill_error(rc: int, stdout: str, stderr: str) -> str:
    detail = (stderr.strip() or stdout.strip())[-400:]
    if rc == KILL_VERIFICATION_FAILED_RC:
        prefix = "kill verification found surviving process identities"
    else:
        prefix = f"kill transport/command failed rc={rc}"
    return f"{prefix}: {detail}" if detail else prefix


@dataclass(frozen=True)
class LocalBackendRuntimeDeps:
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
    task_pids: Callable[[dict], list[int]]
    task_process_groups: Callable[[dict], list[int]]
    descendants_of: Callable[[set[int], dict[int, int]], set[int]]
    running_probe_max_workers: int
    running_probe_timeout_s: int
    now: Callable[[], float]
    sleep: Callable[[float], Any]


class LocalBackend:
    """ssh + nohup + setsid backend for Unix-like nodes."""

    name = "local"

    def __init__(self, deps: LocalBackendRuntimeDeps | Callable[[], LocalBackendRuntimeDeps]):
        self._deps_source = deps

    @property
    def deps(self) -> LocalBackendRuntimeDeps:
        if callable(self._deps_source):
            return self._deps_source()
        return self._deps_source

    def _launch_deps(self) -> LocalLaunchDeps:
        deps = self.deps
        return LocalLaunchDeps(
            state_dir=deps.state_dir,
            node_configs=deps.node_configs,
            remote_path_for_node=deps.remote_path_for_node,
            apply_node_cmd_rewrites=deps.apply_node_cmd_rewrites,
            rewrite_command_paths_for_node=deps.rewrite_command_paths_for_node,
            inject_python_u=deps.inject_python_u,
            maybe_wrap_docker=deps.maybe_wrap_docker,
            run_on=deps.run_on,
            claim_enabled_for=deps.claim_enabled_for,
            claim=deps.claim,
            update_claim_pid=deps.update_claim_pid,
            release_task_claims_and_intents=deps.release_task_claims_and_intents,
            task_required_gpu_idx=deps.task_required_gpu_idx,
            gpu_fits=deps.gpu_fits,
            node_launch_extra_env=deps.node_launch_extra_env,
            safe_extra_env_items=deps.safe_extra_env_items,
            remember_last_placement=deps.remember_last_placement,
            set_current_usage=deps.set_current_usage,
            now=deps.now,
            sleep=deps.sleep,
        )

    def _batch_probe_deps(self) -> LocalBatchProbeDeps:
        deps = self.deps
        return LocalBatchProbeDeps(
            run_on=deps.run_on,
            task_pids=deps.task_pids,
            descendants_of=deps.descendants_of,
            max_workers=deps.running_probe_max_workers,
            timeout_s=deps.running_probe_timeout_s,
        )

    def launch(self, task: dict, node_state: Optional[dict] = None) -> tuple[bool, str]:
        return local_backend_launch(task, node_state=node_state, deps=self._launch_deps())

    def kill(self, task: dict, timeout: int = 15) -> tuple[bool, str]:
        node = task.get("node")
        if not node:
            return False, "no node"
        deps = self.deps
        pids = [int(p) for p in deps.task_pids(task) if p]
        pgids = deps.task_process_groups(task)
        container = task.get("container_name")
        if not pids and not pgids and not container:
            return False, "no pids"
        containers = [str(container)] if container else []
        cmd = _kill_command(containers=containers, pgids=pgids, pids=pids)
        try:
            rc, stdout, stderr = _remote_result(
                deps.run_on(node, cmd, timeout=timeout, check=False)
            )
            if rc != 0:
                return False, _kill_error(rc, stdout, stderr)
            bits = []
            if container:
                bits.append(f"container={container}")
            if pgids:
                bits.append(f"pgids={pgids}")
            if pids:
                bits.append(f"pids={pids}")
            return True, " ".join(bits)
        except Exception as exc:
            return False, str(exc)[:200]

    def kill_many(self, tasks: list[dict], timeout: int = 15) -> dict[str, tuple[bool, str]]:
        """Kill many Unix tasks with one remote command per node."""
        deps = self.deps
        by_node: dict[str, list[dict]] = {}
        results: dict[str, tuple[bool, str]] = {}
        for task in tasks:
            task_id = str(task.get("id") or "")
            node = str(task.get("node") or "")
            if not task_id:
                continue
            if not node:
                results[task_id] = (False, "no node")
                continue
            by_node.setdefault(node, []).append(task)

        node_batches = [
            (node, node_tasks[index:index + MAX_BATCH_KILL_TASKS_PER_COMMAND])
            for node, node_tasks in by_node.items()
            for index in range(0, len(node_tasks), MAX_BATCH_KILL_TASKS_PER_COMMAND)
        ]
        for node, node_tasks in node_batches:
            containers = sorted({
                str(task.get("container_name"))
                for task in node_tasks
                if task.get("container_name")
            })
            pgids = sorted({
                int(pgid)
                for task in node_tasks
                for pgid in deps.task_process_groups(task)
                if pgid
            })
            pids = sorted({
                int(pid)
                for task in node_tasks
                for pid in deps.task_pids(task)
                if pid
            })
            actionable = [
                task for task in node_tasks
                if task.get("container_name")
                or deps.task_process_groups(task)
                or deps.task_pids(task)
            ]
            actionable_ids = {str(task.get("id") or "") for task in actionable}
            for task in node_tasks:
                task_id = str(task.get("id") or "")
                if task_id not in actionable_ids:
                    results[task_id] = (False, "no pids")
            if not actionable:
                continue

            try:
                rc, stdout, stderr = _remote_result(
                    deps.run_on(
                        node,
                        _kill_command(
                            containers=containers,
                            pgids=pgids,
                            pids=pids,
                        ),
                        timeout=timeout,
                        check=False,
                    )
                )
                if rc != 0:
                    message = _kill_error(rc, stdout, stderr)
                    for task_id in actionable_ids:
                        results[task_id] = (False, message)
                    continue
                message = (
                    f"batch node={node} tasks={len(actionable)} "
                    f"containers={len(containers)} pgids={len(pgids)} pids={len(pids)}"
                )
                for task_id in actionable_ids:
                    results[task_id] = (True, message)
            except Exception as exc:
                message = str(exc)[:200]
                for task_id in actionable_ids:
                    results[task_id] = (False, message)
        return results

    def batch_probe(self, state: dict) -> dict:
        probe_tasks = []
        results = {}
        for task in state.get("tasks", []):
            node = str(task.get("node") or "")
            node_info = self.deps.node_configs.get(node, {}) if node else {}
            if task.get("status") == "running" and (node_info or {}).get("retired"):
                results[str(task.get("id") or "")] = {
                    "state": "unknown",
                    "alive_pids": [],
                    "vram_mb": 0,
                    "ram_mb": 0,
                    "pcpu": 0.0,
                    "error": f"node {node} is retired; remote probe skipped",
                }
                continue
            probe_tasks.append(task)
        results.update(
            local_backend_batch_probe(
                {**state, "tasks": probe_tasks},
                deps=self._batch_probe_deps(),
            )
        )
        return results
