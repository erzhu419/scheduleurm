from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

from .hybrid import HybridBackendDeps
from .local_backend import LocalBackendRuntimeDeps
from scheduler_windows.backend import WindowsBackendRuntimeDeps


class Backend:
    """Abstract interface for node launch/kill/probe backends."""

    name = "abstract"

    def requires_local_capacity_check(self, node: str, task: Optional[dict] = None,
                                      node_state: Optional[dict] = None) -> bool:
        return True

    def launch(self, task: dict, node_state: Optional[dict] = None) -> tuple[bool, str]:
        raise NotImplementedError

    def kill(self, task: dict, timeout: int = 15) -> tuple[bool, str]:
        raise NotImplementedError

    def kill_many(self, tasks: list[dict], timeout: int = 15) -> dict[str, tuple[bool, str]]:
        return {
            str(task.get("id") or ""): self.kill(task, timeout=timeout)
            for task in tasks
            if task.get("id")
        }

    def batch_probe(self, state: dict) -> dict:
        raise NotImplementedError


class LegacyExternalBackend(Backend):
    """Read-only probe surface for legacy externally-managed task records."""

    name = "legacy-external"

    def __init__(self, unsupported_message: Callable[[], str]):
        self._unsupported_message = unsupported_message

    def requires_local_capacity_check(self, node: str, task: Optional[dict] = None,
                                      node_state: Optional[dict] = None) -> bool:
        return True

    def launch(self, task: dict, node_state: Optional[dict] = None) -> tuple[bool, str]:
        return False, self._unsupported_message()

    def kill(self, task: dict, timeout: int = 15) -> tuple[bool, str]:
        return False, self._unsupported_message()

    def batch_probe(self, state: dict) -> dict:
        results = {}
        for task in state.get("tasks", []):
            if task.get("status") != "running" or not task.get("slurm_job_id"):
                continue
            results[task["id"]] = {
                "state": "unknown",
                "alive_pids": [],
                "vram_mb": 0,
                "ram_mb": 0,
                "pcpu": 0.0,
                "error": self._unsupported_message(),
            }
        return results


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_windows_backend_runtime_deps(namespace: Mapping[str, Any]) -> WindowsBackendRuntimeDeps:
    """Build WindowsBackend runtime deps from scheduler.py's compatibility namespace."""
    os_mod = _ns(namespace, "os")
    subprocess_mod = _ns(namespace, "subprocess")
    time_mod = _ns(namespace, "time")
    return WindowsBackendRuntimeDeps(
        node_configs=_ns(namespace, "NODES"),
        log_dir=_ns(namespace, "LOG_DIR"),
        wrapper_resource_log_interval_s=_ns(namespace, "WINDOWS_WRAPPER_RESOURCE_LOG_INTERVAL_S"),
        launcher_text=_ns(namespace, "_WINDOWS_LAUNCHER"),
        detacher_text=_ns(namespace, "_WINDOWS_DETACHER"),
        ssh_base_args=_ns(namespace, "_ssh_base_args"),
        run_subprocess=subprocess_mod.run,
        node_cpu_fallback_block_reason=_ns(namespace, "_node_cpu_fallback_block_reason"),
        windows_env_spec_error=_ns(namespace, "_windows_env_spec_error"),
        windows_path_for_node=_ns(namespace, "_windows_path_for_node"),
        ps_quote=_ns(namespace, "_ps_quote"),
        run_windows_ps=_ns(namespace, "_run_windows_ps"),
        apply_cpu_parallel_plan_to_task=_ns(namespace, "_apply_cpu_parallel_plan_to_task"),
        cpu_parallel_env=_ns(namespace, "_cpu_parallel_env"),
        safe_extra_env_items=_ns(namespace, "_safe_extra_env_items"),
        launch_extra_env=_ns(namespace, "_launch_extra_env"),
        windows_prepare_command=_ns(namespace, "_windows_prepare_command"),
        remember_last_placement=_ns(namespace, "_remember_last_placement"),
        set_current_usage=_ns(namespace, "_set_current_usage"),
        collect_windows_probe_targets=_ns(namespace, "_collect_windows_probe_targets"),
        node_is_windows=_ns(namespace, "_node_is_windows"),
        task_pids=_ns(namespace, "_task_pids"),
        windows_path_for_task=_ns(namespace, "_windows_path_for_task"),
        success_patterns=_ns(namespace, "SUCCESS_PATTERNS"),
        windows_log_exit_probe_specs=_ns(namespace, "_windows_log_exit_probe_specs"),
        windows_log_exit_probe_script=_ns(namespace, "_windows_log_exit_probe_script"),
        windows_log_exit_results_from_rows=_ns(namespace, "_windows_log_exit_results_from_rows"),
        windows_process_probe_specs=_ns(namespace, "_windows_process_probe_specs"),
        windows_process_probe_script=_ns(namespace, "_windows_process_probe_script"),
        normalize_windows_json_rows=_ns(namespace, "_normalize_windows_json_rows"),
        windows_queue_accounting_fallback_result=_ns(namespace, "_windows_queue_accounting_fallback_result"),
        windows_process_results_from_rows=_ns(namespace, "_windows_process_results_from_rows"),
        local_launch_transport_alive=_ns(namespace, "_local_launch_transport_alive"),
        now=time_mod.time,
        time_ns=time_mod.time_ns,
        getpid=os_mod.getpid,
        environ=os_mod.environ,
    )


def build_local_backend_runtime_deps(namespace: Mapping[str, Any]) -> LocalBackendRuntimeDeps:
    """Build LocalBackend runtime deps from scheduler.py's compatibility namespace."""
    time_mod = _ns(namespace, "time")
    claim_manager = _ns(namespace, "_ClaimManager")
    return LocalBackendRuntimeDeps(
        state_dir=_ns(namespace, "STATE_DIR"),
        node_configs=_ns(namespace, "NODES"),
        remote_path_for_node=_ns(namespace, "_remote_path_for_node"),
        apply_node_cmd_rewrites=_ns(namespace, "_apply_node_cmd_rewrites"),
        rewrite_command_paths_for_node=_ns(namespace, "_rewrite_command_paths_for_node"),
        inject_python_u=_ns(namespace, "_inject_python_u"),
        maybe_wrap_docker=_ns(namespace, "_maybe_wrap_docker"),
        run_on=_ns(namespace, "run_on"),
        claim_enabled_for=claim_manager.enabled_for,
        claim=claim_manager.claim,
        update_claim_pid=claim_manager.update_pid,
        release_task_claims_and_intents=_ns(namespace, "_release_task_claims_and_intents"),
        task_required_gpu_idx=_ns(namespace, "_task_required_gpu_idx"),
        gpu_fits=_ns(namespace, "_gpu_fits"),
        node_launch_extra_env=_ns(namespace, "_node_launch_extra_env"),
        safe_extra_env_items=_ns(namespace, "_safe_extra_env_items"),
        remember_last_placement=_ns(namespace, "_remember_last_placement"),
        set_current_usage=_ns(namespace, "_set_current_usage"),
        task_pids=_ns(namespace, "_task_pids"),
        task_process_groups=_ns(namespace, "_task_process_groups"),
        descendants_of=_ns(namespace, "_descendants_of"),
        running_probe_max_workers=_ns(namespace, "RUNNING_PROBE_MAX_WORKERS"),
        running_probe_timeout_s=_ns(namespace, "RUNNING_PROBE_TIMEOUT_S"),
        now=time_mod.time,
        sleep=time_mod.sleep,
    )


def build_hybrid_backend_deps(namespace: Mapping[str, Any]) -> HybridBackendDeps:
    return HybridBackendDeps(
        node_is_windows=_ns(namespace, "_node_is_windows"),
        local_backend_factory=_ns(namespace, "LocalBackend"),
        legacy_backend_factory=_ns(namespace, "LegacyExternalBackend"),
        windows_backend_factory=_ns(namespace, "WindowsBackend"),
    )
