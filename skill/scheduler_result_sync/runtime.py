"""Runtime-bound result-sync wrappers for scheduler.py."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Mapping, Optional

from scheduler_result_sync import (
    ResultSyncDeps,
    sync_completed_results_outside_lock as _sync_completed_results_outside_lock_impl,
)
from scheduler_result_sync.executor import (
    WindowsResultSyncDeps,
    sync_one_result as _sync_one_result_impl,
    sync_windows_result as _sync_windows_result_impl,
)
from scheduler_result_sync.background import (
    ResultSyncBackgroundDeps,
    request_result_sync_background as _request_result_sync_background_impl,
    run_result_sync_worker as _run_result_sync_worker_impl,
)
from scheduler_runtime.wiring_data_motion import build_result_sync_executor_deps as _build_result_sync_executor_deps


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_result_sync_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    worker_holder: dict[str, Any] = {"process": None}

    def _result_sync_background_deps():
        subprocess_mod = _ns(namespace, "subprocess")
        scheduler_dir = Path(namespace.get("__file__") or __file__).resolve().parent
        return ResultSyncBackgroundDeps(
            worker_lock=_ns(namespace, "result_sync_worker_lock"),
            sync_results=lambda: _sync_completed_results_outside_lock(),
            lock_timeout_type=_ns(namespace, "SchedulerLockTimeout"),
            scheduler_dir=scheduler_dir,
            python_executable=sys.executable,
            log_path=_ns(namespace, "LOG_DIR") / "result_sync_worker.log",
            popen=subprocess_mod.Popen,
            stderr_to_stdout=subprocess_mod.STDOUT,
        )
    def _sync_windows_result(candidate: dict) -> tuple:
        subprocess_mod = _ns(namespace, "subprocess")
        return _sync_windows_result_impl(
            candidate,
            deps=WindowsResultSyncDeps(
                windows_path_for_node=_ns(namespace, "_windows_path_for_node"),
                ps_quote=_ns(namespace, "_ps_quote"),
                ssh_base_args=_ns(namespace, "_ssh_base_args"),
                result_sync_timeout_s=_ns(namespace, "RESULT_SYNC_TIMEOUT_S"),
                make_local_dir=lambda path: Path(path).mkdir(parents=True, exist_ok=True),
                popen=subprocess_mod.Popen,
                subprocess_run=subprocess_mod.run,
                pipe=subprocess_mod.PIPE,
            ),
        )

    def _result_sync_executor_deps():
        return _build_result_sync_executor_deps(namespace)

    def _sync_one_result(candidate: dict) -> tuple:
        return _sync_one_result_impl(
            candidate,
            deps=_ns(namespace, "_result_sync_executor_deps")(),
        )

    def _sync_completed_results_outside_lock(max_candidates: Optional[int] = None):
        return _sync_completed_results_outside_lock_impl(
            max_candidates=max_candidates,
            deps=ResultSyncDeps(
                state_lock=_ns(namespace, "state_lock"),
                load_state=_ns(namespace, "load_state"),
                save_state=_ns(namespace, "save_state"),
                notify=_ns(namespace, "notify"),
                node_configs=_ns(namespace, "NODES"),
                infer_bapr_result_dirs_from_cmd=_ns(namespace, "_infer_bapr_result_dirs_from_cmd"),
                sync_one_result=_ns(namespace, "_sync_one_result"),
                result_sync_timeout_s=_ns(namespace, "RESULT_SYNC_TIMEOUT_S"),
                result_sync_stale_grace_s=_ns(namespace, "RESULT_SYNC_STALE_GRACE_S"),
                result_sync_max_per_cycle=_ns(namespace, "RESULT_SYNC_MAX_PER_CYCLE"),
                result_sync_dependency_max_per_cycle=_ns(
                    namespace, "RESULT_SYNC_DEPENDENCY_MAX_PER_CYCLE"
                ),
                result_sync_max_attempts=_ns(namespace, "RESULT_SYNC_MAX_ATTEMPTS"),
                now=_ns(namespace, "time").time,
                getpid=_ns(namespace, "os").getpid,
                pid_alive=_ns(namespace, "_pid_alive_local"),
            ),
        )

    def _run_result_sync_worker():
        return _run_result_sync_worker_impl(deps=_result_sync_background_deps())

    def _request_result_sync_background():
        return _request_result_sync_background_impl(
            worker_holder,
            deps=_result_sync_background_deps(),
        )

    return {
        "_sync_windows_result": _sync_windows_result,
        "_result_sync_executor_deps": _result_sync_executor_deps,
        "_sync_one_result": _sync_one_result,
        "_sync_completed_results_outside_lock": _sync_completed_results_outside_lock,
        "_run_result_sync_worker": _run_result_sync_worker,
        "_request_result_sync_background": _request_result_sync_background,
    }
