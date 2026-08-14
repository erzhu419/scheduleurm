"""Runtime-bound launch staging wrappers for scheduler.py."""

from __future__ import annotations

from typing import Any, Mapping, Optional


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_launch_staging_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    def _launch_cwd_staging_deps():
        return _ns(namespace, "_build_launch_cwd_staging_deps")(namespace)

    def _resume_ckpt_staging_deps():
        return _ns(namespace, "_build_resume_ckpt_staging_deps")(namespace)

    def _launch_staging_plan_deps():
        return _ns(namespace, "_build_launch_staging_plan_deps")(namespace)

    def _tar_exclude_args(extra_excludes: Optional[list] = None) -> list:
        return _ns(namespace, "_tar_exclude_args_impl")(extra_excludes)

    def _stage_local_dir_to_windows(
        local_dir: str,
        target_node: str,
        target_dir: str,
        extra_excludes: Optional[list] = None,
        timeout_s: int = 600,
    ) -> tuple:
        subprocess_mod = _ns(namespace, "subprocess")
        return _ns(namespace, "_stage_local_dir_to_windows_impl")(
            local_dir,
            target_node,
            target_dir,
            extra_excludes=extra_excludes,
            timeout_s=timeout_s,
            deps=_ns(namespace, "_WindowsTarStagingDeps")(
                node_is_windows=_ns(namespace, "_node_is_windows"),
                windows_path_for_node=_ns(namespace, "_windows_path_for_node"),
                ps_quote=_ns(namespace, "_ps_quote"),
                ssh_base_args=_ns(namespace, "_ssh_base_args"),
                run_windows_ps=_ns(namespace, "_run_windows_ps"),
                popen=subprocess_mod.Popen,
                run_subprocess=subprocess_mod.run,
            ),
        )

    def _stage_code_tar_for_launch(
        task: dict,
        target_node: str,
        remote_cwd: str,
        cwd_size_mb: int,
    ) -> tuple:
        os_mod = _ns(namespace, "os")
        subprocess_mod = _ns(namespace, "subprocess")
        time_mod = _ns(namespace, "time")
        return _ns(namespace, "_stage_code_tar_for_launch_impl")(
            task,
            target_node,
            remote_cwd,
            cwd_size_mb,
            deps=_ns(namespace, "_CodeTarStagingDeps")(
                node_configs=_ns(namespace, "NODES"),
                ssh_base_args=_ns(namespace, "_ssh_base_args"),
                run_on=_ns(namespace, "run_on"),
                mark_stage_success=_ns(namespace, "_mark_launch_stage_success"),
                getpid=os_mod.getpid,
                now=time_mod.time,
                run_subprocess=subprocess_mod.run,
                tmp_dir="/tmp",
            ),
        )

    def _stage_resume_ckpt_for_launch(
        task: dict,
        target_node: str,
        source_loc: Optional[dict] = None,
        max_ckpt_mb: int = None,
    ) -> tuple:
        if max_ckpt_mb is None:
            max_ckpt_mb = _ns(namespace, "MIGRATION_MAX_CKPT_SIZE_MB")
        return _ns(namespace, "_stage_resume_ckpt_for_launch_impl")(
            task,
            target_node,
            source_loc=source_loc,
            max_ckpt_mb=max_ckpt_mb,
            deps=_ns(namespace, "_resume_ckpt_staging_deps")(),
        )

    def _stage_cwd_for_launch(
        task: dict,
        target_node: str,
        extra_excludes: list = None,
    ) -> tuple:
        return _ns(namespace, "_stage_cwd_for_launch_impl")(
            task,
            target_node,
            extra_excludes=extra_excludes,
            deps=_ns(namespace, "_launch_cwd_staging_deps")(),
        )

    def _launch_input_stage_state(task: dict, target_node: str) -> str:
        return _ns(namespace, "_launch_input_stage_state_impl")(
            task,
            target_node,
            deps=_ns(namespace, "_launch_cwd_staging_deps")(),
        )

    def _stage_input_paths_for_launch(
        task: dict,
        target_node: str,
    ) -> tuple:
        return _ns(namespace, "_stage_input_paths_for_launch_impl")(
            task,
            target_node,
            deps=_ns(namespace, "_launch_cwd_staging_deps")(),
        )

    def _stage_launch_candidates_outside_lock(
        task_ids: Optional[set] = None,
        *,
        background: bool = False,
    ):
        time_mod = _ns(namespace, "time")
        return _ns(namespace, "_stage_launch_candidates_outside_lock_impl")(
            task_ids,
            background=background,
            deps=_ns(namespace, "_LaunchStagingRunnerDeps")(
                state_lock=_ns(namespace, "state_lock"),
                load_state=_ns(namespace, "load_state"),
                collect_launch_staging_plan=lambda state, selected_task_ids: _ns(
                    namespace,
                    "_collect_launch_staging_plan",
                )(
                    state,
                    selected_task_ids,
                    deps=_ns(namespace, "_launch_staging_plan_deps")(),
                ),
                launch_stage_candidate_key=lambda item: _ns(
                    namespace,
                    "_launch_stage_candidate_key",
                )(
                    item,
                    node_configs=_ns(namespace, "NODES"),
                    node_is_windows=_ns(namespace, "_node_is_windows"),
                ),
                max_candidates_per_pass=_ns(namespace, "LAUNCH_STAGING_MAX_CANDIDATES_PER_PASS"),
                worker_count=_ns(namespace, "LAUNCH_STAGING_WORKERS"),
                stage_cwd_for_launch=_ns(namespace, "_stage_cwd_for_launch"),
                stage_input_paths_for_launch=_ns(namespace, "_stage_input_paths_for_launch"),
                launch_input_stage_key=lambda task, target: _ns(
                    namespace, "_launch_input_stage_key_impl")(
                        task,
                        target,
                        node_configs=_ns(namespace, "NODES"),
                    ),
                stage_resume_ckpt_for_launch=_ns(namespace, "_stage_resume_ckpt_for_launch"),
                resume_ckpt_stage_key=_ns(namespace, "_resume_ckpt_stage_key"),
                staging_cache_hit=_ns(namespace, "_staging_cache_hit"),
                staging_fails=_ns(namespace, "_STAGING_FAILS"),
                notify=_ns(namespace, "notify"),
                now=time_mod.time,
                staging_key_guard=_ns(namespace, "_staging_key_guard"),
            ),
        )

    return {
        "_launch_cwd_staging_deps": _launch_cwd_staging_deps,
        "_resume_ckpt_staging_deps": _resume_ckpt_staging_deps,
        "_launch_staging_plan_deps": _launch_staging_plan_deps,
        "_tar_exclude_args": _tar_exclude_args,
        "_stage_local_dir_to_windows": _stage_local_dir_to_windows,
        "_stage_code_tar_for_launch": _stage_code_tar_for_launch,
        "_stage_resume_ckpt_for_launch": _stage_resume_ckpt_for_launch,
        "_stage_cwd_for_launch": _stage_cwd_for_launch,
        "_launch_input_stage_state": _launch_input_stage_state,
        "_stage_input_paths_for_launch": _stage_input_paths_for_launch,
        "_stage_launch_candidates_outside_lock": _stage_launch_candidates_outside_lock,
    }
