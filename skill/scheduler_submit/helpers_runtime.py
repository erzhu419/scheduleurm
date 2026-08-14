from __future__ import annotations

import os
import time
from typing import Any, Mapping, Optional


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_submit_helpers_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    exports: dict[str, Any] = {}

    simple_impl_exports = {
        "_cmd_looks_like_training": "_cmd_looks_like_training_impl",
        "_safe_read_text": "_safe_read_text_impl",
        "_script_invocation_from_cmd": "_script_invocation_from_cmd_impl",
        "_submit_policy_text": "_submit_policy_text_impl",
        "_shell_split_statements": "_shell_split_statements_impl",
        "_shell_expand_simple": "_shell_expand_simple_impl",
        "_seed_value_substitute": "_seed_value_substitute_impl",
        "_submit_context_is_bapr": "_submit_context_is_bapr_impl",
        "_expand_simple_seed_loop_inner": "_expand_simple_seed_loop_inner_impl",
        "_split_bapr_seed_batch_submit_args": "_split_bapr_seed_batch_submit_args_impl",
        "_simple_shell_env_from_script": "_simple_shell_env_from_script_impl",
        "_extract_script_cd_dir": "_extract_script_cd_dir_impl",
        "_arg_value": "_arg_value_impl",
        "_abs_under": "_abs_under_impl",
        "_infer_direct_jax_train_ckpt": "_infer_direct_jax_train_ckpt_impl",
        "_infer_wrapper_ckpt": "_infer_wrapper_ckpt_impl",
        "_infer_bapr_run_seed_ckpt": "_infer_bapr_run_seed_ckpt_impl",
        "_infer_checkpoint_from_submit": "_infer_checkpoint_from_submit_impl",
        "_candidate_training_source_paths": "_candidate_training_source_paths_impl",
        "_checkpoint_contract_reason": "_checkpoint_contract_reason_impl",
        "_cmd_explicitly_cpu": "_cmd_explicitly_cpu_impl",
        "_task_looks_like_training": "_task_looks_like_training_impl",
        "_cpu_training_policy_reason": "_cpu_training_policy_reason_impl",
        "_cmd_has_resume_flag": "_cmd_has_resume_flag_impl",
        "_resume_capability_reason": "_resume_capability_reason_impl",
        "_conflict_path_key": "_conflict_path_key_impl",
        "_same_declared_path": "_same_declared_path_impl",
        "_active_or_unsynced_result_status": "_active_or_unsynced_result_status_impl",
        "_task_run_identity": "_task_run_identity_impl",
        "_legacy_external_state_is_terminal": "_legacy_external_state_is_terminal_impl",
        "_slurm_state_is_terminal": "_slurm_state_is_terminal_impl",
        "_task_is_descendant_of": "_task_is_descendant_of_impl",
        "_remember_last_placement": "_remember_last_placement_impl",
        "_runtime_history_closest_index": "_runtime_history_closest_index_impl",
        "_runtime_cmd_tokens": "_runtime_cmd_tokens_impl",
        "_runtime_script_name": "_runtime_script_name_impl",
        "_read_text_tail": "_read_text_tail_impl",
        "_flag_present_or_true": "_flag_present_or_true_impl",
        "_doctor_path_key": "_doctor_path_key_impl",
        "_resolve_cmd_path": "_resolve_cmd_path_impl",
        "_task_wait_files": "_task_wait_files_impl",
        "_task_has_wait_file": "_task_has_wait_file_impl",
        "_ready_local_file": "_ready_local_file_impl",
    }
    for public_name, impl_name in simple_impl_exports.items():
        exports[public_name] = _ns(namespace, impl_name)

    namespace.update(exports)

    def _infer_bapr_result_dirs_from_cmd(cmd: str, cwd: str = "") -> list:
        return _ns(namespace, "_infer_bapr_result_dirs_from_cmd_impl")(
            cmd,
            cwd,
            conflict_path_key=_ns(namespace, "_conflict_path_key"),
        )

    def _bapr_run_seed_metas_from_cmd(cmd: str, cwd: str = "") -> list[dict]:
        return _ns(namespace, "_bapr_run_seed_metas_from_cmd_impl")(
            cmd,
            cwd,
            conflict_path_key=_ns(namespace, "_conflict_path_key"),
        )

    def _bapr_batch_projection(task: dict, tail_text: str, elapsed_s: float) -> Optional[dict]:
        return _ns(namespace, "_bapr_batch_projection_impl")(
            task,
            tail_text,
            elapsed_s,
            load_eta_tracker_module=_ns(namespace, "_load_eta_tracker_module"),
            clean_result_path=_ns(namespace, "_clean_result_path"),
            conflict_path_key=_ns(namespace, "_conflict_path_key"),
        )

    def _local_launch_transport_alive(task: dict) -> bool:
        return _ns(namespace, "_local_launch_transport_alive_impl")(
            task,
            alive=_ns(namespace, "_pid_alive_local"),
        )

    def _task_identity_deps():
        return _ns(namespace, "_TaskIdentityDeps")(
            task_pids=_ns(namespace, "_task_pids"),
            local_launch_transport_alive=_ns(namespace, "_local_launch_transport_alive"),
            backend=_ns(namespace, "_BACKEND"),
        )

    def _task_has_recorded_launch_artifacts(task: dict) -> bool:
        return _ns(namespace, "_task_has_recorded_launch_artifacts_impl")(
            task,
            deps=_ns(namespace, "_task_identity_deps")(),
        )

    def _recorded_launch_safety_state(task: dict) -> tuple[str, str]:
        return _ns(namespace, "_recorded_launch_safety_state_impl")(
            task,
            deps=_ns(namespace, "_task_identity_deps")(),
        )

    def _same_run_identity_live_artifact_reason(
        task: dict,
        state: dict,
        run_key=None,
    ) -> Optional[str]:
        return _ns(namespace, "_same_run_identity_live_artifact_reason_impl")(
            task,
            state,
            run_key,
            deps=_ns(namespace, "_task_identity_deps")(),
        )

    def _mark_user_cancelled(task: dict, reason: str = "user cancel") -> None:
        return _ns(namespace, "_mark_user_cancelled_impl")(
            task,
            reason,
            actor_info=_ns(namespace, "_actor_info"),
            now=time.time,
        )

    def _queued_artifact_reconcile_deps():
        return _ns(namespace, "_build_queued_artifact_reconcile_deps")(namespace)

    def _queued_launch_artifacts(task: dict) -> list[str]:
        return _ns(namespace, "_queued_launch_artifacts_impl")(
            task,
            deps=_ns(namespace, "_queued_artifact_reconcile_deps")(),
        )

    def _reconcile_queued_launch_artifacts_before_dispatch(
        task: dict,
        state: dict,
    ) -> Optional[dict]:
        return _ns(namespace, "_reconcile_queued_launch_artifacts_before_dispatch_impl")(
            task,
            state,
            deps=_ns(namespace, "_queued_artifact_reconcile_deps")(),
        )

    def _cancel_related_queued_retries(state: dict, task: dict, reason: str) -> int:
        return _ns(namespace, "_cancel_related_queued_retries_impl")(
            state,
            task,
            reason,
            task_run_identity=_ns(namespace, "_task_run_identity"),
            release_task_claims_and_intents=_ns(namespace, "_release_task_claims_and_intents"),
            mark_user_cancelled=_ns(namespace, "_mark_user_cancelled"),
        )

    def _has_user_cancelled_retry_descendant(
        parent: dict,
        state: dict,
        parent_key=None,
    ) -> bool:
        return _ns(namespace, "_has_user_cancelled_retry_descendant_impl")(
            parent,
            state,
            parent_key,
            task_run_identity=_ns(namespace, "_task_run_identity"),
        )

    def reconcile_requeue_lineage_invariants(state: dict) -> int:
        return _ns(namespace, "_reconcile_requeue_lineage_invariants_impl")(
            state,
            release_task_claims_and_intents=_ns(namespace, "_release_task_claims_and_intents"),
            mark_user_cancelled=_ns(namespace, "_mark_user_cancelled"),
            now=time.time,
        )

    def _queued_cpu_training_block_reason(task):
        if task.get("status") != "queued":
            return None
        if task.get("auto_adopted") or task.get("adopted"):
            return None
        req_node = task.get("require_node")
        if (req_node in _ns(namespace, "_cpu_labor_node_names")()
                and int(task.get("est_vram_mb") or 0) <= 0):
            return None
        return _ns(namespace, "_cpu_training_policy_reason")(
            _ns(namespace, "_submit_policy_text")(task.get("cmd", ""), task.get("cwd", "")),
            task.get("description", ""),
            bool(task.get("allow_cpu_training", False)),
            int(task.get("est_vram_mb") or 0),
        )

    def _queued_wait_for_file_block_reason(task):
        if task.get("status") != "queued":
            return None
        waits = task.get("wait_for_files") or []
        if isinstance(waits, str):
            waits = [waits]
        if not waits:
            return None
        missing = []
        for raw in waits:
            path_s = os.path.expandvars(os.path.expanduser(str(raw)))
            try:
                st = os.stat(path_s)
            except OSError:
                missing.append(path_s)
                continue
            if not os.path.isfile(path_s) or st.st_size <= 0:
                missing.append(path_s)
        if not missing:
            return None
        shown = ", ".join(missing[:3])
        more = "" if len(missing) <= 3 else f" (+{len(missing) - 3} more)"
        return f"waiting for prerequisite file(s): {shown}{more}"

    exports.update({
        "_infer_bapr_result_dirs_from_cmd": _infer_bapr_result_dirs_from_cmd,
        "_bapr_run_seed_metas_from_cmd": _bapr_run_seed_metas_from_cmd,
        "_bapr_batch_projection": _bapr_batch_projection,
        "_local_launch_transport_alive": _local_launch_transport_alive,
        "_task_identity_deps": _task_identity_deps,
        "_task_has_recorded_launch_artifacts": _task_has_recorded_launch_artifacts,
        "_recorded_launch_safety_state": _recorded_launch_safety_state,
        "_same_run_identity_live_artifact_reason": _same_run_identity_live_artifact_reason,
        "_mark_user_cancelled": _mark_user_cancelled,
        "_queued_artifact_reconcile_deps": _queued_artifact_reconcile_deps,
        "_queued_launch_artifacts": _queued_launch_artifacts,
        "_reconcile_queued_launch_artifacts_before_dispatch": _reconcile_queued_launch_artifacts_before_dispatch,
        "_cancel_related_queued_retries": _cancel_related_queued_retries,
        "_has_user_cancelled_retry_descendant": _has_user_cancelled_retry_descendant,
        "reconcile_requeue_lineage_invariants": reconcile_requeue_lineage_invariants,
        "_queued_cpu_training_block_reason": _queued_cpu_training_block_reason,
        "_queued_wait_for_file_block_reason": _queued_wait_for_file_block_reason,
    })
    return exports
