from __future__ import annotations

import builtins
from typing import Any, Mapping

from scheduler_adopt.command import AdoptCommandDeps
from scheduler_commands.bulk_submit import BulkSubmitDeps
from scheduler_commands.cli import SchedulerCliDeps
from scheduler_commands.edit import EditCommandDeps
from scheduler_runtime.history import HistoryCommandDeps
from scheduler_commands.status import StatusCommandDeps
from scheduler_submit.command import SubmitCommandDeps, SubmitJsonlCommandDeps
from scheduler_submit.cpu_batch import SubmitCpuBatchCommandDeps
from scheduler_submit.preflight import SubmitPreflightDeps
from scheduler_submit.task import SubmitTaskDeps
from scheduler_task.control import TaskControlDeps
from scheduler_commands.why import WhyCommandDeps


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def _dispatchable_node_configs(namespace: Mapping[str, Any]) -> dict:
    return {
        name: info
        for name, info in _ns(namespace, "NODES").items()
        if not (info or {}).get("retired")
    }


def build_submit_preflight_deps(namespace: Mapping[str, Any]) -> SubmitPreflightDeps:
    return SubmitPreflightDeps(
        submit_policy_text=_ns(namespace, "_submit_policy_text"),
        infer_checkpoint_from_submit=_ns(namespace, "_infer_checkpoint_from_submit"),
        canonical_node_name=_ns(namespace, "_canonical_node_name"),
        canonicalize_node_list=_ns(namespace, "_canonicalize_node_list"),
        simple_sac_large_data_reason=_ns(namespace, "_simple_sac_large_data_reason"),
        cpu_training_policy_reason=_ns(namespace, "_cpu_training_policy_reason"),
        task_looks_like_training=_ns(namespace, "_task_looks_like_training"),
        cmd_looks_like_training=_ns(namespace, "_cmd_looks_like_training"),
        resume_capability_reason=_ns(namespace, "_resume_capability_reason"),
        checkpoint_contract_reason=_ns(namespace, "_checkpoint_contract_reason"),
        same_declared_path=_ns(namespace, "_same_declared_path"),
    )


def build_submit_task_deps(namespace: Mapping[str, Any]) -> SubmitTaskDeps:
    time_mod = _ns(namespace, "time")
    claim_manager = _ns(namespace, "_ClaimManager")
    return SubmitTaskDeps(
        known_nodes=_dispatchable_node_configs(namespace),
        parse_env=_ns(namespace, "_parse_env"),
        task_run_identity=_ns(namespace, "_task_run_identity"),
        task_has_recorded_launch_artifacts=_ns(namespace, "_task_has_recorded_launch_artifacts"),
        recorded_launch_safety_state=_ns(namespace, "_recorded_launch_safety_state"),
        same_declared_path=_ns(namespace, "_same_declared_path"),
        active_or_unsynced_result_status=_ns(namespace, "_active_or_unsynced_result_status"),
        history_get=_ns(namespace, "history_get"),
        load_history=_ns(namespace, "load_history"),
        effective_est_vram=_ns(namespace, "_effective_est_vram"),
        effective_est_ram=_ns(namespace, "_effective_est_ram"),
        project_from_path=_ns(namespace, "_project_from_path"),
        cpu_worker_plan_for_items=_ns(namespace, "_cpu_worker_plan_for_items"),
        node_physical_cores=_ns(namespace, "_node_physical_cores"),
        cpu_labor_node_names=_ns(namespace, "_cpu_labor_node_names"),
        allocate_task_id=_ns(namespace, "_allocate_task_id"),
        local_user=_ns(namespace, "_local_user"),
        local_host_short=_ns(namespace, "_local_host_short"),
        scheduler_id=claim_manager.scheduler_id,
        apply_test_log_runtime_profile=_ns(namespace, "_apply_test_log_runtime_profile"),
        seed_pending_eta_from_history=_ns(namespace, "_seed_pending_eta_from_history"),
        now=time_mod.time,
    )


def build_submit_command_deps(namespace: Mapping[str, Any]) -> SubmitCommandDeps:
    return SubmitCommandDeps(
        split_bapr_seed_batch_submit_args=_ns(namespace, "_split_bapr_seed_batch_submit_args"),
        submit_one=_ns(namespace, "cmd_submit"),
        run_submit_preflight=_ns(namespace, "_run_submit_preflight"),
        submit_preflight_deps=_ns(namespace, "_submit_preflight_deps"),
        submit_preflight_refusal_type=_ns(namespace, "_SubmitPreflightRefusal"),
        history_record=_ns(namespace, "history_record"),
        state_lock=_ns(namespace, "state_lock"),
        load_state=_ns(namespace, "load_state"),
        build_submit_task_for_state=_ns(namespace, "_build_submit_task_for_state"),
        submit_task_deps=_ns(namespace, "_submit_task_deps"),
        save_state=_ns(namespace, "save_state"),
        submit_task_refusal_type=_ns(namespace, "_SubmitTaskRefusal"),
        default_vram_mb=_ns(namespace, "DEFAULT_VRAM_MB"),
        default_ram_mb=_ns(namespace, "DEFAULT_RAM_MB"),
        default_cpu_cores=_ns(namespace, "DEFAULT_CPU_CORES"),
        print_fn=builtins.print,
        exit_fn=_ns(namespace, "sys").exit,
    )


def build_bulk_submit_deps(namespace: Mapping[str, Any]) -> BulkSubmitDeps:
    time_mod = _ns(namespace, "time")
    claim_manager = _ns(namespace, "_ClaimManager")
    return BulkSubmitDeps(
        known_nodes=_dispatchable_node_configs(namespace),
        default_vram_mb=_ns(namespace, "DEFAULT_VRAM_MB"),
        default_ram_mb=_ns(namespace, "DEFAULT_RAM_MB"),
        default_cpu_cores=_ns(namespace, "DEFAULT_CPU_CORES"),
        canonical_node_name=_ns(namespace, "_canonical_node_name"),
        canonicalize_node_list=_ns(namespace, "_canonicalize_node_list"),
        parse_env=_ns(namespace, "_parse_env"),
        task_run_identity=_ns(namespace, "_task_run_identity"),
        history_get=_ns(namespace, "history_get"),
        project_from_path=_ns(namespace, "_project_from_path"),
        allocate_task_id=_ns(namespace, "_allocate_task_id"),
        local_user=_ns(namespace, "_local_user"),
        local_host_short=_ns(namespace, "_local_host_short"),
        scheduler_id=claim_manager.scheduler_id,
        seed_pending_eta_from_history=_ns(namespace, "_seed_pending_eta_from_history"),
        now=time_mod.time,
    )


def build_submit_jsonl_command_deps(namespace: Mapping[str, Any]) -> SubmitJsonlCommandDeps:
    return SubmitJsonlCommandDeps(
        load_submit_jsonl_specs=_ns(namespace, "_load_submit_jsonl_specs"),
        write_dispatch_intent=_ns(namespace, "_write_dispatch_intent"),
        clear_dispatch_intent=_ns(namespace, "_clear_dispatch_intent"),
        load_history=_ns(namespace, "load_history"),
        load_runtime_history=_ns(namespace, "load_runtime_history"),
        runtime_history_closest_index=_ns(namespace, "_runtime_history_closest_index"),
        state_lock=_ns(namespace, "state_lock"),
        load_state=_ns(namespace, "load_state"),
        allocate_task_ids=_ns(namespace, "_allocate_task_ids"),
        build_bulk_submit_tasks=_ns(namespace, "_build_bulk_submit_tasks"),
        save_state=_ns(namespace, "save_state"),
        dispatch_intent_ttl_s=_ns(namespace, "DISPATCH_INTENT_TTL_S"),
        json_dumps=_ns(namespace, "json").dumps,
        print_fn=builtins.print,
        exit_fn=_ns(namespace, "sys").exit,
    )


def build_submit_cpu_batch_command_deps(namespace: Mapping[str, Any]) -> SubmitCpuBatchCommandDeps:
    return SubmitCpuBatchCommandDeps(
        cpu_batch_item_counts=_ns(namespace, "_cpu_batch_item_counts"),
        parse_cpu_node_list=_ns(namespace, "_parse_cpu_node_list"),
        cpu_plan_live_node_states=_ns(namespace, "_cpu_plan_live_node_states"),
        cpu_batch_plan=_ns(namespace, "_cpu_batch_plan"),
        cpu_parallel_template_values=_ns(namespace, "_cpu_parallel_template_values"),
        rewrite_cpu_parallel_cmd=_ns(namespace, "_rewrite_cpu_parallel_cmd"),
        format_cpu_parallel_template=_ns(namespace, "_format_cpu_parallel_template"),
        print_cpu_plan=_ns(namespace, "_print_cpu_plan"),
        cpu_batch_log_payload=_ns(namespace, "_cpu_batch_log_payload"),
        notify=_ns(namespace, "notify"),
        cpu_batch_submit_spec=_ns(namespace, "_cpu_batch_submit_spec"),
        validate_cpu_batch_submit_spec=_ns(namespace, "_validate_cpu_batch_submit_spec"),
        load_history=_ns(namespace, "load_history"),
        load_runtime_history=_ns(namespace, "load_runtime_history"),
        runtime_history_closest_index=_ns(namespace, "_runtime_history_closest_index"),
        write_dispatch_intent=_ns(namespace, "_write_dispatch_intent"),
        state_lock=_ns(namespace, "state_lock"),
        load_state=_ns(namespace, "load_state"),
        validate_bulk_submit_result_conflicts=_ns(namespace, "_validate_bulk_submit_result_conflicts"),
        allocate_task_ids=_ns(namespace, "_allocate_task_ids"),
        build_bulk_submit_tasks=_ns(namespace, "_build_bulk_submit_tasks"),
        save_state=_ns(namespace, "save_state"),
        clear_dispatch_intent=_ns(namespace, "_clear_dispatch_intent"),
        dispatch_intent_ttl_s=_ns(namespace, "DISPATCH_INTENT_TTL_S"),
    )


def build_status_command_deps(namespace: Mapping[str, Any]) -> StatusCommandDeps:
    return StatusCommandDeps(
        status_readonly_lock_timeout_s=_ns(namespace, "STATUS_READONLY_LOCK_TIMEOUT_S"),
        lock_timeout_error=_ns(namespace, "SchedulerLockTimeout"),
        running_probe_snapshot_outside_lock=_ns(namespace, "_running_probe_snapshot_outside_lock"),
        eta_tail_snapshot_outside_lock=_ns(namespace, "_eta_tail_snapshot_outside_lock"),
        state_lock=_ns(namespace, "state_lock"),
        load_state=_ns(namespace, "load_state"),
        recover_stale_launching_tasks=_ns(namespace, "recover_stale_launching_tasks"),
        update_running_tasks=_ns(namespace, "update_running_tasks"),
        seed_pending_eta_from_history=_ns(namespace, "_seed_pending_eta_from_history"),
        reconcile_requeue_lineage_invariants=_ns(namespace, "reconcile_requeue_lineage_invariants"),
        save_state=_ns(namespace, "save_state"),
        compute_node_load_seconds=_ns(namespace, "compute_node_load_seconds"),
        probe_all=_ns(namespace, "probe_all"),
        apply_cpu_slot_accounting_to_nodes=_ns(namespace, "_apply_cpu_slot_accounting_to_nodes"),
        iter_node_summary_nodes=_ns(namespace, "_iter_node_summary_nodes"),
        node_display_name=_ns(namespace, "_node_display_name"),
        format_mem_gb=_ns(namespace, "_format_mem_gb"),
        node_configs=_ns(namespace, "NODES"),
        format_node_claim_summary=_ns(namespace, "_format_node_claim_summary"),
        format_node_ram_summary=_ns(namespace, "_format_node_ram_summary"),
        format_task_location=_ns(namespace, "_format_task_location"),
        format_task_vram_usage=_ns(namespace, "_format_task_vram_usage"),
        format_task_ram_usage=_ns(namespace, "_format_task_ram_usage"),
        format_task_eta=_ns(namespace, "_format_task_eta"),
        format_task_owner=_ns(namespace, "_format_task_owner"),
    )


def build_task_control_deps(namespace: Mapping[str, Any]) -> TaskControlDeps:
    time_mod = _ns(namespace, "time")
    return TaskControlDeps(
        read_dispatch_intent=_ns(namespace, "_read_dispatch_intent"),
        cancel_defers_to_dispatch_intent=_ns(namespace, "CANCEL_DEFERS_TO_DISPATCH_INTENT"),
        dispatch_intent_message=_ns(namespace, "_dispatch_intent_message"),
        state_lock=_ns(namespace, "state_lock"),
        load_state=_ns(namespace, "load_state"),
        save_state=_ns(namespace, "save_state"),
        recover_stale_launching_tasks=_ns(namespace, "recover_stale_launching_tasks"),
        release_task_claims_and_intents=_ns(namespace, "_release_task_claims_and_intents"),
        mark_user_cancelled=_ns(namespace, "_mark_user_cancelled"),
        cancel_related_queued_retries=_ns(namespace, "_cancel_related_queued_retries"),
        notify=_ns(namespace, "notify"),
        task_pids=_ns(namespace, "_task_pids"),
        record_task_kill_actor=_ns(namespace, "_record_task_kill_actor"),
        kill_task_processes=_ns(namespace, "_kill_task_processes"),
        actor_info=_ns(namespace, "_actor_info"),
        now=time_mod.time,
        write_dispatch_intent=_ns(namespace, "_write_dispatch_intent"),
        clear_dispatch_intent=_ns(namespace, "_clear_dispatch_intent"),
        kill_task_processes_batch=_ns(namespace, "_kill_task_processes_batch"),
        cancel_kill_max_workers=_ns(namespace, "CANCEL_KILL_MAX_WORKERS"),
    )


def build_adopt_command_deps(namespace: Mapping[str, Any]) -> AdoptCommandDeps:
    time_mod = _ns(namespace, "time")
    return AdoptCommandDeps(
        node_is_windows=_ns(namespace, "_node_is_windows"),
        run_on=_ns(namespace, "run_on"),
        project_from_path=_ns(namespace, "_project_from_path"),
        project_from_pid=_ns(namespace, "_project_from_pid"),
        history_get=_ns(namespace, "history_get"),
        state_lock=_ns(namespace, "state_lock"),
        load_state=_ns(namespace, "load_state"),
        allocate_task_id=_ns(namespace, "_allocate_task_id"),
        save_state=_ns(namespace, "save_state"),
        local_user=_ns(namespace, "_local_user"),
        default_vram_mb=_ns(namespace, "DEFAULT_VRAM_MB"),
        default_ram_mb=_ns(namespace, "DEFAULT_RAM_MB"),
        now=time_mod.time,
    )


def build_edit_command_deps(namespace: Mapping[str, Any]) -> EditCommandDeps:
    return EditCommandDeps(
        node_configs=_ns(namespace, "NODES"),
        canonicalize_node_list=_ns(namespace, "_canonicalize_node_list"),
        release_task_claims_and_intents=_ns(namespace, "_release_task_claims_and_intents"),
        state_lock=_ns(namespace, "state_lock"),
        load_state=_ns(namespace, "load_state"),
        save_state=_ns(namespace, "save_state"),
    )


def build_why_command_deps(namespace: Mapping[str, Any]) -> WhyCommandDeps:
    claim_manager = _ns(namespace, "_ClaimManager")
    return WhyCommandDeps(
        node_configs=_ns(namespace, "NODES"),
        vram_margin_mb=_ns(namespace, "VRAM_MARGIN_MB"),
        hpc_cpu_pool_soft_require_nodes=_ns(namespace, "_hpc_cpu_pool_soft_require_nodes"),
        blocked_nodes_for_task=_ns(namespace, "_blocked_nodes_for_task"),
        launch_failed_nodes_for_task=_ns(namespace, "_launch_failed_nodes_for_task"),
        node_resources_ok=_ns(namespace, "_node_resources_ok"),
        node_is_windows=_ns(namespace, "_node_is_windows"),
        node_cpu_fallback_block_reason=_ns(namespace, "_node_cpu_fallback_block_reason"),
        gpu_fits=_ns(namespace, "_gpu_fits"),
        algorithm_gpu_fit_block_reason=_ns(namespace, "_algorithm_gpu_fit_block_reason"),
        hard_rule_bypassed=_ns(namespace, "_hard_rule_bypassed"),
        gpu_freeze_line_mb=_ns(namespace, "_gpu_freeze_line_mb"),
        task_ignores_one_third_pack_rule=_ns(namespace, "_task_ignores_one_third_pack_rule"),
        node_gpu_util_limit=_ns(namespace, "_node_gpu_util_limit"),
        state_lock=_ns(namespace, "state_lock"),
        lock_timeout_error=_ns(namespace, "SchedulerLockTimeout"),
        load_state=_ns(namespace, "load_state"),
        load_history=_ns(namespace, "load_history"),
        probe_all=_ns(namespace, "probe_all"),
        apply_cpu_slot_accounting_to_nodes=_ns(
            namespace,
            "_apply_cpu_slot_accounting_to_nodes",
        ),
        claim_enabled_for=claim_manager.enabled_for,
        format_claim_intent_hint_for_task=_ns(namespace, "_format_claim_intent_hint_for_task"),
    )


def build_history_command_deps(namespace: Mapping[str, Any]) -> HistoryCommandDeps:
    time_mod = _ns(namespace, "time")
    return HistoryCommandDeps(
        load_history=_ns(namespace, "load_history"),
        save_history=_ns(namespace, "save_history"),
        now=time_mod.time,
        print_fn=builtins.print,
        exit_fn=_ns(namespace, "sys").exit,
    )


def build_scheduler_cli_deps(namespace: Mapping[str, Any]) -> SchedulerCliDeps:
    nodes = _dispatchable_node_configs(namespace)
    return SchedulerCliDeps(
        node_names=list(nodes.keys()),
        default_vram_mb=_ns(namespace, "DEFAULT_VRAM_MB"),
        default_ram_mb=_ns(namespace, "DEFAULT_RAM_MB"),
        default_cpu_cores=_ns(namespace, "DEFAULT_CPU_CORES"),
        dispatch_intent_ttl_s=_ns(namespace, "DISPATCH_INTENT_TTL_S"),
        resource_log_interval_s=_ns(namespace, "RESOURCE_LOG_INTERVAL_S"),
        archive_age_days=_ns(namespace, "ARCHIVE_AGE_DAYS"),
        archive_max_hot_terminal=_ns(namespace, "ARCHIVE_MAX_HOT_TERMINAL"),
        node_down_requeue_s=_ns(namespace, "NODE_DOWN_REQUEUE_S"),
        cmd_submit=_ns(namespace, "cmd_submit"),
        cmd_submit_jsonl=_ns(namespace, "cmd_submit_jsonl"),
        cmd_cpu_plan=_ns(namespace, "cmd_cpu_plan"),
        cmd_submit_cpu_batch=_ns(namespace, "cmd_submit_cpu_batch"),
        cmd_dispatch=_ns(namespace, "cmd_dispatch"),
        cmd_wait_for=_ns(namespace, "cmd_wait_for"),
        cmd_watch=_ns(namespace, "cmd_watch"),
        cmd_status=_ns(namespace, "cmd_status"),
        cmd_compact=_ns(namespace, "cmd_compact"),
        cmd_doctor=_ns(namespace, "cmd_doctor"),
        cmd_profile_local=_ns(namespace, "cmd_profile_local"),
        cmd_claims=_ns(namespace, "cmd_claims"),
        cmd_show=_ns(namespace, "cmd_show"),
        cmd_task_log=_ns(namespace, "cmd_task_log"),
        cmd_results=_ns(namespace, "cmd_results"),
        cmd_cancel=_ns(namespace, "cmd_cancel"),
        cmd_forget=_ns(namespace, "cmd_forget"),
        cmd_clear_queue=_ns(namespace, "cmd_clear_queue"),
        cmd_adopt=_ns(namespace, "cmd_adopt"),
        cmd_record_vram=_ns(namespace, "cmd_record_vram"),
        cmd_history=_ns(namespace, "cmd_history"),
        cmd_priority=_ns(namespace, "cmd_priority"),
        cmd_edit=_ns(namespace, "cmd_edit"),
        cmd_why=_ns(namespace, "cmd_why"),
        cmd_tui=_ns(namespace, "cmd_tui"),
    )
