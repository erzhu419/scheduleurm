from __future__ import annotations

import builtins

from skill.scheduler_commands.wiring import (
    build_submit_command_deps,
    build_submit_jsonl_command_deps,
)


def _same_bound_method(left, right) -> bool:
    return (
        getattr(left, "__self__", None) is getattr(right, "__self__", None)
        and getattr(left, "__func__", None) is getattr(right, "__func__", None)
    )


def _active_nodes(sch):
    return {
        name: info
        for name, info in sch.NODES.items()
        if not (info or {}).get("retired")
    }


def test_submit_wiring_uses_scheduler_namespace(sch):
    preflight = sch._submit_preflight_deps()
    assert preflight.submit_policy_text is sch._submit_policy_text
    assert preflight.infer_checkpoint_from_submit is sch._infer_checkpoint_from_submit
    assert preflight.canonicalize_node_list is sch._canonicalize_node_list
    assert preflight.same_declared_path is sch._same_declared_path

    task = sch._submit_task_deps()
    assert task.known_nodes == _active_nodes(sch)
    assert task.parse_env is sch._parse_env
    assert task.task_run_identity is sch._task_run_identity
    assert task.history_get is sch.history_get
    assert task.effective_est_vram is sch._effective_est_vram
    assert _same_bound_method(task.scheduler_id, sch._ClaimManager.scheduler_id)
    assert task.seed_pending_eta_from_history is sch._seed_pending_eta_from_history
    assert task.now is sch.time.time

    submit = build_submit_command_deps(vars(sch))
    assert submit.split_bapr_seed_batch_submit_args is sch._split_bapr_seed_batch_submit_args
    assert submit.submit_one is sch.cmd_submit
    assert submit.run_submit_preflight is sch._run_submit_preflight
    assert submit.submit_preflight_deps is sch._submit_preflight_deps
    assert submit.build_submit_task_for_state is sch._build_submit_task_for_state
    assert submit.default_vram_mb == sch.DEFAULT_VRAM_MB
    assert submit.print_fn is builtins.print
    assert submit.exit_fn is sch.sys.exit


def test_bulk_submit_and_cpu_batch_wiring_uses_scheduler_namespace(sch):
    bulk = sch._bulk_submit_deps()
    assert bulk.known_nodes == _active_nodes(sch)
    assert bulk.default_vram_mb == sch.DEFAULT_VRAM_MB
    assert bulk.canonical_node_name is sch._canonical_node_name
    assert bulk.allocate_task_id is sch._allocate_task_id
    assert _same_bound_method(bulk.scheduler_id, sch._ClaimManager.scheduler_id)
    assert bulk.now is sch.time.time

    jsonl = build_submit_jsonl_command_deps(vars(sch))
    assert jsonl.load_submit_jsonl_specs is sch._load_submit_jsonl_specs
    assert jsonl.write_dispatch_intent is sch._write_dispatch_intent
    assert jsonl.clear_dispatch_intent is sch._clear_dispatch_intent
    assert jsonl.allocate_task_ids is sch._allocate_task_ids
    assert jsonl.build_bulk_submit_tasks is sch._build_bulk_submit_tasks
    assert jsonl.dispatch_intent_ttl_s == sch.DISPATCH_INTENT_TTL_S
    assert jsonl.json_dumps is sch.json.dumps

    cpu = sch._submit_cpu_batch_command_deps()
    assert cpu.cpu_batch_item_counts is sch._cpu_batch_item_counts
    assert cpu.cpu_batch_plan is sch._cpu_batch_plan
    assert cpu.cpu_batch_submit_spec is sch._cpu_batch_submit_spec
    assert cpu.validate_bulk_submit_result_conflicts is sch._validate_bulk_submit_result_conflicts
    assert cpu.build_bulk_submit_tasks is sch._build_bulk_submit_tasks
    assert cpu.dispatch_intent_ttl_s == sch.DISPATCH_INTENT_TTL_S


def test_status_and_control_wiring_uses_scheduler_namespace(sch):
    status = sch._status_command_deps()
    assert status.status_readonly_lock_timeout_s == sch.STATUS_READONLY_LOCK_TIMEOUT_S
    assert status.lock_timeout_error is sch.SchedulerLockTimeout
    assert status.running_probe_snapshot_outside_lock is sch._running_probe_snapshot_outside_lock
    assert status.state_lock is sch.state_lock
    assert status.update_running_tasks is sch.update_running_tasks
    assert status.node_configs is sch.NODES
    assert status.format_task_eta is sch._format_task_eta

    control = sch._task_control_deps()
    assert control.read_dispatch_intent is sch._read_dispatch_intent
    assert control.cancel_defers_to_dispatch_intent == sch.CANCEL_DEFERS_TO_DISPATCH_INTENT
    assert control.state_lock is sch.state_lock
    assert control.release_task_claims_and_intents is sch._release_task_claims_and_intents
    assert control.kill_task_processes is sch._kill_task_processes
    assert control.kill_task_processes_batch is sch._kill_task_processes_batch
    assert control.write_dispatch_intent is sch._write_dispatch_intent
    assert control.clear_dispatch_intent is sch._clear_dispatch_intent
    assert control.cancel_kill_max_workers == sch.CANCEL_KILL_MAX_WORKERS
    assert control.now is sch.time.time


def test_misc_command_wiring_uses_scheduler_namespace(sch):
    adopt = sch._adopt_command_deps()
    assert adopt.node_is_windows is sch._node_is_windows
    assert adopt.run_on is sch.run_on
    assert adopt.allocate_task_id is sch._allocate_task_id
    assert adopt.default_ram_mb == sch.DEFAULT_RAM_MB

    edit = sch._edit_command_deps()
    assert edit.node_configs is sch.NODES
    assert edit.canonicalize_node_list is sch._canonicalize_node_list
    assert edit.release_task_claims_and_intents is sch._release_task_claims_and_intents

    why = sch._why_command_deps()
    assert why.node_configs is sch.NODES
    assert why.vram_margin_mb == sch.VRAM_MARGIN_MB
    assert why.node_resources_ok is sch._node_resources_ok
    assert why.gpu_fits is sch._gpu_fits
    assert _same_bound_method(why.claim_enabled_for, sch._ClaimManager.enabled_for)

    history = sch._history_command_deps()
    assert history.load_history is sch.load_history
    assert history.save_history is sch.save_history
    assert history.now is sch.time.time
    assert history.exit_fn is sch.sys.exit


def test_scheduler_cli_wiring_uses_scheduler_namespace(sch):
    cli = sch._scheduler_cli_deps()
    assert cli.node_names == list(_active_nodes(sch))
    assert cli.default_vram_mb == sch.DEFAULT_VRAM_MB
    assert cli.dispatch_intent_ttl_s == sch.DISPATCH_INTENT_TTL_S
    assert cli.cmd_submit is sch.cmd_submit
    assert cli.cmd_submit_jsonl is sch.cmd_submit_jsonl
    assert cli.cmd_submit_cpu_batch is sch.cmd_submit_cpu_batch
    assert cli.cmd_dispatch is sch.cmd_dispatch
    assert cli.cmd_watch is sch.cmd_watch
    assert cli.cmd_status is sch.cmd_status
    assert cli.cmd_cancel is sch.cmd_cancel
    assert cli.cmd_edit is sch.cmd_edit
    assert cli.cmd_why is sch.cmd_why
