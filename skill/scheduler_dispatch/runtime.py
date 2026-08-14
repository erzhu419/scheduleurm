"""Runtime-bound dispatch-loop and deferred-launch wrappers for scheduler.py."""

from __future__ import annotations

from typing import Any, Mapping, Optional


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_dispatch_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    def _is_legacy_external_managed(task: dict) -> bool:
        return bool(task.get("slurm_job_id"))

    def _is_slurm_managed(task: dict) -> bool:
        return _is_legacy_external_managed(task)

    def _counts_against_node_concurrency(task: dict) -> bool:
        if task.get("status") not in ("running", "launching"):
            return False
        if not task.get("node"):
            return False
        return True

    def _reserve_inflight_vram(state, nodes):
        return _ns(namespace, "_reserve_inflight_vram_impl")(
            state,
            nodes,
            deps=_ns(namespace, "_InflightVramReservationDeps")(
                node_configs=_ns(namespace, "NODES"),
                startup_floor_mb=_ns(namespace, "STARTUP_FLOOR_MB"),
                is_slurm_managed=_ns(namespace, "_is_legacy_external_managed"),
                hard_rule_bypassed=_ns(namespace, "_hard_rule_bypassed"),
            ),
        )

    def _signature_batch_key(signature):
        return _ns(namespace, "_signature_batch_key_impl")(signature)

    def _detect_batch_completions(state, transitioned_ids):
        return _ns(namespace, "_detect_batch_completions_impl")(state, transitioned_ids)

    def _enforce_post_dispatch_thresholds(state, nodes):
        return _ns(namespace, "_enforce_post_dispatch_thresholds_impl")(
            state,
            nodes,
            deps=_ns(namespace, "_post_dispatch_threshold_deps")(),
        )

    def _post_dispatch_threshold_deps():
        return _ns(namespace, "_build_post_dispatch_threshold_deps")(namespace)

    def _preemption_deps():
        return _ns(namespace, "_build_preemption_deps")(namespace)

    def _eviction_deps():
        return _ns(namespace, "_build_eviction_deps")(namespace)

    def _evict_to_queue(victim, state, reason, eviction_kind: str = "preempt"):
        return _ns(namespace, "_evict_to_queue_impl")(
            victim,
            state,
            reason,
            eviction_kind,
            deps=_ns(namespace, "_eviction_deps")(),
        )

    def _eviction_cooldown_block_reason(task: dict, now: Optional[float] = None) -> str:
        return _ns(namespace, "_eviction_cooldown_block_reason_impl")(
            task,
            now=now,
            deps=_ns(namespace, "_eviction_deps")(),
        )

    def _evict_node_cooldown_block_reason(
        task: dict,
        node: str,
        now: Optional[float] = None,
        node_state: Optional[dict] = None,
    ) -> str:
        return _ns(namespace, "_evict_node_cooldown_block_reason_impl")(
            task,
            node,
            now=now,
            node_state=node_state,
            deps=_ns(namespace, "_eviction_deps")(),
        )

    def _preempt_for_high_priority(state, nodes):
        return _ns(namespace, "_preempt_for_high_priority_impl")(
            state,
            nodes,
            deps=_ns(namespace, "_preemption_deps")(),
        )

    def _new_launch_token() -> str:
        time_mod = _ns(namespace, "time")
        os_mod = _ns(namespace, "os")
        threading_mod = _ns(namespace, "threading")
        return f"{time_mod.time_ns()}:{os_mod.getpid()}:{threading_mod.get_ident()}"

    def _debit_node_resources_for_launch(nodes: list, task: dict) -> None:
        for node_state in nodes:
            if node_state["name"] != task.get("node"):
                continue
            node_info = _ns(namespace, "NODES").get(node_state["name"], {}) or {}
            cpu_debit = (
                0 if (
                    node_state["name"] == "local"
                    and _ns(namespace, "_task_is_gpu_capacity_task")(task)
                )
                else task.get("cpu_cores", _ns(namespace, "DEFAULT_CPU_CORES"))
            )
            node_state["free_cpu"] = max(0, node_state.get("free_cpu", 0) - cpu_debit)
            if "cpu_slot_reserved" in node_state:
                node_state["cpu_slot_reserved"] = int(
                    node_state.get("cpu_slot_reserved") or 0
                ) + int(cpu_debit or 0)
                if "cpu_slot_free" in node_state:
                    node_state["cpu_slot_free"] = max(
                        0,
                        int(node_state.get("cpu_slot_free") or 0) - int(cpu_debit or 0),
                    )
                node_state["cpu_slot_overcommitted"] = max(
                    0,
                    int(node_state.get("cpu_slot_reserved") or 0)
                    - int(node_state.get("total_cpu") or 0),
                )
            if "cpu_hard_free" in node_state:
                node_state["cpu_hard_reserved"] = int(
                    node_state.get("cpu_hard_reserved") or 0
                ) + int(cpu_debit or 0)
                node_state["cpu_startup_declared_reserved"] = int(
                    node_state.get("cpu_startup_declared_reserved") or 0
                ) + int(cpu_debit or 0)
                node_state["cpu_hard_free"] = max(
                    0,
                    int(node_state.get("cpu_hard_free") or 0) - int(cpu_debit or 0),
                )
                if int(cpu_debit or 0) > 0:
                    if int(cpu_debit or 0) <= 1:
                        configured_grace = node_info.get(
                            "live_cpu_backfill_single_thread_startup_grace_s"
                        )
                        if configured_grace is None:
                            configured_grace = min(
                                10,
                                int(node_info.get(
                                    "live_cpu_backfill_startup_grace_s", 300
                                )),
                            )
                    else:
                        configured_grace = node_info.get(
                            "live_cpu_backfill_startup_grace_s"
                        )
                    startup_grace_s = max(
                        0,
                        int(300 if configured_grace is None else configured_grace),
                    )
                    node_state["cpu_startup_hold_remaining_s"] = max(
                        int(node_state.get("cpu_startup_hold_remaining_s") or 0),
                        startup_grace_s,
                    )
            # Do not mutate observed_free_cpu here. It is probe telemetry, while
            # free_cpu/cpu_hard_free above are the in-wave admission budgets.
            node_state["free_ram_mb"] = max(
                0,
                node_state.get("free_ram_mb", 0) - task.get("ram_mb", _ns(namespace, "DEFAULT_RAM_MB")),
            )
            node_state["running_count"] = node_state.get("running_count", 0) + 1
            if int(task.get("est_vram_mb") or 0) > 0:
                thread_reserve = max(0, int(
                    node_info.get("gpu_launch_user_thread_reserve") or 0))
            elif _ns(namespace, "_task_is_thread_heavy_jax_cpu_impl")(task):
                thread_reserve = max(0, int(
                    node_info.get("jax_cpu_launch_user_thread_reserve") or 0))
            else:
                thread_reserve = 0
            if thread_reserve:
                if node_state.get("user_threads") is not None:
                    node_state["user_threads"] = (
                        int(node_state.get("user_threads") or 0) + thread_reserve)
                if node_state.get("free_user_threads") is not None:
                    node_state["free_user_threads"] = max(
                        0,
                        int(node_state.get("free_user_threads") or 0)
                        - thread_reserve,
                    )
            if task.get("gpu_idx") is None:
                continue
            for gpu in node_state["gpus"]:
                if gpu["idx"] != task["gpu_idx"]:
                    continue
                gpu["used_mb"] += task["est_vram_mb"]
                gpu["free_mb"] = max(0, gpu["free_mb"] - task["est_vram_mb"])
                gpu["running_task_count"] = int(gpu.get("running_task_count") or 0) + 1

    def _apply_launch_result_to_task(task: dict, ok: bool, msg: str, events: list) -> bool:
        time_mod = _ns(namespace, "time")
        return _ns(namespace, "_apply_launch_result_to_task_impl")(
            task,
            ok,
            msg,
            events,
            deps=_ns(namespace, "_LaunchResultDeps")(
                max_launch_retry=_ns(namespace, "MAX_LAUNCH_RETRY"),
                remember_claim_intent=_ns(namespace, "_remember_claim_intent"),
                release_task_claims_and_intents=_ns(namespace, "_release_task_claims_and_intents"),
                write_escalation=_ns(namespace, "_write_escalation"),
                clear_claim_intent_markers=_ns(namespace, "_clear_claim_intent_markers"),
                now=time_mod.time,
            ),
        )

    def _dispatch_accounting_deps():
        return _ns(namespace, "_build_dispatch_accounting_deps")(namespace)

    def _resource_estimate_deps():
        return _ns(namespace, "_build_resource_estimate_deps")(namespace)

    def _dispatch_launch_staging_deps():
        return _ns(namespace, "_build_dispatch_launch_staging_deps")(namespace)

    def _dispatch_launch_execution_deps():
        return _ns(namespace, "_build_dispatch_launch_execution_deps")(namespace)

    def _dispatch_placement_apply_deps():
        return _ns(namespace, "_build_dispatch_placement_apply_deps")(namespace)

    def _dispatch_resume_placement_deps():
        return _ns(namespace, "_build_dispatch_resume_placement_deps")(namespace)

    def _dispatch_task_gate_deps():
        return _ns(namespace, "_build_dispatch_task_gate_deps")(namespace)

    def _dispatch_loop_deps():
        return _ns(namespace, "_build_dispatch_loop_deps")(namespace)

    def _watch_state_phase_deps():
        return _ns(namespace, "_build_watch_state_phase_deps")(namespace)

    def _do_dispatch(
        state,
        nodes,
        target_task_ids: Optional[set] = None,
        defer_launches: bool = False,
        max_queued: Optional[int] = None,
    ):
        return _ns(namespace, "_do_dispatch_impl")(
            state,
            nodes,
            target_task_ids=target_task_ids,
            defer_launches=defer_launches,
            max_queued=max_queued,
            deps=_ns(namespace, "_dispatch_loop_deps")(),
        )

    def _launch_commit_deps():
        return _ns(namespace, "_build_launch_commit_deps")(namespace)

    def _commit_deferred_launch_result(
        intent: dict,
        launched_task: dict,
        ok: bool,
        msg: str,
        lock_timeout=None,
        mark_notified_launch: bool = False,
    ) -> list:
        return _ns(namespace, "_commit_deferred_launch_result_impl")(
            intent,
            launched_task,
            ok,
            msg,
            lock_timeout=lock_timeout,
            mark_notified_launch=mark_notified_launch,
            deps=_ns(namespace, "_launch_commit_deps")(),
        )

    def _commit_deferred_launch_results_batch(
        records: list[dict],
        lock_timeout=None,
        mark_notified_launch: bool = False,
    ) -> dict[int, list]:
        return _ns(namespace, "_commit_deferred_launch_results_batch_impl")(
            records,
            lock_timeout=lock_timeout,
            mark_notified_launch=mark_notified_launch,
            deps=_ns(namespace, "_launch_commit_deps")(),
        )

    def _execute_deferred_launches(
        events: list,
        lock_timeout=None,
        mark_notified_launch: bool = False,
    ) -> list:
        return _ns(namespace, "_execute_deferred_launches_impl")(
            events,
            lock_timeout=lock_timeout,
            mark_notified_launch=mark_notified_launch,
            deps=_ns(namespace, "_DeferredLaunchDeps")(
                max_workers=_ns(namespace, "LAUNCH_EXEC_MAX_WORKERS"),
                launch=_ns(namespace, "launch"),
                launch_exec_slot_lock=lambda node: _ns(namespace, "launch_exec_slot_lock")(
                    node,
                    purpose="launch-exec",
                ),
                execute_launch_intents=_ns(namespace, "_execute_launch_intents_impl"),
                notify=_ns(namespace, "notify"),
                commit_deps=_ns(namespace, "_launch_commit_deps")(),
            ),
        )

    def _load_state_shared_snapshot(purpose: str) -> dict:
        with _ns(namespace, "state_lock")(shared=True, purpose=purpose):
            return _ns(namespace, "load_state")()

    return {
        "_is_legacy_external_managed": _is_legacy_external_managed,
        "_is_slurm_managed": _is_slurm_managed,
        "_counts_against_node_concurrency": _counts_against_node_concurrency,
        "_reserve_inflight_vram": _reserve_inflight_vram,
        "_signature_batch_key": _signature_batch_key,
        "_detect_batch_completions": _detect_batch_completions,
        "_enforce_post_dispatch_thresholds": _enforce_post_dispatch_thresholds,
        "_post_dispatch_threshold_deps": _post_dispatch_threshold_deps,
        "_preemption_deps": _preemption_deps,
        "_eviction_deps": _eviction_deps,
        "_evict_to_queue": _evict_to_queue,
        "_eviction_cooldown_block_reason": _eviction_cooldown_block_reason,
        "_evict_node_cooldown_block_reason": _evict_node_cooldown_block_reason,
        "_preempt_for_high_priority": _preempt_for_high_priority,
        "_new_launch_token": _new_launch_token,
        "_debit_node_resources_for_launch": _debit_node_resources_for_launch,
        "_apply_launch_result_to_task": _apply_launch_result_to_task,
        "_dispatch_accounting_deps": _dispatch_accounting_deps,
        "_resource_estimate_deps": _resource_estimate_deps,
        "_dispatch_launch_staging_deps": _dispatch_launch_staging_deps,
        "_dispatch_launch_execution_deps": _dispatch_launch_execution_deps,
        "_dispatch_placement_apply_deps": _dispatch_placement_apply_deps,
        "_dispatch_resume_placement_deps": _dispatch_resume_placement_deps,
        "_dispatch_task_gate_deps": _dispatch_task_gate_deps,
        "_dispatch_loop_deps": _dispatch_loop_deps,
        "_watch_state_phase_deps": _watch_state_phase_deps,
        "_do_dispatch": _do_dispatch,
        "_launch_commit_deps": _launch_commit_deps,
        "_commit_deferred_launch_result": _commit_deferred_launch_result,
        "_commit_deferred_launch_results_batch": _commit_deferred_launch_results_batch,
        "_execute_deferred_launches": _execute_deferred_launches,
        "_load_state_shared_snapshot": _load_state_shared_snapshot,
    }
