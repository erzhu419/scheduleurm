from __future__ import annotations

from types import SimpleNamespace

from skill.scheduler_dispatch_command import DispatchCommandDeps, run_dispatch_command


class _Lock:
    def __init__(self, calls, label):
        self.calls = calls
        self.label = label

    def __enter__(self):
        self.calls.append(("lock_enter", self.label))
        return self

    def __exit__(self, *args):
        self.calls.append(("lock_exit", self.label))
        return False


def _deps(
    *,
    calls=None,
    states=None,
    dispatch_events=None,
    qcount=0,
    execute_events=None,
    fail_preflight=None,
    recover_count=0,
):
    calls = calls if calls is not None else []
    states = list(states or [{"tasks": []}, {"tasks": []}])
    dispatch_events = list(dispatch_events or [])
    execute_events = execute_events if execute_events is not None else dispatch_events
    fail_preflight = fail_preflight or set()

    def state_lock(**kwargs):
        calls.append(("state_lock", kwargs))
        return _Lock(calls, kwargs.get("purpose"))

    def load_state():
        calls.append(("load_state",))
        return states.pop(0) if states else {"tasks": []}

    def save_state(state):
        calls.append(("save_state", state))

    def notify(event, payload, **kwargs):
        calls.append(("notify", event, payload, kwargs))

    def maybe_fail(name):
        if name in fail_preflight:
            raise RuntimeError(f"{name} failed")
        calls.append((name,))

    def preload(**kwargs):
        if "preload" in fail_preflight:
            raise RuntimeError("preload failed")
        calls.append(("preload", kwargs))

    def stage_launch(**kwargs):
        if "stage_launch" in fail_preflight:
            raise RuntimeError("stage_launch failed")
        calls.append(("stage_launch", kwargs))

    def update_running(state, **kwargs):
        calls.append(("update_running", state, kwargs))

    def do_dispatch(state, nodes, **kwargs):
        calls.append(("do_dispatch", state, nodes, kwargs))
        return list(dispatch_events), qcount

    def execute(events, **kwargs):
        calls.append(("execute", events, kwargs))
        return list(execute_events)

    return DispatchCommandDeps(
        set_hard_rule_mode=lambda mode: calls.append(("hard_rule", mode)),
        configure_algorithm=lambda name: calls.append(("algorithm", name)),
        state_lock=state_lock,
        load_state=load_state,
        save_state=save_state,
        recover_stale_launching_tasks_outside_lock=lambda **kwargs: (
            calls.append(("recover_outside", kwargs)) or recover_count
        ),
        notify=notify,
        preload_docker_images_outside_lock=preload,
        stage_migration_candidates_outside_lock=lambda: maybe_fail("stage_migration"),
        stage_launch_candidates_outside_lock=stage_launch,
        sync_completed_results_outside_lock=lambda: maybe_fail("result_sync"),
        running_probe_snapshot_outside_lock=lambda purpose: (calls.append(("running_probe", purpose)) or ({"r": 1}, {"tr"})),
        eta_tail_snapshot_outside_lock=lambda purpose: (calls.append(("eta_tail", purpose)) or ({"e": 1}, {"te"})),
        probe_all=lambda: (calls.append(("probe_all",)) or [{"name": "n"}]),
        probe_all_cached=lambda **kwargs: (
            calls.append(("probe_all_cached", kwargs)) or [{"name": "n-cached"}]
        ),
        prefetch_resume_scans_outside_lock=lambda nodes, **kwargs: calls.append(
            ("prefetch_resume", nodes, kwargs)
        ),
        dispatch_probe_cache_max_age_s=30,
        update_running_tasks=update_running,
        seed_pending_eta_from_history=lambda state: calls.append(("seed_eta", state)),
        remember_running_resource_snapshots=lambda state, nodes: calls.append(("remember_resources", state, nodes)),
        reconcile_aggregate_only_vram=lambda state, nodes: calls.append(("reconcile_vram", state, nodes)),
        reserve_inflight_vram=lambda state, nodes: calls.append(("reserve_vram", state, nodes)),
        print_node_summary=lambda nodes: calls.append(("print_nodes", nodes)),
        do_dispatch=do_dispatch,
        execute_deferred_launches=execute,
        load_state_shared_snapshot=lambda purpose: (calls.append(("snapshot", purpose)) or {"tasks": ["post"]}),
        dispatch_cycle_log_payload=lambda state, nodes, events, queued: {
            "state": state,
            "nodes": nodes,
            "events": events,
            "queued": queued,
        },
        format_task_location=lambda task: f"{task.get('node')}:{task.get('gpu_idx')}",
        format_mem_gb=lambda mb: f"{mb}MB",
        max_launch_retry=3,
        defer_launch_outside_lock=True,
        print_fn=lambda *args: calls.append(("print", " ".join(str(arg) for arg in args))),
    )


def _args(**overrides):
    base = {
        "hard_rule_mode": None,
        "algorithm": None,
        "dispatch_task_ids": [],
        "lock_timeout": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_dispatch_command_runs_lock_light_prefetch_then_main_dispatch():
    calls = []
    states = [{"tasks": [], "recover": 1}, {"tasks": [], "recover": 2}]

    run_dispatch_command(
        _args(hard_rule_mode="off", algorithm="legacy"),
        deps=_deps(calls=calls, states=states, recover_count=1),
    )

    assert ("hard_rule", "off") in calls
    assert ("algorithm", "legacy") in calls
    assert ("preload", {"task_ids": None}) in calls
    assert ("stage_migration",) in calls
    assert ("stage_launch", {"task_ids": None}) in calls
    assert ("result_sync",) in calls
    assert (
        "recover_outside",
        {"purpose": "dispatch:recover-launching", "lock_timeout": None},
    ) in calls
    assert ("running_probe", "dispatch:running-probe") in calls
    assert ("eta_tail", "dispatch:eta-tail") in calls
    assert ("probe_all",) in calls
    assert ("prefetch_resume", [{"name": "n"}], {"task_ids": None}) in calls
    names = [call[0] for call in calls]
    assert names.index("prefetch_resume") < names.index("stage_launch")
    assert any(call[0] == "do_dispatch" and call[3]["defer_launches"] is True for call in calls)
    assert any(call[0] == "notify" and call[1] == "launching_state_recovered" for call in calls)
    assert any(call[0] == "notify" and call[1] == "dispatch_cycle" for call in calls)
    assert ("print", "=== dispatch === (nothing queued)") in calls


def test_targeted_dispatch_skips_migration_and_result_sync():
    calls = []

    run_dispatch_command(
        _args(dispatch_task_ids=["t1", ""]),
        deps=_deps(calls=calls, qcount=1),
    )

    assert ("preload", {"task_ids": {"t1"}}) in calls
    assert ("stage_launch", {"task_ids": {"t1"}}) in calls
    assert not any(call[0] == "stage_migration" for call in calls)
    assert not any(call[0] == "result_sync" for call in calls)
    assert not any(call[0] == "running_probe" for call in calls)
    assert not any(call[0] == "eta_tail" for call in calls)
    assert not any(call[0] == "update_running" for call in calls)
    assert ("probe_all_cached", {"max_age_s": 30}) in calls
    assert (
        "prefetch_resume",
        [{"name": "n-cached"}],
        {"task_ids": {"t1"}},
    ) in calls
    names = [call[0] for call in calls]
    assert names.index("prefetch_resume") < names.index("stage_launch")
    assert not any(call[0] == "probe_all" for call in calls)
    assert any(call[0] == "do_dispatch" and call[3]["target_task_ids"] == {"t1"} for call in calls)


def test_preflight_errors_notify_and_continue_to_dispatch():
    calls = []

    run_dispatch_command(
        _args(),
        deps=_deps(calls=calls, fail_preflight={"preload", "stage_migration", "stage_launch", "result_sync"}),
    )

    notified = [call[1] for call in calls if call[0] == "notify"]
    assert "preload_error" in notified
    assert "migration_staging_error_outer" in notified
    assert "launch_staging_error_outer" in notified
    assert "result_sync_error_outer" in notified
    assert any(call[0] == "do_dispatch" for call in calls)


def test_deferred_launches_reload_post_launch_snapshot_for_dispatch_payload():
    calls = []
    dispatch_events = [{"type": "launch_intent", "task_id": "t1"}]
    execute_events = [{"type": "launched", "task_id": "t1", "task": {"node": "n", "gpu_idx": 0, "log_path": "/tmp/log"}, "msg": "pid=1"}]

    run_dispatch_command(
        _args(),
        deps=_deps(
            calls=calls,
            dispatch_events=dispatch_events,
            execute_events=execute_events,
            qcount=1,
        ),
    )

    assert ("snapshot", "dispatch:post-launch-snapshot") in calls
    dispatch_notify = next(call for call in calls if call[0] == "notify" and call[1] == "dispatch_cycle")
    assert dispatch_notify[2]["state"] == {"tasks": ["post"]}
    assert ("print", "  [t1] LAUNCH on n:0  pid=1  log=/tmp/log") in calls


def test_dispatch_executes_deferred_launches_after_main_state_lock_exits():
    calls = []
    dispatch_events = [{"type": "launch_intent", "task_id": "t1"}]

    run_dispatch_command(
        _args(),
        deps=_deps(calls=calls, dispatch_events=dispatch_events, qcount=1),
    )

    execute_idx = next(i for i, call in enumerate(calls) if call[0] == "execute")
    main_exit_idx = next(
        i
        for i, call in enumerate(calls)
        if call == ("lock_exit", "dispatch:main")
    )
    assert main_exit_idx < execute_idx


def test_empty_queue_global_plan_event_is_printed_without_task_id():
    calls = []
    events = [
        {
            "type": "algorithm_global_batch_plan",
            "algorithm": "global_theorem_maxweight_v1",
            "candidate_tasks": 0,
            "planned_tasks": 0,
            "scheduler_hook_ready": False,
        }
    ]

    run_dispatch_command(
        _args(),
        deps=_deps(calls=calls, dispatch_events=events, execute_events=events),
    )

    printed = [call[1] for call in calls if call[0] == "print"]
    assert printed[-2:] == [
        "=== dispatch === (0 queued)",
        "  [algorithm] GLOBAL-PLAN global_theorem_maxweight_v1: "
        "0/0 task(s), hook=not-ready",
    ]


def test_global_plan_event_is_printed_before_legacy_task_event():
    calls = []
    events = [
        {
            "type": "algorithm_global_batch_plan",
            "algorithm": "global_theorem_maxweight_v1",
            "candidate_tasks": 2,
            "planned_tasks": 1,
            "scheduler_hook_ready": True,
        },
        {
            "type": "no_fit",
            "task_id": "t1",
            "task": {"last_block_reason": "no ram", "description": "desc"},
        },
    ]

    run_dispatch_command(
        _args(),
        deps=_deps(calls=calls, dispatch_events=events, execute_events=events, qcount=2),
    )

    printed = [call[1] for call in calls if call[0] == "print"]
    assert printed[-3:] == [
        "=== dispatch === (2 queued)",
        "  [algorithm] GLOBAL-PLAN global_theorem_maxweight_v1: "
        "1/2 task(s), hook=ready",
        "  [t1] WAIT  no ram (desc)",
    ]


def test_global_plan_error_is_printed_before_legacy_task_event():
    calls = []
    events = [
        {
            "type": "algorithm_global_batch_plan_error",
            "algorithm": "global_theorem_maxweight_v1",
            "reason": "oracle unavailable",
        },
        {"type": "blocked", "task_id": "t2", "reason": "dependency pending"},
    ]

    run_dispatch_command(
        _args(),
        deps=_deps(calls=calls, dispatch_events=events, execute_events=events, qcount=1),
    )

    printed = [call[1] for call in calls if call[0] == "print"]
    assert printed[-3:] == [
        "=== dispatch === (1 queued)",
        "  [algorithm] GLOBAL-PLAN-ERROR global_theorem_maxweight_v1: "
        "oracle unavailable",
        "  [t2] BLOCK dependency pending",
    ]


def test_global_rejection_and_execution_events_do_not_require_task_id():
    calls = []
    events = [
        {
            "type": "algorithm_global_batch_plan_rejected",
            "algorithm": "global_theorem_maxweight_v1",
            "reason": "missing cache certificate",
        },
        {
            "type": "algorithm_global_batch_execution",
            "planned_task_ids": ["t1", "t2"],
            "executed_task_ids": ["t1"],
            "unresolved_task_ids": ["t2"],
            "exact_execution_ready": False,
        },
    ]

    run_dispatch_command(
        _args(),
        deps=_deps(calls=calls, dispatch_events=events, execute_events=events, qcount=2),
    )

    printed = [call[1] for call in calls if call[0] == "print"]
    assert printed[-3:] == [
        "=== dispatch === (2 queued)",
        "  [algorithm] GLOBAL-PLAN-REJECTED global_theorem_maxweight_v1: "
        "missing cache certificate",
        "  [algorithm] GLOBAL-EXECUTION 1/2 task(s), status=incomplete",
    ]


def test_dispatch_event_printing_covers_common_events():
    calls = []
    events = [
        {"type": "lineage_repaired", "count": 2},
        {"type": "no_fit", "task_id": "t1", "task": {"last_block_reason": "no ram", "description": "desc"}},
        {"type": "blocked", "task_id": "t2", "reason": "blocked"},
        {"type": "resume_found", "task_id": "t3", "resume_from": "/ckpt"},
        {"type": "launch_failed_retry", "task_id": "t4", "task": {"launch_fail_count": 1}, "error": "bad"},
        {"type": "launch_failed_terminal", "task_id": "t5", "error": "perm"},
        {"type": "migrated", "task_id": "t6", "from_node": "a", "to_node": "b", "eta_seconds": 9},
        {"type": "preempted", "task_id": "t7", "freed_node": "n", "cpu_freed": 2, "ram_freed": 1024},
        {"type": "claim_race", "task_id": "t8", "reason": "busy"},
        {"type": "launch_commit_skipped", "task_id": "t9", "reason": "changed"},
    ]

    run_dispatch_command(_args(), deps=_deps(calls=calls, dispatch_events=events, execute_events=events, qcount=9))

    printed = [call[1] for call in calls if call[0] == "print"]
    assert any("[lineage] repaired 2" in line for line in printed)
    assert any("[t1] WAIT" in line for line in printed)
    assert any("[t6] MIGRATE a → b" in line for line in printed)
    assert any("[t8] CLAIM-RACE busy" in line for line in printed)
