from __future__ import annotations

from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass, field
import threading
import time

from skill.scheduler_launch.staging_runner import (
    _BACKGROUND_ACTIVE_KEYS,
    _BACKGROUND_LANES,
    _background_lane_done,
    LaunchStagingRunnerDeps,
    stage_launch_candidates_outside_lock,
)
from skill.scheduler_watch.shutdown import WatcherShutdownRequested


@dataclass
class _Plan:
    protected_under_cwd: dict
    candidates: list
    ckpt_candidates: list
    input_candidates: list = field(default_factory=list)


class _Lock:
    def __init__(self, calls):
        self.calls = calls

    def __enter__(self):
        self.calls.append(("lock_enter",))
        return self

    def __exit__(self, *args):
        self.calls.append(("lock_exit",))
        return False


def _deps(
    *,
    plan=None,
    calls=None,
    staging_fails=None,
    max_candidates=0,
    cwd_results=None,
    ckpt_results=None,
    input_results=None,
    cache_hits=None,
    snapshot_error=False,
    guard_acquired=True,
    worker_count=1,
    cwd_hook=None,
):
    calls = calls if calls is not None else []
    staging_fails = staging_fails if staging_fails is not None else {}
    cwd_results = cwd_results or {}
    ckpt_results = ckpt_results or {}
    input_results = input_results or {}
    cache_hits = set(cache_hits or [])
    plan = plan or _Plan({}, [], [])

    def state_lock(**kwargs):
        calls.append(("state_lock", kwargs))
        if snapshot_error:
            raise RuntimeError("lock failed")
        return _Lock(calls)

    def collect(state, task_ids):
        calls.append(("collect", state, task_ids))
        return plan

    def stage_cwd(task, target_node, **kwargs):
        calls.append(("stage_cwd", target_node, task.get("cwd"), kwargs))
        if cwd_hook is not None:
            return cwd_hook(task, target_node, **kwargs)
        result = cwd_results.get((target_node, task.get("cwd")), (True, "ok"))
        if isinstance(result, BaseException):
            raise result
        return result

    def stage_ckpt(task, target_node, source_location):
        calls.append(("stage_ckpt", task.get("id"), target_node, source_location.get("node")))
        result = ckpt_results.get((task.get("id"), target_node), (True, "ok"))
        if isinstance(result, BaseException):
            raise result
        return result

    def stage_inputs(task, target_node):
        calls.append(("stage_inputs", task.get("id"), target_node))
        return input_results.get((task.get("id"), target_node), (True, "ok"))

    def ckpt_key(source, target, ckpt_dir, source_location=None):
        return ("ckpt", source, target, ckpt_dir)

    def notify(event, payload, **kwargs):
        calls.append(("notify", event, payload, kwargs))

    @contextmanager
    def staging_key_guard(key):
        calls.append(("staging_key_guard", key))
        yield guard_acquired

    return LaunchStagingRunnerDeps(
        state_lock=state_lock,
        load_state=lambda: {"tasks": [{"id": "queued"}]},
        collect_launch_staging_plan=collect,
        launch_stage_candidate_key=lambda item: (item[0], item[1]),
        max_candidates_per_pass=max_candidates,
        worker_count=worker_count,
        stage_cwd_for_launch=stage_cwd,
        stage_input_paths_for_launch=stage_inputs,
        launch_input_stage_key=lambda task, target: (
            "launch_input", target, tuple(task.get("stage_input_paths") or [])),
        stage_resume_ckpt_for_launch=stage_ckpt,
        resume_ckpt_stage_key=ckpt_key,
        staging_cache_hit=lambda key: key in cache_hits,
        staging_fails=staging_fails,
        notify=notify,
        now=lambda: 123.0,
        staging_key_guard=staging_key_guard,
    )


def test_no_candidates_only_takes_snapshot_lock():
    calls = []

    stage_launch_candidates_outside_lock({"t1"}, deps=_deps(calls=calls))

    assert ("state_lock", {"shared": True, "purpose": "watch:post-reboot-snapshot"}) in calls
    assert ("lock_enter",) in calls
    assert ("lock_exit",) in calls
    assert not any(call[0] == "stage_cwd" for call in calls)


def test_stage_cwd_candidates_are_sorted_limited_and_get_extra_excludes():
    calls = []
    plan = _Plan(
        protected_under_cwd={"/work/a": {"ckpt", "results"}},
        candidates=[("nodeB", "/work/b"), ("nodeA", "/work/a")],
        ckpt_candidates=[],
    )

    stage_launch_candidates_outside_lock(
        deps=_deps(plan=plan, calls=calls, max_candidates=1),
    )

    assert ("stage_cwd", "nodeA", "/work/a", {"extra_excludes": ["ckpt", "results"]}) in calls
    assert not any(call[0] == "stage_cwd" and call[1] == "nodeB" for call in calls)


def test_cwd_cap_exceeded_is_not_recorded_as_transport_failure():
    calls = []
    fails = {}
    plan = _Plan({}, [("nodeA", "/work/a")], [])

    stage_launch_candidates_outside_lock(
        deps=_deps(
            plan=plan,
            calls=calls,
            staging_fails=fails,
            cwd_results={("nodeA", "/work/a"): (False, "CAP_EXCEEDED: too large")},
        ),
    )

    assert fails == {}
    assert not any(call[0] == "notify" for call in calls)


def test_cwd_failure_and_exception_are_recorded_and_notified():
    calls = []
    fails = {}
    plan = _Plan({}, [("nodeA", "/work/a"), ("nodeB", "/work/b")], [])

    stage_launch_candidates_outside_lock(
        deps=_deps(
            plan=plan,
            calls=calls,
            staging_fails=fails,
            cwd_results={
                ("nodeA", "/work/a"): (False, "ssh failed"),
                ("nodeB", "/work/b"): RuntimeError("boom"),
            },
        ),
    )

    assert fails[("local", "nodeA", "/work/a")] == (123.0, "ssh failed")
    assert fails[("local", "nodeB", "/work/b")] == (123.0, "exception: boom")
    assert any(call[0] == "notify" and call[1] == "launch_staging_failed" for call in calls)
    assert any(call[0] == "notify" and call[1] == "launch_staging_exception" for call in calls)


def test_checkpoint_cache_hit_skips_stage():
    calls = []
    task = {"id": "t1", "ckpt_dir": "/work/ckpt"}
    src = {"node": "local"}
    key = ("ckpt", "local", "nodeA", "/work/ckpt")
    plan = _Plan({}, [], [(task, "nodeA", src)])

    stage_launch_candidates_outside_lock(
        deps=_deps(plan=plan, calls=calls, cache_hits={key}),
    )

    assert not any(call[0] == "stage_ckpt" for call in calls)


def test_checkpoint_stage_precedes_cwd_on_the_same_node():
    calls = []
    task = {"id": "t1", "ckpt_dir": "/work/ckpt", "cwd": "/work/repo"}
    plan = _Plan(
        {},
        [("nodeA", "/work/repo")],
        [(task, "nodeA", {"node": "local"})],
    )

    stage_launch_candidates_outside_lock(
        deps=_deps(plan=plan, calls=calls, worker_count=2),
    )

    kinds = [call[0] for call in calls if call[0] in {"stage_ckpt", "stage_cwd"}]
    assert kinds == ["stage_ckpt", "stage_cwd"]


def test_task_inputs_stage_on_the_selected_target_only():
    calls = []
    plan = _Plan(
        {}, [], [],
        [({"id": "t-input", "stage_input_paths": ["/work/input"]}, "nodeB")],
    )

    stage_launch_candidates_outside_lock(deps=_deps(plan=plan, calls=calls))

    assert ("stage_inputs", "t-input", "nodeB") in calls


def test_busy_cross_process_guard_deduplicates_input_stage():
    calls = []
    plan = _Plan(
        {}, [], [],
        [({"id": "t-input", "stage_input_paths": ["/work/input"]}, "nodeB")],
    )

    stage_launch_candidates_outside_lock(
        deps=_deps(plan=plan, calls=calls, guard_acquired=False))

    assert not any(call[0] == "stage_inputs" for call in calls)
    assert any(
        call[0] == "notify"
        and call[1] == "launch_input_staging_deduplicated"
        for call in calls
    )


def test_shared_cwd_runs_after_first_checkpoint_before_later_checkpoints():
    calls = []
    first = {
        "id": "t1",
        "ckpt_dir": "/work/ckpt1",
        "cwd": "/work/repo",
        "submitted_at": 1,
    }
    second = {
        "id": "t2",
        "ckpt_dir": "/work/ckpt2",
        "cwd": "/work/repo",
        "submitted_at": 2,
    }
    plan = _Plan(
        {},
        [("nodeA", "/work/repo")],
        [
            (second, "nodeA", {"node": "source"}),
            (first, "nodeA", {"node": "source"}),
        ],
    )

    stage_launch_candidates_outside_lock(
        deps=_deps(plan=plan, calls=calls, worker_count=1),
    )

    operations = [
        (call[0], call[1] if call[0] == "stage_ckpt" else call[2])
        for call in calls
        if call[0] in {"stage_ckpt", "stage_cwd"}
    ]
    assert operations == [
        ("stage_ckpt", "t1"),
        ("stage_cwd", "/work/repo"),
        ("stage_ckpt", "t2"),
    ]


def test_staging_lanes_run_distinct_nodes_concurrently():
    calls = []
    barrier = threading.Barrier(2)

    def wait_for_peer(task, target_node, **kwargs):
        del task, target_node, kwargs
        barrier.wait(timeout=1)
        return True, "ok"

    plan = _Plan(
        {},
        [("nodeA", "/work/a"), ("nodeB", "/work/b")],
        [],
    )

    stage_launch_candidates_outside_lock(
        deps=_deps(
            plan=plan,
            calls=calls,
            worker_count=2,
            cwd_hook=wait_for_peer,
        ),
    )

    assert {call[1] for call in calls if call[0] == "stage_cwd"} == {
        "nodeA", "nodeB"
    }


def test_background_staging_returns_promptly_and_deduplicates_busy_node():
    calls = []
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def blocking_stage(task, target_node, **kwargs):
        del task, target_node, kwargs
        started.set()
        release.wait(timeout=2)
        finished.set()
        return True, "ok"

    plan = _Plan({}, [("nodeBackground", "/work/background")], [])
    deps = _deps(
        plan=plan,
        calls=calls,
        worker_count=2,
        cwd_hook=blocking_stage,
    )

    t0 = time.monotonic()
    stage_launch_candidates_outside_lock(deps=deps, background=True)
    assert time.monotonic() - t0 < 0.5
    assert started.wait(timeout=1)

    stage_launch_candidates_outside_lock(deps=deps, background=True)
    assert sum(call[0] == "stage_cwd" for call in calls) == 1

    release.set()
    assert finished.wait(timeout=1)


def test_background_staging_consumes_watcher_shutdown_exception():
    target = "nodeShutdownCallback"
    future = Future()
    future.set_exception(WatcherShutdownRequested("stop"))
    _BACKGROUND_LANES[target] = future
    _BACKGROUND_ACTIVE_KEYS[target] = {("cwd", target, "/work")}

    _background_lane_done(target, future, _deps())

    assert target not in _BACKGROUND_LANES
    assert target not in _BACKGROUND_ACTIVE_KEYS


def test_background_staging_queues_new_work_for_busy_node():
    calls = []
    started = threading.Event()
    release = threading.Event()
    second_finished = threading.Event()

    def stage_in_order(task, target_node, **kwargs):
        del target_node, kwargs
        if task.get("cwd") == "/work/first":
            started.set()
            release.wait(timeout=2)
        if task.get("cwd") == "/work/second":
            second_finished.set()
        return True, "ok"

    plan = _Plan({}, [("nodeBusyQueue", "/work/first")], [])
    deps = _deps(
        plan=plan,
        calls=calls,
        worker_count=1,
        cwd_hook=stage_in_order,
    )

    stage_launch_candidates_outside_lock(deps=deps, background=True)
    assert started.wait(timeout=1)

    plan.candidates = [("nodeBusyQueue", "/work/second")]
    stage_launch_candidates_outside_lock(deps=deps, background=True)
    release.set()

    assert second_finished.wait(timeout=1)
    staged_paths = [
        call[2] for call in calls if call[0] == "stage_cwd"
    ]
    assert staged_paths == ["/work/first", "/work/second"]


def test_background_staging_queues_other_target_when_workers_are_full():
    calls = []
    started = threading.Event()
    release = threading.Event()
    queued_finished = threading.Event()

    def stage_with_saturated_worker(task, target_node, **kwargs):
        del kwargs
        if target_node == "nodeWorkerBusy":
            started.set()
            release.wait(timeout=2)
        if target_node == "nodeWorkerQueued":
            queued_finished.set()
        return True, "ok"

    plan = _Plan({}, [("nodeWorkerBusy", "/work/first")], [])
    deps = _deps(
        plan=plan,
        calls=calls,
        worker_count=1,
        cwd_hook=stage_with_saturated_worker,
    )

    stage_launch_candidates_outside_lock(deps=deps, background=True)
    assert started.wait(timeout=1)

    plan.candidates = [("nodeWorkerQueued", "/work/second")]
    stage_launch_candidates_outside_lock(deps=deps, background=True)
    release.set()

    assert queued_finished.wait(timeout=1)
    assert (
        "stage_cwd",
        "nodeWorkerQueued",
        "/work/second",
        {"extra_excludes": []},
    ) in calls


def test_background_staging_prioritizes_lane_with_more_operations():
    calls = []
    started = threading.Event()
    release = threading.Event()

    def blocking_stage(task, target_node, **kwargs):
        del task, target_node, kwargs
        started.set()
        release.wait(timeout=2)
        return True, "ok"

    plan = _Plan(
        {},
        [
            ("nodeA", "/work/a"),
            ("nodeB", "/work/b1"),
            ("nodeB", "/work/b2"),
        ],
        [],
    )
    deps = _deps(
        plan=plan,
        calls=calls,
        worker_count=1,
        cwd_hook=blocking_stage,
    )

    stage_launch_candidates_outside_lock(deps=deps, background=True)
    assert started.wait(timeout=1)
    assert next(call for call in calls if call[0] == "stage_cwd")[1] == "nodeB"
    release.set()


def test_busy_cross_process_guard_deduplicates_cwd_stage():
    calls = []
    plan = _Plan({}, [("nodeA", "/work/a")], [])

    stage_launch_candidates_outside_lock(
        deps=_deps(plan=plan, calls=calls, guard_acquired=False),
    )

    assert not any(call[0] == "stage_cwd" for call in calls)
    assert any(
        call[0] == "notify" and call[1] == "launch_staging_deduplicated"
        for call in calls
    )


def test_checkpoint_failure_and_exception_are_recorded_and_notified():
    calls = []
    fails = {}
    task_a = {"id": "t1", "ckpt_dir": "/ckpt/a"}
    task_b = {"id": "t2", "ckpt_dir": "/ckpt/b"}
    plan = _Plan({}, [], [(task_a, "nodeA", {"node": "local"}), (task_b, "nodeB", {"node": "local"})])

    stage_launch_candidates_outside_lock(
        deps=_deps(
            plan=plan,
            calls=calls,
            staging_fails=fails,
            ckpt_results={
                ("t1", "nodeA"): (False, "rsync failed"),
                ("t2", "nodeB"): RuntimeError("ckpt boom"),
            },
        ),
    )

    assert fails[("ckpt", "local", "nodeA", "/ckpt/a")] == (123.0, "rsync failed")
    assert fails[("ckpt", "local", "nodeB", "/ckpt/b")] == (123.0, "exception: ckpt boom")
    assert any(call[0] == "notify" and call[1] == "launch_ckpt_staging_failed" for call in calls)
    assert any(call[0] == "notify" and call[1] == "launch_ckpt_staging_exception" for call in calls)


def test_snapshot_error_notifies_and_returns():
    calls = []

    stage_launch_candidates_outside_lock(deps=_deps(calls=calls, snapshot_error=True))

    assert any(call[0] == "notify" and call[1] == "launch_staging_snapshot_error" for call in calls)
    assert not any(call[0] == "collect" for call in calls)
