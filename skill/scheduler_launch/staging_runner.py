from __future__ import annotations

from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable, Optional

try:
    from ..scheduler_watch.shutdown import WatcherShutdownRequested
except ImportError:
    from scheduler_watch.shutdown import WatcherShutdownRequested


_BACKGROUND_EXECUTOR = ThreadPoolExecutor(
    max_workers=32,
    thread_name_prefix="scheduleurm-launch-stage",
)
_BACKGROUND_LANES: dict[str, Future] = {}
_BACKGROUND_ACTIVE_KEYS: dict[str, set[tuple]] = {}
_BACKGROUND_PENDING: dict[str, list[tuple]] = {}
_BACKGROUND_PENDING_KEYS: dict[str, set[tuple]] = {}
_BACKGROUND_WORKER_LIMIT = 1
_BACKGROUND_LANES_LOCK = Lock()


@dataclass(frozen=True)
class LaunchStagingRunnerDeps:
    state_lock: Callable[..., Any]
    load_state: Callable[[], dict]
    collect_launch_staging_plan: Callable[[dict, Optional[set]], Any]
    launch_stage_candidate_key: Callable[[tuple], tuple]
    max_candidates_per_pass: int
    worker_count: int
    stage_cwd_for_launch: Callable[..., tuple[bool, str]]
    stage_input_paths_for_launch: Callable[[dict, str], tuple[bool, str]]
    launch_input_stage_key: Callable[[dict, str], tuple]
    stage_resume_ckpt_for_launch: Callable[[dict, str, dict], tuple[bool, str]]
    resume_ckpt_stage_key: Callable[..., tuple]
    staging_cache_hit: Callable[[tuple], bool]
    staging_fails: dict
    notify: Callable[..., Any]
    now: Callable[[], float]
    staging_key_guard: Callable[[tuple], Any] = lambda key: nullcontext(True)


def _notify(deps: LaunchStagingRunnerDeps, event: str, payload: dict) -> None:
    try:
        deps.notify(event, payload, feishu_enabled=False)
    except Exception:
        pass


def _stage_cwd_candidate(
    target_node: str,
    cwd: str,
    protected_under_cwd: dict,
    deps: LaunchStagingRunnerDeps,
) -> None:
    cwd_norm = cwd.rstrip("/")
    extra = sorted(protected_under_cwd.get(cwd_norm, set()))
    cwd_key = ("local", target_node, cwd)
    try:
        with deps.staging_key_guard(cwd_key) as acquired:
            if not acquired:
                _notify(
                    deps,
                    "launch_staging_deduplicated",
                    {"target": target_node, "cwd": cwd},
                )
                return
            if deps.staging_cache_hit(cwd_key):
                return
            ok, msg = deps.stage_cwd_for_launch(
                {"cwd": cwd},
                target_node,
                extra_excludes=extra,
            )
            if not ok:
                if msg.startswith(("CAP_EXCEEDED:", "UNSAFE_CWD:")):
                    return
                deps.staging_fails[cwd_key] = (deps.now(), msg[:200])
                _notify(
                    deps,
                    "launch_staging_failed",
                    {"target": target_node, "cwd": cwd, "reason": msg[:200]},
                )
    except Exception as exc:
        deps.staging_fails[cwd_key] = (
            deps.now(),
            f"exception: {str(exc)[:150]}",
        )
        _notify(
            deps,
            "launch_staging_exception",
            {"target": target_node, "cwd": cwd, "error": str(exc)[:200]},
        )


def _stage_ckpt_candidate(
    task_snapshot: dict,
    target_node: str,
    source_location: dict,
    deps: LaunchStagingRunnerDeps,
) -> None:
    source_node = source_location.get("node")
    ckpt_dir = task_snapshot.get("ckpt_dir")
    ckpt_key = deps.resume_ckpt_stage_key(
        source_node, target_node, ckpt_dir, source_location)
    if deps.staging_cache_hit(ckpt_key):
        return
    try:
        with deps.staging_key_guard(ckpt_key) as acquired:
            if not acquired:
                _notify(
                    deps,
                    "launch_ckpt_staging_deduplicated",
                    {
                        "task_id": task_snapshot.get("id"),
                        "source": source_node,
                        "target": target_node,
                        "ckpt_dir": ckpt_dir,
                    },
                )
                return
            if deps.staging_cache_hit(ckpt_key):
                return
            ok, msg = deps.stage_resume_ckpt_for_launch(
                task_snapshot,
                target_node,
                source_location,
            )
            if not ok:
                if msg.startswith("CAP_EXCEEDED:"):
                    return
                deps.staging_fails[ckpt_key] = (deps.now(), msg[:200])
                _notify(
                    deps,
                    "launch_ckpt_staging_failed",
                    {
                        "task_id": task_snapshot.get("id"),
                        "source": source_node,
                        "target": target_node,
                        "ckpt_dir": ckpt_dir,
                        "reason": msg[:200],
                    },
                )
    except Exception as exc:
        deps.staging_fails[ckpt_key] = (
            deps.now(),
            f"exception: {str(exc)[:150]}",
        )
        _notify(
            deps,
            "launch_ckpt_staging_exception",
            {
                "task_id": task_snapshot.get("id"),
                "source": source_node,
                "target": target_node,
                "ckpt_dir": ckpt_dir,
                "error": str(exc)[:200],
            },
        )


def _stage_input_candidate(
    task_snapshot: dict,
    target_node: str,
    deps: LaunchStagingRunnerDeps,
) -> None:
    key = deps.launch_input_stage_key(task_snapshot, target_node)
    try:
        with deps.staging_key_guard(key) as acquired:
            if not acquired:
                _notify(
                    deps,
                    "launch_input_staging_deduplicated",
                    {
                        "task_id": task_snapshot.get("id"),
                        "target": target_node,
                    },
                )
                return
            if deps.staging_cache_hit(key):
                return
            ok, msg = deps.stage_input_paths_for_launch(
                task_snapshot, target_node)
            if not ok:
                deps.staging_fails[key] = (deps.now(), msg[:200])
                _notify(
                    deps,
                    "launch_input_staging_failed",
                    {
                        "task_id": task_snapshot.get("id"),
                        "target": target_node,
                        "reason": msg[:200],
                    },
                )
    except Exception as exc:
        deps.staging_fails[key] = (deps.now(), f"exception: {str(exc)[:150]}")
        _notify(
            deps,
            "launch_input_staging_exception",
            {"task_id": task_snapshot.get("id"), "target": target_node, "error": str(exc)[:200]},
        )


def _freeze_background_key(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted(
            (str(key), _freeze_background_key(item))
            for key, item in value.items()
        ))
    if isinstance(value, (list, tuple, set)):
        return tuple(_freeze_background_key(item) for item in value)
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _background_operation_key(
    kind: str,
    payload: tuple,
    deps: LaunchStagingRunnerDeps,
) -> tuple:
    if kind == "cwd":
        target_node, cwd = payload
        return ("cwd", target_node, str(cwd).rstrip("/"))
    if kind == "input":
        task_snapshot, target_node = payload
        return (
            "input",
            _freeze_background_key(
                deps.launch_input_stage_key(task_snapshot, target_node)
            ),
        )
    task_snapshot, target_node, source_location = payload
    return (
        "ckpt",
        _freeze_background_key(
            deps.resume_ckpt_stage_key(
                source_location.get("node"),
                target_node,
                task_snapshot.get("ckpt_dir"),
                source_location,
            )
        ),
    )


def _run_background_jobs(jobs: list[tuple]) -> None:
    first_error: Exception | None = None
    for run_lane, operations, _deps, _keys in jobs:
        try:
            run_lane(operations)
        except Exception as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error


def _drain_background_pending_locked() -> list[tuple[str, Future, LaunchStagingRunnerDeps]]:
    callbacks: list[tuple[str, Future, LaunchStagingRunnerDeps]] = []
    while (
        len(_BACKGROUND_LANES) < _BACKGROUND_WORKER_LIMIT
        and _BACKGROUND_PENDING
    ):
        target_node = min(
            _BACKGROUND_PENDING,
            key=lambda target: (
                -sum(
                    len(job[1])
                    for job in _BACKGROUND_PENDING.get(target, [])
                ),
                target,
            ),
        )
        jobs = _BACKGROUND_PENDING.pop(target_node)
        keys = _BACKGROUND_PENDING_KEYS.pop(target_node, set())
        future = _BACKGROUND_EXECUTOR.submit(_run_background_jobs, jobs)
        _BACKGROUND_LANES[target_node] = future
        _BACKGROUND_ACTIVE_KEYS[target_node] = keys
        callbacks.append((target_node, future, jobs[-1][2]))
    return callbacks


def _register_background_callbacks(
    callbacks: list[tuple[str, Future, LaunchStagingRunnerDeps]],
) -> None:
    for target_node, future, deps in callbacks:
        future.add_done_callback(
            lambda completed, node=target_node, lane_deps=deps:
                _background_lane_done(node, completed, lane_deps)
        )


def _background_lane_done(
    target_node: str,
    future: Future,
    deps: LaunchStagingRunnerDeps,
) -> None:
    error: Exception | None = None
    shutdown_requested = False
    try:
        future.result()
    except WatcherShutdownRequested:
        shutdown_requested = True
    except Exception as exc:
        error = exc

    with _BACKGROUND_LANES_LOCK:
        if _BACKGROUND_LANES.get(target_node) is not future:
            return
        _BACKGROUND_LANES.pop(target_node, None)
        _BACKGROUND_ACTIVE_KEYS.pop(target_node, None)
        if shutdown_requested:
            _BACKGROUND_PENDING.clear()
            _BACKGROUND_PENDING_KEYS.clear()
            callbacks = []
        else:
            callbacks = _drain_background_pending_locked()

    if error is not None:
        _notify(
            deps,
            "launch_staging_background_exception",
            {"target": target_node, "error": str(error)[:200]},
        )
    _register_background_callbacks(callbacks)


def _submit_background_lanes(
    lanes: dict[str, list[tuple[str, tuple]]],
    run_lane: Callable[[list[tuple[str, tuple]]], None],
    deps: LaunchStagingRunnerDeps,
) -> None:
    global _BACKGROUND_WORKER_LIMIT
    with _BACKGROUND_LANES_LOCK:
        _BACKGROUND_WORKER_LIMIT = max(1, int(deps.worker_count or 1))
        for target_node, operations in lanes.items():
            active_keys = _BACKGROUND_ACTIVE_KEYS.get(target_node, set())
            pending_keys = _BACKGROUND_PENDING_KEYS.setdefault(
                target_node, set()
            )
            new_operations: list[tuple[str, tuple]] = []
            new_keys: set[tuple] = set()
            for kind, payload in operations:
                key = _background_operation_key(kind, payload, deps)
                if key in active_keys or key in pending_keys or key in new_keys:
                    continue
                new_operations.append((kind, payload))
                new_keys.add(key)
            if not new_operations:
                if not _BACKGROUND_PENDING.get(target_node):
                    _BACKGROUND_PENDING_KEYS.pop(target_node, None)
                continue
            _BACKGROUND_PENDING.setdefault(target_node, []).append(
                (run_lane, new_operations, deps, new_keys)
            )
            pending_keys.update(new_keys)
        callbacks = _drain_background_pending_locked()

    # add_done_callback may invoke synchronously for a short lane.
    _register_background_callbacks(callbacks)


def stage_launch_candidates_outside_lock(
    task_ids: Optional[set] = None,
    *,
    deps: LaunchStagingRunnerDeps,
    background: bool = False,
) -> None:
    try:
        with deps.state_lock(shared=True, purpose="watch:post-reboot-snapshot"):
            state = deps.load_state()
        plan = deps.collect_launch_staging_plan(state, task_ids)
        protected_under_cwd = plan.protected_under_cwd
        candidates = plan.candidates
        ckpt_candidates = plan.ckpt_candidates
        input_candidates = plan.input_candidates
    except Exception as exc:
        _notify(deps, "launch_staging_snapshot_error", {"error": str(exc)[:200]})
        return

    if not candidates and not ckpt_candidates and not input_candidates:
        return

    ordered_candidates = sorted(candidates, key=deps.launch_stage_candidate_key)
    if deps.max_candidates_per_pass > 0:
        ordered_candidates = ordered_candidates[: deps.max_candidates_per_pass]

    # Run one staging lane per target node.  This parallelizes independent SSH
    # routes while keeping rsyncs to the same host serialized.  Put a task's
    # shared cwd immediately after the first checkpoint that needs it; placing
    # every checkpoint before one shared cwd delays all otherwise-ready tasks.
    lanes: dict[str, list[tuple[str, tuple]]] = defaultdict(list)
    pending_cwds: dict[str, dict[str, tuple[str, str]]] = defaultdict(dict)
    for target_node, cwd in ordered_candidates:
        pending_cwds[target_node][cwd.rstrip("/")] = (target_node, cwd)

    for task_snapshot, target_node, source_location in sorted(
        ckpt_candidates,
        key=lambda item: (
            item[1],
            float(item[0].get("submitted_at") or 0),
            str(item[0].get("id") or ""),
        ),
    ):
        lanes[target_node].append(
            ("ckpt", (task_snapshot, target_node, source_location))
        )
        task_cwd = str(task_snapshot.get("cwd") or "").rstrip("/")
        cwd_candidate = pending_cwds[target_node].pop(task_cwd, None)
        if cwd_candidate is not None:
            lanes[target_node].append(("cwd", cwd_candidate))

    for task_snapshot, target_node in sorted(
        input_candidates,
        key=lambda item: (item[1], str(item[0].get("id") or "")),
    ):
        lanes[target_node].append(("input", (task_snapshot, target_node)))

    for target_node, cwd in ordered_candidates:
        cwd_candidate = pending_cwds[target_node].pop(
            cwd.rstrip("/"), None)
        if cwd_candidate is not None:
            lanes[target_node].append(("cwd", cwd_candidate))

    def run_lane(operations: list[tuple[str, tuple]]) -> None:
        for kind, payload in operations:
            if kind == "ckpt":
                _stage_ckpt_candidate(*payload, deps)
            elif kind == "input":
                _stage_input_candidate(*payload, deps)
            else:
                _stage_cwd_candidate(*payload, protected_under_cwd, deps)

    if background:
        _submit_background_lanes(lanes, run_lane, deps)
        return

    workers = max(1, min(int(deps.worker_count or 1), len(lanes)))
    if workers <= 1:
        for node in sorted(lanes):
            run_lane(lanes[node])
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_lane, lanes[node]) for node in sorted(lanes)]
        for future in as_completed(futures):
            future.result()
