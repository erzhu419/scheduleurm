from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

from skill.scheduler_runtime.history_batch import HistoryBatchCoordinator, HistoryBatchDeps


@dataclass(frozen=True)
class _ResourceDeps:
    load_history: Callable[[], dict]
    save_history: Callable[[dict], None]


@dataclass(frozen=True)
class _RuntimeDeps:
    load_runtime_history: Callable[[], dict]
    save_runtime_history: Callable[[dict], None]


def test_history_batch_folds_many_samples_with_one_save_per_file():
    calls = []
    resource_store = {}
    runtime_store = {}

    @contextmanager
    def history_lock(**kwargs):
        calls.append(("lock_enter", kwargs))
        yield
        calls.append(("lock_exit", kwargs))

    def save_resource(history):
        resource_store.clear()
        resource_store.update(history)
        calls.append(("save_resource", dict(history)))

    def save_runtime(history):
        runtime_store.clear()
        runtime_store.update(history)
        calls.append(("save_runtime", dict(history)))

    def resource_record(sig, peak_ram_mb=0, **kwargs):
        deps = kwargs["deps"]
        history = deps.load_history()
        history[sig] = history.get(sig, 0) + peak_ram_mb
        deps.save_history(history)

    def runtime_record(task, duration_s=0, **kwargs):
        deps = kwargs["deps"]
        history = deps.load_runtime_history()
        history[task["id"]] = duration_s
        deps.save_runtime_history(history)

    batch = HistoryBatchCoordinator(
        HistoryBatchDeps(
            history_lock=history_lock,
            load_history=lambda: dict(resource_store),
            save_history=save_resource,
            resource_record=resource_record,
            resource_deps_factory=lambda: _ResourceDeps(lambda: {}, lambda _history: None),
            load_runtime_history=lambda: dict(runtime_store),
            save_runtime_history=save_runtime,
            runtime_record=runtime_record,
            runtime_deps_factory=lambda: _RuntimeDeps(lambda: {}, lambda _history: None),
        )
    )

    task = {"id": "before"}
    batch.begin()
    assert batch.queue_resource({"sig": "family", "peak_ram_mb": 100}) is True
    assert batch.queue_resource({"sig": "family", "peak_ram_mb": 200}) is True
    assert batch.queue_runtime(task, 10) is True
    task["id"] = "after"
    batch.end()

    assert batch.flush() == {"resource": 2, "runtime": 1}
    assert resource_store == {"family": 300}
    assert runtime_store == {"before": 10}
    assert sum(call[0] == "save_resource" for call in calls) == 1
    assert sum(call[0] == "save_runtime" for call in calls) == 1
    assert calls[0][0] == "lock_enter"
    assert calls[-1][0] == "lock_exit"
    assert batch.pending_counts() == {"resource": 0, "runtime": 0}


def test_history_batch_rejects_flush_while_active():
    @contextmanager
    def history_lock(**_kwargs):
        yield

    batch = HistoryBatchCoordinator(
        HistoryBatchDeps(
            history_lock=history_lock,
            load_history=lambda: {},
            save_history=lambda _history: None,
            resource_record=lambda *args, **kwargs: None,
            resource_deps_factory=lambda: _ResourceDeps(lambda: {}, lambda _history: None),
            load_runtime_history=lambda: {},
            save_runtime_history=lambda _history: None,
            runtime_record=lambda *args, **kwargs: None,
            runtime_deps_factory=lambda: _RuntimeDeps(lambda: {}, lambda _history: None),
        )
    )
    batch.begin()

    try:
        batch.flush()
    except RuntimeError as exc:
        assert "batch is active" in str(exc)
    else:
        raise AssertionError("flush should fail while the batch is active")
    finally:
        batch.end()
