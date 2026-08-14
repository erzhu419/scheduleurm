"""In-process batching for resource and runtime history updates."""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass, replace
from typing import Any, Callable


@dataclass(frozen=True)
class HistoryBatchDeps:
    history_lock: Callable[..., Any]
    load_history: Callable[[], dict]
    save_history: Callable[[dict], Any]
    resource_record: Callable[..., Any]
    resource_deps_factory: Callable[[], Any]
    load_runtime_history: Callable[[], dict]
    save_runtime_history: Callable[[dict], Any]
    runtime_record: Callable[..., Any]
    runtime_deps_factory: Callable[[], Any]


class HistoryBatchCoordinator:
    """Collect history samples under state lock and persist them once outside it."""

    def __init__(self, deps: HistoryBatchDeps):
        self._deps = deps
        self._mutex = threading.Lock()
        self._depth = 0
        self._resource_samples: list[dict] = []
        self._runtime_samples: list[tuple[dict, int]] = []

    def begin(self) -> None:
        with self._mutex:
            self._depth += 1

    def end(self) -> None:
        with self._mutex:
            if self._depth <= 0:
                raise RuntimeError("history batch end without matching begin")
            self._depth -= 1

    def active(self) -> bool:
        with self._mutex:
            return self._depth > 0

    def queue_resource(self, sample: dict) -> bool:
        with self._mutex:
            if self._depth <= 0:
                return False
            self._resource_samples.append(dict(sample))
            return True

    def queue_runtime(self, task: dict, duration_s: int) -> bool:
        with self._mutex:
            if self._depth <= 0:
                return False
            self._runtime_samples.append((copy.deepcopy(task), int(duration_s or 0)))
            return True

    def pending_counts(self) -> dict[str, int]:
        with self._mutex:
            return {
                "resource": len(self._resource_samples),
                "runtime": len(self._runtime_samples),
            }

    def _take_pending(self) -> tuple[list[dict], list[tuple[dict, int]]]:
        with self._mutex:
            if self._depth > 0:
                raise RuntimeError("cannot flush history while a batch is active")
            resources = self._resource_samples
            runtimes = self._runtime_samples
            self._resource_samples = []
            self._runtime_samples = []
            return resources, runtimes

    def _restore_pending(
        self,
        resources: list[dict],
        runtimes: list[tuple[dict, int]],
    ) -> None:
        with self._mutex:
            self._resource_samples = resources + self._resource_samples
            self._runtime_samples = runtimes + self._runtime_samples

    def _flush_resource_samples(self, samples: list[dict]) -> None:
        if not samples:
            return
        holder = {"history": self._deps.load_history()}
        base_deps = self._deps.resource_deps_factory()
        in_memory_deps = replace(
            base_deps,
            load_history=lambda: holder["history"],
            save_history=lambda history: holder.__setitem__("history", history),
        )
        for sample in samples:
            self._deps.resource_record(**sample, deps=in_memory_deps)
        self._deps.save_history(holder["history"])

    def _flush_runtime_samples(self, samples: list[tuple[dict, int]]) -> None:
        if not samples:
            return
        holder = {"history": self._deps.load_runtime_history()}
        base_deps = self._deps.runtime_deps_factory()
        in_memory_deps = replace(
            base_deps,
            load_runtime_history=lambda: holder["history"],
            save_runtime_history=lambda history: holder.__setitem__("history", history),
        )
        for task, duration_s in samples:
            self._deps.runtime_record(task, duration_s=duration_s, deps=in_memory_deps)
        self._deps.save_runtime_history(holder["history"])

    def flush(self) -> dict[str, int]:
        resources, runtimes = self._take_pending()
        if not resources and not runtimes:
            return {"resource": 0, "runtime": 0}
        with self._deps.history_lock(purpose="history:batch-flush"):
            try:
                self._flush_resource_samples(resources)
            except Exception:
                self._restore_pending(resources, runtimes)
                raise
            try:
                self._flush_runtime_samples(runtimes)
            except Exception:
                self._restore_pending([], runtimes)
                raise
        return {"resource": len(resources), "runtime": len(runtimes)}
