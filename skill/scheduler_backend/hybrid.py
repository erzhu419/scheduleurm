from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class HybridBackendDeps:
    node_is_windows: Callable[[str], bool]
    local_backend_factory: Callable[[], Any]
    legacy_backend_factory: Callable[[], Any]
    windows_backend_factory: Callable[[], Any]


class HybridBackend:
    """Route scheduleurm launches to the node-native backend implementation."""

    name = "hybrid"

    def __init__(self, deps: HybridBackendDeps | Callable[[], HybridBackendDeps]):
        self._deps_source = deps
        self._local = None
        self._legacy = None
        self._windows = None
        self._cache: dict = {}  # legacy compatibility; new launches use local/windows only

    @property
    def deps(self) -> HybridBackendDeps:
        if callable(self._deps_source):
            return self._deps_source()
        return self._deps_source

    @property
    def local_backend(self):
        if self._local is None:
            self._local = self.deps.local_backend_factory()
        return self._local

    @property
    def legacy_backend(self):
        if self._legacy is None:
            self._legacy = self.deps.legacy_backend_factory()
        return self._legacy

    @property
    def windows_backend(self):
        if self._windows is None:
            self._windows = self.deps.windows_backend_factory()
        return self._windows

    def requires_local_capacity_check(self, node: str, task: Optional[dict] = None,
                                      node_state: Optional[dict] = None) -> bool:
        """All new launches are scheduleurm-managed and must pass local capacity checks."""
        return True

    def _backend_for(self, node: str, task: Optional[dict] = None,
                     node_state: Optional[dict] = None):
        if self.deps.node_is_windows(node):
            return self.windows_backend
        return self.local_backend

    def _backend_for_task(self, task: dict, node_state: Optional[dict] = None):
        """Route by launch artifacts on the task record, not mutable node state."""
        node = task.get("node")
        if node and self.deps.node_is_windows(node):
            return self.windows_backend
        if task.get("slurm_job_id"):
            return self.legacy_backend
        if task.get("remote_pids"):
            return self.local_backend
        if not node:
            return self.local_backend
        return self._backend_for(node, task, node_state=node_state)

    def launch(self, task: dict, node_state: Optional[dict] = None) -> tuple[bool, str]:
        return self._backend_for_task(task, node_state=node_state).launch(
            task,
            node_state=node_state,
        )

    def kill(self, task: dict, timeout: int = 15) -> tuple[bool, str]:
        return self._backend_for_task(task).kill(task, timeout=timeout)

    def kill_many(self, tasks: list[dict], timeout: int = 15) -> dict[str, tuple[bool, str]]:
        grouped: dict[Any, list[dict]] = {}
        for task in tasks:
            grouped.setdefault(self._backend_for_task(task), []).append(task)
        results: dict[str, tuple[bool, str]] = {}
        for backend, backend_tasks in grouped.items():
            results.update(backend.kill_many(backend_tasks, timeout=timeout))
        return results

    def batch_probe(self, state: dict) -> dict:
        local_tasks, legacy_tasks, windows_tasks = [], [], []
        for task in state.get("tasks", []):
            if task.get("status") != "running":
                continue
            backend = self._backend_for_task(task)
            if backend is self.legacy_backend:
                legacy_tasks.append(task)
            elif backend is self.windows_backend:
                windows_tasks.append(task)
            else:
                local_tasks.append(task)

        merged: dict = {}
        if local_tasks:
            merged.update(self.local_backend.batch_probe({"tasks": local_tasks}))
        if legacy_tasks:
            merged.update(self.legacy_backend.batch_probe({"tasks": legacy_tasks}))
        if windows_tasks:
            merged.update(self.windows_backend.batch_probe({"tasks": windows_tasks}))
        return merged
