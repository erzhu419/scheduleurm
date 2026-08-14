"""Runtime-bound backend assembly wrappers."""

from __future__ import annotations

from typing import Any, Mapping, Optional

try:
    from scheduler_backend.facade import (
        Backend,
        LegacyExternalBackend as _LegacyExternalBackend,
        build_hybrid_backend_deps as _build_hybrid_backend_deps,
        build_local_backend_runtime_deps as _build_local_backend_runtime_deps,
        build_windows_backend_runtime_deps as _build_windows_backend_runtime_deps,
    )
    from scheduler_backend.runtime import build_backend_runtime as _build_backend_runtime
    from scheduler_backend.hybrid import HybridBackend as _HybridBackendImpl
    from scheduler_backend.local_backend import LocalBackend as _LocalBackendImpl
    from scheduler_windows.backend import WindowsBackend as _WindowsBackendImpl
except ModuleNotFoundError:
    from ..scheduler_backend_facade import (
        Backend,
        LegacyExternalBackend as _LegacyExternalBackend,
        build_hybrid_backend_deps as _build_hybrid_backend_deps,
        build_local_backend_runtime_deps as _build_local_backend_runtime_deps,
        build_windows_backend_runtime_deps as _build_windows_backend_runtime_deps,
    )
    from ..scheduler_backend_runtime import build_backend_runtime as _build_backend_runtime
    from ..scheduler_hybrid_backend import HybridBackend as _HybridBackendImpl
    from ..scheduler_local_backend import LocalBackend as _LocalBackendImpl
    from ..scheduler_windows_backend import WindowsBackend as _WindowsBackendImpl


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_launch_backend_runtime_exports(namespace: Mapping[str, Any]) -> dict[str, Any]:
    def _windows_backend_runtime_deps():
        return _build_windows_backend_runtime_deps(namespace)

    def _local_backend_runtime_deps():
        return _build_local_backend_runtime_deps(namespace)

    def _hybrid_backend_deps():
        return _build_hybrid_backend_deps(namespace)

    def _legacy_external_message_fallback() -> str:
        message = namespace.get("_legacy_external_backend_message")
        if callable(message):
            return message()
        return "Legacy external scheduler records are read-only; use scheduleurm local backend"

    backend_runtime = _build_backend_runtime(
        backend_base=Backend,
        windows_backend_impl=_WindowsBackendImpl,
        local_backend_impl=_LocalBackendImpl,
        legacy_external_backend_impl=_LegacyExternalBackend,
        hybrid_backend_impl=_HybridBackendImpl,
        windows_deps_factory=_windows_backend_runtime_deps,
        local_deps_factory=_local_backend_runtime_deps,
        hybrid_deps_factory=_hybrid_backend_deps,
        legacy_external_message=_legacy_external_message_fallback,
    )

    def _requires_local_capacity_check(
        node: str,
        task: Optional[dict] = None,
        node_state: Optional[dict] = None,
    ) -> bool:
        try:
            return _ns(namespace, "_BACKEND").requires_local_capacity_check(
                node,
                task=task,
                node_state=node_state,
            )
        except TypeError:
            try:
                return _ns(namespace, "_BACKEND").requires_local_capacity_check(node, task=task)
            except TypeError:
                return _ns(namespace, "_BACKEND").requires_local_capacity_check(node)

    def launch(task, node_state=None):
        return _ns(namespace, "_BACKEND").launch(task, node_state=node_state)

    return {
        "Backend": Backend,
        "_windows_backend_runtime_deps": _windows_backend_runtime_deps,
        "_local_backend_runtime_deps": _local_backend_runtime_deps,
        "_hybrid_backend_deps": _hybrid_backend_deps,
        "_BACKEND_RUNTIME": backend_runtime,
        "WindowsBackend": backend_runtime.WindowsBackend,
        "LocalBackend": backend_runtime.LocalBackend,
        "LegacyExternalBackend": backend_runtime.LegacyExternalBackend,
        "HybridBackend": backend_runtime.HybridBackend,
        "_BACKEND": backend_runtime.backend,
        "_requires_local_capacity_check": _requires_local_capacity_check,
        "launch": launch,
    }
