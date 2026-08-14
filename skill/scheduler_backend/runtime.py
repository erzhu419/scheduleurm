from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class BackendRuntime:
    Backend: type
    WindowsBackend: type
    LocalBackend: type
    LegacyExternalBackend: type
    HybridBackend: type
    backend: Any


def build_backend_runtime(
    *,
    backend_base: type,
    windows_backend_impl: type,
    local_backend_impl: type,
    legacy_external_backend_impl: type,
    hybrid_backend_impl: type,
    windows_deps_factory: Callable[[], Any],
    local_deps_factory: Callable[[], Any],
    hybrid_deps_factory: Callable[[], Any],
    legacy_external_message: Callable[[], str],
) -> BackendRuntime:
    """Build scheduler backend classes while keeping scheduler.py as wiring only."""

    class WindowsBackend(windows_backend_impl, backend_base):
        def __init__(self):
            super().__init__(windows_deps_factory)

    class LocalBackend(local_backend_impl, backend_base):
        """Local backend wrapper; concrete launch/probe code lives in local modules."""

        def __init__(self):
            super().__init__(local_deps_factory)

    class LegacyExternalBackend(legacy_external_backend_impl):
        """Compatibility backend for legacy externally-managed task records."""

        def __init__(self):
            super().__init__(legacy_external_message)

    class HybridBackend(hybrid_backend_impl, backend_base):
        def __init__(self):
            super().__init__(hybrid_deps_factory)

    # Make reprs, tracebacks, and user-facing introspection match the historical
    # scheduler.py classes even though the implementation now lives here.
    for cls in (WindowsBackend, LocalBackend, LegacyExternalBackend, HybridBackend):
        cls.__module__ = "scheduler"

    return BackendRuntime(
        Backend=backend_base,
        WindowsBackend=WindowsBackend,
        LocalBackend=LocalBackend,
        LegacyExternalBackend=LegacyExternalBackend,
        HybridBackend=HybridBackend,
        backend=HybridBackend(),
    )
