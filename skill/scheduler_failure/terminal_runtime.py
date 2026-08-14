"""Runtime-bound terminal diagnosis wrappers."""

from __future__ import annotations

from typing import Any, Mapping


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_failure_terminal_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    def _diagnose_terminal(task):
        return _ns(namespace, "_diagnose_terminal_impl")(
            task,
            deps=_ns(namespace, "_terminal_diagnosis_deps")(),
        )

    def _terminal_diagnosis_deps():
        return _ns(namespace, "_build_terminal_diagnosis_deps")(namespace)

    return {
        "_diagnose_terminal": _diagnose_terminal,
        "_terminal_diagnosis_deps": _terminal_diagnosis_deps,
    }
