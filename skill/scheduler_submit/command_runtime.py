from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Optional


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_submit_command_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    def _submit_preflight_deps():
        return _ns(namespace, "_build_submit_preflight_deps")(namespace)

    def _submit_task_deps():
        return _ns(namespace, "_build_submit_task_deps")(namespace)

    def cmd_submit(args):
        return _ns(namespace, "_cmd_submit_impl")(
            args,
            deps=_ns(namespace, "_build_submit_command_deps")(namespace),
        )

    def _load_submit_jsonl_specs(args) -> list:
        if getattr(args, "stdin", False):
            text = sys.stdin.read()
            source = "stdin"
        else:
            path = getattr(args, "file", "") or ""
            if not path:
                sys.exit("submit-jsonl requires --stdin or --file")
            source = path
            text = Path(path).read_text(encoding="utf-8")
        try:
            return _ns(namespace, "_load_submit_jsonl_specs_from_text")(text, source=source)
        except ValueError as exc:
            sys.exit(str(exc))

    def _spec_bool(spec: dict, key: str, default: bool = False) -> bool:
        return _ns(namespace, "_bulk_spec_bool")(spec, key, default)

    def _spec_int(spec: dict, key: str, default=None):
        return _ns(namespace, "_bulk_spec_int")(spec, key, default)

    def _bulk_submit_deps():
        return _ns(namespace, "_build_bulk_submit_deps")(namespace)

    def _build_bulk_submit_task(
        state: dict,
        spec: dict,
        task_id: Optional[str] = None,
        resource_history: Optional[dict] = None,
        runtime_history: Optional[dict] = None,
        runtime_closest_index: Optional[list[dict]] = None,
    ) -> dict:
        return _ns(namespace, "_build_bulk_submit_task_impl")(
            state,
            spec,
            deps=_ns(namespace, "_bulk_submit_deps")(),
            task_id=task_id,
            resource_history=resource_history,
            runtime_history=runtime_history,
            runtime_closest_index=runtime_closest_index,
        )

    def _build_bulk_submit_tasks(
        state: dict,
        specs: list[dict],
        task_ids: list[str],
        resource_history: Optional[dict] = None,
        runtime_history: Optional[dict] = None,
        runtime_closest_index: Optional[list[dict]] = None,
    ) -> list[dict]:
        return _ns(namespace, "_build_bulk_submit_tasks_impl")(
            state,
            specs,
            deps=_ns(namespace, "_bulk_submit_deps")(),
            task_ids=task_ids,
            resource_history=resource_history,
            runtime_history=runtime_history,
            runtime_closest_index=runtime_closest_index,
        )

    def cmd_submit_jsonl(args):
        return _ns(namespace, "_cmd_submit_jsonl_impl")(
            args,
            deps=_ns(namespace, "_build_submit_jsonl_command_deps")(namespace),
        )

    return {
        "_submit_preflight_deps": _submit_preflight_deps,
        "_submit_task_deps": _submit_task_deps,
        "cmd_submit": cmd_submit,
        "_load_submit_jsonl_specs": _load_submit_jsonl_specs,
        "_spec_bool": _spec_bool,
        "_spec_int": _spec_int,
        "_bulk_submit_deps": _bulk_submit_deps,
        "_build_bulk_submit_task": _build_bulk_submit_task,
        "_build_bulk_submit_tasks": _build_bulk_submit_tasks,
        "cmd_submit_jsonl": cmd_submit_jsonl,
    }
