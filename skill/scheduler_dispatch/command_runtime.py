from __future__ import annotations

from typing import Any, Mapping


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_dispatch_command_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    def cmd_dispatch(args):
        bulk_window = bool(getattr(args, "bulk_window", False))
        if bulk_window:
            ttl = getattr(args, "intent_ttl", None) or _ns(namespace, "DISPATCH_INTENT_TTL_S")
            label = getattr(args, "intent_label", "") or "bulk-dispatch"
            _ns(namespace, "_write_dispatch_intent")(label=label, ttl_s=ttl)
        try:
            return _ns(namespace, "_cmd_dispatch_impl")(args)
        finally:
            if bulk_window:
                _ns(namespace, "_clear_dispatch_intent")()

    def _cmd_dispatch_impl(args):
        return _ns(namespace, "_run_dispatch_command")(
            args,
            deps=_ns(namespace, "_build_dispatch_command_deps")(namespace),
        )

    return {
        "cmd_dispatch": cmd_dispatch,
        "_cmd_dispatch_impl": _cmd_dispatch_impl,
    }
