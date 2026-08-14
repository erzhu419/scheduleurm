from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_notify_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    exports: dict[str, Any] = {
        "WATCHER_LOG": _ns(namespace, "STATE_DIR") / "logs" / "watcher.log",
        "FEISHU_CONFIG": Path.home() / ".claude" / "feishu.json",
        "SLURM_SUPPORT_REMOVED": True,
    }

    def _pid_alive(pid):
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def _notify_deps():
        return _ns(namespace, "_build_notify_deps")(namespace)

    def _load_feishu_cfg():
        return _ns(namespace, "_load_feishu_cfg_impl")(_ns(namespace, "FEISHU_CONFIG"))

    def _send_feishu(webhook_url, text):
        return _ns(namespace, "_send_feishu_impl")(
            webhook_url,
            text,
            deps=_ns(namespace, "_notify_deps")(),
        )

    def _format_feishu(event_type, payload):
        return _ns(namespace, "_format_feishu_impl")(
            event_type,
            payload,
            deps=_ns(namespace, "_notify_deps")(),
        )

    def _compact_task_event_payload(task: dict) -> dict:
        return _ns(namespace, "_compact_task_event_payload_impl")(task)

    def notify(event_type, payload, feishu_enabled=True):
        return _ns(namespace, "_notify_impl")(
            event_type,
            payload,
            feishu_enabled=feishu_enabled,
            deps=_ns(namespace, "_notify_deps")(),
        )

    def _legacy_external_backend_message() -> str:
        return (
            "Legacy external scheduler records are read-only in scheduleurm. "
            "Use scheduleurm local dispatch on node001-node007 for new launches."
        )

    exports.update({
        "_pid_alive": _pid_alive,
        "_notify_deps": _notify_deps,
        "_load_feishu_cfg": _load_feishu_cfg,
        "_send_feishu": _send_feishu,
        "_format_feishu": _format_feishu,
        "_compact_task_event_payload": _compact_task_event_payload,
        "notify": notify,
        "_legacy_external_backend_message": _legacy_external_backend_message,
    })
    return exports
