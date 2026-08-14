"""Runtime wrappers for failure healing and escalation."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_failure_heal_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    latest_escalations_cache: dict[str, dict] = {}
    latest_escalations_stamp: dict[str, tuple[int, int] | None] = {"value": None}
    exports: dict[str, Any] = {
        "_HEAL_FIRE_LOCK": _ns(namespace, "STATE_DIR") / ".heal_fire.lock",
        "_HEAL_DEBOUNCE_S": 90,
        "_CLAUDE_BIN": os.environ.get(
            "CLAUDE_BIN",
            "/home/erzhu419/.nvm/versions/node/v22.17.1/bin/claude",
        ),
        "_NODE_BIN": "/home/erzhu419/.nvm/versions/node/v22.17.1/bin/node",
        "_CLAUDE_CLI_JS": (
            "/home/erzhu419/.nvm/versions/node/v22.17.1/lib/node_modules/"
            "@anthropic-ai/claude-code/cli.js"
        ),
        "_HEAL_FIRE_LOG": _ns(namespace, "LOG_DIR") / "heal_fires.log",
    }

    def _fire_heal_session():
        return _ns(namespace, "_fire_heal_session_impl")(
            deps=_ns(namespace, "_HealSessionDeps")(
                state_dir=_ns(namespace, "STATE_DIR"),
                log_dir=_ns(namespace, "LOG_DIR"),
                heal_fire_lock=_ns(namespace, "_HEAL_FIRE_LOCK"),
                heal_debounce_s=_ns(namespace, "_HEAL_DEBOUNCE_S"),
                claude_bin=_ns(namespace, "_CLAUDE_BIN"),
                node_bin=_ns(namespace, "_NODE_BIN"),
                claude_cli_js=_ns(namespace, "_CLAUDE_CLI_JS"),
                heal_fire_log=_ns(namespace, "_HEAL_FIRE_LOG"),
                home=Path.home,
                env_get=os.environ.get,
                now=time.time,
                strftime=time.strftime,
                popen=subprocess.Popen,
                devnull=subprocess.DEVNULL,
            ),
        )

    def _write_escalation(task, category, diag):
        record = _ns(namespace, "_write_escalation_impl")(
            task,
            category,
            diag,
            deps=_ns(namespace, "_EscalationDeps")(
                escalations_file=_ns(namespace, "ESCALATIONS_FILE"),
                fire_heal_session=_ns(namespace, "_fire_heal_session"),
                now=time.time,
            ),
        )
        latest_escalations_stamp["value"] = None
        return record

    def _refresh_escalations_outside_lock(force=False):
        path = _ns(namespace, "ESCALATIONS_FILE")
        try:
            stat = path.stat()
            stamp = (int(stat.st_mtime_ns), int(stat.st_size))
        except FileNotFoundError:
            stamp = (0, 0)
        if not force and latest_escalations_stamp["value"] == stamp:
            return len(latest_escalations_cache)
        latest = _ns(namespace, "_latest_escalations_impl")(path)
        latest_escalations_cache.clear()
        latest_escalations_cache.update(latest)
        try:
            stat = path.stat()
            latest_escalations_stamp["value"] = (int(stat.st_mtime_ns), int(stat.st_size))
        except FileNotFoundError:
            latest_escalations_stamp["value"] = (0, 0)
        return len(latest_escalations_cache)

    def _cached_latest_escalations():
        return latest_escalations_cache

    def _resolve_staged_cwd_escalations(cwd, nodes):
        resolved = _ns(namespace, "_resolve_staged_cwd_escalations_impl")(
            cwd,
            nodes,
            path=_ns(namespace, "ESCALATIONS_FILE"),
            now=time.time,
        )
        if resolved:
            latest_escalations_stamp["value"] = None
            _refresh_escalations_outside_lock()
        return resolved

    def _recover_resolved_staging_failures(state):
        return _ns(namespace, "_recover_resolved_staging_failures_impl")(
            state,
            latest_escalations_cache,
            requeue_after_crash=_ns(namespace, "_requeue_after_crash"),
        )

    def _blocked_nodes_for_task(task):
        return _ns(namespace, "_blocked_nodes_for_task_impl")(
            task,
            deps=_ns(namespace, "_BlockedNodesDeps")(
                escalations_file=_ns(namespace, "ESCALATIONS_FILE"),
                project_wide_env_block_ttl_s=_ns(namespace, "PROJECT_WIDE_ENV_BLOCK_TTL_S"),
                latest_escalations=_cached_latest_escalations,
                now=time.time,
            ),
        )

    def _blocked_nodes_for_signature(sig):
        return _ns(namespace, "_blocked_nodes_for_signature_impl")(
            sig,
            deps=_ns(namespace, "_BlockedNodesDeps")(
                escalations_file=_ns(namespace, "ESCALATIONS_FILE"),
                project_wide_env_block_ttl_s=_ns(namespace, "PROJECT_WIDE_ENV_BLOCK_TTL_S"),
                latest_escalations=_cached_latest_escalations,
                now=time.time,
            ),
        )

    exports.update({
        "_fire_heal_session": _fire_heal_session,
        "_write_escalation": _write_escalation,
        "_refresh_escalations_outside_lock": _refresh_escalations_outside_lock,
        "_cached_latest_escalations": _cached_latest_escalations,
        "_resolve_staged_cwd_escalations": _resolve_staged_cwd_escalations,
        "_recover_resolved_staging_failures": _recover_resolved_staging_failures,
        "_blocked_nodes_for_task": _blocked_nodes_for_task,
        "_blocked_nodes_for_signature": _blocked_nodes_for_signature,
    })
    return exports
