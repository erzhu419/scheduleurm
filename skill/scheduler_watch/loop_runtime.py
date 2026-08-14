from __future__ import annotations

import sys
from typing import Any, Mapping


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_watch_loop_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    exports: dict[str, Any] = {
        "WATCHER_STATE": _ns(namespace, "STATE_DIR") / ".watcher_state.json",
    }

    def _env_smoke_deps():
        return _ns(namespace, "_build_env_smoke_deps")(namespace)

    def _smoke_test_envs():
        return _ns(namespace, "_smoke_test_envs_impl")(
            deps=_ns(namespace, "_env_smoke_deps")(),
        )

    def _smoke_test_envs_enabled() -> bool:
        return _ns(namespace, "_smoke_test_envs_enabled_impl")(_ns(namespace, "os").environ)

    def _post_reboot_triage_announce():
        try:
            with open("/proc/uptime") as handle:
                uptime_s = float(handle.read().split()[0])
        except Exception:
            return
        if uptime_s > _ns(namespace, "REBOOT_DETECT_WINDOW_S"):
            return
        try:
            with _ns(namespace, "state_lock")():
                state = _ns(namespace, "load_state")()
            affected = [
                t for t in state["tasks"]
                if t.get("status") == "running"
                and (t.get("node") or t.get("assigned_node")) == "local"
            ]
        except Exception:
            affected = []
        _ns(namespace, "notify")(
            "post_reboot_triage",
            {
                "uptime_s": int(uptime_s),
                "local_running_pre_reboot": len(affected),
                "affected_ids": [t["id"] for t in affected[:20]],
                "note": (
                    "local box rebooted; all listed tasks have stale PIDs and will be flagged "
                    "as crashed in the first dispatch cycle, then auto-requeued. Tasks with "
                    "--resume-flag set resume from latest ckpt; --allow-no-resume tasks restart "
                    "from step 0. Remote-node tasks are unaffected (setsid kept them alive)."
                ),
            },
            feishu_enabled=False,
        )

    def _claim_tending_deps():
        return _ns(namespace, "_build_claim_tending_deps")(namespace)

    def _tend_claims_for_watch() -> None:
        try:
            with _ns(namespace, "state_lock")(shared=True, purpose="watch:claim-renew-snapshot"):
                cur_state = _ns(namespace, "load_state")()
            _ns(namespace, "_tend_scheduler_claims")(
                cur_state.get("tasks", []),
                _ns(namespace, "NODES"),
                deps=_ns(namespace, "_claim_tending_deps")(),
            )
        except Exception as exc:
            _ns(namespace, "notify")(
                "claims_tend_outer_error",
                {"error": str(exc)[:200]},
                feishu_enabled=False,
            )

    def cmd_watch(args):
        try:
            with _ns(namespace, "watcher_lifetime_lock")(timeout_s=0, purpose="watch:lifetime"):
                return _ns(namespace, "_cmd_watch_locked")(args)
        except _ns(namespace, "SchedulerLockTimeout") as exc:
            sys.exit(str(exc))

    def _watch_command_deps():
        return _ns(namespace, "_build_watch_command_deps")(namespace)

    def _cmd_watch_locked(args):
        return _ns(namespace, "_cmd_watch_locked_impl")(
            args,
            deps=_ns(namespace, "_watch_command_deps")(),
        )

    def _watch_iteration(args):
        return _ns(namespace, "_run_watch_iteration")(
            args,
            deps=_ns(namespace, "_watch_iteration_deps")(),
        )

    def _get_last_auto_adopt_at() -> float:
        return float(namespace.get("_LAST_AUTO_ADOPT_AT") or 0.0)

    def _set_last_auto_adopt_at(value: float) -> None:
        namespace["_LAST_AUTO_ADOPT_AT"] = value

    def _watch_iteration_deps():
        return _ns(namespace, "_build_watch_iteration_deps")(namespace)

    exports.update({
        "_env_smoke_deps": _env_smoke_deps,
        "_smoke_test_envs": _smoke_test_envs,
        "_smoke_test_envs_enabled": _smoke_test_envs_enabled,
        "_post_reboot_triage_announce": _post_reboot_triage_announce,
        "_claim_tending_deps": _claim_tending_deps,
        "_tend_claims_for_watch": _tend_claims_for_watch,
        "cmd_watch": cmd_watch,
        "_watch_command_deps": _watch_command_deps,
        "_cmd_watch_locked": _cmd_watch_locked,
        "_watch_iteration": _watch_iteration,
        "_get_last_auto_adopt_at": _get_last_auto_adopt_at,
        "_set_last_auto_adopt_at": _set_last_auto_adopt_at,
        "_watch_iteration_deps": _watch_iteration_deps,
    })
    return exports
