"""watch command daemon lifecycle runner."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class WatchCommandDeps:
    resource_log_interval_s: int
    reset_watcher_shutdown: Callable[[], Any]
    set_hard_rule_mode: Callable[[Any], Any]
    configure_algorithm: Callable[[Any], Any]
    watcher_state_exists: Callable[[], bool]
    read_watcher_state_text: Callable[[], str]
    ensure_watcher_state_parent: Callable[[], Any]
    write_watcher_state_text: Callable[[str], Any]
    unlink_watcher_state: Callable[[], Any]
    live_watcher_pid: Callable[..., Any]
    pid_alive: Callable[[Any], bool]
    register_signal_handler: Callable[[Any, Any], Any]
    sigterm: Any
    sigint: Any
    request_watcher_shutdown: Callable[..., Any]
    initial_watcher_state: Callable[..., dict]
    getpid: Callable[[], int]
    now: Callable[[], float]
    notify: Callable[..., Any]
    smoke_test_envs_enabled: Callable[[], bool]
    smoke_test_envs: Callable[[], Any]
    state_lock: Callable[..., Any]
    load_state: Callable[[], dict]
    save_state: Callable[[dict], Any]
    recover_stale_launching_tasks_outside_lock: Callable[..., int]
    post_reboot_triage_announce: Callable[[], Any]
    watch_iteration: Callable[[Any], Any]
    watcher_shutdown_requested_type: type[BaseException]
    sleep: Callable[[float], Any]
    start_eta_periodic_refresh: Callable[[], bool] = lambda: False
    stop_eta_periodic_refresh: Callable[[], Any] = lambda: None


def _refuse_duplicate_watcher(deps: WatchCommandDeps) -> None:
    if not deps.watcher_state_exists():
        return
    try:
        existing = json.loads(deps.read_watcher_state_text())
        existing_pid = deps.live_watcher_pid(existing, pid_alive=deps.pid_alive)
        if existing_pid:
            raise SystemExit(
                f"another watcher is already running (pid={existing_pid}). "
                f"Stop it first: kill {existing_pid}"
            )
    except SystemExit:
        raise
    except Exception:
        pass


def _write_initial_state(args, deps: WatchCommandDeps) -> dict:
    me = deps.initial_watcher_state(
        pid=deps.getpid(),
        started_at=deps.now(),
        interval=args.interval,
        heartbeat=args.heartbeat,
        resource_log_interval=getattr(
            args,
            "resource_log_interval",
            deps.resource_log_interval_s,
        ),
    )
    deps.ensure_watcher_state_parent()
    deps.write_watcher_state_text(json.dumps(me))
    return me


def _run_env_smoke(args, deps: WatchCommandDeps) -> None:
    if deps.smoke_test_envs_enabled():
        try:
            deps.smoke_test_envs()
        except Exception as exc:
            deps.notify("env_smoke_test_error", {"error": str(exc)[:200]}, feishu_enabled=False)
        return
    deps.notify(
        "env_smoke_test_skipped",
        {"reason": "disabled; set SCHEDULEURM_ENV_SMOKE_TEST=1 to enable"},
        feishu_enabled=False,
    )


def _recover_stale_launching_at_start(deps: WatchCommandDeps) -> None:
    try:
        reverted = deps.recover_stale_launching_tasks_outside_lock(
            purpose="watch-start:recover-launching"
        )
        if reverted > 0:
            deps.notify(
                "launching_state_recovered",
                {"reverted_count": reverted},
                feishu_enabled=False,
            )
    except Exception as exc:
        deps.notify("launching_recovery_error", {"error": str(exc)[:200]}, feishu_enabled=False)


def _announce_post_reboot_triage(deps: WatchCommandDeps) -> None:
    try:
        deps.post_reboot_triage_announce()
    except Exception as exc:
        deps.notify("post_reboot_triage_error", {"error": str(exc)[:200]}, feishu_enabled=False)


def cmd_watch_locked(args, *, deps: WatchCommandDeps) -> None:
    """Run the watcher after the process-wide watcher lifetime lock is held."""
    deps.reset_watcher_shutdown()
    deps.set_hard_rule_mode(getattr(args, "hard_rule_mode", None))
    deps.configure_algorithm(getattr(args, "algorithm", None) or None)
    _refuse_duplicate_watcher(deps)

    stop_flag = {"stop": False}

    def _signal(sig, frame):
        stop_flag["stop"] = True
        deps.request_watcher_shutdown(sig, frame)

    deps.register_signal_handler(deps.sigterm, _signal)
    deps.register_signal_handler(deps.sigint, _signal)

    me = _write_initial_state(args, deps)
    deps.notify("watcher_started", me)
    _run_env_smoke(args, deps)
    _recover_stale_launching_at_start(deps)
    _announce_post_reboot_triage(deps)
    try:
        eta_periodic_started = bool(deps.start_eta_periodic_refresh())
        if eta_periodic_started:
            deps.notify(
                "eta_periodic_refresh_started",
                {"started": True},
                feishu_enabled=False,
            )
    except Exception as exc:
        deps.notify(
            "eta_periodic_refresh_start_error",
            {"error": str(exc)[:200]},
            feishu_enabled=False,
        )

    try:
        while not stop_flag["stop"]:
            iteration_started_at = deps.now()
            try:
                deps.watch_iteration(args)
            except deps.watcher_shutdown_requested_type:
                break
            except Exception as exc:
                deps.notify("iteration_error", {"error": str(exc)[:300]}, feishu_enabled=False)
            elapsed = max(0.0, deps.now() - iteration_started_at)
            remaining = max(0.0, float(args.interval) - elapsed)
            while remaining > 0:
                if stop_flag["stop"]:
                    break
                step = min(1.0, remaining)
                deps.sleep(step)
                remaining -= step
    finally:
        try:
            deps.stop_eta_periodic_refresh()
        except Exception as exc:
            deps.notify(
                "eta_periodic_refresh_stop_error",
                {"error": str(exc)[:200]},
                feishu_enabled=False,
            )
        deps.notify("watcher_stopped", {"pid": me["pid"]})
        try:
            deps.unlink_watcher_state()
        except Exception:
            pass
