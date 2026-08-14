from skill.scheduler_watch.health import (
    decide_watcher_health,
    watcher_phase_stale_after,
)


def _state(**overrides):
    state = {
        "pid": 123,
        "started_at": 100.0,
        "last_progress_ts": 900.0,
        "last_resource_log_ts": 850.0,
        "last_heartbeat_ts": 800.0,
        "phase": "state_update_and_dispatch_plan",
    }
    state.update(overrides)
    return state


def test_health_starts_missing_service():
    decision = decide_watcher_health(
        service_active=False,
        main_pid=0,
        process_alive=False,
        process_age_s=0,
        state=None,
        now=1000.0,
    )
    assert decision.action == "start"


def test_health_restarts_live_but_stale_watcher():
    decision = decide_watcher_health(
        service_active=True,
        main_pid=123,
        process_alive=True,
        process_age_s=4000.0,
        state=_state(
            started_at=1.0,
            last_progress_ts=1.0,
            last_resource_log_ts=1.0,
            last_heartbeat_ts=1.0,
        ),
        now=1000.0,
        stale_after_s=900.0,
    )
    assert decision.action == "restart"
    assert decision.phase == "state_update_and_dispatch_plan"


def test_health_does_not_restart_current_or_starting_watcher():
    current = decide_watcher_health(
        service_active=True,
        main_pid=123,
        process_alive=True,
        process_age_s=4000.0,
        state=_state(),
        now=1000.0,
    )
    starting = decide_watcher_health(
        service_active=True,
        main_pid=123,
        process_alive=True,
        process_age_s=30.0,
        state=None,
        now=1000.0,
    )
    assert current.action == "none"
    assert starting.action == "none"


def test_health_gives_launch_and_staging_longer_deadline():
    assert watcher_phase_stale_after("launching", 900.0) == 3600.0
    assert watcher_phase_stale_after("staging", 900.0) == 3600.0
    decision = decide_watcher_health(
        service_active=True,
        main_pid=123,
        process_alive=True,
        process_age_s=4000.0,
        state=_state(phase="launching", last_progress_ts=1.0),
        now=2000.0,
    )
    assert decision.action == "none"


def test_health_restarts_mismatched_state_pid_after_grace():
    decision = decide_watcher_health(
        service_active=True,
        main_pid=123,
        process_alive=True,
        process_age_s=4000.0,
        state=_state(pid=456),
        now=1000.0,
    )
    assert decision.action == "restart"
