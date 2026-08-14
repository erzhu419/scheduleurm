from __future__ import annotations

from pathlib import Path

from skill.scheduler_config import build_scheduler_config_exports


def test_config_exports_defaults_and_paths(tmp_path):
    cfg = build_scheduler_config_exports(environ={}, home=tmp_path)

    assert cfg["STATE_DIR"] == Path(tmp_path) / ".claude" / "scheduler"
    assert cfg["QUEUE_FILE"] == cfg["STATE_DIR"] / "queue.json"
    assert cfg["DEFAULT_VRAM_MB"] == 512
    assert cfg["DEFAULT_RAM_MB"] == 4096
    assert cfg["MIGRATION_MAX_PER_DISPATCH"] == 1
    assert cfg["LAUNCH_FAILED_NODE_TTL_S"] == 90
    assert cfg["RUNNING_PROBE_MIN_INTERVAL_S"] == 30
    assert cfg["RUNNING_PROBE_MAX_WORKERS"] == 8
    assert cfg["RUNNING_PROBE_TIMEOUT_S"] == 25
    assert cfg["CANCEL_KILL_MAX_WORKERS"] == 4
    assert cfg["WATCH_DISPATCH_MAX_QUEUED_PER_CYCLE"] == 1024
    assert cfg["LAUNCH_EXEC_MAX_WORKERS"] == 48
    assert cfg["LAUNCH_EXEC_MAX_PER_NODE"] == 12
    assert cfg["LAUNCH_STAGING_MAX_CANDIDATES_PER_PASS"] == 256
    assert cfg["LAUNCH_STAGING_WORKERS"] == 12
    assert cfg["ETA_TAIL_PROBE_TIMEOUT_S"] == 10
    assert cfg["ETA_REFRESH_MIN_INTERVAL_S"] == 60
    assert cfg["ETA_ANALYSIS_INTERVAL_S"] == 60
    assert cfg["ETA_ANALYSIS_DIR"] == cfg["LOG_DIR"] / "eta_analysis"
    assert cfg["WATCH_RESPECTS_DISPATCH_INTENT"] is False
    assert cfg["DISPATCH_PROBE_CACHE_MAX_AGE_S"] == 30
    assert cfg["WATCH_PROBE_CACHE_MAX_AGE_S"] == 30
    assert cfg["WATCH_PRE_DISPATCH_PROBE_MAX_AGE_S"] == 20
    assert cfg["PROBE_ALL_MAX_WORKERS"] == 8
    assert cfg["PROBE_ROUTE_MAX_WORKERS"] == 4
    assert cfg["TERMINAL_DIAGNOSTICS_MAX_PER_CYCLE"] == 8
    assert cfg["TERMINAL_DIAGNOSTICS_MAX_WALL_S"] == 20
    assert cfg["TERMINAL_DIAGNOSTICS_INCLUDE_LOG_ARTIFACTS"] is False
    assert cfg["TERMINAL_EVIDENCE_MAX_WORKERS"] == 4
    assert cfg["TERMINAL_EVIDENCE_BATCH_SIZE"] == 32
    assert cfg["SSH_ROUTE_CACHE_TTL_S"] == 300
    assert cfg["SSH_LOGIN_FAILURE_CACHE_TTL_S"] == 60
    assert cfg["RECOVER_QUEUED_LIVE_REMOTE_NODES"] is False
    assert cfg["SSH_LOGIN_PREFLIGHT_TIMEOUT_S"] == 15
    assert cfg["LAUNCH_MAX_CWD_SIZE_MB"] == 2048
    assert cfg["LAUNCH_STAGING_MAX_CANDIDATES_PER_PASS"] == 256
    assert cfg["LAUNCH_STAGING_WORKERS"] == 12
    assert cfg["RESULT_SYNC_MAX_ATTEMPTS"] == 5
    assert cfg["REBOOT_DETECT_WINDOW_S"] == 600
    assert "CUDA out of memory" in cfg["OOM_PATTERNS"]


def test_config_exports_respects_env_overrides(tmp_path):
    cfg = build_scheduler_config_exports(
        environ={
            "SCHEDULEURM_DEFER_LAUNCH_OUTSIDE_LOCK": "0",
            "SCHEDULEURM_LAUNCHING_RESET_S": "77",
            "SCHEDULEURM_MIGRATION_MIN_TASK_ETA_S": "444",
            "SCHEDULEURM_RESUME_CKPT_MIGRATE_MIN_WAIT_S": "555",
            "SCHEDULEURM_RESULT_SYNC_MAX_ATTEMPTS": "9",
            "SCHEDULEURM_LAUNCH_FAILED_NODE_TTL_S": "12",
            "SCHEDULEURM_RUNNING_PROBE_MIN_INTERVAL_S": "17",
            "SCHEDULEURM_RUNNING_PROBE_MAX_WORKERS": "5",
            "SCHEDULEURM_RUNNING_PROBE_TIMEOUT_S": "11",
            "SCHEDULEURM_CANCEL_KILL_MAX_WORKERS": "7",
            "SCHEDULEURM_WATCH_DISPATCH_MAX_QUEUED_PER_CYCLE": "33",
            "SCHEDULEURM_ETA_TAIL_PROBE_TIMEOUT_S": "6",
            "SCHEDULEURM_ETA_REFRESH_MIN_INTERVAL_S": "45",
            "SCHEDULEURM_ETA_ANALYSIS_INTERVAL_S": "30",
            "SCHEDULEURM_WATCH_RESPECTS_DISPATCH_INTENT": "true",
            "SCHEDULEURM_WATCH_PROBE_CACHE_MAX_AGE_S": "12",
            "SCHEDULEURM_WATCH_PRE_DISPATCH_PROBE_MAX_AGE_S": "7",
            "SCHEDULEURM_PROBE_ALL_MAX_WORKERS": "6",
            "SCHEDULEURM_PROBE_ROUTE_MAX_WORKERS": "3",
            "SCHEDULEURM_TERMINAL_DIAGNOSTICS_MAX_PER_CYCLE": "7",
            "SCHEDULEURM_TERMINAL_DIAGNOSTICS_MAX_WALL_S": "13.5",
            "SCHEDULEURM_TERMINAL_DIAGNOSTICS_INCLUDE_LOG_ARTIFACTS": "true",
            "SCHEDULEURM_TERMINAL_EVIDENCE_MAX_WORKERS": "4",
            "SCHEDULEURM_TERMINAL_EVIDENCE_BATCH_SIZE": "16",
            "SCHEDULEURM_RECOVER_QUEUED_LIVE_REMOTE_NODES": "true",
            "SCHEDULEURM_SSH_DISABLE_MUX": "false",
            "SCHEDULEURM_SSH_LOGIN_FAILURE_CACHE_TTL_S": "17",
            "SCHEDULEURM_LAUNCH_STAGING_MAX_CANDIDATES_PER_PASS": "9",
            "SCHEDULEURM_LAUNCH_STAGING_WORKERS": "3",
        },
        home=tmp_path,
    )

    assert cfg["DEFER_LAUNCH_OUTSIDE_LOCK"] is False
    assert cfg["LAUNCHING_RESET_S"] == 77
    assert cfg["MIGRATION_MIN_TASK_ETA_S"] == 444
    assert cfg["RESUME_CKPT_MIGRATE_MIN_WAIT_S"] == 555
    assert cfg["RESULT_SYNC_MAX_ATTEMPTS"] == 9
    assert cfg["LAUNCH_FAILED_NODE_TTL_S"] == 12
    assert cfg["RUNNING_PROBE_MIN_INTERVAL_S"] == 17
    assert cfg["RUNNING_PROBE_MAX_WORKERS"] == 5
    assert cfg["RUNNING_PROBE_TIMEOUT_S"] == 11
    assert cfg["CANCEL_KILL_MAX_WORKERS"] == 7
    assert cfg["WATCH_DISPATCH_MAX_QUEUED_PER_CYCLE"] == 33
    assert cfg["ETA_TAIL_PROBE_TIMEOUT_S"] == 6
    assert cfg["ETA_REFRESH_MIN_INTERVAL_S"] == 45
    assert cfg["ETA_ANALYSIS_INTERVAL_S"] == 30
    assert cfg["WATCH_RESPECTS_DISPATCH_INTENT"] is True
    assert cfg["WATCH_PROBE_CACHE_MAX_AGE_S"] == 12
    assert cfg["WATCH_PRE_DISPATCH_PROBE_MAX_AGE_S"] == 7
    assert cfg["PROBE_ALL_MAX_WORKERS"] == 6
    assert cfg["PROBE_ROUTE_MAX_WORKERS"] == 3
    assert cfg["TERMINAL_DIAGNOSTICS_MAX_PER_CYCLE"] == 7
    assert cfg["TERMINAL_DIAGNOSTICS_MAX_WALL_S"] == 13.5
    assert cfg["TERMINAL_DIAGNOSTICS_INCLUDE_LOG_ARTIFACTS"] is True
    assert cfg["TERMINAL_EVIDENCE_MAX_WORKERS"] == 4
    assert cfg["TERMINAL_EVIDENCE_BATCH_SIZE"] == 16
    assert cfg["RECOVER_QUEUED_LIVE_REMOTE_NODES"] is True
    assert cfg["SSH_DISABLE_MUX"] is False
    assert cfg["SSH_LOGIN_FAILURE_CACHE_TTL_S"] == 17
    assert cfg["LAUNCH_STAGING_MAX_CANDIDATES_PER_PASS"] == 9
    assert cfg["LAUNCH_STAGING_WORKERS"] == 3


def test_config_exports_new_runtime_caches_per_build(tmp_path):
    first = build_scheduler_config_exports(environ={}, home=tmp_path)
    second = build_scheduler_config_exports(environ={}, home=tmp_path)

    first["_SSH_ROUTE_JUMP_CACHE"]["node001"] = "jump"

    assert second["_SSH_ROUTE_JUMP_CACHE"] == {}
    assert first["_SSH_ROUTE_JUMP_CACHE"] is not second["_SSH_ROUTE_JUMP_CACHE"]
