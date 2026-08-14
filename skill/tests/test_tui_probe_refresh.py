import json
import os
from types import SimpleNamespace

from skill import tui


def test_node_probe_cache_snapshot_keeps_inventory_visible(monkeypatch, tmp_path):
    cache_file = tmp_path / "node_probe_cache.json"
    cache_file.write_text("{}")
    monkeypatch.setattr(tui.sch, "NODE_PROBE_CACHE_FILE", cache_file)
    monkeypatch.setattr(tui.sch, "NODES", {
        "node001": {},
        "node002": {},
        "jump": {"monitor_only": True},
    })
    monkeypatch.setattr(tui.sch, "_load_node_probe_cache", lambda max_age_s: {
        "node001": {
            "name": "node001",
            "alive": False,
            "error": "timeout",
            "probe_fallback": "stale_node_probe_cache",
            "probe_cache_age_s": 4,
        },
    })

    snap = tui._node_probe_cache_snapshot(300)

    assert [node["name"] for node in snap["nodes"]] == ["node001", "node002"]
    assert snap["nodes"][0]["error"] == "timeout"
    assert "probe_fallback" not in snap["nodes"][0]
    assert snap["nodes"][1]["probe_pending"] is True
    assert snap["complete"] is False


def test_probe_snapshot_uses_complete_watcher_cache_without_live_probe(monkeypatch):
    monkeypatch.setattr(tui, "_node_probe_cache_snapshot", lambda max_age_s: {
        "nodes": [{"name": "node001", "alive": False, "error": "offline"}],
        "ts": 123.0,
        "complete": True,
        "source": "watcher-cache",
    })
    monkeypatch.setattr(
        tui.sch,
        "probe_all",
        lambda: (_ for _ in ()).throw(AssertionError("fresh cache must avoid live SSH")),
    )
    monkeypatch.setattr(tui, "_load_display_state", lambda: ({"tasks": [{"id": "t1"}]}, {"sig": {}}))

    snap = tui._probe_snapshot()

    assert snap["nodes"][0]["name"] == "node001"
    assert snap["state"]["tasks"][0]["id"] == "t1"
    assert snap["ts"] == 123.0


def test_probe_snapshot_reads_task_state_after_slow_live_probe(monkeypatch):
    calls = []
    monkeypatch.setattr(tui, "_node_probe_cache_snapshot", lambda max_age_s: {
        "nodes": [{"name": "node001", "alive": False, "probe_pending": True}],
        "ts": 0.0,
        "complete": False,
        "source": "pending",
    })
    monkeypatch.setattr(
        tui.sch,
        "probe_all",
        lambda: calls.append("probe") or [{"name": "node001", "alive": False, "error": "offline"}],
    )
    monkeypatch.setattr(
        tui,
        "_load_display_state",
        lambda: calls.append("state") or ({"tasks": []}, {}),
    )

    snap = tui._probe_snapshot()

    assert calls == ["probe", "state"]
    assert snap["node_source"] == "live-probe"


def test_fast_snapshot_merge_does_not_replace_newer_live_telemetry():
    previous = {
        "state": {"tasks": [{"id": "old"}]},
        "hist": {},
        "nodes": [{"name": "node001", "alive": True}],
        "ts": 200.0,
        "node_source": "live-probe",
    }
    fast = {
        "state": {"tasks": [{"id": "new"}]},
        "hist": {},
        "nodes": [{"name": "node001", "alive": False}],
        "ts": 100.0,
        "node_source": "watcher-cache",
    }

    merged = tui._merge_fast_snapshot(previous, fast)

    assert merged["state"]["tasks"][0]["id"] == "new"
    assert merged["nodes"][0]["alive"] is True
    assert merged["ts"] == 200.0


def test_kick_probe_never_starts_an_overlapping_generation(monkeypatch):
    calls = []
    fake_app = SimpleNamespace(
        _probing=True,
        _refresh_fast_state=lambda: calls.append("fast"),
        run_worker=lambda *args, **kwargs: calls.append("worker"),
    )
    monkeypatch.setattr(tui, "_scheduler_source_changed", lambda: False)

    tui.SchedulerTUI._kick_probe(fake_app)

    assert calls == ["fast"]


def test_node_summary_distinguishes_pending_from_down():
    summary = tui._node_summary_line([
        {"name": "node001", "alive": False, "probe_pending": True, "error": "first probe"},
        {"name": "node002", "alive": False, "error": "timeout"},
    ])

    assert "node001" in summary and "WAIT" in summary
    assert "node002" in summary and "DOWN" in summary


def test_node_summary_rounds_gpu_memory_percentage_like_status():
    summary = tui._node_summary_line([{
        "name": "node007",
        "alive": True,
        "free_cpu": 64,
        "total_cpu": 64,
        "free_ram_mb": 100_000,
        "total_ram_mb": 192_000,
        "gpus": [{
            "idx": 3,
            "used_mb": 430,
            "total_mb": 11_264,
            "free_mb": 10_834,
            "util_pct": 0,
        }],
    }])

    assert "0.42GB/11.0GB" in summary
    assert " 4%" in summary


def test_terminal_task_does_not_display_stale_eta():
    task = {
        "status": "cancelled",
        "eta_seconds": 636 * 3600,
        "eta_source": "growing_iter_cost",
    }

    assert tui._fmt_eta(task, {}) == "-"


def test_live_eta_displays_unified_source_and_iteration_progress():
    task = {
        "status": "running",
        "eta_seconds": 7200,
        "eta_source": "saas_native_eta",
        "runtime_current_unit": 50,
        "runtime_total_units": 80,
    }

    assert tui._fmt_eta(task, {}) == "~2.0h live 50/80"


def test_history_overrun_does_not_display_bounded_floor_as_eta():
    task = {
        "status": "running",
        "eta_seconds": 300,
        "eta_source": "runtime_history_overrun",
    }

    assert tui._fmt_eta(task, {}) == "? hist"


def test_completed_live_progress_displays_finishing_state():
    task = {
        "status": "running",
        "eta_seconds": 0,
        "eta_source": "progress_window",
        "runtime_current_unit": 190,
        "runtime_total_units": 190,
    }

    assert tui._fmt_eta(task, {}) == "finishing live"


def test_node_tail_summary_shows_live_cpu_separately_from_admission_controls():
    summary = tui._node_tail_summary({
        "free_cpu": 0,
        "cpu_slot_free": 0,
        "total_cpu": 192,
        "observed_free_cpu": 65,
        "cpu_hard_free": 35,
        "cpu_hard_reserved": 30,
        "loadavg": 127.0,
        "observed_loadavg": 127.0,
        "free_ram_mb": 170_000,
        "total_ram_mb": 192_000,
    })

    assert "cpu=65/192 live(slot=0,guard=35,startup=30)" in summary
    assert "load=127.0" in summary


def test_node_summary_exposes_measurement_reservation_before_free_resources():
    summary = tui._node_tail_summary({
        "name": "jtl110gpu2",
        "measurement_reservation_active": True,
        "measurement_reservation": {
            "purpose": "theorem-facing natural-completion GPU calibration",
        },
        "free_cpu": 10,
        "total_cpu": 12,
        "free_ram_mb": 500_000,
        "total_ram_mb": 512_000,
    })

    assert summary.startswith(
        "MEASUREMENT-RESERVED(theorem-facing natural-completion GPU calibration)"
    )


def test_watcher_status_prefers_in_cycle_progress_over_old_heartbeat(monkeypatch, tmp_path):
    watcher_state = tmp_path / "watcher_state.json"
    watcher_state.write_text(json.dumps({
        "pid": os.getpid(),
        "started_at": 100.0,
        "last_heartbeat_ts": 100.0,
        "last_resource_log_ts": 100.0,
        "last_progress_ts": 995.0,
        "phase": "running_task_probe",
        "resource_log_interval": 300,
    }))
    monkeypatch.setattr(tui.sch, "WATCHER_STATE", watcher_state)
    monkeypatch.setattr(tui.time, "time", lambda: 1000.0)

    assert tui._watcher_status_line() == ""
