from __future__ import annotations

from skill.scheduler_watch_phases import WatchPhaseRecorder


def test_watch_phase_recorder_skips_phases_below_threshold():
    clock = iter([10.9])
    recorder = WatchPhaseRecorder(1.0, clock=lambda: next(clock))

    recorded = recorder.record("load_state", 10.0, tasks=3)

    assert recorded is None
    assert not recorder
    assert recorder.payload() == {"threshold_s": 1.0, "phases": []}


def test_watch_phase_recorder_records_slow_phase_with_scheduler_payload_shape():
    clock = iter([13.4567])
    recorder = WatchPhaseRecorder(3.0, clock=lambda: next(clock))

    recorded = recorder.record(
        "update_running_tasks",
        10.0,
        probed=2,
        ignored=None,
    )

    assert recorded == {
        "phase": "update_running_tasks",
        "seconds": 3.457,
        "probed": 2,
    }
    assert bool(recorder)
    assert recorder.payload() == {
        "threshold_s": 3.0,
        "phases": [recorded],
    }
