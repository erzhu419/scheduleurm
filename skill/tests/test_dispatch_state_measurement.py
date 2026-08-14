from __future__ import annotations

import pytest

from algorithm.experiments import file_progress_completion_wrapper as wrapper
from algorithm.experiments import native_cpu_colocation_completion_matrix as colocation


def test_dispatch_cpu_sample_uses_multiple_fixed_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counters = iter(
        (
            (100, 1_000),
            (130, 1_100),
            (130, 1_100),
            (150, 1_200),
            (150, 1_200),
            (160, 1_300),
        )
    )
    clocks = iter((20.0, 23.0))
    sleep_calls: list[float] = []
    monkeypatch.setattr(wrapper, "_cpu_counters", lambda: next(counters))
    monkeypatch.setattr(wrapper.time, "perf_counter", lambda: next(clocks))
    monkeypatch.setattr(wrapper.time, "sleep", sleep_calls.append)
    monkeypatch.setattr(wrapper.os, "cpu_count", lambda: 8)

    state = wrapper._sample_dispatch_cpu_state(
        window_s=1.0,
        sample_count=3,
    )

    assert sleep_calls == [1.0, 1.0, 1.0]
    assert state["dispatch_state_ready"] is True
    assert state["dispatch_state_observed_before_child"] is True
    assert state["dispatch_sample_window_s"] == pytest.approx(3.0)
    assert state["dispatch_sample_subwindow_s"] == pytest.approx(1.0)
    assert state["dispatch_sample_count"] == 3
    assert state["dispatch_system_busy_fraction_samples"] == pytest.approx(
        [0.3, 0.2, 0.1]
    )
    assert state["dispatch_system_busy_fraction"] == pytest.approx(0.2)
    assert state["dispatch_system_busy_fraction_median"] == pytest.approx(0.2)
    assert state["dispatch_system_busy_fraction_p95"] == pytest.approx(0.3)
    assert state["dispatch_system_busy_core_equiv"] == pytest.approx(1.6)
    assert state["dispatch_external_cpu_core_equiv"] == pytest.approx(1.6)
    assert state["dispatch_external_cpu_regime"] == "cpu_external_moderate"


def test_colocation_dispatch_sample_relabels_pre_child_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        colocation,
        "_sample_child_dispatch_cpu_state",
        lambda **_: {
            "dispatch_state_ready": True,
            "dispatch_state_observed_before_child": True,
            "dispatch_sample_count": 5,
            "dispatch_external_cpu_regime": "cpu_external_light",
            "dispatch_external_cpu_core_equiv": 1.5,
        },
    )

    state = colocation._sample_dispatch_cpu_state(
        window_s=1.0,
        sample_count=5,
    )

    assert "dispatch_state_observed_before_child" not in state
    assert state["dispatch_state_observed_before_controlled_group"] is True
    assert state["dispatch_sample_count"] == 5
    assert state["dispatch_external_cpu_regime"] == "cpu_external_light"
