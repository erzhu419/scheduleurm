"""Replay cold whole selectors inside a fixed virtual computation window.

The FIFO and finite-cutoff rules come from the existing delayed replay. Actual
wall and CPU costs are measured separately: spare window time is virtual wait
padding, and a wall overrun advances the clock by its full measured duration.
"""
from __future__ import annotations

from math import isfinite
from time import perf_counter_ns, process_time_ns

from algorithm.experiments.integer_capacity_decision_delay_replay import (
    simulate_delayed_trajectory,
)
from algorithm.experiments.small_dimension_exact_native_decision_delay_audit import (
    audit_trajectory,
)


def simulate_common_cadence_trajectory(
    model, tape, arrival_horizon, observation_end, budget, selector,
    computation_window_s, policy="candidate", wall_clock=perf_counter_ns,
    cpu_clock=process_time_ns,
):
    """Pay ``max(window, actual wall)`` while retaining each original result.

    A fresh whole call is made at every decision, including repeated queues and
    terminal decisions which finish too late to dispatch. No real sleep or
    selector/reference cache is introduced by the computation window. The
    offline ``full_reference`` policy keeps the fixed window even if its actual
    solve exceeds it; that policy is a normative trajectory, not a timed arm.
    """
    if not isfinite(computation_window_s) or computation_window_s <= 0:
        raise ValueError("computation window must be finite and positive")
    window_ns = round(computation_window_s * 1_000_000_000)
    measurements = []
    virtual_ns = 0

    def measured_selector(options, queue, lanes, cap):
        nonlocal virtual_ns
        wall_started, cpu_started = wall_clock(), cpu_clock()
        result = selector(options, queue, lanes, cap)
        cpu_ns = cpu_clock() - cpu_started
        wall_ns = wall_clock() - wall_started
        if not (isfinite(wall_ns) and isfinite(cpu_ns)) or wall_ns < 0 or cpu_ns < 0:
            raise ValueError("selector clocks must be finite nondecreasing nanoseconds")
        wall_s, cpu_s = wall_ns / 1_000_000_000, cpu_ns / 1_000_000_000
        measurements.append({
            "selector_wall_s": wall_s, "selector_cpu_s": cpu_s,
            "window_padding_s": max(0.0, computation_window_s - wall_s),
            "window_overrun_s": max(0.0, wall_s - computation_window_s),
        })
        virtual_ns += window_ns if policy == "full_reference" else max(window_ns, wall_ns)
        return result

    trajectory, selections = simulate_delayed_trajectory(
        model, tape, arrival_horizon, observation_end, budget, measured_selector,
        policy=policy, clock=lambda: virtual_ns,
    )
    for frame, cost in zip(trajectory["frames"], measurements, strict=True):
        frame.update(cost)
    trajectory.update({
        "computation_window_s": computation_window_s,
        "total_selector_wall_s": sum(cost["selector_wall_s"] for cost in measurements),
        "total_selector_cpu_s": sum(cost["selector_cpu_s"] for cost in measurements),
        "total_window_padding_s": sum(cost["window_padding_s"] for cost in measurements),
        "total_window_overrun_s": sum(cost["window_overrun_s"] for cost in measurements),
        "window_overrun_count": sum(cost["window_overrun_s"] > 0 for cost in measurements),
    })
    return trajectory, selections


def simulate_fixed_reference_trajectory(
    model, tape, arrival_horizon, observation_end, budget, selector,
    computation_window_s, wall_clock=perf_counter_ns, cpu_clock=process_time_ns,
):
    """Measure offline reference cost while advancing by its fixed window.

    Reference overruns are recorded, but cannot alter this canonical baseline's
    decision clock or count as a deployment deadline failure.
    """
    return simulate_common_cadence_trajectory(
        model, tape, arrival_horizon, observation_end, budget, selector,
        computation_window_s, policy="full_reference", wall_clock=wall_clock,
        cpu_clock=cpu_clock,
    )


def audit_common_cadence_trajectory(model, trajectory, selections, reference_audit, protocol):
    """Keep native evidence checks and charge only actual runtime to metrics."""
    audit_trajectory(model, trajectory, selections, reference_audit, protocol)
    trajectory["selector_metrics"]["operational_elapsed_s"] = trajectory["total_selector_wall_s"]
    trajectory["selector_metrics"]["operational_cpu_s"] = trajectory["total_selector_cpu_s"]
    return trajectory
