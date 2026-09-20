from __future__ import annotations

import pytest

from algorithm.experiments.unified_hardware_or_replay import (
    EvaluationAction,
    LowerAction,
    PolicyDescriptor,
    ReplayAction,
    Scenario,
    TraceJob,
    TraceSpec,
    _common_complete_runs,
    _run_policy,
)


CPU = "cpu_heavy_local_bench"
HYBRID = "hybrid_rl_resac_ant"
POLICY = PolicyDescriptor("ours", "ours", "robust_maxweight", {})


def _scenario(workload=CPU, widths=(3, 4), lanes=1):
    actions = []
    for width in widths:
        rate = 2.0 if width == min(widths) else 1.0
        action_id = f"batch_{width}"
        actions.append(ReplayAction(
            LowerAction(
                action_id=action_id,
                workload_key=workload,
                profile=width,
                node_bucket="fixture",
                resource_state="idle",
                resident_mix="none",
                lower_service=rate,
                lower_service_vector=((workload, rate),),
                penalty_units=0.0,
            ),
            EvaluationAction(
                action_id=action_id,
                phase_service_rate=rate,
                completion_group_units=float(width),
                replay_job_units=1.0,
                work_unit="fixture_units",
                fixed_overhead_s=0.0,
                completion_model_sample_count=3,
                eta_source="synthetic_fixture",
                completion_model_kind="natural_completion",
            ),
        ))
    return Scenario(
        scenario_id="causal_fixture",
        quadrant="q11" if workload == HYBRID else "q10",
        node_bucket="fixture",
        resource_count=lanes,
        workload_keys=(workload,),
        actions=tuple(actions),
    )


def _run(scenario, *, initial_count=7, suffix_count=0, suffix_arrival=100.0):
    workload = scenario.workload_keys[0]
    jobs = tuple(
        TraceJob(f"prefix_{index}", workload, 0.0, 1.0)
        for index in range(initial_count)
    ) + tuple(
        TraceJob(f"suffix_{index}", workload, suffix_arrival, 1.0 + index / 4)
        for index in range(suffix_count)
    )
    trace = TraceSpec(
        trace_id="same_history",
        arrival_family="bursty",
        arrival_process="bursty",
        load_factor=0.5,
        seed=0,
        jobs=jobs,
        workload_unit_scales=((workload, 1.0),),
    )
    return _run_policy(
        scenario=scenario, trace=trace, policy=POLICY, migration_mode="without_migration"
    )


def _prefix(row):
    return [audit for audit in row["selection_audit"] if audit["frame_start_s"] < 20.0]


@pytest.mark.parametrize("suffix_count,suffix_arrival", [(1, 100.0), (3, 50.0), (6, 200.0)])
def test_widths_three_four_current_seven_have_identical_candidates_and_selection(
    suffix_count, suffix_arrival
):
    scenario = _scenario()
    without_suffix = _run(scenario)
    with_suffix = _run(scenario, suffix_count=suffix_count, suffix_arrival=suffix_arrival)

    assert _prefix(without_suffix) == _prefix(with_suffix)
    first = with_suffix["selection_audit"][0]
    assert first["ready_count_before"] == 7
    assert set(first["feasible_action_ids"]) == {"batch_3", "batch_4"}
    assert first["action_id"] == "batch_3"
    assert without_suffix["completion_status"] == "complete"
    if suffix_count == 1:
        assert with_suffix["completion_status"] == "incomplete_tail"
        assert with_suffix["unfinished_jobs"] == 1


@pytest.mark.parametrize("lanes", [2, 3])
def test_same_history_keeps_all_lane_decisions_identical_before_different_suffixes(lanes):
    scenario = _scenario(lanes=lanes)
    reference = _run(scenario, initial_count=14)
    changed = _run(scenario, initial_count=14, suffix_count=1, suffix_arrival=100.0)

    assert len(_prefix(reference)) >= 4
    assert len({audit["resource_lane"] for audit in _prefix(reference)}) == lanes
    assert _prefix(reference) == _prefix(changed)
    assert reference["completion_status"] == "complete"
    assert changed["completion_status"] == "incomplete_tail"


def test_registered_q11_widths_two_five_keep_unknown_singleton_tail_unfinished():
    scenario = _scenario(workload=HYBRID, widths=(2, 5))
    complete_prefix = _run(scenario)
    unknown_singleton = _run(scenario, suffix_count=1)

    assert _prefix(complete_prefix) == _prefix(unknown_singleton)
    assert [audit["action_id"] for audit in _prefix(unknown_singleton)] == ["batch_2", "batch_5"]
    assert unknown_singleton["job_count"] == 8
    assert unknown_singleton["completed_jobs"] == 7
    assert unknown_singleton["unfinished_jobs"] == 1
    assert unknown_singleton["completion_status"] == "incomplete_tail"
    assert unknown_singleton["observation_horizon_s"] == 100.0
    assert unknown_singleton["makespan_s"] is None
    assert unknown_singleton["mean_flow_s"] is None
    assert unknown_singleton["p90_flow_s"] is None


def test_incomplete_policy_excludes_the_same_trace_from_every_performance_comparator():
    rows = [
        {
            "scenario_id": "fixture",
            "trace_id": trace,
            "migration_mode": "without_migration",
            "policy": policy,
            "completion_status": status,
        }
        for trace, policy, status in (
            ("shared_complete", "ours", "complete"),
            ("shared_complete", "baseline", "complete"),
            ("partial", "ours", "complete"),
            ("partial", "baseline", "incomplete_tail"),
        )
    ]

    retained = _common_complete_runs(rows)

    assert retained == rows[:2]
