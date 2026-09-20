from __future__ import annotations

from dataclasses import replace
import random

import pytest

from algorithm.experiments import unified_hardware_or_replay as replay


WORKLOADS = (
    "gpu_heavy_jax_matmul",
    "gpu_cnn_torch_resnet50",
    "gpu_llm_distilgpt2",
)


def _scenario(workloads=WORKLOADS):
    actions = []
    for index, workload in enumerate(workloads, start=1):
        for profile, rate in ((1, 2.0 * index), (2, 3.0 * index)):
            action_id = f"{workload}:{profile}"
            actions.append(
                replay.ReplayAction(
                    lower=replay.LowerAction(
                        action_id=action_id,
                        workload_key=workload,
                        profile=profile,
                        node_bucket="dual_gpu",
                        resource_state="idle",
                        resident_mix="none",
                        lower_service=rate,
                        lower_service_vector=((workload, rate),),
                        penalty_units=0.0,
                    ),
                    evaluation=replay.EvaluationAction(
                        action_id=action_id,
                        phase_service_rate=rate,
                        completion_group_units=10.0 * index * profile,
                        replay_job_units=10.0 * index,
                        work_unit="native_step",
                        fixed_overhead_s=0.0,
                        completion_model_sample_count=5,
                        eta_source="synthetic_natural_completion",
                        completion_model_kind="synthetic_phase_model",
                    ),
                )
            )
    # A much faster loaded action must not redefine the dedicated reference.
    first = actions[0]
    actions.append(
        replace(
            first,
            lower=replace(
                first.lower,
                action_id="loaded:fast",
                loaded_action=True,
                lower_service=1000.0,
                lower_service_vector=((workloads[0], 1000.0),),
            ),
        )
    )
    return replay.Scenario(
        scenario_id="q01:dual_gpu",
        quadrant="q01",
        node_bucket="dual_gpu",
        resource_count=2,
        workload_keys=tuple(workloads),
        actions=tuple(actions),
    )


def _protocol(load=0.85, process="poisson"):
    return {
        "arrival_family": "load_sweep",
        "arrival_process": process,
        "load_factor": load,
        "seed": 41,
    }


@pytest.mark.parametrize("load", (0.50, 0.85, 0.95))
@pytest.mark.parametrize("process", ("poisson", "bursty"))
def test_mixed_q01_arrival_intensities_share_one_joint_load(monkeypatch, load, process):
    scenario = _scenario()
    arrival_inputs = []
    generate_arrivals = replay._arrival_times

    def capture_arrivals(**kwargs):
        arrival_inputs.append(kwargs)
        return generate_arrivals(**kwargs)

    monkeypatch.setattr(replay, "_arrival_times", capture_arrivals)
    trace = replay._build_trace(scenario, _protocol(load, process), jobs_per_workload=12)
    reference = replay._dedicated_load_reference(scenario)

    assert len(trace.jobs) == 36
    assert reference["workload_time_shares"] == dict.fromkeys(WORKLOADS, 1.0 / 3.0)
    assert reference["dedicated_lower_service_per_lane"] == dict(zip(WORKLOADS, (3.0, 6.0, 9.0)))
    assert reference["reference_work_rates"] == dict(zip(WORKLOADS, (2.0, 4.0, 6.0)))
    joint_load = 0.0
    for workload, inputs in zip(WORKLOADS, arrival_inputs):
        work_arrival_rate = inputs["load_factor"] * inputs["lower_capacity"]
        exclusive_capacity = 2 * reference["dedicated_lower_service_per_lane"][workload]
        joint_load += work_arrival_rate / exclusive_capacity
    assert joint_load == pytest.approx(load)


@pytest.mark.parametrize("process", ("poisson", "bursty"))
def test_single_class_preserves_exclusive_capacity_arrivals(process):
    scenario = _scenario(WORKLOADS[:1])
    protocol = _protocol(process=process)
    trace = replay._build_trace(scenario, protocol, jobs_per_workload=12)
    rng = random.Random(f"{scenario.scenario_id}|load_sweep|{process}|0.85|41")
    exclusive_arrivals = replay._arrival_times(
        count=12,
        process=process,
        load_factor=0.85,
        lower_capacity=2 * 3.0,
        canonical_units=10.0,
        rng=rng,
    )
    assert [job.arrival_s for job in trace.jobs] == exclusive_arrivals
    assert replay._dedicated_load_reference(scenario)["workload_time_shares"] == {
        WORKLOADS[0]: 1.0
    }


def test_changing_physical_unit_scale_preserves_arrival_times():
    scenario = _scenario()
    scale = {WORKLOADS[0]: 1.0, WORKLOADS[1]: 1000.0, WORKLOADS[2]: 0.01}
    scaled = replace(
        scenario,
        actions=tuple(
            replace(
                action,
                lower=replace(
                    action.lower,
                    lower_service=action.lower.lower_service * scale[action.lower.workload_key],
                    lower_service_vector=tuple(
                        (workload, rate * scale[workload])
                        for workload, rate in action.lower.lower_service_vector
                    ),
                ),
                evaluation=replace(
                    action.evaluation,
                    phase_service_rate=action.evaluation.phase_service_rate * scale[action.lower.workload_key],
                    completion_group_units=action.evaluation.completion_group_units * scale[action.lower.workload_key],
                    replay_job_units=action.evaluation.replay_job_units * scale[action.lower.workload_key],
                    work_unit=f"rescaled_{action.lower.workload_key}_step",
                ),
            )
            for action in scenario.actions
        ),
    )
    original_trace = replay._build_trace(scenario, _protocol(), jobs_per_workload=12)
    scaled_trace = replay._build_trace(scaled, _protocol(), jobs_per_workload=12)
    original_jobs = {job.job_id: job for job in original_trace.jobs}
    for job in scaled_trace.jobs:
        original = original_jobs[job.job_id]
        assert job.arrival_s == pytest.approx(original.arrival_s)
        assert job.total_units == pytest.approx(original.total_units * scale[job.workload_key])
