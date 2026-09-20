from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import random

import pytest

from algorithm.experiments.unified_hardware_or_replay import (
    ABLATION_POLICIES,
    EvaluationAction,
    LowerAction,
    PolicyDescriptor,
    ReplayAction,
    ReplayJobState,
    TraceJob,
    TraceSpec,
    _job_order_key,
    _select_lower_action,
    _simulate_scenario,
)


CPU = "cpu_heavy_local_bench"
GPU = "gpu_heavy_jax_matmul"
CNN = "gpu_cnn_torch_resnet50"


def _policy(semantic: str) -> PolicyDescriptor:
    return PolicyDescriptor(semantic, "test", semantic, {})


def _lower(action_id, workload, rate, *, width=1, penalty=0.0, residents=()):
    return LowerAction(
        action_id=action_id,
        workload_key=workload,
        profile=width,
        node_bucket="fixture",
        resource_state="loaded" if residents else "idle",
        resident_mix="fixture" if residents else "none",
        lower_service=rate,
        lower_service_vector=((workload, rate),) + tuple(
            (key, service) for key, _, service in residents
        ),
        penalty_units=penalty,
        loaded_action=bool(residents),
        target_batch_width=width,
        required_job_counts=((workload, width),) + tuple(
            (key, count) for key, count, _ in residents
        ),
    )


def _select(actions, backlog, scales=None, semantic="robust_maxweight"):
    return _select_lower_action(
        policy=_policy(semantic),
        actions=actions,
        quadrant="q01",
        remaining_count=100,
        backlog_vector=backlog,
        workload_unit_scales=scales or {key: 1.0 for key in backlog},
        remaining_counts={key: 100 for key in backlog},
        oldest_ready_workload=next(iter(backlog)),
    )


def _oracle_key(action, backlog, scales):
    """Independent rational rate oracle: each resident is capped by count / time."""
    duration = (
        Fraction(action.batch_width())
        * Fraction(scales[action.workload_key])
        / Fraction(action.lower_service)
    )
    requirements = dict(action.required_job_counts)
    score = sum(
        Fraction(backlog[key])
        * min(
            Fraction(rate) / Fraction(scales[key]),
            Fraction(requirements[key]) / duration,
        )
        for key, rate in action.lower_service_vector
    ) - Fraction(action.penalty_units)
    return (-score, duration, action.profile, action.penalty_units, action.action_id)


def test_single_job_selector_prioritizes_backlog_weight_over_delay():
    slow = _lower("slow", GPU, 1.0)
    fast = _lower("fast", CNN, 2.0)

    selected = _select((slow, fast), {GPU: 100.0, CNN: 1.0})

    assert selected == slow  # Supports are 100 and 2; delay-first loses 98.


def test_positive_support_gap_is_not_treated_as_a_delay_tie():
    slow = _lower("slow", GPU, 1.0, width=2)
    fast = _lower("fast", CNN, 2.0, width=2, penalty=5e-10)

    assert _select((slow, fast), {GPU: 2.0, CNN: 1.0}) == slow


def test_delay_breaks_an_exact_support_tie():
    slow = _lower("slow", GPU, 1.0)
    fast = _lower("fast", CNN, 2.0)

    assert _select((slow, fast), {GPU: 2.0, CNN: 1.0}) == fast


def test_frame_score_caps_service_at_admitted_resident_work():
    joint = _lower("joint", GPU, 1.0, residents=((CNN, 1, 100.0),))
    dedicated = _lower("dedicated", GPU, 4.0)

    assert _select((joint, dedicated), {GPU: 1.0, CNN: 1.0}) == dedicated


def test_support_only_removes_delay_tie_break_without_changing_frame_objective():
    slow = _lower("slow", GPU, 1.0)
    fast = _lower("fast", CNN, 4.0, width=2)
    backlog = {GPU: 4.0, CNN: 1.0}
    assert _select((slow, fast), backlog) == fast
    assert _select((slow, fast), backlog, semantic="ablation_support_only") == slow

    joint = _lower("joint", GPU, 1.0, residents=((CNN, 1, 1.0),))
    dedicated = _lower("dedicated", GPU, 2.0)
    assert _select(
        (joint, dedicated), {GPU: 1.0, CNN: 10.0}, semantic="ablation_support_only"
    ) == joint


def test_delay_only_preserves_frame_support_as_its_secondary_objective():
    joint = _lower("joint", GPU, 1.0, residents=((CNN, 1, 1.0),))
    dedicated = _lower("dedicated", CNN, 1.0)

    assert _select(
        (joint, dedicated), {GPU: 1.0, CNN: 10.0}, semantic="ablation_delay_only"
    ) == joint


def test_robust_selector_matches_independent_exhaustive_rational_argmax():
    rng = random.Random(20260909)
    keys = (CPU, GPU, CNN)
    for case in range(120):
        # Dyadic inputs keep the oracle's exact ties exact in floating point.
        scales = {key: rng.choice((0.25, 1.0, 4.0)) for key in keys}
        backlog = {key: rng.randint(1, 400) / 4 for key in keys}
        actions = []
        for index in range(rng.randint(2, 8)):
            key = rng.choice(keys)
            residents = tuple(
                (other, rng.randint(1, 3), rng.choice((0.25, 1.0, 4.0, 16.0)))
                for other in keys
                if other != key and rng.choice((False, True))
            )
            action = _lower(
                f"{case}:{index}",
                key,
                rng.choice((0.25, 1.0, 4.0, 16.0)),
                width=rng.choice((1, 2, 4)),
                penalty=rng.randint(0, 32) / 16,
                residents=residents,
            )
            if not residents and rng.choice((False, True)):
                action = replace(action, migration_action=True)
            actions.append(action)
        expected = min(actions, key=lambda action: _oracle_key(action, backlog, scales))
        assert _select(actions, backlog, scales) == expected, case
        rng.shuffle(actions)
        assert _select(actions, backlog, scales) == expected, case
        nonmigration = [action for action in actions if not action.migration_action]
        if nonmigration:
            assert _select(nonmigration, backlog, scales) == _select(
                nonmigration, backlog, scales, semantic="ablation_no_migration"
            ), case


@pytest.mark.parametrize("semantic", [name for name, _ in ABLATION_POLICIES])
@pytest.mark.parametrize("workload", [CPU, GPU])
@pytest.mark.parametrize("width", [1, 2])
def test_ablations_preserve_the_full_policy_job_order(semantic, workload, width):
    states = [
        ReplayJobState(TraceJob(str(index), workload, arrival, units), units, arrival)
        for index, (arrival, units) in enumerate(((0.0, 3.0), (0.0, 1.0), (1.0, 2.0)))
    ]
    expected = sorted(
        states,
        key=lambda state: _job_order_key(state, _policy("robust_maxweight"), batch_width=width),
    )
    actual = sorted(
        states,
        key=lambda state: _job_order_key(state, _policy(semantic), batch_width=width),
    )
    assert actual == expected


@pytest.mark.parametrize("workload,quadrant", [(CPU, "q10"), (GPU, "q01")])
def test_no_migration_matches_full_decisions_and_replay_when_action_family_is_identical(
    workload, quadrant
):
    lower_actions = (
        _lower("single", workload, 1.0),
        _lower("batch", workload, 3.0, width=2),
    )
    backlog = {workload: 20.0}
    assert _select(lower_actions, backlog) == _select(
        lower_actions, backlog, semantic="ablation_no_migration"
    )
    actions = tuple(
        ReplayAction(
            lower,
            EvaluationAction(
                action_id=lower.action_id,
                phase_service_rate=lower.lower_service,
                completion_group_units=float(lower.batch_width()),
                replay_job_units=1.0,
                work_unit="fixture_units",
                fixed_overhead_s=0.1,
                completion_model_sample_count=3,
                eta_source="synthetic_fixture",
                completion_model_kind="natural_completion",
            ),
        )
        for lower in lower_actions
    )
    trace = TraceSpec(
        trace_id="no_migration_equivalence",
        arrival_family="poisson",
        arrival_process="poisson",
        load_factor=0.5,
        seed=0,
        jobs=tuple(
            TraceJob(str(index), workload, arrival, units)
            for index, (arrival, units) in enumerate(
                ((0.0, 1.0), (0.0, 3.0), (0.0, 2.0), (0.2, 4.0), (0.3, 1.5))
            )
        ),
        workload_unit_scales=((workload, 1.0),),
    )
    kwargs = dict(trace=trace, actions=actions, quadrant=quadrant, resource_count=2)
    assert _simulate_scenario(policy=_policy("robust_maxweight"), **kwargs) == (
        _simulate_scenario(policy=_policy("ablation_no_migration"), **kwargs)
    )
