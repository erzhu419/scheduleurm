from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from algorithm.experiments import aligned_hardware_joint95_baseline_confirmation as baseline
from algorithm.experiments.aligned_hardware_frame_model import FrameOption, load_model


def _option(requirements, duration, name):
    return FrameOption(
        action_id=name,
        requirements=tuple(requirements),
        duration=float(duration),
        completion_offsets=tuple(
            (float(duration),) * count for count in requirements
        ),
    )


def _exhaustive(options, queues, lanes):
    best = None
    option_count = len(options)
    for raw in itertools.product((None, *range(option_count)), repeat=lanes):
        indices = tuple(sorted(raw, key=lambda value: option_count if value is None else value))
        service = tuple(
            sum(
                options[index].requirements[coordinate]
                for index in indices if index is not None
            )
            for coordinate in range(len(queues))
        )
        if any(done > waiting for done, waiting in zip(service, queues, strict=True)):
            continue
        duration = max((1.0, *(options[index].duration for index in indices if index is not None)))
        score = sum(q * d for q, d in zip(queues, service, strict=True)) / duration
        key = (duration, tuple(option_count if value is None else value for value in indices))
        if best is None or score > best[0] or (score == best[0] and key < best[1]):
            best = (score, key, indices)
    return best


def _write(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _copied_registration(path: Path, source: Path) -> Path:
    _write(path, json.loads(source.read_text(encoding="utf-8")))
    return path


def test_independent_exact_mw_matches_exhaustive_score_and_canonical_action():
    options = (
        _option((1, 0), 1.0, "a1"),
        _option((0, 1), 2.0, "b1"),
        _option((1, 1), 1.5, "ab"),
    )
    queues = (3, 2)
    lanes = 3
    expected = _exhaustive(options, queues, lanes)
    selected = baseline.select_exact_renewal_mw(options, queues, lanes)
    assert selected.exact
    assert selected.score == pytest.approx(expected[0])
    assert selected.option_indices == expected[2]
    assert selected.solver_calls == len({1.0, 1.5, 2.0}) + 1


def test_independent_exact_mw_uses_registered_neutral_tie_break():
    options = (
        _option((1,), 1.0, "first"),
        _option((1,), 1.0, "second"),
    )
    selected = baseline.select_exact_renewal_mw(options, (1,), 2)
    assert selected.option_indices == (0, None)
    assert selected.score == pytest.approx(1.0)


def test_independent_exact_mw_tie_solve_survives_dependent_objective_row():
    # The primary objective equality is exactly 4*service_0 + 5*service_1.
    # HiGHS presolve used to misclassify this reachable integer tie solve as
    # infeasible even though (option 2, option 2, option 3, idle) is feasible.
    options = (
        _option((1, 0), 2.0, "a1"),
        _option((0, 1), 1.5, "b1"),
        _option((1, 2), 1.25, "ab2"),
        _option((2, 1), 1.5, "a2b"),
    )
    expected = _exhaustive(options, (4, 5), 4)
    selected = baseline.select_exact_renewal_mw(options, (4, 5), 4)
    assert selected.score == pytest.approx(expected[0])
    assert selected.option_indices == expected[2]


def test_exact_mw_call_path_is_independent_of_candidate_selector(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("exact oracle called the candidate B&B selector")

    monkeypatch.setattr(baseline, "select_canonical_count_space_bnb_frame", forbidden)
    options = (
        _option((1, 0), 1.0, "a1"),
        _option((0, 1), 2.0, "b1"),
        _option((1, 1), 1.5, "ab"),
    )
    selected = baseline.select_exact_renewal_mw(options, (3, 2), 3)
    assert selected.exact
    assert selected.score > 0.0


def test_plain_heuristics_are_feasible_and_keep_their_named_semantics():
    model = load_model()
    fixed = baseline.select_fixed_balanced_partition(model.options, (10, 10, 20), 8)
    simple = baseline.select_simple_queue_delay(model.options, (10, 10, 20), 8)
    assert fixed.policy == "fixed_balanced_partition"
    assert simple.policy == "simple_queue_delay"
    assert all(done <= waiting for done, waiting in zip(fixed.service, (10, 10, 20), strict=True))
    assert all(done <= waiting for done, waiting in zip(simple.service, (10, 10, 20), strict=True))
    assert not any(
        model.options[index].loaded
        for selection in (fixed, simple)
        for index in selection.option_indices
        if index is not None
    )


def test_registered_task_grid_has_exactly_120_common_tapes_and_480_runs():
    model = load_model()
    rates = {
        (lanes, load): (0.1, 0.2, 0.3)
        for lanes in baseline.LANES
        for load in baseline.LOADS
    }
    tasks = baseline._build_tasks(model, rates)
    identities = {(lanes, load, seed) for _, lanes, load, seed, _ in tasks}
    assert len(tasks) == len(identities) == 120
    assert len(tasks) * len(baseline.POLICIES) == 480


def test_case_seed_builds_one_tape_and_returns_all_four_runs(monkeypatch):
    model = load_model()
    tape = ((0.25, 0, 0),)
    observed_tapes = []

    monkeypatch.setattr(baseline, "poisson_arrivals", lambda rates, horizon, seed: tape)

    def fake_simulate(model, arrivals, horizon, policy):
        observed_tapes.append(arrivals)
        return {"policy": policy}

    monkeypatch.setattr(baseline, "_simulate_policy", fake_simulate)
    row = baseline._run_case_seed((model, 8, 0.5, 12041, (0.1, 0.2, 0.3)))
    assert row["arrival_tape_generation_count"] == 1
    assert row["arrival_count"] == 1
    assert set(row["policies"]) == set(baseline.POLICIES)
    assert observed_tapes == [tape] * len(baseline.POLICIES)


def test_registration_is_outcome_blind_and_freezes_the_18_claim_family(tmp_path, monkeypatch):
    output = tmp_path / "baseline.json"
    capacity_protocol = _copied_registration(
        tmp_path / "capacity-protocol.json", baseline.DEFAULT_CAPACITY_REGISTRATION
    )
    static_protocol = _copied_registration(
        tmp_path / "static-protocol.json", baseline.DEFAULT_STATIC_REGISTRATION
    )
    canonical_protocol = _copied_registration(
        tmp_path / "canonical-protocol.json", baseline.DEFAULT_CANONICAL_REGISTRATION
    )
    result_paths = {
        (tmp_path / "capacity-result.json").resolve(),
        (tmp_path / "static-result.json").resolve(),
        (tmp_path / "canonical-result.json").resolve(),
    }
    original = baseline._read_object
    reads = []

    def guarded(path):
        resolved = Path(path).resolve()
        reads.append(resolved)
        if resolved in result_paths:
            raise AssertionError("registration read a terminal result")
        return original(path)

    monkeypatch.setattr(baseline, "_read_object", guarded)
    protocol = baseline.register(
        output,
        capacity_registration=capacity_protocol,
        capacity_result=tmp_path / "capacity-result.json",
        static_registration=static_protocol,
        static_result=tmp_path / "static-result.json",
        canonical_registration=canonical_protocol,
        canonical_result=tmp_path / "canonical-result.json",
    )
    result, registered, fixed = baseline._paths(output)
    assert not result.exists()
    assert registered.is_file() and fixed.is_file()
    assert not (set(reads) & result_paths)
    assert protocol["status"] == "REGISTERED_AWAITING_CANONICAL_PHYSICAL_SELECTOR"
    assert protocol["scenario"]["arrival_seeds"] == list(range(12041, 12061))
    assert protocol["scenario"]["loads"] == [0.5, 0.8, 0.95]
    assert protocol["expected_work"]["case_seed_tapes"] == 120
    assert protocol["expected_work"]["trajectory_runs"] == 480
    assert protocol["statistics"]["primary_family_size"] == 18
    assert len(protocol["statistics"]["primary_claims"]) == 18
    assert protocol["registration_scope"]["canonical_selector_protocol_read"] is True
    assert protocol["registration_scope"]["canonical_selector_result_read"] is False


def test_runner_lists_all_missing_prerequisites_before_tape_or_selector_work(tmp_path, monkeypatch):
    output = tmp_path / "baseline.json"
    capacity_protocol = _copied_registration(
        tmp_path / "capacity-protocol.json", baseline.DEFAULT_CAPACITY_REGISTRATION
    )
    static_protocol = _copied_registration(
        tmp_path / "static-protocol.json", baseline.DEFAULT_STATIC_REGISTRATION
    )
    canonical_protocol = _copied_registration(
        tmp_path / "canonical-protocol.json", baseline.DEFAULT_CANONICAL_REGISTRATION
    )
    baseline.register(
        output,
        capacity_registration=capacity_protocol,
        capacity_result=tmp_path / "capacity-result.json",
        static_registration=static_protocol,
        static_result=tmp_path / "static-result.json",
        canonical_registration=canonical_protocol,
        canonical_result=tmp_path / "canonical-result.json",
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("trajectory work began before prerequisites passed")

    monkeypatch.setattr(baseline, "poisson_arrivals", forbidden)
    monkeypatch.setattr(baseline, "select_canonical_count_space_bnb_frame", forbidden)
    monkeypatch.setattr(baseline, "select_exact_renewal_mw", forbidden)
    monkeypatch.setattr(baseline, "_campaign_fail_stop", lambda protocol: None)
    with pytest.raises(baseline.BaselineConfirmationRefusal) as caught:
        baseline.run_registered(output)
    message = str(caught.value)
    assert "capacity_result" in message
    assert "score_only_static_result" in message
    assert "canonical_selector_result" in message
    assert not baseline._paths(output)[0].exists()


def test_terminal_campaign_fail_stop_blocks_the_baseline_gate(tmp_path):
    results = tmp_path / "joint-results"
    results.mkdir()
    payload = {
        "schema_version": 1,
        "gate": "critical_gpu_joint95_campaign_fail_stop",
        "status": "FAIL_STOP",
        "terminal": True,
        "pass": False,
        "decision": "JOINT95_V1_EXTERNAL_OVERLAP_FAIL_STOP",
        "campaign": {"failed_wave": 23},
        "adoption": {"baseline_confirmation_authorized": False},
    }
    _write(results / "campaign_fail_stop.json", payload)
    observed = baseline._campaign_fail_stop(
        {"source_paths": {"joint_results_directory": str(results)}}
    )
    assert observed == payload


def test_paired_rule_uses_all_twenty_seeds_and_frozen_familywise_limit():
    rows = []
    for seed in baseline.ARRIVAL_SEEDS:
        rows.append(
            {
                "lanes": 8,
                "load": 0.5,
                "seed": seed,
                "policies": {
                    "candidate_canonical_k": {
                        "mean_flow_s": 9.9,
                        "makespan_s": 99.0,
                        "p90_flow_s": 19.8,
                    },
                    "exact_renewal_mw_milp": {
                        "mean_flow_s": 10.0,
                        "makespan_s": 100.0,
                        "p90_flow_s": 20.0,
                    },
                    "fixed_balanced_partition": {
                        "mean_flow_s": 11.0,
                        "makespan_s": 110.0,
                        "p90_flow_s": 22.0,
                    },
                    "simple_queue_delay": {
                        "mean_flow_s": 12.0,
                        "makespan_s": 120.0,
                        "p90_flow_s": 24.0,
                    },
                },
            }
        )
    exact = baseline._paired_ratio_summary(rows, 8, 0.5, "exact_renewal_mw_milp")
    fixed = baseline._paired_ratio_summary(rows, 8, 0.5, "fixed_balanced_partition")
    assert exact["sample_size"] == 20
    assert exact["one_sided_simultaneous_upper_ratio"] == pytest.approx(0.99)
    assert exact["upper_ratio_limit"] == pytest.approx(1.01)
    assert exact["confirmed"]
    assert fixed["one_sided_simultaneous_upper_ratio"] == pytest.approx(0.9)
    assert fixed["confirmed"]
    assert baseline.FAMILYWISE_ALPHA / baseline.PRIMARY_FAMILY_SIZE == pytest.approx(0.05 / 18)
