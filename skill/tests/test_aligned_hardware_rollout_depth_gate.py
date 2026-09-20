"""Develop the depth gate without accessing new arrival seeds or confirmation data."""
import json
from pathlib import Path

import pytest

import algorithm.experiments.aligned_hardware_rollout_depth_gate as experiment
from algorithm.experiments.aligned_hardware_control_variate import _assert_control_reproduced
from algorithm.experiments.aligned_hardware_frame_model import capacity_certificate, load_model
from algorithm.experiments.aligned_hardware_lookahead import terminal_service_rates
from algorithm.experiments.aligned_hardware_mixed_successor_rollout import _run_seed as frozen_run_seed


@pytest.fixture(scope="module")
def example():
    model = load_model()
    certificate = capacity_certificate(model, .5)
    mu = terminal_service_rates(model, certificate)
    return model, certificate, mu


def test_authored_tape_retains_eleven_run_roster_controls_and_gate_work(example, monkeypatch):
    model, certificate, mu = example
    tape = [(0., coordinate, job) for coordinate, count in enumerate((1, 9, 8)) for job in range(count)]
    calls = []

    def arrivals(*args):
        calls.append(args)
        return tape

    monkeypatch.setattr(experiment, "poisson_arrivals", arrivals)
    monkeypatch.setattr("algorithm.experiments.aligned_hardware_mixed_successor_rollout.poisson_arrivals", lambda *args: tape)
    protocol = {"arrival_horizon_s": 2., "budget": 26,
                "planning_seeds": (0, 1, 2), "terminal_service_rates": mu}
    rows = experiment._run_seed((model, certificate, .5, 41, protocol))
    expected = frozen_run_seed((model, certificate, .5, 41,
                               {"arrival_horizon_s": 2., "budgets": [26]}, mu))
    assert len(calls) == 1 and len(rows) == 11
    assert {(row["policy"], row["planning_seed"]) for row in rows} == {
        *( (policy, None) for policy in experiment.CONTROLS[:2]),
        *( (policy, seed) for policy in (experiment.CONTROLS[-1], experiment.MIXED_POLICY, experiment.NEW_POLICY)
           for seed in (0, 1, 2)),
    }
    reference = {(row["policy"], row["planning_seed"]): json.loads(json.dumps(row)) for row in expected}
    for row in rows:
        assert row["metrics"]["completed_jobs"] == row["metrics"]["job_count"] == len(tape)
        assert row["decomposition"]["mean_flow_s"] == pytest.approx(row["metrics"]["mean_flow_s"])
        if row["policy"] in experiment.CONTROL_POLICIES:
            _assert_control_reproduced(row, reference[(row["policy"], row["planning_seed"])])
        else:
            counts = row["selector_accounting"]
            assert counts["mixed_branch_frames"] > 0 and counts["v4_branch_frames"] > 0
            assert counts["mixed_branch_frames"] + counts["v4_branch_frames"] == row["metrics"]["frame_count"]
            assert counts["gate_queue_coordinate_evaluations"] == 3 * row["metrics"]["frame_count"]
            assert counts["gate_setup_requirement_reads"] == 54
            assert counts["gate_threshold_comparisons"] == row["metrics"]["frame_count"]
            assert counts["root_attempts"] + counts["rollout_attempts"] == row["metrics"]["candidate_attempts"]
            assert counts["initial_root_attempts"] + counts["recycled_candidate_attempts"] == counts["root_attempts"]
            assert counts["initial_cost_candidate_evaluations"] + counts["recycled_cost_candidate_evaluations"] == counts["cost_candidate_evaluations"]
            assert counts["rollout_attempts"] == 4 * counts["rollout_frames"]


def test_inactive_boundary_charges_v4_attempts_without_mixed_work(example):
    model, certificate, mu = example
    recorder = experiment.RolloutDepthGateRecorder(model.options, model.lanes, certificate["arrival_rates"], mu, 0)
    before = [rng.getstate() for rng in recorder.arrival_rngs]
    selected = recorder(model.options, (1, 8, 8), model.lanes, 26)  # Weighted load exactly 96.
    counts = recorder.accounting()
    assert [rng.getstate() for rng in recorder.arrival_rngs] == before
    assert counts["mixed_branch_frames"] == 0 and counts["v4_branch_frames"] == 1
    assert counts["root_attempts"] == counts["initial_root_attempts"] == selected.attempts
    assert counts["rollout_attempts"] == counts["rollout_random_draws"] == counts["cost_candidate_evaluations"] == 0
    assert counts["minimum_seed_support_margin"] is counts["minimum_support_floor_margin"] is None


def test_registration_freezes_only_development_inputs_and_never_generates_arrivals(tmp_path, monkeypatch):
    output = tmp_path / "development.json"
    read_text = Path.read_text

    def allowed_read(path, *args, **kwargs):
        assert "mixed_successor_confirmation" not in str(path)
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", allowed_read)
    monkeypatch.setattr(experiment, "poisson_arrivals", lambda *args: pytest.fail("registration must not generate arrivals"))
    monkeypatch.setattr(experiment, "ProcessPoolExecutor", lambda *args, **kwargs: pytest.fail("registration must not start workers"))
    protocol = experiment.register(output)
    assert protocol["seeds"] == (41, 42, 43) and protocol["planning_seeds"] == (0, 1, 2)
    assert protocol["budget"] == 26
    assert protocol["gate_weights"] == (8, 8, 3)
    assert protocol["gate_capacity"] == 48 and protocol["strict_threshold"] == 96
    assert protocol["advance_criteria"] == experiment.ADVANCE_CRITERIA
    assert protocol["retained_continuation_summary"]["new_replays"] == 0
    assert {path.name for path in tmp_path.iterdir()} == {"development_protocol.json", "development_fixed_sources.json"}
    saved = tmp_path / "development_protocol.json"
    changed = json.loads(saved.read_text())
    changed["seeds"] = [4041, 4042, 4043]
    saved.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="roster, gate or advancement criteria changed"):
        experiment.run_registered(output)
    assert not (tmp_path / "development_runs.jsonl").exists()


def test_seed_layer_summary_and_development_gate_do_not_pool_planning_seeds():
    rows = []
    for load in (.5, .8, .95):
        for seed in (41, 42, 43):
            for policy in experiment.CONTROL_POLICIES:
                for p in ((0, 1, 2) if policy in (experiment.CONTROLS[-1], experiment.MIXED_POLICY) else (None,)):
                    rows.append({"load": load, "seed": seed, "policy": policy, "planning_seed": p,
                                 "metrics": {"job_count": 10, "mean_flow_s": 100., "p90_flow_s": 150.}})
            for p, flow in enumerate((80., 90., 100.)):
                rows.append({"load": load, "seed": seed, "policy": experiment.NEW_POLICY, "planning_seed": p,
                             "metrics": {"job_count": 10, "mean_flow_s": flow, "p90_flow_s": 140.}})
    paired, summary = experiment.summarize_runs(rows, (.5, .8, .95))
    assert len(paired) == 27
    assert all(row["mean_paired_flow_ratio_over_v4_support"] == pytest.approx(.9) for row in summary)
    assert all(experiment.advance_decision(summary, {"execution": True}).values())
    summary[1]["by_arrival_seed"][0]["flow_ratio_over_v4_support"] = 1.
    assert not experiment.advance_decision(summary, {"execution": True})["every_medium_high_real_seed_improves_flow_over_v4"]


def test_retained_continuation_filter_is_strict_and_has_no_new_replay(example):
    model, _, _ = example
    gate = experiment.build_backlog_gate(model.options, model.lanes)
    cases = [{"queue": queue, "state_id": str(i),
              "effect": {"outcome": outcome, "realized_total_delta_job_s": delta}}
             for i, (queue, outcome, delta) in enumerate((((1, 8, 8), "improved", -20.),
                                                         ((1, 9, 8), "worsened", 3.)))]
    source = {"protocol": {"loads": [.5]},
              "runs": [{"load": .5, "source_policy": experiment.CONTROLS[-1], "cases": cases}]}
    result = experiment.continuation_summary(source, gate)
    assert result["summary"]["all_disagreement_occurrences"] == 2
    assert result["summary"]["retained_occurrences"] == result["summary"]["worsened"] == 1
    assert result["gate_queue_coordinate_evaluations"] == 6
    assert result["new_selector_calls"] == result["new_replays"] == result["new_random_draws"] == 0
