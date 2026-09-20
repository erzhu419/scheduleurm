"""The frozen confirmation roster on an authored short arrival tape."""
import json

import pytest

import algorithm.experiments.aligned_hardware_mixed_successor_confirmation as confirmation
from algorithm.experiments.aligned_hardware_frame_model import capacity_certificate, load_model
from algorithm.experiments.aligned_hardware_lookahead import terminal_service_rates
from algorithm.experiments.aligned_hardware_waiting_cost import COMPONENTS


def test_one_task_runs_eight_registered_policies_on_one_tape_with_complete_accounting(monkeypatch):
    model = load_model()
    certificate = capacity_certificate(model, .5)
    mu = terminal_service_rates(model, certificate)
    tape = [(0., coordinate, index) for coordinate, count in enumerate((1, 4, 8))
            for index in range(count)] + [(.5, 2, 8), (1., 0, 1)]
    generated, recorded = [], []
    native_simulate = confirmation.simulate_frames

    def arrivals(*args):
        generated.append(args)
        return tape

    def simulate(model_arg, arrivals_arg, horizon, certificate_arg, budget, *, selector, **kwargs):
        assert arrivals_arg is tape and horizon == 2. and budget == 26
        result = native_simulate(model_arg, arrivals_arg, horizon, certificate_arg, budget,
                                 selector=selector, **kwargs)
        recorded.append(selector)
        return result

    monkeypatch.setattr(confirmation, "poisson_arrivals", arrivals)
    monkeypatch.setattr(confirmation, "simulate_frames", simulate)
    protocol = {"budget": 26, "arrival_horizon_s": 2., "terminal_service_rates": mu,
                "planning_seeds": [101, 102, 103]}
    rows = confirmation._run_seed((model, certificate, .5, 4041, protocol))
    assert generated == [(certificate["arrival_rates"], 2., 4041)]
    expected = {("v4_support", None), ("lookahead_cap_1pct", None)} | {
        (policy, seed) for policy in ("poisson_rollout_cap_1pct", "mixed_successor_rollout_cap_1pct")
        for seed in (101, 102, 103)}
    assert len(rows) == len(recorded) == len(expected) == 8
    assert {(row["policy"], row["planning_seed"]) for row in rows} == expected
    recorders = {(row["policy"], row["planning_seed"]): recorder
                 for row, recorder in zip(rows, recorded, strict=True)}
    indexed = {(row["policy"], row["planning_seed"]): row for row in rows}
    for row in rows:
        assert (row["load"], row["seed"], row["budget"]) == (.5, 4041, 26)
        metrics, parts = row["metrics"], row["decomposition"]
        assert metrics["job_count"] == metrics["completed_jobs"] == len(tape)
        assert metrics["mean_flow_s"] == pytest.approx(parts["mean_flow_s"])
        assert metrics["mean_flow_s"] == pytest.approx(sum(parts[key] for key in COMPONENTS))
        assert metrics["maximum_candidate_attempts_per_frame"] <= 26
        if row["planning_seed"] is not None:
            counts = row["selector_accounting"]
            assert counts["rollout_frames"] == 1
            assert counts["root_attempts"] + counts["rollout_attempts"] == metrics["candidate_attempts"]
            assert counts["rollout_attempts"] == 4 * counts["rollout_frames"]
            assert counts["rollout_random_draws"] == counts["rollout_arrival_event_generations"] + 3 * counts["rollout_frames"]
            assert counts["minimum_seed_support_margin"] >= 0
            assert counts["minimum_support_floor_margin"] >= 0
            assert counts["maximum_relative_support_loss"] <= .01 + 1e-12
            assert counts["maximum_budget_identity_error"] == counts["maximum_stage_identity_error"] == 0
    for planning_seed in (101, 102, 103):
        raw_key, mixed_key = ("poisson_rollout_cap_1pct", planning_seed), ("mixed_successor_rollout_cap_1pct", planning_seed)
        assert [rng.getstate() for rng in recorders[raw_key].arrival_rngs] == [
            rng.getstate() for rng in recorders[mixed_key].arrival_rngs]
        raw, mixed = indexed[raw_key]["selector_accounting"], indexed[mixed_key]["selector_accounting"]
        assert raw["rollout_random_draws"] == mixed["rollout_random_draws"]
        assert mixed["rollout_mixed_priority_coordinate_evaluations"] == mixed["rollout_class_evaluations"]
        assert mixed["rollout_mixed_profile_probes"] == mixed["rollout_profile_probes"]
        assert mixed["rollout_temporary_queue_updates"] <= mixed["rollout_queue_coordinate_evaluations"]
        assert mixed["rollout_temporary_queue_updates"] <= mixed["rollout_lane_fill_evaluations"] <= 8 * mixed["rollout_frames"]


def test_confirmation_constants_are_unique_and_disjoint_from_consumed_seed_ranges():
    assert tuple(confirmation.REAL_SEEDS) == tuple(range(4041, 4061))
    assert tuple(confirmation.PLANNING_SEEDS) == (101, 102, 103)
    consumed_real = {41, 42, 43} | set(range(1041, 1051)) | set(range(2041, 2061)) | set(range(3041, 3061))
    assert not set(confirmation.REAL_SEEDS).intersection(consumed_real)
    assert not set(confirmation.PLANNING_SEEDS).intersection((0, 1, 2))
    assert confirmation.BUDGET == 26 and confirmation.HORIZON == 20000.


def _forbid_fresh_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("registration validation must finish without fresh arrivals or workers")

    monkeypatch.setattr(confirmation, "poisson_arrivals", forbidden)
    monkeypatch.setattr(confirmation, "ProcessPoolExecutor", forbidden)


def test_registration_writes_only_protocol_and_snapshot_without_drawing_arrivals(tmp_path, monkeypatch):
    _forbid_fresh_execution(monkeypatch)
    output = tmp_path / "confirmation.json"
    protocol = confirmation.register(output)
    protocol_path = tmp_path / "confirmation_protocol.json"
    snapshot_path = tmp_path / "confirmation_fixed_sources.json"
    assert set(tmp_path.iterdir()) == {protocol_path, snapshot_path}
    assert json.loads(protocol_path.read_text()) == json.loads(json.dumps(protocol))
    snapshot = json.loads(snapshot_path.read_text())
    for name in confirmation.NEW_SOURCE_FILES:
        assert snapshot[name] == (confirmation.ROOT / name).read_text()
    assert protocol["seeds"] == confirmation.REAL_SEEDS
    assert protocol["planning_seeds"] == confirmation.PLANNING_SEEDS
    assert not output.exists() and not (tmp_path / "confirmation_runs.jsonl").exists()


@pytest.mark.parametrize("changed", ("model", "sources"))
def test_registered_drift_is_rejected_before_arrivals_workers_or_run_file(tmp_path, monkeypatch, changed):
    _forbid_fresh_execution(monkeypatch)
    output = tmp_path / "confirmation.json"
    confirmation.register(output)
    if changed == "model":
        path = tmp_path / "confirmation_protocol.json"
        saved = json.loads(path.read_text())
        saved["model"]["node"] = "different-registered-model"
    else:
        path = tmp_path / "confirmation_fixed_sources.json"
        saved = json.loads(path.read_text())
        saved[confirmation.NEW_SOURCE_FILES[0]] += "\n# different registered source\n"
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match=f"registered {changed} changed; fresh confirmation seeds remain unused"):
        confirmation.run_registered(output)
    assert not output.exists() and not (tmp_path / "confirmation_runs.jsonl").exists()


def test_single_stage_entrypoint_refuses_to_register_or_run(tmp_path, monkeypatch):
    _forbid_fresh_execution(monkeypatch)
    with pytest.raises(ValueError, match="separate confirmation stages"):
        confirmation.build_confirmation(tmp_path / "confirmation.json")
    assert not list(tmp_path.iterdir())
