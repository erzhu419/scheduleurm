"""Exact reuse changes execution work without changing frozen science rows."""
from copy import deepcopy
import json

import pytest

from algorithm.background_competition_continuous_lane_rollout import BackgroundCompetitionContinuousLaneRollout
from algorithm.continuous_arrival_age_tail_lane_rollout import ContinuousArrivalAgeTailLaneRollout
from algorithm.experiments import aligned_hardware_exact_reuse_sharded as fast
from algorithm.experiments import aligned_hardware_two_step_current_policy_bg_incumbent_development as frozen
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory
from algorithm.two_step_bg_incumbent_original_risk_lane_rollout import TwoStepBGIncumbentOriginalRiskLaneRollout


@pytest.fixture(scope="module")
def native_pair():
    options = (
        FrameOption("a1", (1, 0, 0), 1., ((1.,), (), ())),
        FrameOption("b1", (0, 1, 0), 1.25, ((), (1.25,), ())),
        FrameOption("c1", (0, 0, 1), 1.5, ((), (), (1.5,))),
        FrameOption("loaded-ab", (1, 1, 0), 1.75, ((.5,), (1.75,), ()), loaded=True),
        FrameOption("a2", (2, 0, 0), 1.5, ((1.5, 1.5), (), ())),
    )
    model = FrameModel("toy", "exact-run-tape", 2, ("a", "b", "c"), (1., 1., 1.), options)
    tape = {"tape_id": "toy:R2:load0.8:seed41", "node": "toy", "lanes": 2,
            "load": .8, "seed": 41,
            "arrivals": [(0., 0, 0), (0., 0, 1), (0., 1, 0), (0., 2, 0),
                         (3., 0, 2), (3., 1, 1), (3., 2, 1), (6., 0, 3), (8., 2, 2)]}
    protocol = {
        "planning_samples": 2, "planning_event_cap": 8,
        "environment_arrival_end": 12., "observation_end": 16., "arrival_horizon": 12.,
        "birth_cohort": [0., 12.], "models": [{"node": "toy", "lanes": 2}],
        "tapes": [tape], "loads": [.8], "arrival_seeds": [41],
        "performance": {"min_gain": .01, "group_secondary_tolerance": .05},
        "experiment": "native-toy", "registered_at_utc": "scientific-test-registration",
        "limitations": "Small deterministic native trajectory used only for equivalence.",
        "cutoff_semantics": "Same literal physical service clock.",
        "continuation_parent_result": "toy-parent", "previous_result": "toy-BG",
    }
    seed = "arrival-plan-v1:toy:0.8:online"

    def simulate(planner, label, maxweight=False):
        return simulate_observed_rate_contextual_trajectory(
            model, tape["arrivals"], 12., 16.,
            planner.maxweight if maxweight else planner, policy=label)

    mw = simulate(ContinuousArrivalAgeTailLaneRollout(options, seed, 2, 8), "maxweight", True)
    bg = simulate(BackgroundCompetitionContinuousLaneRollout(options, seed, 2, 8), "arrival_rollout")
    parent = simulate(TwoStepBGIncumbentOriginalRiskLaneRollout(options, seed, 2, 8), "arrival_rollout")
    checks = {"baseline_reproduction_passed": True, "rollout_checks_passed": True}
    # The registered references are loaded from gzip JSON, including list normalization.
    references = (json.loads(json.dumps({"maxweight": mw, "arrival_rollout": bg})), checks,
                  json.loads(json.dumps({"maxweight": mw, "arrival_rollout": parent})), checks)
    old_row, old_policies = frozen.run_tape(model, tape, protocol, *references)
    captured = {}
    real_check = fast.check_rollout

    def capture_check(result, check_model, check_protocol, planner):
        captured["before_check"] = planner.reuse_work
        passed = real_check(result, check_model, check_protocol, planner)
        captured["after_check"] = planner.reuse_work
        return passed

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(fast, "check_rollout", capture_check)
        new_row, new_policies = fast.run_tape(model, tape, protocol, *references)
    return protocol, old_row, old_policies, new_row, new_policies, captured


def test_native_run_tape_keeps_all_scientific_fields_and_clock_and_excludes_checker_work(native_pair):
    _, old_row, old_policies, new_row, new_policies, captured = native_pair
    assert new_policies == old_policies
    assert {key: value for key, value in new_row.items() if key != "executed_reuse_work"} == old_row
    assert new_row["baseline_reproduction_passed"] and new_row["rollout_checks_passed"]
    assert any(3 in frame["option_indices"] for frame in new_policies["arrival_rollout"]["frames"])
    assert len(new_row["class_snapshots"]) == 3
    assert new_row["executed_reuse_work"] == captured["before_check"]
    before = captured["before_check"]["mw_reuse_work"]
    after = captured["after_check"]["mw_reuse_work"]
    assert after["queries"] > before["queries"]
    assert before["cache_hits"] > 0 and before["actual_option_probes"] < before["logical_option_probes"]
    assert new_row["executed_reuse_work"]["inner_decision_reuse_work"]["current_cache_hits"] > 0


def test_merge_retains_native_gates_and_logical_work_while_summing_actual_counts_and_maximizing_peaks(
        native_pair, tmp_path, monkeypatch):
    original_protocol, old_row, _, new_row, _, _ = native_pair
    scientific = deepcopy(original_protocol)
    scientific["arrival_seeds"] = [41, 42, 43]
    scientific["tapes"] = []
    old_rows, new_rows = [], []
    # These copies are merge inputs, not additional executed trajectories.
    for index, seed in enumerate(scientific["arrival_seeds"]):
        tape = {**deepcopy(original_protocol["tapes"][0]), "seed": seed,
                "tape_id": f"toy:R2:load0.8:seed{seed}"}
        scientific["tapes"].append(tape)
        old = {**deepcopy(old_row), "seed": seed, "tape_id": tape["tape_id"]}
        new = {**deepcopy(new_row), "seed": seed, "tape_id": tape["tape_id"]}
        for fields in new["executed_reuse_work"].values():
            for name, value in fields.items():
                fields[name] = value * ((3, 1, 2)[index] if "_peak_" in name else index + 1)
        old_rows.append(old)
        new_rows.append(new)
    expected = fast.previous.summarize_rows(old_rows, scientific)
    registration = {"scientific_protocol": scientific,
                    "distribution_registered_at_utc": "distribution-test-registration"}
    monkeypatch.setattr(fast, "load_registration", lambda artifact_dir: registration)
    directory = tmp_path / (fast.STEM + "_shards")
    directory.mkdir()
    for index in (2, 0, 1):
        # Filenames intentionally disagree with literal tape order.
        document = {
            "shard_index": index, "tape_id": scientific["tapes"][index]["tape_id"],
            "scientific_registered_at_utc": scientific["registered_at_utc"],
            "distribution_registered_at_utc": registration["distribution_registered_at_utc"],
            "execution_host": "node001", "trajectory_file": f"/remote/{index}.json.gz",
            "row": new_rows[index],
        }
        (directory / f"{2 - index:02d}.json").write_text(json.dumps(document))
    actual = json.loads(fast.merge(tmp_path).read_text())
    assert actual["scientific_gate_decision"] == expected["decision"]
    assert actual["decision"] == expected["decision"].replace(
        "TWO_STEP_CURRENT_POLICY_BG_INCUMBENT", "EXACT_REUSE_TWO_STEP_CURRENT_POLICY_BG_INCUMBENT", 1)
    for key, value in expected.items():
        if key == "decision":
            continue
        actual_value = deepcopy(actual[key])
        if key == "paired_runs":
            for row in actual_value:
                row.pop("executed_reuse_work")
        elif key in ("reference_comparison", "parent_comparison"):
            actual_value["work_scope"] = value["work_scope"]
            for row in actual_value["paired_runs"]:
                row.pop("executed_reuse_work")
        assert actual_value == json.loads(json.dumps(value)), key
    assert [row["tape_id"] for row in actual["paired_runs"]] == [tape["tape_id"] for tape in scientific["tapes"]]
    for group, fields in new_row["executed_reuse_work"].items():
        for name, value in fields.items():
            assert actual["executed_reuse_work"][group][name] == value * (3 if "_peak_" in name else 6)
    assert actual["executed_reuse_work"]["inner_decision_reuse_work"]["current_cache_peak_entries"] > 0
    assert len(actual["per_tape_executed_reuse_work"]) == 3
