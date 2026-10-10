"""Single-root replay preserves class protection, partial work, and free replanning."""
from dataclasses import replace
from fractions import Fraction
from itertools import chain, repeat
import json

import pytest

import algorithm.background_competition_continuous_lane_rollout as background_module
import algorithm.class_protected_bg_incumbent_background_redecision_lane_rollout as policy_module
from algorithm.arrival_aware_lane_rollout import PlanningArrivals
from algorithm.class_protected_bg_incumbent_background_redecision_lane_rollout import (
    CLASS_PROTECTION_WORK, ClassProtectedBGIncumbentBackgroundRedecisionLaneRollout,
)
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.class_protected_bg_incumbent_counterfactual_replay import (
    ClassProtectedBGIncumbentCounterfactualSelector, collect_work, registered_targets,
    restore_planner_state,
)
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory
from algorithm.experiments.single_action_counterfactual_replay import prepare_checkpoint, resume_single_action_trajectory


OPTIONS = (FrameOption("single", (1,), 1., ((1.,),)),
           FrameOption("loaded-pair", (2,), 1.5, ((.5, 1.),), loaded=True),
           FrameOption("triple", (3,), 2., ((2., 2., 2.),)))
MODEL = FrameModel("toy", "class-incumbent-replay", 2, ("a",), (1.,), OPTIONS)
TAPE = [(0., 0, 0), (0., 0, 1), (3., 0, 2), (3., 0, 3), (3., 0, 4), (6., 0, 5)]


def normalized(value):
    return json.loads(json.dumps(value))


def controlled_planner():
    candidate = ClassProtectedBGIncumbentBackgroundRedecisionLaneRollout(
        OPTIONS, "class-replay", sample_count=1)
    score = candidate.score_background_roots

    def contrast(*args):
        actual = score(*args)
        if tuple(args[0]) == (3,) and args[2] == 3.:
            # Native ordinary BG selects loaded-pair1, while MW selects2.
            # Only the later objective contrast is controlled; its per-class
            # and aggregate S2 stay consistent and all native fees remain.
            class_cost = (actual.class_squared_costs[1][0] - Fraction(1),)
            return replace(actual,
                costs={**actual.costs, 0: actual.costs[1] - Fraction(1, 2)},
                squared_costs={**actual.squared_costs, 0: sum(class_cost, Fraction(0))},
                class_squared_costs={**actual.class_squared_costs, 0: class_cost})
        return actual

    candidate.score_background_roots = contrast
    return candidate


@pytest.fixture(scope="module")
def native():
    with pytest.MonkeyPatch.context() as patch:
        def no_future(rates, window, rng, cap):
            positive = sum(rate > 0 for rate in rates)
            for _ in range(positive):
                rng.random()
            return PlanningArrivals((), False, positive, len(rates))

        patch.setattr(policy_module, "sample_planning_arrivals", no_future)
        patch.setattr(background_module, "sample_planning_arrivals", no_future)
        selector = ClassProtectedBGIncumbentCounterfactualSelector(
            controlled_planner(), capture=range(40))
        factual = simulate_observed_rate_contextual_trajectory(
            MODEL, TAPE, 8., 10., selector, policy="factual")
        targets = registered_targets([{"tape_id": "toy"}], {"toy": factual})
        assert len(targets) == 1
        target = targets[0]
        assert (target["start"], target["factual_option_index"],
                target["replacement_option_index"], target["maxweight_option_index"]) == (3., 0, 1, 2)
        yield factual, selector, target


@pytest.fixture(scope="module")
def forced(native):
    factual, captured, target = native
    index = target["frame_index"]
    state = captured.states[index]
    candidate = controlled_planner()
    fees = restore_planner_state(candidate, state)
    selector = ClassProtectedBGIncumbentCounterfactualSelector(
        candidate, capture=(1,), force_first=True)
    branch = resume_single_action_trajectory(
        MODEL, 8., 10., selector, prepare_checkpoint(factual, index), policy="forced")
    return branch, selector, state, fees


def test_native_null_replays_full_metadata_rng_clock_and_loaded_commitments(native):
    factual, captured, target = native
    busy = prepare_checkpoint(factual, 1)
    assert busy["available"] == (1.5, 0.)
    assert [job["completion"] for job in busy["jobs"] if job["start"] is not None] == [.5, 1.]
    for index in (1, target["frame_index"]):
        candidate = controlled_planner()
        state = captured.states[index]
        fees = restore_planner_state(candidate, state)
        assert fees == {"checkpoint_rng_words_restored": 625,
                        "checkpoint_decision_counter_writes": 1}
        selector = ClassProtectedBGIncumbentCounterfactualSelector(candidate)
        resumed = resume_single_action_trajectory(
            MODEL, 8., 10., selector, prepare_checkpoint(factual, index), policy="null")
        assert resumed["jobs"] == factual["jobs"]
        assert normalized(factual["frames"][:index] + resumed["frames"]) == normalized(factual["frames"])
        assert normalized(selector.first_proposal["rollout"]) == normalized(factual["frames"][index]["rollout"])
        assert factual["lane_events"][:index] + resumed["lane_events"] == factual["lane_events"]
        assert resumed["last_frame_end"] == factual["last_frame_end"]
        assert candidate.rng.getstate() == captured.planner.rng.getstate()
        assert candidate.decision_counter == captured.planner.decision_counter


def test_first_nonmw_bg_restore_copies_class_vector_and_then_freely_replans(native, forced):
    factual, _, target = native
    branch, selector, _, _ = forced
    assert branch["frames"][0]["option_indices"] == (1,)
    assert selector.intervention["original"] == 0
    assert selector.intervention["replacement"] == selector.intervention["actual_applied_option_index"] == 1
    assert selector.intervention["maxweight_option_index"] == 2
    proposed = selector.first_proposal["rollout"]
    applied = branch["frames"][0]["rollout"]
    assert normalized(proposed) == normalized(factual["frames"][target["frame_index"]]["rollout"])
    assert proposed["chosen_predicted_class_squared_flow_s2"] != proposed["baseline_predicted_class_squared_flow_s2"]
    assert applied["chosen_predicted_class_squared_flow_s2"] == proposed["baseline_predicted_class_squared_flow_s2"]
    assert not applied["improved_incumbent"] and applied["changed"]
    completions = {job["index"]: job["completion"] for job in branch["jobs"]}
    assert completions[0] == .5 and completions[1] == 1.
    assert completions[2] == 3.5 and completions[3] == 4.
    assert branch["frames"][1]["available_before"] == (3.5, 4.5)
    assert branch["frames"][1]["start"] == 3.5
    assert branch["frames"][1]["option_indices"] == (0,)
    assert branch["frames"][1]["rollout"]["eligible"]
    assert branch["frames"][1]["rollout"]["sample_count"] == 1
    assert selector.work["override_calls"] == 1
    assert not hasattr(selector.planner, "pending_successor_index")


def test_proposal_state_advances_once_class_fees_and_extra_mw_scan_are_separate(native, forced):
    factual, _, target = native
    branch, selector, state, restore_fees = forced
    frame = factual["frames"][target["frame_index"]]
    reference = controlled_planner()
    restore_planner_state(reference, state)
    proposal = reference(OPTIONS, frame["queue"], frame["available_before"], frame["start"],
                         frame["lane_ids"][0], frame["waiting_ages"], frame["observed_arrival_counts"])
    assert proposal.improved_incumbent and proposal.incumbent_root_index == 1
    assert selector.states[1] == {"rng_state": reference.rng.getstate(),
                                  "decision_counter": reference.decision_counter}
    assert selector.planner.decision_counter == state["decision_counter"] + len(branch["frames"])
    assert selector.work["override_root_option_probes"] == len(OPTIONS)
    for key in CLASS_PROTECTION_WORK + (
            "root_option_probes", "incumbent_rollout_option_probes",
            "recourse_rollout_option_probes", "planning_random_draws"):
        assert selector.work[key] == sum(item["rollout"][key] for item in branch["frames"])
        assert selector.work[key] > 0
    work = collect_work(selector, branch)
    assert work["controller_constructors"] == work["recourse_controller_constructors"] == 1
    assert work["preprocessing_option_probes"] == work["preprocessing_recourse_option_probes"] == len(OPTIONS)
    assert work["root_option_probes"] == selector.work["root_option_probes"]
    assert work["override_root_option_probes"] == len(OPTIONS)
    for key in CLASS_PROTECTION_WORK:
        assert work[key] == selector.work[key]
    assert restore_fees["checkpoint_rng_words_restored"] == 625
    assert selector.status["proposal_calls"] == selector.status["applied_calls"] == selector.calls
    assert selector.status["proposal_improved_incumbent"] == 1
    assert selector.status["applied_improved_incumbent"] == 0


def test_native_stage_overflows_keep_partial_class_and_original_work_once(monkeypatch):
    class ScriptedRandom:
        def __init__(self, seed):
            self.gaps = chain((.1, .1, .1), repeat(1000.))

        def expovariate(self, rate):
            return next(self.gaps)

    monkeypatch.setattr(background_module, "Random", ScriptedRandom)
    for target in (policy_module, background_module):
        monkeypatch.setattr(target, "sample_planning_arrivals",
                            lambda *args: PlanningArrivals((), False, 0, 0))
    options = (FrameOption("single", (1,), 1., ((1.,),)),
               FrameOption("pair", (2,), 1.5, ((1.5, 1.5),)),
               FrameOption("triple", (3,), 2., ((2., 2., 2.),)),
               FrameOption("slow-single", (1,), 2., ((2.,),)))
    candidate = ClassProtectedBGIncumbentBackgroundRedecisionLaneRollout(
        options, "partial-new", sample_count=1, event_cap=2)
    original_call = background_module.BackgroundCompetitionContinuousLaneRollout.__call__

    def slow_child(self, *args):
        decision = original_call(self, *args)
        if self is candidate.inner_planner:
            return replace(decision, selection=replace(decision.selection,
                option_counts=(0, 0, 0, 1), option_indices=(3,), service=(1,), duration=2., exact=False),
                changed=True)
        return decision

    monkeypatch.setattr(background_module.BackgroundCompetitionContinuousLaneRollout,
                        "__call__", slow_child)
    new_selector = ClassProtectedBGIncumbentCounterfactualSelector(candidate)
    new = new_selector(options, (3,), (1000., 1005.), 1000., 0, ((10., 0., 0.),), (20,))
    assert new.background_rollout_overflow and not new.incumbent_background_rollout_overflow
    assert new.selection.option_indices == (1,) and not new.improved_incumbent
    assert new.class_squared_initializations == 4
    assert new.class_squared_accumulations == new.flow_evaluations == 2
    assert new.class_squared_score_reads == new.class_risk_comparisons == 0
    assert new.recourse_calls == 1
    old_options = options[:1]
    old_selector = ClassProtectedBGIncumbentCounterfactualSelector(
        ClassProtectedBGIncumbentBackgroundRedecisionLaneRollout(
            old_options, "partial-old", sample_count=1, event_cap=2))
    old = old_selector(old_options, (3,), (1000., 1005., 1005.), 1000., 0, ((0., 0., 0.),), (20,))
    assert old.incumbent_background_rollout_overflow and old.incumbent_flow_evaluations == 2
    assert not old.background_rollout_overflow
    for selector, decision in ((new_selector, new), (old_selector, old)):
        for key in CLASS_PROTECTION_WORK + (
                "incumbent_flow_evaluations", "incumbent_background_planning_events",
                "recourse_calls", "flow_evaluations"):
            assert selector.work[key] == getattr(decision, key)
        assert not selector.work["override_root_option_probes"]
        assert normalized(selector.first_proposal["rollout"])["class_squared_accumulations"] == decision.class_squared_accumulations
