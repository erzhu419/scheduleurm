"""Terminal native proposals preserve cached state and fixed-population costs."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction
import json
from random import Random
from types import SimpleNamespace
from unittest.mock import patch

from algorithm import parallel_root_lane_rollout as native
from algorithm.background_competition_continuous_lane_rollout import _LazyBackgroundArrivals
from algorithm.exact_reuse_lane_rollout import ExactReuseTwoStepCurrentPolicyBGIncumbentLaneRollout
from algorithm.experiments.aligned_hardware_frame_model import FrameOption
from algorithm.experiments.bg_incumbent_counterfactual_replay import FEES, STATUS
from algorithm.experiments.common_horizon_flow_rollout import evaluate_root_sample
from algorithm.experiments.native_terminal_flow_rollout import _simulate_native_terminal, evaluate_cached
from algorithm.experiments.persistent_continuation_cross_selector import (
    capture_planner_tree, restore_planner_tree,
)
from algorithm.two_step_bg_incumbent_background_redecision_lane_rollout import TWO_STEP_WORK
from skill.tests.test_competing_terminal_flow_rollout import NativeMWFixture, FixedSuffix, job, json_value


F = Fraction


class CompleteProposalFixture(NativeMWFixture):
    """Known decisions with complete proposal metadata and observable state use."""
    def __init__(self, options):
        super().__init__(options)
        self.rng = Random(17)
        self.decision_counter = 7
        self.arrival_rates = (0.,) * len(self.requirements[0])
        self._outer_observed_counts = (0,) * len(self.arrival_rates)
        self.calls = []
        self.reuse_work = {}

    def __call__(self, options, queue, available, now, lane, ages, counts):
        self.calls.append((options, queue, available, now, lane, ages, counts))
        self.decision_counter += 1
        self.rng.random()
        self.arrival_rates = tuple(count / now for count in counts)
        self._outer_observed_counts = tuple(counts)
        index, _ = self._best_index(queue)
        fees = dict.fromkeys(FEES + STATUS + TWO_STEP_WORK, 0)
        fees.update(root_option_probes=len(options), planning_random_draws=1)
        return SimpleNamespace(**fees, selection=SimpleNamespace(option_indices=(index,)),
            inner_current_work={"fixture": 2}, inner_current_fallbacks={})


def test_full_proposals_recover_H_fifo_keep_busy_release_and_score_only_fixed_jobs():
    options = (FrameOption("A", (1, 0), 2., ((.5,), ())),
        FrameOption("B", (0, 1), 2., ((), (1.5,))))
    jobs = [job(("visible", 0, 0), 0, -1, "visible", 0, F(1, 2)),
        job(("prefix", 0), 0, 1, "prefix", 5, F(11, 2)),
        job(("prefix", 1), 1, 2, "prefix", 4, F(11, 2)),
        job(("background", 0), 1, 3, "background", 6, F(15, 2)),
        job(("visible", 1, 0), 1, 0, "visible", 3, F(9, 2))]
    saved = deepcopy(jobs)
    state = dict(queue=(1, 2), waiting_ids=((jobs[1]["id"],), (jobs[2]["id"], jobs[3]["id"])),
        lane_ends=(F(4), F(5)), counts=(2, 3))
    planner = CompleteProposalFixture(options)
    result = _simulate_native_terminal(planner, jobs, state, FixedSuffix(), origin=F(100),
        horizon=F(4), common_area=F(21, 2), waiting_area=F(9), inflight_area=F(3, 2))
    assert jobs == saved
    assert result["complete"] and not result["overflow"]
    # B0@4, B1@5, unscored B@6/@7, A@8: fixed residual 9, common area 10.5.
    assert result["cost"] == F(39, 2) and result["terminal_cost"] == 9
    assert result["common_area"] == F(21, 2)
    assert [call[3] for call in planner.calls] == [104., 105., 106., 107., 108.]
    assert planner.calls[0][2] == (104., 105.)
    assert planner.calls[0][5] == ((3.,), (2., 1.))
    assert [call[6] for call in planner.calls] == [(2, 3), (2, 5), (2, 6), (2, 6), (2, 6)]
    assert planner.decision_counter == 12
    assert result["logical_work"]["root_option_probes"] == 10
    assert result["logical_work"]["planning_random_draws"] == 5
    assert result["logical_work"]["inner_current_fixture"] == 10
    restored = {entry["id"]: entry for entry in result["population"]["jobs"]}
    assert set(restored) == {entry["id"] for entry in jobs}
    assert restored[jobs[1]["id"]]["completion"] == F(17, 2)
    assert restored[jobs[3]["id"]]["completion"] == F(13, 2)
    assert restored[jobs[4]["id"]] == jobs[4]
    assert result["population"]["remaining"] == 0


class RecordedAnchor(ExactReuseTwoStepCurrentPolicyBGIncumbentLaneRollout):
    def score_background_roots(self, *context):
        self.paths = tuple(context[-1])
        return super().score_background_roots(*context)


def test_public_real_JSON_restores_all_native_states_and_replays_each_complete_proposal():
    options = (FrameOption("single", (1,), 2., ((.5,),)),)
    seed, samples, cap = 0, 1, 128
    context = dict(queue=(2,), available=(100., 101.), now=100., lane=0,
        ages=((2., 1.),), counts=(200,))
    reference = RecordedAnchor(options, seed, samples, cap)
    before = capture_planner_tree(reference, Counter())
    proposal = reference(options, context["queue"], context["available"], context["now"],
        context["lane"], context["ages"], context["counts"])
    assert proposal.eligible and sum(context["queue"]) <= proposal.queue_limit
    # Seed 0 at rate 2 has three actual prefix births before t=2. The two
    # initial jobs fit queue_limit=2; only four singletons can start before H=4,
    # so a fixed H-waiting job remains without bypassing native eligibility.
    assert len(reference.paths) == 1 and len(reference.paths[0].events) == 3
    after = capture_planner_tree(reference, Counter())
    old = evaluate_root_sample(options, seed, reference.paths, before, after, context,
        0, 0, sample_count=samples, event_cap=cap, workers=2)
    assert old["complete"] and sum(old["flow"]["terminal_state"]["queue"]) > 0
    saved_tree = old["flow"]["final_tree"]
    assert len(saved_tree["planners"]) == 3
    cached = json.loads(json.dumps(json_value(old), default=lambda value: json_value(asdict(value))))
    saved = deepcopy(cached)
    traced = []
    cls = native.ParallelRootTwoStepCurrentPolicyBGIncumbentLaneRollout
    native_call = cls.__call__

    def observe(planner, *arguments):
        entry = capture_planner_tree(planner, Counter())
        decision = native_call(planner, *arguments)
        traced.append((arguments, entry, decision, capture_planner_tree(planner, Counter())))
        return decision

    with patch.object(cls, "__call__", observe):
        result = evaluate_cached(options, seed, cached, context["now"],
            workers=2, sample_count=samples, event_cap=cap)
    assert cached == saved and result["complete"]
    assert result["root"] == result["sample_index"] == 0
    assert result["environment"] == cached["environment"]
    flow = result["flow"]
    assert traced and traced[0][1] == saved_tree
    assert flow["initial_tree"] == saved_tree
    assert traced[0][0][3] == 104.
    restore_planner_tree(reference, saved_tree)
    logical = Counter()
    for current, entry, proposed, exit_state in traced:
        assert capture_planner_tree(reference, Counter()) == entry
        expected = reference(*current)
        assert expected == proposed
        assert capture_planner_tree(reference, Counter()) == exit_state
        logical.update({name: getattr(expected, name) for name in FEES + TWO_STEP_WORK})
        for values in (expected.inner_current_work, expected.inner_current_fallbacks):
            logical.update({"inner_current_" + name: value for name, value in values.items()})
    assert flow["logical_work"] == dict(logical)
    assert flow["final_tree"] == capture_planner_tree(reference, Counter())
    assert flow["final_tree"]["planners"][0]["decision_counter"] == (
        saved_tree["planners"][0]["decision_counter"] + len(traced))
    assert flow["common_area"] == old["flow"]["common_area"]
    # There is one feasible service action. Later births cannot pass fixed H jobs.
    assert flow["terminal_cost"] == old["flow"]["terminal_cost"]
    assert flow["cost"] == old["flow"]["cost"]
    assert flow["population"]["total"] == old["flow"]["population"]["total"]
    environment = old["environment"]
    suffix = _LazyBackgroundArrivals(environment["rates"], environment["window_s"],
        environment["seed"], cap, len(environment["prefix_events"]))
    assert suffix.release_through(old["flow"]["horizon"])
    assert tuple(suffix.events) == old["flow"]["background_events"]
    assert suffix.release_through(F(traced[-1][0][3]) - F(context["now"]))
    assert tuple(suffix.events) == flow["background_events"]
    assert flow["suffix_work"] == (len(suffix.events), suffix.random_draws,
        suffix.rate_reads, suffix.seed_initializations)


def test_external_cap_keeps_partial_population_and_consumed_proposal():
    options = (FrameOption("single", (1,), 2., ((.5,),)),)
    suffix = _LazyBackgroundArrivals((1.,), 2., 0, 1, 0)
    assert suffix.release_through(F(4)) and len(suffix.events) == 1
    birth = suffix.events[0][0]
    jobs = [job(("visible", 0, 0), 0, 0, "visible", 5, F(11, 2)),
        job(("background", 0), 0, birth, "background", F(11, 2), 6),
        job(("visible", 0, 1), 0, -1, "visible", 3, F(7, 2)),
        job(("visible", 0, 2), 0, -2, "visible", F(7, 2), 4)]
    state = dict(queue=(2,), waiting_ids=((jobs[0]["id"], jobs[1]["id"]),),
        lane_ends=(F(5), F(11, 2)), counts=(4,))
    common = 4 + (4 - birth) + F(7, 2) + 4
    waiting = 4 + (4 - birth) + 3 + F(7, 2)
    planner = CompleteProposalFixture(options)
    result = _simulate_native_terminal(planner, jobs, state, suffix, origin=F(100),
        horizon=F(4), common_area=common, waiting_area=waiting, inflight_area=F(1))
    assert result["overflow"] and not result["complete"]
    assert len(suffix.events) == 2 and result["population"]["remaining"] == 1
    assert result["population"]["total"] == 4
    assert planner.decision_counter == 8 and len(planner.calls) == 1
    assert result["logical_work"]["planning_random_draws"] == 1
    assert result["terminal_cost"] == F(3, 2) + suffix.events[-1][0] - 4
    assert result["cost"] == common + result["terminal_cost"]


if __name__ == "__main__":
    test_full_proposals_recover_H_fifo_keep_busy_release_and_score_only_fixed_jobs()
    test_public_real_JSON_restores_all_native_states_and_replays_each_complete_proposal()
    test_external_cap_keeps_partial_population_and_consumed_proposal()
    print("3 passed", flush=True)
