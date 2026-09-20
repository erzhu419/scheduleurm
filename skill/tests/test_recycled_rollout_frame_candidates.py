"""Unused rollout reserve continues one paid frontier; no arrival tape needed."""
from dataclasses import dataclass, fields
import itertools

import pytest

from algorithm.batch_rollout_frame_candidates import BatchRolloutFrameSelection, select_batch_rollout_frame
from algorithm.budgeted_frame_candidates import select_budgeted_frame
from algorithm.experiments.aligned_hardware_frame_model import capacity_certificate, load_model
from algorithm.experiments.aligned_hardware_lookahead import terminal_service_rates
from algorithm.lookahead_frame_candidates import select_lookahead_frame
import algorithm.recycled_rollout_frame_candidates as recycled


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float
    completion_offsets: tuple[tuple[float, ...], ...]


RATES = {"arrival_rates": (0.0, 0.0), "terminal_service_rates": (1.0, 1.0)}


def _toy(batch, duration, other_duration=1.0):
    return (Option((batch, 0), duration, ((duration,) * batch, ())),
            Option((0, 1), other_duration, ((), (other_duration,)))) + tuple(
        Option((100 + i, 0), 100.0, ((100.0,) * (100 + i), ())) for i in range(7))


def _assert_old_fields_equal(new, old):
    for field in fields(BatchRolloutFrameSelection):
        assert getattr(new, field.name) == getattr(old, field.name), field.name


def test_real_225_queues_preserve_rollout_or_match_one_full_budget_v4_search():
    model = load_model()
    rates = {"arrival_rates": (0.05, 0.15, 0.2), "terminal_service_rates": (0.4, 0.9, 1.2)}
    rollout_cases = resumed_cases = 0
    for queues in itertools.product(range(5), range(5), range(9)):
        for budget in (10, 14, 26, 120):
            old = select_batch_rollout_frame(model.options, queues, model.lanes, budget, **rates)
            new = recycled.select_recycled_rollout_frame(model.options, queues, model.lanes, budget, **rates)
            context = (queues, budget)
            assert new.initial_root_budget == old.root_budget, context
            assert new.initial_root_attempts == old.root_attempts, context
            assert new.initial_cost_candidate_evaluations == old.cost_candidate_evaluations, context
            assert new.recycled_candidate_attempts == new.root_attempts - old.root_attempts, context
            assert new.current_search_resumed == (new.recycled_candidate_attempts > 0), context
            assert new.cost_candidate_evaluations == (new.initial_cost_candidate_evaluations
                                                      + new.recycled_cost_candidate_evaluations), context
            assert new.rollout_attempts in (0, 4), context
            assert new.attempts == new.root_attempts + new.rollout_attempts <= budget, context
            if old.rollout_used or budget == 10:
                _assert_old_fields_equal(new, old)
                assert not new.reclaimed_reserve and not new.current_search_resumed, context
                rollout_cases += old.rollout_used
            else:
                full = select_budgeted_frame(model.options, queues, model.lanes, budget)
                full_cost = select_lookahead_frame(model.options, queues, model.lanes, budget, **rates)
                assert new.reclaimed_reserve and new.root_budget == budget, context
                resumed_cases += new.current_search_resumed
                for name in ("option_indices", "service", "duration", "score", "upper_bound", "gap", "epsilon",
                             "attempts", "seed_attempts", "feasible_candidate_count", "local_score_evaluations",
                             "bound_option_probes", "bound_threshold_evaluations", "frontier_peak", "exact"):
                    assert getattr(new, name) == getattr(full, name), (name, context)
                for name in ("seed_support", "support_floor", "eligible_candidate_count",
                             "support_cap_excluded_candidate_count", "support_incumbent_lookahead_cost"):
                    assert getattr(new, name) == getattr(full_cost, name), (name, context)
                assert new.rollout_attempts == new.rollout_cost_candidate_evaluations == 0, context
                assert new.lookahead_cost == full_cost.support_incumbent_lookahead_cost, context
                assert new.root_lookahead_cost == full_cost.lookahead_cost, context
                assert new.cost_candidate_evaluations >= new.eligible_candidate_count, context
            assert new.score >= new.seed_support, context
            assert new.score >= 0.99 * new.support_incumbent_score, context
            departures = tuple(sum(model.options[j].requirements[i] for j in new.option_indices if j is not None)
                               for i in range(len(queues)))
            duration = max((1.0, *(model.options[j].duration for j in new.option_indices if j is not None)))
            assert departures == new.service and all(d <= q for d, q in zip(departures, queues)), context
            assert new.score == sum(q * d for q, d in zip(queues, departures)) / duration, context
            assert new.gap == max(0.0, new.upper_bound - new.score), context
            assert new.exact == (new.gap == 0.0), context
            assert new.epsilon == (new.gap / sum(queues) if sum(queues) else 0.0), context
    assert rollout_cases > 0 and resumed_cases > 0


def test_recovery_improves_support_and_preserves_one_frontier_and_cumulative_cost(monkeypatch):
    options = _toy(3, 3.0)
    counts = {"push": 0, "pop": 0}
    frontier_ids = set()
    original_push, original_pop = recycled.heapq.heappush, recycled.heapq.heappop

    def push(heap, item):
        counts["push"] += 1
        frontier_ids.add(id(heap))
        return original_push(heap, item)

    def pop(heap):
        counts["pop"] += 1
        frontier_ids.add(id(heap))
        return original_pop(heap)

    monkeypatch.setattr(recycled.heapq, "heappush", push)
    monkeypatch.setattr(recycled.heapq, "heappop", pop)
    full = select_budgeted_frame(options, (3, 1), 2, 14)
    expected_counts = counts.copy()
    counts.update(push=0, pop=0)
    frontier_ids.clear()
    new = recycled.select_recycled_rollout_frame(options, (3, 1), 2, 14, **RATES)
    assert counts == expected_counts
    assert len(frontier_ids) == 1
    assert new.initial_root_budget == new.initial_root_attempts == 10
    assert new.root_budget == new.root_attempts == 14
    assert new.recycled_candidate_attempts == 4 and new.current_search_resumed
    assert new.rollout_attempts == 0
    assert new.seed_support == 3.0
    assert new.score == full.score == pytest.approx(10.0 / 3.0)
    assert new.option_indices == (0, 1)
    # The initial A candidate was priced, then excluded by the raised floor.
    # Final eligibility is not the number of analytic evaluations already paid.
    assert new.initial_cost_candidate_evaluations == new.recycled_cost_candidate_evaluations == 1
    assert new.cost_candidate_evaluations == 2 > new.eligible_candidate_count == 1
    assert new.completion_offset_reads == 7  # initial A:3; final A+B:4.


class CountedTuple(tuple):
    def __new__(cls, values, reads, key):
        instance = super().__new__(cls, values)
        instance.reads, instance.key = reads, key
        return instance

    def __iter__(self):
        for value in super().__iter__():
            self.reads[self.key] = self.reads.get(self.key, 0) + 1
            yield value


def test_second_pricing_reuses_existing_objects_without_rereading_their_offsets():
    reads = {}
    options = tuple(Option(CountedTuple(option.requirements, reads, "requirements"), option.duration,
                           tuple(CountedTuple(group, reads, "offsets") for group in option.completion_offsets))
                    for option in _toy(10, 1.0))
    new = recycled.select_recycled_rollout_frame(options, (10, 1), 2, 14, **RATES)
    assert new.current_search_resumed and new.rollout_attempts == 0
    assert new.seed_support == 100.0 and new.score == 101.0
    # Initial A remains above the final 99.99 floor. The new A prefix is a
    # distinct paid object; initial A is cached, prefix A and A+B are priced.
    assert new.eligible_candidate_count == new.cost_candidate_evaluations == 3
    assert new.initial_cost_candidate_evaluations == 1
    assert new.recycled_cost_candidate_evaluations == 2
    assert new.completion_offset_reads == reads["offsets"] == 31
    assert reads["requirements"] == 18  # no second seed/profile setup.
    assert new.next_queue_coordinate_evaluations == new.terminal_value_coordinate_evaluations == 6
    assert new.physical_action_completion_offset_reads == 0


def test_new_unrolled_challenger_does_not_replace_the_final_support_incumbent():
    options = _toy(10, 1.0, other_duration=1.009)
    rates = {"arrival_rates": (0.0, 0.0), "terminal_service_rates": (1.0, 100.0)}
    full_cost = select_lookahead_frame(options, (10, 1), 2, 14, **rates)
    new = recycled.select_recycled_rollout_frame(options, (10, 1), 2, 14, **rates)
    assert new.current_search_resumed and new.rollout_attempts == 0
    assert full_cost.option_indices == (0, None)
    assert new.option_indices == (0, 1)
    assert new.score > full_cost.score
    assert new.root_lookahead_cost == full_cost.lookahead_cost < new.lookahead_cost
    assert new.selected_rollout_cost is None


def test_naturally_finished_search_reclaims_reserve_without_search_or_pricing_again():
    model = load_model()
    rates = {"arrival_rates": (0.05, 0.15, 0.2), "terminal_service_rates": (0.4, 0.9, 1.2)}
    old = select_batch_rollout_frame(model.options, (0, 0, 0), 2, 14, **rates)
    new = recycled.select_recycled_rollout_frame(model.options, (0, 0, 0), 2, 14, **rates)
    assert old.support_search_exact
    assert new.reclaimed_reserve and not new.current_search_resumed
    assert new.recycled_candidate_attempts == new.recycled_cost_candidate_evaluations == 0
    for field in fields(BatchRolloutFrameSelection):
        if field.name != "root_budget":
            assert getattr(new, field.name) == getattr(old, field.name), field.name
    assert new.root_budget == 14 and new.initial_root_budget == 10


def test_no_reserve_k10_does_not_inspect_future_or_restart_current_search(monkeypatch):
    def forbidden(*args):
        pytest.fail("K10 has no reserve to compare or recycle")

    monkeypatch.setattr(recycled, "_same_physical_action", forbidden)
    monkeypatch.setattr(recycled, "_rollout_old_jobs", forbidden)
    options = _toy(3, 3.0)
    old = select_batch_rollout_frame(options, (3, 1), 2, 10, **RATES)
    new = recycled.select_recycled_rollout_frame(options, (3, 1), 2, 10, **RATES)
    _assert_old_fields_equal(new, old)
    assert not new.reclaimed_reserve
    assert new.recycled_candidate_attempts == new.recycled_cost_candidate_evaluations == 0


def test_known_low_load_counterexample_keeps_the_frozen_batch_proxy_choice():
    model = load_model()
    certificate = capacity_certificate(model, 0.5)
    rates = {"arrival_rates": certificate["arrival_rates"],
             "terminal_service_rates": terminal_service_rates(model, certificate)}
    old = select_batch_rollout_frame(model.options, (1, 4, 8), 2, 26, **rates)
    new = recycled.select_recycled_rollout_frame(model.options, (1, 4, 8), 2, 26, **rates)
    assert old.rollout_used and not old.rollout_vetoed
    assert old.option_indices == (7, 7)
    _assert_old_fields_equal(new, old)
    assert not new.reclaimed_reserve and new.rollout_attempts == 4
