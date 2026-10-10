"""Exact computation reuse for the frozen two-current-call lane policy."""
from collections import Counter

from algorithm.arrival_aware_lane_rollout import ArrivalAwareLaneRollout
from algorithm.background_competition_continuous_lane_rollout import (
    BackgroundCompetitionContinuousLaneRollout,
)
from algorithm.dyadic_background_rollout import score_background_roots
from algorithm.exact_reuse_inner_decision import InnerDecisionReuse
from algorithm.two_step_bg_incumbent_original_risk_lane_rollout import (
    TwoStepBGIncumbentOriginalRiskLaneRollout,
)
from algorithm.two_step_current_policy_bg_incumbent_lane_rollout import (
    TwoStepCurrentPolicyBGIncumbentLaneRollout,
)


class MWSelectionReuse:
    """Share exact MW scans within one real decision and one fixed catalogue.

    All users are constructed from the same options in the same order. Only
    the queue-dependent pair (selected index, feasible count) is cached. FIFO
    ages, arrival samples and root scores are evaluated by the native policy.
    Counts are cumulative actual cache work; native decision fees stay logical.
    """
    def __init__(self, option_count, event_cap):
        self.option_count = option_count
        self.capacity = max(1, event_cap * option_count)
        self.cache = {}
        self.queries = self.cache_hits = self.cache_misses = 0
        self.cache_key_coordinate_reads = 0
        self.cache_clear_operations = self.cache_cleared_entries = 0

    def _clear(self):
        if self.cache:
            self.cache_clear_operations += 1
            self.cache_cleared_entries += len(self.cache)
            self.cache.clear()

    def begin_outer_decision(self):
        self._clear()

    def finish_outer_decision(self):
        self._clear()

    def best_index(self, planner, queue):
        queue = tuple(queue)
        self.queries += 1
        self.cache_key_coordinate_reads += len(queue)
        cached = self.cache.get(queue)
        if cached is not None:
            self.cache_hits += 1
            return cached
        self.cache_misses += 1
        selected = ArrivalAwareLaneRollout._best_index(planner, queue)
        if len(self.cache) >= self.capacity:
            self._clear()
        self.cache[queue] = selected
        return selected

    @property
    def work(self):
        return {
            "queries": self.queries, "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "logical_option_probes": self.queries * self.option_count,
            "actual_option_probes": self.cache_misses * self.option_count,
            "cache_key_coordinate_reads": self.cache_key_coordinate_reads,
            "cache_lookup_operations": self.queries,
            "cache_store_operations": self.cache_misses,
            "cache_clear_operations": self.cache_clear_operations,
            "cache_cleared_entries": self.cache_cleared_entries,
        }


class _ExactMWSelection:
    def _best_index(self, queue):
        return self._mw_reuse.best_index(self, queue)


class ExactReuseBackgroundCompetitionContinuousLaneRollout(
        _ExactMWSelection, BackgroundCompetitionContinuousLaneRollout):
    """Ordinary BG with reusable MW scans and exact integer flow arithmetic."""
    def __init__(self, options, planning_seed, sample_count=8, event_cap=128,
                 mw_reuse=None):
        super().__init__(options, planning_seed, sample_count, event_cap)
        self._mw_reuse = (MWSelectionReuse(len(self.options), event_cap)
                          if mw_reuse is None else mw_reuse)
        self.kernel_work = Counter()

    def score_background_roots(self, queues, available, now, lane, waiting_ages, paths):
        return score_background_roots(
            self, queues, available, now, lane, waiting_ages, paths)


class ExactReuseTwoStepBGIncumbentOriginalRiskLaneRollout(
        _ExactMWSelection, TwoStepBGIncumbentOriginalRiskLaneRollout):
    """The frozen current continuation, sharing the same pure MW memo."""
    def __init__(self, options, planning_seed, sample_count=8, event_cap=128,
                 mw_reuse=None):
        BackgroundCompetitionContinuousLaneRollout.__init__(
            self, options, planning_seed, sample_count, event_cap)
        self._mw_reuse = (MWSelectionReuse(len(self.options), event_cap)
                          if mw_reuse is None else mw_reuse)
        self.inner_planner = ExactReuseBackgroundCompetitionContinuousLaneRollout(
            self.options, planning_seed, sample_count, event_cap, self._mw_reuse)
        self.preprocessing_work.update({"recourse_" + key: value
            for key, value in self.inner_planner.preprocessing_work.items()})
        self._outer_observed_counts = (0,) * len(self.arrival_rates)
        self._recourse_work, self._two_step_work = Counter(), Counter()


class ExactReuseTwoStepCurrentPolicyBGIncumbentLaneRollout(
        _ExactMWSelection, TwoStepCurrentPolicyBGIncumbentLaneRollout):
    """Execute the frozen policy exactly, reporting physical reuse separately.

    Each real decision clears the bounded MW memo and resets the current/BG
    decision memo scopes. Cached decisions restore their exact RNG and causal
    state; native wrappers still update counts, rates and call counters. The
    returned decision and its logical work fields retain their original type
    and values. ``reuse_work`` records cumulative executed optimization work;
    ``last_reuse_work`` contains the latest real-decision increments.
    """
    def __init__(self, options, planning_seed, sample_count=8, event_cap=128):
        BackgroundCompetitionContinuousLaneRollout.__init__(
            self, options, planning_seed, sample_count, event_cap)
        self._mw_reuse = MWSelectionReuse(len(self.options), event_cap)
        self.inner_planner = ExactReuseTwoStepBGIncumbentOriginalRiskLaneRollout(
            self.options, planning_seed, sample_count, event_cap, self._mw_reuse)
        self.preprocessing_work.update({"recourse_" + key: value
            for key, value in self.inner_planner.preprocessing_work.items()})
        self._outer_observed_counts = (0,) * len(self.arrival_rates)
        self._recourse_work, self._two_step_work = Counter(), Counter()
        self._inner_current_work, self._inner_current_fallbacks = Counter(), Counter()
        self._inner_decision_reuse = InnerDecisionReuse(self.inner_planner)
        self._inner_decision_reuse_work = Counter()
        self.last_reuse_work = {}

    @property
    def reuse_work(self):
        return {
            "mw_reuse_work": self._mw_reuse.work,
            "inner_decision_reuse_work": dict(self._inner_decision_reuse_work),
            "kernel_work": dict(self.inner_planner.inner_planner.kernel_work),
        }

    def _choose(self, options, queues, available, now, lane, waiting_ages):
        before = self.reuse_work
        self._mw_reuse.begin_outer_decision()
        self._inner_decision_reuse.begin_outer_decision()
        try:
            decision = super()._choose(options, queues, available, now, lane, waiting_ages)
        finally:
            self._mw_reuse.finish_outer_decision()
            scope_work = self._inner_decision_reuse.work
            peaks = ("current_cache_peak_entries", "BG_cache_peak_entries")
            self._inner_decision_reuse_work.update(
                {name: value for name, value in scope_work.items() if name not in peaks})
            for name in peaks:
                self._inner_decision_reuse_work[name] = max(
                    self._inner_decision_reuse_work[name], scope_work.get(name, 0))
            after = self.reuse_work
            self.last_reuse_work = {
                group: {name: values.get(name, 0) - before[group].get(name, 0)
                        for name in values.keys() | before[group].keys()}
                for group, values in after.items()
            }
            self.last_reuse_work["inner_decision_reuse_work"].update(
                {name: scope_work.get(name, 0) for name in peaks})
        return decision
