"""Reuse identical native inner chooses within one real outer decision."""
from collections import Counter

from algorithm.bg_incumbent_background_redecision_lane_rollout import INCUMBENT_WORK
from algorithm.one_step_background_redecision_lane_rollout import RECOURSE_WORK
from algorithm.two_step_bg_incumbent_background_redecision_lane_rollout import TWO_STEP_WORK


MAIN_WORK = (
    "root_option_probes", "rollout_calls", "rollout_option_probes", "planning_events",
    "planning_random_draws", "planning_rate_reads", "rollout_idle_calls", "waiting_age_reads",
    "flow_evaluations", "rate_estimation_coordinate_reads", "rate_estimation_divisions",
    "tagged_planning_events", "background_planning_events", "background_planning_random_draws",
    "background_planning_rate_reads", "background_seed_initializations", "admitted_job_reads",
    "tag_flag_reads", "background_admissions",
)
NATIVE_WORK = MAIN_WORK + INCUMBENT_WORK + RECOURSE_WORK + TWO_STEP_WORK
BG_COMPARISONS = (
    "risk_comparisons", "ordering_comparisons", "strict_improvement_comparisons",
)
REUSE_WORK = (
    "choose_calls", "native_choose_calls", "cache_hits", "cache_lookups", "cache_inserts",
    "key_option_reads", "key_queue_coordinate_reads", "key_lane_reads", "key_age_reads",
    "key_rate_reads", "key_observed_count_reads", "key_rng_words_copied", "key_counter_reads",
    "post_planners_copied", "post_rng_words_copied", "post_counter_reads",
    "post_rate_reads", "post_observed_count_reads", "post_planners_restored",
    "post_rng_words_restored", "post_counter_writes", "post_rate_writes",
    "post_observed_count_writes", "cache_peak_entries",
)


class InnerDecisionReuse:
    """Keep current choices for one outer call and BG choices for one miss.

    The observed-context ``__call__`` wrappers still run on every invocation.
    A hit supplies the unchanged native ``_choose`` decision and restores only
    semantic state. Cached native work is logical work; its saved fields must
    be subtracted once when reporting actual execution. A current hit skips
    its whole subtree, whereas BG hits occur only inside executed current
    misses. MW and leaf-kernel caches and their actual ledgers are untouched.
    """
    def __init__(self, inner_current):
        self.current = inner_current
        self.background = inner_current.inner_planner
        self.caches = {"current": {}, "BG": {}}
        self.work = Counter()
        self._install(self.current, "current")
        self._install(self.background, "BG")
        self.begin_outer_decision()

    def begin_outer_decision(self):
        self.caches["current"].clear()
        self.caches["BG"].clear()
        self.work = Counter({level + "_" + name: 0
            for level in ("current", "BG") for name in REUSE_WORK})
        self.work.update({level + "_saved_" + name: 0
            for level in ("current", "BG") for name in NATIVE_WORK})
        self.work.update({"BG_saved_" + name: 0 for name in BG_COMPARISONS})
        self.work["outer_scope_resets"] = 1
        self.work["BG_scope_resets"] = 0

    @property
    def saved_work(self):
        return {level: Counter({name.removeprefix(level + "_saved_"): value
                for name, value in self.work.items() if name.startswith(level + "_saved_")})
                for level in ("current", "BG")}

    def _key(self, planner, level, context):
        options, queue, available, now, lane, waiting_ages = context
        ages = tuple(tuple(group) for group in waiting_ages)
        rates = tuple(planner.arrival_rates)
        counts = (tuple(planner._outer_observed_counts)
                  if hasattr(planner, "_outer_observed_counts") else None)
        rng = planner.rng.getstate()
        counter = planner.decision_counter
        prefix = level + "_key_"
        self.work[prefix + "option_reads"] += len(options)
        self.work[prefix + "queue_coordinate_reads"] += len(queue)
        self.work[prefix + "lane_reads"] += len(available)
        self.work[prefix + "age_reads"] += sum(map(len, ages))
        self.work[prefix + "rate_reads"] += len(rates)
        self.work[prefix + "observed_count_reads"] += len(counts) if counts is not None else 0
        self.work[prefix + "rng_words_copied"] += len(rng[1])
        self.work[prefix + "counter_reads"] += 1
        return tuple(options), tuple(queue), tuple(available), now, lane, ages, rates, counts, rng, counter

    def _capture(self, planner, level, touched_child):
        nodes = [planner]
        if touched_child:
            nodes.append(planner.inner_planner)
        states = []
        for node in nodes:
            rng = node.rng.getstate()
            rates = tuple(node.arrival_rates)
            counts = (tuple(node._outer_observed_counts)
                      if hasattr(node, "_outer_observed_counts") else None)
            states.append((node, rng, node.decision_counter, rates, counts))
            prefix = level + "_post_"
            self.work[prefix + "planners_copied"] += 1
            self.work[prefix + "rng_words_copied"] += len(rng[1])
            self.work[prefix + "counter_reads"] += 1
            self.work[prefix + "rate_reads"] += len(rates)
            self.work[prefix + "observed_count_reads"] += len(counts) if counts is not None else 0
        return states

    def _restore(self, states, level):
        for node, rng, counter, rates, counts in states:
            node.rng.setstate(rng)
            node.decision_counter = counter
            node.arrival_rates = rates
            if counts is not None:
                node._outer_observed_counts = counts
            prefix = level + "_post_"
            self.work[prefix + "planners_restored"] += 1
            self.work[prefix + "rng_words_restored"] += len(rng[1])
            self.work[prefix + "counter_writes"] += 1
            self.work[prefix + "rate_writes"] += len(rates)
            self.work[prefix + "observed_count_writes"] += len(counts) if counts is not None else 0

    def _saved(self, decision, level):
        for name in NATIVE_WORK:
            self.work[level + "_saved_" + name] += getattr(decision, name, 0)
        if level == "BG" and decision.eligible and not decision.planning_overflow:
            self.work["BG_saved_risk_comparisons"] += decision.candidate_count
            self.work["BG_saved_ordering_comparisons"] += (
                decision.candidate_count + decision.risk_feasible_candidates - 2)
            self.work["BG_saved_strict_improvement_comparisons"] += 1

    def _install(self, planner, level):
        native_choose = planner._choose
        cache = self.caches[level]

        def choose(*context):
            self.work[level + "_choose_calls"] += 1
            key = self._key(planner, level, context)
            self.work[level + "_cache_lookups"] += 1
            cached = cache.get(key)
            if cached is not None:
                decision, states = cached
                self.work[level + "_cache_hits"] += 1
                self._restore(states, level)
                self._saved(decision, level)
                return decision
            self.work[level + "_native_choose_calls"] += 1
            if level == "current":
                self.caches["BG"].clear()
                self.work["BG_scope_resets"] += 1
            decision = native_choose(*context)
            # A no-recourse choose does not touch the child at all. Preserve
            # whichever child state the calling branch brought to that hit.
            touched_child = level == "current" and decision.recourse_calls > 0
            states = self._capture(planner, level, touched_child)
            cache[key] = decision, states
            self.work[level + "_cache_inserts"] += 1
            self.work[level + "_cache_peak_entries"] = max(
                self.work[level + "_cache_peak_entries"], len(cache))
            return decision

        planner._choose = choose
