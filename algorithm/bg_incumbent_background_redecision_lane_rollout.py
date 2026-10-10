"""Improve the ordinary BG root using one BG continuation redecision."""
from collections import Counter
from dataclasses import dataclass, fields, replace
from fractions import Fraction
from math import inf, nextafter

from algorithm.age_tail_constrained_lane_rollout import AgeRootScores
from algorithm.arrival_aware_lane_rollout import sample_planning_arrivals
from algorithm.background_competition_continuous_lane_rollout import (
    BackgroundCompetitionContinuousLaneRollout, BackgroundRootScores,
)
from algorithm.one_step_background_redecision_lane_rollout import (
    RECOURSE_WORK, OneStepBackgroundRedecisionDecision,
    OneStepBackgroundRedecisionLaneRollout,
)


INCUMBENT_SCORE_WORK = (
    "root_option_probes", "rollout_calls", "rollout_option_probes", "rollout_idle_calls",
    "waiting_age_reads", "flow_evaluations", "background_planning_events",
    "background_planning_random_draws", "background_planning_rate_reads",
    "background_seed_initializations", "admitted_job_reads", "tag_flag_reads",
    "background_admissions",
)
INCUMBENT_WORK = tuple("incumbent_" + name for name in INCUMBENT_SCORE_WORK) + (
    "incumbent_risk_comparisons", "incumbent_ordering_comparisons",
    "incumbent_strict_improvement_comparisons", "incumbent_membership_checks",
    "incumbent_lookahead_risk_comparisons", "incumbent_lookahead_ordering_comparisons",
    "incumbent_lookahead_strict_improvement_comparisons",
)


@dataclass(frozen=True)
class BGIncumbentBackgroundRedecisionDecision(OneStepBackgroundRedecisionDecision):
    incumbent_maxweight_root_index: int | None = None
    incumbent_root_index: int | None = None
    incumbent_changed: bool = False
    incumbent_background_rollout_overflow: bool = False
    improved_incumbent: bool = False
    incumbent_candidate_count: int = 0
    incumbent_risk_feasible_candidates: int = 0
    incumbent_risk_filtered_candidates: int = 0
    incumbent_mw_predicted_completion_s: float | None = None
    incumbent_predicted_completion_s: float | None = None
    incumbent_mw_predicted_squared_flow_s2: float | None = None
    incumbent_predicted_squared_flow_s2: float | None = None
    incumbent_actual_predicted_squared_flow_s2: float | None = None
    incumbent_root_option_probes: int = 0
    incumbent_rollout_calls: int = 0
    incumbent_rollout_option_probes: int = 0
    incumbent_rollout_idle_calls: int = 0
    incumbent_waiting_age_reads: int = 0
    incumbent_flow_evaluations: int = 0
    incumbent_background_planning_events: int = 0
    incumbent_background_planning_random_draws: int = 0
    incumbent_background_planning_rate_reads: int = 0
    incumbent_background_seed_initializations: int = 0
    incumbent_admitted_job_reads: int = 0
    incumbent_tag_flag_reads: int = 0
    incumbent_background_admissions: int = 0
    incumbent_risk_comparisons: int = 0
    incumbent_ordering_comparisons: int = 0
    incumbent_strict_improvement_comparisons: int = 0
    incumbent_membership_checks: int = 0
    incumbent_lookahead_risk_comparisons: int = 0
    incumbent_lookahead_ordering_comparisons: int = 0
    incumbent_lookahead_strict_improvement_comparisons: int = 0


class BGIncumbentBackgroundRedecisionLaneRollout(OneStepBackgroundRedecisionLaneRollout):
    """Use ordinary BG as the root incumbent before one-step improvement.

    Both scorers consume the same outer sampled prefix population. Their
    background suffixes are generated and charged separately. The second
    stage admits only roots feasible under ordinary BG's S2-versus-MW rule
    whose one-step S2 is no worse than the BG incumbent's. It requires strict
    aggregate S1 improvement, with BG-first ties. An ordinary-BG overflow
    falls back to MW; a later one-step overflow retains the verified BG root.
    The actual controller freely replans, with no successor commitment.
    """
    def maxweight(self, options, queues, available, now, lane, waiting_ages,
                  observed_arrival_counts=None):
        decision = super().maxweight(options, queues, available, now, lane, waiting_ages)
        indices = decision.selection.option_indices
        return BGIncumbentBackgroundRedecisionDecision(
            **{field.name: getattr(decision, field.name)
               for field in fields(OneStepBackgroundRedecisionDecision)},
            incumbent_maxweight_root_index=indices[0] if indices else None)

    def _choose(self, options, queues, available, now, lane, waiting_ages):
        self._recourse_work = Counter(recourse_root_count_reads=len(self._outer_observed_counts))
        incumbent_work = Counter()

        def finish(decision):
            return replace(decision,
                **{name: self._recourse_work[name] for name in RECOURSE_WORK},
                **{name: incumbent_work[name] for name in INCUMBENT_WORK})

        baseline = self.maxweight(options, queues, available, now, lane, waiting_ages)
        if not 0 < sum(queues) <= baseline.queue_limit:
            return finish(baseline)
        maxweight_root, = baseline.selection.option_indices
        paths = []
        for _ in range(self.sample_count):
            paths.append(sample_planning_arrivals(
                self.arrival_rates, self.maximum_duration, self.rng, self.event_cap))
            if paths[-1].overflow:
                break
        metadata = dict(
            eligible=True, tagged_planning_events=sum(len(path.events) for path in paths),
            planning_events=sum(len(path.events) for path in paths),
            planning_random_draws=sum(path.random_draws for path in paths),
            planning_rate_reads=sum(path.rate_reads for path in paths),
            planning_overflow=any(path.overflow for path in paths),
            window_s=self.maximum_duration, sample_count=len(paths))
        if metadata["planning_overflow"]:
            return finish(replace(baseline, **metadata))
        original = BackgroundCompetitionContinuousLaneRollout.score_background_roots(
            self, queues, available, now, lane, waiting_ages, paths)
        for name in INCUMBENT_SCORE_WORK:
            incumbent_work["incumbent_" + name] = getattr(original, name)
        baseline = replace(baseline,
            incumbent_candidate_count=original.candidate_count,
            incumbent_background_rollout_overflow=original.background_rollout_overflow)
        if original.background_rollout_overflow:
            metadata["planning_overflow"] = True
            return finish(replace(baseline, **metadata))

        original_permitted = frozenset(index for index in original.costs
            if original.squared_costs[index] <= original.squared_costs[maxweight_root])

        def original_ordering(index):
            return original.costs[index], index != maxweight_root, self.durations[index], index

        original_unconstrained = min(original.costs, key=original_ordering)
        incumbent = min(original_permitted, key=original_ordering)
        original_changed = original.costs[incumbent] < original.costs[maxweight_root]
        if not original_changed:
            incumbent = maxweight_root
        incumbent_work["incumbent_risk_comparisons"] = original.candidate_count
        incumbent_work["incumbent_ordering_comparisons"] = (
            original.candidate_count - 1 + len(original_permitted) - 1)
        incumbent_work["incumbent_strict_improvement_comparisons"] = 1
        baseline = replace(baseline,
            incumbent_root_index=incumbent, incumbent_changed=incumbent != maxweight_root,
            incumbent_risk_feasible_candidates=len(original_permitted),
            incumbent_risk_filtered_candidates=original.candidate_count - len(original_permitted),
            incumbent_mw_predicted_completion_s=float(original.costs[maxweight_root] / len(paths)),
            incumbent_predicted_completion_s=float(original.costs[incumbent] / len(paths)),
            incumbent_mw_predicted_squared_flow_s2=float(
                original.squared_costs[maxweight_root] / len(paths)),
            incumbent_predicted_squared_flow_s2=float(original.squared_costs[incumbent] / len(paths)),
            incumbent_actual_predicted_squared_flow_s2=float(
                original.squared_costs[incumbent] / len(paths)))

        def selection_for_root(root):
            selected = baseline.selection
            if root == maxweight_root:
                return selected
            support = Fraction(sum(q * d for q, d in zip(
                queues, self.requirements[root], strict=True))) / self.durations[root]
            best_support = Fraction(sum(q * d for q, d in zip(
                queues, selected.service, strict=True))) / self.durations[maxweight_root]
            score, score_exact = float(support), support == best_support
            upper = score if score_exact else selected.upper_bound
            if not score_exact and Fraction.from_float(upper) < best_support:
                upper = nextafter(upper, inf)
            return replace(selected,
                option_counts=tuple(int(i == root) for i in range(len(self.options))),
                option_indices=(root,), service=self.requirements[root],
                duration=float(self.durations[root]), score=score, upper_bound=upper,
                gap=max(0., upper - score), exact=False, score_exact=score_exact,
                canonical_exact=False)

        scored = self.score_background_roots(queues, available, now, lane, waiting_ages, paths)
        metadata["planning_events"] += scored.background_planning_events
        metadata["planning_random_draws"] += scored.background_planning_random_draws
        metadata["planning_rate_reads"] += scored.background_planning_rate_reads
        work = dict(
            candidate_count=scored.candidate_count,
            rollout_calls=scored.rollout_calls,
            rollout_option_probes=scored.rollout_option_probes,
            root_option_probes=baseline.root_option_probes + scored.root_option_probes,
            rollout_idle_calls=scored.rollout_idle_calls,
            waiting_age_reads=scored.waiting_age_reads, flow_evaluations=scored.flow_evaluations,
            **{field.name: getattr(scored, field.name)
               for field in fields(BackgroundRootScores)
               if field.name not in {base.name for base in fields(AgeRootScores)}})
        if scored.background_rollout_overflow:
            metadata["planning_overflow"] = True
            return finish(replace(baseline,
                selection=selection_for_root(incumbent), changed=incumbent != maxweight_root,
                **work, **metadata))

        permitted = []
        for index in scored.costs:
            incumbent_work["incumbent_membership_checks"] += 1
            if index not in original_permitted:
                continue
            incumbent_work["incumbent_lookahead_risk_comparisons"] += 1
            if scored.squared_costs[index] <= scored.squared_costs[incumbent]:
                permitted.append(index)

        def ordering(index):
            return scored.costs[index], index != incumbent, self.durations[index], index

        unconstrained = min(scored.costs, key=ordering)
        chosen = min(permitted, key=ordering)
        improved = scored.costs[chosen] < scored.costs[incumbent]
        incumbent_work["incumbent_lookahead_ordering_comparisons"] = (
            scored.candidate_count - 1 + len(permitted) - 1)
        incumbent_work["incumbent_lookahead_strict_improvement_comparisons"] = 1
        actual = chosen if improved else incumbent
        return finish(replace(baseline,
            selection=selection_for_root(actual), changed=actual != maxweight_root,
            improved_incumbent=improved,
            incumbent_actual_predicted_squared_flow_s2=float(original.squared_costs[actual] / len(paths)),
            baseline_predicted_completion_s=float(scored.costs[incumbent] / len(paths)),
            chosen_predicted_completion_s=float(scored.costs[actual] / len(paths)),
            baseline_predicted_squared_flow_s2=float(scored.squared_costs[incumbent] / len(paths)),
            chosen_predicted_squared_flow_s2=float(scored.squared_costs[actual] / len(paths)),
            risk_filtered_candidates=scored.candidate_count - len(permitted),
            risk_feasible_candidates=len(permitted), unconstrained_root_index=unconstrained,
            risk_changed_unconstrained_root=actual != unconstrained, **work, **metadata))
