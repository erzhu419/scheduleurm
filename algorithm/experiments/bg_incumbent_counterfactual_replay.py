"""Replay one improved root as its same-call BG incumbent, then freely replan."""
from collections import Counter
from dataclasses import asdict, replace
from fractions import Fraction
from math import inf, nextafter

from algorithm.bg_incumbent_background_redecision_lane_rollout import (
    BGIncumbentBackgroundRedecisionLaneRollout, INCUMBENT_WORK,
)
from algorithm.one_step_background_redecision_lane_rollout import RECOURSE_WORK


FEES = (
    "root_option_probes", "rollout_calls", "rollout_option_probes", "planning_events",
    "planning_random_draws", "planning_rate_reads", "rollout_idle_calls", "waiting_age_reads",
    "flow_evaluations", "rate_estimation_coordinate_reads", "rate_estimation_divisions",
    "tagged_planning_events", "background_planning_events", "background_planning_random_draws",
    "background_planning_rate_reads", "background_seed_initializations", "admitted_job_reads",
    "tag_flag_reads", "background_admissions",
) + INCUMBENT_WORK + RECOURSE_WORK
STATUS = (
    "eligible", "changed", "planning_overflow", "background_rollout_overflow",
    "risk_changed_unconstrained_root", "candidate_count", "risk_filtered_candidates",
    "risk_feasible_candidates", "incumbent_changed", "improved_incumbent",
    "incumbent_background_rollout_overflow", "incumbent_candidate_count",
    "incumbent_risk_feasible_candidates", "incumbent_risk_filtered_candidates",
)


def make_planner(model, tape, protocol):
    return BGIncumbentBackgroundRedecisionLaneRollout(model.options,
        f"arrival-plan-v1:{model.node}:{tape['load']}:online",
        protocol["planning_samples"], protocol["planning_event_cap"])


def registered_targets(tapes, factual):
    """Include every cached strict incumbent improvement, at every call time."""
    targets = []
    for tape in tapes:
        for index, frame in enumerate(factual[tape["tape_id"]]["frames"]):
            info = frame["rollout"]
            if info["improved_incumbent"]:
                targets.append({"tape_id": tape["tape_id"], "frame_index": index,
                    "start": frame["start"], "factual_option_index": frame["option_indices"][0],
                    "replacement_option_index": info["incumbent_root_index"],
                    "maxweight_option_index": info["incumbent_maxweight_root_index"]})
    return targets


def restore_planner_state(planner, state):
    """Restore persistent online randomness; call-local context is recomputed."""
    planner.rng.setstate(state["rng_state"])
    planner.decision_counter = state["decision_counter"]
    return {"checkpoint_rng_words_restored": len(state["rng_state"][1]),
            "checkpoint_decision_counter_writes": 1}


class BGIncumbentCounterfactualSelector:
    """Consume the full factual proposal before an optional first BG swap.

    Captures contain only the outer prefix RNG and decision counter. The next
    call overwrites causal counts/rates, and every virtual recourse restores
    the reusable inner BG state, so these are the persistent replay states.
    Proposal work/status remain separate from applied action status. The
    extra MW certificate scan has its own override_root_option_probes column;
    it is not also added to the proposal's root_option_probes column.
    """
    def __init__(self, planner, capture=(), force_first=False):
        self.planner, self.capture, self.force_first = planner, set(capture), force_first
        self.calls, self.states, self.work, self.status = 0, {}, Counter(), Counter()
        self.first_proposal = self.intervention = None

    def __call__(self, *context):
        if self.calls in self.capture:
            state = {"rng_state": self.planner.rng.getstate(),
                     "decision_counter": self.planner.decision_counter}
            self.states[self.calls] = state
            self.work["checkpoint_rng_words_copied"] += len(state["rng_state"][1])
            self.work["checkpoint_decision_counter_reads"] += 1
        proposed = self.planner(*context)
        self.work.update({key: getattr(proposed, key) for key in FEES})
        self.status["proposal_calls"] += 1
        self.status.update({"proposal_" + key: getattr(proposed, key) for key in STATUS})
        selected = proposed
        if self.calls == 0:
            before, after = proposed.baseline_predicted_completion_s, proposed.chosen_predicted_completion_s
            rollout = asdict(proposed)
            del rollout["selection"]
            self.first_proposal = {
                "original": proposed.selection.option_indices[0],
                "replacement": proposed.incumbent_root_index,
                "changed": proposed.changed, "improved_incumbent": proposed.improved_incumbent,
                "predicted_S1_advantage": before - after if before is not None and after is not None else None,
                "rollout": rollout,
            }
            if self.force_first:
                original, replacement = self.first_proposal["original"], self.first_proposal["replacement"]
                if not proposed.improved_incumbent or replacement is None or original == replacement:
                    raise ValueError("forced target must be a strict factual BG-incumbent improvement")
                control = self.planner.maxweight(*context)
                self.work["override_root_option_probes"] += control.root_option_probes
                self.work["override_calls"] += 1
                maxweight_root, = control.selection.option_indices
                selection = control.selection
                if replacement != maxweight_root:
                    queues = context[1]
                    support = Fraction(sum(q * d for q, d in zip(
                        queues, self.planner.requirements[replacement], strict=True))) / self.planner.durations[replacement]
                    best_support = Fraction(sum(q * d for q, d in zip(
                        queues, selection.service, strict=True))) / self.planner.durations[maxweight_root]
                    score, score_exact = float(support), support == best_support
                    upper = score if score_exact else selection.upper_bound
                    if not score_exact and Fraction.from_float(upper) < best_support:
                        upper = nextafter(upper, inf)
                    selection = replace(selection,
                        option_counts=tuple(int(i == replacement) for i in range(len(self.planner.options))),
                        option_indices=(replacement,), service=self.planner.requirements[replacement],
                        duration=float(self.planner.durations[replacement]), score=score, upper_bound=upper,
                        gap=max(0., upper - score), exact=False, score_exact=score_exact, canonical_exact=False)
                selected = replace(proposed, selection=selection,
                    changed=replacement != maxweight_root, improved_incumbent=False,
                    chosen_predicted_completion_s=proposed.baseline_predicted_completion_s,
                    chosen_predicted_squared_flow_s2=proposed.baseline_predicted_squared_flow_s2,
                    incumbent_actual_predicted_squared_flow_s2=proposed.incumbent_predicted_squared_flow_s2,
                    risk_changed_unconstrained_root=replacement != proposed.unconstrained_root_index)
                self.intervention = {"original": original, "replacement": replacement,
                    "maxweight_option_index": maxweight_root, "actual_applied_option_index": replacement,
                    "predicted_S1_advantage": self.first_proposal["predicted_S1_advantage"],
                    "override_root_option_probes": control.root_option_probes}
        self.status["applied_calls"] += 1
        self.status.update({"applied_" + key: getattr(selected, key) for key in STATUS})
        self.calls += 1
        return selected


def collect_work(selector, result):
    """Add constructors and engine context work once, without selector aliases."""
    work = Counter(selector.work)
    work["controller_constructors"] += 1
    work["recourse_controller_constructors"] += 1
    work.update({"preprocessing_" + key: value for key, value in selector.planner.preprocessing_work.items()})
    for key in ("age_context_job_reads", "observed_arrival_counter_updates",
                "observed_arrival_context_coordinate_reads"):
        work[key] += result["rollout_work"][key]
    return work
