"""Freeze cohort-visible roots and cross their first action with free continuation."""
from collections import Counter, deque

from algorithm.arrival_aware_lane_rollout import sample_planning_arrivals
from algorithm.background_competition_continuous_lane_rollout import (
    BackgroundCompetitionContinuousLaneRollout,
)
from algorithm.experiments.bg_incumbent_counterfactual_replay import (
    BGIncumbentCounterfactualSelector, FEES, STATUS,
    collect_work as incumbent_collect_work, restore_planner_state,
)
from algorithm.two_step_bg_incumbent_background_redecision_lane_rollout import TWO_STEP_WORK
from algorithm.two_step_bg_incumbent_original_risk_lane_rollout import (
    TwoStepBGIncumbentOriginalRiskLaneRollout,
)


BG_FEES = tuple(key for key in FEES if not key.startswith(("incumbent_", "recourse_")))
BG_STATUS = tuple(key for key in STATUS
                  if not key.startswith("incumbent_") and key != "improved_incumbent")


def make_current_planner(model, tape, protocol):
    return TwoStepBGIncumbentOriginalRiskLaneRollout(model.options,
        f"arrival-plan-v1:{model.node}:{tape['load']}:online",
        protocol["planning_samples"], protocol["planning_event_cap"])


def _causal_frames(factual):
    """Walk cached FIFO admissions using canonical births, before each action."""
    jobs = factual["jobs"]
    dimension = len(factual["frames"][0]["queue"])
    waiting, counts, cursor = [deque() for _ in range(dimension)], [0] * dimension, 0
    for index, frame in enumerate(factual["frames"]):
        now = frame["start"]
        previous_cursor = cursor
        while cursor < len(jobs) and jobs[cursor]["arrival"] <= now:
            coordinate = jobs[cursor]["coordinate"]
            waiting[coordinate].append(cursor)
            counts[coordinate] += 1
            cursor += 1
        ages = tuple(tuple(now - jobs[job_index]["arrival"] for job_index in group)
                     for group in waiting)
        cohort_ids = [(jobs[job_index]["coordinate"], jobs[job_index]["index"])
                      for group in waiting for job_index in group
                      if 300 <= jobs[job_index]["arrival"] <= 900]
        yield index, frame, tuple(counts), ages, cohort_ids, cursor - previous_cursor
        for coordinate, admitted in enumerate(frame["service"]):
            for _ in range(admitted):
                waiting[coordinate].popleft()


def registered_anchors(tapes, factual):
    """Pick the first complete strict, cohort-visible root in [300,900)."""
    anchors, missing = [], []
    for tape in tapes:
        for index, frame, _, _, cohort_ids, _ in _causal_frames(factual[tape["tape_id"]]):
            info = frame["rollout"]
            root = frame["option_indices"][0]
            if (300 <= frame["start"] < 900 and cohort_ids
                    and info["improved_incumbent"] and not info["planning_overflow"]
                    and root != info["incumbent_root_index"]):
                anchors.append({"tape_id": tape["tape_id"], "frame_index": index,
                    "decision_counter": index, "start": frame["start"],
                    "factual_option_index": root,
                    "replacement_option_index": info["incumbent_root_index"],
                    "maxweight_option_index": info["incumbent_maxweight_root_index"],
                    "visible_cohort_jobs": len(cohort_ids),
                    "visible_cohort_job_ids": cohort_ids})
                break
        else:
            missing.append(tape["tape_id"])
    return {"anchors": anchors, "missing_tape_ids": missing}


def reconstruct_prefix_states(planner, factual, anchor_indices, work):
    """Recreate prefix randomness and verify all cached causal FIFO contexts."""
    wanted, states = set(anchor_indices), {}
    for index, frame, counts, ages, cohort_ids, released in _causal_frames(factual):
        info, now = frame["rollout"], frame["start"]
        queue = tuple(map(len, ages))
        rates = tuple(count / now if now > 0 else 0. for count in counts)
        work["cached_frame_reads"] += 1
        work["prefix_fifo_job_birth_reads"] += released
        work["prefix_arrival_counter_updates"] += released
        work["prefix_fifo_context_job_reads"] += sum(queue)
        work["prefix_cohort_birth_reads"] += sum(queue)
        work["prefix_fifo_service_coordinate_reads"] += len(queue)
        work["prefix_fifo_admission_job_reads"] += sum(frame["service"])
        work["rate_estimation_coordinate_reads"] += len(counts)
        work["rate_estimation_divisions"] += len(counts) if now > 0 else 0
        if (planner.decision_counter != index
                or tuple(frame["queue"]) != queue
                or tuple(tuple(group) for group in frame["waiting_ages"]) != ages
                or tuple(frame["observed_arrival_counts"]) != counts
                or tuple(info["observed_arrival_counts"]) != counts
                or tuple(info["estimated_arrival_rates"]) != rates
                or info["eligible"] != (0 < sum(queue) <= info["queue_limit"])):
            raise ValueError("cached causal FIFO/rates/counts/counter differ at frame " + str(index))
        planner._outer_observed_counts, planner.arrival_rates = counts, rates
        if index in wanted:
            state = {"rng_state": planner.rng.getstate(), "decision_counter": planner.decision_counter}
            states[index] = state
            work["checkpoint_rng_words_copied"] += len(state["rng_state"][1])
            work["checkpoint_decision_counter_reads"] += 1
        paths = []
        if info["eligible"]:
            for _ in range(planner.sample_count):
                paths.append(sample_planning_arrivals(
                    rates, planner.maximum_duration, planner.rng, planner.event_cap))
                if paths[-1].overflow:
                    break
        events = sum(len(path.events) for path in paths)
        draws = sum(path.random_draws for path in paths)
        reads = sum(path.rate_reads for path in paths)
        overflow = any(path.overflow for path in paths)
        if (len(paths) != info["sample_count"] or events != info["tagged_planning_events"]
                or draws != info["planning_random_draws"] - info["background_planning_random_draws"]
                or reads != info["planning_rate_reads"] - info["background_planning_rate_reads"]
                or (overflow and (not info["planning_overflow"] or info["incumbent_candidate_count"]))):
            raise ValueError("cached outer prefix population/work differ at frame " + str(index))
        work["prefix_planning_events"] += events
        work["prefix_planning_random_draws"] += draws
        work["prefix_planning_rate_reads"] += reads
        work["prefix_sample_calls"] += len(paths)
        work["prefix_overflow_frames"] += overflow
        planner.decision_counter += 1
        work["replayed_decision_counter_reads"] += 1
        work["replayed_decision_counter_writes"] += 1
    if states.keys() != wanted:
        raise ValueError("every requested anchor state must be captured")
    return states


class TwoStepContinuationCrossSelector(BGIncumbentCounterfactualSelector):
    """Consume the current proposal first, then freely run the chosen suffix.

    First-action candidate/feasibility metadata belongs to the native current
    proposal. A BG swap uses its already scored baseline values. Ordinary BG
    begins only after that complete first call and receives its post-proposal
    outer RNG/counter state; its later decisions retain native BG metadata.
    """
    def __init__(self, planner, first_root="original", suffix="current", capture=()):
        if first_root not in ("original", "BG") or suffix not in ("current", "BG"):
            raise ValueError("cross arms require original/BG first root and current/BG suffix")
        super().__init__(planner, capture=capture, force_first=first_root == "BG")
        self.first_root, self.suffix = first_root, suffix
        self.BG_planner = (BackgroundCompetitionContinuousLaneRollout(
            planner.options, planner.planning_seed, planner.sample_count, planner.event_cap)
            if suffix == "BG" else None)
        self.active_planner = planner

    def __call__(self, *context):
        if self.calls == 0 or self.suffix == "current":
            selected = super().__call__(*context)
            self.work.update({key: getattr(selected, key) for key in TWO_STEP_WORK})
            self.status["current_proposal_calls"] += 1
            if self.calls == 1:
                if (not self.first_proposal["improved_incumbent"]
                        or self.first_proposal["replacement"] is None
                        or self.first_proposal["original"] == self.first_proposal["replacement"]):
                    raise ValueError("cross anchor must be a strict native BG-incumbent improvement")
                applied, = selected.selection.option_indices
                self.first_proposal["metadata_scope"] = "native_current_two_step_original_risk_proposal"
                self.intervention = {**(self.intervention or {}),
                    "original": self.first_proposal["original"],
                    "replacement": applied, "actual_applied_option_index": applied,
                    "first_root": self.first_root, "suffix": self.suffix,
                    "forced_single_action": self.first_root == "BG",
                    "retained_feasibility_metadata_scope": "native_current_two_step_original_risk_proposal"}
                if self.BG_planner is not None:
                    state = self.planner.rng.getstate()
                    self.BG_planner.rng.setstate(state)
                    self.BG_planner.decision_counter = self.planner.decision_counter
                    self.work["suffix_rng_words_copied"] += len(state[1])
                    self.work["suffix_rng_words_restored"] += len(state[1])
                    self.work["suffix_decision_counter_reads"] += 1
                    self.work["suffix_decision_counter_writes"] += 1
                    self.active_planner = self.BG_planner
            return selected

        if self.calls in self.capture:
            state = {"rng_state": self.active_planner.rng.getstate(),
                     "decision_counter": self.active_planner.decision_counter}
            self.states[self.calls] = state
            self.work["checkpoint_rng_words_copied"] += len(state["rng_state"][1])
            self.work["checkpoint_decision_counter_reads"] += 1
        selected = self.active_planner(*context)
        self.work.update({key: getattr(selected, key) for key in BG_FEES})
        if selected.eligible and not selected.planning_overflow:
            self.work["BG_suffix_risk_comparisons"] += selected.candidate_count
            self.work["BG_suffix_ordering_comparisons"] += (
                selected.candidate_count - 1 + selected.risk_feasible_candidates - 1)
            self.work["BG_suffix_strict_improvement_comparisons"] += 1
        self.status["proposal_calls"] += 1
        self.status["BG_suffix_proposal_calls"] += 1
        self.status.update({"proposal_" + key: getattr(selected, key) for key in BG_STATUS})
        self.status["applied_calls"] += 1
        self.status.update({"applied_" + key: getattr(selected, key) for key in BG_STATUS})
        self.calls += 1
        return selected


def collect_work(selector, result):
    """Count each constructed planner and engine context once, without aliases."""
    work = incumbent_collect_work(selector, result)
    if selector.BG_planner is not None:
        work["controller_constructors"] += 1
        work.update({"preprocessing_BG_suffix_" + key: value
                     for key, value in selector.BG_planner.preprocessing_work.items()})
    return work
