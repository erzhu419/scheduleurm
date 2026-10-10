"""Try a bounded integer certificate, then continue the existing LP frontier."""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Any, Sequence

from algorithm.canonical_count_space_bnb_frame_candidates import CanonicalCountSpaceBnbSelection
from algorithm.integer_capacity_shared_threshold_frame_candidates import IntegerCapacityThresholdSelection
from algorithm.resumable_integer_capacity_frame_candidates import IntegerCapacitySearch
from algorithm.score_face_canonical_completion_frame_candidates import ScoreFaceCompletionWork
from algorithm.strict_score_proof_canonical_completion_frame_candidates import (
    StrictScoreProofCanonicalCompletionSelection, StrictScoreProofWork,
    _complete_canonical, _prove_no_strict_improvement,
)


@dataclass(frozen=True)
class ResumableProbeSelection:
    selection: CanonicalCountSpaceBnbSelection
    initial_lp: IntegerCapacityThresholdSelection
    final_lp: IntegerCapacityThresholdSelection
    probe: StrictScoreProofCanonicalCompletionSelection | None
    resumed: bool
    initial_lp_budget: int
    lp_node_budget: int
    integer_state_budget: int
    status: str


def _probe(options, queue, lanes, stage, budget, idle_duration):
    """Use the paid LP snapshot; the integer helpers perform no LP calls."""
    baseline = stage.selection
    certified = baseline
    if baseline.score_exact:
        proof = StrictScoreProofWork("SKIPPED_SCORE_CERTIFIED", False, True, budget)
    else:
        proof = _prove_no_strict_improvement(options, queue, lanes, baseline, budget, idle_duration)
        if proof.complete:
            certified = replace(baseline, score_exact=True, upper_bound=baseline.score,
                                gap=0.0, epsilon=0.0)
    if certified.score_exact:
        selected, completion = _complete_canonical(
            options, queue, lanes, certified, budget - proof.states_visited, idle_duration)
    else:
        selected = baseline
        completion = ScoreFaceCompletionWork(
            "SKIPPED_SCORE_UNPROVEN", False, False, budget - proof.states_visited)
    return StrictScoreProofCanonicalCompletionSelection(
        selected, baseline, certified, stage.preprocessing, stage.capacity_propagation,
        proof, completion, budget)


def select_resumable_probe_frame(
    options: Sequence[Any], queues: Sequence[int], lanes: int,
    initial_lp_budget: int, lp_node_budget: int, integer_state_budget: int,
    idle_duration: float = 1.0,
) -> ResumableProbeSelection:
    """Charge one cumulative LP search and at most one shared-budget probe.

    Integer exhaustion or a better-action witness returns control to the same
    LP frontier. No LP prefix or integer search is restarted. The final LP
    snapshot contains all paid LP work, including the initial snapshot.
    """
    for value, name in ((initial_lp_budget, "initial_lp_budget"),
                        (lp_node_budget, "lp_node_budget"),
                        (integer_state_budget, "integer_state_budget")):
        if int(value) != value or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    initial_budget, total_budget, probe_budget = map(
        int, (initial_lp_budget, lp_node_budget, integer_state_budget))
    if total_budget < initial_budget:
        raise ValueError("total LP budget must cover the initial LP budget")
    options, queue = tuple(options), tuple(queues)
    search = IntegerCapacitySearch(options, queue, lanes, idle_duration=idle_duration)
    initial = search.advance_to(initial_budget)

    def finish(selected, final_lp, probe, resumed, status):
        return ResumableProbeSelection(selected, initial, final_lp, probe, resumed,
                                      initial_budget, total_budget, probe_budget, status)

    if initial.selection.canonical_exact:
        return finish(initial.selection, initial, None, False, "LP_INITIAL_COMPLETE")
    probe = _probe(options, queue, lanes, initial, probe_budget, idle_duration)
    if probe.selection.canonical_exact:
        return finish(probe.selection, initial, probe, False, "INTEGER_PROBE_COMPLETE")

    final_lp = search.advance_to(total_budget)
    selected = final_lp.selection
    # A completed integer score proof remains valid when the LP frontier is
    # resumed. Its certificate is separate from the untouched raw LP snapshot.
    if probe.selection.score_exact:
        score = lambda action: Fraction(sum(q * s for q, s in
            zip(queue, action.service, strict=True))) / Fraction.from_float(action.duration)
        if score(selected) != score(probe.selection):
            raise ArithmeticError("resumed LP action contradicts the proved optimum")
        if not selected.score_exact:
            selected = replace(selected, score_exact=True, upper_bound=selected.score,
                               gap=0.0, epsilon=0.0)
    return finish(selected, final_lp, probe, True,
                  "LP_RESUME_COMPLETE" if selected.canonical_exact else "LP_RESUME_INCOMPLETE")
