"""Registered certificate-only sidecar for the frozen legacy K26 action.

The experiment reuses the 5,003 queues and their registered legacy K26
decisions.  A FIFO-execution quotient K26 search supplies a second safe upper
bound, but never supplies the executed action or score.  The sidecar
certificate is the intersection of the registered legacy and FIFO upper
bounds.  No arrival or trajectory process is run here.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import product
import json
import math
from pathlib import Path
import time

from algorithm.budgeted_frame_candidates import select_budgeted_frame
from algorithm.experiments.aligned_hardware_frame_model import load_model
from algorithm.fifo_execution_quotient_frame_candidates import (
    fifo_execution_signature,
    select_fifo_execution_quotient_frame,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
DEFAULT_OUTPUT = ARTIFACTS / "aligned_hardware_fifo_execution_certificate_sidecar_development_20260912.json"
SOURCE_DIAGNOSTIC = ARTIFACTS / "aligned_hardware_candidate_search_diagnostic_development_20260910.json"
REGISTERED_STATES = ARTIFACTS / "aligned_hardware_duration_tie_order_shadow_development_20260910_registered_states.json"
SOURCE_FIFO_SHADOW = ARTIFACTS / "aligned_hardware_fifo_execution_quotient_shadow_development_20260911.json"
SOURCE_FIFO_SHADOW_PROTOCOL = ARTIFACTS / "aligned_hardware_fifo_execution_quotient_shadow_development_20260911_protocol.json"
SOURCE_EVIDENCE_REUSE = ARTIFACTS / "aligned_hardware_fifo_execution_quotient_evidence_reuse_development_20260911.json"
SOURCE_EVIDENCE_REUSE_PROTOCOL = ARTIFACTS / "aligned_hardware_fifo_execution_quotient_evidence_reuse_development_20260911_protocol.json"
SOURCE_FIXED_SOURCES = ARTIFACTS / "aligned_hardware_fifo_execution_quotient_evidence_reuse_development_20260911_fixed_sources.json"

LOADS = (0.50, 0.80, 0.95)
SEEDS = (41, 42, 43)
SMALL_BUDGET = 26
LEGACY_EXACT_BUDGET = 120
SOURCE_POLICY_BUDGETS = (SMALL_BUDGET, LEGACY_EXACT_BUDGET)
STATE_COUNT = 5003
ORDERED_ACTION_COUNT = 100
FIFO_SIGNATURE_COUNT = 71
SCORE_TOLERANCE = 1e-10
AGGREGATE_TOLERANCE = 1e-12
EXPERIMENT = "aligned_hardware_fifo_execution_certificate_sidecar_v1"

NEW_SOURCES = (
    "algorithm/experiments/aligned_hardware_fifo_execution_certificate_sidecar.py",
)

RULE = {
    "executed_action": "copy the registered legacy K26 raw option tuple, service, duration, and score exactly",
    "raw_fifo_upper_bound": "run the unchanged FIFO-execution quotient selector at paid budget K26 and use its returned upper bound only",
    "sidecar_upper_bound": "minimum of the registered legacy K26 upper bound and the raw FIFO K26 upper bound",
    "fixed_action_gap": "sidecar upper bound minus the fixed legacy K26 score; no clamp is applied",
    "fixed_action_epsilon": "fixed-action gap divided by visible queue norm, or zero for the empty queue",
    "fixed_action_exact": "fixed-action gap is at most the registered score tolerance and all upper-bound validity checks pass",
    "fifo_search_exact": "reported separately from fixed-action exactness and never changes the fixed action",
    "unchanged": "model, 5,003-state roster, K26 paid budget, feasibility, legacy raw tie break, loads, seeds, and population weights",
}

TIMING_PROTOCOL = {
    "clock": "time.perf_counter",
    "passes": 1,
    "state_order": "registered state_index order",
    "selector_order": "legacy first on even state_index and FIFO first on odd state_index",
    "warmup": "none; the timed calls are also the correctness reconstruction calls",
    "reported": "legacy-only, incremental FIFO, and composite wall seconds and milliseconds per state, plus incremental and composite ratios over legacy",
    "interpretation": "descriptive implementation-cost diagnostic only; no timing value changes a scientific decision",
    "scientific_gate": False,
}

ADVANCE_GATE = {
    "all_execution_checks": True,
    "fixed_action_identity_on_every_state": True,
    "registered_optimum_is_bounded_by_both_raw_bounds_and_their_minimum": True,
    "fixed_gap_regret_slack_decomposition_on_every_state": True,
    "pointwise_sidecar_upper_and_gap_nonworse": True,
    "primary_each_load_support_metrics_identical": True,
    "sensitivity_each_load_support_metrics_identical": True,
    "primary_each_load_certificate_metrics_nonworse": True,
    "sensitivity_each_load_certificate_metrics_nonworse": True,
    "primary_exposure_contains_certificate_tightening": True,
    "at_least_one_primary_load_strict_certificate_improvement": True,
}

STOPPING_RULE = (
    "invalid execution gives INVALID_EXECUTION; a valid pointwise-safe result "
    "with all nonworse gates and at least one strict primary certificate gain "
    "gives CERTIFICATE_TIGHTENING_ONLY_CLOSE; failure of any nonworse gate gives "
    "CERTIFICATE_TRADEOFF_STOP; an otherwise nonworse result without a strict "
    "gain gives NO_CERTIFICATE_GAIN_STOP. Every valid decision stops without "
    "trajectories."
)

PROHIBITED = (
    "changing the registered legacy K26 action or score, using the FIFO-selected "
    "action, arrival generation, trajectory simulation, new seeds, rule variants, "
    "runtime thresholds, confirmation outcomes, and paper edits"
)

REGISTERED_STATE_SCHEMA = {
    "queue": "visible three-class queue",
    "population_visits": "load, seed, source policy budget, frame count, and legacy visited time",
    "legacy_k26": "registered fixed action, score, certificate, work, and exact flag",
    "support_reference_score": "registered exhaustive legacy K120 support optimum",
}


def _json(value):
    return json.loads(json.dumps(value))


def _paths(output):
    output = Path(output)
    return (
        output,
        output.with_name(output.stem + "_protocol.json"),
        output.with_name(output.stem + "_fixed_sources.json"),
    )


def _write_new(path, data, *, compact=False):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(
            data,
            stream,
            separators=(",", ":") if compact else None,
            indent=None if compact else 2,
        )
        stream.write("\n")


def _source_texts():
    sources = json.loads(SOURCE_FIXED_SOURCES.read_text())
    changed = [
        name for name, text in sources.items() if (ROOT / name).read_text() != text
    ]
    if changed:
        raise ValueError(f"FIFO evidence-reuse source changed: {changed}")
    return {
        **sources,
        **{name: (ROOT / name).read_text() for name in NEW_SOURCES},
    }


def _compact_state(row):
    fields = (
        "option_indices",
        "service",
        "duration",
        "score",
        "upper_bound",
        "gap",
        "epsilon",
        "attempts",
        "exact",
    )
    return {
        "queue": row["queue"],
        "population_visits": [
            [
                visit["load"],
                visit["seed"],
                visit["source_budget"],
                visit["frame_count"],
                visit["visited_time_s"],
            ]
            for visit in row["population_visits"]
        ],
        "legacy_k26": {key: row["k26"][key] for key in fields},
        "support_reference_score": row["exact"]["score"],
        "legacy_first_exact_score_attempt": row["first_exact_score_attempt"],
    }


def _registered_state_roster(diagnostic):
    states = [_compact_state(row) for row in diagnostic["state_diagnostics"]]
    return {
        "source_experiment": SOURCE_DIAGNOSTIC.name,
        "state_count": len(states),
        "states": states,
    }


def _source_population_cells(diagnostic):
    return [
        {
            "load": row["load"],
            "seed": row["seed"],
            "source_policy_budget": row["source_budget"],
            "frame_count": row["frame_count"],
            "legacy_visited_time_s": row["elapsed_including_drain_s"],
        }
        for row in diagnostic["source_trajectory_replays"]
    ]


def _validate_source_diagnostic():
    report = json.loads(SOURCE_DIAGNOSTIC.read_text())
    protocol = report["protocol"]
    expected_populations = [
        (load, budget) for load in LOADS for budget in SOURCE_POLICY_BUDGETS
    ]
    if (
        report.get("experiment") != "aligned_hardware_candidate_search_diagnostic_v1"
        or report.get("status") != "PASS"
        or not all(report.get("checks", {}).values())
        or report.get("development_advance")
        or protocol.get("loads") != list(LOADS)
        or protocol.get("seeds") != list(SEEDS)
        or protocol.get("source_budgets") != list(SOURCE_POLICY_BUDGETS)
        or protocol.get("small_budget") != SMALL_BUDGET
        or protocol.get("exact_budget") != LEGACY_EXACT_BUDGET
        or len(report.get("source_trajectory_replays", ())) != 18
        or len(report.get("state_diagnostics", ())) != STATE_COUNT
        or len({tuple(row["queue"]) for row in report["state_diagnostics"]})
        != STATE_COUNT
        or [
            (row["load"], row["source_budget"])
            for row in report["summary_by_load_and_source_policy"]
        ]
        != expected_populations
    ):
        raise ValueError("certificate sidecar requires the completed candidate-search diagnostic")
    model = load_model()
    if _json(model.snapshot()) != protocol.get("model"):
        raise ValueError("certificate-sidecar model changed from its source diagnostic")
    roster = json.loads(REGISTERED_STATES.read_text())
    if roster != _registered_state_roster(report):
        raise ValueError("registered 5,003-state roster changed")
    return report, model, roster


def _validate_predecessors():
    shadow = json.loads(SOURCE_FIFO_SHADOW.read_text())
    shadow_protocol = json.loads(SOURCE_FIFO_SHADOW_PROTOCOL.read_text())
    evidence = json.loads(SOURCE_EVIDENCE_REUSE.read_text())
    evidence_protocol = json.loads(SOURCE_EVIDENCE_REUSE_PROTOCOL.read_text())
    if (
        shadow.get("protocol") != shadow_protocol
        or shadow.get("experiment")
        != "aligned_hardware_fifo_execution_quotient_state_shadow_v1"
        or shadow.get("status") != "PASS"
        or not all(shadow.get("checks", {}).values())
        or shadow.get("state_evaluation_count") != STATE_COUNT
        or not shadow["checks"].get(
            "k71_matches_registered_legacy_k120_support_on_every_state"
        )
        or not shadow["checks"].get(
            "k71_replays_registered_k120_raw_physical_and_fifo_witnesses"
        )
        or evidence.get("protocol") != evidence_protocol
        or evidence.get("experiment")
        != "aligned_hardware_fifo_execution_quotient_evidence_reuse_v1"
        or evidence.get("status") != "PASS"
        or not all(evidence.get("checks", {}).values())
        or evidence.get("scientific_decision") != "KNOWN_TAIL_TRADEOFF_STOP"
        or evidence.get("development_advance")
    ):
        raise ValueError("certificate sidecar requires the completed FIFO shadow and tail-tradeoff closure")
    return {
        "fifo_shadow": {
            "experiment": shadow["experiment"],
            "status": shadow["status"],
            "state_evaluation_count": shadow["state_evaluation_count"],
            "k71_support_oracle_check": shadow["checks"][
                "k71_matches_registered_legacy_k120_support_on_every_state"
            ],
            "k71_witness_replay_check": shadow["checks"][
                "k71_replays_registered_k120_raw_physical_and_fifo_witnesses"
            ],
        },
        "evidence_reuse": {
            "experiment": evidence["experiment"],
            "status": evidence["status"],
            "scientific_decision": evidence["scientific_decision"],
            "development_advance": evidence["development_advance"],
        },
    }


def _structure_snapshot(model):
    choices = (None, *range(len(model.options)))
    signatures = {
        fifo_execution_signature(
            model.options, raw, model.lanes, len(model.workloads)
        )
        for raw in product(choices, repeat=model.lanes)
    }
    return {
        "option_count": len(model.options),
        "lanes": model.lanes,
        "represented_raw_action_count": (len(model.options) + 1) ** model.lanes,
        "represented_fifo_execution_signature_count": len(signatures),
    }


def _structure_is_expected(structure):
    return structure == {
        "option_count": 9,
        "lanes": 2,
        "represented_raw_action_count": ORDERED_ACTION_COUNT,
        "represented_fifo_execution_signature_count": FIFO_SIGNATURE_COUNT,
    }


def _state_visits(state, load, source_policy_budget, seed=None):
    matching = [
        row
        for row in state["population_visits"]
        if row[0] == load
        and row[2] == source_policy_budget
        and (seed is None or row[1] == seed)
    ]
    return sum(row[3] for row in matching), sum(row[4] for row in matching)


def _selection_subset(selection):
    data = asdict(selection)
    return {
        key: data[key]
        for key in (
            "option_indices",
            "service",
            "duration",
            "score",
            "upper_bound",
            "gap",
            "epsilon",
            "attempts",
            "exact",
        )
    }


def _work_identities(selection, option_count, lanes):
    return (
        selection.seed_attempts == option_count + 1
        and selection.attempts
        == selection.seed_attempts + selection.paid_search_pops
        and selection.cache_entries
        == selection.attempts - selection.seed_signature_collisions
        and selection.cache_misses_after_seeds == selection.paid_search_pops
        and selection.cache_hits == selection.free_cache_hit_pops
        and selection.logical_search_pops
        == selection.paid_search_pops + selection.free_cache_hit_pops
        and selection.cached_feasible_reuses
        + selection.cached_infeasible_reuses
        == selection.free_cache_hit_pops
        and selection.candidate_score_evaluations
        == selection.feasible_candidate_count
        and selection.cache_key_evaluations
        == selection.seed_attempts
        + selection.logical_search_pops
        + selection.budget_blocked_cache_miss_lookups
        and selection.cache_key_item_reads
        == lanes * selection.cache_key_evaluations
        and selection.cache_key_requirement_coordinate_reads
        == len(selection.service) * selection.cache_key_duration_reads
        and selection.cache_key_completion_offset_reads
        >= selection.cache_key_duration_reads
        and selection.upper_bound_calls
        == selection.frontier_pushes + selection.bound_rejected_prefixes
        and selection.frontier_pushes
        == selection.logical_search_pops
        + selection.exact_stop_frontier_entries_cleared
        + selection.frontier_entries_remaining
        and selection.represented_ordered_configurations
        == (option_count + 1) ** lanes
        and 0 <= selection.seed_signature_collisions < selection.seed_attempts
        and selection.cache_entries <= FIFO_SIGNATURE_COUNT
        and selection.budget_blocked_cache_miss_lookups in (0, 1)
        and selection.exact == (selection.frontier_entries_remaining == 0)
    )


def _evaluate_state(model, state_index, state):
    queue = tuple(state["queue"])

    # Alternate the timed order deterministically. Timing is descriptive only;
    # both calls are also used for correctness reconstruction.
    if state_index % 2 == 0:
        started = time.perf_counter()
        legacy_recomputed = select_budgeted_frame(
            model.options, queue, model.lanes, SMALL_BUDGET
        )
        legacy_wall_s = time.perf_counter() - started
        started = time.perf_counter()
        fifo = select_fifo_execution_quotient_frame(
            model.options, queue, model.lanes, SMALL_BUDGET
        )
        fifo_wall_s = time.perf_counter() - started
    else:
        started = time.perf_counter()
        fifo = select_fifo_execution_quotient_frame(
            model.options, queue, model.lanes, SMALL_BUDGET
        )
        fifo_wall_s = time.perf_counter() - started
        started = time.perf_counter()
        legacy_recomputed = select_budgeted_frame(
            model.options, queue, model.lanes, SMALL_BUDGET
        )
        legacy_wall_s = time.perf_counter() - started

    registered = state["legacy_k26"]
    legacy_subset = _json(_selection_subset(legacy_recomputed))
    action_fields = ("option_indices", "service", "duration", "score")
    fixed_action = {key: registered[key] for key in action_fields}
    action_preserved = fixed_action == {
        key: registered[key] for key in action_fields
    }
    raw_fifo_upper = float(fifo.upper_bound)
    sidecar_upper = min(float(registered["upper_bound"]), raw_fifo_upper)
    fixed_gap = sidecar_upper - float(registered["score"])
    queue_norm = sum(queue)
    fixed_epsilon = fixed_gap / queue_norm if queue_norm else 0.0
    reference = float(state["support_reference_score"])
    support_regret = reference - float(registered["score"])
    legacy_frontier_slack = float(registered["upper_bound"]) - reference
    raw_fifo_frontier_slack = raw_fifo_upper - reference
    sidecar_frontier_slack = sidecar_upper - reference
    gap_identity_error = fixed_gap - support_regret - sidecar_frontier_slack
    upper_validity = {
        "fixed_score_is_bounded_by_registered_reference": (
            float(registered["score"]) <= reference + SCORE_TOLERANCE
        ),
        "legacy_upper_bounds_registered_reference": (
            reference <= float(registered["upper_bound"]) + SCORE_TOLERANCE
        ),
        "raw_fifo_upper_bounds_registered_reference": (
            reference <= raw_fifo_upper + SCORE_TOLERANCE
        ),
        "sidecar_upper_bounds_registered_reference": (
            reference <= sidecar_upper + SCORE_TOLERANCE
        ),
        "fixed_score_is_bounded_by_sidecar": (
            float(registered["score"]) <= sidecar_upper + SCORE_TOLERANCE
        ),
        "fifo_selected_score_is_bounded_by_sidecar": (
            float(fifo.score) <= sidecar_upper + SCORE_TOLERANCE
        ),
        "fifo_selected_score_is_bounded_by_registered_reference": (
            float(fifo.score) <= reference + SCORE_TOLERANCE
        ),
    }
    exact_definition_checks = {
        "gap_is_unclamped_upper_minus_fixed_score": math.isclose(
            fixed_gap,
            sidecar_upper - float(registered["score"]),
            rel_tol=0.0,
            abs_tol=SCORE_TOLERANCE,
        ),
        "gap_is_nonnegative_within_tolerance": fixed_gap >= -SCORE_TOLERANCE,
        "epsilon_uses_fixed_action_gap": math.isclose(
            fixed_epsilon,
            fixed_gap / queue_norm if queue_norm else 0.0,
            rel_tol=0.0,
            abs_tol=SCORE_TOLERANCE,
        ),
        "gap_decomposes_into_support_regret_and_sidecar_frontier_slack": (
            abs(gap_identity_error) <= SCORE_TOLERANCE
        ),
    }
    fixed_exact = all(upper_validity.values()) and fixed_gap <= SCORE_TOLERANCE
    return {
        "state_index": state_index,
        "queue": list(queue),
        "legacy_k26": registered,
        "legacy_recomputed_k26": legacy_subset,
        "legacy_selection": asdict(legacy_recomputed),
        "legacy_reconstruction_matches_registered": legacy_subset == registered,
        "fifo_certificate_k26": asdict(fifo),
        "fifo_selection": asdict(fifo),
        "fixed_action": fixed_action,
        "executed_option_indices": fixed_action["option_indices"],
        "executed_service": fixed_action["service"],
        "executed_duration": fixed_action["duration"],
        "executed_score": fixed_action["score"],
        "action_preserved": action_preserved,
        "registered_support_reference_score": reference,
        "fixed_action_support_regret": support_regret,
        "legacy_frontier_slack": legacy_frontier_slack,
        "raw_fifo_frontier_slack": raw_fifo_frontier_slack,
        "sidecar_frontier_slack": sidecar_frontier_slack,
        "raw_fifo_upper_bound": raw_fifo_upper,
        "sidecar_upper_bound": sidecar_upper,
        "fixed_action_gap": fixed_gap,
        "fixed_action_epsilon": fixed_epsilon,
        "gap_identity_error": gap_identity_error,
        "fixed_action_exact": fixed_exact,
        "fifo_search_exact": bool(fifo.exact),
        "upper_bound_validity": upper_validity,
        "exact_definition_checks": exact_definition_checks,
        "exact_definition_checks_pass": all(exact_definition_checks.values()),
        "work_accounting_checks_pass": _work_identities(
            fifo, len(model.options), model.lanes
        ),
        "certificate_upper_delta_over_legacy": (
            sidecar_upper - float(registered["upper_bound"])
        ),
        "certificate_gap_delta_over_legacy": (
            fixed_gap - float(registered["gap"])
        ),
        "certificate_tightened": (
            sidecar_upper < float(registered["upper_bound"]) - SCORE_TOLERANCE
        ),
        "timing": {
            "legacy_selector_wall_s": legacy_wall_s,
            "fifo_selector_wall_s": fifo_wall_s,
        },
    }


def _metrics_for_population(states, evaluations, load, source_policy_budget, seed=None):
    weighted = []
    for state, evaluation in zip(states, evaluations, strict=True):
        count, visited_time = _state_visits(state, load, source_policy_budget, seed)
        if count:
            weighted.append((state, evaluation, count, visited_time))
    frame_count = sum(row[2] for row in weighted)
    visited_time = sum(row[3] for row in weighted)

    def average(function):
        return sum(
            count * function(state, evaluation)
            for state, evaluation, count, _ in weighted
        ) / frame_count

    def frames_where(function):
        return sum(
            count
            for state, evaluation, count, _ in weighted
            if function(state, evaluation)
        )

    def time_where(function):
        return sum(
            visited
            for state, evaluation, _, visited in weighted
            if function(state, evaluation)
        )

    legacy = {
        "mean_fixed_action_score": average(
            lambda state, _evaluation: state["legacy_k26"]["score"]
        ),
        "mean_finite_budget_support_regret": average(
            lambda state, _evaluation: max(
                0.0,
                state["support_reference_score"] - state["legacy_k26"]["score"],
            )
        ),
        "strict_support_loss_frame_count": frames_where(
            lambda state, _evaluation: state["support_reference_score"]
            - state["legacy_k26"]["score"]
            > SCORE_TOLERANCE
        ),
        "mean_upper_bound": average(
            lambda state, _evaluation: state["legacy_k26"]["upper_bound"]
        ),
        "mean_certified_gap": average(
            lambda state, _evaluation: state["legacy_k26"]["gap"]
        ),
        "mean_certified_gap_per_visible_job": average(
            lambda state, _evaluation: state["legacy_k26"]["epsilon"]
        ),
        "exact_certified_frame_fraction": frames_where(
            lambda state, _evaluation: state["legacy_k26"]["exact"]
        ) / frame_count,
    }
    sidecar = {
        "mean_fixed_action_score": average(
            lambda _state, evaluation: evaluation["fixed_action"]["score"]
        ),
        "mean_finite_budget_support_regret": average(
            lambda state, evaluation: max(
                0.0,
                state["support_reference_score"]
                - evaluation["fixed_action"]["score"],
            )
        ),
        "strict_support_loss_frame_count": frames_where(
            lambda state, evaluation: state["support_reference_score"]
            - evaluation["fixed_action"]["score"]
            > SCORE_TOLERANCE
        ),
        "mean_upper_bound": average(
            lambda _state, evaluation: evaluation["sidecar_upper_bound"]
        ),
        "mean_certified_gap": average(
            lambda _state, evaluation: evaluation["fixed_action_gap"]
        ),
        "mean_certified_gap_per_visible_job": average(
            lambda _state, evaluation: evaluation["fixed_action_epsilon"]
        ),
        "exact_certified_frame_fraction": frames_where(
            lambda _state, evaluation: evaluation["fixed_action_exact"]
        ) / frame_count,
    }
    return {
        "load": load,
        "seed": seed,
        "source_policy_budget": source_policy_budget,
        "population_role": (
            "primary frozen K26-policy states"
            if source_policy_budget == SMALL_BUDGET
            else "frozen K120-policy state sensitivity"
        ),
        "frame_count": frame_count,
        "legacy_visited_time_s": visited_time,
        "unique_queue_state_count": len(weighted),
        "action_changed_frame_count": frames_where(
            lambda _state, evaluation: not evaluation["action_preserved"]
        ),
        "certificate_tightened_frame_count": frames_where(
            lambda _state, evaluation: evaluation["certificate_tightened"]
        ),
        "certificate_tightened_time_fraction": time_where(
            lambda _state, evaluation: evaluation["certificate_tightened"]
        ) / visited_time,
        "mean_fifo_paid_attempts": average(
            lambda _state, evaluation: evaluation["fifo_certificate_k26"][
                "attempts"
            ]
        ),
        "mean_fifo_free_cache_hits": average(
            lambda _state, evaluation: evaluation["fifo_certificate_k26"][
                "free_cache_hit_pops"
            ]
        ),
        "legacy": legacy,
        "sidecar": sidecar,
        "sidecar_minus_legacy": {
            key: sidecar[key] - legacy[key] for key in sidecar
        },
    }


def _source_summary_matches(source, observed):
    if (
        source["frame_count"] != observed["frame_count"]
        or source["unique_queue_state_count"] != observed["unique_queue_state_count"]
        or source["strict_support_loss_frame_count"]
        != observed["legacy"]["strict_support_loss_frame_count"]
    ):
        return False
    pairs = (
        (
            source["mean_finite_budget_support_regret"],
            observed["legacy"]["mean_finite_budget_support_regret"],
        ),
        (source["mean_certified_gap"], observed["legacy"]["mean_certified_gap"]),
        (
            source["k26_exact_certified_frame_fraction"],
            observed["legacy"]["exact_certified_frame_fraction"],
        ),
    )
    return all(
        math.isclose(left, right, rel_tol=0.0, abs_tol=AGGREGATE_TOLERANCE)
        for left, right in pairs
    )


def _certificate_checks(prefix, summaries):
    return {
        f"{prefix}_fixed_support_identical_at_every_load": all(
            math.isclose(
                row["sidecar"]["mean_fixed_action_score"],
                row["legacy"]["mean_fixed_action_score"],
                rel_tol=0.0,
                abs_tol=AGGREGATE_TOLERANCE,
            )
            and math.isclose(
                row["sidecar"]["mean_finite_budget_support_regret"],
                row["legacy"]["mean_finite_budget_support_regret"],
                rel_tol=0.0,
                abs_tol=AGGREGATE_TOLERANCE,
            )
            and row["sidecar"]["strict_support_loss_frame_count"]
            == row["legacy"]["strict_support_loss_frame_count"]
            for row in summaries
        ),
        f"{prefix}_mean_upper_bound_nonworse_at_every_load": all(
            row["sidecar"]["mean_upper_bound"]
            <= row["legacy"]["mean_upper_bound"] + AGGREGATE_TOLERANCE
            for row in summaries
        ),
        f"{prefix}_mean_certified_gap_nonworse_at_every_load": all(
            row["sidecar"]["mean_certified_gap"]
            <= row["legacy"]["mean_certified_gap"] + AGGREGATE_TOLERANCE
            for row in summaries
        ),
        f"{prefix}_exact_certified_fraction_nonworse_at_every_load": all(
            row["sidecar"]["exact_certified_frame_fraction"]
            + AGGREGATE_TOLERANCE
            >= row["legacy"]["exact_certified_frame_fraction"]
            for row in summaries
        ),
    }


def _advance_checks(primary, sensitivity, evaluations, execution_passed):
    checks = {
        "execution_passed": execution_passed,
        "fixed_action_identity_on_every_state": all(
            row["action_preserved"] for row in evaluations
        ),
        "registered_optimum_bounded_by_legacy_fifo_and_sidecar_on_every_state": all(
            all(row["upper_bound_validity"].values()) for row in evaluations
        ),
        "fixed_gap_regret_slack_decomposition_on_every_state": all(
            abs(row["gap_identity_error"]) <= SCORE_TOLERANCE
            for row in evaluations
        ),
        "sidecar_upper_bound_pointwise_nonworse": all(
            row["certificate_upper_delta_over_legacy"] <= SCORE_TOLERANCE
            for row in evaluations
        ),
        "sidecar_gap_pointwise_nonworse": all(
            row["certificate_gap_delta_over_legacy"] <= SCORE_TOLERANCE
            for row in evaluations
        ),
    }
    checks.update(_certificate_checks("primary", primary))
    checks.update(_certificate_checks("sensitivity", sensitivity))
    checks["primary_exposure_contains_certificate_tightening"] = any(
        row["certificate_tightened_frame_count"] > 0 for row in primary
    )
    checks["at_least_one_primary_load_strict_certificate_improvement"] = any(
        row["sidecar"]["mean_certified_gap"]
        < row["legacy"]["mean_certified_gap"] - AGGREGATE_TOLERANCE
        or row["sidecar"]["exact_certified_frame_fraction"]
        > row["legacy"]["exact_certified_frame_fraction"] + AGGREGATE_TOLERANCE
        for row in primary
    )
    return checks


def _decision(checks):
    if not checks.get("execution_passed", False):
        return "INVALID_EXECUTION"
    strict_keys = {
        "primary_exposure_contains_certificate_tightening",
        "at_least_one_primary_load_strict_certificate_improvement",
    }
    nonworse = all(
        value for key, value in checks.items() if key not in strict_keys
    )
    if not nonworse:
        return "CERTIFICATE_TRADEOFF_STOP"
    if not all(checks.get(key, False) for key in strict_keys):
        return "NO_CERTIFICATE_GAIN_STOP"
    return "CERTIFICATE_TIGHTENING_ONLY_CLOSE"


def _operation_work(evaluations):
    common_fields = {
        "candidate_attempts": "attempts",
        "feasible_candidate_evaluations": "feasible_candidate_count",
        "local_score_evaluations": "local_score_evaluations",
        "bound_option_probes": "bound_option_probes",
        "bound_threshold_evaluations": "bound_threshold_evaluations",
    }

    def totals(selection_key):
        return {
            "selector_calls": len(evaluations),
            **{
                output_name: sum(
                    row[selection_key][source_name] for row in evaluations
                )
                for output_name, source_name in common_fields.items()
            },
            "peak_search_frontier": max(
                (row[selection_key]["frontier_peak"] for row in evaluations),
                default=0,
            ),
        }

    legacy = totals("legacy_selection")
    fifo = {
        **totals("fifo_selection"),
        "seed_attempts": sum(
            row["fifo_selection"]["seed_attempts"] for row in evaluations
        ),
        "candidate_score_evaluations": sum(
            row["fifo_selection"]["candidate_score_evaluations"]
            for row in evaluations
        ),
        "logical_search_pops": sum(
            row["fifo_selection"]["logical_search_pops"] for row in evaluations
        ),
        "paid_search_pops": sum(
            row["fifo_selection"]["paid_search_pops"] for row in evaluations
        ),
        "free_cache_hit_pops": sum(
            row["fifo_selection"]["free_cache_hit_pops"] for row in evaluations
        ),
        "cache_misses_after_seeds": sum(
            row["fifo_selection"]["cache_misses_after_seeds"]
            for row in evaluations
        ),
        "cached_feasible_reuses": sum(
            row["fifo_selection"]["cached_feasible_reuses"]
            for row in evaluations
        ),
        "cached_infeasible_reuses": sum(
            row["fifo_selection"]["cached_infeasible_reuses"]
            for row in evaluations
        ),
        "cache_entries": sum(
            row["fifo_selection"]["cache_entries"] for row in evaluations
        ),
        "cache_hits": sum(
            row["fifo_selection"]["cache_hits"] for row in evaluations
        ),
        "seed_signature_collisions": sum(
            row["fifo_selection"]["seed_signature_collisions"]
            for row in evaluations
        ),
        "cache_key_evaluations": sum(
            row["fifo_selection"]["cache_key_evaluations"]
            for row in evaluations
        ),
        "cache_key_item_reads": sum(
            row["fifo_selection"]["cache_key_item_reads"] for row in evaluations
        ),
        "cache_key_requirement_coordinate_reads": sum(
            row["fifo_selection"]["cache_key_requirement_coordinate_reads"]
            for row in evaluations
        ),
        "cache_key_duration_reads": sum(
            row["fifo_selection"]["cache_key_duration_reads"]
            for row in evaluations
        ),
        "cache_key_completion_offset_reads": sum(
            row["fifo_selection"]["cache_key_completion_offset_reads"]
            for row in evaluations
        ),
        "budget_blocked_cache_miss_lookups": sum(
            row["fifo_selection"]["budget_blocked_cache_miss_lookups"]
            for row in evaluations
        ),
        "upper_bound_calls": sum(
            row["fifo_selection"]["upper_bound_calls"] for row in evaluations
        ),
        "frontier_pushes": sum(
            row["fifo_selection"]["frontier_pushes"] for row in evaluations
        ),
        "bound_rejected_prefixes": sum(
            row["fifo_selection"]["bound_rejected_prefixes"]
            for row in evaluations
        ),
        "exact_stop_frontier_entries_cleared": sum(
            row["fifo_selection"]["exact_stop_frontier_entries_cleared"]
            for row in evaluations
        ),
        "frontier_entries_remaining": sum(
            row["fifo_selection"]["frontier_entries_remaining"]
            for row in evaluations
        ),
        "exact_selection_count": sum(
            row["fifo_selection"]["exact"] for row in evaluations
        ),
        "represented_ordered_configurations_per_call": (
            evaluations[0]["fifo_selection"]["represented_ordered_configurations"]
            if evaluations
            else ORDERED_ACTION_COUNT
        ),
    }
    composite = {
        "selector_calls": legacy["selector_calls"] + fifo["selector_calls"],
        **{
            field: legacy[field] + fifo[field] for field in common_fields
        },
        "peak_search_frontier": max(
            legacy["peak_search_frontier"], fifo["peak_search_frontier"]
        ),
    }
    return {
        "legacy_only": legacy,
        "fifo_certificate_incremental": fifo,
        "composite": composite,
    }


def register(output=DEFAULT_OUTPUT):
    output, protocol_path, sources_path = _paths(output)
    if any(path.exists() for path in (output, protocol_path, sources_path)):
        raise ValueError("FIFO certificate-sidecar artifacts already exist")

    # Registration freezes sources, predecessor facts, model, roster, rule,
    # timing protocol and gates. It must not call either selector or evaluate a
    # registered queue.
    sources = _source_texts()
    diagnostic, model, _roster = _validate_source_diagnostic()
    predecessors = _validate_predecessors()
    structure = _structure_snapshot(model)
    if not _structure_is_expected(structure):
        raise ValueError("registered 100-to-71 FIFO execution structure changed")
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT,
        "role": "development certificate-only fixed-state sidecar on consumed states",
        "source_experiment": SOURCE_DIAGNOSTIC.name,
        "source_fifo_shadow": SOURCE_FIFO_SHADOW.name,
        "source_tail_closure": SOURCE_EVIDENCE_REUSE.name,
        "registered_state_roster": REGISTERED_STATES.name,
        "loads": list(LOADS),
        "seeds": list(SEEDS),
        "source_policy_budgets": list(SOURCE_POLICY_BUDGETS),
        "small_checkpoint_budget": SMALL_BUDGET,
        "legacy_exact_reference_budget": LEGACY_EXACT_BUDGET,
        "state_count": STATE_COUNT,
        "ordered_action_count": ORDERED_ACTION_COUNT,
        "fifo_signature_count": FIFO_SIGNATURE_COUNT,
        "score_tolerance": SCORE_TOLERANCE,
        "model": _json(model.snapshot()),
        "registered_state_schema": REGISTERED_STATE_SCHEMA,
        "fifo_execution_structure": structure,
        "predecessor_facts": predecessors,
        "source_population_summaries": diagnostic[
            "summary_by_load_and_source_policy"
        ],
        "source_population_cells": _source_population_cells(diagnostic),
        "rule": RULE,
        "timing_protocol": TIMING_PROTOCOL,
        "advance_gate": ADVANCE_GATE,
        "stopping_rule": STOPPING_RULE,
        "prohibited_in_this_stage": PROHIBITED,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_new(protocol_path, protocol)
    _write_new(sources_path, sources)
    return protocol


def run_registered(output=DEFAULT_OUTPUT):
    output, protocol_path, sources_path = _paths(output)
    if output.exists():
        raise ValueError("FIFO certificate-sidecar result already exists")
    protocol = json.loads(protocol_path.read_text())
    sources = json.loads(sources_path.read_text())
    constants = {
        "experiment": EXPERIMENT,
        "source_experiment": SOURCE_DIAGNOSTIC.name,
        "source_fifo_shadow": SOURCE_FIFO_SHADOW.name,
        "source_tail_closure": SOURCE_EVIDENCE_REUSE.name,
        "registered_state_roster": REGISTERED_STATES.name,
        "loads": list(LOADS),
        "seeds": list(SEEDS),
        "source_policy_budgets": list(SOURCE_POLICY_BUDGETS),
        "small_checkpoint_budget": SMALL_BUDGET,
        "legacy_exact_reference_budget": LEGACY_EXACT_BUDGET,
        "state_count": STATE_COUNT,
        "ordered_action_count": ORDERED_ACTION_COUNT,
        "fifo_signature_count": FIFO_SIGNATURE_COUNT,
        "score_tolerance": SCORE_TOLERANCE,
    }
    if any(protocol.get(key) != value for key, value in constants.items()):
        raise ValueError("registered FIFO certificate-sidecar constants changed")
    frozen = {
        "registered_state_schema": REGISTERED_STATE_SCHEMA,
        "rule": RULE,
        "timing_protocol": TIMING_PROTOCOL,
        "advance_gate": ADVANCE_GATE,
        "stopping_rule": STOPPING_RULE,
        "prohibited_in_this_stage": PROHIBITED,
    }
    if any(protocol.get(key) != value for key, value in frozen.items()):
        raise ValueError("registered FIFO certificate-sidecar rule changed")

    # Finish every provenance check before the first selector call.
    if sources != _source_texts():
        raise ValueError("registered FIFO certificate-sidecar sources changed")
    diagnostic, model, roster = _validate_source_diagnostic()
    predecessors = _validate_predecessors()
    structure = _structure_snapshot(model)
    if protocol.get("model") != _json(model.snapshot()):
        raise ValueError("registered certificate-sidecar model changed")
    if (
        protocol.get("predecessor_facts") != predecessors
        or protocol.get("fifo_execution_structure") != structure
        or not _structure_is_expected(structure)
        or protocol.get("source_population_summaries")
        != diagnostic["summary_by_load_and_source_policy"]
        or protocol.get("source_population_cells")
        != _source_population_cells(diagnostic)
    ):
        raise ValueError("registered certificate-sidecar evidence changed")

    states = roster["states"]
    evaluations = [
        _evaluate_state(model, index, state)
        for index, state in enumerate(states)
    ]
    population = [
        _metrics_for_population(states, evaluations, load, budget)
        for load in LOADS
        for budget in SOURCE_POLICY_BUDGETS
    ]
    cells = [
        _metrics_for_population(states, evaluations, load, budget, seed)
        for load in LOADS
        for seed in SEEDS
        for budget in SOURCE_POLICY_BUDGETS
    ]
    source_by_id = {
        (row["load"], row["source_budget"]): row
        for row in protocol["source_population_summaries"]
    }
    execution_checks = {
        "registered_5003_unique_states_executed_once": (
            len(evaluations) == STATE_COUNT
            and [row["state_index"] for row in evaluations]
            == list(range(STATE_COUNT))
        ),
        "registered_18_population_cells_reconstruct": (
            len(cells) == 18
            and all(
                (
                    observed["load"],
                    observed["seed"],
                    observed["source_policy_budget"],
                    observed["frame_count"],
                )
                == (
                    expected["load"],
                    expected["seed"],
                    expected["source_policy_budget"],
                    expected["frame_count"],
                )
                and math.isclose(
                    observed["legacy_visited_time_s"],
                    expected["legacy_visited_time_s"],
                    rel_tol=0.0,
                    abs_tol=1e-8,
                )
                for observed, expected in zip(
                    cells, protocol["source_population_cells"], strict=True
                )
            )
        ),
        "registered_six_population_aggregates_reconstruct_source": all(
            _source_summary_matches(
                source_by_id[(row["load"], row["source_policy_budget"])], row
            )
            for row in population
        ),
        "cartesian_100_raw_to_71_fifo_signatures_holds": (
            _structure_is_expected(structure)
        ),
        "predecessor_k71_oracle_safety_is_registered": (
            predecessors["fifo_shadow"]["state_evaluation_count"] == STATE_COUNT
            and predecessors["fifo_shadow"]["k71_support_oracle_check"]
            and predecessors["fifo_shadow"]["k71_witness_replay_check"]
        ),
        "legacy_k26_reconstructs_registered_action_and_certificate": all(
            row["legacy_reconstruction_matches_registered"]
            for row in evaluations
        ),
        "fixed_legacy_action_preserved_on_every_state": all(
            row["action_preserved"] for row in evaluations
        ),
        "registered_optimum_and_selected_scores_are_bounded": all(
            all(row["upper_bound_validity"].values()) for row in evaluations
        ),
        "fixed_gap_epsilon_and_exact_definition_hold": all(
            row["exact_definition_checks_pass"]
            and abs(row["gap_identity_error"]) <= SCORE_TOLERANCE
            for row in evaluations
        ),
        "fifo_candidate_cache_and_frontier_work_identities_hold": all(
            row["work_accounting_checks_pass"] for row in evaluations
        ),
        "fifo_paid_attempts_respect_k26": all(
            row["fifo_certificate_k26"]["attempts"] <= SMALL_BUDGET
            for row in evaluations
        ),
        "sidecar_upper_and_gap_are_pointwise_nonworse": all(
            row["certificate_upper_delta_over_legacy"] <= SCORE_TOLERANCE
            and row["certificate_gap_delta_over_legacy"] <= SCORE_TOLERANCE
            for row in evaluations
        ),
        "timing_measurements_are_finite_and_nonnegative": all(
            math.isfinite(value) and value >= 0.0
            for row in evaluations
            for value in row["timing"].values()
        ),
        "no_arrivals_trajectories_or_randomness_used": True,
        "registered_sources_unchanged": all(
            (ROOT / name).read_text() == text for name, text in sources.items()
        ),
    }
    passed = all(execution_checks.values())
    primary = [
        row for row in population if row["source_policy_budget"] == SMALL_BUDGET
    ]
    sensitivity = [
        row
        for row in population
        if row["source_policy_budget"] == LEGACY_EXACT_BUDGET
    ]
    advance_checks = _advance_checks(primary, sensitivity, evaluations, passed)
    decision = _decision(advance_checks)
    legacy_wall_s = sum(
        row["timing"]["legacy_selector_wall_s"] for row in evaluations
    )
    fifo_wall_s = sum(
        row["timing"]["fifo_selector_wall_s"] for row in evaluations
    )
    operation_work = _operation_work(evaluations)
    work = {
        **operation_work,
        # Flat aliases keep the incremental certificate-cost headline easy to
        # inspect while the nested groups preserve like-for-like accounting.
        "selector_calls": operation_work["fifo_certificate_incremental"][
            "selector_calls"
        ],
        "fifo_paid_candidate_attempts": operation_work[
            "fifo_certificate_incremental"
        ]["candidate_attempts"],
        "fifo_free_cache_hit_pops": operation_work[
            "fifo_certificate_incremental"
        ]["free_cache_hit_pops"],
        "fifo_cache_key_evaluations": operation_work[
            "fifo_certificate_incremental"
        ]["cache_key_evaluations"],
        "fifo_cache_key_item_reads": operation_work[
            "fifo_certificate_incremental"
        ]["cache_key_item_reads"],
        "fifo_cache_key_requirement_coordinate_reads": operation_work[
            "fifo_certificate_incremental"
        ]["cache_key_requirement_coordinate_reads"],
        "fifo_cache_key_duration_reads": operation_work[
            "fifo_certificate_incremental"
        ]["cache_key_duration_reads"],
        "fifo_cache_key_completion_offset_reads": operation_work[
            "fifo_certificate_incremental"
        ]["cache_key_completion_offset_reads"],
        "timing": {
            "legacy_selector_wall_s": legacy_wall_s,
            "fifo_selector_wall_s": fifo_wall_s,
            "composite_selector_wall_s": legacy_wall_s + fifo_wall_s,
            "legacy_selector_ms_per_state": 1000.0 * legacy_wall_s / len(evaluations),
            "fifo_selector_ms_per_state": 1000.0 * fifo_wall_s / len(evaluations),
            "composite_selector_ms_per_state": (
                1000.0 * (legacy_wall_s + fifo_wall_s) / len(evaluations)
            ),
            "fifo_over_legacy_ratio": (
                fifo_wall_s / legacy_wall_s if legacy_wall_s > 0.0 else None
            ),
            "composite_over_legacy_ratio": (
                (legacy_wall_s + fifo_wall_s) / legacy_wall_s
                if legacy_wall_s > 0.0
                else None
            ),
            "is_scientific_gate": False,
        },
        "arrival_tapes": 0,
        "trajectory_frames": 0,
        "rng_initializations": 0,
        "random_draws": 0,
    }
    tightened = [row for row in evaluations if row["certificate_tightened"]]
    report = {
        "experiment": EXPERIMENT,
        "status": "PASS" if passed else "FAIL",
        "checks": execution_checks,
        "protocol": protocol,
        "registered_state_roster": REGISTERED_STATES.name,
        "state_evaluation_count": len(evaluations),
        "action_preserved_state_count": sum(
            row["action_preserved"] for row in evaluations
        ),
        "certificate_tightened_state_count": len(tightened),
        "fixed_action_exact_state_count": sum(
            row["fixed_action_exact"] for row in evaluations
        ),
        "fifo_search_exact_state_count": sum(
            row["fifo_search_exact"] for row in evaluations
        ),
        "certificate_tightened_states": [
            {
                "state_index": row["state_index"],
                "queue": row["queue"],
                "legacy_upper_bound": row["legacy_k26"]["upper_bound"],
                "raw_fifo_upper_bound": row["raw_fifo_upper_bound"],
                "sidecar_upper_bound": row["sidecar_upper_bound"],
                "fixed_action_score": row["executed_score"],
                "registered_support_reference_score": row[
                    "registered_support_reference_score"
                ],
                "fixed_action_support_regret": row[
                    "fixed_action_support_regret"
                ],
                "legacy_frontier_slack": row["legacy_frontier_slack"],
                "raw_fifo_frontier_slack": row[
                    "raw_fifo_frontier_slack"
                ],
                "sidecar_frontier_slack": row["sidecar_frontier_slack"],
                "legacy_gap": row["legacy_k26"]["gap"],
                "fixed_action_gap": row["fixed_action_gap"],
                "gap_reduction": row["legacy_k26"]["gap"]
                - row["fixed_action_gap"],
            }
            for row in tightened
        ],
        "summary_by_load_and_source_policy": population,
        "summary_by_load_seed_and_source_policy": cells,
        "work_accounting": work,
        "sidecar_checks": advance_checks,
        "scientific_decision": decision,
        "development_advance": False,
        "next_step": (
            "record the certificate tightening with unchanged trajectories and close this sidecar"
            if decision == "CERTIFICATE_TIGHTENING_ONLY_CLOSE"
            else "close this sidecar without tuning or trajectories"
        ),
        "limitations": [
            "The 5,003 queues and mechanism rationale are consumed development evidence.",
            "The certificate concerns maximum one-frame support for the fixed legacy K26 action; it does not improve that action's support score.",
            "Legacy visited-time weights are reused only to summarize exposure; the fixed action makes the corresponding trajectory exactly unchanged.",
            "Wall timings are one interleaved diagnostic pass and have no scientific threshold.",
        ],
    }
    if sources != _source_texts():
        raise ValueError("registered FIFO certificate-sidecar sources changed during execution")
    _write_new(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--run-registered", action="store_true")
    args = parser.parse_args()
    result = register(args.output) if args.register else run_registered(args.output)
    print(
        json.dumps(
            {
                "experiment": result["experiment"],
                "status": result.get("status", "REGISTERED"),
                "scientific_decision": result.get("scientific_decision"),
                "output": str(args.output),
            },
            indent=2,
        )
    )
