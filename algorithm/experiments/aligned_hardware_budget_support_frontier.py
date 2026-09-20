"""Registered finite-box budget-to-support-margin bridge for legacy v4.

The runner turns the selector's returned support gap into an approximate-oracle
coefficient over the complete axis-aligned queue box induced by the frozen
development roster.  It keeps that generic, finite-box calculation separate
from the stronger K-independent homogeneous-seed drift certificate already
proved for this particular frame model.  No arrivals or trajectories are run.
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
from algorithm.experiments.aligned_budgeted_frame_search_curve import (
    DEFAULT_CHECKPOINT_BUDGETS,
    trace_budgeted_frame_search_curve,
)
from algorithm.experiments.aligned_hardware_frame_model import (
    capacity_certificate,
    load_model,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
DEFAULT_OUTPUT = ARTIFACTS / (
    "aligned_hardware_budget_support_frontier_development_20260912.json"
)
SOURCE_DEVELOPMENT = ARTIFACTS / (
    "aligned_hardware_frame_experiment_v4_20260909.json"
)
SOURCE_DIAGNOSTIC = ARTIFACTS / (
    "aligned_hardware_candidate_search_diagnostic_development_20260910.json"
)
SOURCE_DIAGNOSTIC_PROTOCOL = ARTIFACTS / (
    "aligned_hardware_candidate_search_diagnostic_development_20260910_protocol.json"
)
SOURCE_REGISTERED_STATES = ARTIFACTS / (
    "aligned_hardware_duration_tie_order_shadow_development_20260910_registered_states.json"
)
SOURCE_FIXED_SOURCES = ARTIFACTS / (
    "aligned_hardware_candidate_search_diagnostic_development_20260910_fixed_sources.json"
)

LOADS = (0.50, 0.80, 0.95)
CHECKPOINT_BUDGETS = DEFAULT_CHECKPOINT_BUDGETS
ANCHOR_BUDGETS = (10, 14, 26, 120)
MINIMUM_BUDGET = 10
EXACT_BUDGET = 120
REGISTERED_STATE_COUNT = 5003
ENVELOPE_MIN = (0, 0, 0)
ENVELOPE_MAX = (40, 25, 24)
ENVELOPE_SHAPE = tuple(value + 1 for value in ENVELOPE_MAX)
ENVELOPE_STATE_COUNT = math.prod(ENVELOPE_SHAPE)
ENVELOPE_MAX_L1 = sum(ENVELOPE_MAX)
RAW_ACTION_COUNT = 100
SCORE_TOLERANCE = 1e-10
AGGREGATE_TOLERANCE = 1e-12
EXPERIMENT = "aligned_hardware_budget_support_frontier_v1"

NEW_SOURCES = (
    "algorithm/experiments/aligned_budgeted_frame_search_curve.py",
    "algorithm/experiments/aligned_hardware_budget_support_frontier.py",
)

RULE = {
    "domain": (
        "complete integer Cartesian queue box from zero through the coordinatewise "
        "maxima of the frozen 5,003-state roster"
    ),
    "budget_curve": (
        "one continuous frozen legacy-v4 search per queue with every integer "
        "checkpoint K10 through K120; an exact endpoint is carried forward"
    ),
    "independent_oracle": (
        "enumerate all 100 ordered two-lane raw actions for maximum support; "
        "global tied raw action is diagnostic because equality pruning only "
        "certifies support optimality"
    ),
    "certified_coefficient": (
        "alpha_D(K)=max over nonempty D of returned (upper-score)/L1(queue)"
    ),
    "diagnostic_coefficients": (
        "actual oracle regret and frontier slack require the exhaustive oracle "
        "and are not deployable selector outputs"
    ),
    "generic_residual_margin": (
        "capacity effective_slack_eta minus alpha_D(K); this is a conservative "
        "finite-box approximate-oracle bridge"
    ),
    "seed_specific_baseline": (
        "the existing homogeneous-seed all-state eta and backlog threshold apply "
        "unchanged to every K>=10 and are not added to the generic bridge"
    ),
}

ADVANCE_GATE = {
    "all_execution_checks": True,
    "subexact_budget_positive_generic_margin_at_all_three_loads": True,
    "strictly_less_paid_work_than_k120": True,
    "next_stage": "analytic all-queue or polyhedral-cone extension only",
}

DECISION_RULE = {
    "INVALID_EXECUTION": "one or more frozen execution checks fail",
    "FINITE_BOX_MARGIN_VACUOUS_STOP": (
        "no K below 120 leaves positive generic residual margin at all loads"
    ),
    "POSITIVE_MARGIN_WITHOUT_COMPUTE_SAVING_STOP": (
        "a subexact all-load positive margin exists but paid work is not below K120"
    ),
    "POSITIVE_FINITE_BOX_MARGIN_ADVANCE_TO_GLOBAL_EXTENSION": (
        "the first subexact all-load positive-margin K uses less paid work than K120"
    ),
}

PROHIBITED = (
    "arrivals, trajectory or drain simulation, RNG, server data, FIFO or other "
    "action-changing selectors, post-outcome budget changes, confirmation runs, "
    "and paper edits"
)


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
            indent=None if compact else 2,
            separators=(",", ":") if compact else None,
        )
        stream.write("\n")


def _source_texts():
    sources = json.loads(SOURCE_FIXED_SOURCES.read_text())
    changed = [
        name for name, text in sources.items() if (ROOT / name).read_text() != text
    ]
    if changed:
        raise ValueError(f"frozen aligned-frame source changed: {changed}")
    return {
        **sources,
        **{name: (ROOT / name).read_text() for name in NEW_SOURCES},
    }


def _selection_subset(selection):
    data = asdict(selection)
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
    return _json({field: data[field] for field in fields})


def _registered_reference_rows(diagnostic):
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
    return [
        {
            "queue": row["queue"],
            "k26": {field: row["k26"][field] for field in fields},
            "k120": {field: row["exact"][field] for field in fields},
        }
        for row in diagnostic["state_diagnostics"]
    ]


def _validate_inputs():
    development = json.loads(SOURCE_DEVELOPMENT.read_text())
    diagnostic = json.loads(SOURCE_DIAGNOSTIC.read_text())
    diagnostic_protocol = json.loads(SOURCE_DIAGNOSTIC_PROTOCOL.read_text())
    roster = json.loads(SOURCE_REGISTERED_STATES.read_text())
    model = load_model()
    certificates = [capacity_certificate(model, load) for load in LOADS]

    if (
        development.get("experiment") != "aligned_hardware_frame_and_budget_v4"
        or development.get("status") != "PASS"
        or not all(development.get("checks", {}).values())
        or development.get("protocol", {}).get("loads") != list(LOADS)
        or development.get("protocol", {}).get("budgets") != list(ANCHOR_BUDGETS)
        or development.get("protocol", {}).get("exact_reference_budget")
        != EXACT_BUDGET
        or _json(model.snapshot()) != development.get("model")
        or _json(certificates) != development.get("certificates")
    ):
        raise ValueError("budget frontier requires the frozen aligned-frame v4 result")
    if (
        diagnostic.get("protocol") != diagnostic_protocol
        or diagnostic.get("experiment")
        != "aligned_hardware_candidate_search_diagnostic_v1"
        or diagnostic.get("status") != "PASS"
        or not all(diagnostic.get("checks", {}).values())
        or diagnostic.get("development_advance")
        or len(diagnostic.get("state_diagnostics", ()))
        != REGISTERED_STATE_COUNT
        or diagnostic_protocol.get("small_budget") != 26
        or diagnostic_protocol.get("exact_budget") != EXACT_BUDGET
        or diagnostic_protocol.get("model") != _json(model.snapshot())
        or diagnostic_protocol.get("certificates") != _json(certificates)
    ):
        raise ValueError("budget frontier requires the frozen candidate diagnostic")
    states = roster.get("states", ())
    queues = [tuple(row["queue"]) for row in states]
    diagnostic_queues = [
        tuple(row["queue"]) for row in diagnostic["state_diagnostics"]
    ]
    minima = tuple(min(queue[i] for queue in queues) for i in range(3))
    maxima = tuple(max(queue[i] for queue in queues) for i in range(3))
    if (
        roster.get("state_count") != REGISTERED_STATE_COUNT
        or len(states) != REGISTERED_STATE_COUNT
        or len(set(queues)) != REGISTERED_STATE_COUNT
        or queues != diagnostic_queues
        or minima != ENVELOPE_MIN
        or maxima != ENVELOPE_MAX
        or len(CHECKPOINT_BUDGETS) != 111
        or CHECKPOINT_BUDGETS[0] != MINIMUM_BUDGET
        or CHECKPOINT_BUDGETS[-1] != EXACT_BUDGET
    ):
        raise ValueError("registered queues do not induce the frozen finite box")
    references = _registered_reference_rows(diagnostic)
    if any(
        row["queue"] != reference["queue"]
        or row["legacy_k26"] != reference["k26"]
        or not math.isclose(
            row["support_reference_score"],
            reference["k120"]["score"],
            rel_tol=0.0,
            abs_tol=SCORE_TOLERANCE,
        )
        for row, reference in zip(states, references, strict=True)
    ):
        raise ValueError("registered roster does not reconstruct K26/K120 references")
    return development, diagnostic, roster, model, certificates


def _domain_queues():
    return product(*(range(low, high + 1) for low, high in zip(
        ENVELOPE_MIN, ENVELOPE_MAX, strict=True
    )))


def _raw_action_family(model):
    choices = (None, *range(len(model.options)))
    family = []
    for indices in product(choices, repeat=model.lanes):
        service = tuple(
            sum(
                model.options[index].requirements[coordinate]
                for index in indices
                if index is not None
            )
            for coordinate in range(len(model.workloads))
        )
        duration = max(
            (1.0, *(model.options[index].duration for index in indices if index is not None))
        )
        key = (
            duration,
            tuple(len(model.options) if index is None else index for index in indices),
        )
        family.append((indices, service, duration, key))
    if len(family) != RAW_ACTION_COUNT:
        raise ValueError("independent raw action family changed")
    return tuple(family)


def _independent_oracle(queue, raw_actions):
    best = None
    for indices, service, duration, key in raw_actions:
        if any(done > waiting for done, waiting in zip(service, queue, strict=True)):
            continue
        score = sum(q * done for q, done in zip(queue, service, strict=True)) / duration
        if best is None or score > best["score"] or (
            score == best["score"] and key < best["key"]
        ):
            best = {
                "option_indices": indices,
                "service": service,
                "duration": duration,
                "score": score,
                "key": key,
            }
    if best is None:
        raise RuntimeError("all-idle independent action must be feasible")
    return best


def _new_budget_accumulator(budget):
    return {
        "budget": budget,
        "domain_state_count": 0,
        "nonempty_state_count": 0,
        "exact_score_state_count": 0,
        "exact_certificate_state_count": 0,
        "positive_gap_state_count": 0,
        "sum_normalized_certified_gap": 0.0,
        "sum_normalized_actual_regret": 0.0,
        "max_certified_normalized_gap": -1.0,
        "max_actual_normalized_regret_diagnostic": -1.0,
        "max_frontier_normalized_slack_diagnostic": -1.0,
        "alpha_witness": None,
        "actual_regret_witness": None,
        "frontier_slack_witness": None,
        "selector_work": {
            "selector_calls": 0,
            "candidate_attempts": 0,
            "feasible_candidate_evaluations": 0,
            "local_score_evaluations": 0,
            "bound_option_probes": 0,
            "bound_threshold_evaluations": 0,
            "peak_search_frontier": 0,
        },
    }


def _witness(queue, selection, optimum, regret, slack, gap):
    norm = sum(queue)
    return {
        "queue": queue,
        "queue_l1": norm,
        "selected_score": selection.score,
        "returned_upper_bound": selection.upper_bound,
        "independent_optimum_score": optimum,
        "actual_support_regret": regret,
        "frontier_bound_slack": slack,
        "returned_certified_gap": gap,
        "normalized_actual_regret": regret / norm if norm else 0.0,
        "normalized_frontier_slack": slack / norm if norm else 0.0,
        "normalized_certified_gap": gap / norm if norm else 0.0,
        "paid_attempts": selection.attempts,
        "exact_certificate": selection.exact,
    }


def _update_maximum(accumulator, field, witness_field, value, witness):
    current = accumulator[field]
    current_queue = (
        tuple(accumulator[witness_field]["queue"])
        if accumulator[witness_field] is not None
        else None
    )
    queue = tuple(witness["queue"])
    if value > current or (
        value == current and (current_queue is None or queue < current_queue)
    ):
        accumulator[field] = value
        accumulator[witness_field] = witness


def _update_budget(accumulator, queue, selection, optimum):
    regret = optimum - selection.score
    slack = selection.upper_bound - optimum
    gap = selection.upper_bound - selection.score
    norm = sum(queue)
    normalized_regret = regret / norm if norm else 0.0
    normalized_slack = slack / norm if norm else 0.0
    normalized_gap = gap / norm if norm else 0.0
    witness = _witness(queue, selection, optimum, regret, slack, gap)

    accumulator["domain_state_count"] += 1
    accumulator["nonempty_state_count"] += int(norm > 0)
    accumulator["exact_score_state_count"] += int(regret <= SCORE_TOLERANCE)
    accumulator["exact_certificate_state_count"] += int(selection.exact)
    accumulator["positive_gap_state_count"] += int(gap > SCORE_TOLERANCE)
    accumulator["sum_normalized_certified_gap"] += normalized_gap
    accumulator["sum_normalized_actual_regret"] += normalized_regret
    if norm:
        _update_maximum(
            accumulator,
            "max_certified_normalized_gap",
            "alpha_witness",
            normalized_gap,
            witness,
        )
        _update_maximum(
            accumulator,
            "max_actual_normalized_regret_diagnostic",
            "actual_regret_witness",
            normalized_regret,
            witness,
        )
        _update_maximum(
            accumulator,
            "max_frontier_normalized_slack_diagnostic",
            "frontier_slack_witness",
            normalized_slack,
            witness,
        )
    work = accumulator["selector_work"]
    work["selector_calls"] += 1
    work["candidate_attempts"] += selection.attempts
    work["feasible_candidate_evaluations"] += selection.feasible_candidate_count
    work["local_score_evaluations"] += selection.local_score_evaluations
    work["bound_option_probes"] += selection.bound_option_probes
    work["bound_threshold_evaluations"] += selection.bound_threshold_evaluations
    work["peak_search_frontier"] = max(
        work["peak_search_frontier"], selection.frontier_peak
    )


def _finalize_budget(accumulator, certificates):
    count = accumulator["domain_state_count"]
    nonempty = accumulator["nonempty_state_count"]
    accumulator["mean_normalized_certified_gap"] = (
        accumulator.pop("sum_normalized_certified_gap") / nonempty
    )
    accumulator["mean_normalized_actual_regret_diagnostic"] = (
        accumulator.pop("sum_normalized_actual_regret") / nonempty
    )
    accumulator["exact_score_domain_fraction"] = (
        accumulator["exact_score_state_count"] / count
    )
    accumulator["exact_certificate_domain_fraction"] = (
        accumulator["exact_certificate_state_count"] / count
    )
    work = accumulator["selector_work"]
    work["mean_candidate_attempts"] = work["candidate_attempts"] / count
    alpha = accumulator["max_certified_normalized_gap"]
    accumulator["capacity_bridge"] = [
        _capacity_bridge_row(certificate, alpha) for certificate in certificates
    ]
    return accumulator


def _capacity_bridge_row(certificate, alpha):
    support_slack = float(certificate["support_slack_delta"])
    projection_loss = float(certificate["projection_support_loss_rate"])
    eta = float(certificate["effective_slack_eta"])
    eta_reconstructed = support_slack - projection_loss
    residual = eta - alpha
    drift_constant = float(certificate["drift_constant"])
    second_moment = float(certificate["second_moment_B"])
    availability_loss = float(certificate["availability_loss_C"])
    tau_max = float(certificate["tau_max"])
    reconstructed = second_moment + tau_max * availability_loss
    tau_min = float(certificate["tau_min"])
    positive = residual > 0.0
    threshold = drift_constant / (tau_min * residual) if positive else None
    seed_threshold = float(certificate["negative_drift_backlog_threshold"])
    seed_threshold_reconstructed = drift_constant / (tau_min * eta)
    return {
        "load": certificate["load_factor"],
        "support_slack_delta": support_slack,
        "projection_support_loss_rate": projection_loss,
        "seed_specific_effective_slack_eta": eta,
        "seed_specific_effective_slack_eta_reconstructed": eta_reconstructed,
        "seed_specific_negative_drift_backlog_threshold": seed_threshold,
        "seed_specific_negative_drift_backlog_threshold_reconstructed": (
            seed_threshold_reconstructed
        ),
        "seed_specific_certificate_applies_to_every_k_at_least_10": True,
        "finite_box_approximate_oracle_alpha_1": alpha,
        "generic_finite_box_residual_margin": residual,
        "generic_residual_margin_positive": positive,
        "second_moment_B": second_moment,
        "availability_loss_C": availability_loss,
        "tau_min": tau_min,
        "tau_max": tau_max,
        "drift_constant": drift_constant,
        "drift_constant_reconstructed": reconstructed,
        "algebraic_threshold_if_alpha_extended_globally": threshold,
        "minimum_integer_backlog_above_threshold": (
            math.floor(threshold) + 1 if threshold is not None else None
        ),
        "threshold_over_seed_specific_ratio": (
            threshold / seed_threshold if threshold is not None else None
        ),
        "threshold_relative_increase": (
            threshold / seed_threshold - 1.0 if threshold is not None else None
        ),
        "threshold_absolute_increase": (
            threshold - seed_threshold if threshold is not None else None
        ),
        "domain_max_l1_backlog": ENVELOPE_MAX_L1,
        "negative_drift_region_intersects_domain": (
            threshold is not None and threshold < ENVELOPE_MAX_L1
        ),
        "generic_drift_upper_bound_at_domain_max_l1": (
            drift_constant - tau_min * residual * ENVELOPE_MAX_L1
        ),
    }


def _work_identities(curve, option_count):
    work = curve.work
    final = curve.checkpoints[-1].selection
    return (
        work.trace_calls == 1
        and work.checkpoint_reads == len(CHECKPOINT_BUDGETS)
        and work.seed_attempts == option_count + 1
        and work.paid_candidate_attempts == final.attempts
        and work.feasible_candidate_evaluations == final.feasible_candidate_count
        and work.infeasible_candidate_attempts
        == work.paid_candidate_attempts - work.feasible_candidate_evaluations
        and work.local_score_evaluations == option_count
        and work.bound_option_probes == final.bound_option_probes
        and work.bound_threshold_evaluations == final.bound_threshold_evaluations
        and work.frontier_push_attempts
        == work.frontier_pushes + work.bound_rejected_pushes
        and work.frontier_pops
        == work.expanded_prefixes + work.certificate_closure_pops
        and work.frontier_pushes
        == work.frontier_pops
        + work.frontier_entries_discarded_on_exact
        + work.frontier_remaining
        and work.frontier_peak == final.frontier_peak
        and work.frontier_remaining == 0
    )


def _capacity_bridge_reconstructs(row):
    eta = row["support_slack_delta"] - row["projection_support_loss_rate"]
    drift_constant = row["second_moment_B"] + (
        row["tau_max"] * row["availability_loss_C"]
    )
    seed_threshold = drift_constant / (row["tau_min"] * eta)
    residual = eta - row["finite_box_approximate_oracle_alpha_1"]
    positive = residual > 0.0
    threshold = (
        drift_constant / (row["tau_min"] * residual) if positive else None
    )
    return (
        math.isclose(
            row["seed_specific_effective_slack_eta"],
            eta,
            rel_tol=0.0,
            abs_tol=SCORE_TOLERANCE,
        )
        and math.isclose(
            row["seed_specific_effective_slack_eta_reconstructed"],
            eta,
            rel_tol=0.0,
            abs_tol=SCORE_TOLERANCE,
        )
        and math.isclose(
            row["drift_constant"],
            drift_constant,
            rel_tol=0.0,
            abs_tol=SCORE_TOLERANCE,
        )
        and math.isclose(
            row["drift_constant_reconstructed"],
            drift_constant,
            rel_tol=0.0,
            abs_tol=SCORE_TOLERANCE,
        )
        and math.isclose(
            row["seed_specific_negative_drift_backlog_threshold"],
            seed_threshold,
            rel_tol=0.0,
            abs_tol=SCORE_TOLERANCE,
        )
        and math.isclose(
            row["seed_specific_negative_drift_backlog_threshold_reconstructed"],
            seed_threshold,
            rel_tol=0.0,
            abs_tol=SCORE_TOLERANCE,
        )
        and math.isclose(
            row["generic_finite_box_residual_margin"],
            residual,
            rel_tol=0.0,
            abs_tol=SCORE_TOLERANCE,
        )
        and row["generic_residual_margin_positive"] is positive
        and (
            row["algebraic_threshold_if_alpha_extended_globally"] is None
            if threshold is None
            else math.isclose(
                row["algebraic_threshold_if_alpha_extended_globally"],
                threshold,
                rel_tol=0.0,
                abs_tol=SCORE_TOLERANCE,
            )
        )
    )


def _curve_monotone(curve):
    selections = [checkpoint.selection for checkpoint in curve.checkpoints]
    return all(
        right.score + SCORE_TOLERANCE >= left.score
        and right.upper_bound <= left.upper_bound + SCORE_TOLERANCE
        and right.gap <= left.gap + SCORE_TOLERANCE
        and right.attempts >= left.attempts
        and right.attempts <= budget
        and (not left.exact or right.exact)
        for left, right, budget in zip(
            selections[:-1], selections[1:], CHECKPOINT_BUDGETS[1:], strict=True
        )
    )


def _selection_reconstructs(model, queue, selection):
    service = tuple(
        sum(
            model.options[index].requirements[coordinate]
            for index in selection.option_indices
            if index is not None
        )
        for coordinate in range(len(queue))
    )
    duration = max(
        (1.0, *(
            model.options[index].duration
            for index in selection.option_indices
            if index is not None
        ))
    )
    score = sum(
        q * done for q, done in zip(queue, service, strict=True)
    ) / duration
    return (
        selection.service == service
        and all(done <= waiting for done, waiting in zip(service, queue, strict=True))
        and selection.duration == duration
        and selection.score == score
    )


def _registered_maps(diagnostic, roster):
    diagnostic_map = {
        tuple(row["queue"]): row for row in diagnostic["state_diagnostics"]
    }
    roster_map = {tuple(row["queue"]): row for row in roster["states"]}
    if set(diagnostic_map) != set(roster_map):
        raise ValueError("diagnostic and registered queue rosters differ")
    return diagnostic_map, roster_map


def _decision(execution_passed, summaries):
    if not execution_passed:
        return "INVALID_EXECUTION", None
    feasible = [
        row
        for row in summaries
        if row["budget"] < EXACT_BUDGET
        and all(
            bridge["generic_residual_margin_positive"]
            for bridge in row["capacity_bridge"]
        )
    ]
    if not feasible:
        return "FINITE_BOX_MARGIN_VACUOUS_STOP", None
    first = feasible[0]
    exact = summaries[-1]
    saves_work = (
        first["selector_work"]["candidate_attempts"]
        < exact["selector_work"]["candidate_attempts"]
    )
    if not saves_work:
        return "POSITIVE_MARGIN_WITHOUT_COMPUTE_SAVING_STOP", first["budget"]
    return (
        "POSITIVE_FINITE_BOX_MARGIN_ADVANCE_TO_GLOBAL_EXTENSION",
        first["budget"],
    )


def register(output=DEFAULT_OUTPUT):
    output, protocol_path, sources_path = _paths(output)
    if any(path.exists() for path in (output, protocol_path, sources_path)):
        raise ValueError("budget-support frontier artifacts already exist")
    sources = _source_texts()
    _development, _diagnostic, _roster, model, certificates = _validate_inputs()
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT,
        "role": "development finite-box budget-to-approximate-oracle bridge",
        "source_development": SOURCE_DEVELOPMENT.name,
        "source_diagnostic": SOURCE_DIAGNOSTIC.name,
        "source_registered_states": SOURCE_REGISTERED_STATES.name,
        "loads": list(LOADS),
        "checkpoint_budgets": list(CHECKPOINT_BUDGETS),
        "anchor_budgets": list(ANCHOR_BUDGETS),
        "minimum_budget": MINIMUM_BUDGET,
        "exact_budget": EXACT_BUDGET,
        "registered_state_count": REGISTERED_STATE_COUNT,
        "envelope_min": list(ENVELOPE_MIN),
        "envelope_max": list(ENVELOPE_MAX),
        "envelope_shape": list(ENVELOPE_SHAPE),
        "envelope_state_count": ENVELOPE_STATE_COUNT,
        "envelope_max_l1": ENVELOPE_MAX_L1,
        "raw_action_count": RAW_ACTION_COUNT,
        "score_tolerance": SCORE_TOLERANCE,
        "model": _json(model.snapshot()),
        "capacity_certificates": _json(certificates),
        "registered_reference_rows": _registered_reference_rows(_diagnostic),
        "rule": RULE,
        "advance_gate": ADVANCE_GATE,
        "decision_rule": DECISION_RULE,
        "prohibited": PROHIBITED,
        "timing_is_scientific_gate": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_new(protocol_path, protocol, compact=True)
    _write_new(sources_path, sources)
    return protocol


def run_registered(output=DEFAULT_OUTPUT):
    output, protocol_path, sources_path = _paths(output)
    if output.exists():
        raise ValueError("budget-support frontier result already exists")
    protocol = json.loads(protocol_path.read_text())
    sources = json.loads(sources_path.read_text())
    expected_constants = {
        "experiment": EXPERIMENT,
        "source_development": SOURCE_DEVELOPMENT.name,
        "source_diagnostic": SOURCE_DIAGNOSTIC.name,
        "source_registered_states": SOURCE_REGISTERED_STATES.name,
        "loads": list(LOADS),
        "checkpoint_budgets": list(CHECKPOINT_BUDGETS),
        "anchor_budgets": list(ANCHOR_BUDGETS),
        "minimum_budget": MINIMUM_BUDGET,
        "exact_budget": EXACT_BUDGET,
        "registered_state_count": REGISTERED_STATE_COUNT,
        "envelope_min": list(ENVELOPE_MIN),
        "envelope_max": list(ENVELOPE_MAX),
        "envelope_shape": list(ENVELOPE_SHAPE),
        "envelope_state_count": ENVELOPE_STATE_COUNT,
        "envelope_max_l1": ENVELOPE_MAX_L1,
        "raw_action_count": RAW_ACTION_COUNT,
        "score_tolerance": SCORE_TOLERANCE,
        "rule": RULE,
        "advance_gate": ADVANCE_GATE,
        "decision_rule": DECISION_RULE,
        "prohibited": PROHIBITED,
        "timing_is_scientific_gate": False,
    }
    if any(protocol.get(key) != value for key, value in expected_constants.items()):
        raise ValueError("registered budget-support frontier protocol changed")
    if sources != _source_texts():
        raise ValueError("registered budget-support frontier sources changed")
    development, diagnostic, roster, model, certificates = _validate_inputs()
    if (
        protocol.get("model") != _json(model.snapshot())
        or protocol.get("capacity_certificates") != _json(certificates)
        or protocol.get("registered_reference_rows")
        != _registered_reference_rows(diagnostic)
    ):
        raise ValueError("registered model or capacity certificate changed")

    diagnostic_map, roster_map = _registered_maps(diagnostic, roster)
    raw_actions = _raw_action_family(model)
    accumulators = {
        budget: _new_budget_accumulator(budget) for budget in CHECKPOINT_BUDGETS
    }
    domain_seen = set()
    trace_work = None
    execution = {
        "all_states_have_all_checkpoints": True,
        "support_score_and_upper_bound_cover_oracle": True,
        "gap_regret_slack_decomposition": True,
        "every_checkpoint_selection_is_realizable": True,
        "normalized_empty_queue_is_zero": True,
        "per_state_budget_curve_is_monotone": True,
        "k120_is_exact_support_oracle": True,
        "curve_work_identities_hold": True,
        "anchor_checkpoints_match_fresh_selector": True,
        "registered_k26_and_k120_reconstruct": True,
    }
    global_tie_action_match_count = 0
    loop_started = time.perf_counter()
    trace_wall_s = 0.0
    oracle_wall_s = 0.0
    anchor_wall_s = 0.0
    oracle_state_count = 0
    anchor_selector_call_count = 0
    registered_match_count = 0

    domain_sequence = tuple(_domain_queues())
    for queue in domain_sequence:
        queue = tuple(queue)
        domain_seen.add(queue)
        started = time.perf_counter()
        oracle = _independent_oracle(queue, raw_actions)
        oracle_state_count += 1
        oracle_wall_s += time.perf_counter() - started
        started = time.perf_counter()
        curve = trace_budgeted_frame_search_curve(
            model.options,
            queue,
            model.lanes,
            checkpoint_budgets=CHECKPOINT_BUDGETS,
        )
        trace_wall_s += time.perf_counter() - started
        if trace_work is None:
            trace_work = {key: 0 for key in asdict(curve.work)}
            trace_work["frontier_peak"] = 0
        for key, value in asdict(curve.work).items():
            if key == "frontier_peak":
                trace_work[key] = max(trace_work[key], value)
            else:
                trace_work[key] += value
        execution["all_states_have_all_checkpoints"] &= (
            curve.checkpoint_budgets == CHECKPOINT_BUDGETS
            and len(curve.checkpoints) == len(CHECKPOINT_BUDGETS)
        )
        execution["per_state_budget_curve_is_monotone"] &= _curve_monotone(curve)
        execution["curve_work_identities_hold"] &= _work_identities(
            curve, len(model.options)
        )
        exact = curve.selection_at(EXACT_BUDGET)
        execution["k120_is_exact_support_oracle"] &= (
            exact.exact
            and exact.gap <= SCORE_TOLERANCE
            and math.isclose(
                exact.score,
                oracle["score"],
                rel_tol=0.0,
                abs_tol=SCORE_TOLERANCE,
            )
        )
        global_tie_action_match_count += int(
            tuple(exact.option_indices) == tuple(oracle["option_indices"])
        )

        for budget, checkpoint in zip(
            CHECKPOINT_BUDGETS, curve.checkpoints, strict=True
        ):
            selection = checkpoint.selection
            regret = oracle["score"] - selection.score
            slack = selection.upper_bound - oracle["score"]
            gap = selection.upper_bound - selection.score
            execution["support_score_and_upper_bound_cover_oracle"] &= (
                selection.score <= oracle["score"] + SCORE_TOLERANCE
                and oracle["score"] <= selection.upper_bound + SCORE_TOLERANCE
            )
            execution["gap_regret_slack_decomposition"] &= (
                regret >= -SCORE_TOLERANCE
                and slack >= -SCORE_TOLERANCE
                and gap >= -SCORE_TOLERANCE
                and math.isclose(
                    gap,
                    regret + slack,
                    rel_tol=0.0,
                    abs_tol=SCORE_TOLERANCE,
                )
                and math.isclose(
                    selection.gap,
                    gap,
                    rel_tol=0.0,
                    abs_tol=SCORE_TOLERANCE,
                )
                and math.isclose(
                    selection.epsilon,
                    gap / sum(queue) if sum(queue) else 0.0,
                    rel_tol=0.0,
                    abs_tol=SCORE_TOLERANCE,
                )
            )
            execution["every_checkpoint_selection_is_realizable"] &= (
                _selection_reconstructs(model, queue, selection)
            )
            if sum(queue) == 0:
                execution["normalized_empty_queue_is_zero"] &= (
                    selection.score == 0.0
                    and selection.upper_bound == 0.0
                    and selection.gap == 0.0
                    and selection.epsilon == 0.0
                )
            _update_budget(
                accumulators[budget], queue, selection, oracle["score"]
            )

        started = time.perf_counter()
        for budget in ANCHOR_BUDGETS:
            fresh = select_budgeted_frame(
                model.options, queue, model.lanes, budget
            )
            anchor_selector_call_count += 1
            execution["anchor_checkpoints_match_fresh_selector"] &= (
                fresh == curve.selection_at(budget)
            )
        anchor_wall_s += time.perf_counter() - started

        if queue in diagnostic_map:
            source = diagnostic_map[queue]
            registered = roster_map[queue]
            execution["registered_k26_and_k120_reconstruct"] &= (
                _selection_subset(curve.selection_at(26))
                == registered["legacy_k26"]
                and _selection_subset(curve.selection_at(26))
                == _json({
                    key: source["k26"][key]
                    for key in registered["legacy_k26"]
                })
                and _selection_subset(curve.selection_at(EXACT_BUDGET))
                == _json({
                    key: source["exact"][key]
                    for key in registered["legacy_k26"]
                })
                and math.isclose(
                    curve.selection_at(EXACT_BUDGET).score,
                    registered["support_reference_score"],
                    rel_tol=0.0,
                    abs_tol=SCORE_TOLERANCE,
                )
            )
            registered_match_count += 1

    combined_loop_wall_s = time.perf_counter() - loop_started
    summaries = [
        _finalize_budget(accumulators[budget], certificates)
        for budget in CHECKPOINT_BUDGETS
    ]
    alphas = [row["max_certified_normalized_gap"] for row in summaries]
    execution.update({
        "finite_box_is_complete_unique_and_ordered": (
            len(domain_sequence) == ENVELOPE_STATE_COUNT
            and len(domain_seen) == ENVELOPE_STATE_COUNT
            and domain_sequence[0] == ENVELOPE_MIN
            and domain_sequence[-1] == ENVELOPE_MAX
            and all(
                left < right
                for left, right in zip(
                    domain_sequence[:-1], domain_sequence[1:], strict=True
                )
            )
            and ENVELOPE_STATE_COUNT == 26650
        ),
        "continuous_trace_and_checkpoint_accounting": (
            trace_work["trace_calls"] == ENVELOPE_STATE_COUNT
            and trace_work["checkpoint_reads"]
            == ENVELOPE_STATE_COUNT * len(CHECKPOINT_BUDGETS)
            and trace_work["paid_candidate_attempts"]
            <= ENVELOPE_STATE_COUNT * EXACT_BUDGET
        ),
        "registered_roster_is_contained_and_reconstructed": (
            registered_match_count == REGISTERED_STATE_COUNT
            and set(diagnostic_map).issubset(domain_seen)
        ),
        "independent_oracle_action_accounting": (
            oracle_state_count == ENVELOPE_STATE_COUNT
            and oracle_state_count * len(raw_actions) == 2_665_000
        ),
        "fresh_anchor_selector_accounting": (
            anchor_selector_call_count
            == ENVELOPE_STATE_COUNT * len(ANCHOR_BUDGETS)
        ),
        "budget_summary_accounting": (
            len(summaries) == len(CHECKPOINT_BUDGETS)
            and [row["budget"] for row in summaries]
            == list(CHECKPOINT_BUDGETS)
            and all(
                row["domain_state_count"] == ENVELOPE_STATE_COUNT
                and row["nonempty_state_count"] == ENVELOPE_STATE_COUNT - 1
                and row["selector_work"]["selector_calls"]
                == ENVELOPE_STATE_COUNT
                for row in summaries
            )
        ),
        "domain_alpha_curve_is_nonincreasing_and_exact_at_k120": (
            all(
                right <= left + AGGREGATE_TOLERANCE
                for left, right in zip(alphas[:-1], alphas[1:], strict=True)
            )
            and alphas[-1] <= SCORE_TOLERANCE
        ),
        "capacity_bridge_reconstructs_without_double_counting": all(
            _capacity_bridge_reconstructs(row)
            for summary in summaries
            for row in summary["capacity_bridge"]
        ),
        "no_arrivals_trajectories_drains_or_rng": True,
        "registered_sources_unchanged": all(
            (ROOT / name).read_text() == text for name, text in sources.items()
        ),
    })
    execution_passed = all(execution.values())
    decision, first_all_load_positive_budget = _decision(
        execution_passed, summaries
    )
    first_summary = (
        summaries[first_all_load_positive_budget - MINIMUM_BUDGET]
        if first_all_load_positive_budget is not None
        else None
    )
    negative_region_all_loads = bool(
        first_summary is not None
        and all(
            row["negative_drift_region_intersects_domain"]
            for row in first_summary["capacity_bridge"]
        )
    )
    report = {
        "experiment": EXPERIMENT,
        "status": "PASS" if execution_passed else "FAIL",
        "checks": execution,
        "protocol": protocol,
        "domain": {
            "kind": "complete_axis_aligned_integer_box_induced_by_registered_coordinate_extrema",
            "minimum": ENVELOPE_MIN,
            "maximum": ENVELOPE_MAX,
            "shape": ENVELOPE_SHAPE,
            "state_count": ENVELOPE_STATE_COUNT,
            "maximum_l1_backlog": ENVELOPE_MAX_L1,
            "registered_state_count": REGISTERED_STATE_COUNT,
            "registered_maximum_l1_backlog": max(
                sum(queue) for queue in diagnostic_map
            ),
            "is_invariant_under_poisson_arrivals": False,
        },
        "budget_summaries": summaries,
        "first_subexact_all_load_positive_margin_budget": (
            first_all_load_positive_budget
        ),
        "negative_drift_region_intersects_domain_at_all_loads_for_first_budget": (
            negative_region_all_loads
        ),
        "finite_box_negative_drift_status_for_first_budget": (
            "FINITE_BOX_CONTAINS_STRICT_NEGATIVE_DRIFT_STATES_AT_ALL_LOADS"
            if negative_region_all_loads
            else "NO_FINITE_BOX_NEGATIVE_DRIFT_STATE_AT_ALL_LOADS"
        ),
        "independent_oracle": {
            "raw_actions_per_state": len(raw_actions),
            "state_count": ENVELOPE_STATE_COUNT,
            "raw_action_feasibility_checks": ENVELOPE_STATE_COUNT * len(raw_actions),
            "k120_global_tie_action_match_state_count_diagnostic": (
                global_tie_action_match_count
            ),
            "k120_global_tie_action_mismatch_state_count_diagnostic": (
                ENVELOPE_STATE_COUNT - global_tie_action_match_count
            ),
            "support_score_is_execution_gate": True,
            "global_tied_raw_action_is_execution_gate": False,
        },
        "actual_continuous_trace_work": trace_work,
        "validation_work": {
            "independent_oracle_state_calls": oracle_state_count,
            "independent_oracle_raw_action_feasibility_checks": (
                oracle_state_count * len(raw_actions)
            ),
            "fresh_anchor_selector_calls": anchor_selector_call_count,
            "registered_reference_state_checks": registered_match_count,
            "charged_to_deployable_budget": False,
        },
        "result_retention": {
            "budget_summary_count": len(summaries),
            "per_state_checkpoint_rows_retained": 0,
        },
        "timing": {
            "combined_loop_wall_s": combined_loop_wall_s,
            "continuous_trace_wall_s_inside_loop": trace_wall_s,
            "independent_oracle_wall_s_inside_loop": oracle_wall_s,
            "fresh_anchor_selector_wall_s_inside_loop": anchor_wall_s,
            "is_scientific_gate": False,
        },
        "scientific_decision": decision,
        "development_advance": (
            decision
            == "POSITIVE_FINITE_BOX_MARGIN_ADVANCE_TO_GLOBAL_EXTENSION"
        ),
        "next_step": (
            "derive an all-queue or polyhedral-cone alpha envelope; do not run trajectories"
            if decision
            == "POSITIVE_FINITE_BOX_MARGIN_ADVANCE_TO_GLOBAL_EXTENSION"
            else "close the generic finite-box bridge without tuning its budgets or domain"
        ),
        "claim_boundary": {
            "supports": (
                "a computable K-to-approximate-oracle coefficient and residual-margin "
                "mapping on the complete registered-coordinate finite box"
            ),
            "does_not_support": (
                "global stability, positive recurrence, a Poisson-invariant finite state "
                "space, or improvement of the existing homogeneous-seed certificate"
            ),
            "seed_specific_certificate": (
                "the K-independent all-state homogeneous-seed certificate remains the "
                "stronger stability result for this model"
            ),
        },
        "limitations": [
            "The finite box is induced by reused development queues and is not an independent holdout.",
            "Poisson arrivals can leave the finite box, so its alpha coefficient cannot be extended outside the box without another proof.",
            "Actual regret and frontier slack diagnostics use the exhaustive oracle; only the returned gap is deployable.",
            "K120 support exactness does not certify the globally earliest tied raw action because equality prefixes may be pruned.",
        ],
    }
    if sources != _source_texts():
        raise ValueError("registered sources changed during budget-frontier execution")
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
    print(json.dumps({
        "experiment": result["experiment"],
        "status": result.get("status", "REGISTERED"),
        "scientific_decision": result.get("scientific_decision"),
        "output": str(args.output),
    }, indent=2))
