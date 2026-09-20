"""Registered envelope-normalized trajectory transition.

The runner executes R8/K48 single-exchange and R16/K9 conditional K144
support-rescue candidates against independent cold K4096 trajectories.  A
second cold K4096 call shadows each candidate state without affecting queue
evolution.  Registration reads and freezes evidence only; it never generates
arrivals or calls a selector.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import mean, stdev
import time
from typing import Any, Mapping, Sequence

from scipy.stats import t as student_t

from algorithm.canonical_count_space_bnb_frame_candidates import select_canonical_count_space_bnb_frame
from algorithm.canonical_count_space_bnb_single_exchange_frame_candidates import (
    select_canonical_count_space_bnb_single_exchange_frame,
    total_work_row as single_exchange_total_work_row,
)
from algorithm.canonical_score_tie_bulk_substitution_frame_candidates import total_work_row as bulk_total_work_row
from algorithm.canonical_support_rescue_bulk_tie_frame_candidates import (
    select_canonical_support_rescue_bulk_tie_frame,
    total_work_row as rescue_total_work_row,
)
from algorithm.experiments import aligned_hardware_joint95_baseline_confirmation as baseline
from algorithm.experiments import aligned_hardware_joint95_v5_baseline_confirmation_v3 as baseline_v5
from algorithm.experiments import aligned_hardware_joint95_v5_envelope_r16_support_rescue_confirmation as confirmation
from algorithm.experiments.aligned_frame_flow_decomposition import decompose_flow
from algorithm.experiments.aligned_hardware_frame_experiment import poisson_arrivals


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
STEM = "aligned_hardware_joint95_v5_envelope_normalized_trajectory_transition_20260920"
DEFAULT_REGISTRATION = ARTIFACTS / f"{STEM}_protocol.json"
DEFAULT_FIXED_SOURCES = ARTIFACTS / f"{STEM}_fixed_sources.json"
DEFAULT_RESULT = ARTIFACTS / f"{STEM}.json"
DEFAULT_CONFIRMATION_REGISTRATION = confirmation.DEFAULT_REGISTRATION
DEFAULT_CONFIRMATION_ROSTER = confirmation.DEFAULT_ROSTER
DEFAULT_CONFIRMATION_FIXED_SOURCES = confirmation.DEFAULT_FIXED_SOURCES
DEFAULT_CONFIRMATION_RESULT = confirmation.DEFAULT_RESULT
DEFAULT_BASELINE_REGISTRATION = ARTIFACTS / "aligned_hardware_joint95_v5_baseline_confirmation_v3_20260919_protocol.json"
DEFAULT_BASELINE_FIXED_SOURCES = ARTIFACTS / "aligned_hardware_joint95_v5_baseline_confirmation_v3_20260919_fixed_sources.json"
DEFAULT_BASELINE_RESULT = ARTIFACTS / "aligned_hardware_joint95_v5_baseline_confirmation_v3_20260919.json"

EXPERIMENT = "aligned_hardware_joint95_v5_envelope_normalized_trajectory_transition_v1"
GATE = "aligned_hardware_joint95_v5_envelope_normalized_trajectory_transition_runner"
REGISTRATION_STATUS = "REGISTERED_AWAITING_ENVELOPE_NORMALIZED_TRAJECTORY"
PASS_DECISION = "ENVELOPE_NORMALIZED_TRAJECTORY_TRANSITION_PASS"
INVALID_DECISION = "INVALID_EXECUTION"
SUPPORT_FAIL_DECISION = "ENVELOPE_NORMALIZED_TRAJECTORY_SUPPORT_FAIL_STOP"
ACTION_FAIL_DECISION = "ENVELOPE_NORMALIZED_TRAJECTORY_ACTION_OR_REGRET_FAIL_STOP"
FLOW_FAIL_DECISION = "ENVELOPE_NORMALIZED_TRAJECTORY_FLOW_NONINFERIORITY_FAIL_STOP"
COMPUTE_FAIL_DECISION = "ENVELOPE_NORMALIZED_TRAJECTORY_COMPUTE_FAIL_STOP"
LANES = (8, 16)
LOADS = (0.5, 0.8, 0.95)
ARRIVAL_SEEDS = tuple(range(12041, 12061))
ARRIVAL_HORIZON_S = 600.0
CANDIDATE_BUDGETS = {8: 48, 16: 9}
RESCUE_BUDGET = 144
REFERENCE_BUDGET = 4096
SUPPORT_EPSILON_LIMIT = confirmation.SUPPORT_EPSILON_LIMIT
WORK_FIELDS = confirmation.canonical.WORK_FIELDS
WORK_GATE_FIELDS = confirmation.WORK_GATE_FIELDS
SCORE_TOLERANCE = confirmation.canonical.SCORE_TOLERANCE
EXECUTION_TOLERANCE = 1e-8
NONINFERIORITY_MARGIN = 0.01
FAMILY_ALPHA = 0.05
PRIMARY_FAMILY_SIZE = 6
EXECUTION_WORKERS = 4
THIS_SOURCE = "algorithm/experiments/aligned_hardware_joint95_v5_envelope_normalized_trajectory_transition.py"
FLOW_SOURCE = "algorithm/experiments/aligned_frame_flow_decomposition.py"
ARRIVAL_SOURCE = "algorithm/experiments/aligned_hardware_frame_experiment.py"
BASELINE_SOURCE = "algorithm/experiments/aligned_hardware_joint95_baseline_confirmation.py"


class EnvelopeTrajectoryRefusal(RuntimeError):
    """The frozen trajectory contract is not satisfied."""


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EnvelopeTrajectoryRefusal(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EnvelopeTrajectoryRefusal(f"JSON payload is not an object: {path}")
    return value


def _write_new(path: Path, payload: Mapping[str, Any], *, compact: bool = False) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=None if compact else 2,
                  separators=(",", ":") if compact else None)
        stream.write("\n")


def _new_work() -> dict[str, int]:
    return {field: 0 for field in WORK_FIELDS}


def _add_work(target: dict[str, int], source: Mapping[str, int]) -> None:
    for field in WORK_FIELDS:
        target[field] += int(source[field])


def _selection_work(selection: Any) -> dict[str, int]:
    return {field: int(getattr(selection, field)) for field in WORK_FIELDS}


def _current_fixed_sources() -> dict[str, str]:
    rows = {**baseline_v5._current_fixed_sources(), **confirmation._current_fixed_sources()}
    for name in (FLOW_SOURCE, ARRIVAL_SOURCE, BASELINE_SOURCE, THIS_SOURCE):
        rows[name] = (ROOT / name).read_text(encoding="utf-8")
    return rows


DECISION_RULE = {
    INVALID_DECISION: "one or more registered execution-integrity checks fail",
    SUPPORT_FAIL_DECISION: "valid execution fails an envelope support gate",
    ACTION_FAIL_DECISION: "support passes but action identity or certified regret fails",
    COMPUTE_FAIL_DECISION: "support/action pass but charged work is not strictly lower",
    FLOW_FAIL_DECISION: "support/action/compute pass but paired mean-flow noninferiority fails",
    PASS_DECISION: "all execution, support, action, flow and charged-work gates pass",
}
CLAIM_BOUNDARY = (
    "A preregistered 600-second plus full-drain FIFO trajectory comparison on "
    "120 frozen synthetic Poisson tapes. It compares the fixed envelope-normalized "
    "candidate with a cold K4096 trajectory and audits each candidate state with an "
    "independent nonexecuted cold K4096 shadow. It does not reclassify capacity, prove "
    "stability or closure, or establish a physical multi-GPU result."
)
PROHIBITED = (
    "changes after registration to evidence, model, rates, eta, seeds, horizon, loads, "
    "methods, budgets, support trigger, reference, endpoints, accounting or gates; "
    "future-arrival access, shadow-driven execution, capacity recomputation, server "
    "pulls, and README or paper edits"
)


def _source_paths(
    confirmation_registration: Path = DEFAULT_CONFIRMATION_REGISTRATION,
    confirmation_roster: Path = DEFAULT_CONFIRMATION_ROSTER,
    confirmation_fixed_sources: Path = DEFAULT_CONFIRMATION_FIXED_SOURCES,
    confirmation_result: Path = DEFAULT_CONFIRMATION_RESULT,
    baseline_registration: Path = DEFAULT_BASELINE_REGISTRATION,
    baseline_fixed_sources: Path = DEFAULT_BASELINE_FIXED_SOURCES,
) -> dict[str, str]:
    return {
        "confirmation_registration": str(Path(confirmation_registration).resolve()),
        "confirmation_roster": str(Path(confirmation_roster).resolve()),
        "confirmation_fixed_sources": str(Path(confirmation_fixed_sources).resolve()),
        "confirmation_result": str(Path(confirmation_result).resolve()),
        "baseline_registration": str(Path(baseline_registration).resolve()),
        "baseline_fixed_sources": str(Path(baseline_fixed_sources).resolve()),
    }


def _validated_confirmation_bundle(paths: Mapping[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = _read_object(Path(paths["confirmation_registration"]))
    roster = _read_object(Path(paths["confirmation_roster"]))
    frozen = _read_object(Path(paths["confirmation_fixed_sources"]))
    result = _read_object(Path(paths["confirmation_result"]))
    try:
        confirmation._validate_protocol(protocol)
    except Exception as exc:
        raise EnvelopeTrajectoryRefusal(f"final confirmation protocol is invalid: {exc}") from exc
    if frozen != confirmation._current_fixed_sources():
        raise EnvelopeTrajectoryRefusal("final confirmation frozen source map changed")
    if roster != protocol.get("queue_roster_contract"):
        raise EnvelopeTrajectoryRefusal("final confirmation roster differs from its protocol")
    p = protocol.get("source_paths") or {}
    try:
        contract = confirmation._validated_development_contract(
            _read_object(Path(p["development_result"])),
            _read_object(Path(p["development_registration"])),
        )
        exclusions = confirmation._load_exclusion_rosters(p)
        checked_roster = confirmation._validate_roster(roster, protocol, exclusions)
    except Exception as exc:
        raise EnvelopeTrajectoryRefusal(f"final confirmation lineage is invalid: {exc}") from exc
    false_fields = (
        "retained_v5_capacity_established",
        "retained_v5_closure_passed",
        "capacity_closure_reclassified",
    )
    zero_fields = (
        "trajectory_runs", "arrival_generations",
        "rng_draws_during_execution", "server_pulls",
    )
    rows = result.get("lane_summaries") or ()
    aggregate = result.get("aggregate_actual_compute") or {}
    if (
        result.get("schema_version") != 1
        or result.get("gate") != confirmation.GATE
        or result.get("experiment") != confirmation.EXPERIMENT
        or result.get("status") != "PASS"
        or result.get("scientific_decision") != confirmation.PASS_DECISION
        or result.get("development_scientific_decision") != confirmation.SOURCE_DECISION
        or result.get("protocol") != protocol
        or result.get("independent_static_confirmation_passed") is not True
        or result.get("envelope_normalized_trajectory_registration_authorized") is not True
        or result.get("trajectory_registration_prerequisite_passed") is not True
        or any(result.get(key) is not False for key in false_fields)
        or any(result.get(key) != 0 for key in zero_fields)
        or not result.get("checks") or not all(result["checks"].values())
        or contract != protocol.get("development_contract")
        or len(checked_roster) != confirmation.ROSTER_SIZE
        or len(rows) != 2
        or any(not row.get("checks") or not all(row["checks"].values()) for row in rows)
        or any(row.get("lane_confirmation_gate_passed") is not True for row in rows)
        or aggregate.get("gate_passed") is not True
    ):
        raise EnvelopeTrajectoryRefusal("final confirmation PASS/authorization contract changed")
    source_contract = _json(result)
    source_contract["confirmation_roster_payload"] = _json(roster)
    source_contract["confirmation_fixed_source_keys"] = sorted(frozen)
    return source_contract, protocol


def _validated_baseline_seed_contract(paths: Mapping[str, str]) -> dict[str, Any]:
    baseline_path = Path(paths["baseline_registration"])
    forbidden_result = baseline_path.with_name(
        baseline_path.name.removesuffix("_protocol.json") + ".json"
    )
    if forbidden_result.exists():
        raise EnvelopeTrajectoryRefusal("baseline trajectory seeds are no longer unexecuted")
    protocol = _read_object(baseline_path)
    frozen_sources = _read_object(Path(paths["baseline_fixed_sources"]))
    scope = protocol.get("registration_scope") or {}
    expected_scope = {
        "arrival_tape_constructed": False,
        "exact_mw_invoked": False,
        "paper_edited": False,
        "rng_draws": 0,
        "rng_initializations": 0,
        "selector_invoked": False,
        "v1_score_only_static_protocol_read": False,
        "v1_score_only_static_result_read": False,
        "v4_score_only_static_protocol_read": False,
        "v4_score_only_static_result_read": False,
        "v5_canonical_protocol_read": True,
        "v5_canonical_result_read": False,
        "v5_capacity_protocol_read": True,
        "v5_capacity_result_read": False,
        "v5_results_directory_read": False,
    }
    try:
        baseline_v5._validate_protocol(protocol)
    except Exception as exc:
        raise EnvelopeTrajectoryRefusal(f"baseline seed protocol is invalid: {exc}") from exc
    trajectory = protocol.get("trajectory_contract") or {}
    if (
        protocol.get("schema_version") != 1
        or protocol.get("gate") != baseline_v5.GATE
        or protocol.get("experiment") != baseline_v5.EXPERIMENT
        or protocol.get("status") != baseline_v5.REGISTRATION_STATUS
        or protocol.get("registered") is not True
        or frozen_sources != baseline_v5._current_fixed_sources()
        or scope != expected_scope
        or trajectory.get("arrival_seeds") != list(ARRIVAL_SEEDS)
        or trajectory.get("arrival_horizon_s") != ARRIVAL_HORIZON_S
        or trajectory.get("lanes") != list(LANES)
        or trajectory.get("loads") != list(LOADS)
        or trajectory.get("case_seed_tapes") != 120
        or trajectory.get("trajectory_runs") != 480
        or protocol.get("statistics", {}).get("primary_family_size") != 18
    ):
        raise EnvelopeTrajectoryRefusal("baseline seed/horizon roster contract changed")
    return {
        "experiment": protocol["experiment"],
        "gate": protocol["gate"],
        "status": protocol["status"],
        "registration_path": str(baseline_path.resolve()),
        "arrival_seeds": list(ARRIVAL_SEEDS),
        "arrival_horizon_s": ARRIVAL_HORIZON_S,
        "lanes": list(LANES),
        "loads": list(LOADS),
        "case_seed_tapes": 120,
        "source_trajectory_contract": _json(trajectory),
        "source_registration_scope": _json(scope),
        "source_protocol": _json(protocol),
        "source_primary_family_size": 18,
        "disposition": (
            "inherit only the unused seeds, horizon, lanes and loads; supersede the "
            "old four-policy, fixed-budget and 18-primary-claim design"
        ),
        "corresponding_result_required_absent": True,
        "corresponding_result_path": str(forbidden_result.resolve()),
        "fixed_sources_path": paths["baseline_fixed_sources"],
    }


TRAJECTORY_CONTRACT = {
    "lanes": list(LANES),
    "loads": list(LOADS),
    "arrival_seeds": list(ARRIVAL_SEEDS),
    "arrival_horizon_s": ARRIVAL_HORIZON_S,
    "case_seed_tapes": 120,
    "executed_policies": ["candidate", "cold_k4096_reference"],
    "trajectory_runs": 240,
    "candidate_state_shadow": (
        "one fresh cold K4096 call on every candidate pre-dispatch state; audit only, "
        "never executed and never read by the candidate"
    ),
    "shadow_trajectory_runs": 0,
    "common_tape_rule": "one immutable tape per R/load/seed replayed by both executed arms",
    "dispatch": "global synchronous frames with FIFO service within each class",
    "online_information": "selectors receive only the released current queue, model, lanes and fixed budgets",
    "stopping_rule": "stop arrivals at 600 seconds and fully drain both executed arms",
    "candidate_by_lane": {
        "8": "cold K48 canonical B&B plus frozen single-lane exchange",
        "16": "cold K9 bulk-tie primary plus conditional fresh cold K144 rescue",
    },
    "reference": "fresh cold canonical K4096 at every executed-reference or shadow state",
    "paired_endpoint": "candidate/reference arithmetic mean-flow ratio on common tapes",
    "noninferiority_margin": NONINFERIORITY_MARGIN,
    "family_alpha": FAMILY_ALPHA,
    "primary_family_size": PRIMARY_FAMILY_SIZE,
}


REGISTRATION_SCOPE = {
    "confirmation_result_read": True,
    "confirmation_protocol_roster_and_fixed_sources_read": True,
    "baseline_seed_protocol_read": True,
    "rng_initializations": 0,
    "rng_draws": 0,
    "arrival_generations": 0,
    "selector_calls": 0,
    "trajectory_runs": 0,
    "capacity_recomputations": 0,
    "server_pulls": 0,
    "paper_or_readme_edits": 0,
}


def build_registration(**kwargs: Any) -> dict[str, Any]:
    paths = _source_paths(**kwargs)
    source_contract, source_protocol = _validated_confirmation_bundle(paths)
    baseline_contract = _validated_baseline_seed_contract(paths)
    development = source_protocol["development_contract"]
    model = development["physical_complete_frame_envelope"]["model"]
    capacity = development["physical_envelope_capacity_normalized_loads"]
    if (
        len(model.get("options") or ()) != confirmation.canonical.OPTION_COUNT
        or RESCUE_BUDGET != 16 * len(model["options"])
        or set(capacity) != {"8", "16"}
        or any([float(row["load"]) for row in capacity[str(lanes)]] != list(LOADS) for lanes in LANES)
    ):
        raise EnvelopeTrajectoryRefusal("frozen model, capacity rows or rescue budget changed")
    return {
        "schema_version": 1,
        "gate": GATE,
        "experiment": EXPERIMENT,
        "status": REGISTRATION_STATUS,
        "registered": True,
        "source_paths": paths,
        "source_confirmation_contract": source_contract,
        "baseline_seed_contract": baseline_contract,
        "physical_complete_frame_model": _json(model),
        "physical_envelope_capacity_normalized_loads": _json(capacity),
        "trajectory_contract": _json(TRAJECTORY_CONTRACT),
        "selector_contract": _json(confirmation.SELECTOR_CONTRACT),
        "work_fields": list(WORK_FIELDS),
        "work_gate_fields": list(WORK_GATE_FIELDS),
        "decision_rule": dict(DECISION_RULE),
        "claim_boundary": CLAIM_BOUNDARY,
        "prohibited": PROHIBITED,
        "registration_scope": dict(REGISTRATION_SCOPE),
        "retained_v5_capacity_established": False,
        "retained_v5_closure_passed": False,
        "capacity_closure_reclassified": False,
    }


def _validate_protocol(protocol: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    required = {"registered_at_utc", *expected.keys()}
    if set(protocol) != required or not isinstance(protocol.get("registered_at_utc"), str):
        raise EnvelopeTrajectoryRefusal("registered trajectory protocol keys changed")
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise EnvelopeTrajectoryRefusal("registered trajectory protocol changed")


def register(
    registration: Path = DEFAULT_REGISTRATION,
    fixed_sources: Path = DEFAULT_FIXED_SOURCES,
    result: Path = DEFAULT_RESULT,
    **kwargs: Any,
) -> dict[str, Any]:
    if any(Path(path).exists() for path in (registration, fixed_sources, result)):
        raise EnvelopeTrajectoryRefusal("trajectory artifact already exists")
    body = build_registration(**kwargs)
    protocol = {"registered_at_utc": datetime.now(timezone.utc).isoformat(), **body}
    _write_new(Path(fixed_sources), _current_fixed_sources())
    _write_new(Path(registration), protocol)
    return protocol


def _certificate_valid(model: Any, queue: Sequence[int], lanes: int,
                       budget: int, selection: Any, *, exact: bool = False) -> bool:
    valid = bool(
        confirmation.canonical._selection_reconstructs(model, queue, lanes, selection)
        and confirmation.canonical._selection_work_holds(queue, lanes, budget, selection)
        and confirmation.canonical._certificate_valid(selection, queue)
    )
    if exact:
        valid = bool(
            valid and selection.score_exact and selection.canonical_exact
            and math.isclose(selection.score, selection.upper_bound,
                             rel_tol=0.0, abs_tol=SCORE_TOLERANCE)
        )
    return valid


def _single_exchange_relation(result: Any, queue: Sequence[int], option_count: int) -> bool:
    base, selected = result.base_selection, result.selection
    triggered = not base.score_exact
    occupied = sum(int(value > 0) for value in base.option_counts)
    occupied += int(sum(base.option_counts) < 8)
    expected_attempts = occupied * option_count if triggered else 0
    gap = max(0.0, selected.upper_bound - selected.score)
    epsilon = gap / sum(queue) if sum(queue) else 0.0
    expected_work = _selection_work(base)
    expected_work["candidate_evaluations"] += result.exchange_attempts
    expected_work["residual_fill_option_probes"] += result.exchange_attempts
    return bool(
        result.polish_triggered is triggered
        and result.exchange_attempts == expected_attempts
        and result.feasible_exchange_count <= result.exchange_attempts
        and result.score_improving_exchange_count <= result.feasible_exchange_count
        and result.canonical_tie_improving_exchange_count <= result.feasible_exchange_count
        and result.incumbent_changed == (selected.option_indices != base.option_indices)
        and selected.upper_bound == base.upper_bound
        and selected.score_exact == base.score_exact
        and selected.canonical_exact == base.canonical_exact
        and all(getattr(selected, field) == getattr(base, field) for field in WORK_FIELDS)
        and selected.score + SCORE_TOLERANCE >= base.score
        and math.isclose(selected.gap, gap, rel_tol=0.0, abs_tol=SCORE_TOLERANCE)
        and math.isclose(selected.epsilon, epsilon, rel_tol=0.0, abs_tol=SCORE_TOLERANCE)
        and single_exchange_total_work_row(result) == expected_work
    )


def _bulk_relation(result: Any, queue: Sequence[int], options: Sequence[Any]) -> bool:
    base, selected = result.base_selection, result.selection
    pairs = tuple(
        (source, destination)
        for source in range(len(options))
        for destination in range(source)
        if sum(options[source].requirements) == sum(options[destination].requirements)
    )
    triggered = bool(base.score_exact and not base.canonical_exact)
    expected_checks = len(pairs) if triggered else 0
    expected_attempts = (
        sum(int(base.option_counts[source] > 0) for source, _ in pairs)
        if triggered else 0
    )
    gap = max(0.0, selected.upper_bound - selected.score)
    epsilon = gap / sum(queue) if sum(queue) else 0.0
    expected_work = _selection_work(base)
    expected_work["candidate_evaluations"] += result.pair_checks
    expected_work["residual_fill_option_probes"] += result.pair_checks
    return bool(
        result.polish_triggered is triggered
        and result.pair_checks == expected_checks
        and result.bulk_candidate_attempts == expected_attempts
        and result.feasible_bulk_candidate_count <= result.bulk_candidate_attempts
        and result.accepted_bulk_substitution_count == int(selected.option_indices != base.option_indices)
        and result.incumbent_changed == (selected.option_indices != base.option_indices)
        and selected.upper_bound == base.upper_bound
        and selected.score_exact == base.score_exact
        and selected.canonical_exact == base.canonical_exact
        and all(getattr(selected, field) == getattr(base, field) for field in WORK_FIELDS)
        and selected.score + SCORE_TOLERANCE >= base.score
        and math.isclose(selected.gap, gap, rel_tol=0.0, abs_tol=SCORE_TOLERANCE)
        and math.isclose(selected.epsilon, epsilon, rel_tol=0.0, abs_tol=SCORE_TOLERANCE)
        and bulk_total_work_row(result) == expected_work
    )


def _candidate_call(model: Any, queue: Sequence[int]) -> dict[str, Any]:
    lanes = int(model.lanes)
    if lanes == 8:
        composite = select_canonical_count_space_bnb_single_exchange_frame(
            model.options, queue, lanes, CANDIDATE_BUDGETS[lanes]
        )
        selected = composite.selection
        primary_work = _selection_work(composite.base_selection)
        rescue_work = _new_work()
        total_work = single_exchange_total_work_row(composite)
        relation = _single_exchange_relation(composite, queue, len(model.options))
        valid = bool(
            relation
            and _certificate_valid(model, queue, lanes, CANDIDATE_BUDGETS[lanes], composite.base_selection)
            and confirmation.canonical._selection_reconstructs(model, queue, lanes, selected)
            and confirmation.canonical._certificate_valid(selected, queue)
        )
        return {
            "selection": selected, "valid": valid, "method_relation": relation,
            "primary_work": primary_work, "rescue_work": rescue_work,
            "work": total_work, "rescue_triggered": False,
            "rescue_exact": True, "exchange_triggered": composite.polish_triggered,
            "exchange_attempts": composite.exchange_attempts,
            "primary_bulk_triggered": False, "primary_pair_checks": 0,
            "rescue_bulk_triggered": False, "rescue_pair_checks": 0,
        }
    if lanes != 16:
        raise EnvelopeTrajectoryRefusal("only R8 and R16 are registered")
    composite = select_canonical_support_rescue_bulk_tie_frame(
        model.options, queue, lanes, CANDIDATE_BUDGETS[lanes], RESCUE_BUDGET,
        SUPPORT_EPSILON_LIMIT,
    )
    primary = composite.primary_result
    rescue = composite.rescue_result
    selected = composite.selection
    expected_trigger = SUPPORT_EPSILON_LIMIT - primary.selection.epsilon <= SCORE_TOLERANCE
    primary_work = bulk_total_work_row(primary)
    rescue_work = bulk_total_work_row(rescue) if rescue is not None else _new_work()
    total_work = rescue_total_work_row(composite)
    relation = bool(
        _bulk_relation(primary, queue, model.options)
        and composite.rescue_triggered is expected_trigger
        and ((rescue is None and not expected_trigger and selected == primary.selection)
             or (rescue is not None and expected_trigger and selected == rescue.selection
                 and _bulk_relation(rescue, queue, model.options)
                 and rescue.selection.score_exact and rescue.selection.canonical_exact
                 and rescue.selection.score + SCORE_TOLERANCE >= primary.selection.score
                 and rescue.selection.upper_bound <= primary.selection.upper_bound + SCORE_TOLERANCE))
        and total_work == {field: primary_work[field] + rescue_work[field] for field in WORK_FIELDS}
    )
    valid = bool(
        relation
        and _certificate_valid(model, queue, lanes, CANDIDATE_BUDGETS[lanes], primary.base_selection)
        and confirmation.canonical._selection_reconstructs(model, queue, lanes, primary.selection)
        and confirmation.canonical._certificate_valid(primary.selection, queue)
        and (rescue is None or (
            _certificate_valid(model, queue, lanes, RESCUE_BUDGET, rescue.base_selection)
            and confirmation.canonical._selection_reconstructs(model, queue, lanes, rescue.selection)
            and confirmation.canonical._certificate_valid(rescue.selection, queue)
        ))
    )
    return {
        "selection": selected, "valid": valid, "method_relation": relation,
        "primary_work": primary_work, "rescue_work": rescue_work,
        "work": total_work, "rescue_triggered": composite.rescue_triggered,
        "rescue_exact": rescue is None or (rescue.selection.score_exact and rescue.selection.canonical_exact),
        "exchange_triggered": False, "exchange_attempts": 0,
        "primary_bulk_triggered": primary.polish_triggered,
        "primary_pair_checks": primary.pair_checks,
        "rescue_bulk_triggered": bool(rescue and rescue.polish_triggered),
        "rescue_pair_checks": rescue.pair_checks if rescue else 0,
    }


def _select_candidate(model: Any, queue: Sequence[int]) -> dict[str, Any]:
    """Stable trajectory hook for one frozen candidate call."""
    return _candidate_call(model, queue)


def _reference_call(model: Any, queue: Sequence[int]) -> tuple[Any, dict[str, int], bool]:
    selected = select_canonical_count_space_bnb_frame(
        model.options, queue, model.lanes, REFERENCE_BUDGET
    )
    return selected, _selection_work(selected), _certificate_valid(
        model, queue, model.lanes, REFERENCE_BUDGET, selected, exact=True
    )


def _compact_decomposition(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": row["status"], "job_count": row["job_count"],
        "completed_jobs": row["completed_jobs"], "mean_flow_s": row["mean_flow_s"],
        "flow_area_job_seconds": row["flow_area_job_seconds"],
        "flow_identity_error_job_seconds": row["flow_identity_error_job_seconds"],
        "mean_deferred_visible_wait_s": row["mean_deferred_visible_wait_s"],
        "mean_arrival_barrier_wait_s": row["mean_arrival_barrier_wait_s"],
        "mean_dispatch_to_completion_s": row["mean_dispatch_to_completion_s"],
        "arrival_window_mean_flow_contribution_s": row["time_regions"]["arrival_window"]["mean_flow_contribution_s"],
        "drain_mean_flow_contribution_s": row["time_regions"]["drain"]["mean_flow_contribution_s"],
    }


def simulate_trajectory(model: Any, arrivals: Sequence[tuple[float, int, int]],
                        horizon: float, policy: str, *, audit_limit: int = 3) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if policy not in {"candidate", "cold_k4096_reference"}:
        raise ValueError("unknown trajectory policy")
    queues = [deque() for _ in model.workloads]
    tape = tuple(sorted(arrivals))
    pending = iter(tape)
    next_arrival = next(pending, None)
    now = 0.0
    frames: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    flows: list[float] = []
    last_completion = 0.0
    peak_backlog = backlog_sum = recursion_violation = 0
    valid_frames = exact_frames = action_matches = bounded_regrets = 0
    candidate_valid_frames = shadow_valid_frames = shadow_exact_frames = 0
    method_relations = rescue_triggers = exact_rescues = 0
    exchange_triggers = exchange_attempts = 0
    primary_bulk_triggers = primary_pair_checks = 0
    rescue_bulk_triggers = rescue_pair_checks = 0
    max_epsilon = max_regret = max_gap = 0.0
    candidate_work, primary_work, rescue_work = _new_work(), _new_work(), _new_work()
    reference_work, shadow_work = _new_work(), _new_work()
    selected_counts: Counter[str] = Counter()
    selector_wall = shadow_wall = 0.0
    first_mismatch = first_regret_violation = None

    while now < horizon or next_arrival is not None or any(queues):
        while next_arrival is not None and next_arrival[0] <= now:
            at, coordinate, index = next_arrival
            queues[coordinate].append((at, index))
            next_arrival = next(pending, None)
        before = tuple(len(row) for row in queues)
        started = time.perf_counter()
        if policy == "candidate":
            call = _select_candidate(model, before)
            selected = call["selection"]
            selector_wall += time.perf_counter() - started
            _add_work(candidate_work, call["work"])
            _add_work(primary_work, call["primary_work"])
            _add_work(rescue_work, call["rescue_work"])
            candidate_valid = bool(call["valid"])
            candidate_valid_frames += int(candidate_valid)
            method_relations += int(call["method_relation"])
            rescue_triggers += int(call["rescue_triggered"])
            exact_rescues += int(call["rescue_triggered"] and call["rescue_exact"])
            exchange_triggers += int(call["exchange_triggered"])
            exchange_attempts += int(call["exchange_attempts"])
            primary_bulk_triggers += int(call["primary_bulk_triggered"])
            primary_pair_checks += int(call["primary_pair_checks"])
            rescue_bulk_triggers += int(call["rescue_bulk_triggered"])
            rescue_pair_checks += int(call["rescue_pair_checks"])
            shadow_started = time.perf_counter()
            shadow, work, shadow_valid = _reference_call(model, before)
            shadow_wall += time.perf_counter() - shadow_started
            _add_work(shadow_work, work)
            shadow_valid_frames += int(shadow_valid)
            shadow_is_exact = bool(shadow.score_exact and shadow.canonical_exact)
            shadow_exact_frames += int(shadow_is_exact)
            valid = bool(candidate_valid and shadow_valid)
            norm = sum(before)
            regret = (shadow.score - selected.score) / norm if norm else 0.0
            bounded = -SCORE_TOLERANCE <= regret <= selected.epsilon + SCORE_TOLERANCE
            matched = selected.option_indices == shadow.option_indices
            action_matches += int(matched)
            bounded_regrets += int(bounded)
            max_regret = max(max_regret, regret)
            if not matched and first_mismatch is None:
                first_mismatch = {"frame": len(frames), "queue": list(before),
                                  "candidate": list(selected.option_indices),
                                  "shadow": list(shadow.option_indices), "regret": regret}
            if not bounded and first_regret_violation is None:
                first_regret_violation = {"frame": len(frames), "queue": list(before),
                                          "epsilon": selected.epsilon, "regret": regret}
        else:
            selected, work, valid = _reference_call(model, before)
            selector_wall += time.perf_counter() - started
            _add_work(reference_work, work)
            shadow = None
            shadow_is_exact = False
            regret = 0.0
            matched = bounded = True

        valid_frames += int(valid)
        exact_frames += int(selected.score_exact and selected.canonical_exact)
        max_epsilon = max(max_epsilon, selected.epsilon)
        max_gap = max(max_gap, selected.gap)
        peak_backlog = max(peak_backlog, sum(before))
        backlog_sum += sum(before)
        end = now + selected.duration
        departures = [0] * len(queues)
        for option_index in selected.option_indices:
            if option_index is None:
                selected_counts["idle"] += 1
                continue
            option = model.options[option_index]
            selected_counts[option.action_id] += 1
            for coordinate, offsets in enumerate(option.completion_offsets):
                for offset in offsets:
                    arrival_time, _ = queues[coordinate].popleft()
                    completion = now + offset
                    flows.append(completion - arrival_time)
                    last_completion = max(last_completion, completion)
                    departures[coordinate] += 1
        if tuple(departures) != selected.service:
            raise EnvelopeTrajectoryRefusal("selected service differs from FIFO departures")
        during = [0] * len(queues)
        while next_arrival is not None and next_arrival[0] <= end:
            at, coordinate, index = next_arrival
            queues[coordinate].append((at, index))
            during[coordinate] += 1
            next_arrival = next(pending, None)
        after = tuple(len(row) for row in queues)
        recursion_violation = max(recursion_violation, max(
            abs(q1 - (q0 - done + added))
            for q0, done, added, q1 in zip(before, departures, during, after, strict=True)
        ))
        frame = {"start_s": now, "duration_s": selected.duration,
                 "queue_before": before, "departures": tuple(departures),
                 "option_indices": selected.option_indices}
        frames.append(frame)
        if len(audits) < audit_limit:
            audits.append({**frame, "arrivals_during_frame": tuple(during),
                           "queue_after": after, "score": selected.score,
                           "upper_bound": selected.upper_bound, "epsilon": selected.epsilon,
                           "score_exact": selected.score_exact,
                           "canonical_exact": selected.canonical_exact,
                           "same_state_shadow_score": shadow.score if shadow else None,
                           "same_state_shadow_score_exact": shadow.score_exact if shadow else None,
                           "same_state_shadow_canonical_exact": shadow.canonical_exact if shadow else None,
                           "same_state_shadow_action_match": matched if shadow else None,
                           "actual_normalized_regret": regret if shadow else None})
        now = end

    ordered = sorted(flows)
    frame_count = len(frames)
    metrics = {
        "policy": policy, "arrival_horizon_s": horizon,
        "elapsed_including_drain_s": now, "job_count": len(tape),
        "completed_jobs": len(flows), "makespan_s": last_completion,
        "mean_flow_s": mean(flows) if flows else None,
        "p90_flow_s": ordered[math.ceil(0.9 * len(ordered)) - 1] if ordered else None,
        "frame_count": frame_count, "valid_selection_frame_count": valid_frames,
        "exact_selection_frame_count": exact_frames,
        "method_relation_frame_count": method_relations if policy == "candidate" else None,
        "same_state_shadow_calls": frame_count if policy == "candidate" else 0,
        "candidate_valid_selection_frame_count": candidate_valid_frames if policy == "candidate" else None,
        "same_state_shadow_valid_count": shadow_valid_frames if policy == "candidate" else None,
        "same_state_shadow_exact_count": shadow_exact_frames if policy == "candidate" else None,
        "same_state_action_match_count": action_matches if policy == "candidate" else None,
        "regret_bounded_by_epsilon_count": bounded_regrets if policy == "candidate" else None,
        "maximum_certified_gap_per_job": max_epsilon,
        "maximum_actual_normalized_regret": max_regret if policy == "candidate" else None,
        "maximum_certified_gap": max_gap,
        "maximum_queue_recursion_violation": recursion_violation,
        "frame_boundary_peak_backlog": peak_backlog,
        "frame_boundary_mean_backlog": backlog_sum / frame_count,
        "work": candidate_work if policy == "candidate" else reference_work,
        "primary_work": primary_work if policy == "candidate" else None,
        "rescue_work": rescue_work if policy == "candidate" else None,
        "same_state_shadow_work": shadow_work if policy == "candidate" else None,
        "rescue_trigger_frame_count": rescue_triggers,
        "exact_rescue_frame_count": exact_rescues,
        "single_exchange_trigger_frame_count": exchange_triggers,
        "single_exchange_attempts": exchange_attempts,
        "primary_bulk_trigger_frame_count": primary_bulk_triggers,
        "primary_bulk_pair_checks": primary_pair_checks,
        "rescue_bulk_trigger_frame_count": rescue_bulk_triggers,
        "rescue_bulk_pair_checks": rescue_pair_checks,
        "first_action_mismatch": first_mismatch,
        "first_regret_bound_violation": first_regret_violation,
        "selected_option_counts": dict(selected_counts),
        "selector_wall_s": selector_wall, "shadow_selector_wall_s": shadow_wall,
        "frame_audit_prefix": audits,
    }
    return metrics, frames


def _run_tape(task: tuple[Any, Mapping[str, Any], int, float, int, float]) -> dict[str, Any]:
    model, capacity, lanes, load, seed, horizon = task
    lane_model = replace(model, lanes=lanes)
    tape = tuple(poisson_arrivals(capacity["arrival_rates"], horizon, seed))
    policies = []
    for policy in ("candidate", "cold_k4096_reference"):
        metrics, frames = simulate_trajectory(lane_model, tape, horizon, policy)
        decomposition = decompose_flow(lane_model, tape, frames, horizon)
        policies.append({
            "policy": policy,
            "metrics": metrics,
            "flow_decomposition": _compact_decomposition(decomposition),
        })
    jobs_by_class = [0] * len(model.workloads)
    for _, coordinate, _ in tape:
        jobs_by_class[coordinate] += 1
    return {
        "lanes": lanes, "load": load, "seed": seed,
        "arrival_tape": {
            "job_count": len(tape), "jobs_by_class": jobs_by_class,
            "first_arrival_s": tape[0][0] if tape else None,
            "last_arrival_s": tape[-1][0] if tape else None,
            "arrival_generation_count": 1, "executed_policy_count": 2,
            "shadow_trajectory_runs": 0,
        },
        "policies": policies,
    }


def _work_comparison(candidate: Mapping[str, int], reference: Mapping[str, int]) -> dict[str, dict[str, Any]]:
    return {
        field: {
            "candidate": int(candidate[field]),
            "same_state_cold_k4096": int(reference[field]),
            "candidate_fraction_of_shadow": (
                candidate[field] / reference[field] if reference[field] else None
            ),
            "strictly_lower": candidate[field] < reference[field],
            "scientific_gate_field": field in WORK_GATE_FIELDS,
        }
        for field in WORK_FIELDS
    }


def _sum_run_work(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
    candidate, shadow = _new_work(), _new_work()
    for row in rows:
        policies = {item["policy"]: item for item in row["policies"]}
        metrics = policies["candidate"]["metrics"]
        _add_work(candidate, metrics["work"])
        _add_work(shadow, metrics["same_state_shadow_work"])
    return candidate, shadow


def _paired_case_summary(runs: Sequence[Mapping[str, Any]], lanes: int,
                         load: float, eta: float) -> dict[str, Any]:
    members = sorted(
        (row for row in runs if row["lanes"] == lanes and row["load"] == load),
        key=lambda row: row["seed"],
    )
    if [row["seed"] for row in members] != list(ARRIVAL_SEEDS):
        raise EnvelopeTrajectoryRefusal("case does not contain the registered seed roster")
    ratios: list[float] = []
    seed_rows = []
    jobs_complete = certificates_valid = flow_identities = True
    shadows_exact = method_relations = regret_bounded = action_identity = True
    rescues_exact = True
    max_epsilon = max_regret = 0.0
    for row in members:
        policies = {item["policy"]: item for item in row["policies"]}
        candidate, reference = policies["candidate"], policies["cold_k4096_reference"]
        cm, rm = candidate["metrics"], reference["metrics"]
        if cm["mean_flow_s"] is None or rm["mean_flow_s"] in (None, 0):
            raise EnvelopeTrajectoryRefusal("paired mean-flow ratio is undefined")
        ratio = cm["mean_flow_s"] / rm["mean_flow_s"]
        ratios.append(ratio)
        max_epsilon = max(max_epsilon, cm["maximum_certified_gap_per_job"])
        max_regret = max(max_regret, cm["maximum_actual_normalized_regret"])
        jobs_complete &= bool(
            cm["job_count"] == cm["completed_jobs"] == rm["job_count"]
            and rm["job_count"] == rm["completed_jobs"] == row["arrival_tape"]["job_count"]
        )
        certificates_valid &= bool(
            cm["valid_selection_frame_count"] == cm["frame_count"]
            and rm["valid_selection_frame_count"] == rm["frame_count"]
            and rm["exact_selection_frame_count"] == rm["frame_count"]
        )
        shadows_exact &= bool(
            cm["same_state_shadow_valid_count"] == cm["frame_count"]
            and cm["same_state_shadow_exact_count"] == cm["frame_count"]
        )
        method_relations &= cm["method_relation_frame_count"] == cm["frame_count"]
        regret_bounded &= cm["regret_bounded_by_epsilon_count"] == cm["frame_count"]
        action_identity &= cm["same_state_action_match_count"] == cm["frame_count"]
        rescues_exact &= cm["rescue_trigger_frame_count"] == cm["exact_rescue_frame_count"]
        flow_identities &= all(
            item["flow_decomposition"]["status"] == "PASS"
            and abs(item["flow_decomposition"]["mean_flow_s"] - item["metrics"]["mean_flow_s"])
            <= EXECUTION_TOLERANCE
            for item in policies.values()
        )
        seed_rows.append({
            "seed": row["seed"], "job_count": cm["job_count"],
            "candidate_mean_flow_s": cm["mean_flow_s"],
            "reference_mean_flow_s": rm["mean_flow_s"],
            "candidate_over_reference_mean_flow_ratio": ratio,
            "candidate_frame_count": cm["frame_count"],
            "reference_frame_count": rm["frame_count"],
            "candidate_maximum_epsilon": cm["maximum_certified_gap_per_job"],
            "candidate_maximum_actual_regret": cm["maximum_actual_normalized_regret"],
        })
    estimate = mean(ratios)
    standard_error = stdev(ratios) / math.sqrt(len(ratios))
    per_case_alpha = FAMILY_ALPHA / PRIMARY_FAMILY_SIZE
    critical = float(student_t.ppf(1.0 - per_case_alpha, df=len(ratios) - 1))
    upper = estimate + critical * standard_error
    flow_pass = upper <= 1.0 + NONINFERIORITY_MARGIN + SCORE_TOLERANCE
    support_pass = bool(
        eta - max_epsilon > SCORE_TOLERANCE
        and eta - max_regret > SCORE_TOLERANCE
    )
    action_pass = bool(action_identity and regret_bounded)
    case_candidate, case_shadow = _sum_run_work(members)
    comparison = _work_comparison(case_candidate, case_shadow)
    compute_pass = all(comparison[field]["strictly_lower"] for field in WORK_GATE_FIELDS)
    execution_pass = bool(
        jobs_complete and certificates_valid and flow_identities and shadows_exact
        and method_relations and rescues_exact
    )
    return {
        "lanes": lanes, "load": load,
        "candidate_primary_budget": CANDIDATE_BUDGETS[lanes],
        "conditional_rescue_budget": RESCUE_BUDGET if lanes == 16 else None,
        "cold_reference_budget": REFERENCE_BUDGET,
        "seeds": list(ARRIVAL_SEEDS), "seed_count": len(members),
        "paired_seed_rows": seed_rows, "paired_mean_flow_ratios": ratios,
        "paired_ratio_mean": estimate, "paired_ratio_standard_error": standard_error,
        "one_sided_confidence_level": 1.0 - per_case_alpha,
        "critical_t": critical, "paired_ratio_upper_confidence_bound": upper,
        "noninferiority_acceptance_upper": 1.0 + NONINFERIORITY_MARGIN,
        "paired_mean_flow_noninferiority_passed": flow_pass,
        "effective_slack_eta": eta,
        "maximum_candidate_epsilon": max_epsilon,
        "maximum_actual_normalized_regret": max_regret,
        "eta_minus_candidate_epsilon": eta - max_epsilon,
        "eta_minus_actual_regret": eta - max_regret,
        "trajectory_support_gate_passed": support_pass,
        "same_state_action_and_regret_gate_passed": action_pass,
        "charged_candidate_work": case_candidate,
        "same_state_cold_k4096_shadow_work": case_shadow,
        "charged_work_comparison": comparison,
        "case_compute_gate_passed": compute_pass,
        "execution_integrity_passed": execution_pass,
        "case_gate_passed": execution_pass and support_pass and action_pass and flow_pass and compute_pass,
    }


def _aggregate_work_gates(runs: Sequence[Mapping[str, Any]],
                          cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lane_rows = []
    for lanes in LANES:
        members = [row for row in runs if row["lanes"] == lanes]
        candidate, shadow = _sum_run_work(members)
        comparison = _work_comparison(candidate, shadow)
        lane_rows.append({
            "lanes": lanes, "candidate": candidate,
            "same_state_cold_k4096_shadow": shadow, "comparison": comparison,
            "gate_passed": all(comparison[field]["strictly_lower"] for field in WORK_GATE_FIELDS),
        })
    candidate, shadow = _sum_run_work(runs)
    comparison = _work_comparison(candidate, shadow)
    aggregate_pass = all(comparison[field]["strictly_lower"] for field in WORK_GATE_FIELDS)
    return {
        "case_gate_passed": all(row["case_compute_gate_passed"] for row in cases),
        "by_lane": lane_rows,
        "aggregate": {"candidate": candidate, "same_state_cold_k4096_shadow": shadow,
                      "comparison": comparison, "gate_passed": aggregate_pass},
        "gate_passed": bool(
            aggregate_pass
            and all(row["gate_passed"] for row in lane_rows)
            and all(row["case_compute_gate_passed"] for row in cases)
        ),
    }


def _decision(execution_passed: bool, cases: Sequence[Mapping[str, Any]],
              compute_passed: bool) -> str:
    if not execution_passed:
        return INVALID_DECISION
    if not all(row["trajectory_support_gate_passed"] for row in cases):
        return SUPPORT_FAIL_DECISION
    if not all(row["same_state_action_and_regret_gate_passed"] for row in cases):
        return ACTION_FAIL_DECISION
    if not compute_passed:
        return COMPUTE_FAIL_DECISION
    if not all(row["paired_mean_flow_noninferiority_passed"] for row in cases):
        return FLOW_FAIL_DECISION
    return PASS_DECISION


def run_registered(registration: Path = DEFAULT_REGISTRATION,
                   fixed_sources: Path = DEFAULT_FIXED_SOURCES,
                   output: Path = DEFAULT_RESULT) -> dict[str, Any]:
    if Path(output).exists():
        raise EnvelopeTrajectoryRefusal(f"trajectory result already exists: {output}")
    protocol = _read_object(Path(registration))
    paths = protocol.get("source_paths") or {}
    if set(paths) != {
        "confirmation_registration", "confirmation_roster",
        "confirmation_fixed_sources", "confirmation_result",
        "baseline_registration", "baseline_fixed_sources",
    }:
        raise EnvelopeTrajectoryRefusal("registered source paths changed")
    # Complete preflight before the first model/RNG/arrival/selector operation.
    source_contract, source_protocol = _validated_confirmation_bundle(paths)
    baseline_contract = _validated_baseline_seed_contract(paths)
    expected = build_registration(
        confirmation_registration=Path(paths["confirmation_registration"]),
        confirmation_roster=Path(paths["confirmation_roster"]),
        confirmation_fixed_sources=Path(paths["confirmation_fixed_sources"]),
        confirmation_result=Path(paths["confirmation_result"]),
        baseline_registration=Path(paths["baseline_registration"]),
        baseline_fixed_sources=Path(paths["baseline_fixed_sources"]),
    )
    _validate_protocol(protocol, expected)
    if source_contract != protocol["source_confirmation_contract"]:
        raise EnvelopeTrajectoryRefusal("registered confirmation contract changed")
    if baseline_contract != protocol["baseline_seed_contract"]:
        raise EnvelopeTrajectoryRefusal("registered baseline seed contract changed")
    registered_sources = _read_object(Path(fixed_sources))
    if registered_sources != _current_fixed_sources():
        raise EnvelopeTrajectoryRefusal("registered trajectory source map changed")

    model = baseline._model_from_snapshot(protocol["physical_complete_frame_model"])
    if _json(model.snapshot()) != protocol["physical_complete_frame_model"]:
        raise EnvelopeTrajectoryRefusal("frozen physical model failed exact reconstruction")
    capacity = protocol["physical_envelope_capacity_normalized_loads"]
    by_case: dict[tuple[int, float], Mapping[str, Any]] = {}
    for lanes in LANES:
        for row in capacity[str(lanes)]:
            key = (lanes, float(row["load"]))
            if (
                int(row["lanes"]) != lanes
                or row.get("capacity_projection_passed") is not True
                or row.get("conditional_negative_drift_certified") is not True
                or row.get("positive_effective_slack") is not True
            ):
                raise EnvelopeTrajectoryRefusal("frozen capacity row changed")
            by_case[key] = row
    if set(by_case) != {(lanes, load) for lanes in LANES for load in LOADS}:
        raise EnvelopeTrajectoryRefusal("frozen R/load capacity grid changed")
    tasks = [
        (model, by_case[(lanes, load)], lanes, load, seed, ARRIVAL_HORIZON_S)
        for lanes in LANES for load in LOADS for seed in ARRIVAL_SEEDS
    ]
    started = time.perf_counter()
    runs = []
    with ProcessPoolExecutor(max_workers=EXECUTION_WORKERS) as pool:
        runs.extend(pool.map(_run_tape, tasks))

    # A long run is publishable only if the exact registered evidence and code
    # still validate when execution finishes.  Refuse before writing a result
    # if any source artifact changed while the workers were active.
    post_expected = build_registration(
        confirmation_registration=Path(paths["confirmation_registration"]),
        confirmation_roster=Path(paths["confirmation_roster"]),
        confirmation_fixed_sources=Path(paths["confirmation_fixed_sources"]),
        confirmation_result=Path(paths["confirmation_result"]),
        baseline_registration=Path(paths["baseline_registration"]),
        baseline_fixed_sources=Path(paths["baseline_fixed_sources"]),
    )
    _validate_protocol(protocol, post_expected)
    if (
        _read_object(Path(fixed_sources)) != registered_sources
        or registered_sources != _current_fixed_sources()
    ):
        raise EnvelopeTrajectoryRefusal(
            "registered trajectory sources changed during execution"
        )
    cases = [
        _paired_case_summary(runs, lanes, load,
                             float(by_case[(lanes, load)]["effective_slack_eta"]))
        for lanes in LANES for load in LOADS
    ]
    work_gates = _aggregate_work_gates(runs, cases)
    expected_tapes = len(tasks)
    policy_rows = [policy for run in runs for policy in run["policies"]]
    candidate_rows = [row for row in policy_rows if row["policy"] == "candidate"]
    reference_rows = [row for row in policy_rows if row["policy"] == "cold_k4096_reference"]
    total_candidate_frames = sum(row["metrics"]["frame_count"] for row in candidate_rows)
    total_reference_frames = sum(row["metrics"]["frame_count"] for row in reference_rows)
    checks = {
        "source_confirmation_and_unused_baseline_seed_contracts_unchanged": True,
        "protocol_and_fixed_sources_revalidated_after_execution": True,
        "arrival_tape_count_is_registered_120": len(runs) == expected_tapes == 120,
        "executed_trajectory_count_is_registered_240": len(policy_rows) == 240,
        "one_arrival_generation_per_common_tape": sum(run["arrival_tape"]["arrival_generation_count"] for run in runs) == 120,
        "same_state_shadow_never_executes_a_trajectory": all(run["arrival_tape"]["shadow_trajectory_runs"] == 0 for run in runs),
        "every_candidate_frame_has_one_fresh_shadow": sum(row["metrics"]["same_state_shadow_calls"] for row in candidate_rows) == total_candidate_frames,
        "every_policy_fully_drains_and_flow_decomposes": all(row["flow_decomposition"]["status"] == "PASS" for row in policy_rows),
        "every_case_execution_integrity_passes": all(row["execution_integrity_passed"] for row in cases),
        "no_capacity_recomputation_or_reclassification": True,
    }
    execution_passed = all(checks.values())
    decision = _decision(execution_passed, cases, work_gates["gate_passed"])
    report = {
        "schema_version": 1, "gate": GATE, "experiment": EXPERIMENT,
        "status": "PASS" if execution_passed else "FAIL", "protocol": protocol,
        "checks": checks, "case_summaries": cases, "runs": runs,
        "work_gates": work_gates,
        "work_accounting": {
            "arrival_tapes": len(runs), "arrival_generations": len(runs),
            "executed_trajectory_runs": len(policy_rows), "candidate_trajectory_runs": len(candidate_rows),
            "cold_reference_trajectory_runs": len(reference_rows), "shadow_trajectory_runs": 0,
            "candidate_selector_calls": total_candidate_frames,
            "same_state_shadow_selector_calls": total_candidate_frames,
            "executed_reference_selector_calls": total_reference_frames,
            "ordered_action_evaluations": 0, "count_vector_enumerations": 0,
            "server_pulls": 0,
        },
        "scientific_decision": decision,
        "trajectory_transition_passed": decision == PASS_DECISION,
        "development_advance": decision == PASS_DECISION,
        "trajectory_runs": len(policy_rows),
        "arrival_generations": len(runs),
        "shadow_trajectory_runs": 0,
        "server_pulls": 0,
        "retained_v5_capacity_established": False,
        "retained_v5_closure_passed": False,
        "capacity_closure_reclassified": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "timing": {"wall_s": time.perf_counter() - started, "scientific_gate": False},
    }
    _write_new(Path(output), report)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--register", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    parser.add_argument("--fixed-sources", type=Path, default=DEFAULT_FIXED_SOURCES)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULT)
    args = parser.parse_args(argv)
    try:
        payload = (register(args.registration, args.fixed_sources, args.output)
                   if args.register else run_registered(args.registration, args.fixed_sources, args.output))
    except EnvelopeTrajectoryRefusal as exc:
        raise SystemExit(f"REFUSED: {exc}") from exc
    print(json.dumps({"status": payload["status"], "experiment": payload["experiment"],
                      "scientific_decision": payload.get("scientific_decision")}, indent=2))


if __name__ == "__main__":
    main()
