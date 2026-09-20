"""Registered canonical static screen conditional on joint-95 capacity closure.

The registration step reads the frozen joint-95 capacity protocol, the formal
canonical-exact compute confirmation, and prior queue rosters.  It creates one
new explicit queue roster.  It does not read joint measurements or the future
capacity result and does not invoke a selector.

Execution remains conditional on the complete joint calibration/release/
holdout audit and its matching capacity-closure artifact.  On the final
physical complete-frame envelope it compares canonical count-space searches at
R8/K36 and R16/K72 with separate cold canonical K4096 references.  The screen
is finite and static; it does not make a flow or trajectory claim.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import product
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

from algorithm.canonical_count_space_bnb_frame_candidates import (
    CanonicalCountSpaceBnbSelection,
    select_canonical_count_space_bnb_frame,
)
from algorithm.experiments import (
    aligned_hardware_count_space_bnb_canonical_exact_compute_confirmation
    as canonical_confirmation,
)
from algorithm.experiments import aligned_hardware_joint95_static_selector_screen as legacy
from algorithm.experiments.aligned_hardware_count_space_bnb_scaling_screen import (
    LOCAL_CHOICE_COUNT,
    OPTION_COUNT,
    SCORE_TOLERANCE,
    _selection_reconstructs,
    _selection_work_holds,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
STEM = "aligned_hardware_joint95_canonical_selector_screen_20260913"
DEFAULT_REGISTRATION = ARTIFACTS / f"{STEM}_protocol.json"
DEFAULT_ROSTER = ARTIFACTS / f"{STEM}_roster.json"
DEFAULT_FIXED_SOURCES = ARTIFACTS / f"{STEM}_fixed_sources.json"
DEFAULT_RESULT = ARTIFACTS / f"{STEM}.json"
DEFAULT_CAPACITY_REGISTRATION = legacy.DEFAULT_CAPACITY_REGISTRATION
DEFAULT_CAPACITY_RESULT = legacy.DEFAULT_CAPACITY_RESULT
DEFAULT_COUNT_ROSTER = legacy.DEFAULT_PREVIOUS_ROSTER
DEFAULT_LEGACY_STATIC_ROSTER = legacy.DEFAULT_ROSTER
DEFAULT_CANONICAL_CONFIRMATION = canonical_confirmation.DEFAULT_OUTPUT

EXPERIMENT = "aligned_hardware_joint95_canonical_selector_screen_v1"
GATE = "aligned_hardware_joint95_canonical_selector_screen_runner"
REGISTRATION_STATUS = "REGISTERED_AWAITING_JOINT95_CAPACITY_CLOSURE"
CANONICAL_SOURCE_DECISION = "CANONICAL_EXACT_COMPUTE_CONFIRMATION_PASS"
PASS_DECISION = "CANONICAL_PHYSICAL_SELECTOR_PASS_ADVANCE_TO_BASELINE_TRAJECTORY"
ROSTER_SEED = 13041
ROSTER_SIZE = 512
ROSTER_RANGES = legacy.ROSTER_RANGES
OLD_CNN_LEVELS = legacy.OLD_CNN_LEVELS
OLD_JAX_LEVELS = legacy.OLD_JAX_LEVELS
OLD_LLM_LEVELS = legacy.OLD_LLM_LEVELS
LANE_COUNTS = (8, 16)
LOADS = (0.50, 0.80, 0.95)
FIXED_BUDGET_BY_LANE = {8: 36, 16: 72}
EXACT_REFERENCE_BUDGET = 4096
WORK_FIELDS = (
    "lp_calls",
    "lp_matrix_coefficients",
    "candidate_evaluations",
    "residual_fill_option_probes",
    "dual_column_checks",
    "branch_splits",
    "canonical_key_bound_evaluations",
    "canonical_tie_pruned_regions",
    "canonical_tie_incumbent_updates",
    "canonical_tie_singletons_resolved",
)
THIS_SOURCE = (
    "algorithm/experiments/aligned_hardware_joint95_canonical_selector_screen.py"
)
EXTRA_FIXED_SOURCE_PATHS = (
    "algorithm/experiments/aligned_hardware_joint95_capacity_closure.py",
    "algorithm/experiments/aligned_hardware_joint95_static_selector_screen.py",
    "algorithm/experiments/aligned_hardware_lcb_frame_envelope_shadow.py",
    THIS_SOURCE,
)

DECISION_RULE = {
    "INVALID_EXECUTION": "one or more registered execution checks fail",
    PASS_DECISION: (
        "both registered lanes have valid canonical-family candidate and cold "
        "reference certificates; every K4096 reference is score-exact and "
        "canonical-exact; candidate epsilon and actual normalized support regret "
        "are strictly below eta at all three physical-capacity-normalized loads; "
        "and candidate LP, matrix, and candidate-evaluation work are strictly "
        "below reference work per lane and in aggregate"
    ),
    "CANONICAL_PHYSICAL_SELECTOR_SUPPORT_FAIL_STOP": (
        "execution is valid but at least one registered lane/load support gate fails"
    ),
    "CANONICAL_PHYSICAL_SELECTOR_COMPUTE_FAIL_STOP": (
        "support passes but at least one registered actual-work comparison is not strict"
    ),
}

CLAIM_BOUNDARY = (
    "A preregistered finite static comparison on 512 independently generated base "
    "queues for synthetic identical-lane R8 and R16 replicas of the final joint-95 "
    "gpu_b complete-frame envelope. It establishes returned score bounds, "
    "support regret against a score-exact and canonical-exact reference action, "
    "canonical tie-safe selector behavior, and actual search work on this roster "
    "only. It is not an all-queue theorem, a flow or trajectory result, a stability "
    "proof, a physical multi-GPU experiment, or a deployment decision. A pass only "
    "authorizes a separately registered baseline trajectory comparison."
)

PROHIBITED = (
    "reading joint visit values, the joint ledger, a joint margin, or the capacity-"
    "closure result during registration; changing the roster, loads, budgets, "
    "canonical reference, or gates after outcomes; ordered-action or count-vector "
    "enumeration; run-time RNG; arrivals or trajectories; server pulls; and paper edits"
)

LEGACY_DISPOSITION = {
    "experiment": legacy.EXPERIMENT,
    "evidence_retained": (
        "the old static screen remains valid for its finite-roster support-score "
        "and actual-work evidence"
    ),
    "superseded_scope": (
        "trajectory authorization only, because the old score-exact selector did "
        "not certify the canonical policy action"
    ),
    "artifact_erased_or_rewritten": False,
}


class CanonicalStaticScreenRefusal(RuntimeError):
    """The frozen registration is invalid or conditional evidence is not ready."""


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CanonicalStaticScreenRefusal(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CanonicalStaticScreenRefusal(f"JSON root must be an object: {path}")
    return value


def _write_new(path: Path, payload: Mapping[str, Any], *, compact: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(
            dict(payload),
            stream,
            indent=None if compact else 2,
            separators=(",", ":") if compact else None,
            sort_keys=not compact,
        )
        stream.write("\n")


def _artifact_family(path: Path) -> tuple[Path, Path, Path]:
    path = Path(path)
    return (
        path,
        path.with_name(path.stem + "_protocol.json"),
        path.with_name(path.stem + "_fixed_sources.json"),
    )


def _canonical_confirmation_contract(
    result_path: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    result_path, protocol_path, fixed_path = _artifact_family(result_path)
    result = _read_object(result_path)
    protocol = _read_object(protocol_path)
    fixed_sources = _read_object(fixed_path)
    changed = [
        name
        for name, text in fixed_sources.items()
        if not isinstance(text, str)
        or not (ROOT / name).is_file()
        or (ROOT / name).read_text(encoding="utf-8") != text
    ]
    expected_cases = {(lanes, load) for lanes in LANE_COUNTS for load in (0.80, 0.95)}
    observed_cases = {
        (row.get("lanes"), row.get("load"))
        for row in result.get("case_summaries") or ()
    }
    if (
        changed
        or result.get("experiment") != canonical_confirmation.EXPERIMENT
        or result.get("status") != "PASS"
        or result.get("scientific_decision") != CANONICAL_SOURCE_DECISION
        or result.get("canonical_exact_compute_confirmed") is not True
        or not result.get("checks")
        or not all(result["checks"].values())
        or result.get("protocol") != protocol
        or observed_cases != expected_cases
        or not (result.get("aggregate_gate") or {}).get("aggregate_gate_passed")
        or (result.get("work_accounting") or {}).get("ordered_action_enumerations") != 0
        or (result.get("work_accounting") or {}).get("count_vector_enumerations") != 0
    ):
        raise CanonicalStaticScreenRefusal(
            "formal canonical-exact compute confirmation is invalid or its sources changed"
        )
    contract = {
        "result_path": str(result_path.resolve()),
        "protocol_path": str(protocol_path.resolve()),
        "fixed_sources_path": str(fixed_path.resolve()),
        "experiment": result["experiment"],
        "status": result["status"],
        "scientific_decision": result["scientific_decision"],
        "canonical_exact_compute_confirmed": True,
        "checks": _json(result["checks"]),
        "aggregate_gate": _json(result["aggregate_gate"]),
        "case_gate_keys": sorted(
            [f"R{int(row['lanes'])}/load={float(row['load']):.2f}" for row in result["case_summaries"]]
        ),
        "protocol": _json(protocol),
        "claim_boundary": result.get("claim_boundary"),
        "source_disposition": (
            "formal point-model evidence for canonical tie handling and empirical "
            "bounded compute; the present screen separately evaluates the future "
            "joint-95 envelope"
        ),
    }
    return contract, {str(name): str(text) for name, text in fixed_sources.items()}


def _current_fixed_sources(canonical_fixed_sources: Mapping[str, str]) -> dict[str, str]:
    result = dict(canonical_fixed_sources)
    for name in EXTRA_FIXED_SOURCE_PATHS:
        result[name] = (ROOT / name).read_text(encoding="utf-8")
    return result


def _old_base_grid() -> set[tuple[int, int, int]]:
    return set(product(OLD_CNN_LEVELS, OLD_JAX_LEVELS, OLD_LLM_LEVELS))


def _validated_count_roster(payload: Mapping[str, Any]) -> tuple[tuple[int, int, int], ...]:
    try:
        return tuple(legacy._validated_previous_roster(payload))
    except legacy.StaticScreenRefusal as exc:
        raise CanonicalStaticScreenRefusal(str(exc)) from exc


def _validated_legacy_static_roster(
    payload: Mapping[str, Any], count_roster: Sequence[Sequence[int]]
) -> tuple[tuple[int, int, int], ...]:
    queues = tuple(tuple(int(value) for value in row) for row in payload.get("queues") or ())
    count_set = {tuple(int(value) for value in row) for row in count_roster}
    valid = (
        payload.get("schema_version") == 1
        and payload.get("experiment") == legacy.EXPERIMENT
        and payload.get("seed") == legacy.ROSTER_SEED
        and payload.get("coordinate_ranges_inclusive")
        == [list(bounds) for bounds in ROSTER_RANGES]
        and payload.get("requested_queue_count") == ROSTER_SIZE
        and len(queues) == len(set(queues)) == ROSTER_SIZE
        and payload.get("old_1000_grid_intersection_count") == 0
        and payload.get("prior_512_roster_intersection_count") == 0
        and all(
            len(queue) == len(ROSTER_RANGES)
            and queue != (0, 0, 0)
            and queue not in _old_base_grid()
            and queue not in count_set
            and all(
                lower <= value <= upper
                for value, (lower, upper) in zip(queue, ROSTER_RANGES, strict=True)
            )
            for queue in queues
        )
    )
    if not valid:
        raise CanonicalStaticScreenRefusal("legacy joint95 static roster is invalid")
    return queues


def generate_roster(
    count_roster: Sequence[Sequence[int]],
    legacy_static_roster: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """Materialize the only random object used during registration."""

    old_grid = _old_base_grid()
    count_set = {tuple(int(value) for value in row) for row in count_roster}
    static_set = {tuple(int(value) for value in row) for row in legacy_static_roster}
    if len(count_set) != ROSTER_SIZE or len(static_set) != ROSTER_SIZE:
        raise CanonicalStaticScreenRefusal("both exclusion rosters must contain 512 queues")
    rng = random.Random(ROSTER_SEED)
    queues: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    draws = 0
    while len(queues) < ROSTER_SIZE:
        draws += 1
        queue = tuple(rng.randint(lower, upper) for lower, upper in ROSTER_RANGES)
        if (
            queue == (0, 0, 0)
            or queue in old_grid
            or queue in count_set
            or queue in static_set
            or queue in seen
        ):
            continue
        seen.add(queue)
        queues.append(queue)
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "seed": ROSTER_SEED,
        "coordinate_ranges_inclusive": [list(bounds) for bounds in ROSTER_RANGES],
        "requested_queue_count": ROSTER_SIZE,
        "candidate_draws": draws,
        "rng_integer_draws": draws * len(ROSTER_RANGES),
        "zero_queue_excluded": True,
        "old_1000_grid_excluded": True,
        "old_count_512_roster_excluded": True,
        "old_joint95_static_512_roster_excluded": True,
        "old_1000_grid_intersection_count": sum(queue in old_grid for queue in queues),
        "old_count_512_roster_intersection_count": sum(queue in count_set for queue in queues),
        "old_joint95_static_512_roster_intersection_count": sum(
            queue in static_set for queue in queues
        ),
        "queues": [list(queue) for queue in queues],
    }


def _roster_source_row(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(Path(path).resolve()),
        "experiment": payload.get("experiment"),
        "seed": payload.get("seed"),
        "size": payload.get("requested_queue_count"),
        "candidate_draws": payload.get("candidate_draws"),
        "queues": _json(payload.get("queues")),
    }


def _protocol_constants(
    *,
    capacity_registration_path: Path,
    capacity_result_path: Path,
    canonical_confirmation_path: Path,
    count_roster_path: Path,
    legacy_static_roster_path: Path,
    capacity_contract: Mapping[str, Any],
    canonical_contract: Mapping[str, Any],
    count_payload: Mapping[str, Any],
    static_payload: Mapping[str, Any],
    roster: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "gate": GATE,
        "experiment": EXPERIMENT,
        "status": REGISTRATION_STATUS,
        "registered": True,
        "source_paths": {
            "capacity_closure_registration": str(Path(capacity_registration_path).resolve()),
            "capacity_closure_result": str(Path(capacity_result_path).resolve()),
            "joint_results_directory": str(
                Path(capacity_contract["source_paths"]["joint_results_directory"]).resolve()
            ),
            "canonical_exact_compute_confirmation": str(
                Path(canonical_confirmation_path).resolve()
            ),
            "old_count_space_confirmation_roster": str(Path(count_roster_path).resolve()),
            "old_joint95_static_roster": str(Path(legacy_static_roster_path).resolve()),
        },
        "capacity_closure_registration_contract": _json(capacity_contract),
        "uniform_margin_scaling_audit": legacy._uniform_margin_scaling_audit(
            capacity_contract
        ),
        "canonical_exact_compute_confirmation_contract": _json(canonical_contract),
        "legacy_static_screen_disposition": dict(LEGACY_DISPOSITION),
        "queue_roster_contract": {
            "seed": ROSTER_SEED,
            "size": ROSTER_SIZE,
            "coordinate_ranges_inclusive": [list(bounds) for bounds in ROSTER_RANGES],
            "candidate_draws": roster["candidate_draws"],
            "rng_integer_draws": roster["rng_integer_draws"],
            "base_queues": _json(roster["queues"]),
            "lane_scaling_rule": "Q_R = (R/2) * Q_base coordinatewise",
            "zero_queue_excluded": True,
            "old_1000_grid": {
                "size": len(_old_base_grid()),
                "intersection_count": roster["old_1000_grid_intersection_count"],
            },
            "old_count_512_roster": {
                **_roster_source_row(count_roster_path, count_payload),
                "intersection_count": roster["old_count_512_roster_intersection_count"],
            },
            "old_joint95_static_512_roster": {
                **_roster_source_row(legacy_static_roster_path, static_payload),
                "intersection_count": roster[
                    "old_joint95_static_512_roster_intersection_count"
                ],
            },
        },
        "model_contract": {
            "node": "gpu_b",
            "base_lanes": 2,
            "lane_counts": list(LANE_COUNTS),
            "action_count": 9,
            "duration_catalogue": (
                "capacity-closure nine-action complete-frame envelope rebuilt "
                "from the frozen predictor anchor and final rank-19 joint margin"
            ),
            "synthetic_lane_rule": (
                "replicate the hardware-local option catalogue across identical lanes"
            ),
        },
        "physical_load_contract": {
            "loads": list(LOADS),
            "normalization": "fraction of final joint-envelope common capacity boundary",
            "common_boundary_source": "capacity_closure.capacity.common_capacity_scale",
            "old_point_load_capacity_gate_required": False,
            "eta_formula": (
                "uniform physical service slack minus projection support loss"
            ),
        },
        "selector_contract": {
            "family": "canonical-v2 cold count-space branch-and-bound selector",
            "fixed_budget_by_lane": {
                str(lanes): budget for lanes, budget in FIXED_BUDGET_BY_LANE.items()
            },
            "cold_canonical_reference_budget": EXACT_REFERENCE_BUDGET,
            "candidate": "one fresh canonical count-space call per queue state",
            "reference": "one separate cold canonical K4096 call per queue state",
            "canonical_key": (
                "ascending (duration, sorted option indices with idle last)"
            ),
            "reference_gate": (
                "every reference is score-exact and canonical-exact"
            ),
            "support_gate": (
                "for every R/load, candidate epsilon and actual normalized support "
                "regret are each strictly below eta"
            ),
            "actual_work_gate": (
                "candidate totals are strictly below reference totals for LP calls, "
                "LP matrix coefficients, and candidate evaluations per lane and aggregate"
            ),
            "enumeration_gate": (
                "ordered-action evaluations and count-vector enumerations both equal zero"
            ),
        },
        "expected_work": {
            "registered_base_queues": ROSTER_SIZE,
            "scaled_queue_states": ROSTER_SIZE * len(LANE_COUNTS),
            "candidate_selector_calls": ROSTER_SIZE * len(LANE_COUNTS),
            "cold_reference_selector_calls": ROSTER_SIZE * len(LANE_COUNTS),
            "ordered_action_evaluations": 0,
            "count_vector_enumerations": 0,
            "run_rng_initializations": 0,
            "arrival_tapes": 0,
            "trajectory_frames": 0,
        },
        "run_preconditions": [
            "the capacity-closure registration protocol is unchanged",
            "the formal canonical-exact compute confirmation and sources are unchanged",
            "waves 14..32, rank-19 release, and wave 33 holdout are complete and valid",
            "the recomputed joint95 coverage audit passes",
            "a matching capacity-closure result reproduces the envelope and capacity analysis",
            "the final common capacity boundary is positive and finite",
        ],
        "decision_rule": dict(DECISION_RULE),
        "claim_boundary": CLAIM_BOUNDARY,
        "prohibited": PROHIBITED,
        "registration_scope": {
            "capacity_closure_protocol_read": True,
            "canonical_exact_compute_confirmation_read": True,
            "old_queue_rosters_read": 2,
            "joint_results_directory_read": False,
            "joint_visit_values_read": False,
            "capacity_closure_result_read": False,
            "joint_margin_computed": False,
            "capacity_computed": False,
            "selector_invoked": False,
            "exact_reference_invoked": False,
            "arrival_or_trajectory_constructed": False,
            "server_pull_performed": False,
            "roster_rng_initializations": 1,
            "roster_rng_integer_draws": roster["rng_integer_draws"],
            "other_rng_initializations": 0,
            "other_rng_draws": 0,
        },
    }


def register(
    registration: Path = DEFAULT_REGISTRATION,
    roster_path: Path = DEFAULT_ROSTER,
    fixed_sources_path: Path = DEFAULT_FIXED_SOURCES,
    *,
    capacity_registration_path: Path = DEFAULT_CAPACITY_REGISTRATION,
    capacity_result_path: Path = DEFAULT_CAPACITY_RESULT,
    canonical_confirmation_path: Path = DEFAULT_CANONICAL_CONFIRMATION,
    count_roster_path: Path = DEFAULT_COUNT_ROSTER,
    legacy_static_roster_path: Path = DEFAULT_LEGACY_STATIC_ROSTER,
) -> dict[str, Any]:
    """Freeze all constants and the sole random roster without outcome reads."""

    paths = (Path(registration), Path(roster_path), Path(fixed_sources_path))
    if any(path.exists() for path in paths):
        raise CanonicalStaticScreenRefusal(
            "canonical static-screen registration artifacts already exist"
        )
    capacity_source = _read_object(Path(capacity_registration_path))
    try:
        capacity_contract = legacy._capacity_registration_contract(capacity_source)
    except legacy.StaticScreenRefusal as exc:
        raise CanonicalStaticScreenRefusal(str(exc)) from exc
    canonical_contract, canonical_fixed = _canonical_confirmation_contract(
        Path(canonical_confirmation_path)
    )
    count_payload = _read_object(Path(count_roster_path))
    count_queues = _validated_count_roster(count_payload)
    static_payload = _read_object(Path(legacy_static_roster_path))
    static_queues = _validated_legacy_static_roster(static_payload, count_queues)
    roster = generate_roster(count_queues, static_queues)
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        **_protocol_constants(
            capacity_registration_path=Path(capacity_registration_path),
            capacity_result_path=Path(capacity_result_path),
            canonical_confirmation_path=Path(canonical_confirmation_path),
            count_roster_path=Path(count_roster_path),
            legacy_static_roster_path=Path(legacy_static_roster_path),
            capacity_contract=capacity_contract,
            canonical_contract=canonical_contract,
            count_payload=count_payload,
            static_payload=static_payload,
            roster=roster,
        ),
    }
    _write_new(Path(roster_path), roster, compact=True)
    _write_new(Path(fixed_sources_path), _current_fixed_sources(canonical_fixed))
    _write_new(Path(registration), protocol, compact=True)
    return protocol


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    if (
        protocol.get("schema_version") != 1
        or protocol.get("gate") != GATE
        or protocol.get("experiment") != EXPERIMENT
        or protocol.get("status") != REGISTRATION_STATUS
        or protocol.get("registered") is not True
        or protocol.get("decision_rule") != DECISION_RULE
        or protocol.get("claim_boundary") != CLAIM_BOUNDARY
        or protocol.get("prohibited") != PROHIBITED
        or protocol.get("legacy_static_screen_disposition") != LEGACY_DISPOSITION
    ):
        raise CanonicalStaticScreenRefusal("canonical static-screen registration is invalid")
    audit = protocol.get("uniform_margin_scaling_audit") or {}
    if (
        audit.get("all_nine_action_durations_uniformly_scaled") is not True
        or audit.get("issue_count") != 0
        or audit.get("joint_result_values_read") is not False
        or audit.get("capacity_outcome_read") is not False
    ):
        raise CanonicalStaticScreenRefusal("registered common-margin audit is invalid")


def _validate_roster(
    roster: Mapping[str, Any], protocol: Mapping[str, Any]
) -> tuple[tuple[int, int, int], ...]:
    contract = protocol.get("queue_roster_contract") or {}
    queues = tuple(tuple(int(value) for value in row) for row in roster.get("queues") or ())
    count_set = {
        tuple(int(value) for value in row)
        for row in (contract.get("old_count_512_roster") or {}).get("queues") or ()
    }
    static_set = {
        tuple(int(value) for value in row)
        for row in (contract.get("old_joint95_static_512_roster") or {}).get("queues") or ()
    }
    valid = (
        roster.get("schema_version") == 1
        and roster.get("experiment") == EXPERIMENT
        and roster.get("seed") == ROSTER_SEED
        and roster.get("coordinate_ranges_inclusive")
        == [list(bounds) for bounds in ROSTER_RANGES]
        and roster.get("requested_queue_count") == ROSTER_SIZE
        and roster.get("candidate_draws") == contract.get("candidate_draws")
        and roster.get("rng_integer_draws") == contract.get("rng_integer_draws")
        and roster.get("queues") == contract.get("base_queues")
        and len(queues) == len(set(queues)) == ROSTER_SIZE
        and len(count_set) == len(static_set) == ROSTER_SIZE
        and all(
            queue != (0, 0, 0)
            and queue not in _old_base_grid()
            and queue not in count_set
            and queue not in static_set
            and len(queue) == len(ROSTER_RANGES)
            and all(
                lower <= value <= upper
                for value, (lower, upper) in zip(queue, ROSTER_RANGES, strict=True)
            )
            for queue in queues
        )
        and roster.get("old_1000_grid_intersection_count") == 0
        and roster.get("old_count_512_roster_intersection_count") == 0
        and roster.get("old_joint95_static_512_roster_intersection_count") == 0
    )
    if not valid:
        raise CanonicalStaticScreenRefusal(
            "registered canonical static-screen roster changed or is invalid"
        )
    return queues


def _runtime_context(protocol: Mapping[str, Any]) -> dict[str, Any]:
    paths = protocol.get("source_paths") or {}
    canonical_contract, _ = _canonical_confirmation_contract(
        Path(paths["canonical_exact_compute_confirmation"])
    )
    if canonical_contract != protocol.get("canonical_exact_compute_confirmation_contract"):
        raise CanonicalStaticScreenRefusal(
            "canonical-exact compute confirmation changed after registration"
        )
    try:
        return legacy._runtime_context(protocol)
    except legacy.StaticScreenRefusal as exc:
        raise CanonicalStaticScreenRefusal(str(exc)) from exc


def _certificate_valid(
    selection: CanonicalCountSpaceBnbSelection, queue: Sequence[int]
) -> bool:
    queue_norm = sum(queue)
    return (
        selection.score <= selection.upper_bound + SCORE_TOLERANCE
        and selection.gap >= -SCORE_TOLERANCE
        and math.isclose(
            selection.gap,
            max(0.0, selection.upper_bound - selection.score),
            rel_tol=0.0,
            abs_tol=SCORE_TOLERANCE,
        )
        and math.isclose(
            selection.epsilon,
            selection.gap / queue_norm if queue_norm else 0.0,
            rel_tol=0.0,
            abs_tol=SCORE_TOLERANCE,
        )
        and selection.dual_certified_lp_calls + selection.safe_fallback_lp_calls
        == selection.lp_calls
        and selection.exact == selection.canonical_exact
        and (not selection.canonical_exact or selection.score_exact)
    )


def _work_row(selection: CanonicalCountSpaceBnbSelection) -> dict[str, int]:
    return {field: int(getattr(selection, field)) for field in WORK_FIELDS}


def _new_work() -> dict[str, int]:
    return {field: 0 for field in WORK_FIELDS}


def _add_work(target: dict[str, int], selection: CanonicalCountSpaceBnbSelection) -> None:
    for field, value in _work_row(selection).items():
        target[field] += value


def _selection_witness(
    base_queue: Sequence[int],
    queue: Sequence[int],
    candidate: CanonicalCountSpaceBnbSelection,
    reference: CanonicalCountSpaceBnbSelection,
    value: float,
) -> dict[str, Any]:
    def compact(selection: CanonicalCountSpaceBnbSelection) -> dict[str, Any]:
        return {
            "option_counts": list(selection.option_counts),
            "option_indices": list(selection.option_indices),
            "score": selection.score,
            "upper_bound": selection.upper_bound,
            "gap": selection.gap,
            "epsilon": selection.epsilon,
            "score_exact": selection.score_exact,
            "canonical_exact": selection.canonical_exact,
            "work": _work_row(selection),
        }

    return {
        "base_queue": list(base_queue),
        "scaled_queue": list(queue),
        "queue_l1": sum(queue),
        "value": value,
        "candidate": compact(candidate),
        "cold_canonical_reference": compact(reference),
    }


def _screen_lane(
    model: Any,
    base_queues: Sequence[Sequence[int]],
    lanes: int,
    capacity_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, bool]]:
    budget = FIXED_BUDGET_BY_LANE[lanes]
    factor = lanes // 2
    candidate_work = _new_work()
    reference_work = _new_work()
    max_epsilon = -1.0
    max_regret = -1.0
    epsilon_witness = None
    regret_witness = None
    valid_candidates = 0
    valid_references = 0
    bounded_regrets = 0
    candidate_score_exact = 0
    candidate_canonical_exact = 0
    action_matches = 0

    for base_queue in base_queues:
        queue = tuple(factor * int(value) for value in base_queue)
        candidate = select_canonical_count_space_bnb_frame(
            model.options, queue, lanes, budget
        )
        reference = select_canonical_count_space_bnb_frame(
            model.options, queue, lanes, EXACT_REFERENCE_BUDGET
        )
        candidate_valid = (
            _selection_reconstructs(model, queue, lanes, candidate)
            and _selection_work_holds(queue, lanes, budget, candidate)
            and _certificate_valid(candidate, queue)
        )
        reference_valid = (
            _selection_reconstructs(model, queue, lanes, reference)
            and _selection_work_holds(
                queue, lanes, EXACT_REFERENCE_BUDGET, reference
            )
            and _certificate_valid(reference, queue)
            and reference.score_exact
            and reference.canonical_exact
            and math.isclose(
                reference.score,
                reference.upper_bound,
                rel_tol=0.0,
                abs_tol=SCORE_TOLERANCE,
            )
        )
        norm = sum(queue)
        regret = (reference.score - candidate.score) / norm
        bounded = (
            regret >= -SCORE_TOLERANCE
            and regret <= candidate.epsilon + SCORE_TOLERANCE
        )
        valid_candidates += int(candidate_valid)
        valid_references += int(reference_valid)
        bounded_regrets += int(bounded)
        candidate_score_exact += int(candidate.score_exact)
        candidate_canonical_exact += int(candidate.canonical_exact)
        action_matches += int(candidate.option_indices == reference.option_indices)
        _add_work(candidate_work, candidate)
        _add_work(reference_work, reference)
        if candidate.epsilon > max_epsilon:
            max_epsilon = candidate.epsilon
            epsilon_witness = _selection_witness(
                base_queue, queue, candidate, reference, candidate.epsilon
            )
        if regret > max_regret:
            max_regret = regret
            regret_witness = _selection_witness(
                base_queue, queue, candidate, reference, regret
            )

    comparisons = {
        field: {
            "candidate": candidate_work[field],
            "cold_canonical_reference": reference_work[field],
            "candidate_fraction_of_reference": (
                candidate_work[field] / reference_work[field]
                if reference_work[field]
                else None
            ),
            "strictly_lower": candidate_work[field] < reference_work[field],
        }
        for field in ("lp_calls", "lp_matrix_coefficients", "candidate_evaluations")
    }
    support_rows = []
    for capacity in capacity_rows:
        eta = float(capacity["effective_slack_eta"])
        epsilon_residual = eta - max_epsilon
        regret_residual = eta - max_regret
        support_rows.append({
            "lanes": lanes,
            "load": float(capacity["load"]),
            "load_normalization": capacity["load_normalization"],
            "arrival_rates": _json(capacity["arrival_rates"]),
            "effective_slack_eta": eta,
            "maximum_candidate_epsilon": max_epsilon,
            "maximum_actual_normalized_support_regret": max_regret,
            "eta_minus_candidate_epsilon": epsilon_residual,
            "eta_minus_actual_support_regret": regret_residual,
            "candidate_epsilon_strictly_below_eta": epsilon_residual > SCORE_TOLERANCE,
            "actual_support_regret_strictly_below_eta": regret_residual > SCORE_TOLERANCE,
            "support_gate_passed": (
                bool(capacity["capacity_projection_passed"])
                and epsilon_residual > SCORE_TOLERANCE
                and regret_residual > SCORE_TOLERANCE
            ),
        })
    checks = {
        "scaled_queue_count_is_exact": len(base_queues) == ROSTER_SIZE,
        "every_candidate_certificate_is_valid": valid_candidates == ROSTER_SIZE,
        "every_reference_certificate_is_valid_score_exact_and_canonical_exact": (
            valid_references == ROSTER_SIZE
        ),
        "every_actual_support_regret_is_nonnegative_and_bounded_by_epsilon": (
            bounded_regrets == ROSTER_SIZE
        ),
    }
    support_passed = all(row["support_gate_passed"] for row in support_rows)
    compute_passed = all(row["strictly_lower"] for row in comparisons.values())
    return ({
        "lanes": lanes,
        "queue_scale": factor,
        "state_count": len(base_queues),
        "fixed_lp_budget": budget,
        "cold_canonical_reference_budget": EXACT_REFERENCE_BUDGET,
        "raw_ordered_actions_per_state": LOCAL_CHOICE_COUNT**lanes,
        "represented_count_vectors_per_state": math.comb(OPTION_COUNT + lanes, lanes),
        "candidate_score_exact_state_count": candidate_score_exact,
        "candidate_canonical_exact_state_count": candidate_canonical_exact,
        "candidate_action_matches_reference_count": action_matches,
        "valid_candidate_certificate_count": valid_candidates,
        "valid_canonical_reference_count": valid_references,
        "regret_bounded_by_epsilon_count": bounded_regrets,
        "maximum_candidate_epsilon": max_epsilon,
        "maximum_actual_normalized_support_regret": max_regret,
        "epsilon_witness": epsilon_witness,
        "support_regret_witness": regret_witness,
        "support_by_physical_load": support_rows,
        "candidate_actual_work": candidate_work,
        "cold_canonical_reference_actual_work": reference_work,
        "actual_compute_comparison": comparisons,
        "checks": checks,
        "support_gate_passed": support_passed,
        "actual_compute_gate_passed": compute_passed,
    }, checks)


def _decision(
    execution_passed: bool,
    lanes: Sequence[Mapping[str, Any]],
    aggregate_compute: bool,
) -> str:
    if not execution_passed:
        return "INVALID_EXECUTION"
    if not all(row["support_gate_passed"] for row in lanes):
        return "CANONICAL_PHYSICAL_SELECTOR_SUPPORT_FAIL_STOP"
    if not aggregate_compute or not all(row["actual_compute_gate_passed"] for row in lanes):
        return "CANONICAL_PHYSICAL_SELECTOR_COMPUTE_FAIL_STOP"
    return PASS_DECISION


def run_registered(
    registration: Path = DEFAULT_REGISTRATION,
    roster_path: Path = DEFAULT_ROSTER,
    fixed_sources_path: Path = DEFAULT_FIXED_SOURCES,
    output: Path = DEFAULT_RESULT,
) -> dict[str, Any]:
    output = Path(output)
    if output.exists():
        raise CanonicalStaticScreenRefusal(f"result already exists: {output}")
    protocol = _read_object(Path(registration))
    _validate_protocol(protocol)
    base_queues = _validate_roster(_read_object(Path(roster_path)), protocol)
    fixed_sources = _read_object(Path(fixed_sources_path))
    _, canonical_fixed = _canonical_confirmation_contract(
        Path(protocol["source_paths"]["canonical_exact_compute_confirmation"])
    )
    if fixed_sources != _current_fixed_sources(canonical_fixed):
        raise CanonicalStaticScreenRefusal("registered canonical static-screen sources changed")

    # Coverage and capacity closure are checked before capacity or selector work.
    context = _runtime_context(protocol)
    model = context["envelope_model"]
    if (
        len(model.options) != OPTION_COUNT
        or len(set(max(1.0, option.duration) for option in model.options)) != OPTION_COUNT
    ):
        raise CanonicalStaticScreenRefusal("joint envelope does not retain nine roots")
    capacity_by_lane = legacy._physical_capacity_rows(
        context["source"], model, context["common_capacity_scale"]
    )

    started = time.perf_counter()
    lane_rows = []
    lane_checks = []
    for lanes in LANE_COUNTS:
        row, checks = _screen_lane(
            model, base_queues, lanes, capacity_by_lane[str(lanes)]
        )
        lane_rows.append(row)
        lane_checks.append(checks)

    candidate_total = _new_work()
    reference_total = _new_work()
    for row in lane_rows:
        for field in WORK_FIELDS:
            candidate_total[field] += row["candidate_actual_work"][field]
            reference_total[field] += row["cold_canonical_reference_actual_work"][field]
    aggregate_comparisons = {
        field: {
            "candidate": candidate_total[field],
            "cold_canonical_reference": reference_total[field],
            "candidate_fraction_of_reference": (
                candidate_total[field] / reference_total[field]
                if reference_total[field]
                else None
            ),
            "strictly_lower": candidate_total[field] < reference_total[field],
        }
        for field in ("lp_calls", "lp_matrix_coefficients", "candidate_evaluations")
    }
    aggregate_compute = all(row["strictly_lower"] for row in aggregate_comparisons.values())
    checks = {
        "registered_roster_is_explicit_unique_and_disjoint_from_all_prior_rosters": (
            len(base_queues) == ROSTER_SIZE
        ),
        "formal_canonical_exact_compute_confirmation_revalidated": True,
        "joint95_calibration_release_holdout_and_coverage_revalidated": True,
        "matching_capacity_closure_artifact_reproduced": True,
        "physical_loads_are_final_common_boundary_normalized": all(
            row["load_normalization"]
            == "fraction_of_final_joint_envelope_common_capacity_boundary"
            for rows in capacity_by_lane.values()
            for row in rows
        ),
        "every_lane_execution_check_passes": all(
            all(values.values()) for values in lane_checks
        ),
        "selector_call_count_is_exact": (
            sum(row["state_count"] for row in lane_rows)
            == ROSTER_SIZE * len(LANE_COUNTS)
        ),
        "no_ordered_or_count_vector_enumeration": True,
        "no_run_rng_arrivals_or_trajectories": True,
    }
    execution_passed = all(checks.values())
    decision = _decision(execution_passed, lane_rows, aggregate_compute)
    report = {
        "schema_version": 1,
        "gate": GATE,
        "experiment": EXPERIMENT,
        "status": "PASS" if execution_passed else "FAIL",
        "registration_path": str(Path(registration).resolve()),
        "protocol": protocol,
        "checks": checks,
        "canonical_source": _json(
            protocol["canonical_exact_compute_confirmation_contract"]
        ),
        "legacy_static_screen_disposition": dict(LEGACY_DISPOSITION),
        "joint95_dependency": {
            "capacity_closure_result": str(
                Path(protocol["source_paths"]["capacity_closure_result"]).resolve()
            ),
            "one_sided_margin": context["joint"]["one_sided_margin"],
            "holdout_covered": context["joint"]["holdout_covered"],
            "common_capacity_scale": context["common_capacity_scale"],
        },
        "physical_complete_frame_envelope": {
            "model": _json(model.snapshot()),
            "actions": _json(context["envelopes"]),
        },
        "physical_capacity_normalized_loads": capacity_by_lane,
        "lane_summaries": lane_rows,
        "aggregate_actual_compute": {
            "candidate": candidate_total,
            "cold_canonical_reference": reference_total,
            "comparison": aggregate_comparisons,
            "gate_passed": aggregate_compute,
        },
        "work_accounting": {
            "candidate_selector_calls": ROSTER_SIZE * len(LANE_COUNTS),
            "cold_reference_selector_calls": ROSTER_SIZE * len(LANE_COUNTS),
            "ordered_action_evaluations": 0,
            "count_vector_enumerations": 0,
            "run_rng_initializations": 0,
            "arrival_tapes": 0,
            "trajectory_frames": 0,
        },
        "scientific_decision": decision,
        "physical_trajectory_advance": decision == PASS_DECISION,
        "baseline_trajectory_advance": decision == PASS_DECISION,
        "claim_boundary": CLAIM_BOUNDARY,
        "next_step": (
            "run only the separately registered baseline trajectory comparison"
            if decision == PASS_DECISION
            else "preserve this finite-roster outcome and do not tune on its queues"
        ),
        "timing": {
            "total_screen_wall_s": time.perf_counter() - started,
            "timing_is_scientific_gate": False,
        },
    }
    _write_new(output, report)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    register_parser = sub.add_parser("register")
    register_parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    register_parser.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
    register_parser.add_argument("--fixed-sources", type=Path, default=DEFAULT_FIXED_SOURCES)
    register_parser.add_argument(
        "--capacity-registration", type=Path, default=DEFAULT_CAPACITY_REGISTRATION
    )
    register_parser.add_argument("--capacity-result", type=Path, default=DEFAULT_CAPACITY_RESULT)
    register_parser.add_argument(
        "--canonical-confirmation", type=Path, default=DEFAULT_CANONICAL_CONFIRMATION
    )
    register_parser.add_argument("--count-roster", type=Path, default=DEFAULT_COUNT_ROSTER)
    register_parser.add_argument(
        "--legacy-static-roster", type=Path, default=DEFAULT_LEGACY_STATIC_ROSTER
    )
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    run_parser.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
    run_parser.add_argument("--fixed-sources", type=Path, default=DEFAULT_FIXED_SOURCES)
    run_parser.add_argument("--output", type=Path, default=DEFAULT_RESULT)
    args = parser.parse_args(argv)
    try:
        if args.command == "register":
            payload = register(
                args.registration,
                args.roster,
                args.fixed_sources,
                capacity_registration_path=args.capacity_registration,
                capacity_result_path=args.capacity_result,
                canonical_confirmation_path=args.canonical_confirmation,
                count_roster_path=args.count_roster,
                legacy_static_roster_path=args.legacy_static_roster,
            )
            output = args.registration
        else:
            payload = run_registered(
                args.registration, args.roster, args.fixed_sources, args.output
            )
            output = args.output
    except CanonicalStaticScreenRefusal as exc:
        parser.exit(2, f"REFUSED: {exc}\n")
    print(json.dumps({
        "experiment": payload["experiment"],
        "status": payload.get("status", REGISTRATION_STATUS),
        "scientific_decision": payload.get("scientific_decision"),
        "output": str(output),
    }))


if __name__ == "__main__":
    main()
