"""Post-scout raw fixed-cap statewise development sidecar.

The completed rescue-budget frontier showed that fresh cold rescue repeats
search work. This sidecar consumes that frozen negative result, reuses its
4,445 unique states, occurrence weights, and exact K4096 shadows, and evaluates
native canonical search checkpoints from one cumulative trace per state.

This is development evidence only. It does not replay a tape, execute a policy,
or support a trajectory or flow-time claim. A pass authorizes a raw fixed-cap
implementation followed by full-trajectory confirmation on new seeds.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import argparse
import itertools
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

from algorithm.canonical_count_space_bnb_frame_candidates import (
    select_canonical_count_space_bnb_frame,
    trace_canonical_count_space_bnb_frame,
)
from algorithm.experiments import (
    aligned_hardware_joint95_v5_envelope_trajectory_failure_budget_frontier_development
    as source,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
STEM = (
    "aligned_hardware_joint95_v5_envelope_"
    "cumulative_continuation_development_20260920"
)
DEFAULT_REGISTRATION = ARTIFACTS / f"{STEM}_protocol.json"
DEFAULT_FIXED_SOURCES = ARTIFACTS / f"{STEM}_fixed_sources.json"
DEFAULT_RESULT = ARTIFACTS / f"{STEM}.json"
DEFAULT_SOURCE_REGISTRATION = source.DEFAULT_REGISTRATION
DEFAULT_SOURCE_FIXED_SOURCES = source.DEFAULT_FIXED_SOURCES
DEFAULT_SOURCE_RESULT = source.DEFAULT_RESULT

EXPERIMENT = (
    "aligned_hardware_joint95_v5_envelope_"
    "cumulative_continuation_development_v1"
)
GATE = (
    "aligned_hardware_joint95_v5_envelope_"
    "cumulative_continuation_development_runner"
)
REGISTERED_STATUS = "REGISTERED_POST_SCOUT_RAW_FIXED_CAP_DEVELOPMENT"
PASS_DECISION = "RAW_FIXED_CAP_STATEWISE_DEVELOPMENT_PASS"
FAIL_DECISION = "RAW_FIXED_CAP_STATEWISE_DEVELOPMENT_FAIL_STOP"
INVALID_DECISION = "INVALID_RAW_FIXED_CAP_DEVELOPMENT_EXECUTION"
LANES = source.LANES
LOADS = source.LOADS
CHECKPOINTS = {8: (48, 96, 144, 288), 16: (9, 96, 144, 288, 576)}
WORK_FIELDS = source.WORK_FIELDS
WORK_GATE_FIELDS = source.WORK_GATE_FIELDS
SCORE_TOLERANCE = source.SCORE_TOLERANCE
THIS_SOURCE = (
    "algorithm/experiments/"
    "aligned_hardware_joint95_v5_envelope_"
    "cumulative_continuation_development.py"
)


class CumulativeContinuationRefusal(RuntimeError):
    pass


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CumulativeContinuationRefusal(f"expected JSON object: {path}")
    return payload


def _write_new(path: Path, payload: Mapping[str, Any], *, compact: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=None if compact else 2, sort_keys=True)
        handle.write("\n")


def _new_work() -> dict[str, int]:
    return {field: 0 for field in WORK_FIELDS}


def _checked_work(work: Mapping[str, Any]) -> dict[str, int]:
    if set(work) != set(WORK_FIELDS):
        raise CumulativeContinuationRefusal("work row does not cover WORK_FIELDS")
    row = {field: int(work[field]) for field in WORK_FIELDS}
    if any(row[field] < 0 or row[field] != work[field] for field in WORK_FIELDS):
        raise CumulativeContinuationRefusal("invalid work count")
    return row


def _add_scaled_work(
    target: dict[str, int], source_work: Mapping[str, Any], scale: int
) -> None:
    row = _checked_work(source_work)
    if int(scale) != scale or scale < 0:
        raise CumulativeContinuationRefusal("work scale must be nonnegative integer")
    for field in WORK_FIELDS:
        target[field] += row[field] * int(scale)


def _current_fixed_sources() -> dict[str, str]:
    frozen = _read_object(DEFAULT_SOURCE_FIXED_SOURCES)
    frozen[THIS_SOURCE] = (ROOT / THIS_SOURCE).read_text(encoding="utf-8")
    return dict(sorted(frozen.items()))


def _compact_state_roster(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for state in result.get("unique_states", []):
        shadow = state.get("shadow") or {}
        row = {
            "lanes": int(state["lanes"]),
            "queue": [int(value) for value in state["queue"]],
            "total_occurrences": int(state["total_occurrences"]),
            "occurrences_by_case": {
                str(key): int(value)
                for key, value in sorted(
                    state["occurrences_by_case"].items(),
                    key=lambda item: float(item[0]),
                )
            },
            "first_occurrence": _json(state["first_occurrence"]),
            "shadow": {
                "option_indices": list(shadow["option_indices"]),
                "score": float(shadow["score"]),
                "upper_bound": float(shadow["upper_bound"]),
                "score_exact": bool(shadow["score_exact"]),
                "canonical_exact": bool(shadow["canonical_exact"]),
                "valid": bool(shadow["valid"]),
                "work": _checked_work(shadow["work"]),
            },
        }
        if sum(row["occurrences_by_case"].values()) != row["total_occurrences"]:
            raise CumulativeContinuationRefusal("state occurrence weights changed")
        if len(row["queue"]) != 3 or any(value < 0 for value in row["queue"]):
            raise CumulativeContinuationRefusal("invalid frozen queue")
        if row["lanes"] not in LANES:
            raise CumulativeContinuationRefusal("invalid frozen lane count")
        if not (
            row["shadow"]["valid"]
            and row["shadow"]["score_exact"]
            and row["shadow"]["canonical_exact"]
        ):
            raise CumulativeContinuationRefusal("frozen shadow is not valid exact")
        rows.append(row)
    rows.sort(key=lambda row: (row["lanes"], tuple(row["queue"])))
    if len({(row["lanes"], tuple(row["queue"])) for row in rows}) != len(rows):
        raise CumulativeContinuationRefusal("duplicate frozen unique state")
    return rows


def _validated_source_bundle(
    registration: Path,
    fixed_sources: Path,
    result_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    protocol = _read_object(registration)
    registered_sources = _read_object(fixed_sources)
    result = _read_object(result_path)
    if result.get("protocol") != protocol:
        raise CumulativeContinuationRefusal("source result/protocol binding changed")
    if registered_sources != source._current_fixed_sources():
        raise CumulativeContinuationRefusal("source fixed sources changed")
    if (
        result.get("experiment") != source.EXPERIMENT
        or result.get("gate") != source.GATE
        or result.get("status") != "COMPLETE"
        or result.get("scientific_decision") != source.COMPLETE_DECISION
        or result.get("independent_confirmation_eligible") is not False
        or result.get("trajectory_or_flow_claim_authorized") is not False
        or result.get("new_tape_confirmation_required") is not True
        or result.get("unique_state_count") != 4445
        or result.get("state_occurrence_count") != 5411
        or result.get("server_pulls") != 0
        or result.get("counterfactual_trajectory_runs") != 0
    ):
        raise CumulativeContinuationRefusal(
            "source is not the completed negative rescue-budget development"
        )
    if not all((result.get("checks") or {}).values()):
        raise CumulativeContinuationRefusal("source development checks changed")
    source_scope = protocol.get("registration_scope") or {}
    if (
        source_scope.get("post_result_development") is not True
        or source_scope.get("independent_confirmation") is not False
        or source_scope.get("trajectory_or_flow_evidence") is not False
    ):
        raise CumulativeContinuationRefusal("source development boundary changed")
    roster = _compact_state_roster(result)
    if len(roster) != 4445 or sum(row["total_occurrences"] for row in roster) != 5411:
        raise CumulativeContinuationRefusal("source state roster count changed")
    contract = {
        "experiment": result["experiment"],
        "gate": result["gate"],
        "status": result["status"],
        "scientific_decision": result["scientific_decision"],
        "unique_state_count": result["unique_state_count"],
        "failure_state_count": result["failure_state_count"],
        "state_occurrence_count": result["state_occurrence_count"],
        "checks": _json(result["checks"]),
        "diagnostic_selector_calls": _json(result["diagnostic_selector_calls"]),
        "independent_confirmation_eligible": False,
        "trajectory_or_flow_claim_authorized": False,
        "new_tape_confirmation_required": True,
        "source_fixed_source_keys": sorted(registered_sources),
    }
    return contract, protocol, roster


DEVELOPMENT_BOUNDARY = (
    "This is a post-result and post-scout statewise development screen on the "
    "already observed 4,445 states. It reuses frozen occurrence weights and "
    "K4096 shadows, generates no tapes, and executes no policy trajectory. One "
    "raw native trace per state exposes the registered fixed-cap snapshots; the "
    "main decision uses each snapshot and its native cumulative work. A passing "
    "result authorizes implementation of the selected raw fixed-cap selector "
    "and a full-trajectory confirmation on new seeds only. Native exactness is "
    "reported diagnostically and is not a statewise-screen gate, so the new-seed "
    "confirmation must retain per-frame K4096 shadow-action checks."
)


def build_registration(
    *,
    source_registration: Path = DEFAULT_SOURCE_REGISTRATION,
    source_fixed_sources: Path = DEFAULT_SOURCE_FIXED_SOURCES,
    source_result: Path = DEFAULT_SOURCE_RESULT,
) -> dict[str, Any]:
    contract, protocol, roster = _validated_source_bundle(
        Path(source_registration), Path(source_fixed_sources), Path(source_result)
    )
    eta = _json(protocol["lane_minimum_effective_slack_eta"])
    if any(float(eta[str(lanes)]) <= SCORE_TOLERANCE for lanes in LANES):
        raise CumulativeContinuationRefusal("lane minimum eta is not positive")
    return {
        "schema_version": 1,
        "gate": GATE,
        "experiment": EXPERIMENT,
        "status": REGISTERED_STATUS,
        "registered": True,
        "source_paths": {
            "frontier_registration": str(Path(source_registration).resolve()),
            "frontier_fixed_sources": str(Path(source_fixed_sources).resolve()),
            "frontier_result": str(Path(source_result).resolve()),
        },
        "source_frontier_contract": contract,
        "state_roster": roster,
        "physical_complete_frame_model": _json(
            protocol["physical_complete_frame_model"]
        ),
        "lane_minimum_effective_slack_eta": eta,
        "selector_contract": {
            "checkpoints_by_lane": {
                str(lanes): list(CHECKPOINTS[lanes]) for lanes in LANES
            },
            "possible_fixed_caps_by_lane": {
                str(lanes): list(CHECKPOINTS[lanes]) for lanes in LANES
            },
            "decision_candidate": "raw_no_polish_native_fixed_cap",
            "decision_selection_rule": (
                "run the native raw canonical selector once to the configured "
                "fixed cap and execute that cap snapshot"
            ),
            "decision_policy_has_runtime_trigger": False,
            "reference_or_shadow_is_not_a_selection_input": True,
            "one_shared_cumulative_trace_per_unique_state": True,
            "trace_runs_to_registered_lane_max_for_frontier_diagnostics": True,
            "fixed_cap_selection_and_work_are_the_native_cap_snapshot": True,
            "decision_work_rule": (
                "configured cap snapshot cumulative native WORK_FIELDS exactly "
                "once, with no exchange, bulk polish, old K144 rescue, or "
                "repeated primary search"
            ),
            "first_checkpoint_equals_standalone_native_primary": (
                "covered by focused selector unit tests rather than a second "
                "cold call on every registered state"
            ),
            "earliest_registered_native_exact_checkpoint": "diagnostic_only",
            "native_score_and_canonical_exactness": (
                "reported but not a statewise-screen gate; fixed-cap shadow "
                "action identity, support, regret, certificate, and compute are gates"
            ),
            "shadow_budget": source.REFERENCE_BUDGET,
            "shadow_calls": 0,
            "work_fields": list(WORK_FIELDS),
            "work_gate_fields": list(WORK_GATE_FIELDS),
            "case_compute_gate": (
                "candidate <= K4096 shadow in every WORK_GATE_FIELD, uniformly "
                "applied to all six cases; no case-specific exception"
            ),
            "lane_and_aggregate_compute_gate": (
                "candidate < K4096 shadow in every WORK_GATE_FIELD"
            ),
            "per_state_gate_work_relations_are_descriptive_not_additional_gates": True,
        },
        "analysis_contract": {
            "unique_states": 4445,
            "state_occurrences": 5411,
            "case_count": len(LANES) * len(LOADS),
            "case_compute_rule_uniformly_applied_without_exceptions": True,
            "tape_replays": 0,
            "trajectory_runs": 0,
            "arrival_generations": 0,
            "k4096_shadow_calls": 0,
            "aggregate_cap_profiles": len(CHECKPOINTS[8]) * len(CHECKPOINTS[16]),
            "select_smallest_lane_cap_requiring_all_cases_and_lane_to_pass": True,
            "mixed_selected_caps_must_pass_aggregate": True,
        },
        "registration_scope": {
            "post_result_development": True,
            "exploratory_scout_preceded_registration": True,
            "post_scout_development": True,
            "independent_blind_registration": False,
            "registration_selector_calls": 0,
            "registration_rng_initializations": 0,
            "registration_arrival_generations": 0,
            "registration_trajectory_runs": 0,
            "registration_server_pulls": 0,
        },
        "decision_rule": {
            "invalid": (
                "any source/roster/protocol drift, invalid native checkpoint "
                "certificate, trace monotonicity failure, incomplete work row, "
                "or postflight mismatch"
            ),
            "fail_stop": (
                "either lane has no cap passing action, support, regret, "
                "certificate, all three uniformly applied case no-regression "
                "gates, and strict lane compute gates, or the mixed selected "
                "caps fail the strict aggregate gate"
            ),
            "pass": (
                "both lanes have a smallest qualifying raw native cap and their "
                "mixed aggregate qualifies; PASS authorizes implementation and "
                "new-seed full-trajectory confirmation only"
            ),
        },
        "development_boundary": DEVELOPMENT_BOUNDARY,
        "independent_confirmation_eligible": False,
        "paper_claim_authorized": False,
    }


def register(
    registration: Path = DEFAULT_REGISTRATION,
    fixed_sources: Path = DEFAULT_FIXED_SOURCES,
    result: Path = DEFAULT_RESULT,
    **kwargs: Any,
) -> dict[str, Any]:
    if any(Path(path).exists() for path in (registration, fixed_sources, result)):
        raise CumulativeContinuationRefusal("development artifact already exists")
    body = build_registration(**kwargs)
    protocol = {"registered_at_utc": datetime.now(timezone.utc).isoformat(), **body}
    _write_new(Path(fixed_sources), _current_fixed_sources())
    _write_new(Path(registration), protocol, compact=True)
    return protocol


def _validate_protocol(protocol: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    observed = dict(protocol)
    registered_at = observed.pop("registered_at_utc", None)
    if not isinstance(registered_at, str) or observed != expected:
        raise CumulativeContinuationRefusal("registered continuation protocol changed")


def _selection_work(selection: Any) -> dict[str, int]:
    return _checked_work(source.source._selection_work(selection))


def _selection_valid(model: Any, queue: Sequence[int], budget: int, selection: Any) -> bool:
    return bool(
        source.source._certificate_valid(
            model, queue, model.lanes, budget, selection
        )
    )


def _raw_trace(model: Any, queue: Sequence[int]) -> dict[str, Any]:
    checkpoints = CHECKPOINTS[int(model.lanes)]
    rows = trace_canonical_count_space_bnb_frame(
        model.options, queue, model.lanes, checkpoints
    )
    if len(rows) != len(checkpoints):
        raise CumulativeContinuationRefusal("trace checkpoint count changed")
    snapshots = dict(zip(checkpoints, rows, strict=True))
    for budget, snapshot in snapshots.items():
        if not _selection_valid(model, queue, budget, snapshot):
            raise CumulativeContinuationRefusal(
                f"invalid native trace checkpoint K{budget}"
            )
    for earlier, later in zip(rows[:-1], rows[1:], strict=True):
        if later.score + SCORE_TOLERANCE < earlier.score:
            raise CumulativeContinuationRefusal("trace score decreased")
        if later.upper_bound > earlier.upper_bound + SCORE_TOLERANCE:
            raise CumulativeContinuationRefusal("trace upper bound increased")
        earlier_work, later_work = _selection_work(earlier), _selection_work(later)
        if any(later_work[field] < earlier_work[field] for field in WORK_FIELDS):
            raise CumulativeContinuationRefusal("trace work prefix decreased")
    return {
        "snapshots": snapshots,
        "trace_to_registered_lane_max_work": _selection_work(rows[-1]),
    }


def _raw_snapshot_variant(
    model: Any,
    queue: Sequence[int],
    snapshot: Any,
    budget: int,
    shadow: Mapping[str, Any],
    lane_eta: float,
) -> dict[str, Any]:
    normalizer = sum(queue)
    regret = (
        (float(shadow["score"]) - float(snapshot.score)) / normalizer
        if normalizer
        else 0.0
    )
    epsilon = float(snapshot.epsilon)
    return {
        "configured_cap": budget,
        "option_indices": list(snapshot.option_indices),
        "score": float(snapshot.score),
        "upper_bound": float(snapshot.upper_bound),
        "epsilon": epsilon,
        "score_exact": bool(snapshot.score_exact),
        "canonical_exact": bool(snapshot.canonical_exact),
        "certificate_valid": _selection_valid(model, queue, budget, snapshot),
        "same_state_shadow_action_match": (
            list(snapshot.option_indices) == list(shadow["option_indices"])
        ),
        "actual_normalized_regret": regret,
        "within_lane_min_eta": bool(
            lane_eta - epsilon > SCORE_TOLERANCE
            and lane_eta - regret > SCORE_TOLERANCE
        ),
        "regret_bounded_by_epsilon": bool(
            -SCORE_TOLERANCE <= regret <= epsilon + SCORE_TOLERANCE
        ),
        "fixed_cap_native_work": _selection_work(snapshot),
        "shadow_work": _checked_work(shadow["work"]),
    }


def _selection_row(selection: Any, *, budget: int, valid: bool) -> dict[str, Any]:
    return {
        "method": f"raw_native_k{budget}_no_polish",
        "option_indices": list(selection.option_indices),
        "score": float(selection.score),
        "upper_bound": float(selection.upper_bound),
        "epsilon": float(selection.epsilon),
        "score_exact": bool(selection.score_exact),
        "canonical_exact": bool(selection.canonical_exact),
        "valid": bool(valid),
        "work": _selection_work(selection),
    }


def _evaluate_state(
    model: Any,
    state: Mapping[str, Any],
    lane_eta: float,
) -> dict[str, Any]:
    queue = tuple(int(value) for value in state["queue"])
    checkpoints = CHECKPOINTS[int(model.lanes)]
    trace = _raw_trace(model, queue)
    snapshots = trace["snapshots"]
    primary = snapshots[checkpoints[0]]
    primary_valid = _selection_valid(model, queue, checkpoints[0], primary)
    if primary.canonical_exact and (
        primary.epsilon != 0.0
        or not primary.score_exact
        or list(primary.option_indices) != list(state["shadow"]["option_indices"])
    ):
        raise CumulativeContinuationRefusal(
            "native canonical-exact primary disagrees with exact shadow"
        )
    earliest_exact = next(
        (budget for budget in checkpoints if snapshots[budget].canonical_exact),
        None,
    )
    variants = {
        budget: _raw_snapshot_variant(
            model,
            queue,
            snapshots[budget],
            budget,
            state["shadow"],
            lane_eta,
        )
        for budget in checkpoints
    }
    return {
        "lanes": int(model.lanes),
        "queue": queue,
        "total_occurrences": int(state["total_occurrences"]),
        "occurrences_by_case": {
            float(key): int(value)
            for key, value in state["occurrences_by_case"].items()
        },
        "first_occurrence": _json(state["first_occurrence"]),
        "raw_primary": _selection_row(
            primary, budget=checkpoints[0], valid=primary_valid
        ),
        "earliest_native_exact_registered_checkpoint_diagnostic_only": earliest_exact,
        "shadow": _json(state["shadow"]),
        "diagnostic_actual_work": {
            "trace_called_once": True,
            "trace_to_registered_lane_max_work": trace[
                "trace_to_registered_lane_max_work"
            ],
        },
        "raw_variants": variants,
    }


def _work_comparison(
    candidate: Mapping[str, Any], shadow: Mapping[str, Any]
) -> dict[str, Any]:
    candidate_row, shadow_row = _checked_work(candidate), _checked_work(shadow)
    fields = {
        field: {
            "candidate": candidate_row[field],
            "same_state_cold_k4096": shadow_row[field],
            "candidate_fraction_of_shadow": (
                candidate_row[field] / shadow_row[field]
                if shadow_row[field]
                else None
            ),
            "strictly_lower": candidate_row[field] < shadow_row[field],
            "equal": candidate_row[field] == shadow_row[field],
            "scientific_gate_field": field in WORK_GATE_FIELDS,
        }
        for field in WORK_FIELDS
    }
    return {
        "fields": fields,
        "work_gate_fields_strictly_lower": all(
            fields[field]["strictly_lower"] for field in WORK_GATE_FIELDS
        ),
        "work_gate_fields_no_greater": all(
            candidate_row[field] <= shadow_row[field] for field in WORK_GATE_FIELDS
        ),
        "all_positive_shadow_fields_strictly_lower": all(
            fields[field]["strictly_lower"]
            for field in WORK_FIELDS
            if shadow_row[field] > 0
        ),
        "all_registered_fields_strictly_lower": all(
            fields[field]["strictly_lower"] for field in WORK_FIELDS
        ),
        "zero_equal_fields": [
            field
            for field in WORK_FIELDS
            if shadow_row[field] == candidate_row[field] == 0
        ],
    }


def _scope_summary(
    evaluated: Sequence[Mapping[str, Any]],
    cap_by_lane: Mapping[int, int],
    occurrence_count: Callable[[Mapping[str, Any]], int],
    *,
    scope: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_work, shadow_work = _new_work(), _new_work()
    counts = Counter()
    maximum_epsilon = maximum_regret = 0.0
    unique_states = 0
    for state in evaluated:
        count = int(occurrence_count(state))
        if count <= 0:
            continue
        unique_states += 1
        cap = int(cap_by_lane[state["lanes"]])
        variant = state["raw_variants"][cap]
        counts["occurrences"] += count
        for key, condition in (
            ("certificate_valid", variant["certificate_valid"]),
            ("action_match", variant["same_state_shadow_action_match"]),
            ("score_exact", variant["score_exact"]),
            ("canonical_exact", variant["canonical_exact"]),
            ("within_eta", variant["within_lane_min_eta"]),
            ("regret_bounded", variant["regret_bounded_by_epsilon"]),
        ):
            counts[key] += count * int(condition)
        candidate_state_work = _checked_work(variant["fixed_cap_native_work"])
        shadow_state_work = _checked_work(variant["shadow_work"])
        all_gate_no_greater = all(
            candidate_state_work[field] <= shadow_state_work[field]
            for field in WORK_GATE_FIELDS
        )
        all_gate_equal = all(
            candidate_state_work[field] == shadow_state_work[field]
            for field in WORK_GATE_FIELDS
        )
        counts["gate_work_no_greater_unique"] += int(all_gate_no_greater)
        counts["gate_work_no_greater_occurrences"] += count * int(
            all_gate_no_greater
        )
        counts["gate_work_equal_unique"] += int(all_gate_equal)
        counts["gate_work_equal_occurrences"] += count * int(all_gate_equal)
        maximum_epsilon = max(maximum_epsilon, variant["epsilon"])
        maximum_regret = max(
            maximum_regret, variant["actual_normalized_regret"]
        )
        _add_scaled_work(candidate_work, candidate_state_work, count)
        _add_scaled_work(shadow_work, shadow_state_work, count)
    total = counts["occurrences"]
    comparison = _work_comparison(candidate_work, shadow_work)
    compute_gate = (
        comparison["work_gate_fields_no_greater"]
        if scope["scope"] == "case"
        else comparison["work_gate_fields_strictly_lower"]
    )
    gates = {
        "nonempty_scope": total > 0,
        "all_occurrences_certificate_valid": counts["certificate_valid"] == total,
        "all_occurrences_match_shadow_action": counts["action_match"] == total,
        "all_occurrences_within_lane_min_eta": counts["within_eta"] == total,
        "all_occurrences_regret_bounded": counts["regret_bounded"] == total,
        "registered_scope_compute_gate_passed": compute_gate,
    }
    return {
        **scope,
        "cap_by_lane": {str(key): int(value) for key, value in cap_by_lane.items()},
        "unique_state_count": unique_states,
        "state_occurrence_count": total,
        "certificate_valid_occurrence_count": counts["certificate_valid"],
        "shadow_action_match_occurrence_count": counts["action_match"],
        "score_exact_occurrence_count": counts["score_exact"],
        "score_inexact_occurrence_count": total - counts["score_exact"],
        "canonical_exact_occurrence_count": counts["canonical_exact"],
        "canonical_inexact_occurrence_count": total - counts["canonical_exact"],
        "exactness_is_diagnostic_not_a_registered_gate": True,
        "within_lane_min_eta_occurrence_count": counts["within_eta"],
        "regret_bounded_by_epsilon_occurrence_count": counts["regret_bounded"],
        "maximum_epsilon": maximum_epsilon,
        "maximum_actual_normalized_regret": maximum_regret,
        "per_state_gate_work_relation_counts": {
            "all_three_no_greater_unique_state_count": counts[
                "gate_work_no_greater_unique"
            ],
            "all_three_no_greater_occurrence_count": counts[
                "gate_work_no_greater_occurrences"
            ],
            "all_three_equal_unique_state_count": counts["gate_work_equal_unique"],
            "all_three_equal_occurrence_count": counts[
                "gate_work_equal_occurrences"
            ],
            "relation_is_descriptive_not_an_additional_gate": True,
        },
        "candidate_occurrence_weighted_work": candidate_work,
        "candidate_work_interpretation": "native_fixed_cap_actual_selector_work",
        "same_state_k4096_occurrence_weighted_work": shadow_work,
        "work_comparison": comparison,
        "work_gate_fields_strictly_lower": comparison[
            "work_gate_fields_strictly_lower"
        ],
        "work_gate_fields_no_greater": comparison["work_gate_fields_no_greater"],
        "registered_scope_compute_rule": (
            "all_three_fields_no_greater_than_shadow_uniformly_for_every_case"
            if scope["scope"] == "case"
            else "all_three_fields_strictly_lower_than_shadow"
        ),
        **gates,
        "all_registered_gates_passed": all(gates.values()),
    }


def _frontier(evaluated: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cases, lane_rows, aggregate_profiles = [], [], []
    for lanes in LANES:
        members = [state for state in evaluated if state["lanes"] == lanes]
        for cap in CHECKPOINTS[lanes]:
            cap_map = {lanes: cap}
            for load in LOADS:
                cases.append(
                    _scope_summary(
                        members,
                        cap_map,
                        lambda state, selected_load=load: state[
                            "occurrences_by_case"
                        ].get(selected_load, 0),
                        scope={
                            "scope": "case",
                            "lanes": lanes,
                            "load": load,
                            "configured_cap": cap,
                        },
                    )
                )
            lane_rows.append(
                _scope_summary(
                    members,
                    cap_map,
                    lambda state: state["total_occurrences"],
                    scope={
                        "scope": "lane",
                        "lanes": lanes,
                        "configured_cap": cap,
                    },
                )
            )
    for r8_cap, r16_cap in itertools.product(CHECKPOINTS[8], CHECKPOINTS[16]):
        cap_map = {8: r8_cap, 16: r16_cap}
        aggregate_profiles.append(
            _scope_summary(
                evaluated,
                cap_map,
                lambda state: state["total_occurrences"],
                scope={"scope": "aggregate_profile"},
            )
        )
    return {
        "case": cases,
        "by_lane": lane_rows,
        "aggregate_profiles": aggregate_profiles,
    }


def _select_lane_caps(frontier: Mapping[str, Any]) -> dict[int, int | None]:
    selected: dict[int, int | None] = {}
    for lanes in LANES:
        selected[lanes] = None
        for cap in CHECKPOINTS[lanes]:
            cases = [
                row
                for row in frontier["case"]
                if row["lanes"] == lanes and row["configured_cap"] == cap
            ]
            lane = next(
                row
                for row in frontier["by_lane"]
                if row["lanes"] == lanes and row["configured_cap"] == cap
            )
            if (
                len(cases) == len(LOADS)
                and {row["load"] for row in cases} == set(LOADS)
                and all(row["all_registered_gates_passed"] for row in cases)
                and lane["all_registered_gates_passed"]
            ):
                selected[lanes] = cap
                break
    return selected


def _serialize_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "lanes": state["lanes"],
        "queue": list(state["queue"]),
        "total_occurrences": state["total_occurrences"],
        "occurrences_by_case": {
            f"{key:.12g}": value
            for key, value in sorted(state["occurrences_by_case"].items())
        },
        "first_occurrence": state["first_occurrence"],
        "raw_primary": _json(state["raw_primary"]),
        "earliest_native_exact_registered_checkpoint_diagnostic_only": state[
            "earliest_native_exact_registered_checkpoint_diagnostic_only"
        ],
        "shadow": _json(state["shadow"]),
        "diagnostic_actual_work": _json(state["diagnostic_actual_work"]),
        "raw_fixed_cap_variants": {
            str(cap): _json(state["raw_variants"][cap])
            for cap in CHECKPOINTS[state["lanes"]]
        },
    }


def _postflight_revalidate(
    registration: Path,
    fixed_sources: Path,
    in_memory_protocol: Mapping[str, Any],
    source_paths: Mapping[str, str],
) -> None:
    on_disk = _read_object(Path(registration))
    if on_disk != in_memory_protocol:
        raise CumulativeContinuationRefusal(
            "continuation registration changed during execution"
        )
    expected = build_registration(
        source_registration=Path(source_paths["frontier_registration"]),
        source_fixed_sources=Path(source_paths["frontier_fixed_sources"]),
        source_result=Path(source_paths["frontier_result"]),
    )
    _validate_protocol(on_disk, expected)
    if _read_object(Path(fixed_sources)) != _current_fixed_sources():
        raise CumulativeContinuationRefusal(
            "continuation fixed sources changed during execution"
        )


def run_registered(
    registration: Path = DEFAULT_REGISTRATION,
    fixed_sources: Path = DEFAULT_FIXED_SOURCES,
    output: Path = DEFAULT_RESULT,
) -> dict[str, Any]:
    if Path(output).exists():
        raise CumulativeContinuationRefusal(f"development result exists: {output}")
    protocol = _read_object(Path(registration))
    paths = protocol.get("source_paths") or {}
    if set(paths) != {
        "frontier_registration",
        "frontier_fixed_sources",
        "frontier_result",
    }:
        raise CumulativeContinuationRefusal("registered source paths changed")
    expected = build_registration(
        source_registration=Path(paths["frontier_registration"]),
        source_fixed_sources=Path(paths["frontier_fixed_sources"]),
        source_result=Path(paths["frontier_result"]),
    )
    _validate_protocol(protocol, expected)
    if _read_object(Path(fixed_sources)) != _current_fixed_sources():
        raise CumulativeContinuationRefusal("registered fixed sources changed")

    started = time.perf_counter()
    model = source.source.baseline._model_from_snapshot(
        protocol["physical_complete_frame_model"]
    )
    if _json(model.snapshot()) != protocol["physical_complete_frame_model"]:
        raise CumulativeContinuationRefusal("physical model reconstruction changed")
    evaluated = []
    for frozen in protocol["state_roster"]:
        lanes = int(frozen["lanes"])
        evaluated.append(
            _evaluate_state(
                replace(model, lanes=lanes),
                frozen,
                float(protocol["lane_minimum_effective_slack_eta"][str(lanes)]),
            )
        )

    frontier = _frontier(evaluated)
    selected = _select_lane_caps(frontier)
    mixed = None
    if all(selected[lanes] is not None for lanes in LANES):
        selected_map = {lanes: int(selected[lanes]) for lanes in LANES}
        mixed = next(
            row
            for row in frontier["aggregate_profiles"]
            if row["cap_by_lane"]
            == {str(lanes): selected_map[lanes] for lanes in LANES}
        )
    selected_pass = bool(mixed and mixed["all_registered_gates_passed"])

    expected_occurrences = sum(
        row["total_occurrences"] for row in protocol["state_roster"]
    )
    observed_occurrences = sum(row["total_occurrences"] for row in evaluated)
    diagnostic_trace_work = _new_work()
    for state in evaluated:
        _add_scaled_work(
            diagnostic_trace_work,
            state["diagnostic_actual_work"]["trace_to_registered_lane_max_work"],
            1,
        )
    checks = {
        "source_frontier_and_compact_roster_revalidated": True,
        "all_4445_unique_states_evaluated": len(evaluated) == 4445,
        "state_occurrence_count_is_frozen_5411": (
            expected_occurrences == observed_occurrences == 5411
        ),
        "no_tape_rng_trajectory_or_k4096_shadow_call": True,
        "every_trace_checkpoint_certificate_and_monotonicity_valid": True,
        "all_raw_fixed_cap_variants_have_native_work_and_valid_certificates": all(
            set(variant["fixed_cap_native_work"]) == set(WORK_FIELDS)
            and variant["certificate_valid"]
            for state in evaluated
            for variant in state["raw_variants"].values()
        ),
        "case_gate_uniformly_applied_to_all_six_cases_without_exception": all(
            len(
                {
                    row["load"]
                    for row in frontier["case"]
                    if row["lanes"] == lanes
                    and row["configured_cap"] == cap
                }
            )
            == len(LOADS)
            for lanes in LANES
            for cap in CHECKPOINTS[lanes]
        ),
        "raw_decision_work_contains_no_polish_rescue_or_repeated_primary_bnb": True,
        "native_exactness_reported_but_not_used_as_registered_gate": all(
            row["exactness_is_diagnostic_not_a_registered_gate"]
            for row in (
                frontier["case"]
                + frontier["by_lane"]
                + frontier["aggregate_profiles"]
            )
        ),
        "postflight_protocol_fixed_sources_and_source_bundle_revalidated": True,
    }
    _postflight_revalidate(Path(registration), Path(fixed_sources), protocol, paths)
    status = "PASS" if selected_pass else "FAIL"
    decision = PASS_DECISION if selected_pass else FAIL_DECISION
    if not all(checks.values()):
        status, decision = "INVALID", INVALID_DECISION
    result = {
        "schema_version": 1,
        "gate": GATE,
        "experiment": EXPERIMENT,
        "status": status,
        "scientific_decision": decision,
        "protocol": protocol,
        "checks": checks,
        "unique_state_count": len(evaluated),
        "state_occurrence_count": observed_occurrences,
        "state_rows": [_serialize_state(state) for state in evaluated],
        "raw_fixed_cap_frontier": frontier,
        "selected_lane_caps": {str(key): value for key, value in selected.items()},
        "selected_mixed_aggregate": mixed,
        "diagnostic_actual_deduplicated_selector_accounting": {
            "trace_to_registered_lane_max_calls": len(evaluated),
            "trace_to_registered_lane_max_work": diagnostic_trace_work,
            "used_by_scientific_compute_gate": False,
        },
        "implementation_and_new_seed_trajectory_confirmation_authorized": (
            decision == PASS_DECISION
        ),
        "independent_confirmation_eligible": False,
        "trajectory_or_flow_claim_authorized": False,
        "paper_claim_authorized": False,
        "new_seed_full_trajectory_confirmation_required": True,
        "tape_replays": 0,
        "trajectory_runs": 0,
        "arrival_generations": 0,
        "k4096_shadow_calls": 0,
        "server_pulls": 0,
        "development_boundary": DEVELOPMENT_BOUNDARY,
        "timing": {"wall_s": time.perf_counter() - started},
    }
    _write_new(Path(output), result, compact=True)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    parser.add_argument("--fixed-sources", type=Path, default=DEFAULT_FIXED_SOURCES)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULT)
    args = parser.parse_args(argv)
    if args.register == args.run:
        parser.error("choose exactly one of --register or --run")
    if args.register:
        payload = register(args.registration, args.fixed_sources, args.output)
        summary = {
            "status": payload["status"],
            "registration": str(args.registration.resolve()),
            "fixed_sources": str(args.fixed_sources.resolve()),
            "planned_result": str(args.output.resolve()),
        }
    else:
        payload = run_registered(args.registration, args.fixed_sources, args.output)
        summary = {
            "status": payload["status"],
            "scientific_decision": payload["scientific_decision"],
            "output": str(args.output.resolve()),
            "selected_lane_caps": payload["selected_lane_caps"],
            "unique_state_count": payload["unique_state_count"],
            "state_occurrence_count": payload["state_occurrence_count"],
            "timing": payload["timing"],
        }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
