"""Post-result statewise rescue-budget diagnostic for the v5 trajectory failure.

This runner deliberately replays the *observed* 120 candidate tapes from the
frozen envelope-normalized trajectory transition.  It keeps that candidate's
visited-state distribution fixed, deduplicates the visited states, and asks a
development-only question: on those states, can a reference-free trigger plus
a fresh cold canonical rescue recover the K4096 action/certificate while still
using less occurrence-weighted work?

The diagnostic is not an independent confirmation and it does not execute a
counterfactual rescue trajectory.  Any selected rule/budget must be frozen and
run on new tapes before it can support trajectory or flow claims.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import replace
from datetime import datetime, timezone
import argparse
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from algorithm.canonical_count_space_bnb_frame_candidates import (
    select_canonical_count_space_bnb_frame,
)
from algorithm.canonical_score_tie_bulk_substitution_frame_candidates import (
    select_canonical_score_tie_bulk_substitution_frame,
    total_work_row as bulk_total_work_row,
)
from algorithm.experiments import (
    aligned_hardware_joint95_v5_envelope_normalized_trajectory_transition as source,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
STEM = (
    "aligned_hardware_joint95_v5_envelope_trajectory_failure_"
    "budget_frontier_development_20260920"
)
DEFAULT_REGISTRATION = ARTIFACTS / f"{STEM}_protocol.json"
DEFAULT_FIXED_SOURCES = ARTIFACTS / f"{STEM}_fixed_sources.json"
DEFAULT_RESULT = ARTIFACTS / f"{STEM}.json"
DEFAULT_SOURCE_REGISTRATION = source.DEFAULT_REGISTRATION
DEFAULT_SOURCE_FIXED_SOURCES = source.DEFAULT_FIXED_SOURCES
DEFAULT_SOURCE_RESULT = source.DEFAULT_RESULT

EXPERIMENT = (
    "aligned_hardware_joint95_v5_envelope_trajectory_failure_"
    "budget_frontier_development_v1"
)
GATE = (
    "aligned_hardware_joint95_v5_envelope_trajectory_failure_"
    "budget_frontier_development_runner"
)
REGISTERED_STATUS = "REGISTERED_POST_RESULT_DEVELOPMENT_ONLY"
COMPLETE_DECISION = "POST_RESULT_RESCUE_BUDGET_FRONTIER_DEVELOPMENT_COMPLETE"
INVALID_DECISION = "INVALID_POST_RESULT_DEVELOPMENT_EXECUTION"
LANES = source.LANES
LOADS = source.LOADS
ARRIVAL_SEEDS = source.ARRIVAL_SEEDS
ARRIVAL_HORIZON_S = source.ARRIVAL_HORIZON_S
RESCUE_BUDGETS = (96, 144, 288, 576, 1024)
REFERENCE_BUDGET = source.REFERENCE_BUDGET
TRIGGER_RULES = ("epsilon_near_eta_or_canonical_inexact",)
WORK_FIELDS = source.WORK_FIELDS
WORK_GATE_FIELDS = source.WORK_GATE_FIELDS
SCORE_TOLERANCE = source.SCORE_TOLERANCE
THIS_SOURCE = (
    "algorithm/experiments/"
    "aligned_hardware_joint95_v5_envelope_trajectory_failure_"
    "budget_frontier_development.py"
)


class PostResultDevelopmentRefusal(RuntimeError):
    pass


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PostResultDevelopmentRefusal(f"expected JSON object: {path}")
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
        raise PostResultDevelopmentRefusal("work row does not cover frozen WORK_FIELDS")
    row = {field: int(work[field]) for field in WORK_FIELDS}
    if any(row[field] < 0 or row[field] != work[field] for field in WORK_FIELDS):
        raise PostResultDevelopmentRefusal("work row contains an invalid count")
    return row


def _add_scaled_work(
    target: dict[str, int], source_work: Mapping[str, Any], scale: int
) -> None:
    row = _checked_work(source_work)
    if int(scale) != scale or scale < 0:
        raise PostResultDevelopmentRefusal("work scale must be nonnegative integer")
    for field in WORK_FIELDS:
        target[field] += row[field] * int(scale)


def _current_fixed_sources() -> dict[str, str]:
    frozen = _read_object(DEFAULT_SOURCE_FIXED_SOURCES)
    frozen[THIS_SOURCE] = (ROOT / THIS_SOURCE).read_text(encoding="utf-8")
    return dict(sorted(frozen.items()))


def _tape_key(lanes: int, load: float, seed: int) -> str:
    return f"R{lanes}:load={load:.12g}:seed={seed}"


def _source_tape_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for run in result.get("runs", []):
        policies = run.get("policies") or []
        candidate = next(
            (row for row in policies if row.get("policy") == "candidate"), None
        )
        if candidate is None:
            raise PostResultDevelopmentRefusal("source run lacks candidate policy")
        metrics = candidate.get("metrics") or {}
        rows.append(
            {
                "lanes": int(run["lanes"]),
                "load": float(run["load"]),
                "seed": int(run["seed"]),
                "arrival_tape": _json(run["arrival_tape"]),
                "candidate_frame_count": int(metrics["frame_count"]),
            }
        )
    rows.sort(key=lambda row: (row["lanes"], row["load"], row["seed"]))
    return rows


def _validated_source_bundle(
    registration: Path,
    fixed_sources: Path,
    result_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = _read_object(registration)
    registered_sources = _read_object(fixed_sources)
    result = _read_object(result_path)
    if result.get("protocol") != protocol:
        raise PostResultDevelopmentRefusal("source result/protocol binding changed")
    if registered_sources != source._current_fixed_sources():
        raise PostResultDevelopmentRefusal("source transition fixed sources changed")
    if (
        result.get("experiment") != source.EXPERIMENT
        or result.get("gate") != source.GATE
        or result.get("status") != "FAIL"
        or result.get("scientific_decision") != source.INVALID_DECISION
        or result.get("trajectory_transition_passed") is not False
        or result.get("development_advance") is not False
        or result.get("trajectory_runs") != 240
        or result.get("arrival_generations") != 120
        or result.get("shadow_trajectory_runs") != 0
        or result.get("server_pulls") != 0
        or result.get("work_gates", {}).get("gate_passed") is not True
    ):
        raise PostResultDevelopmentRefusal(
            "source is not the frozen failed envelope-normalized transition"
        )
    frozen = protocol.get("trajectory_contract") or {}
    if (
        frozen.get("lanes") != list(LANES)
        or frozen.get("loads") != list(LOADS)
        or frozen.get("arrival_seeds") != list(ARRIVAL_SEEDS)
        or frozen.get("arrival_horizon_s") != ARRIVAL_HORIZON_S
        or frozen.get("case_seed_tapes") != 120
        or protocol.get("work_fields") != list(WORK_FIELDS)
        or protocol.get("work_gate_fields") != list(WORK_GATE_FIELDS)
    ):
        raise PostResultDevelopmentRefusal("source trajectory contract changed")
    expected_keys = {
        _tape_key(lanes, load, seed)
        for lanes in LANES
        for load in LOADS
        for seed in ARRIVAL_SEEDS
    }
    tape_rows = _source_tape_rows(result)
    actual_keys = {
        _tape_key(row["lanes"], row["load"], row["seed"]) for row in tape_rows
    }
    if len(tape_rows) != 120 or actual_keys != expected_keys:
        raise PostResultDevelopmentRefusal("source result does not contain 120 tapes")
    checks = result.get("checks") or {}
    if not all(
        checks.get(key) is True
        for key in (
            "arrival_tape_count_is_registered_120",
            "executed_trajectory_count_is_registered_240",
            "one_arrival_generation_per_common_tape",
            "every_candidate_frame_has_one_fresh_shadow",
            "every_policy_fully_drains_and_flow_decomposes",
            "no_capacity_recomputation_or_reclassification",
        )
    ):
        raise PostResultDevelopmentRefusal("source execution checks changed")
    source_contract = {
        "experiment": result["experiment"],
        "gate": result["gate"],
        "status": result["status"],
        "scientific_decision": result["scientific_decision"],
        "trajectory_transition_passed": result["trajectory_transition_passed"],
        "development_advance": result["development_advance"],
        "trajectory_runs": result["trajectory_runs"],
        "arrival_generations": result["arrival_generations"],
        "shadow_trajectory_runs": result["shadow_trajectory_runs"],
        "server_pulls": result["server_pulls"],
        "checks": _json(checks),
        "work_gates": _json(result["work_gates"]),
        "case_failure_summary": [
            {
                key: row[key]
                for key in (
                    "lanes",
                    "load",
                    "execution_integrity_passed",
                    "trajectory_support_gate_passed",
                    "same_state_action_and_regret_gate_passed",
                    "paired_mean_flow_noninferiority_passed",
                    "case_compute_gate_passed",
                    "effective_slack_eta",
                    "maximum_candidate_epsilon",
                    "maximum_actual_normalized_regret",
                )
            }
            for row in result["case_summaries"]
        ],
        "tape_rows": tape_rows,
        "source_fixed_source_keys": sorted(registered_sources),
    }
    return source_contract, protocol


def _capacity_contract(protocol: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    capacity = _json(protocol["physical_envelope_capacity_normalized_loads"])
    minimum_eta: dict[str, float] = {}
    for lanes in LANES:
        rows = capacity.get(str(lanes))
        if not isinstance(rows, list) or {
            float(row["load"]) for row in rows
        } != set(LOADS):
            raise PostResultDevelopmentRefusal("source capacity case grid changed")
        if any(
            int(row["lanes"]) != lanes
            or row.get("capacity_projection_passed") is not True
            or row.get("conditional_negative_drift_certified") is not True
            or row.get("positive_effective_slack") is not True
            for row in rows
        ):
            raise PostResultDevelopmentRefusal("source capacity row changed")
        minimum_eta[str(lanes)] = min(float(row["effective_slack_eta"]) for row in rows)
    return capacity, minimum_eta


DEVELOPMENT_BOUNDARY = (
    "This is a post-result statewise development screen on the already observed "
    "candidate-state distribution from the frozen 120 tapes. It is not an "
    "independent confirmation, does not execute a rescue trajectory, and does "
    "not establish flow noninferiority, capacity, stability, closure, or a "
    "physical multi-GPU result. A selected rule and budget require a frozen run "
    "on new tapes."
)


def build_registration(
    *,
    source_registration: Path = DEFAULT_SOURCE_REGISTRATION,
    source_fixed_sources: Path = DEFAULT_SOURCE_FIXED_SOURCES,
    source_result: Path = DEFAULT_SOURCE_RESULT,
) -> dict[str, Any]:
    contract, protocol = _validated_source_bundle(
        Path(source_registration), Path(source_fixed_sources), Path(source_result)
    )
    capacity, minimum_eta = _capacity_contract(protocol)
    return {
        "schema_version": 1,
        "gate": GATE,
        "experiment": EXPERIMENT,
        "status": REGISTERED_STATUS,
        "registered": True,
        "source_paths": {
            "trajectory_registration": str(Path(source_registration).resolve()),
            "trajectory_fixed_sources": str(Path(source_fixed_sources).resolve()),
            "trajectory_result": str(Path(source_result).resolve()),
        },
        "source_transition_contract": contract,
        "physical_complete_frame_model": _json(protocol["physical_complete_frame_model"]),
        "physical_envelope_capacity_normalized_loads": capacity,
        "lane_minimum_effective_slack_eta": minimum_eta,
        "diagnostic_contract": {
            "lanes": list(LANES),
            "loads": list(LOADS),
            "arrival_seeds": list(ARRIVAL_SEEDS),
            "arrival_horizon_s": ARRIVAL_HORIZON_S,
            "tape_count": 120,
            "trajectory_replays": 120,
            "counterfactual_trajectory_runs": 0,
            "visited_state_distribution": "frozen_current_candidate",
            "state_replay_policy": (
                "observed R8 K48 single-exchange / R16 K9 conditional K144 candidate"
            ),
            "frontier_primary_by_lane": {
                "8": "fresh K48 single-exchange primary",
                "16": "fresh K9 bulk primary without old conditional K144 rescue",
            },
            "deduplicate_selector_diagnostics_by": ["lanes", "queue"],
            "rescue_budgets_by_lane": {
                str(lanes): list(RESCUE_BUDGETS) for lanes in LANES
            },
            "trigger_rules": list(TRIGGER_RULES),
            "trigger_diagnostic_components": [
                "epsilon_near_lane_min_eta",
                "canonical_inexact",
            ],
            "trigger_relation": (
                "epsilon_near_lane_min_eta implies canonical_inexact; therefore "
                "their OR is exactly canonical_inexact and is not counted as a "
                "third method"
            ),
            "trigger_tolerance": SCORE_TOLERANCE,
            "shadow_budget": REFERENCE_BUDGET,
            "each_budget_is_an_independent_fresh_cold_call": True,
            "overlapping_or_trigger_executes_one_rescue": True,
            "reference_is_not_a_trigger_input": True,
            "base_incumbent_is_never_degraded": True,
            "occurrence_weighted_work_includes_base_and_executed_rescue": True,
            "shadow_work_is_k4096_per_occurrence": True,
            "work_fields": list(WORK_FIELDS),
            "work_gate_fields": list(WORK_GATE_FIELDS),
        },
        "registration_scope": {
            "post_result_development": True,
            "source_result_observed_before_registration": True,
            "independent_confirmation": False,
            "trajectory_or_flow_evidence": False,
            "rng_initializations": 0,
            "arrival_generations": 0,
            "trajectory_replays": 0,
            "selector_calls": 0,
            "server_pulls": 0,
        },
        "decision_rule": {
            "invalid": (
                "any source/protocol/tape replay mismatch, invalid selector "
                "certificate, nonexact K4096 shadow, or incomplete work row"
            ),
            "complete": (
                "all registered observed tapes replay and every unique state/budget "
                "diagnostic is accounted; completion is not confirmation"
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
        raise PostResultDevelopmentRefusal("development artifact already exists")
    body = build_registration(**kwargs)
    protocol = {"registered_at_utc": datetime.now(timezone.utc).isoformat(), **body}
    _write_new(Path(fixed_sources), _current_fixed_sources())
    _write_new(Path(registration), protocol)
    return protocol


def _validate_protocol(protocol: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    observed = dict(protocol)
    registered_at = observed.pop("registered_at_utc", None)
    if not isinstance(registered_at, str) or observed != expected:
        raise PostResultDevelopmentRefusal("registered development protocol changed")


def _trigger_flags(selection: Any, lane_eta: float) -> dict[str, bool]:
    near = float(lane_eta) - float(selection.epsilon) <= SCORE_TOLERANCE
    inexact = not bool(selection.canonical_exact)
    if near and not inexact:
        raise PostResultDevelopmentRefusal(
            "epsilon-near-eta must imply canonical inexactness"
        )
    flags = {
        "epsilon_near_lane_min_eta": near,
        "canonical_inexact": inexact,
        "epsilon_near_eta_or_canonical_inexact": near or inexact,
    }
    if flags["epsilon_near_eta_or_canonical_inexact"] != inexact:
        raise PostResultDevelopmentRefusal(
            "registered OR trigger must equal canonical inexactness"
        )
    return flags


def _canonical_key(selection: Any, option_count: int) -> tuple[Any, ...]:
    return (
        float(selection.duration),
        tuple(option_count if value is None else int(value) for value in selection.option_indices),
    )


def _adopted_selection(base: Any, rescue: Any, option_count: int) -> tuple[Any, str]:
    if rescue.score > base.score:
        return rescue, "rescue"
    if (
        rescue.score == base.score
        and _canonical_key(rescue, option_count) < _canonical_key(base, option_count)
    ):
        return rescue, "rescue"
    return base, "base"


def _selection_certificate_valid(model: Any, queue: Sequence[int], selection: Any) -> bool:
    return bool(
        source.confirmation.canonical._selection_reconstructs(
            model, queue, model.lanes, selection
        )
        and source.confirmation.canonical._certificate_valid(selection, queue)
    )


def _diagnostic_base_call(model: Any, queue: Sequence[int]) -> dict[str, Any]:
    """Recompute the frontier primary without inheriting the old R16 K144 rescue."""
    if int(model.lanes) == 8:
        call = source._candidate_call(model, queue)
        if call["rescue_triggered"] or any(call["rescue_work"].values()):
            raise PostResultDevelopmentRefusal("R8 frontier base unexpectedly rescued")
        return {
            "selection": call["selection"],
            "work": _checked_work(call["work"]),
            "valid": bool(call["valid"]),
            "method_relation": bool(call["method_relation"]),
            "method": "r8_k48_single_exchange_primary",
            "lp_node_budget": source.CANDIDATE_BUDGETS[8],
        }
    if int(model.lanes) != 16:
        raise PostResultDevelopmentRefusal("only R8 and R16 are registered")
    primary = select_canonical_score_tie_bulk_substitution_frame(
        model.options,
        queue,
        model.lanes,
        source.CANDIDATE_BUDGETS[16],
    )
    relation = source._bulk_relation(primary, queue, model.options)
    valid = bool(
        relation
        and source._certificate_valid(
            model,
            queue,
            model.lanes,
            source.CANDIDATE_BUDGETS[16],
            primary.base_selection,
        )
        and _selection_certificate_valid(model, queue, primary.selection)
    )
    return {
        "selection": primary.selection,
        "work": _checked_work(bulk_total_work_row(primary)),
        "valid": valid,
        "method_relation": relation,
        "method": "r16_k9_bulk_primary_without_old_k144_rescue",
        "lp_node_budget": source.CANDIDATE_BUDGETS[16],
    }


def _variant(
    model: Any,
    queue: Sequence[int],
    base_call: Mapping[str, Any],
    rescue: Mapping[str, Any] | None,
    shadow: Any,
    shadow_work: Mapping[str, Any],
) -> dict[str, Any]:
    base = base_call["selection"]
    base_work = _checked_work(base_call["work"])
    if rescue is None:
        selected, adopted = base, "base"
        upper = float(base.upper_bound)
        work = base_work
        rescue_valid = True
        canonical_witnesses = (base,)
    else:
        rescue_selection = rescue["selection"]
        selected, adopted = _adopted_selection(
            base, rescue_selection, len(model.options)
        )
        upper = min(float(base.upper_bound), float(rescue_selection.upper_bound))
        work = dict(base_work)
        _add_scaled_work(work, rescue["work"], 1)
        rescue_valid = bool(rescue["valid"])
        canonical_witnesses = (base, rescue_selection)
    if float(selected.score) > upper + SCORE_TOLERANCE:
        raise PostResultDevelopmentRefusal(
            "adopted incumbent exceeds the combined certified upper bound"
        )
    gap = max(0.0, upper - float(selected.score))
    normalizer = sum(queue)
    epsilon = gap / normalizer if normalizer else 0.0
    combined_bound_closed = gap <= SCORE_TOLERANCE
    exact_witness = any(
        witness.score_exact
        and witness.option_indices == selected.option_indices
        and witness.score == selected.score
        for witness in canonical_witnesses
    )
    exact_bound_equality = upper == float(selected.score)
    score_exact = exact_witness
    canonical_exact = bool(
        score_exact
        and any(
            witness.canonical_exact
            and witness.option_indices == selected.option_indices
            and witness.score == selected.score
            for witness in canonical_witnesses
        )
    )
    regret = (
        (float(shadow.score) - float(selected.score)) / normalizer
        if normalizer
        else 0.0
    )
    return {
        "adopted": adopted,
        "option_indices": list(selected.option_indices),
        "score": float(selected.score),
        "upper_bound": upper,
        "epsilon": epsilon,
        "combined_bound_closed_within_tolerance": combined_bound_closed,
        "combined_bound_closed_by_exact_float_equality": exact_bound_equality,
        "score_exact_witness_present": exact_witness,
        "score_exact_provenance": (
            "same_action_native_selector_exact_witness" if exact_witness else None
        ),
        "score_exact": score_exact,
        "canonical_exact": canonical_exact,
        "exact": score_exact and canonical_exact,
        "certificate_valid": bool(
            base_call["valid"]
            and _selection_certificate_valid(model, queue, base)
            and rescue_valid
            and _selection_certificate_valid(model, queue, selected)
        ),
        "same_state_shadow_action_match": (
            selected.option_indices == shadow.option_indices
        ),
        "actual_normalized_regret": regret,
        "regret_bounded_by_epsilon": bool(
            -SCORE_TOLERANCE <= regret <= epsilon + SCORE_TOLERANCE
        ),
        "work": work,
        "shadow_work": _checked_work(shadow_work),
    }


def _rescue_call(model: Any, queue: Sequence[int], budget: int) -> dict[str, Any]:
    selection = select_canonical_count_space_bnb_frame(
        model.options, queue, model.lanes, budget
    )
    return {
        "selection": selection,
        "work": source._selection_work(selection),
        "valid": source._certificate_valid(
            model, queue, model.lanes, budget, selection
        ),
    }


def _tape_metadata(tape: Sequence[tuple[float, int, int]], classes: int) -> dict[str, Any]:
    jobs = [0] * classes
    for _, coordinate, _ in tape:
        jobs[coordinate] += 1
    return {
        "job_count": len(tape),
        "jobs_by_class": jobs,
        "first_arrival_s": tape[0][0] if tape else None,
        "last_arrival_s": tape[-1][0] if tape else None,
        "arrival_generation_count": 1,
        "executed_policy_count": 2,
        "shadow_trajectory_runs": 0,
    }


def _base_signature(call: Mapping[str, Any]) -> tuple[Any, ...]:
    selection = call["selection"]
    return (
        selection,
        tuple(_checked_work(call["work"]).items()),
        bool(call["valid"]),
        bool(call["method_relation"]),
    )


def _replay_candidate_tape(
    model: Any,
    tape: Sequence[tuple[float, int, int]],
    horizon: float,
    *,
    lanes: int,
    load: float,
    seed: int,
    states: dict[tuple[int, tuple[int, ...]], dict[str, Any]],
) -> int:
    queues = [deque() for _ in model.workloads]
    pending = iter(tuple(sorted(tape)))
    next_arrival = next(pending, None)
    now = 0.0
    frames = 0
    while now < horizon or next_arrival is not None or any(queues):
        while next_arrival is not None and next_arrival[0] <= now:
            at, coordinate, index = next_arrival
            queues[coordinate].append((at, index))
            next_arrival = next(pending, None)
        before = tuple(len(row) for row in queues)
        call = source._select_candidate(model, before)
        key = (lanes, before)
        record = states.get(key)
        if record is None:
            record = {
                "lanes": lanes,
                "queue": before,
                "observed_candidate_call": call,
                "occurrences_by_case": Counter(),
                "occurrences_by_tape": Counter(),
                "first_occurrence": {
                    "load": load,
                    "seed": seed,
                    "frame": frames,
                },
            }
            states[key] = record
        elif _base_signature(record["observed_candidate_call"]) != _base_signature(call):
            raise PostResultDevelopmentRefusal(
                "current candidate is not deterministic on a repeated state"
            )
        record["occurrences_by_case"][float(load)] += 1
        record["occurrences_by_tape"][(float(load), int(seed))] += 1

        selected = call["selection"]
        end = now + float(selected.duration)
        departures = [0] * len(queues)
        for option_index in selected.option_indices:
            if option_index is None:
                continue
            option = model.options[option_index]
            for coordinate, offsets in enumerate(option.completion_offsets):
                for _ in offsets:
                    if not queues[coordinate]:
                        raise PostResultDevelopmentRefusal(
                            "candidate tried to serve an empty FIFO queue"
                        )
                    queues[coordinate].popleft()
                    departures[coordinate] += 1
        if tuple(departures) != selected.service:
            raise PostResultDevelopmentRefusal(
                "candidate service differs from replayed FIFO departures"
            )
        while next_arrival is not None and next_arrival[0] <= end:
            at, coordinate, index = next_arrival
            queues[coordinate].append((at, index))
            next_arrival = next(pending, None)
        now = end
        frames += 1
    return frames


def _failure_reasons(
    base_call: Mapping[str, Any],
    shadow: Any,
    lane_eta: float,
    queue: Sequence[int],
) -> list[str]:
    selection = base_call["selection"]
    normalizer = sum(queue)
    regret = (
        (float(shadow.score) - float(selection.score)) / normalizer
        if normalizer
        else 0.0
    )
    reasons = []
    if not base_call["valid"]:
        reasons.append("current_candidate_contract_invalid")
    if not selection.score_exact:
        reasons.append("score_inexact")
    if not selection.canonical_exact:
        reasons.append("canonical_inexact")
    if selection.option_indices != shadow.option_indices:
        reasons.append("k4096_action_mismatch")
    if lane_eta - selection.epsilon <= SCORE_TOLERANCE:
        reasons.append("epsilon_not_strictly_below_lane_min_eta")
    if lane_eta - regret <= SCORE_TOLERANCE:
        reasons.append("regret_not_strictly_below_lane_min_eta")
    if not (-SCORE_TOLERANCE <= regret <= selection.epsilon + SCORE_TOLERANCE):
        reasons.append("regret_outside_certificate")
    return reasons


def _evaluate_unique_state(
    model: Any,
    record: Mapping[str, Any],
    lane_eta: float,
) -> dict[str, Any]:
    queue = tuple(record["queue"])
    observed_call = record["observed_candidate_call"]
    base_call = _diagnostic_base_call(model, queue)
    base = base_call["selection"]
    shadow, shadow_work, shadow_valid = source._reference_call(model, queue)
    if not shadow_valid or not (shadow.score_exact and shadow.canonical_exact):
        raise PostResultDevelopmentRefusal("K4096 shadow is not canonical exact")
    flags = _trigger_flags(base, lane_eta)
    rescues: dict[int, dict[str, Any]] = {}
    if flags[TRIGGER_RULES[0]]:
        for budget in RESCUE_BUDGETS:
            rescues[budget] = _rescue_call(model, queue, budget)
            if not rescues[budget]["valid"]:
                raise PostResultDevelopmentRefusal(
                    f"invalid cold rescue certificate at K{budget}"
                )
    variants: dict[tuple[str, int], dict[str, Any]] = {}
    for rule in TRIGGER_RULES:
        for budget in RESCUE_BUDGETS:
            triggered = flags[rule]
            row = _variant(
                model,
                queue,
                base_call,
                rescues[budget] if triggered else None,
                shadow,
                shadow_work,
            )
            row["triggered"] = triggered
            row["within_lane_min_eta"] = bool(
                lane_eta - row["epsilon"] > SCORE_TOLERANCE
                and lane_eta - row["actual_normalized_regret"] > SCORE_TOLERANCE
            )
            variants[(rule, budget)] = row
    normalizer = sum(queue)
    observed_selection = observed_call["selection"]
    observed_regret = (
        (float(shadow.score) - float(observed_selection.score)) / normalizer
        if normalizer
        else 0.0
    )
    base_regret = (
        (float(shadow.score) - float(base.score)) / normalizer
        if normalizer
        else 0.0
    )
    return {
        "lanes": int(record["lanes"]),
        "queue": queue,
        "total_occurrences": sum(record["occurrences_by_case"].values()),
        "occurrences_by_case": dict(record["occurrences_by_case"]),
        "occurrences_by_tape": dict(record["occurrences_by_tape"]),
        "first_occurrence": dict(record["first_occurrence"]),
        "observed_candidate": {
            "option_indices": list(observed_selection.option_indices),
            "score": float(observed_selection.score),
            "upper_bound": float(observed_selection.upper_bound),
            "epsilon": float(observed_selection.epsilon),
            "score_exact": bool(observed_selection.score_exact),
            "canonical_exact": bool(observed_selection.canonical_exact),
            "current_contract_valid": bool(observed_call["valid"]),
            "method_relation": bool(observed_call["method_relation"]),
            "work": _checked_work(observed_call["work"]),
            "same_state_shadow_action_match": (
                observed_selection.option_indices == shadow.option_indices
            ),
            "actual_normalized_regret": observed_regret,
        },
        "base_call": base_call,
        "frontier_base": {
            "option_indices": list(base.option_indices),
            "score": float(base.score),
            "upper_bound": float(base.upper_bound),
            "epsilon": float(base.epsilon),
            "score_exact": bool(base.score_exact),
            "canonical_exact": bool(base.canonical_exact),
            "current_contract_valid": bool(base_call["valid"]),
            "method_relation": bool(base_call["method_relation"]),
            "method": base_call["method"],
            "lp_node_budget": base_call["lp_node_budget"],
            "work": _checked_work(base_call["work"]),
            "same_state_shadow_action_match": (
                base.option_indices == shadow.option_indices
            ),
            "actual_normalized_regret": base_regret,
        },
        "shadow": {
            "option_indices": list(shadow.option_indices),
            "score": float(shadow.score),
            "upper_bound": float(shadow.upper_bound),
            "score_exact": bool(shadow.score_exact),
            "canonical_exact": bool(shadow.canonical_exact),
            "valid": bool(shadow_valid),
            "work": _checked_work(shadow_work),
        },
        "trigger_flags": flags,
        "failure_reasons": _failure_reasons(
            observed_call, shadow, lane_eta, queue
        ),
        "frontier_base_failure_reasons": _failure_reasons(
            base_call, shadow, lane_eta, queue
        ),
        "rescues": rescues,
        "variants": variants,
    }


def _work_comparison(
    candidate: Mapping[str, Any], shadow: Mapping[str, Any]
) -> dict[str, Any]:
    candidate_row = _checked_work(candidate)
    shadow_row = _checked_work(shadow)
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
    positive = [field for field in WORK_FIELDS if shadow_row[field] > 0]
    zero_equal = [
        field
        for field in WORK_FIELDS
        if shadow_row[field] == 0 and candidate_row[field] == 0
    ]
    return {
        "fields": fields,
        "work_gate_fields_strictly_lower": all(
            fields[field]["strictly_lower"] for field in WORK_GATE_FIELDS
        ),
        "all_positive_shadow_fields_strictly_lower": all(
            fields[field]["strictly_lower"] for field in positive
        ),
        "all_registered_fields_strictly_lower": all(
            fields[field]["strictly_lower"] for field in WORK_FIELDS
        ),
        "zero_equal_fields": zero_equal,
    }


def _scope_summary(
    evaluated: Sequence[Mapping[str, Any]],
    rule: str,
    budget: int,
    occurrence_count: Any,
    *,
    scope: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_work, shadow_work = _new_work(), _new_work()
    occurrence_total = trigger_total = adopted_total = 0
    valid_total = action_total = score_exact_total = canonical_exact_total = 0
    support_total = regret_bounded_total = 0
    unique_total = trigger_unique = 0
    maximum_epsilon = maximum_regret = 0.0
    for row in evaluated:
        count = int(occurrence_count(row))
        if count <= 0:
            continue
        unique_total += 1
        occurrence_total += count
        variant = row["variants"][(rule, budget)]
        trigger_unique += int(variant["triggered"])
        trigger_total += count * int(variant["triggered"])
        adopted_total += count * int(variant["adopted"] == "rescue")
        valid_total += count * int(variant["certificate_valid"])
        action_total += count * int(variant["same_state_shadow_action_match"])
        score_exact_total += count * int(variant["score_exact"])
        canonical_exact_total += count * int(variant["canonical_exact"])
        support_total += count * int(variant["within_lane_min_eta"])
        regret_bounded_total += count * int(variant["regret_bounded_by_epsilon"])
        maximum_epsilon = max(maximum_epsilon, variant["epsilon"])
        maximum_regret = max(
            maximum_regret, variant["actual_normalized_regret"]
        )
        _add_scaled_work(candidate_work, variant["work"], count)
        _add_scaled_work(shadow_work, variant["shadow_work"], count)
    comparison = _work_comparison(candidate_work, shadow_work)
    return {
        **scope,
        "trigger_rule": rule,
        "rescue_budget": budget,
        "unique_state_count": unique_total,
        "state_occurrence_count": occurrence_total,
        "triggered_unique_state_count": trigger_unique,
        "triggered_occurrence_count": trigger_total,
        "rescue_adopted_occurrence_count": adopted_total,
        "certificate_valid_occurrence_count": valid_total,
        "shadow_action_match_occurrence_count": action_total,
        "score_exact_occurrence_count": score_exact_total,
        "canonical_exact_occurrence_count": canonical_exact_total,
        "within_lane_min_eta_occurrence_count": support_total,
        "regret_bounded_by_epsilon_occurrence_count": regret_bounded_total,
        "maximum_epsilon": maximum_epsilon,
        "maximum_actual_normalized_regret": maximum_regret,
        "candidate_occurrence_weighted_work": candidate_work,
        "same_state_k4096_occurrence_weighted_work": shadow_work,
        "work_comparison": comparison,
        "all_occurrences_certificate_valid": valid_total == occurrence_total,
        "all_occurrences_match_shadow_action": action_total == occurrence_total,
        "all_occurrences_score_exact": score_exact_total == occurrence_total,
        "all_occurrences_canonical_exact": canonical_exact_total == occurrence_total,
        "all_occurrences_within_lane_min_eta": support_total == occurrence_total,
        "all_occurrences_regret_bounded": regret_bounded_total == occurrence_total,
    }


def _serialize_state(row: Mapping[str, Any]) -> dict[str, Any]:
    rescues = {
        str(budget): {
            "option_indices": list(item["selection"].option_indices),
            "score": float(item["selection"].score),
            "upper_bound": float(item["selection"].upper_bound),
            "epsilon": float(item["selection"].epsilon),
            "score_exact": bool(item["selection"].score_exact),
            "canonical_exact": bool(item["selection"].canonical_exact),
            "valid": bool(item["valid"]),
            "work": _checked_work(item["work"]),
        }
        for budget, item in row["rescues"].items()
    }
    variants = {
        rule: {
            str(budget): _json(row["variants"][(rule, budget)])
            for budget in RESCUE_BUDGETS
        }
        for rule in TRIGGER_RULES
    }
    return {
        "lanes": row["lanes"],
        "queue": list(row["queue"]),
        "total_occurrences": row["total_occurrences"],
        "occurrences_by_case": {
            f"{load:.12g}": count
            for load, count in sorted(row["occurrences_by_case"].items())
        },
        "first_occurrence": row["first_occurrence"],
        "observed_candidate": _json(row["observed_candidate"]),
        "frontier_base": _json(row["frontier_base"]),
        "shadow": _json(row["shadow"]),
        "trigger_flags": dict(row["trigger_flags"]),
        "failure_reasons": list(row["failure_reasons"]),
        "frontier_base_failure_reasons": list(
            row["frontier_base_failure_reasons"]
        ),
        "rescues": rescues,
        "variants": variants,
    }


def _frontier(evaluated: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cases, lanes_rows, aggregate = [], [], []
    for lanes in LANES:
        lane_members = [row for row in evaluated if row["lanes"] == lanes]
        for load in LOADS:
            for rule in TRIGGER_RULES:
                for budget in RESCUE_BUDGETS:
                    cases.append(
                        _scope_summary(
                            lane_members,
                            rule,
                            budget,
                            lambda row, selected_load=load: row[
                                "occurrences_by_case"
                            ].get(selected_load, 0),
                            scope={"scope": "case", "lanes": lanes, "load": load},
                        )
                    )
        for rule in TRIGGER_RULES:
            for budget in RESCUE_BUDGETS:
                lanes_rows.append(
                    _scope_summary(
                        lane_members,
                        rule,
                        budget,
                        lambda row: row["total_occurrences"],
                        scope={"scope": "lane", "lanes": lanes},
                    )
                )
    for rule in TRIGGER_RULES:
        for budget in RESCUE_BUDGETS:
            aggregate.append(
                _scope_summary(
                    evaluated,
                    rule,
                    budget,
                    lambda row: row["total_occurrences"],
                    scope={"scope": "aggregate"},
                )
            )
    return {"case": cases, "by_lane": lanes_rows, "aggregate": aggregate}


def _all_frontier_certificates_valid(
    evaluated: Sequence[Mapping[str, Any]],
) -> bool:
    return all(
        row["frontier_base"]["current_contract_valid"]
        and all(
            row["variants"][(rule, budget)]["certificate_valid"]
            for rule in TRIGGER_RULES
            for budget in RESCUE_BUDGETS
        )
        for row in evaluated
    )


def _postflight_revalidate(
    registration: Path,
    fixed_sources: Path,
    in_memory_protocol: Mapping[str, Any],
    source_paths: Mapping[str, str],
) -> None:
    on_disk_protocol = _read_object(Path(registration))
    if on_disk_protocol != in_memory_protocol:
        raise PostResultDevelopmentRefusal(
            "development registration changed during execution"
        )
    expected = build_registration(
        source_registration=Path(source_paths["trajectory_registration"]),
        source_fixed_sources=Path(source_paths["trajectory_fixed_sources"]),
        source_result=Path(source_paths["trajectory_result"]),
    )
    _validate_protocol(on_disk_protocol, expected)
    if _read_object(Path(fixed_sources)) != _current_fixed_sources():
        raise PostResultDevelopmentRefusal(
            "diagnostic sources changed during execution"
        )


def run_registered(
    registration: Path = DEFAULT_REGISTRATION,
    fixed_sources: Path = DEFAULT_FIXED_SOURCES,
    output: Path = DEFAULT_RESULT,
) -> dict[str, Any]:
    if Path(output).exists():
        raise PostResultDevelopmentRefusal(f"development result exists: {output}")
    protocol = _read_object(Path(registration))
    paths = protocol.get("source_paths") or {}
    if set(paths) != {
        "trajectory_registration",
        "trajectory_fixed_sources",
        "trajectory_result",
    }:
        raise PostResultDevelopmentRefusal("registered source paths changed")
    expected = build_registration(
        source_registration=Path(paths["trajectory_registration"]),
        source_fixed_sources=Path(paths["trajectory_fixed_sources"]),
        source_result=Path(paths["trajectory_result"]),
    )
    _validate_protocol(protocol, expected)
    if _read_object(Path(fixed_sources)) != _current_fixed_sources():
        raise PostResultDevelopmentRefusal("registered diagnostic sources changed")

    started = time.perf_counter()
    model = source.baseline._model_from_snapshot(
        protocol["physical_complete_frame_model"]
    )
    if _json(model.snapshot()) != protocol["physical_complete_frame_model"]:
        raise PostResultDevelopmentRefusal("physical model reconstruction changed")
    capacity_by_case = {
        (lanes, float(row["load"])): row
        for lanes in LANES
        for row in protocol["physical_envelope_capacity_normalized_loads"][str(lanes)]
    }
    source_tapes = {
        _tape_key(row["lanes"], row["load"], row["seed"]): row
        for row in protocol["source_transition_contract"]["tape_rows"]
    }
    states: dict[tuple[int, tuple[int, ...]], dict[str, Any]] = {}
    tape_rows = []
    for lanes in LANES:
        lane_model = replace(model, lanes=lanes)
        for load in LOADS:
            capacity = capacity_by_case[(lanes, load)]
            for seed in ARRIVAL_SEEDS:
                tape = tuple(
                    source.poisson_arrivals(
                        capacity["arrival_rates"], ARRIVAL_HORIZON_S, seed
                    )
                )
                metadata = _tape_metadata(tape, len(model.workloads))
                frozen = source_tapes[_tape_key(lanes, load, seed)]
                if metadata != frozen["arrival_tape"]:
                    raise PostResultDevelopmentRefusal(
                        "reconstructed arrival tape metadata changed"
                    )
                frame_count = _replay_candidate_tape(
                    lane_model,
                    tape,
                    ARRIVAL_HORIZON_S,
                    lanes=lanes,
                    load=load,
                    seed=seed,
                    states=states,
                )
                if frame_count != frozen["candidate_frame_count"]:
                    raise PostResultDevelopmentRefusal(
                        "candidate replay frame count changed"
                    )
                tape_rows.append(
                    {
                        "lanes": lanes,
                        "load": load,
                        "seed": seed,
                        "candidate_frame_count": frame_count,
                        "arrival_tape_metadata_match": True,
                    }
                )
    evaluated = []
    diagnostic_rescue_calls = diagnostic_shadow_calls = 0
    for (lanes, _), record in sorted(states.items()):
        lane_model = replace(model, lanes=lanes)
        row = _evaluate_unique_state(
            lane_model,
            record,
            float(protocol["lane_minimum_effective_slack_eta"][str(lanes)]),
        )
        evaluated.append(row)
        diagnostic_shadow_calls += 1
        diagnostic_rescue_calls += len(row["rescues"])
    frontier = _frontier(evaluated)
    failure_states = [row for row in evaluated if row["failure_reasons"]]
    expected_occurrences = sum(
        row["candidate_frame_count"]
        for row in protocol["source_transition_contract"]["tape_rows"]
    )
    observed_occurrences = sum(row["total_occurrences"] for row in evaluated)
    all_shadows_exact = all(
        row["shadow"]["valid"]
        and row["shadow"]["score_exact"]
        and row["shadow"]["canonical_exact"]
        for row in evaluated
    )
    trigger_relation_holds = all(
        row["trigger_flags"]["epsilon_near_eta_or_canonical_inexact"]
        == row["trigger_flags"]["canonical_inexact"]
        for row in evaluated
    )

    # Long-run postflight: refuse to write if any registered or consumed input
    # changed while the development diagnostic was executing.
    _postflight_revalidate(Path(registration), Path(fixed_sources), protocol, paths)
    result = {
        "schema_version": 1,
        "gate": GATE,
        "experiment": EXPERIMENT,
        "status": "COMPLETE",
        "scientific_decision": COMPLETE_DECISION,
        "protocol": protocol,
        "checks": {
            "source_post_result_contract_revalidated": True,
            "all_120_tapes_replayed": len(tape_rows) == 120,
            "all_tape_metadata_and_frame_counts_match": all(
                row["arrival_tape_metadata_match"] for row in tape_rows
            ),
            "state_occurrence_count_matches_frozen_candidate_frame_total": (
                observed_occurrences == expected_occurrences
            ),
            "every_unique_state_has_one_valid_exact_k4096_shadow": (
                diagnostic_shadow_calls == len(evaluated) and all_shadows_exact
            ),
            "or_trigger_equals_canonical_inexact_on_every_unique_state": (
                trigger_relation_holds
            ),
            "every_frontier_base_and_variant_certificate_valid": (
                _all_frontier_certificates_valid(evaluated)
            ),
            "no_counterfactual_rescue_trajectory_executed": True,
            "full_work_fields_reported": True,
            "protocol_fixed_sources_and_source_bundle_revalidated_after_execution": True,
        },
        "tape_replays": tape_rows,
        "unique_state_count": len(evaluated),
        "failure_state_count": len(failure_states),
        "state_occurrence_count": observed_occurrences,
        "frozen_candidate_frame_count_total": expected_occurrences,
        "diagnostic_selector_calls": {
            "current_candidate_replay_calls": sum(
                row["candidate_frame_count"] for row in tape_rows
            ),
            "deduplicated_frontier_primary_calls": len(evaluated),
            "deduplicated_k4096_shadow_calls": diagnostic_shadow_calls,
            "deduplicated_fresh_cold_rescue_calls": diagnostic_rescue_calls,
        },
        "unique_states": [_serialize_state(row) for row in evaluated],
        "failure_states": [
            {
                "lanes": row["lanes"],
                "queue": list(row["queue"]),
                "total_occurrences": row["total_occurrences"],
                "failure_reasons": list(row["failure_reasons"]),
            }
            for row in failure_states
        ],
        "frontier": frontier,
        "development_boundary": DEVELOPMENT_BOUNDARY,
        "independent_confirmation_eligible": False,
        "trajectory_or_flow_claim_authorized": False,
        "new_tape_confirmation_required": True,
        "server_pulls": 0,
        "counterfactual_trajectory_runs": 0,
        "timing": {"wall_s": time.perf_counter() - started},
    }
    if not all(result["checks"].values()):
        result["status"] = "INVALID"
        result["scientific_decision"] = INVALID_DECISION
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
            "unique_state_count": payload["unique_state_count"],
            "failure_state_count": payload["failure_state_count"],
            "state_occurrence_count": payload["state_occurrence_count"],
            "timing": payload["timing"],
        }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
