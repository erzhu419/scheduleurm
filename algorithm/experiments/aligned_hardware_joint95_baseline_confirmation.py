"""Registered joint95 trajectory comparison with an independent exact baseline.

Registration freezes the fresh tapes, policies, endpoint, and familywise rule
without reading a capacity, static-screen, or canonical-selector result.  A run
is permitted only after the joint95 capacity closure, the original score-only
static screen, and a separate canonical-policy physical screen all pass.

The exact renewal-frame MaxWeight comparator is implemented here with a generic
integer-programming oracle.  It does not call either count-space B&B selector.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from fractions import Fraction
import json
import math
from pathlib import Path
from statistics import mean, stdev
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.stats import t as student_t

from algorithm.canonical_count_space_bnb_frame_candidates import (
    select_canonical_count_space_bnb_frame,
)
from algorithm.experiments.aligned_hardware_frame_experiment import poisson_arrivals
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
DEFAULT_OUTPUT = ARTIFACTS / (
    "aligned_hardware_joint95_baseline_confirmation_v3_20260913.json"
)
DEFAULT_CAPACITY_REGISTRATION = ARTIFACTS / (
    "aligned_hardware_joint95_capacity_closure_20260913_protocol.json"
)
DEFAULT_CAPACITY_RESULT = ARTIFACTS / (
    "aligned_hardware_joint95_capacity_closure_20260913.json"
)
DEFAULT_STATIC_REGISTRATION = ARTIFACTS / (
    "aligned_hardware_joint95_static_selector_screen_20260913_protocol.json"
)
DEFAULT_STATIC_RESULT = ARTIFACTS / (
    "aligned_hardware_joint95_static_selector_screen_20260913.json"
)
DEFAULT_CANONICAL_REGISTRATION = ARTIFACTS / (
    "aligned_hardware_joint95_canonical_selector_screen_20260913_protocol.json"
)
DEFAULT_CANONICAL_RESULT = ARTIFACTS / (
    "aligned_hardware_joint95_canonical_selector_screen_20260913.json"
)

EXPERIMENT = "aligned_hardware_joint95_baseline_confirmation_v3"
GATE = "aligned_hardware_joint95_baseline_confirmation_v3_runner"
REGISTRATION_STATUS = "REGISTERED_AWAITING_CANONICAL_PHYSICAL_SELECTOR"
CAPACITY_EXPERIMENT = "aligned_hardware_joint95_capacity_closure_v1"
CAPACITY_GATE = "aligned_hardware_joint95_capacity_closure_runner"
STATIC_EXPERIMENT = "aligned_hardware_joint95_static_selector_screen_v1"
STATIC_GATE = "aligned_hardware_joint95_static_selector_screen_runner"
STATIC_PASS = "STATIC_SUPPORT_AND_COMPUTE_PASS_ADVANCE_TO_SEPARATE_PHYSICAL_TRAJECTORY"
CANONICAL_EXPERIMENT = "aligned_hardware_joint95_canonical_selector_screen_v1"
CANONICAL_GATE = "aligned_hardware_joint95_canonical_selector_screen_runner"
CANONICAL_PASS = "CANONICAL_PHYSICAL_SELECTOR_PASS_ADVANCE_TO_BASELINE_TRAJECTORY"

LANES = (8, 16)
LOADS = (0.50, 0.80, 0.95)
ARRIVAL_SEEDS = tuple(range(12041, 12061))
PREVIOUSLY_USED_ARRIVAL_SEEDS = (
    41,
    42,
    43,
    *range(1041, 1051),
    *range(2041, 2061),
    *range(3041, 3061),
    *range(4041, 4061),
    *range(6041, 6049),
    *range(7041, 7049),
    *range(8041, 8049),
    *range(9041, 9049),
    *range(11041, 11049),
)
STATIC_ROSTER_SEED = 10041
ARRIVAL_HORIZON_S = 600.0
EXECUTION_WORKERS = 4
FIXED_BUDGET_BY_LANE = {8: 36, 16: 72}
POLICIES = (
    "candidate_canonical_k",
    "exact_renewal_mw_milp",
    "fixed_balanced_partition",
    "simple_queue_delay",
)
COMPARATORS = (
    "exact_renewal_mw_milp",
    "fixed_balanced_partition",
    "simple_queue_delay",
)
FIXED_PARTITION_BY_LANE = {8: (3, 3, 2), 16: (6, 5, 5)}
FIXED_BATCH_WIDTHS = (3, 3, 8)
PRIMARY_ENDPOINT = "mean_flow_s"
FAMILYWISE_ALPHA = 0.05
PRIMARY_FAMILY_SIZE = len(LANES) * len(LOADS) * len(COMPARATORS)
EXACT_NONINFERIORITY_MARGIN = 0.01
NUMERIC_TOLERANCE = 1e-9

THIS_SOURCE = (
    "algorithm/experiments/aligned_hardware_joint95_baseline_confirmation.py"
)
FIXED_SOURCE_PATHS = (
    "algorithm/canonical_count_space_bnb_frame_candidates.py",
    "algorithm/experiments/aligned_hardware_frame_experiment.py",
    "algorithm/experiments/aligned_hardware_frame_model.py",
    THIS_SOURCE,
)

DECISION_RULE = {
    "INVALID_EXECUTION": "one or more registered execution checks fail",
    "BASELINE_CONFIRMATION_PASS": (
        "all six candidate-versus-exact mean-flow upper bounds are at most 1.01 "
        "and all twelve candidate-versus-heuristic upper bounds are below 1 "
        "under the frozen 18-comparison one-sided Bonferroni family"
    ),
    "BASELINE_CONFIRMATION_FAIL_STOP": (
        "execution is valid, but at least one frozen primary confidence claim fails"
    ),
}

CLAIM_BOUNDARY = (
    "A preregistered paired finite-horizon comparison on twenty fresh Poisson "
    "tapes per physical-capacity-normalized R/load case in the joint95 "
    "complete-frame model. Exact renewal-frame MaxWeight is an independently "
    "implemented deterministic correctness and noninferiority anchor. The two "
    "plainly named policies are local heuristics. This is not a physical "
    "multi-GPU measurement, steady-state theorem, or external-system reproduction."
)

PROHIBITED = (
    "reading capacity, static-screen, canonical-selector, or joint visit outcomes "
    "during registration; changing seeds, loads, horizon, policies, candidate "
    "budgets, endpoint, margin, or familywise rule after outcomes; using future "
    "arrivals in a selector; calling either project count-space B&B implementation "
    "from the independent exact-MW oracle; paper or README edits"
)


class BaselineConfirmationRefusal(RuntimeError):
    """The registration or one of its three prerequisite results is invalid."""


@dataclass(frozen=True)
class PolicySelection:
    policy: str
    option_indices: tuple[int | None, ...]
    service: tuple[int, ...]
    duration: float
    score: float
    exact: bool
    solver_calls: int = 0


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineConfirmationRefusal(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BaselineConfirmationRefusal(f"JSON root must be an object: {path}")
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


def _paths(output: Path = DEFAULT_OUTPUT) -> tuple[Path, Path, Path]:
    output = Path(output)
    return (
        output,
        output.with_name(output.stem + "_protocol.json"),
        output.with_name(output.stem + "_fixed_sources.json"),
    )


def _current_fixed_sources() -> dict[str, str]:
    return {name: (ROOT / name).read_text(encoding="utf-8") for name in FIXED_SOURCE_PATHS}


def _capacity_protocol_contract(source: Mapping[str, Any]) -> dict[str, Any]:
    if (
        source.get("schema_version") != 1
        or source.get("gate") != CAPACITY_GATE
        or source.get("experiment") != CAPACITY_EXPERIMENT
        or source.get("status") != "REGISTERED_AWAITING_JOINT95_DATA"
        or source.get("registered") is not True
    ):
        raise BaselineConfirmationRefusal("joint95 capacity registration is invalid")
    capacity = source.get("fixed_capacity_contract") or {}
    if (
        capacity.get("capacity_lanes") != list(LANES)
        or capacity.get("loads") != list(LOADS)
        or capacity.get("base_lanes") != 2
    ):
        raise BaselineConfirmationRefusal("capacity lane/load contract differs")
    scope = source.get("registration_scope") or {}
    if scope.get("joint_visit_values_read") is not False:
        raise BaselineConfirmationRefusal("capacity protocol was not outcome-blind")
    keys = (
        "schema_version",
        "gate",
        "experiment",
        "status",
        "registered",
        "source_paths",
        "predictor_anchor",
        "action_mapping",
        "point_reference_model",
        "fixed_capacity_contract",
        "score_and_envelope_contract",
        "registration_scope",
    )
    return {key: _json(source.get(key)) for key in keys}


def _static_protocol_contract(
    source: Mapping[str, Any], capacity_contract: Mapping[str, Any]
) -> dict[str, Any]:
    if (
        source.get("schema_version") != 1
        or source.get("gate") != STATIC_GATE
        or source.get("experiment") != STATIC_EXPERIMENT
        or source.get("status") != "REGISTERED_AWAITING_JOINT95_CAPACITY_CLOSURE"
        or source.get("registered") is not True
    ):
        raise BaselineConfirmationRefusal("joint95 score-only static registration is invalid")
    selector = source.get("selector_contract") or {}
    audit = source.get("uniform_margin_scaling_audit") or {}
    if (
        selector.get("fixed_budget_by_lane") != {"8": 36, "16": 72}
        or source.get("physical_load_contract", {}).get("loads") != list(LOADS)
        or source.get("model_contract", {}).get("lane_counts") != list(LANES)
        or audit.get("all_nine_action_durations_uniformly_scaled") is not True
        or audit.get("issue_count") != 0
        or source.get("registration_scope", {}).get("capacity_closure_result_read") is not False
    ):
        raise BaselineConfirmationRefusal("joint95 score-only static contract differs")
    embedded = source.get("capacity_closure_registration_contract") or {}
    if any(embedded.get(key) != capacity_contract.get(key) for key in capacity_contract):
        raise BaselineConfirmationRefusal("static screen embeds a different capacity registration")
    keys = (
        "schema_version",
        "gate",
        "experiment",
        "status",
        "registered",
        "source_paths",
        "model_contract",
        "physical_load_contract",
        "selector_contract",
        "uniform_margin_scaling_audit",
        "decision_rule",
        "claim_boundary",
        "registration_scope",
    )
    return {key: _json(source.get(key)) for key in keys}


def _canonical_protocol_contract(
    source: Mapping[str, Any], capacity_contract: Mapping[str, Any]
) -> dict[str, Any]:
    selector = source.get("selector_contract") or {}
    audit = source.get("uniform_margin_scaling_audit") or {}
    if (
        source.get("schema_version") != 1
        or source.get("gate") != CANONICAL_GATE
        or source.get("experiment") != CANONICAL_EXPERIMENT
        or source.get("status") != "REGISTERED_AWAITING_JOINT95_CAPACITY_CLOSURE"
        or source.get("registered") is not True
        or selector.get("fixed_budget_by_lane") != {"8": 36, "16": 72}
        or selector.get("cold_canonical_reference_budget") != 4096
        or source.get("physical_load_contract", {}).get("loads") != list(LOADS)
        or source.get("model_contract", {}).get("lane_counts") != list(LANES)
        or audit.get("all_nine_action_durations_uniformly_scaled") is not True
        or audit.get("issue_count") != 0
        or source.get("registration_scope", {}).get("capacity_closure_result_read") is not False
        or source.get("registration_scope", {}).get("joint_visit_values_read") is not False
    ):
        raise BaselineConfirmationRefusal("joint95 canonical selector registration differs")
    embedded = source.get("capacity_closure_registration_contract") or {}
    if any(embedded.get(key) != capacity_contract.get(key) for key in capacity_contract):
        raise BaselineConfirmationRefusal("canonical screen embeds a different capacity registration")
    keys = (
        "schema_version",
        "gate",
        "experiment",
        "status",
        "registered",
        "source_paths",
        "canonical_exact_compute_confirmation_contract",
        "legacy_static_screen_disposition",
        "model_contract",
        "physical_load_contract",
        "selector_contract",
        "uniform_margin_scaling_audit",
        "decision_rule",
        "claim_boundary",
        "registration_scope",
    )
    return {key: _json(source.get(key)) for key in keys}


def _primary_claims() -> list[dict[str, Any]]:
    rows = []
    for lanes in LANES:
        for load in LOADS:
            for comparator in COMPARATORS:
                kind = "noninferiority" if comparator == "exact_renewal_mw_milp" else "superiority"
                rows.append(
                    {
                        "lanes": lanes,
                        "load": load,
                        "candidate": "candidate_canonical_k",
                        "comparator": comparator,
                        "endpoint": PRIMARY_ENDPOINT,
                        "claim": kind,
                        "candidate_over_comparator_upper_limit": (
                            1.0 + EXACT_NONINFERIORITY_MARGIN
                            if kind == "noninferiority"
                            else 1.0
                        ),
                    }
                )
    return rows


def _protocol_constants(
    *,
    capacity_registration: Path,
    capacity_result: Path,
    static_registration: Path,
    static_result: Path,
    canonical_registration: Path,
    canonical_result: Path,
    capacity_contract: Mapping[str, Any],
    static_contract: Mapping[str, Any],
    canonical_contract: Mapping[str, Any],
) -> dict[str, Any]:
    if set(ARRIVAL_SEEDS) & set(PREVIOUSLY_USED_ARRIVAL_SEEDS):
        raise BaselineConfirmationRefusal("fresh trajectory seeds overlap prior trajectory seeds")
    if STATIC_ROSTER_SEED in ARRIVAL_SEEDS:
        raise BaselineConfirmationRefusal("arrival seeds include the static roster seed")
    return {
        "schema_version": 1,
        "gate": GATE,
        "experiment": EXPERIMENT,
        "status": REGISTRATION_STATUS,
        "registered": True,
        "source_paths": {
            "capacity_registration": str(Path(capacity_registration).resolve()),
            "capacity_result": str(Path(capacity_result).resolve()),
            "score_only_static_registration": str(Path(static_registration).resolve()),
            "score_only_static_result": str(Path(static_result).resolve()),
            "canonical_selector_registration": str(Path(canonical_registration).resolve()),
            "canonical_selector_result": str(Path(canonical_result).resolve()),
        },
        "capacity_registration_contract": _json(capacity_contract),
        "score_only_static_registration_contract": _json(static_contract),
        "canonical_selector_registration_contract": _json(canonical_contract),
        "unresolved_registration_dependency": {
            "code": "AWAITING_CANONICAL_PHYSICAL_SELECTOR",
            "reason": (
                "the already frozen static screen certifies support score/regret "
                "with the old selector but cannot certify canonical policy flow"
            ),
            "required_experiment": CANONICAL_EXPERIMENT,
            "required_gate": CANONICAL_GATE,
            "required_scientific_decision": CANONICAL_PASS,
            "required_physical_trajectory_advance": True,
            "required_baseline_trajectory_advance": True,
        },
        "scenario": {
            "node": "jtl110gpu2",
            "lanes": list(LANES),
            "loads": list(LOADS),
            "load_normalization": "fraction_of_final_joint95_physical_capacity_boundary",
            "arrival_horizon_s": ARRIVAL_HORIZON_S,
            "arrival_seeds": list(ARRIVAL_SEEDS),
            "previously_used_arrival_seeds": list(PREVIOUSLY_USED_ARRIVAL_SEEDS),
            "seeds_disjoint_from_previous_trajectories": True,
            "static_roster_seed_avoided": STATIC_ROSTER_SEED,
            "arrival_model": (
                "independent class-wise Poisson tapes using aligned-frame:{seed}:{coordinate}"
            ),
            "common_tape_rule": (
                "construct one tape per R/load/seed and replay it unchanged under all four policies"
            ),
            "dispatch": (
                "global synchronous complete-job frames, FIFO within class, arrivals "
                "during a frame eligible only at the next boundary"
            ),
            "stopping_rule": "stop arrivals at 600 seconds and drain every tape",
        },
        "policies": {
            "candidate_canonical_k": {
                "role": "candidate",
                "implementation": "canonical count-space B&B selector",
                "fixed_lp_budget_by_lane": {str(k): v for k, v in FIXED_BUDGET_BY_LANE.items()},
                "dependency": "separate joint95 canonical physical selector screen must pass",
            },
            "exact_renewal_mw_milp": {
                "role": "independent deterministic correctness and noninferiority anchor",
                "objective": "maximize Q dot departures / max(1, frame duration)",
                "information": "current released queue counts and fixed nine-action catalogue only",
                "implementation": (
                    "independent scipy.optimize.milp integer oracle over each duration threshold; "
                    "the exact-oracle call path does not invoke either count-space B&B selector"
                ),
                "tie_break": "shorter duration, then lexicographically smallest sorted option indices with idle last",
                "superiority_claimed": False,
            },
            "fixed_balanced_partition": {
                "role": "plainly named heuristic baseline",
                "lane_partitions": {str(k): list(v) for k, v in FIXED_PARTITION_BY_LANE.items()},
                "fixed_dedicated_batch_widths": list(FIXED_BATCH_WIDTHS),
                "tail_rule": "use the class p1 action inside its fixed partition",
                "unused_partition_lanes": "idle rather than reassigned",
            },
            "simple_queue_delay": {
                "role": "plainly named heuristic baseline",
                "rule": (
                    "repeatedly serve the longest visible queue with its feasible "
                    "dedicated action having maximum jobs per second"
                ),
                "delay_interpretation": "queue length is the same-information delay proxy",
            },
        },
        "statistics": {
            "primary_endpoint": PRIMARY_ENDPOINT,
            "transform": "paired log(candidate/comparator)",
            "estimator": "geometric mean paired ratio",
            "interval": "one-sided Student-t upper confidence bound",
            "familywise_method": "Bonferroni",
            "familywise_alpha": FAMILYWISE_ALPHA,
            "primary_family_size": PRIMARY_FAMILY_SIZE,
            "per_claim_alpha": FAMILYWISE_ALPHA / PRIMARY_FAMILY_SIZE,
            "per_claim_one_sided_confidence_level": 1.0 - FAMILYWISE_ALPHA / PRIMARY_FAMILY_SIZE,
            "exact_mw_noninferiority_margin_ratio": 1.0 + EXACT_NONINFERIORITY_MARGIN,
            "heuristic_superiority_upper_ratio": 1.0,
            "primary_claims": _primary_claims(),
            "secondary_endpoints": {
                "makespan_s": "paired geometric mean ratio, descriptive only",
                "p90_flow_s": "paired geometric mean ratio, descriptive only",
            },
            "complete_case_rule": "all twenty registered seeds are required in every claim",
            "decision_rule": "every one of the eighteen frozen primary claims must pass",
        },
        "expected_work": {
            "case_seed_tapes": len(LANES) * len(LOADS) * len(ARRIVAL_SEEDS),
            "trajectory_runs": len(LANES) * len(LOADS) * len(ARRIVAL_SEEDS) * len(POLICIES),
            "policies_per_tape": len(POLICIES),
            "primary_confidence_claims": PRIMARY_FAMILY_SIZE,
            "execution_workers": EXECUTION_WORKERS,
        },
        "run_preconditions": [
            "the frozen capacity and score-only static registrations are unchanged",
            "joint95 coverage passes and the capacity result establishes positive physical capacity",
            "the score-only physical static screen reaches its registered PASS branch",
            "a separate canonical physical selector screen reaches its registered PASS branch",
            "all three results carry exactly the same joint95 complete-frame model",
            "the canonical result certifies K36 for R8 and K72 for R16 and explicitly advances this baseline trajectory",
        ],
        "decision_rule": dict(DECISION_RULE),
        "claim_boundary": CLAIM_BOUNDARY,
        "prohibited": PROHIBITED,
        "registration_scope": {
            "capacity_protocol_read": True,
            "score_only_static_protocol_read": True,
            "capacity_result_read": False,
            "score_only_static_result_read": False,
            "canonical_selector_protocol_read": True,
            "canonical_selector_result_read": False,
            "joint_visit_values_read": False,
            "physical_model_constructed": False,
            "capacity_computed": False,
            "arrival_tape_constructed": False,
            "selector_invoked": False,
            "rng_initializations": 0,
            "rng_draws": 0,
            "paper_edited": False,
        },
    }


def register(
    output: Path = DEFAULT_OUTPUT,
    *,
    capacity_registration: Path = DEFAULT_CAPACITY_REGISTRATION,
    capacity_result: Path = DEFAULT_CAPACITY_RESULT,
    static_registration: Path = DEFAULT_STATIC_REGISTRATION,
    static_result: Path = DEFAULT_STATIC_RESULT,
    canonical_registration: Path = DEFAULT_CANONICAL_REGISTRATION,
    canonical_result: Path = DEFAULT_CANONICAL_RESULT,
) -> dict[str, Any]:
    """Freeze the comparison without reading any terminal prerequisite result."""

    result_path, protocol_path, fixed_path = _paths(output)
    if any(path.exists() for path in (result_path, protocol_path, fixed_path)):
        raise BaselineConfirmationRefusal("baseline-confirmation artifacts already exist")
    capacity_source = _read_object(Path(capacity_registration))
    capacity_contract = _capacity_protocol_contract(capacity_source)
    static_source = _read_object(Path(static_registration))
    static_contract = _static_protocol_contract(static_source, capacity_contract)
    canonical_source = _read_object(Path(canonical_registration))
    canonical_contract = _canonical_protocol_contract(
        canonical_source, capacity_contract
    )
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        **_protocol_constants(
            capacity_registration=capacity_registration,
            capacity_result=capacity_result,
            static_registration=static_registration,
            static_result=static_result,
            canonical_registration=canonical_registration,
            canonical_result=canonical_result,
            capacity_contract=capacity_contract,
            static_contract=static_contract,
            canonical_contract=canonical_contract,
        ),
    }
    _write_new(protocol_path, protocol, compact=True)
    _write_new(fixed_path, _current_fixed_sources())
    return protocol


def _model_from_snapshot(snapshot: Mapping[str, Any]) -> FrameModel:
    try:
        options = tuple(
            FrameOption(
                action_id=str(row["action_id"]),
                requirements=tuple(int(value) for value in row["requirements"]),
                duration=float(row["duration"]),
                completion_offsets=tuple(
                    tuple(float(value) for value in offsets)
                    for offsets in row["completion_offsets"]
                ),
                loaded=bool(row.get("loaded")),
                source_action_ids=tuple(str(value) for value in row.get("source_action_ids") or ()),
                phases=tuple((str(key), float(value)) for key, value in row.get("phases") or ()),
            )
            for row in snapshot["options"]
        )
        return FrameModel(
            node=str(snapshot["node"]),
            scenario_id=str(snapshot["scenario_id"]),
            lanes=int(snapshot["lanes"]),
            workloads=tuple(str(value) for value in snapshot["workloads"]),
            scales=tuple(float(value) for value in snapshot["scales"]),
            options=options,
            excluded_actions=tuple(dict(row) for row in snapshot.get("excluded_actions") or ()),
            model_scope=str(snapshot.get("model_scope") or ""),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise BaselineConfirmationRefusal(f"invalid physical model snapshot: {exc}") from exc


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    paths = protocol.get("source_paths") or {}
    required_paths = {
        "capacity_registration",
        "capacity_result",
        "score_only_static_registration",
        "score_only_static_result",
        "canonical_selector_registration",
        "canonical_selector_result",
    }
    if (
        protocol.get("schema_version") != 1
        or protocol.get("gate") != GATE
        or protocol.get("experiment") != EXPERIMENT
        or protocol.get("status") != REGISTRATION_STATUS
        or protocol.get("registered") is not True
        or protocol.get("decision_rule") != DECISION_RULE
        or protocol.get("claim_boundary") != CLAIM_BOUNDARY
        or protocol.get("prohibited") != PROHIBITED
        or set(paths) != required_paths
        or protocol.get("statistics", {}).get("primary_claims") != _primary_claims()
        or protocol.get("statistics", {}).get("primary_family_size") != PRIMARY_FAMILY_SIZE
        or protocol.get("scenario", {}).get("arrival_seeds") != list(ARRIVAL_SEEDS)
    ):
        raise BaselineConfirmationRefusal("baseline-confirmation registration changed")


def _campaign_fail_stop(capacity_protocol: Mapping[str, Any]) -> dict[str, Any] | None:
    results_dir = Path(
        (capacity_protocol.get("source_paths") or {}).get(
            "joint_results_directory", ""
        )
    )
    path = results_dir / "campaign_fail_stop.json"
    if not path.is_file():
        return None
    payload = _read_object(path)
    if (
        payload.get("schema_version") != 1
        or payload.get("gate") != "critical_gpu_joint95_campaign_fail_stop"
        or payload.get("status") != "FAIL_STOP"
        or payload.get("terminal") is not True
        or payload.get("pass") is not False
        or payload.get("adoption", {}).get("baseline_confirmation_authorized") is not False
    ):
        raise BaselineConfirmationRefusal("joint95 campaign fail-stop artifact is invalid")
    return payload


def _runtime_context(protocol: Mapping[str, Any]) -> tuple[FrameModel, dict[tuple[int, float], tuple[float, ...]], dict[str, Any]]:
    paths = protocol["source_paths"]
    capacity_protocol = _read_object(Path(paths["capacity_registration"]))
    capacity_contract = _capacity_protocol_contract(capacity_protocol)
    if capacity_contract != protocol.get("capacity_registration_contract"):
        raise BaselineConfirmationRefusal("capacity registration changed after baseline registration")
    static_protocol = _read_object(Path(paths["score_only_static_registration"]))
    static_contract = _static_protocol_contract(static_protocol, capacity_contract)
    if static_contract != protocol.get("score_only_static_registration_contract"):
        raise BaselineConfirmationRefusal("score-only static registration changed after baseline registration")

    canonical_protocol = _read_object(Path(paths["canonical_selector_registration"]))
    canonical_contract = _canonical_protocol_contract(
        canonical_protocol, capacity_contract
    )
    if canonical_contract != protocol.get("canonical_selector_registration_contract"):
        raise BaselineConfirmationRefusal(
            "canonical selector registration changed after baseline registration"
        )
    expected = _protocol_constants(
        capacity_registration=Path(paths["capacity_registration"]),
        capacity_result=Path(paths["capacity_result"]),
        static_registration=Path(paths["score_only_static_registration"]),
        static_result=Path(paths["score_only_static_result"]),
        canonical_registration=Path(paths["canonical_selector_registration"]),
        canonical_result=Path(paths["canonical_selector_result"]),
        capacity_contract=capacity_contract,
        static_contract=static_contract,
        canonical_contract=canonical_contract,
    )
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise BaselineConfirmationRefusal(
            "baseline-confirmation protocol constants changed after registration"
        )

    campaign_stop = _campaign_fail_stop(capacity_protocol)
    if campaign_stop is not None:
        raise BaselineConfirmationRefusal(
            "upstream joint95 campaign is terminal FAIL_STOP: "
            f"{campaign_stop.get('decision')} at wave "
            f"{campaign_stop.get('campaign', {}).get('failed_wave')}; "
            "baseline confirmation is not authorized"
        )

    missing = [
        label
        for label in (
            "capacity_result",
            "score_only_static_result",
            "canonical_selector_result",
        )
        if not Path(paths[label]).is_file()
    ]
    if missing:
        raise BaselineConfirmationRefusal(
            "prerequisite results unavailable before trajectory execution: "
            + ", ".join(missing)
        )

    capacity = _read_object(Path(paths["capacity_result"]))
    if (
        capacity.get("schema_version") != 1
        or capacity.get("gate") != CAPACITY_GATE
        or capacity.get("experiment") != CAPACITY_EXPERIMENT
        or capacity.get("status") != "PASS"
        or capacity.get("protocol") != capacity_protocol
        or capacity.get("joint95_audit", {}).get("holdout_covered") is not True
        or capacity.get("capacity_established") is not True
        or not math.isfinite(float(capacity.get("capacity", {}).get("common_capacity_scale", 0.0)))
        or float(capacity.get("capacity", {}).get("common_capacity_scale", 0.0)) <= 0.0
    ):
        raise BaselineConfirmationRefusal("joint95 coverage/capacity prerequisite has not passed")

    static = _read_object(Path(paths["score_only_static_result"]))
    if (
        static.get("schema_version") != 1
        or static.get("gate") != STATIC_GATE
        or static.get("experiment") != STATIC_EXPERIMENT
        or static.get("status") != "PASS"
        or static.get("protocol") != static_protocol
        or static.get("scientific_decision") != STATIC_PASS
        or static.get("physical_trajectory_advance") is not True
        or not all((static.get("checks") or {}).values())
    ):
        raise BaselineConfirmationRefusal("joint95 score-only static screen has not passed")

    canonical = _read_object(Path(paths["canonical_selector_result"]))
    if (
        canonical.get("schema_version") != 1
        or canonical.get("gate") != CANONICAL_GATE
        or canonical.get("experiment") != CANONICAL_EXPERIMENT
        or canonical.get("status") != "PASS"
        or canonical.get("protocol") != canonical_protocol
        or canonical.get("scientific_decision") != CANONICAL_PASS
        or canonical.get("physical_trajectory_advance") is not True
        or canonical.get("baseline_trajectory_advance") is not True
        or not all((canonical.get("checks") or {}).values())
    ):
        raise BaselineConfirmationRefusal("canonical physical selector prerequisite has not passed")

    capacity_model = (capacity.get("physical_complete_frame_envelope") or {}).get("model")
    static_model = (static.get("physical_complete_frame_envelope") or {}).get("model")
    canonical_model = (canonical.get("physical_complete_frame_envelope") or {}).get("model")
    if not capacity_model or capacity_model != static_model or capacity_model != canonical_model:
        raise BaselineConfirmationRefusal("the prerequisite physical frame models differ")
    model = _model_from_snapshot(capacity_model)
    if len(model.options) != 9 or len(model.workloads) != 3:
        raise BaselineConfirmationRefusal("the physical model is not the frozen nine-action scenario")

    lane_budgets = {
        int(row.get("lanes")): int(row.get("fixed_lp_budget", -1))
        for row in canonical.get("lane_summaries") or ()
    }
    if lane_budgets != FIXED_BUDGET_BY_LANE:
        raise BaselineConfirmationRefusal("canonical selector result does not certify K36/K72")

    rates: dict[tuple[int, float], tuple[float, ...]] = {}
    physical_rows = canonical.get("physical_capacity_normalized_loads") or {}
    if physical_rows != static.get("physical_capacity_normalized_loads"):
        raise BaselineConfirmationRefusal(
            "canonical and score-only static physical load rows differ"
        )
    for lanes in LANES:
        for row in physical_rows.get(str(lanes)) or ():
            key = (lanes, float(row.get("load")))
            values = tuple(float(value) for value in row.get("arrival_rates") or ())
            if len(values) == len(model.workloads) and all(value > 0 for value in values):
                rates[key] = values
    if set(rates) != {(lanes, load) for lanes in LANES for load in LOADS}:
        raise BaselineConfirmationRefusal("physical-capacity-normalized arrival rows are incomplete")
    return model, rates, {"capacity": capacity, "static": static, "canonical": canonical}


def _selection_from_indices(
    options: Sequence[FrameOption],
    queues: Sequence[int],
    lanes: int,
    indices: Sequence[int | None],
    *,
    policy: str,
    exact: bool,
    solver_calls: int = 0,
) -> PolicySelection:
    if len(indices) != lanes:
        raise ValueError("selection must provide one option or idle for every lane")
    normalized = tuple(sorted(indices, key=lambda index: len(options) if index is None else index))
    service = tuple(
        sum(
            options[index].requirements[coordinate]
            for index in normalized
            if index is not None
        )
        for coordinate in range(len(queues))
    )
    if any(done > waiting for done, waiting in zip(service, queues, strict=True)):
        raise ValueError("selection exceeds the current visible queue")
    duration = max((
        1.0,
        *(options[index].duration for index in normalized if index is not None),
    ))
    score = sum(q * d for q, d in zip(queues, service, strict=True)) / duration
    return PolicySelection(
        policy=policy,
        option_indices=normalized,
        service=service,
        duration=duration,
        score=score,
        exact=exact,
        solver_calls=solver_calls,
    )


def _milp_counts(
    requirements: np.ndarray,
    weights: np.ndarray,
    queues: Sequence[int],
    lanes: int,
    eligible: Sequence[bool],
    *,
    primary_numerator: int | None = None,
    lex_objective: np.ndarray | None = None,
    presolve: bool = True,
) -> tuple[int, ...]:
    option_count = requirements.shape[1]
    upper = np.asarray([lanes if keep else 0 for keep in eligible], dtype=float)
    matrix = [np.ones(option_count, dtype=float), *requirements]
    lower = [-np.inf] * (1 + len(queues))
    upper_rows = [float(lanes), *(float(value) for value in queues)]
    if primary_numerator is not None:
        matrix.append(weights)
        lower.append(float(primary_numerator))
        upper_rows.append(float(primary_numerator))
    objective = weights if lex_objective is None else lex_objective
    result = milp(
        c=-np.asarray(objective, dtype=float),
        integrality=np.ones(option_count, dtype=int),
        bounds=Bounds(np.zeros(option_count), upper),
        constraints=LinearConstraint(np.asarray(matrix), np.asarray(lower), np.asarray(upper_rows)),
        options={"mip_rel_gap": 0.0, "presolve": presolve},
    )
    if not result.success or result.x is None:
        raise ValueError(f"independent exact-MW MILP failed: {result.message}")
    rounded = tuple(int(round(value)) for value in result.x)
    if any(abs(value - integer) > 1e-6 for value, integer in zip(result.x, rounded, strict=True)):
        raise ValueError("independent exact-MW MILP returned a nonintegral solution")
    if sum(rounded) > lanes:
        raise ValueError("independent exact-MW MILP exceeds lane count")
    service = requirements @ np.asarray(rounded, dtype=float)
    if any(value > queue + NUMERIC_TOLERANCE for value, queue in zip(service, queues, strict=True)):
        raise ValueError("independent exact-MW MILP exceeds visible queue")
    if primary_numerator is not None:
        reconstructed = sum(int(weight) * count for weight, count in zip(weights, rounded, strict=True))
        if reconstructed != primary_numerator:
            raise ValueError("independent exact-MW lex solve changed the primary optimum")
    return rounded


def select_exact_renewal_mw(
    options: Sequence[FrameOption], queues: Sequence[int], lanes: int
) -> PolicySelection:
    """Solve exact renewal-frame MaxWeight independently of project B&B code."""

    options = tuple(options)
    queue = tuple(int(value) for value in queues)
    if lanes < 1 or any(value < 0 for value in queue):
        raise ValueError("lanes and queues must be nonnegative integers")
    if not any(queue):
        return _selection_from_indices(
            options, queue, lanes, (None,) * lanes,
            policy="exact_renewal_mw_milp", exact=True,
        )
    requirements = np.asarray([option.requirements for option in options], dtype=int).T
    if requirements.shape != (len(queue), len(options)):
        raise ValueError("option requirements do not match queue dimension")
    durations = tuple(Fraction.from_float(max(1.0, float(option.duration))) for option in options)
    weights_int = tuple(
        sum(q * r for q, r in zip(queue, option.requirements, strict=True))
        for option in options
    )
    weights = np.asarray(weights_int, dtype=float)
    thresholds = tuple(sorted(set(durations)))
    solved: dict[Fraction, tuple[int, tuple[int, ...]]] = {}
    best_score = Fraction(0)
    best_duration = Fraction.from_float(1.0)
    for threshold in thresholds:
        eligible = tuple(duration <= threshold for duration in durations)
        counts = _milp_counts(requirements, weights, queue, lanes, eligible)
        numerator = sum(weight * count for weight, count in zip(weights_int, counts, strict=True))
        used = [durations[index] for index, count in enumerate(counts) if count]
        actual_duration = max((Fraction.from_float(1.0), *used))
        score = Fraction(numerator) / actual_duration
        solved[threshold] = (numerator, counts)
        if score > best_score or (score == best_score and actual_duration < best_duration):
            best_score = score
            best_duration = actual_duration

    primary_numerator, _ = solved[best_duration]
    base = lanes + 1
    lex = np.asarray(
        [base ** (len(options) - 1 - index) for index in range(len(options))],
        dtype=float,
    )
    counts = _milp_counts(
        requirements,
        weights,
        queue,
        lanes,
        tuple(duration <= best_duration for duration in durations),
        primary_numerator=primary_numerator,
        lex_objective=lex,
        # HiGHS presolve can incorrectly eliminate the feasible set when this
        # exact integer objective row is a linear combination of the queue
        # constraints.  Keep presolve for the primary solves and disable it
        # only for the deterministic tie solve.
        presolve=False,
    )
    expanded = tuple(index for index, count in enumerate(counts) for _ in range(count))
    selected = _selection_from_indices(
        options,
        queue,
        lanes,
        expanded + (None,) * (lanes - len(expanded)),
        policy="exact_renewal_mw_milp",
        exact=True,
        solver_calls=len(thresholds) + 1,
    )
    if Fraction.from_float(selected.score) != best_score:
        # Float division need not preserve the exact rational bit-for-bit.
        if not math.isclose(selected.score, float(best_score), rel_tol=0.0, abs_tol=NUMERIC_TOLERANCE):
            raise ValueError("independent exact-MW tie solve changed the optimum score")
    return selected


def _dedicated_option_index(
    options: Sequence[FrameOption], coordinate: int, width: int
) -> int:
    target = tuple(width if index == coordinate else 0 for index in range(3))
    matches = [
        index for index, option in enumerate(options)
        if not option.loaded and option.requirements == target
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one dedicated class-{coordinate} p{width} option")
    return matches[0]


def select_fixed_balanced_partition(
    options: Sequence[FrameOption], queues: Sequence[int], lanes: int
) -> PolicySelection:
    partition = FIXED_PARTITION_BY_LANE.get(lanes)
    if partition is None:
        raise ValueError("fixed partition is registered only for R8 and R16")
    indices: list[int | None] = []
    for coordinate, slots in enumerate(partition):
        remaining = int(queues[coordinate])
        batch_width = FIXED_BATCH_WIDTHS[coordinate]
        batch_index = _dedicated_option_index(options, coordinate, batch_width)
        p1_index = _dedicated_option_index(options, coordinate, 1)
        while slots and remaining >= batch_width:
            indices.append(batch_index)
            remaining -= batch_width
            slots -= 1
        while slots and remaining:
            indices.append(p1_index)
            remaining -= 1
            slots -= 1
        indices.extend([None] * slots)
    return _selection_from_indices(
        options, queues, lanes, indices,
        policy="fixed_balanced_partition", exact=False,
    )


def select_simple_queue_delay(
    options: Sequence[FrameOption], queues: Sequence[int], lanes: int
) -> PolicySelection:
    remaining = [int(value) for value in queues]
    indices: list[int | None] = []
    while len(indices) < lanes and any(remaining):
        coordinate = min(
            (index for index, value in enumerate(remaining) if value),
            key=lambda index: (-remaining[index], index),
        )
        feasible = []
        for index, option in enumerate(options):
            if option.loaded:
                continue
            required = option.requirements[coordinate]
            if (
                required > 0
                and required <= remaining[coordinate]
                and sum(option.requirements) == required
            ):
                feasible.append(index)
        if not feasible:
            raise ValueError("simple delay policy cannot serve a nonempty class")
        chosen = min(
            feasible,
            key=lambda index: (
                -sum(options[index].requirements) / max(1.0, options[index].duration),
                options[index].duration,
                index,
            ),
        )
        indices.append(chosen)
        for index, required in enumerate(options[chosen].requirements):
            remaining[index] -= required
    indices.extend([None] * (lanes - len(indices)))
    return _selection_from_indices(
        options, queues, lanes, indices,
        policy="simple_queue_delay", exact=False,
    )


def _candidate_selection(
    options: Sequence[FrameOption], queues: Sequence[int], lanes: int
) -> tuple[PolicySelection, dict[str, Any]]:
    budget = FIXED_BUDGET_BY_LANE[lanes]
    native = select_canonical_count_space_bnb_frame(options, queues, lanes, budget)
    selected = _selection_from_indices(
        options,
        queues,
        lanes,
        native.option_indices,
        policy="candidate_canonical_k",
        exact=bool(native.canonical_exact),
        solver_calls=int(native.lp_calls),
    )
    if (
        not math.isclose(selected.score, native.score, rel_tol=0.0, abs_tol=NUMERIC_TOLERANCE)
        or tuple(selected.service) != tuple(native.service)
        or selected.duration != native.duration
        or native.score > native.upper_bound + NUMERIC_TOLERANCE
        or native.gap < -NUMERIC_TOLERANCE
    ):
        raise ValueError("candidate canonical selector certificate does not reconstruct")
    return selected, {
        "lp_budget": budget,
        "lp_calls": int(native.lp_calls),
        "score_exact": bool(native.score_exact),
        "canonical_exact": bool(native.canonical_exact),
        "gap": float(native.gap),
        "epsilon": float(native.epsilon),
    }


def _simulate_policy(
    model: FrameModel,
    arrivals: Sequence[tuple[float, int, int]],
    horizon: float,
    policy: str,
) -> dict[str, Any]:
    queues = [deque() for _ in model.workloads]
    pending = iter(tuple(sorted(arrivals)))
    next_arrival = next(pending, None)
    now = 0.0
    flows: list[float] = []
    last_completion = 0.0
    frames = 0
    valid_frames = 0
    exact_frames = 0
    solver_calls = 0
    candidate_score_exact = 0
    candidate_canonical_exact = 0
    candidate_max_gap = 0.0
    selected_counts: Counter[str] = Counter()

    selectors: dict[str, Callable[..., PolicySelection]] = {
        "exact_renewal_mw_milp": select_exact_renewal_mw,
        "fixed_balanced_partition": select_fixed_balanced_partition,
        "simple_queue_delay": select_simple_queue_delay,
    }
    while now < horizon or next_arrival is not None or any(queues):
        while next_arrival is not None and next_arrival[0] <= now:
            arrival, coordinate, index = next_arrival
            queues[coordinate].append((arrival, index))
            next_arrival = next(pending, None)
        before = tuple(len(queue) for queue in queues)
        detail: dict[str, Any] = {}
        if policy == "candidate_canonical_k":
            selected, detail = _candidate_selection(model.options, before, model.lanes)
        else:
            selected = selectors[policy](model.options, before, model.lanes)
        frames += 1
        solver_calls += selected.solver_calls
        exact_frames += int(selected.exact)
        candidate_score_exact += int(detail.get("score_exact", False))
        candidate_canonical_exact += int(detail.get("canonical_exact", False))
        candidate_max_gap = max(candidate_max_gap, float(detail.get("gap", 0.0)))
        end = now + selected.duration
        departures = [0] * len(queues)
        for option_index in selected.option_indices:
            if option_index is None:
                continue
            option = model.options[option_index]
            selected_counts[option.action_id] += 1
            for coordinate, offsets in enumerate(option.completion_offsets):
                for offset in offsets:
                    arrival, _ = queues[coordinate].popleft()
                    completion = now + offset
                    flows.append(completion - arrival)
                    last_completion = max(last_completion, completion)
                    departures[coordinate] += 1
        if tuple(departures) != selected.service:
            raise ValueError("executed departures differ from selected service")
        valid_frames += 1
        while next_arrival is not None and next_arrival[0] <= end:
            arrival, coordinate, index = next_arrival
            queues[coordinate].append((arrival, index))
            next_arrival = next(pending, None)
        now = end

    ordered = sorted(flows)
    if len(flows) != len(arrivals):
        raise ValueError("trajectory did not drain every registered job")
    return {
        "policy": policy,
        "job_count": len(arrivals),
        "completed_jobs": len(flows),
        "frame_count": frames,
        "valid_selection_frame_count": valid_frames,
        "exact_selection_frame_count": exact_frames,
        "solver_calls": solver_calls,
        "candidate_score_exact_frame_count": candidate_score_exact,
        "candidate_canonical_exact_frame_count": candidate_canonical_exact,
        "candidate_maximum_score_gap": candidate_max_gap,
        "elapsed_including_drain_s": now,
        "makespan_s": last_completion,
        "mean_flow_s": mean(flows),
        "p90_flow_s": ordered[math.ceil(0.9 * len(ordered)) - 1],
        "selected_option_counts": dict(selected_counts),
    }


def _run_case_seed(task: tuple[FrameModel, int, float, int, tuple[float, ...]]) -> dict[str, Any]:
    base_model, lanes, load, seed, rates = task
    model = replace(base_model, lanes=lanes)
    tape = poisson_arrivals(rates, ARRIVAL_HORIZON_S, seed)
    runs = {
        policy: _simulate_policy(model, tape, ARRIVAL_HORIZON_S, policy)
        for policy in POLICIES
    }
    return {
        "lanes": lanes,
        "load": load,
        "seed": seed,
        "arrival_rates": list(rates),
        "arrival_tape_generation_count": 1,
        "arrival_count": len(tape),
        "policies": runs,
    }


def _build_tasks(
    model: FrameModel,
    rates: Mapping[tuple[int, float], tuple[float, ...]],
) -> list[tuple[FrameModel, int, float, int, tuple[float, ...]]]:
    return [
        (model, lanes, load, seed, rates[(lanes, load)])
        for lanes in LANES
        for load in LOADS
        for seed in ARRIVAL_SEEDS
    ]


def _paired_ratio_summary(
    rows: Sequence[Mapping[str, Any]], lanes: int, load: float, comparator: str
) -> dict[str, Any]:
    chosen = sorted(
        (row for row in rows if row["lanes"] == lanes and row["load"] == load),
        key=lambda row: row["seed"],
    )
    if [row["seed"] for row in chosen] != list(ARRIVAL_SEEDS):
        raise ValueError("paired confidence summary lacks the complete frozen seed roster")
    ratios = [
        row["policies"]["candidate_canonical_k"][PRIMARY_ENDPOINT]
        / row["policies"][comparator][PRIMARY_ENDPOINT]
        for row in chosen
    ]
    logs = [math.log(value) for value in ratios]
    n = len(logs)
    log_mean = mean(logs)
    log_se = stdev(logs) / math.sqrt(n)
    critical = float(student_t.ppf(1.0 - FAMILYWISE_ALPHA / PRIMARY_FAMILY_SIZE, n - 1))
    upper = math.exp(log_mean + critical * log_se)
    claim = "noninferiority" if comparator == "exact_renewal_mw_milp" else "superiority"
    limit = 1.0 + EXACT_NONINFERIORITY_MARGIN if claim == "noninferiority" else 1.0
    secondary = {}
    for endpoint in ("makespan_s", "p90_flow_s"):
        secondary[endpoint] = math.exp(mean([
            math.log(
                row["policies"]["candidate_canonical_k"][endpoint]
                / row["policies"][comparator][endpoint]
            )
            for row in chosen
        ]))
    return {
        "lanes": lanes,
        "load": load,
        "candidate": "candidate_canonical_k",
        "comparator": comparator,
        "endpoint": PRIMARY_ENDPOINT,
        "claim": claim,
        "sample_size": n,
        "paired_ratios_by_seed": [
            {"seed": row["seed"], "candidate_over_comparator": ratio}
            for row, ratio in zip(chosen, ratios, strict=True)
        ],
        "geometric_mean_candidate_over_comparator": math.exp(log_mean),
        "one_sided_simultaneous_upper_ratio": upper,
        "upper_ratio_limit": limit,
        "confirmed": upper <= limit if claim == "noninferiority" else upper < limit,
        "critical_t": critical,
        "degrees_of_freedom": n - 1,
        "secondary_descriptive_geometric_mean_ratios": secondary,
    }


def _decision(execution_passed: bool, claims: Sequence[Mapping[str, Any]]) -> str:
    if not execution_passed:
        return "INVALID_EXECUTION"
    if len(claims) == PRIMARY_FAMILY_SIZE and all(row["confirmed"] for row in claims):
        return "BASELINE_CONFIRMATION_PASS"
    return "BASELINE_CONFIRMATION_FAIL_STOP"


def run_registered(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    result_path, protocol_path, fixed_path = _paths(output)
    if result_path.exists():
        raise BaselineConfirmationRefusal(f"baseline-confirmation result already exists: {result_path}")
    protocol = _read_object(protocol_path)
    _validate_protocol(protocol)
    fixed = _read_object(fixed_path)
    if fixed != _current_fixed_sources():
        raise BaselineConfirmationRefusal("registered baseline-confirmation sources changed")

    # Every prerequisite is checked before a model is constructed, a tape is
    # generated, or any selector is invoked.
    model, rates, dependencies = _runtime_context(protocol)
    tasks = _build_tasks(model, rates)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=EXECUTION_WORKERS) as pool:
        rows = list(pool.map(_run_case_seed, tasks))

    claims = [
        _paired_ratio_summary(rows, lanes, load, comparator)
        for lanes in LANES
        for load in LOADS
        for comparator in COMPARATORS
    ]
    checks = {
        "joint95_coverage_and_positive_capacity_passed": True,
        "score_only_static_support_and_compute_screen_passed": True,
        "canonical_physical_selector_screen_passed": True,
        "all_dependencies_use_the_same_physical_model": True,
        "fresh_seed_roster_is_disjoint": not (set(ARRIVAL_SEEDS) & set(PREVIOUSLY_USED_ARRIVAL_SEEDS)),
        "static_roster_seed_is_avoided": STATIC_ROSTER_SEED not in ARRIVAL_SEEDS,
        "one_common_tape_per_case_seed": (
            len(rows) == len(tasks)
            and all(row["arrival_tape_generation_count"] == 1 for row in rows)
        ),
        "all_four_policies_run_once_per_tape": all(set(row["policies"]) == set(POLICIES) for row in rows),
        "all_jobs_complete_under_every_policy": all(
            metrics["job_count"] == metrics["completed_jobs"] == row["arrival_count"]
            for row in rows for metrics in row["policies"].values()
        ),
        "every_returned_selection_reconstructs": all(
            metrics["valid_selection_frame_count"] == metrics["frame_count"]
            for row in rows for metrics in row["policies"].values()
        ),
        "independent_exact_mw_is_exact_in_every_frame": all(
            row["policies"]["exact_renewal_mw_milp"]["exact_selection_frame_count"]
            == row["policies"]["exact_renewal_mw_milp"]["frame_count"]
            for row in rows
        ),
        "registered_run_count_is_exact": len(rows) * len(POLICIES) == protocol["expected_work"]["trajectory_runs"],
        "primary_family_is_complete": len(claims) == PRIMARY_FAMILY_SIZE and all(row["sample_size"] == len(ARRIVAL_SEEDS) for row in claims),
    }
    execution_passed = all(checks.values())
    decision = _decision(execution_passed, claims)
    report = {
        "schema_version": 1,
        "gate": GATE,
        "experiment": EXPERIMENT,
        "status": "PASS" if execution_passed else "FAIL",
        "protocol": protocol,
        "checks": checks,
        "dependency_evidence": {
            "capacity_scientific_decision": dependencies["capacity"].get("scientific_decision"),
            "joint95_holdout_covered": dependencies["capacity"]["joint95_audit"]["holdout_covered"],
            "score_only_static_scientific_decision": dependencies["static"]["scientific_decision"],
            "canonical_selector_scientific_decision": dependencies["canonical"]["scientific_decision"],
        },
        "physical_complete_frame_envelope": {"model": _json(model.snapshot())},
        "runs": rows,
        "primary_simultaneous_claims": claims,
        "scientific_decision": decision,
        "baseline_confirmation_passed": decision == "BASELINE_CONFIRMATION_PASS",
        "claim_boundary": CLAIM_BOUNDARY,
        "next_step": (
            "preserve the registered result for code-first synthesis"
            if decision == "BASELINE_CONFIRMATION_PASS"
            else "preserve the failed or inconclusive family without tuning on these tapes"
        ),
        "timing": {
            "workers": EXECUTION_WORKERS,
            "wall_s": time.perf_counter() - started,
            "timing_is_scientific_gate": False,
        },
    }
    _write_new(result_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--register", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--capacity-registration", type=Path, default=DEFAULT_CAPACITY_REGISTRATION)
    parser.add_argument("--capacity-result", type=Path, default=DEFAULT_CAPACITY_RESULT)
    parser.add_argument("--static-registration", type=Path, default=DEFAULT_STATIC_REGISTRATION)
    parser.add_argument("--static-result", type=Path, default=DEFAULT_STATIC_RESULT)
    parser.add_argument("--canonical-registration", type=Path, default=DEFAULT_CANONICAL_REGISTRATION)
    parser.add_argument("--canonical-result", type=Path, default=DEFAULT_CANONICAL_RESULT)
    args = parser.parse_args(argv)
    try:
        if args.register:
            payload = register(
                args.output,
                capacity_registration=args.capacity_registration,
                capacity_result=args.capacity_result,
                static_registration=args.static_registration,
                static_result=args.static_result,
                canonical_registration=args.canonical_registration,
                canonical_result=args.canonical_result,
            )
            destination = _paths(args.output)[1]
        else:
            payload = run_registered(args.output)
            destination = _paths(args.output)[0]
    except BaselineConfirmationRefusal as exc:
        raise SystemExit(f"REFUSED: {exc}") from exc
    print(json.dumps({
        "destination": str(destination),
        "status": payload.get("status"),
        "scientific_decision": payload.get("scientific_decision"),
        "unresolved_registration_dependency": payload.get("unresolved_registration_dependency"),
    }, indent=2))


if __name__ == "__main__":
    main()
