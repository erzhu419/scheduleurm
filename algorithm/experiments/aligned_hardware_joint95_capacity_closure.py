"""Registered joint-95 physical envelope and aligned-capacity closure.

Registration freezes the already selected measurement split, predictor anchor,
complete-frame action mapping, arrival direction, load roster, and decision
branches.  It deliberately does not inspect the joint collection results
directory.  Execution is local and becomes available only after the nineteen
calibration visits, explicit rank-19 holdout release, and independent holdout
visit all pass the registered dispatcher audits.

This runner constructs no arrival tape and invokes no scheduling selector.  The
old count-space trajectory epsilon rows are retained as provenance, but a
numeric support comparison is allowed only when both the selector model and the
complete state population match the new physical-envelope contract.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from algorithm.experiments.aligned_hardware_frame_model import (
    FrameModel,
    FrameOption,
    capacity_certificate,
)
from algorithm.experiments.aligned_hardware_lcb_frame_envelope_shadow import (
    _maximum_fixed_arrival_load,
    _uniform_capacity_slack,
)
from algorithm.experiments.critical_gpu_joint95_collection_runner import (
    CALIBRATION_WAVES,
    HOLDOUT_WAVE,
    PROTOCOL as COLLECTION_PROTOCOL,
    SCORE_CONTRACT,
)
from algorithm.experiments.critical_gpu_joint95_manifest_dispatcher import (
    _read_object as _dispatcher_read_object,
    _result_predictor_vector,
    _scan_results,
    validate_holdout_release,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
DEFAULT_COLLECTION_MANIFEST = ARTIFACTS / (
    "critical_gpu_joint95_collection_manifest_20260913.json"
)
DEFAULT_RESULTS_DIR = ARTIFACTS / "critical_gpu_joint95_v1_results"
DEFAULT_LCB_PROTOCOL = ARTIFACTS / (
    "aligned_hardware_lcb_frame_envelope_shadow_development_20260912_protocol.json"
)
DEFAULT_COUNT_PROTOCOL = ARTIFACTS / (
    "aligned_hardware_count_space_bnb_trajectory_transition_20260913_protocol.json"
)
DEFAULT_COUNT_RESULT = ARTIFACTS / (
    "aligned_hardware_count_space_bnb_trajectory_transition_20260913.json"
)
DEFAULT_REGISTRATION = ARTIFACTS / (
    "aligned_hardware_joint95_capacity_closure_20260913_protocol.json"
)
DEFAULT_RESULT = ARTIFACTS / (
    "aligned_hardware_joint95_capacity_closure_20260913.json"
)

EXPERIMENT = "aligned_hardware_joint95_capacity_closure_v1"
GATE = "aligned_hardware_joint95_capacity_closure_runner"
NODE = "gpu_b"
BASE_LANES = 2
CAPACITY_LANES = (8, 16)
LOADS = (0.50, 0.80, 0.95)
CANDIDATE_LOADS = (0.80, 0.95)
TOLERANCE = 1e-12

DECISION_RULE = {
    "JOINT95_HOLDOUT_COVERAGE_FAIL_STOP": (
        "the independent wave-33 joint score exceeds the frozen rank-19 margin"
    ),
    "JOINT95_NO_POSITIVE_CAPACITY_MARGIN_STOP": (
        "coverage passes, but at least one frozen lane/load case has a nonpositive "
        "effective capacity slack or a failed structural projection"
    ),
    "CAPACITY_ESTABLISHED_CANDIDATE_PENDING": (
        "coverage and every frozen capacity case pass, but comparable candidate-"
        "support evidence is unavailable for at least one required case"
    ),
    "JOINT95_CANDIDATE_SUPPORT_FAIL_STOP": (
        "coverage and capacity pass, comparable same-model/same-state epsilon "
        "evidence exists, and at least one epsilon is not strictly below eta"
    ),
    "JOINT95_ALIGNED_CAPACITY_CLOSURE_PASS": (
        "coverage, every frozen capacity case, and every required same-model/same-"
        "state candidate-support comparison pass"
    ),
}


class ClosureRefusal(RuntimeError):
    """The frozen protocol or required terminal evidence is not ready."""


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ClosureRefusal(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ClosureRefusal(f"JSON root must be an object: {path}")
    return value


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(dict(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")


def _positive(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ClosureRefusal(f"{label} must be numeric") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ClosureRefusal(f"{label} must be positive and finite")
    return result


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
                phases=tuple(
                    (str(name), float(value))
                    for name, value in row.get("phases") or ()
                ),
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
        raise ClosureRefusal(f"invalid frame-model snapshot: {exc}") from exc


def _selector_model_contract(model: FrameModel) -> dict[str, Any]:
    """Fields that determine count-space objective values and feasible actions."""

    return {
        "node": model.node,
        "workloads": list(model.workloads),
        "options": [
            {
                "action_id": option.action_id,
                "requirements": list(option.requirements),
                "effective_duration_s": max(1.0, float(option.duration)),
            }
            for option in model.options
        ],
    }


def _collection_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    coverage = manifest.get("coverage_contract") or {}
    validation = manifest.get("validation") or {}
    required = {
        "schema_version": 1,
        "gate": "critical_gpu_joint95_collection_runner",
        "registered": True,
        "protocol": COLLECTION_PROTOCOL,
        "node": NODE,
        "measurements_executed": False,
        "completed_joint_units": 0,
    }
    for key, value in required.items():
        if manifest.get(key) != value:
            raise ClosureRefusal(f"collection manifest {key} differs")
    if validation.get("pass") is not True:
        raise ClosureRefusal("collection manifest does not carry passing validation")
    if (
        coverage.get("calibration_joint_waves") != list(CALIBRATION_WAVES)
        or coverage.get("holdout_joint_wave") != HOLDOUT_WAVE
        or coverage.get("calibration_unit_count") != 19
        or coverage.get("holdout_unit_count") != 1
        or coverage.get("finite_sample_rank") != 19
        or float(coverage.get("miscoverage_alpha") or 0.0) != 0.05
    ):
        raise ClosureRefusal("collection coverage contract differs")
    functionals = list(manifest.get("scored_functionals") or [])
    if len(functionals) != 9 or any(
        row.get("direction") != "upper_time" for row in functionals
    ):
        raise ClosureRefusal("collection must contain nine upper-time functionals")
    if manifest.get("score_contract") != SCORE_CONTRACT:
        raise ClosureRefusal("collection score contract differs")
    history = manifest.get("historical_training_inputs") or {}
    if (
        history.get("role") != "frozen_point_predictors_only"
        or history.get("eligible_joint_calibration_units") != 0
        or history.get("post_hoc_pairing_forbidden") is not True
    ):
        raise ClosureRefusal("historical predictor role differs")
    return {
        "schema_version": manifest["schema_version"],
        "gate": manifest["gate"],
        "status": manifest.get("status"),
        "registered": manifest["registered"],
        "protocol": manifest["protocol"],
        "collection_prefix": manifest.get("collection_prefix"),
        "node": manifest["node"],
        "node_bucket": manifest.get("node_bucket"),
        "hardware_class": manifest.get("hardware_class"),
        "coverage_contract": _json(coverage),
        "score_contract": _json(manifest["score_contract"]),
        "scored_functionals": _json(functionals),
        "joint_work_items": [
            {
                "joint_wave": row.get("joint_wave"),
                "joint_unit_id": row.get("joint_unit_id"),
                "split_role": row.get("split_role"),
                "scored_functional_ids": _json(row.get("scored_functional_ids") or []),
            }
            for row in manifest.get("joint_work_items") or []
        ],
        "historical_training_inputs": {
            "empty_gate": history.get("empty_gate"),
            "loaded_gate": history.get("loaded_gate"),
            "role": history.get("role"),
            "eligible_joint_calibration_units": history.get(
                "eligible_joint_calibration_units"
            ),
            "post_hoc_pairing_forbidden": history.get(
                "post_hoc_pairing_forbidden"
            ),
        },
    }


def _predictor_anchor(
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    history = manifest.get("historical_training_inputs") or {}
    empty_path = Path(str(history.get("empty_gate") or ""))
    loaded_path = Path(str(history.get("loaded_gate") or ""))
    empty = _read_object(empty_path)
    loaded = _read_object(loaded_path)
    empty_values = {
        str(row["service_cell_id"]): _positive(
            row.get("frozen_base_drain_completion_s"),
            f"empty predictor {row.get('service_cell_id')}",
        )
        for row in (empty.get("certificate") or {}).get("rows") or []
        if row.get("service_cell_id")
    }
    loaded_values = {
        str(row["scenario_id"]): _positive(
            row.get("frozen_target_completion_s"),
            f"loaded predictor {row.get('scenario_id')}",
        )
        for row in (loaded.get("certificate") or {}).get("rows") or []
        if row.get("scenario_id")
    }
    anchor = []
    for functional in manifest.get("scored_functionals") or []:
        source = str(functional.get("source") or "")
        identity = str(functional.get("identity") or "")
        if source == "empty" and identity in empty_values:
            predicted = empty_values[identity]
        elif source == "loaded" and identity in loaded_values:
            predicted = loaded_values[identity]
        else:
            raise ClosureRefusal(
                f"registered predictor unavailable for {source}:{identity}"
            )
        anchor.append(
            {
                "functional_id": str(functional["functional_id"]),
                "source": source,
                "identity": identity,
                "functional": str(functional["functional"]),
                "predicted_time_s": predicted,
            }
        )
    if len(anchor) != 9:
        raise ClosureRefusal("predictor anchor must contain exactly nine rows")
    return anchor


def _action_mapping(
    lcb_protocol: Mapping[str, Any],
    predictor_anchor: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    anchor_ids = {str(row["functional_id"]) for row in predictor_anchor}
    envelopes = list(lcb_protocol.get("action_envelopes") or [])
    point_model = _model_from_snapshot(lcb_protocol.get("point_model") or {})
    by_id = {option.action_id: option for option in point_model.options}
    if len(envelopes) != 9 or len(by_id) != 9:
        raise ClosureRefusal("LCB source must expose the frozen nine-action catalogue")
    mappings = []
    for row in envelopes:
        action_id = str(row.get("action_id") or "")
        if action_id not in by_id:
            raise ClosureRefusal(f"LCB mapping names unknown action {action_id!r}")
        if row.get("action_kind") == "dedicated":
            functional_ids = [
                f"empty:{row.get('source_row_key')}:drain_completion"
            ]
            composition = "dedicated_empty_drain_upper"
        elif row.get("action_kind") == "loaded":
            functional_ids = [
                f"loaded:{row.get('source_row_key')}:target_completion_time",
                f"empty:{row.get('resident_source_row_key')}:drain_completion",
            ]
            composition = "serial_loaded_target_plus_fresh_resident_p1_drain_upper"
        else:
            raise ClosureRefusal(f"unknown LCB action kind for {action_id}")
        if any(functional_id not in anchor_ids for functional_id in functional_ids):
            raise ClosureRefusal(f"action {action_id} has an unscored duration component")
        option = by_id[action_id]
        mappings.append(
            {
                "action_id": action_id,
                "action_kind": row["action_kind"],
                "requirements": list(option.requirements),
                "loaded": option.loaded,
                "functional_ids": functional_ids,
                "composition": composition,
                "completion_rule": "all admitted jobs complete at the serial upper duration",
            }
        )
    if {row["action_id"] for row in mappings} != set(by_id):
        raise ClosureRefusal("LCB action mapping is incomplete")
    return sorted(mappings, key=lambda row: row["action_id"])


def _candidate_source_snapshot(
    protocol: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, Any]:
    if result.get("status") != "PASS" or result.get("protocol") != protocol:
        raise ClosureRefusal("count-space result does not embed its registered protocol")
    if protocol.get("experiment") != "aligned_hardware_count_space_bnb_trajectory_transition_v1":
        raise ClosureRefusal("count-space protocol experiment differs")
    cases = []
    for row in result.get("case_summaries") or []:
        epsilon = float(row.get("maximum_fixed_certified_gap_per_job"))
        if not math.isfinite(epsilon) or epsilon < 0.0:
            raise ClosureRefusal("count-space epsilon is invalid")
        cases.append(
            {
                "lanes": int(row["lanes"]),
                "load": float(row["load"]),
                "fixed_lp_budget": int(row["fixed_lp_budget"]),
                "maximum_fixed_certified_gap_per_job": epsilon,
                "source_effective_slack_eta": float(row["effective_slack_eta"]),
                "source_trajectory_support_gate_passed": bool(
                    row["trajectory_support_gate_passed"]
                ),
            }
        )
    expected = sorted(
        (int(row["lanes"]), float(row["load"]), int(row["fixed_lp_budget"]))
        for row in protocol.get("primary_cases") or []
    )
    actual = sorted(
        (row["lanes"], row["load"], row["fixed_lp_budget"])
        for row in cases
    )
    if actual != expected or len(cases) != 4:
        raise ClosureRefusal("count-space epsilon case roster differs")
    return {
        "experiment": result.get("experiment"),
        "status": result.get("status"),
        "scientific_decision": result.get("scientific_decision"),
        "trajectory_transition_advance": result.get("trajectory_transition_advance"),
        "protocol_contract": {
            "experiment": protocol.get("experiment"),
            "model": _json(protocol.get("model")),
            "lane_budget_pairs": _json(protocol.get("lane_budget_pairs")),
            "loads": _json(protocol.get("loads")),
            "arrival_seeds": _json(protocol.get("arrival_seeds")),
            "arrival_horizon_s": protocol.get("arrival_horizon_s"),
            "arrival_model": protocol.get("arrival_model"),
            "dispatch": protocol.get("dispatch"),
            "stopping_rule": protocol.get("stopping_rule"),
            "primary_cases": _json(protocol.get("primary_cases")),
            "trajectory_support_gate": protocol.get("trajectory_support_gate"),
        },
        "case_evidence": sorted(cases, key=lambda row: (row["lanes"], row["load"])),
    }


def _planned_baseline_model(
    point_model: FrameModel,
    predictor_anchor: Sequence[Mapping[str, Any]],
    action_mapping: Sequence[Mapping[str, Any]],
    *,
    margin: float,
) -> tuple[FrameModel, list[dict[str, Any]]]:
    if not math.isfinite(margin) or margin < 0.0:
        raise ClosureRefusal("joint margin must be finite and nonnegative")
    predictors = {
        str(row["functional_id"]): _positive(
            row.get("predicted_time_s"), str(row.get("functional_id"))
        )
        for row in predictor_anchor
    }
    mapping_by_id = {str(row["action_id"]): row for row in action_mapping}
    if set(mapping_by_id) != {option.action_id for option in point_model.options}:
        raise ClosureRefusal("registered action mapping and point catalogue differ")
    options = []
    envelopes = []
    multiplier = 1.0 + margin
    for option in point_model.options:
        mapping = mapping_by_id[option.action_id]
        components = []
        for functional_id in mapping["functional_ids"]:
            predicted = predictors[str(functional_id)]
            upper = predicted * multiplier
            components.append(
                {
                    "functional_id": str(functional_id),
                    "predicted_time_s": predicted,
                    "upper_time_s": upper,
                }
            )
        duration = sum(row["upper_time_s"] for row in components)
        requirements = tuple(option.requirements)
        complete_job_lower_rates = [
            count / max(1.0, duration) for count in requirements
        ]
        native_lower_rates = [
            count * scale / max(1.0, duration)
            for count, scale in zip(requirements, point_model.scales, strict=True)
        ]
        options.append(
            replace(
                option,
                duration=duration,
                completion_offsets=tuple(
                    tuple(duration for _ in range(count)) for count in requirements
                ),
                phases=tuple(
                    (f"upper:{row['functional_id']}", row["upper_time_s"])
                    for row in components
                ),
            )
        )
        envelopes.append(
            {
                "action_id": option.action_id,
                "action_kind": mapping["action_kind"],
                "requirements": list(requirements),
                "components": components,
                "margin_multiplier": multiplier,
                "upper_complete_frame_duration_s": duration,
                "complete_job_lower_service_rates_per_s": complete_job_lower_rates,
                "native_unit_lower_service_rates_per_s": native_lower_rates,
                "completion_offsets_all_equal_upper_duration": True,
                "composition": mapping["composition"],
            }
        )
    return (
        replace(
            point_model,
            options=tuple(options),
            model_scope="joint95_hardware_local_serial_complete_frame_upper_envelope",
        ),
        envelopes,
    )


def _registration_candidate_audit(
    point_model: FrameModel,
    baseline_model: FrameModel,
) -> dict[str, Any]:
    source = _selector_model_contract(point_model)
    target = _selector_model_contract(baseline_model)
    differing = [
        target_row["action_id"]
        for source_row, target_row in zip(
            source["options"], target["options"], strict=True
        )
        if source_row != target_row
    ]
    reasons = []
    if source != target:
        reasons.append(
            {
                "code": "ACTION_DURATION_CATALOGUE_MISMATCH",
                "differing_action_ids_at_zero_margin": differing,
                "detail": (
                    "The joint envelope uses frozen training predictors and a serial "
                    "loaded complete-frame composition; the old trajectory epsilon "
                    "used the point-model durations."
                ),
            }
        )
    reasons.append(
        {
            "code": "VISITED_STATE_POPULATION_MISMATCH",
            "detail": (
                "The old aggregate epsilon covers queues visited by point-duration "
                "trajectories. Those complete states are not materialized as a fixed "
                "static roster, and changed frame durations change physical-time "
                "arrival boundaries."
            ),
        }
    )
    return {
        "status": "CANDIDATE_SUPPORT_EVIDENCE_UNAVAILABLE",
        "numeric_comparison_authorized": False,
        "numeric_comparison_count": 0,
        "reasons": reasons,
        "required_replacement": (
            "a separately registered same-envelope static selector screen over an "
            "explicit fixed queue roster before any new trajectory"
        ),
    }


def _reference_capacity_contract(point_model: FrameModel) -> dict[str, Any]:
    reference = capacity_certificate(point_model, 1.0)
    if not reference.get("capacity_projection_passed"):
        raise ClosureRefusal("point reference capacity projection failed")
    return {
        "base_lanes": BASE_LANES,
        "capacity_lanes": list(CAPACITY_LANES),
        "loads": list(LOADS),
        "candidate_support_loads": list(CANDIDATE_LOADS),
        "fixed_arrival_direction_base_R2": list(reference["arrival_direction"]),
        "fixed_point_boundary_arrivals_base_R2": list(reference["arrival_rates"]),
        "fixed_point_boundary_scale_base_R2": float(reference["full_boundary_scale"]),
        "lane_scaling_rule": (
            "arrival_R(load) = load * (R / 2) * fixed_point_boundary_arrivals_base_R2"
        ),
        "common_capacity_scale_formula": (
            "min over R in {8,16} of max s such that s*(R/2)*lambda_point_R2 "
            "is in the joint-envelope convex complete-frame service region"
        ),
        "eta_formula": (
            "eta_R(load) = max_{w>=0,sum(w)<=1} min_i "
            "[R*sum_a w_a*requirements[a,i]/max(1,tau_upper[a]) "
            "- arrival_R(load)[i]] - projection_support_loss_R"
        ),
        "positive_capacity_gate": (
            "every registered R/load structural projection passes and eta_R(load) > 0"
        ),
    }


def build_registration(
    *,
    collection_manifest_path: Path = DEFAULT_COLLECTION_MANIFEST,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    lcb_protocol_path: Path = DEFAULT_LCB_PROTOCOL,
    count_protocol_path: Path = DEFAULT_COUNT_PROTOCOL,
    count_result_path: Path = DEFAULT_COUNT_RESULT,
) -> dict[str, Any]:
    """Build registration without inspecting ``results_dir`` or any joint visit."""

    collection_manifest_path = Path(collection_manifest_path).resolve()
    lcb_protocol_path = Path(lcb_protocol_path).resolve()
    count_protocol_path = Path(count_protocol_path).resolve()
    count_result_path = Path(count_result_path).resolve()
    # Resolving this name is intentionally the only registration-time operation
    # involving the future results directory.
    frozen_results_dir = Path(results_dir).resolve()

    manifest = _read_object(collection_manifest_path)
    collection = _collection_contract(manifest)
    anchor = _predictor_anchor(manifest)
    lcb_protocol = _read_object(lcb_protocol_path)
    count_protocol = _read_object(count_protocol_path)
    count_result = _read_object(count_result_path)
    if lcb_protocol.get("experiment") != "aligned_hardware_lcb_frame_envelope_shadow_v1":
        raise ClosureRefusal("LCB protocol experiment differs")
    if tuple(float(value) for value in lcb_protocol.get("loads") or ()) != LOADS:
        raise ClosureRefusal("LCB load roster differs")
    point_model = _model_from_snapshot(lcb_protocol.get("point_model") or {})
    count_model = _model_from_snapshot(count_protocol.get("model") or {})
    if point_model.snapshot() != count_model.snapshot():
        raise ClosureRefusal("LCB and count-space point-model contracts differ")
    if point_model.node != NODE or point_model.lanes != BASE_LANES:
        raise ClosureRefusal("point reference must be the registered R2 gpu_b model")
    mappings = _action_mapping(lcb_protocol, anchor)
    baseline_model, baseline_envelopes = _planned_baseline_model(
        point_model, anchor, mappings, margin=0.0
    )
    candidate_source = _candidate_source_snapshot(count_protocol, count_result)
    expected_pairs = sorted(
        (int(row["lanes"]), float(row["load"]))
        for row in count_protocol.get("primary_cases") or []
    )
    if expected_pairs != [
        (lanes, load) for lanes in CAPACITY_LANES for load in CANDIDATE_LOADS
    ]:
        raise ClosureRefusal("count-space candidate lane/load roster differs")
    candidate_audit = _registration_candidate_audit(point_model, baseline_model)
    reference_capacity = _reference_capacity_contract(point_model)
    release_path = frozen_results_dir / "holdout_release.json"
    return {
        "schema_version": 1,
        "gate": GATE,
        "experiment": EXPERIMENT,
        "status": "REGISTERED_AWAITING_JOINT95_DATA",
        "registered": True,
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_paths": {
            "collection_manifest": str(collection_manifest_path),
            "joint_results_directory": str(frozen_results_dir),
            "holdout_release": str(release_path),
            "lcb_envelope_protocol": str(lcb_protocol_path),
            "count_space_trajectory_protocol": str(count_protocol_path),
            "count_space_trajectory_result": str(count_result_path),
        },
        "registration_scope": {
            "joint_results_directory_read": False,
            "joint_visit_values_read": False,
            "margin_computed": False,
            "trajectory_built": False,
            "selector_invoked": False,
            "server_pull_performed": False,
            "rng_draws": 0,
        },
        "collection_contract": collection,
        "predictor_anchor": anchor,
        "score_and_envelope_contract": {
            "cell_score": "actual_time / frozen_predicted_time - 1",
            "joint_wave_score": "max of the exactly nine registered cell scores",
            "finite_sample_rank": 19,
            "raw_rank_score": "sorted(calibration_joint_scores)[18]",
            "one_sided_margin": "max(0, raw_rank_score)",
            "faster_negative_scores": "do not enlarge the upper-time margin",
            "duration_component_upper": "predicted_time * (1 + margin)",
            "dedicated_action_upper": "its empty drain duration component upper",
            "loaded_action_upper": (
                "loaded target duration component upper + fresh resident empty-p1 "
                "drain duration component upper"
            ),
            "complete_frame_offsets": (
                "all admitted jobs complete at the action upper duration"
            ),
            "lower_complete_job_service": (
                "requirements / max(1, upper_complete_frame_duration)"
            ),
            "lower_native_service": (
                "requirements * canonical_native_scale / "
                "max(1, upper_complete_frame_duration)"
            ),
            "forbidden_shortcut": (
                "do not use predicted_service*(1-margin); reciprocal time scaling "
                "is required"
            ),
        },
        "action_mapping": mappings,
        "zero_margin_action_envelopes": baseline_envelopes,
        "point_reference_model": _json(point_model.snapshot()),
        "fixed_capacity_contract": reference_capacity,
        "candidate_evidence_source": candidate_source,
        "candidate_evidence_reuse_pre_audit": candidate_audit,
        "run_preconditions": [
            "waves 14..32 each have exactly one dispatcher-valid PASS terminal",
            "all nineteen calibration terminals equal the registered predictor anchor",
            "the explicit holdout_release validates and freezes rank 19 (the maximum)",
            "wave 33 has exactly one dispatcher-valid independent PASS terminal",
            "wave 33 started no earlier than the explicit release and keeps the predictor anchor",
            "all frozen source contracts still match this registration",
        ],
        "decision_rule": dict(DECISION_RULE),
        "prohibited": [
            "reading joint visit values or the joint results directory during registration",
            "post-result changes to functionals, score, rank, action mapping, lanes, loads, or arrival direction",
            "new trajectory construction or post-result load selection",
            "selector execution in this runner",
            "numeric reuse of epsilon when model or state-population contracts differ",
            "server pulls or paper edits",
        ],
        "claim_boundary": (
            "A one-future-visit 95% joint upper-time audit and conditional local "
            "complete-frame capacity calculation for the registered gpu_b "
            "nine-action catalogue. It is not time-uniform physical coverage, a "
            "trajectory result, or candidate-search closure without separate "
            "same-model static support evidence."
        ),
    }


def register(
    output: Path = DEFAULT_REGISTRATION,
    **kwargs: Any,
) -> dict[str, Any]:
    output = Path(output)
    if output.exists():
        raise ClosureRefusal(f"registration artifact already exists: {output}")
    protocol = build_registration(**kwargs)
    _write_new(output, protocol)
    return protocol


def _current_source_snapshot(protocol: Mapping[str, Any]) -> dict[str, Any]:
    paths = protocol.get("source_paths") or {}
    manifest = _read_object(Path(paths["collection_manifest"]))
    collection = _collection_contract(manifest)
    anchor = _predictor_anchor(manifest)
    lcb_protocol = _read_object(Path(paths["lcb_envelope_protocol"]))
    count_protocol = _read_object(Path(paths["count_space_trajectory_protocol"]))
    count_result = _read_object(Path(paths["count_space_trajectory_result"]))
    point_model = _model_from_snapshot(lcb_protocol.get("point_model") or {})
    mappings = _action_mapping(lcb_protocol, anchor)
    candidate_source = _candidate_source_snapshot(count_protocol, count_result)
    return {
        "manifest": manifest,
        "collection_contract": collection,
        "predictor_anchor": anchor,
        "point_model": point_model,
        "point_reference_model": _json(point_model.snapshot()),
        "action_mapping": mappings,
        "candidate_evidence_source": candidate_source,
    }


def _validate_registered_sources(
    protocol: Mapping[str, Any], current: Mapping[str, Any]
) -> None:
    checks = (
        ("collection contract", current["collection_contract"], protocol.get("collection_contract")),
        ("predictor anchor", current["predictor_anchor"], protocol.get("predictor_anchor")),
        ("point reference model", current["point_reference_model"], protocol.get("point_reference_model")),
        ("action mapping", current["action_mapping"], protocol.get("action_mapping")),
        ("candidate evidence source", current["candidate_evidence_source"], protocol.get("candidate_evidence_source")),
    )
    for label, actual, expected in checks:
        if actual != expected:
            raise ClosureRefusal(f"{label} changed after registration")
    point_model = current["point_model"]
    baseline_model, baseline_envelopes = _planned_baseline_model(
        point_model,
        current["predictor_anchor"],
        current["action_mapping"],
        margin=0.0,
    )
    if _json(baseline_envelopes) != protocol.get("zero_margin_action_envelopes"):
        raise ClosureRefusal("zero-margin action-envelope contract changed")
    if _reference_capacity_contract(point_model) != protocol.get(
        "fixed_capacity_contract"
    ):
        raise ClosureRefusal("fixed capacity direction or roster changed")
    if _registration_candidate_audit(
        point_model, baseline_model
    ) != protocol.get("candidate_evidence_reuse_pre_audit"):
        raise ClosureRefusal("candidate-evidence reuse pre-audit changed")
    paths = protocol.get("source_paths") or {}
    expected_release = Path(paths["joint_results_directory"]) / "holdout_release.json"
    if Path(paths["holdout_release"]) != expected_release:
        raise ClosureRefusal("holdout release path is outside the frozen results directory")


def _score_success(
    payload: Mapping[str, Any],
    predictor_anchor: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected = [
        (str(row["functional_id"]), float(row["predicted_time_s"]))
        for row in predictor_anchor
    ]
    observed = _result_predictor_vector(payload)
    if observed != expected:
        raise ClosureRefusal(
            f"wave {payload.get('joint_wave')} predictor vector differs from registration"
        )
    scores = []
    rows = []
    for observation, (functional_id, predicted) in zip(
        payload.get("observations") or [], expected, strict=True
    ):
        actual = _positive(observation.get("actual_time_s"), functional_id)
        score = actual / predicted - 1.0
        reported = float(observation.get("ratio_score"))
        if not math.isfinite(reported) or not math.isclose(
            reported, score, rel_tol=TOLERANCE, abs_tol=TOLERANCE
        ):
            raise ClosureRefusal(
                f"wave {payload.get('joint_wave')} score differs for {functional_id}"
            )
        scores.append(score)
        rows.append(
            {
                "functional_id": functional_id,
                "actual_time_s": actual,
                "predicted_time_s": predicted,
                "ratio_score": score,
            }
        )
    joint_score = max(scores)
    if not math.isclose(
        float(payload.get("joint_wave_score")),
        joint_score,
        rel_tol=TOLERANCE,
        abs_tol=TOLERANCE,
    ):
        raise ClosureRefusal(
            f"wave {payload.get('joint_wave')} joint score differs from its nine rows"
        )
    return {
        "joint_wave": int(payload["joint_wave"]),
        "joint_unit_id": str(payload["joint_unit_id"]),
        "joint_wave_score": joint_score,
        "observations": rows,
    }


def _rank19_margin(scores: Sequence[float]) -> tuple[float, float]:
    values = tuple(float(value) for value in scores)
    if len(values) != 19 or any(not math.isfinite(value) for value in values):
        raise ClosureRefusal("rank-19 margin requires exactly nineteen finite scores")
    raw_rank_score = sorted(values)[18]
    return raw_rank_score, max(0.0, raw_rank_score)


def _complete_joint_evidence(
    protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    results_dir = Path((protocol.get("source_paths") or {})["joint_results_directory"])
    scans = _scan_results(manifest, results_dir)
    for wave in (*CALIBRATION_WAVES, HOLDOUT_WAVE):
        scan = scans.get(wave) or {}
        if scan.get("invalid_passes"):
            raise ClosureRefusal(f"wave {wave} contains an invalid PASS artifact")
        successes = list(scan.get("successes") or [])
        if len(successes) != 1:
            raise ClosureRefusal(
                f"wave {wave} must have exactly one valid PASS; found {len(successes)}"
            )
    release_path = Path((protocol.get("source_paths") or {})["holdout_release"])
    try:
        release = _dispatcher_read_object(release_path)
    except Exception as exc:  # dispatcher exposes its own refusal type
        raise ClosureRefusal(f"explicit holdout release is unavailable: {exc}") from exc
    release_audit = validate_holdout_release(release, manifest, scans)
    if release_audit.get("pass") is not True:
        raise ClosureRefusal(
            "explicit holdout release is invalid: "
            + "; ".join(str(value) for value in release_audit.get("errors") or [])
        )
    anchor = protocol.get("predictor_anchor") or []
    calibration = []
    for wave in CALIBRATION_WAVES:
        row = scans[wave]["successes"][0]
        calibration.append(
            {
                "path": str(row["path"]),
                **_score_success(row["payload"], anchor),
            }
        )
    holdout_row = scans[HOLDOUT_WAVE]["successes"][0]
    holdout = {
        "path": str(holdout_row["path"]),
        **_score_success(holdout_row["payload"], anchor),
    }
    try:
        released_at = float(release["released_at"])
        holdout_started_at = float(holdout_row["payload"]["started_at"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ClosureRefusal("release/holdout sequencing times are unavailable") from exc
    if (
        not math.isfinite(released_at)
        or not math.isfinite(holdout_started_at)
        or holdout_started_at < released_at
    ):
        raise ClosureRefusal("wave 33 is not an independent post-release holdout")
    scores = [row["joint_wave_score"] for row in calibration]
    raw_rank_score, margin = _rank19_margin(scores)
    frozen_margin = float(release_audit["frozen_margin"])
    if not math.isclose(
        margin, frozen_margin, rel_tol=TOLERANCE, abs_tol=TOLERANCE
    ):
        raise ClosureRefusal("release did not freeze the recomputed rank-19 margin")
    return {
        "calibration": calibration,
        "calibration_joint_scores": scores,
        "raw_rank19_score": raw_rank_score,
        "one_sided_margin": margin,
        "release": {
            "path": str(release_path),
            "released_at": released_at,
            "finite_sample_rank": release.get("finite_sample_rank"),
            "calibration_margin_frozen": release.get("calibration_margin_frozen"),
            "frozen_margin": frozen_margin,
            "audit_passed": True,
        },
        "holdout": holdout,
        "holdout_started_at": holdout_started_at,
        "holdout_covered": holdout["joint_wave_score"] <= margin + TOLERANCE,
    }


def _capacity_analysis(
    protocol: Mapping[str, Any], model: FrameModel
) -> dict[str, Any]:
    contract = protocol.get("fixed_capacity_contract") or {}
    base_arrivals = tuple(
        float(value) for value in contract["fixed_point_boundary_arrivals_base_R2"]
    )
    lane_rows = []
    all_cases = []
    for lanes in contract["capacity_lanes"]:
        lanes = int(lanes)
        lane_model = replace(model, lanes=lanes)
        unit_arrivals = tuple(value * lanes / BASE_LANES for value in base_arrivals)
        structural = capacity_certificate(lane_model, LOADS[0])
        maximum = _maximum_fixed_arrival_load(lane_model, unit_arrivals)
        cases = []
        for load in contract["loads"]:
            load = float(load)
            row = _uniform_capacity_slack(
                lane_model,
                tuple(load * value for value in unit_arrivals),
                structural,
            )
            case = {
                "lanes": lanes,
                "load": load,
                "arrival_rates": row["arrival_rates"],
                "effective_slack_eta": row["effective_slack_eta"],
                "support_slack_delta": row["support_slack_delta"],
                "projection_support_loss_rate": row[
                    "projection_support_loss_rate"
                ],
                "capacity_projection_passed": bool(
                    structural["capacity_projection_passed"]
                ),
                "positive_capacity_margin": bool(
                    structural["capacity_projection_passed"]
                    and row["effective_slack_eta"] > TOLERANCE
                ),
            }
            cases.append(case)
            all_cases.append(case)
        lane_rows.append(
            {
                "lanes": lanes,
                "maximum_fixed_point_load_factor": maximum[
                    "maximum_point_model_load_factor"
                ],
                "cases": cases,
            }
        )
    scales = [row["maximum_fixed_point_load_factor"] for row in lane_rows]
    return {
        "fixed_arrival_basis": "point-model boundary direction frozen at registration",
        "per_lane": lane_rows,
        "common_capacity_scale": min(scales),
        "maximum_lane_scale_disagreement": max(scales) - min(scales),
        "all_registered_capacity_margins_positive": all(
            row["positive_capacity_margin"] for row in all_cases
        ),
        "case_count": len(all_cases),
    }


def _candidate_support_audit(
    protocol: Mapping[str, Any],
    envelope_model: FrameModel,
    capacity: Mapping[str, Any],
) -> dict[str, Any]:
    source = protocol.get("candidate_evidence_source") or {}
    source_model = _model_from_snapshot(
        (source.get("protocol_contract") or {}).get("model") or {}
    )
    model_match = (
        _selector_model_contract(source_model)
        == _selector_model_contract(envelope_model)
    )
    # The old result retains only aggregate maxima and short audit prefixes; it
    # does not materialize the complete visited queue population.  Because the
    # duration catalogue also changes, those physical-time states are not the
    # target population of this closure.
    complete_state_population_match = False
    reasons = []
    if not model_match:
        reasons.append("ACTION_DURATION_CATALOGUE_MISMATCH")
    if not complete_state_population_match:
        reasons.append("VISITED_STATE_POPULATION_MISMATCH")
    eta_by_case = {
        (int(row["lanes"]), float(case["load"])): float(
            case["effective_slack_eta"]
        )
        for row in capacity["per_lane"]
        for case in row["cases"]
    }
    rows = []
    comparable_count = 0
    numeric_failures = 0
    for evidence in source.get("case_evidence") or []:
        key = (int(evidence["lanes"]), float(evidence["load"]))
        if float(evidence["load"]) not in CANDIDATE_LOADS:
            continue
        comparable = model_match and complete_state_population_match and key in eta_by_case
        row = {
            "lanes": key[0],
            "load": key[1],
            "fixed_lp_budget": int(evidence["fixed_lp_budget"]),
            "source_epsilon_retained_for_provenance": float(
                evidence["maximum_fixed_certified_gap_per_job"]
            ),
            "model_contract_matches": model_match,
            "state_population_contract_matches": complete_state_population_match,
            "numeric_comparison_performed": comparable,
        }
        if comparable:
            comparable_count += 1
            eta = eta_by_case[key]
            epsilon = float(evidence["maximum_fixed_certified_gap_per_job"])
            passed = epsilon < eta
            numeric_failures += int(not passed)
            row.update(
                {
                    "joint_envelope_eta": eta,
                    "epsilon_strictly_below_eta": passed,
                    "eta_minus_epsilon": eta - epsilon,
                }
            )
        else:
            row["comparison_unavailable_reasons"] = list(reasons)
        rows.append(row)
    required_count = len(CAPACITY_LANES) * len(CANDIDATE_LOADS)
    if comparable_count < required_count:
        status = "CANDIDATE_SUPPORT_EVIDENCE_UNAVAILABLE"
    elif numeric_failures:
        status = "CANDIDATE_SUPPORT_FAIL"
    else:
        status = "CANDIDATE_SUPPORT_PASS"
    return {
        "status": status,
        "source_evidence_kind": "aggregate maximum epsilon over old point-model trajectory states",
        "numeric_comparison_count": comparable_count,
        "required_comparison_count": required_count,
        "numeric_failure_count": numeric_failures,
        "unavailable_reasons": reasons if comparable_count < required_count else [],
        "cases": rows,
        "next_required_evidence": (
            "register and run a same-envelope static selector screen over a fully "
            "materialized fixed queue roster before any new trajectory"
            if comparable_count < required_count
            else None
        ),
    }


def _decision(
    coverage_passed: bool,
    capacity_passed: bool,
    candidate_status: str,
) -> str:
    if not coverage_passed:
        return "JOINT95_HOLDOUT_COVERAGE_FAIL_STOP"
    if not capacity_passed:
        return "JOINT95_NO_POSITIVE_CAPACITY_MARGIN_STOP"
    if candidate_status == "CANDIDATE_SUPPORT_EVIDENCE_UNAVAILABLE":
        return "CAPACITY_ESTABLISHED_CANDIDATE_PENDING"
    if candidate_status == "CANDIDATE_SUPPORT_FAIL":
        return "JOINT95_CANDIDATE_SUPPORT_FAIL_STOP"
    if candidate_status == "CANDIDATE_SUPPORT_PASS":
        return "JOINT95_ALIGNED_CAPACITY_CLOSURE_PASS"
    raise ClosureRefusal(f"unknown candidate-support status {candidate_status!r}")


def run_registered(
    registration: Path = DEFAULT_REGISTRATION,
    output: Path = DEFAULT_RESULT,
) -> dict[str, Any]:
    registration = Path(registration)
    output = Path(output)
    if output.exists():
        raise ClosureRefusal(f"result artifact already exists: {output}")
    protocol = _read_object(registration)
    if (
        protocol.get("schema_version") != 1
        or protocol.get("gate") != GATE
        or protocol.get("experiment") != EXPERIMENT
        or protocol.get("registered") is not True
        or protocol.get("status") != "REGISTERED_AWAITING_JOINT95_DATA"
        or protocol.get("decision_rule") != DECISION_RULE
    ):
        raise ClosureRefusal("closure registration is invalid")
    current = _current_source_snapshot(protocol)
    _validate_registered_sources(protocol, current)
    joint = _complete_joint_evidence(protocol, current["manifest"])
    envelope_model, envelopes = _planned_baseline_model(
        current["point_model"],
        protocol["predictor_anchor"],
        protocol["action_mapping"],
        margin=float(joint["one_sided_margin"]),
    )
    capacity = _capacity_analysis(protocol, envelope_model)
    candidate = _candidate_support_audit(protocol, envelope_model, capacity)
    decision = _decision(
        bool(joint["holdout_covered"]),
        bool(capacity["all_registered_capacity_margins_positive"]),
        str(candidate["status"]),
    )
    report = {
        "schema_version": 1,
        "gate": GATE,
        "experiment": EXPERIMENT,
        "status": "PASS",
        "registration_path": str(registration.resolve()),
        "protocol": protocol,
        "joint95_audit": joint,
        "physical_complete_frame_envelope": {
            "model": envelope_model.snapshot(),
            "actions": envelopes,
        },
        "capacity": capacity,
        "candidate_support": candidate,
        "scientific_decision": decision,
        "closure_passed": decision == "JOINT95_ALIGNED_CAPACITY_CLOSURE_PASS",
        "capacity_established": bool(
            joint["holdout_covered"]
            and capacity["all_registered_capacity_margins_positive"]
        ),
        "trajectory_runs": 0,
        "selector_calls": 0,
        "server_pulls": 0,
        "rng_draws": 0,
        "claim_boundary": protocol["claim_boundary"],
    }
    _write_new(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    register_parser = sub.add_parser("register")
    register_parser.add_argument("--output", type=Path, default=DEFAULT_REGISTRATION)
    register_parser.add_argument(
        "--collection-manifest", type=Path, default=DEFAULT_COLLECTION_MANIFEST
    )
    register_parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    register_parser.add_argument("--lcb-protocol", type=Path, default=DEFAULT_LCB_PROTOCOL)
    register_parser.add_argument("--count-protocol", type=Path, default=DEFAULT_COUNT_PROTOCOL)
    register_parser.add_argument("--count-result", type=Path, default=DEFAULT_COUNT_RESULT)
    run_parser = sub.add_parser("run")
    run_parser.add_argument(
        "--registration", type=Path, default=DEFAULT_REGISTRATION
    )
    run_parser.add_argument("--output", type=Path, default=DEFAULT_RESULT)
    args = parser.parse_args()
    try:
        if args.command == "register":
            report = register(
                args.output,
                collection_manifest_path=args.collection_manifest,
                results_dir=args.results_dir,
                lcb_protocol_path=args.lcb_protocol,
                count_protocol_path=args.count_protocol,
                count_result_path=args.count_result,
            )
        else:
            report = run_registered(args.registration, args.output)
    except ClosureRefusal as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
