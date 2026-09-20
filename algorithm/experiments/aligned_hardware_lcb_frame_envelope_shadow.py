"""Local LCB duration envelope for the aligned gpu_b frame model.

The registered aligned model uses operational pre-holdout completion points.
This shadow replaces them with complete-frame upper durations inherited from
the existing empty and loaded split-conformal gates.  It does not run a
trajectory or collect new hardware data.

The primary capacity comparison keeps the arrival vectors of the point model.
Loads renormalized to the shadow capacity boundary are reported separately as
a diagnostic, because changing the denominator would change the question.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linprog

from algorithm.experiments.aligned_hardware_frame_model import (
    FrameModel,
    capacity_certificate,
    load_model,
)
from algorithm.experiments.critical_gpu_loaded_stochastic_lcb_gate import (
    build_critical_gpu_loaded_stochastic_lcb_gate,
)
from algorithm.experiments.critical_gpu_stochastic_lcb_gate import (
    build_critical_gpu_stochastic_lcb_gate,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
EMPTY_GATE = ARTIFACTS / (
    "critical_gpu_stochastic_lcb_gate_v8_gpu_b_20260803.json"
)
LOADED_GATE = ARTIFACTS / (
    "critical_gpu_loaded_stochastic_lcb_gate_gpu_b_20260809.json"
)
DEFAULT_OUTPUT = ARTIFACTS / (
    "aligned_hardware_lcb_frame_envelope_shadow_development_20260912.json"
)

EXPERIMENT = "aligned_hardware_lcb_frame_envelope_shadow_v1"
NODE = "gpu_b"
LOADS = (0.50, 0.80, 0.95)
SOURCE_ALPHA = 0.10
SOURCE_COVERAGE = 0.90
COMBINED_UNION_COVERAGE = 0.80
ASPIRATIONAL_COVERAGE = 0.95
SCORE_TOLERANCE = 1e-10
THIS_SOURCE = (
    "algorithm/experiments/aligned_hardware_lcb_frame_envelope_shadow.py"
)
FIXED_CODE_SOURCES = (
    THIS_SOURCE,
    "algorithm/experiments/aligned_hardware_frame_model.py",
)

DEDICATED_ACTIONS = {
    (
        "profile:gpu_cnn_torch_resnet50:gpu_3080ti_12gb_dual_gpu_b:"
        "empty:none:w1:p1"
    ): ("gpu_cnn_torch_resnet50", 1),
    (
        "profile:gpu_cnn_torch_resnet50:gpu_3080ti_12gb_dual_gpu_b:"
        "empty:none:w1:p3"
    ): ("gpu_cnn_torch_resnet50", 3),
    (
        "profile:gpu_heavy_jax_matmul:gpu_3080ti_12gb_dual_gpu_b:"
        "empty:none:w1:p1"
    ): ("gpu_heavy_jax_matmul", 1),
    (
        "profile:gpu_heavy_jax_matmul:gpu_3080ti_12gb_dual_gpu_b:"
        "empty:none:w1:p3"
    ): ("gpu_heavy_jax_matmul", 3),
    (
        "profile:gpu_llm_distilgpt2:gpu_3080ti_12gb_dual_gpu_b:"
        "empty:none:w1:p1"
    ): ("gpu_llm_distilgpt2", 1),
    (
        "profile:gpu_llm_distilgpt2:gpu_3080ti_12gb_dual_gpu_b:"
        "empty:none:w1:p3"
    ): ("gpu_llm_distilgpt2", 3),
    (
        "profile:gpu_llm_distilgpt2:gpu_3080ti_12gb_dual_gpu_b:"
        "empty:none:w1:p8"
    ): ("gpu_llm_distilgpt2", 8),
}

LOADED_ACTIONS = {
    "gpu_loaded:gpu_b:cnn_after_llm": (
        "cnn_after_llm",
        "gpu_llm_distilgpt2",
    ),
    "gpu_loaded:gpu_b:llm_after_cnn": (
        "llm_after_cnn",
        "gpu_cnn_torch_resnet50",
    ),
}

SOURCE_CONTRACT_FIELDS = (
    "gate",
    "schema_version",
    "status",
    "pass",
    "certificate_ready",
    "node",
    "node_bucket",
    "hardware_class",
    "resource_state",
    "measurement_protocol",
    "expected_wave_count",
    "expected_observation_count",
    "ready_observation_count",
    "training_waves",
    "calibration_waves",
    "holdout_wave",
    "miscoverage_alpha",
    "nominal_simultaneous_coverage",
    "measurement_code_sha256",
    "all_measurements_ready",
    "source_artifacts",
    "certificate",
)

ENVELOPE_RULE = {
    "dedicated": (
        "duration_upper = empty certificate simultaneous_upper_drain_completion_s"
    ),
    "loaded": (
        "duration_upper = loaded simultaneous_upper_target_completion_s + "
        "resident empty-p1 simultaneous_upper_drain_completion_s"
    ),
    "loaded_overlap_credit": (
        "zero; the serial envelope never needs a lower bound on overlap duration"
    ),
    "frame_completion": (
        "every admitted job completion offset is set to the action duration upper"
    ),
}

DECISION_BRANCHES = {
    "all_fixed_loads_positive": "ALL_FIXED_LOADS_LCB_CAPACITY_PASS_ADVANCE",
    "only_high_load_nonpositive": "HIGH_LOAD_LCB_CAPACITY_FAIL_STOP",
    "lower_load_nonpositive": "LCB_CAPACITY_FAIL_STOP",
    "invalid_execution": "INVALID_EXECUTION",
}

CLAIM_BOUNDARY = (
    "A local, hardware-specific complete-frame duration shadow for the frozen "
    "nine-action gpu_b catalogue. Its empty and loaded source events each "
    "have nominal simultaneous coverage 0.9, so their intersection has only a "
    "0.8 union-bound lower guarantee. It is not a 0.95 guarantee, a repeated-"
    "frame time-uniform guarantee, or evidence for another node or catalogue."
)


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _paths(output: Path = DEFAULT_OUTPUT) -> tuple[Path, Path, Path]:
    output = Path(output)
    return (
        output,
        output.with_name(output.stem + "_protocol.json"),
        output.with_name(output.stem + "_fixed_sources.json"),
    )


def _write_new(path: Path, payload: Any, *, compact: bool = False) -> None:
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(
            payload,
            stream,
            indent=None if compact else 2,
            separators=(",", ":") if compact else None,
        )
        stream.write("\n")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return payload


def _current_fixed_sources() -> dict[str, str]:
    return {name: (ROOT / name).read_text(encoding="utf-8") for name in FIXED_CODE_SOURCES}


def _campaign_paths(payload: Mapping[str, Any]) -> dict[int, Path]:
    rows = payload.get("source_artifacts")
    if not isinstance(rows, list) or len(rows) != 13:
        raise ValueError("LCB source gate must name exactly 13 wave artifacts")
    paths = {int(row["wave"]): Path(row["path"]) for row in rows}
    if tuple(sorted(paths)) != tuple(range(1, 14)):
        raise ValueError("LCB source gate wave roster changed")
    return paths


def _gate_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {field: _json(payload.get(field)) for field in SOURCE_CONTRACT_FIELDS}


def _validate_gate(
    saved: Mapping[str, Any],
    rebuilt: Mapping[str, Any],
    *,
    resource_state: str,
    row_count: int,
) -> None:
    required = {
        "schema_version": 1,
        "status": "PASS",
        "pass": True,
        "certificate_ready": True,
        "node": NODE,
        "resource_state": resource_state,
        "expected_wave_count": 13,
        "training_waves": [1, 2, 3],
        "calibration_waves": list(range(4, 13)),
        "holdout_wave": 13,
        "miscoverage_alpha": SOURCE_ALPHA,
        "nominal_simultaneous_coverage": SOURCE_COVERAGE,
        "all_measurements_ready": True,
    }
    if any(saved.get(key) != value for key, value in required.items()):
        raise ValueError(f"{resource_state} LCB source gate contract changed")
    if _gate_contract(saved) != _gate_contract(rebuilt):
        raise ValueError(f"{resource_state} LCB gate does not rebuild from its waves")
    certificate = rebuilt.get("certificate") or {}
    if (
        not certificate.get("constructed")
        or not certificate.get("finite_sample_rank_ready")
        or certificate.get("conformal_rank") != 9
        or certificate.get("calibration_wave_count") != 9
        or not certificate.get("all_holdout_bounds_valid")
        or len(certificate.get("rows") or ()) != row_count
    ):
        raise ValueError(f"{resource_state} LCB certificate is not ready")


def _load_sources() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Rebuild both saved gates from their declared local wave JSON files."""
    empty = _read_json(EMPTY_GATE)
    loaded = _read_json(LOADED_GATE)
    rebuilt_empty = build_critical_gpu_stochastic_lcb_gate(
        node=NODE,
        campaign_paths=_campaign_paths(empty),
        miscoverage_alpha=SOURCE_ALPHA,
    )
    rebuilt_loaded = build_critical_gpu_loaded_stochastic_lcb_gate(
        node=NODE,
        campaign_paths=_campaign_paths(loaded),
        miscoverage_alpha=SOURCE_ALPHA,
    )
    _validate_gate(empty, rebuilt_empty, resource_state="empty", row_count=9)
    _validate_gate(
        loaded,
        rebuilt_loaded,
        resource_state="mixed_colocation",
        row_count=4,
    )
    contract = {
        "empty_gate": EMPTY_GATE.name,
        "loaded_gate": LOADED_GATE.name,
        "empty": _gate_contract(rebuilt_empty),
        "loaded": _gate_contract(rebuilt_loaded),
    }
    return rebuilt_empty, rebuilt_loaded, contract


def _positive(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be positive and finite")
    return result


def _completion_offsets(requirements: Sequence[int], duration: float):
    return tuple(tuple(duration for _ in range(count)) for count in requirements)


def _build_shadow_model(
    point_model: FrameModel,
    empty_gate: Mapping[str, Any],
    loaded_gate: Mapping[str, Any],
) -> tuple[FrameModel, list[dict[str, Any]]]:
    """Map all nine actions to frozen LCB rows and construct the shadow."""
    empty_rows = {
        (str(row["workload_key"]), int(row["profile"])): row
        for row in empty_gate["certificate"]["rows"]
    }
    loaded_rows = {
        str(row["scenario_id"]): row
        for row in loaded_gate["certificate"]["rows"]
    }
    expected_ids = set(DEDICATED_ACTIONS) | set(LOADED_ACTIONS)
    actual_ids = {option.action_id for option in point_model.options}
    if (
        point_model.node != NODE
        or point_model.lanes != 2
        or len(point_model.workloads) != 3
        or len(point_model.options) != 9
        or actual_ids != expected_ids
    ):
        raise ValueError("aligned point-model action catalogue changed")

    options = []
    envelopes = []
    for option in point_model.options:
        point_duration = _positive(option.duration, f"{option.action_id}.duration")
        if option.action_id in DEDICATED_ACTIONS:
            workload, profile = DEDICATED_ACTIONS[option.action_id]
            row = empty_rows.get((workload, profile))
            if row is None:
                raise ValueError(f"missing empty LCB row for {option.action_id}")
            upper = _positive(
                row.get("simultaneous_upper_drain_completion_s"),
                f"{option.action_id}.empty_drain_upper",
            )
            envelope = {
                "action_id": option.action_id,
                "action_kind": "dedicated",
                "source_gate": EMPTY_GATE.name,
                "source_row_key": row["service_cell_id"],
                "source_functional": "drain_completion",
                "point_duration_s": point_duration,
                "upper_duration_s": upper,
                "duration_inflation_fraction": upper / point_duration - 1.0,
                "composition": "empty_simultaneous_upper_drain_completion",
                "overlap_credit_units": 0.0,
            }
            phases = (("empty_complete_frame_upper_s", upper),)
        else:
            scenario_id, resident_workload = LOADED_ACTIONS[option.action_id]
            row = loaded_rows.get(scenario_id)
            resident = empty_rows.get((resident_workload, 1))
            if row is None or resident is None:
                raise ValueError(f"missing serial LCB source for {option.action_id}")
            if (
                row.get("resident_workload_key") != resident_workload
                or row.get("target_workload_key") not in point_model.workloads
            ):
                raise ValueError(f"loaded LCB action mapping changed for {option.action_id}")
            target_upper = _positive(
                row.get("simultaneous_upper_target_completion_s"),
                f"{option.action_id}.target_upper",
            )
            resident_upper = _positive(
                resident.get("simultaneous_upper_drain_completion_s"),
                f"{option.action_id}.resident_upper",
            )
            upper = target_upper + resident_upper
            envelope = {
                "action_id": option.action_id,
                "action_kind": "loaded",
                "source_gate": LOADED_GATE.name,
                "source_row_key": scenario_id,
                "resident_source_gate": EMPTY_GATE.name,
                "resident_source_row_key": resident["service_cell_id"],
                "source_functionals": ["target_completion_time", "drain_completion"],
                "point_duration_s": point_duration,
                "target_upper_duration_s": target_upper,
                "resident_fresh_p1_upper_duration_s": resident_upper,
                "target_upper_only_duration_s": target_upper,
                "upper_duration_s": upper,
                "duration_inflation_fraction": upper / point_duration - 1.0,
                "composition": "serial_target_upper_plus_fresh_resident_p1_drain_upper",
                "overlap_credit_units": 0.0,
            }
            phases = (
                ("loaded_target_upper_s", target_upper),
                ("resident_fresh_p1_drain_upper_s", resident_upper),
            )
        if upper + SCORE_TOLERANCE < point_duration:
            raise ValueError(f"LCB envelope is shorter than point duration for {option.action_id}")
        options.append(
            replace(
                option,
                duration=upper,
                completion_offsets=_completion_offsets(option.requirements, upper),
                phases=phases,
            )
        )
        envelopes.append(envelope)

    shadow = replace(
        point_model,
        options=tuple(options),
        model_scope="hardware_local_union_0p8_lcb_serial_complete_frame_shadow",
    )
    return shadow, envelopes


def _coverage_contract() -> dict[str, Any]:
    return {
        "empty_source_nominal_simultaneous_coverage": SOURCE_COVERAGE,
        "loaded_source_nominal_simultaneous_coverage": SOURCE_COVERAGE,
        "combined_event_lower_bound": COMBINED_UNION_COVERAGE,
        "combination_rule": "P(empty and loaded) >= 1-alpha_empty-alpha_loaded",
        "independence_assumed": False,
        "product_0p81_claimed": False,
        "aspirational_combined_coverage": ASPIRATIONAL_COVERAGE,
        "aspirational_0p95_ready": False,
        "current_calibration_wave_count_per_source": 9,
        "minimum_joint_paired_calibration_waves_for_finite_0p95_rank": 19,
        "minimum_per_source_calibration_waves_for_symmetric_bonferroni_0p95": 39,
        "coverage_scope": "one future exchangeable source-wave event, not all future frames",
    }


def _rates(model: FrameModel) -> np.ndarray:
    requirements = np.asarray(
        [option.requirements for option in model.options], dtype=float
    )
    durations = np.maximum(
        1.0, np.asarray([option.duration for option in model.options], dtype=float)
    )
    return requirements / durations[:, None]


def _uniform_capacity_slack(
    model: FrameModel,
    arrivals: Sequence[float],
    structural: Mapping[str, Any],
) -> dict[str, Any]:
    rates = _rates(model)
    arrivals_array = np.asarray(arrivals, dtype=float)
    count = len(model.options)
    fit = linprog(
        np.concatenate((np.zeros(count), [-1.0])),
        A_ub=np.vstack((
            np.column_stack((-model.lanes * rates.T, np.ones(len(arrivals_array)))),
            np.concatenate((np.ones(count), [0.0])),
        )),
        b_ub=np.concatenate((-arrivals_array, [1.0])),
        bounds=[(0.0, None)] * count + [(None, None)],
        method="highs",
    )
    if not fit.success:
        raise ValueError(f"fixed-arrival capacity LP failed: {fit.message}")
    weights = np.maximum(fit.x[:-1], 0.0)
    total = float(weights.sum())
    if total > 1.0:
        weights /= total
    delta = float(np.min(model.lanes * (weights @ rates) - arrivals_array))
    projection_loss = float(structural["projection_support_loss_rate"])
    eta = delta - projection_loss
    tau_max = float(structural["tau_max"])
    max_departures = np.asarray(structural["max_departures"], dtype=float)
    poisson_means = arrivals_array * tau_max
    second_moment = float(0.5 * np.sum(
        poisson_means ** 2 + poisson_means + max_departures ** 2
    ))
    drift_constant = (
        second_moment
        + tau_max * float(structural["availability_loss_C"])
    )
    return {
        "arrival_rates": [float(value) for value in arrivals_array],
        "support_slack_delta": delta,
        "projection_support_loss_rate": projection_loss,
        "effective_slack_eta": eta,
        "capacity_time_weights": [float(value) for value in weights],
        "loaded_time_weight": float(sum(
            weight for weight, option in zip(weights, model.options, strict=True)
            if option.loaded
        )),
        "second_moment_B": second_moment,
        "availability_loss_C": float(structural["availability_loss_C"]),
        "tau_max": tau_max,
        "drift_constant": drift_constant,
        "conditional_negative_drift_certified": eta > SCORE_TOLERANCE,
        "negative_drift_backlog_threshold": (
            drift_constant / eta if eta > SCORE_TOLERANCE else None
        ),
    }


def _maximum_fixed_arrival_load(
    model: FrameModel, unit_arrivals: Sequence[float]
) -> dict[str, Any]:
    rates = _rates(model)
    arrivals = np.asarray(unit_arrivals, dtype=float)
    count = len(model.options)
    fit = linprog(
        np.concatenate((np.zeros(count), [-1.0])),
        A_ub=np.vstack((
            np.column_stack((-model.lanes * rates.T, arrivals)),
            np.concatenate((np.ones(count), [0.0])),
        )),
        b_ub=np.concatenate((np.zeros(len(arrivals)), [1.0])),
        bounds=[(0.0, None)] * count + [(0.0, None)],
        method="highs",
    )
    if not fit.success:
        raise ValueError(f"fixed-direction capacity LP failed: {fit.message}")
    weights = np.maximum(fit.x[:-1], 0.0)
    total = float(weights.sum())
    if total > 1.0:
        weights /= total
    return {
        "maximum_point_model_load_factor": float(fit.x[-1]),
        "capacity_time_weights": [float(value) for value in weights],
        "positive_weight_action_ids": [
            option.action_id
            for option, weight in zip(model.options, weights, strict=True)
            if weight > SCORE_TOLERANCE
        ],
        "loaded_time_weight": float(sum(
            weight for weight, option in zip(weights, model.options, strict=True)
            if option.loaded
        )),
    }


def _loaded_sensitivity(
    point_model: FrameModel,
    shadow_model: FrameModel,
    envelopes: Sequence[Mapping[str, Any]],
    unit_arrivals: Sequence[float],
) -> list[dict[str, Any]]:
    envelope_by_id = {row["action_id"]: row for row in envelopes}
    point_by_id = {option.action_id: option for option in point_model.options}
    rows = []
    for name in ("point_loaded_duration", "target_upper_only", "serial_complete_frame"):
        options = []
        for option in shadow_model.options:
            if not option.loaded or name == "serial_complete_frame":
                duration = option.duration
            elif name == "point_loaded_duration":
                duration = point_by_id[option.action_id].duration
            else:
                duration = envelope_by_id[option.action_id][
                    "target_upper_only_duration_s"
                ]
            options.append(replace(option, duration=float(duration)))
        variant = replace(shadow_model, options=tuple(options))
        rows.append({
            "loaded_duration_variant": name,
            **_maximum_fixed_arrival_load(variant, unit_arrivals),
        })
    return rows


def _capacity_analysis(
    point_model: FrameModel,
    shadow_model: FrameModel,
    envelopes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Run only small LPs; no selector, arrivals, trajectories, or RNG."""
    point_certificates = [capacity_certificate(point_model, load) for load in LOADS]
    structural = capacity_certificate(shadow_model, LOADS[0])
    fixed = []
    relative = []
    for load, point_certificate in zip(LOADS, point_certificates, strict=True):
        row = _uniform_capacity_slack(
            shadow_model,
            point_certificate["arrival_rates"],
            structural,
        )
        fixed.append({
            "load": load,
            "arrival_basis": "point_model_frozen_arrival_vector",
            "point_model_effective_slack_eta": point_certificate[
                "effective_slack_eta"
            ],
            **row,
        })
        shadow_certificate = capacity_certificate(shadow_model, load)
        relative.append({
            "load": load,
            "arrival_basis": "shadow_capacity_renormalized_diagnostic_only",
            "arrival_rates": list(shadow_certificate["arrival_rates"]),
            "support_slack_delta": shadow_certificate["support_slack_delta"],
            "effective_slack_eta": shadow_certificate["effective_slack_eta"],
            "conditional_negative_drift_certified": shadow_certificate[
                "conditional_model_drift_certified"
            ],
            "negative_drift_backlog_threshold": shadow_certificate[
                "negative_drift_backlog_threshold"
            ],
        })
    unit_arrivals = capacity_certificate(point_model, 1.0)["arrival_rates"]
    return {
        "fixed_arrival_capacity": fixed,
        "shadow_relative_capacity_diagnostic": relative,
        "maximum_fixed_arrival_load": _maximum_fixed_arrival_load(
            shadow_model, unit_arrivals
        ),
        "loaded_duration_sensitivity": _loaded_sensitivity(
            point_model, shadow_model, envelopes, unit_arrivals
        ),
        "shadow_structural_certificate": {
            key: structural[key]
            for key in (
                "capacity_projection_passed",
                "projection_check_count",
                "projection_failures",
                "projection_support_loss_rate",
                "availability_loss_C",
                "tau_min",
                "tau_max",
                "max_departures",
            )
        },
    }


def _decision(fixed_arrival_capacity: Sequence[Mapping[str, Any]]) -> str:
    if [float(row["load"]) for row in fixed_arrival_capacity] != list(LOADS):
        raise ValueError("fixed-arrival capacity rows do not match registered loads")
    positive = [
        float(row["effective_slack_eta"]) > SCORE_TOLERANCE
        for row in fixed_arrival_capacity
    ]
    if all(positive):
        return DECISION_BRANCHES["all_fixed_loads_positive"]
    if all(positive[:-1]) and not positive[-1]:
        return DECISION_BRANCHES["only_high_load_nonpositive"]
    return DECISION_BRANCHES["lower_load_nonpositive"]


def _protocol_payload(
    point_model: FrameModel,
    shadow_model: FrameModel,
    envelopes: Sequence[Mapping[str, Any]],
    source_contract: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT,
        "role": "local LCB complete-frame duration envelope shadow",
        "source_gates": [EMPTY_GATE.name, LOADED_GATE.name],
        "loads": list(LOADS),
        "primary_arrival_basis": "point_model_frozen_arrival_vectors",
        "shadow_relative_loads_are_diagnostic_only": True,
        "expected_dedicated_action_count": 7,
        "expected_loaded_action_count": 2,
        "envelope_rule": ENVELOPE_RULE,
        "coverage": _coverage_contract(),
        "decision_branches": DECISION_BRANCHES,
        "advance_gate": "effective_slack_eta > 0 at every fixed point-model arrival vector",
        "point_model": _json(point_model.snapshot()),
        "shadow_model": _json(shadow_model.snapshot()),
        "action_envelopes": _json(envelopes),
        "source_contract": _json(source_contract),
        "trajectory_runs": 0,
        "server_pulls": 0,
        "rng_draws": 0,
        "timing_is_scientific_gate": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def register(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output, protocol_path, fixed_sources_path = _paths(output)
    if any(path.exists() for path in (output, protocol_path, fixed_sources_path)):
        raise ValueError("LCB frame-envelope artifacts already exist")
    empty, loaded, source_contract = _load_sources()
    point_model = load_model()
    shadow_model, envelopes = _build_shadow_model(point_model, empty, loaded)
    protocol = _protocol_payload(
        point_model, shadow_model, envelopes, source_contract
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_new(protocol_path, protocol, compact=True)
    _write_new(fixed_sources_path, _current_fixed_sources())
    return protocol


def run_registered(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output, protocol_path, fixed_sources_path = _paths(output)
    if output.exists():
        raise ValueError("LCB frame-envelope result already exists")
    protocol = _read_json(protocol_path)
    registered_sources = _read_json(fixed_sources_path)
    if registered_sources != _current_fixed_sources():
        raise ValueError("registered LCB frame-envelope code sources changed")

    empty, loaded, source_contract = _load_sources()
    point_model = load_model()
    shadow_model, envelopes = _build_shadow_model(point_model, empty, loaded)
    current_protocol = _protocol_payload(
        point_model, shadow_model, envelopes, source_contract
    )
    current_protocol["registered_at_utc"] = protocol.get("registered_at_utc")
    if protocol != current_protocol:
        raise ValueError("registered LCB frame-envelope protocol or sources changed")

    started = time.perf_counter()
    analysis = _capacity_analysis(point_model, shadow_model, envelopes)
    elapsed = time.perf_counter() - started
    fixed = analysis["fixed_arrival_capacity"]
    sensitivity = analysis["loaded_duration_sensitivity"]
    decision = _decision(fixed)
    sensitivity_factors = [
        row["maximum_point_model_load_factor"] for row in sensitivity
    ]
    checks = {
        "all_nine_actions_have_envelopes": (
            len(envelopes) == 9
            and len({row["action_id"] for row in envelopes}) == 9
        ),
        "exactly_seven_dedicated_and_two_loaded": (
            sum(row["action_kind"] == "dedicated" for row in envelopes) == 7
            and sum(row["action_kind"] == "loaded" for row in envelopes) == 2
        ),
        "every_upper_duration_is_no_shorter_than_point": all(
            row["upper_duration_s"] + SCORE_TOLERANCE
            >= row["point_duration_s"]
            for row in envelopes
        ),
        "loaded_serial_envelopes_use_zero_overlap_credit": all(
            row["composition"]
            == "serial_target_upper_plus_fresh_resident_p1_drain_upper"
            and row["overlap_credit_units"] == 0.0
            for row in envelopes if row["action_kind"] == "loaded"
        ),
        "combined_coverage_is_only_union_bound_0p8": (
            protocol["coverage"]["combined_event_lower_bound"]
            == COMBINED_UNION_COVERAGE
            and not protocol["coverage"]["independence_assumed"]
            and not protocol["coverage"]["aspirational_0p95_ready"]
        ),
        "all_fixed_arrival_capacity_lps_completed": len(fixed) == len(LOADS),
        "shadow_projection_certificate_completed": analysis[
            "shadow_structural_certificate"
        ]["capacity_projection_passed"],
        "all_three_loaded_duration_sensitivities_completed": (
            [row["loaded_duration_variant"] for row in sensitivity]
            == [
                "point_loaded_duration",
                "target_upper_only",
                "serial_complete_frame",
            ]
        ),
        "loaded_sensitivity_boundary_is_invariant_and_has_zero_loaded_weight": (
            max(sensitivity_factors) - min(sensitivity_factors)
            <= SCORE_TOLERANCE
            and all(
                row["loaded_time_weight"] <= SCORE_TOLERANCE
                for row in sensitivity
            )
        ),
        "no_trajectory_server_or_rng_work": (
            protocol["trajectory_runs"]
            == protocol["server_pulls"]
            == protocol["rng_draws"]
            == 0
        ),
        "registered_sources_and_protocol_unchanged": True,
    }
    passed = all(checks.values())
    result = {
        "experiment": EXPERIMENT,
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "protocol": protocol,
        "coverage": protocol["coverage"],
        "action_envelopes": envelopes,
        **analysis,
        "scientific_decision": (
            decision if passed else DECISION_BRANCHES["invalid_execution"]
        ),
        "coverage_decision": "COMBINED_0P8_ONLY_0P95_NOT_CERTIFIED",
        "development_advance": (
            passed
            and decision == DECISION_BRANCHES["all_fixed_loads_positive"]
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "limitations": [
            "The 0.8 guarantee is the union-bound lower coverage of two 0.9 source events.",
            "The source certificates cover a future exchangeable wave, not an unbounded sequence of scheduler frames.",
            "The serial loaded envelope is deliberately conservative and credits no resident overlap service.",
            "A nonpositive fixed-arrival margin is failure to certify that load, not evidence of physical instability.",
            "Changed durations require a fresh budgeted-support proof before any selector claim is reused.",
        ],
        "next_step": (
            "recompute the budgeted-support certificate under the LCB shadow before any trajectory"
            if decision == DECISION_BRANCHES["all_fixed_loads_positive"]
            else "keep the fixed high-load failure; do not launch old-load trajectories or relabel shadow-relative loads as the same experiment"
        ),
        "work": {
            "local_source_wave_json_files": 26,
            "trajectory_runs": 0,
            "server_pulls": 0,
            "rng_draws": 0,
        },
        "timing": {
            "local_lp_wall_seconds": elapsed,
            "timing_is_scientific_gate": False,
        },
    }
    _write_new(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--run", "--run-registered", dest="run", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.register == args.run:
        parser.error("choose exactly one of --register or --run")
    payload = register(args.output) if args.register else run_registered(args.output)
    print(json.dumps({
        "experiment": payload["experiment"],
        "status": payload.get("status", "REGISTERED"),
        "output": str(args.output),
    }))


if __name__ == "__main__":
    main()
