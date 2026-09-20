"""Register a fresh joint95 campaign after the v1, v2, and v3 fail-stops.

The v4 campaign preserves the frozen nine-functional statistical contract and
measurement bundles, but it starts a disjoint 19+1 split.  This module can only
build, validate, and write the registration.  It cannot launch measurements or
consume any v1, v2, or v3 result as calibration evidence.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from algorithm.experiments import critical_gpu_completion_campaign as empty_suite
from algorithm.experiments import critical_gpu_loaded_completion_campaign as loaded_suite


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_V1_MANIFEST = ARTIFACT_ROOT / "critical_gpu_joint95_collection_manifest_20260913.json"
DEFAULT_V1_FAIL_STOP = (
    ARTIFACT_ROOT / "critical_gpu_joint95_v1_results" / "campaign_fail_stop.json"
)
DEFAULT_V2_MANIFEST = ARTIFACT_ROOT / "critical_gpu_joint95_v2_collection_manifest_20260913.json"
DEFAULT_V2_FAIL_STOP = (
    ARTIFACT_ROOT / "critical_gpu_joint95_v2_results" / "campaign_fail_stop.json"
)
DEFAULT_V3_MANIFEST = ARTIFACT_ROOT / "critical_gpu_joint95_v3_collection_manifest_20260913.json"
DEFAULT_V3_FAIL_STOP = (
    ARTIFACT_ROOT / "critical_gpu_joint95_v3_results" / "campaign_fail_stop.json"
)
DEFAULT_OUTPUT = ARTIFACT_ROOT / "critical_gpu_joint95_v4_collection_manifest_20260913.json"
DEFAULT_RESULTS_DIR = ARTIFACT_ROOT / "critical_gpu_joint95_v4_results"
RESERVATION_RELEASE_FILENAME = "v4_reservation_release.ready"
SUPERVISOR_SCHEDULER_SIGNATURE = "ScheduleurmBench/joint95_v4_supervisor_v1"

SCHEMA_VERSION = 1
GATE = "critical_gpu_joint95_v4_registration"
PROTOCOL = "critical_gpu_joint95_natural_completion_v4"
COLLECTION_PREFIX = "critical_gpu_joint95_v4"
NODE = "jtl110gpu2"
MIS_COVERAGE_ALPHA = 0.05
CALIBRATION_WAVES = tuple(range(74, 93))
HOLDOUT_WAVE = 93
JOINT_WAVES = CALIBRATION_WAVES + (HOLDOUT_WAVE,)
V1_WAVES = tuple(range(14, 34))
V2_WAVES = tuple(range(34, 54))
V3_WAVES = tuple(range(54, 74))
V1_FAILURE_DECISION = "JOINT95_V1_EXTERNAL_OVERLAP_FAIL_STOP"
V2_FAILURE_DECISION = "JOINT95_V2_INTERVAL_OVERLAP_FAIL_STOP"
V3_FAILURE_DECISION = "JOINT95_V3_FIRST_ATTEMPT_FAIL_STOP"
REPLACEMENT_CLASSIFICATION = (
    "FULL_INDEPENDENT_REPLACEMENT_AFTER_THREE_FAILED_CAMPAIGNS"
)
V4_EXECUTOR_MODULE = (
    "algorithm.experiments.critical_gpu_joint95_v4_dispatcher"
)
V4_EXECUTOR_TEST_SCOPE = (
    "manifest-derived ordering",
    "v1, v2, and v3 result exclusion",
    "strict attempt01 result identity",
    "unknown JSON result rejection",
    "holdout lock and release",
    "one-step scheduler submission",
    "campaign fail-stop sentinel",
    "dry-run without scheduler mutation",
    "nine mandatory scheduler interval audits",
    "terminal refresh of the original nine scheduler intervals",
    "own-child provenance and process-exit boundary audit",
    "host CPU preflight before every measurement call",
)
HOST_CPU_PREFLIGHT_CONTRACT = {
    "required": True,
    "implementation_module": "algorithm.experiments.critical_gpu_joint95_manifest_dispatcher",
    "function": "_wait_for_clean_node",
    "before_each_measurement_call": True,
    "max_unapproved_user_cpu_fraction": 0.05,
}


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _seed_bases(waves: Sequence[int], *, suite: str) -> list[int]:
    if suite == "empty":
        return [830_000 + int(wave) * 10_000 for wave in waves]
    if suite == "loaded":
        return [920_000 + int(wave) * 100 for wave in waves]
    raise ValueError(f"unknown suite {suite!r}")


def _work_items(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    empty_ids = sorted(
        str(row["service_cell_id"]) for row in source.get("empty_contract") or []
    )
    loaded_ids = sorted(
        str(row["scenario_id"]) for row in source.get("loaded_contract") or []
    )
    scored_ids = [
        *(f"empty:{identity}:drain_completion" for identity in empty_ids),
        *(f"loaded:{identity}:target_completion_time" for identity in loaded_ids),
    ]
    bundles = source.get("measurement_bundles") or {}
    empty_bundle = (bundles.get("empty") or {}).get("manifest") or {}
    loaded_bundle = (bundles.get("loaded") or {}).get("manifest") or {}
    items: list[dict[str, Any]] = []
    for visit_order, wave in enumerate(JOINT_WAVES, start=1):
        role = "calibration" if wave in CALIBRATION_WAVES else "holdout"
        items.append(
            {
                "visit_order": visit_order,
                "joint_wave": wave,
                "joint_unit_id": f"{COLLECTION_PREFIX}_{NODE}_r{wave:02d}",
                "split_role": role,
                "node": NODE,
                "suite_order": ["empty", "loaded"],
                "observation_count": len(scored_ids),
                "scored_functional_ids": scored_ids,
                "empty_suite": {
                    "measurement_protocol": (bundles.get("empty") or {}).get(
                        "protocol"
                    ),
                    "measurement_code_sha256": empty_bundle.get("sha256"),
                    "service_cell_ids": empty_ids,
                    "seed_base": _seed_bases((wave,), suite="empty")[0],
                    "run_sequentially": True,
                    "empty_preflight_required_before_each_cell": True,
                    "scheduler_interval_audit_required_after_each_cell": True,
                },
                "loaded_suite": {
                    "measurement_protocol": (bundles.get("loaded") or {}).get(
                        "protocol"
                    ),
                    "measurement_code_sha256": loaded_bundle.get("sha256"),
                    "scenario_ids": loaded_ids,
                    "seed_base": _seed_bases((wave,), suite="loaded")[0],
                    "run_sequentially": True,
                    "full_resident_target_overlap_required": True,
                    "scheduler_interval_audit_required_after_each_scenario": True,
                },
                "completion_rule": (
                    "Admit exactly one v4 joint unit only after all seven empty "
                    "cells and both loaded scenarios pass and every measurement-"
                    "call scheduler interval audit passes. Failed attempts retain "
                    "this v4 identity and add no unit."
                ),
                "release_condition": (
                    "nineteen v4 calibration units complete and the v4 margin is frozen"
                    if role == "holdout"
                    else "registered v4 manifest, clean preflight, and interval guard"
                ),
                "holdout_lock": {
                    "locked_until_release": role == "holdout",
                    "required_complete_calibration_units": (
                        len(CALIBRATION_WAVES) if role == "holdout" else 0
                    ),
                    "calibration_margin_must_be_frozen": role == "holdout",
                },
            }
        )
    return items


def _expected_interval_guard_contract() -> dict[str, Any]:
    return {
        "required": True,
        "implementation_module": (
            "algorithm.experiments.critical_gpu_joint95_manifest_dispatcher"
        ),
        "audit_function": "_scheduler_history_interval_audit",
        "queue_snapshot_argument_required": True,
        "measurement_boundaries": {
            "empty": {
                "call": "_run_cell",
                "audit_timing": "immediately_after_each_measurement_call",
                "result_field": "joint95_scheduler_interval_audit",
            },
            "loaded": {
                "call": "_run_loaded_trajectory",
                "audit_timing": "immediately_after_each_measurement_call",
                "result_field": "joint95_scheduler_interval_audit",
            },
        },
        "top_level_audit_field": "scheduler_interval_audits",
        "auto_adopted_child_rule": {
            "current_call": (
                "Exclude only an auto-adopted child with exact current-call "
                "run identity and verified command provenance."
            ),
            "prior_call": (
                "Exclude a prior-call child only when exact provenance and "
                "detached-wrapper/child-wait evidence prove process exit "
                "before the current measurement starts."
            ),
            "unknown_or_live_prior_child": "count as overlap and fail-stop",
            "unrelated_target_node_task": "count as overlap and fail-stop",
        },
        "terminal_history_refresh": {
            "required_before_pass": True,
            "intervals": "the original seven empty and two loaded measurement intervals",
            "row_and_top_level_audits_replaced_together": True,
            "purpose": "catch scheduler tasks recorded after an immediate per-call audit",
        },
        "pass_condition": (
            "No unrelated or live target-node process overlaps a measurement "
            "interval; exclusions require exact child provenance and, for a "
            "prior call, proven exit before the current interval."
        ),
        "failure_action": (
            "Fail-stop the complete v4 campaign, retain the affected attempt, "
            "and do not release the holdout or replace the visit."
        ),
    }


def _validate_v1_failure(
    v1_manifest: Mapping[str, Any], v1_fail_stop: Mapping[str, Any]
) -> None:
    if v1_manifest.get("collection_prefix") != "critical_gpu_joint95_v1":
        raise ValueError("source manifest is not the registered joint95 v1 campaign")
    if v1_manifest.get("registered") is not True:
        raise ValueError("source v1 manifest is not registered")
    if v1_fail_stop.get("status") != "FAIL_STOP":
        raise ValueError("v1 terminal is not a fail-stop")
    if v1_fail_stop.get("decision") != V1_FAILURE_DECISION:
        raise ValueError("v1 fail-stop is not the recorded external-overlap incident")
    incident = v1_fail_stop.get("incident") or {}
    if incident.get("kind") != "target_node_scheduler_task_overlap":
        raise ValueError("v1 failure lacks the target-node overlap diagnosis")
    if (v1_fail_stop.get("adoption") or {}).get("retry_wave23_authorized") is not False:
        raise ValueError("v1 fail-stop does not prohibit a wave-23 retry")


def _validate_v2_failure(
    v2_manifest: Mapping[str, Any], v2_fail_stop: Mapping[str, Any]
) -> None:
    if (
        v2_manifest.get("collection_prefix") != "critical_gpu_joint95_v2"
        or v2_manifest.get("registered") is not True
        or v2_manifest.get("launch_authorized") is not True
    ):
        raise ValueError("source v2 manifest is not the authorized joint95 v2 campaign")
    if (
        v2_fail_stop.get("status") != "FAIL_STOP"
        or v2_fail_stop.get("terminal") is not True
        or v2_fail_stop.get("decision") != V2_FAILURE_DECISION
        or v2_fail_stop.get("failed_wave") != 34
        or v2_fail_stop.get("replacement_attempt_authorized") is not False
    ):
        raise ValueError("source v2 fail-stop does not match the recorded wave-34 incident")


def _validate_v3_failure(
    v3_manifest: Mapping[str, Any], v3_fail_stop: Mapping[str, Any]
) -> None:
    if (
        v3_manifest.get("collection_prefix") != "critical_gpu_joint95_v3"
        or v3_manifest.get("registered") is not True
        or v3_manifest.get("launch_authorized") is not True
    ):
        raise ValueError("source v3 manifest is not the authorized joint95 v3 campaign")
    if (
        v3_fail_stop.get("status") != "FAIL_STOP"
        or v3_fail_stop.get("terminal") is not True
        or v3_fail_stop.get("decision") != V3_FAILURE_DECISION
        or v3_fail_stop.get("failed_wave") != 59
        or v3_fail_stop.get("replacement_attempt_authorized") is not False
    ):
        raise ValueError("source v3 fail-stop does not match the recorded wave-59 incident")
    if "UNDECLARED_HOST_PROCESS_DURING_TARGET" not in str(v3_fail_stop.get("reason") or ""):
        raise ValueError("v3 fail-stop lacks the recorded host-process diagnosis")


def build_registration(
    *,
    v1_manifest_path: Path = DEFAULT_V1_MANIFEST,
    v1_fail_stop_path: Path = DEFAULT_V1_FAIL_STOP,
    v2_manifest_path: Path = DEFAULT_V2_MANIFEST,
    v2_fail_stop_path: Path = DEFAULT_V2_FAIL_STOP,
    v3_manifest_path: Path = DEFAULT_V3_MANIFEST,
    v3_fail_stop_path: Path = DEFAULT_V3_FAIL_STOP,
    results_dir: Path = DEFAULT_RESULTS_DIR,
) -> dict[str, Any]:
    """Build a no-evidence v4 registration from frozen measurement contracts."""

    v1_manifest_path = Path(v1_manifest_path).resolve()
    v1_fail_stop_path = Path(v1_fail_stop_path).resolve()
    v2_manifest_path = Path(v2_manifest_path).resolve()
    v2_fail_stop_path = Path(v2_fail_stop_path).resolve()
    v3_manifest_path = Path(v3_manifest_path).resolve()
    v3_fail_stop_path = Path(v3_fail_stop_path).resolve()
    results_dir = Path(results_dir).resolve()
    v1_manifest = _read_object(v1_manifest_path)
    v1_fail_stop = _read_object(v1_fail_stop_path)
    v2_manifest = _read_object(v2_manifest_path)
    v2_fail_stop = _read_object(v2_fail_stop_path)
    v3_manifest = _read_object(v3_manifest_path)
    v3_fail_stop = _read_object(v3_fail_stop_path)
    _validate_v1_failure(v1_manifest, v1_fail_stop)
    _validate_v2_failure(v2_manifest, v2_fail_stop)
    _validate_v3_failure(v3_manifest, v3_fail_stop)

    current_empty_bundle = empty_suite.measurement_code_manifest()
    current_loaded_bundle = loaded_suite.measurement_code_manifest()
    source_bundles = v1_manifest.get("measurement_bundles") or {}
    if (source_bundles.get("empty") or {}).get("manifest") != current_empty_bundle:
        raise ValueError("current empty measurement bundle differs from frozen v1")
    if (source_bundles.get("loaded") or {}).get("manifest") != current_loaded_bundle:
        raise ValueError("current loaded measurement bundle differs from frozen v1")
    for field in (
        "measurement_bundles", "historical_training_inputs", "empty_contract",
        "loaded_contract", "scored_functionals", "score_contract",
    ):
        if v2_manifest.get(field) != v1_manifest.get(field):
            raise ValueError(f"v2 {field} differs from the frozen v1 contract")
        if v3_manifest.get(field) != v1_manifest.get(field):
            raise ValueError(f"v3 {field} differs from the frozen v1 contract")

    rank = math.ceil((len(CALIBRATION_WAVES) + 1) * (1.0 - MIS_COVERAGE_ALPHA))
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "gate": GATE,
        "status": "READY_TO_REGISTER",
        "registered": False,
        "registration_only": True,
        "protocol": PROTOCOL,
        "collection_prefix": COLLECTION_PREFIX,
        "node": NODE,
        "node_bucket": v1_manifest.get("node_bucket"),
        "hardware_class": v1_manifest.get("hardware_class"),
        "launch_capability": False,
        "launch_authorized": False,
        "network_access_performed": False,
        "measurements_executed": False,
        "evidence_status": "NO_V4_EMPIRICAL_EVIDENCE",
        "coverage_contract": {
            "method": "paired_joint_wave_max_split_conformal",
            "miscoverage_alpha": MIS_COVERAGE_ALPHA,
            "nominal_simultaneous_coverage": 1.0 - MIS_COVERAGE_ALPHA,
            "calibration_unit_count": len(CALIBRATION_WAVES),
            "holdout_unit_count": 1,
            "finite_sample_rank_rule": "ceil((n_calibration + 1) * (1 - alpha))",
            "finite_sample_rank": rank,
            "calibration_joint_waves": list(CALIBRATION_WAVES),
            "holdout_joint_wave": HOLDOUT_WAVE,
        },
        "historical_training_inputs": copy.deepcopy(
            v1_manifest.get("historical_training_inputs") or {}
        ),
        "predictor_compatibility_audit": copy.deepcopy(
            v1_manifest.get("compatibility_audit") or {}
        ),
        "measurement_bundles": copy.deepcopy(source_bundles),
        "empty_contract": copy.deepcopy(v1_manifest.get("empty_contract") or []),
        "loaded_contract": copy.deepcopy(v1_manifest.get("loaded_contract") or []),
        "scored_functionals": copy.deepcopy(
            v1_manifest.get("scored_functionals") or []
        ),
        "score_contract": copy.deepcopy(v1_manifest.get("score_contract") or {}),
        "joint_work_items": [],
        "replacement_basis": {
            "classification": REPLACEMENT_CLASSIFICATION,
            "reason": (
                "v1 stopped on external interval overlap; v2 stopped on its "
                "recorded scheduler-overlap incident; v3 stopped at wave 59 "
                "after undeclared host CPU work during a loaded measurement"
            ),
            "v1_manifest_path": str(v1_manifest_path),
            "v1_fail_stop_path": str(v1_fail_stop_path),
            "v1_failure_decision": V1_FAILURE_DECISION,
            "v1_failed_wave": int((v1_fail_stop.get("campaign") or {})["failed_wave"]),
            "v1_valid_prefix_waves": list(
                (v1_fail_stop.get("campaign") or {}).get(
                    "valid_first_attempt_calibration_waves_before_failure"
                )
                or []
            ),
            "v1_data_interpretation": "mechanically_terminated_not_a_stochastic_outcome",
            "v1_is_retry_parent": False,
            "v1_joint_units_eligible_for_v4_calibration": 0,
            "v1_joint_units_eligible_for_v4_holdout": 0,
            "v1_joint_scores_eligible_for_v4_margin": 0,
            "v2_manifest_path": str(v2_manifest_path),
            "v2_fail_stop_path": str(v2_fail_stop_path),
            "v2_failure_decision": V2_FAILURE_DECISION,
            "v2_failed_wave": int(v2_fail_stop["failed_wave"]),
            "v2_data_interpretation": "terminated_execution_not_a_stochastic_outcome",
            "v2_is_retry_parent": False,
            "v2_joint_units_eligible_for_v4_calibration": 0,
            "v2_joint_units_eligible_for_v4_holdout": 0,
            "v2_joint_scores_eligible_for_v4_margin": 0,
            "v3_manifest_path": str(v3_manifest_path),
            "v3_fail_stop_path": str(v3_fail_stop_path),
            "v3_failure_decision": V3_FAILURE_DECISION,
            "v3_failed_wave": int(v3_fail_stop["failed_wave"]),
            "v3_valid_prefix_waves": list(range(54, 59)),
            "v3_data_interpretation": "terminated_execution_not_a_stochastic_outcome",
            "v3_is_retry_parent": False,
            "v3_joint_units_eligible_for_v4_calibration": 0,
            "v3_joint_units_eligible_for_v4_holdout": 0,
            "v3_joint_scores_eligible_for_v4_margin": 0,
        },
        "predictor_provenance": {
            "source": "same frozen August point predictors registered by v1",
            "v1_measurements_used_to_fit_or_select_predictors": False,
            "v1_results_directory_read_for_predictor_values": False,
            "v2_results_directory_read_for_predictor_values": False,
            "v3_results_directory_read_for_predictor_values": False,
            "v1_role": "failure provenance only",
            "v2_role": "failure provenance only",
            "v3_role": "failure provenance only",
        },
        "seed_independence": {
            "v1_joint_waves": list(V1_WAVES),
            "v2_joint_waves": list(V2_WAVES),
            "v3_joint_waves": list(V3_WAVES),
            "v4_joint_waves": list(JOINT_WAVES),
            "wave_identities_disjoint": set((*V1_WAVES, *V2_WAVES, *V3_WAVES)).isdisjoint(JOINT_WAVES),
            "v1_empty_seed_bases": _seed_bases(V1_WAVES, suite="empty"),
            "v2_empty_seed_bases": _seed_bases(V2_WAVES, suite="empty"),
            "v3_empty_seed_bases": _seed_bases(V3_WAVES, suite="empty"),
            "v4_empty_seed_bases": _seed_bases(JOINT_WAVES, suite="empty"),
            "v1_loaded_seed_bases": _seed_bases(V1_WAVES, suite="loaded"),
            "v2_loaded_seed_bases": _seed_bases(V2_WAVES, suite="loaded"),
            "v3_loaded_seed_bases": _seed_bases(V3_WAVES, suite="loaded"),
            "v4_loaded_seed_bases": _seed_bases(JOINT_WAVES, suite="loaded"),
            "empty_seed_bases_disjoint": set(
                _seed_bases((*V1_WAVES, *V2_WAVES, *V3_WAVES), suite="empty")
            ).isdisjoint(_seed_bases(JOINT_WAVES, suite="empty")),
            "loaded_seed_bases_disjoint": set(
                _seed_bases((*V1_WAVES, *V2_WAVES, *V3_WAVES), suite="loaded")
            ).isdisjoint(_seed_bases(JOINT_WAVES, suite="loaded")),
        },
        "scheduler_interval_guard": _expected_interval_guard_contract(),
        "host_cpu_preflight": copy.deepcopy(HOST_CPU_PREFLIGHT_CONTRACT),
        "launch_preconditions": {
            "v4_executor_validation_required": True,
            "registered_manifest_must_be_consumed_without_rewriting": True,
            "clean_preflight_before_each_measurement_required": True,
            "host_cpu_preflight_before_each_measurement_required": True,
            "scheduler_history_interval_audit_after_each_measurement_required": True,
            "holdout_release_requires_nineteen_v4_calibration_units": True,
            "registration_grants_launch_authority": False,
        },
        "results_contract": {
            "results_directory": str(results_dir),
            "v4_result_prefix": COLLECTION_PREFIX,
            "v1_result_prefix_forbidden": "critical_gpu_joint95_v1",
            "v2_result_prefix_forbidden": "critical_gpu_joint95_v2",
            "v3_result_prefix_forbidden": "critical_gpu_joint95_v3",
            "must_be_empty_at_registration": True,
            "campaign_fail_stop_sentinel": "campaign_fail_stop.json",
        },
        "counting_rule": (
            "Only a complete v4 visit with nine passing functionals and nine "
            "passing scheduler interval audits contributes one v4 joint unit. "
            "No v1/v2/v3 visit, partial v4 visit, failed attempt, or replacement visit counts."
        ),
        "observations": [],
        "completed_joint_units": 0,
        "claim_boundary": (
            "This is a prospective independent-replacement registration. It is "
            "not a v1/v2/v3 retry and contains no v4 measurement, margin, coverage "
            "result, capacity certificate, selector result, or paper claim."
        ),
    }
    report["joint_work_items"] = _work_items(report)
    report["validation"] = validate_registration(report, v1_manifest=v1_manifest)
    if not report["validation"]["pass"]:
        report["status"] = "FAIL_PROTOCOL_VALIDATION"
    return report


def validate_registration(
    report: Mapping[str, Any], *, v1_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    coverage = report.get("coverage_contract") or {}
    replacement = report.get("replacement_basis") or {}
    seed_audit = report.get("seed_independence") or {}
    guard = report.get("scheduler_interval_guard") or {}
    host_preflight = report.get("host_cpu_preflight") or {}
    results = report.get("results_contract") or {}
    items = list(report.get("joint_work_items") or [])

    require(report.get("gate") == GATE, "gate differs")
    require(report.get("protocol") == PROTOCOL, "protocol differs")
    require(report.get("collection_prefix") == COLLECTION_PREFIX, "prefix differs")
    require(report.get("node") == NODE, "node differs")
    require(report.get("registration_only") is True, "registration-only flag missing")
    require(report.get("launch_capability") is False, "registration must not launch")
    launch_authorized = report.get("launch_authorized") is True
    if launch_authorized:
        executor = report.get("executor_validation") or {}
        require(report.get("registered") is True, "only a registered manifest can authorize launch")
        require(
            report.get("status") == "REGISTERED_READY_FOR_SEPARATE_LAUNCH",
            "authorized registration status differs",
        )
        require(executor.get("status") == "PASS", "v4 executor validation did not pass")
        require(
            executor.get("module") == V4_EXECUTOR_MODULE,
            "v4 executor module differs",
        )
        require(
            executor.get("validated_scope") == list(V4_EXECUTOR_TEST_SCOPE),
            "v4 executor validation scope differs",
        )
        require(
            executor.get("requires_nine_scheduler_interval_audits") is True,
            "v4 executor does not require nine interval audits",
        )
        require(
            executor.get("measurements_executed_before_authorization") is False,
            "executor authorization followed measurement execution",
        )
    else:
        require(
            report.get("status")
            in {
                "READY_TO_REGISTER",
                "REGISTERED_AWAITING_V4_EXECUTOR_VALIDATION",
                "FAIL_PROTOCOL_VALIDATION",
            },
            "unauthorized registration status differs",
        )
    require(report.get("network_access_performed") is False, "network activity is forbidden")
    require(report.get("measurements_executed") is False, "registration cannot execute measurements")
    require(report.get("evidence_status") == "NO_V4_EMPIRICAL_EVIDENCE", "registration claims evidence")
    require(not (report.get("observations") or []), "registration contains observations")
    require(int(report.get("completed_joint_units", -1)) == 0, "registration contains completed units")

    require(int(coverage.get("calibration_unit_count") or 0) == 19, "need 19 calibration visits")
    require(int(coverage.get("holdout_unit_count") or 0) == 1, "need one holdout")
    require(int(coverage.get("finite_sample_rank") or 0) == 19, "rank must be 19")
    require(coverage.get("calibration_joint_waves") == list(CALIBRATION_WAVES), "calibration waves differ")
    require(coverage.get("holdout_joint_wave") == HOLDOUT_WAVE, "holdout wave differs")

    require(report.get("measurement_bundles") == v1_manifest.get("measurement_bundles"), "measurement bundles differ from v1")
    require(report.get("historical_training_inputs") == v1_manifest.get("historical_training_inputs"), "frozen predictor inputs differ from v1")
    require(report.get("predictor_compatibility_audit") == v1_manifest.get("compatibility_audit"), "predictor compatibility audit differs from v1")
    require(report.get("empty_contract") == v1_manifest.get("empty_contract"), "empty contract differs from v1")
    require(report.get("loaded_contract") == v1_manifest.get("loaded_contract"), "loaded contract differs from v1")
    require(report.get("scored_functionals") == v1_manifest.get("scored_functionals"), "scored functionals differ from v1")
    require(report.get("score_contract") == v1_manifest.get("score_contract"), "score contract differs from v1")
    require(len(report.get("empty_contract") or []) == 7, "need seven empty functionals")
    require(len(report.get("loaded_contract") or []) == 2, "need two loaded functionals")
    require(len(report.get("scored_functionals") or []) == 9, "need nine scored functionals")

    require(len(items) == 20, "need twenty v4 visits")
    require([int(row.get("joint_wave") or -1) for row in items] == list(JOINT_WAVES), "v4 wave identities differ")
    require([row.get("split_role") for row in items] == ["calibration"] * 19 + ["holdout"], "v4 split roles differ")
    require(items == _work_items(report), "v4 work items or seed assignments differ")
    ids = [str(row.get("joint_unit_id") or "") for row in items]
    require(len(set(ids)) == 20 and all(value.startswith(COLLECTION_PREFIX) for value in ids), "v4 joint identities are not unique")
    require(all(int(row.get("observation_count") or 0) == 9 for row in items), "each visit needs nine observations")
    require(sum(int(row.get("observation_count") or 0) for row in items) == 180, "need 180 registered observations")

    require(replacement.get("classification") == REPLACEMENT_CLASSIFICATION, "replacement classification differs")
    require(replacement.get("v1_failure_decision") == V1_FAILURE_DECISION, "v1 mechanical failure not preserved")
    require(replacement.get("v2_failure_decision") == V2_FAILURE_DECISION, "v2 execution failure not preserved")
    require(replacement.get("v3_failure_decision") == V3_FAILURE_DECISION, "v3 execution failure not preserved")
    require(replacement.get("v2_failed_wave") == 34, "v2 failed wave differs")
    require(replacement.get("v3_failed_wave") == 59, "v3 failed wave differs")
    require(replacement.get("v2_manifest_path") == str(DEFAULT_V2_MANIFEST.resolve()), "v2 manifest path differs")
    require(replacement.get("v2_fail_stop_path") == str(DEFAULT_V2_FAIL_STOP.resolve()), "v2 fail-stop path differs")
    require(replacement.get("v3_manifest_path") == str(DEFAULT_V3_MANIFEST.resolve()), "v3 manifest path differs")
    require(replacement.get("v3_fail_stop_path") == str(DEFAULT_V3_FAIL_STOP.resolve()), "v3 fail-stop path differs")
    require(replacement.get("v1_data_interpretation") == "mechanically_terminated_not_a_stochastic_outcome", "v1 is treated as a stochastic outcome")
    require(replacement.get("v2_data_interpretation") == "terminated_execution_not_a_stochastic_outcome", "v2 is treated as a stochastic outcome")
    require(replacement.get("v3_data_interpretation") == "terminated_execution_not_a_stochastic_outcome", "v3 is treated as a stochastic outcome")
    require(replacement.get("v1_is_retry_parent") is False, "v4 must not be a v1 retry")
    require(replacement.get("v2_is_retry_parent") is False, "v4 must not be a v2 retry")
    require(replacement.get("v3_is_retry_parent") is False, "v4 must not be a v3 retry")
    for field in (
        "v1_joint_units_eligible_for_v4_calibration",
        "v1_joint_units_eligible_for_v4_holdout",
        "v1_joint_scores_eligible_for_v4_margin",
        "v2_joint_units_eligible_for_v4_calibration",
        "v2_joint_units_eligible_for_v4_holdout",
        "v2_joint_scores_eligible_for_v4_margin",
        "v3_joint_units_eligible_for_v4_calibration",
        "v3_joint_units_eligible_for_v4_holdout",
        "v3_joint_scores_eligible_for_v4_margin",
    ):
        require(int(replacement.get(field, -1)) == 0, f"{field} must be zero")

    require(seed_audit.get("wave_identities_disjoint") is True, "v1/v4 waves overlap")
    require(seed_audit.get("empty_seed_bases_disjoint") is True, "empty seeds overlap")
    require(seed_audit.get("loaded_seed_bases_disjoint") is True, "loaded seeds overlap")
    require(set(seed_audit.get("v1_joint_waves") or []).isdisjoint(seed_audit.get("v4_joint_waves") or []), "recorded v1/v4 waves overlap")
    require(seed_audit.get("v2_joint_waves") == list(V2_WAVES), "recorded v2 waves differ")
    require(seed_audit.get("v4_joint_waves") == list(JOINT_WAVES), "recorded v4 waves differ")
    require(set(seed_audit.get("v2_joint_waves") or []).isdisjoint(seed_audit.get("v4_joint_waves") or []), "recorded v2/v4 waves overlap")
    require(seed_audit.get("v3_joint_waves") == list(V3_WAVES), "recorded v3 waves differ")
    require(set(seed_audit.get("v3_joint_waves") or []).isdisjoint(seed_audit.get("v4_joint_waves") or []), "recorded v3/v4 waves overlap")
    require(set(seed_audit.get("v1_empty_seed_bases") or []).isdisjoint(seed_audit.get("v4_empty_seed_bases") or []), "recorded empty seed bases overlap")
    require(seed_audit.get("v2_empty_seed_bases") == _seed_bases(V2_WAVES, suite="empty"), "recorded v2 empty seeds differ")
    require(seed_audit.get("v4_empty_seed_bases") == _seed_bases(JOINT_WAVES, suite="empty"), "recorded v4 empty seeds differ")
    require(set(seed_audit.get("v2_empty_seed_bases") or []).isdisjoint(seed_audit.get("v4_empty_seed_bases") or []), "recorded v2/v4 empty seed bases overlap")
    require(seed_audit.get("v3_empty_seed_bases") == _seed_bases(V3_WAVES, suite="empty"), "recorded v3 empty seeds differ")
    require(set(seed_audit.get("v3_empty_seed_bases") or []).isdisjoint(seed_audit.get("v4_empty_seed_bases") or []), "recorded v3/v4 empty seed bases overlap")
    require(set(seed_audit.get("v1_loaded_seed_bases") or []).isdisjoint(seed_audit.get("v4_loaded_seed_bases") or []), "recorded loaded seed bases overlap")
    require(seed_audit.get("v2_loaded_seed_bases") == _seed_bases(V2_WAVES, suite="loaded"), "recorded v2 loaded seeds differ")
    require(seed_audit.get("v4_loaded_seed_bases") == _seed_bases(JOINT_WAVES, suite="loaded"), "recorded v4 loaded seeds differ")
    require(set(seed_audit.get("v2_loaded_seed_bases") or []).isdisjoint(seed_audit.get("v4_loaded_seed_bases") or []), "recorded v2/v4 loaded seed bases overlap")
    require(seed_audit.get("v3_loaded_seed_bases") == _seed_bases(V3_WAVES, suite="loaded"), "recorded v3 loaded seeds differ")
    require(set(seed_audit.get("v3_loaded_seed_bases") or []).isdisjoint(seed_audit.get("v4_loaded_seed_bases") or []), "recorded v3/v4 loaded seed bases overlap")

    require(guard == _expected_interval_guard_contract(), "interval-overlap guard contract differs")
    require(host_preflight == HOST_CPU_PREFLIGHT_CONTRACT, "host CPU preflight contract differs")
    require(guard.get("required") is True, "interval-overlap guard is optional")
    require((report.get("launch_preconditions") or {}).get("scheduler_history_interval_audit_after_each_measurement_required") is True, "launch lacks interval-audit prerequisite")
    require((report.get("launch_preconditions") or {}).get("host_cpu_preflight_before_each_measurement_required") is True, "launch lacks host CPU preflight")
    require(results.get("results_directory") != str(DEFAULT_V1_FAIL_STOP.parent), "v4 results reuse the v1 directory")
    require(results.get("results_directory") != str(DEFAULT_V2_FAIL_STOP.parent), "v4 results reuse the v2 directory")
    require(results.get("results_directory") != str(DEFAULT_V3_FAIL_STOP.parent), "v4 results reuse the v3 directory")
    require(results.get("v4_result_prefix") == COLLECTION_PREFIX, "v4 result prefix differs")
    require(results.get("v1_result_prefix_forbidden") == "critical_gpu_joint95_v1", "v1 results are not excluded")
    require(results.get("v2_result_prefix_forbidden") == "critical_gpu_joint95_v2", "v2 results are not excluded")
    require(results.get("v3_result_prefix_forbidden") == "critical_gpu_joint95_v3", "v3 results are not excluded")
    require(results.get("must_be_empty_at_registration") is True, "v4 results need a fresh directory")

    return {
        "pass": not errors,
        "status": "PASS" if not errors else "FAIL_VALIDATION",
        "error_count": len(errors),
        "errors": errors,
        "joint_unit_count": len(items),
        "observation_count": sum(int(row.get("observation_count") or 0) for row in items),
    }


def write_registration(
    report: Mapping[str, Any],
    *,
    output: Path = DEFAULT_OUTPUT,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    v1_manifest_path: Path = DEFAULT_V1_MANIFEST,
) -> dict[str, Any]:
    """Write the manifest once and create a fresh, empty v4 results directory."""

    output = Path(output).resolve()
    results_dir = Path(results_dir).resolve()
    v1_manifest = _read_object(Path(v1_manifest_path).resolve())
    replacement = report.get("replacement_basis") or {}
    v2_manifest = _read_object(Path(str(replacement.get("v2_manifest_path") or "")))
    v2_fail_stop = _read_object(Path(str(replacement.get("v2_fail_stop_path") or "")))
    v3_manifest = _read_object(Path(str(replacement.get("v3_manifest_path") or "")))
    v3_fail_stop = _read_object(Path(str(replacement.get("v3_fail_stop_path") or "")))
    _validate_v2_failure(v2_manifest, v2_fail_stop)
    _validate_v3_failure(v3_manifest, v3_fail_stop)
    validation = validate_registration(report, v1_manifest=v1_manifest)
    if not validation["pass"]:
        raise ValueError(f"refusing invalid v4 registration: {validation['errors']!r}")
    if output.exists():
        raise FileExistsError(output)
    registered_results_dir = str(
        (report.get("results_contract") or {}).get("results_directory") or ""
    )
    if registered_results_dir != str(results_dir):
        raise ValueError(
            "write results directory differs from the registered results directory"
        )
    if results_dir.exists() and any(results_dir.iterdir()):
        raise ValueError(f"v4 results directory is not empty: {results_dir}")
    results_dir.mkdir(parents=True, exist_ok=True)

    registered = copy.deepcopy(dict(report))
    registered["registered"] = True
    registered["status"] = "REGISTERED_AWAITING_V4_EXECUTOR_VALIDATION"
    registered["registration_snapshot"] = {
        "results_directory_exists": True,
        "results_file_count": 0,
        "measurements_executed": False,
        "launch_authorized": False,
    }
    registered["validation"] = validate_registration(
        registered, v1_manifest=v1_manifest
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(registered, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return registered


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Register the independent no-evidence joint95 v4 campaign."
    )
    parser.add_argument("--v1-manifest", type=Path, default=DEFAULT_V1_MANIFEST)
    parser.add_argument("--v1-fail-stop", type=Path, default=DEFAULT_V1_FAIL_STOP)
    parser.add_argument("--v3-manifest", type=Path, default=DEFAULT_V3_MANIFEST)
    parser.add_argument("--v3-fail-stop", type=Path, default=DEFAULT_V3_FAIL_STOP)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = build_registration(
        v1_manifest_path=args.v1_manifest,
        v1_fail_stop_path=args.v1_fail_stop,
        v3_manifest_path=args.v3_manifest,
        v3_fail_stop_path=args.v3_fail_stop,
        results_dir=args.results_dir,
    )
    registered = write_registration(
        report,
        output=args.output,
        results_dir=args.results_dir,
        v1_manifest_path=args.v1_manifest,
    )
    print(
        json.dumps(
            {
                "status": registered["status"],
                "registered": registered["registered"],
                "validation": registered["validation"],
                "results_directory": registered["results_contract"][
                    "results_directory"
                ],
                "launch_authorized": registered["launch_authorized"],
                "evidence_status": registered["evidence_status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
