"""Freeze and execute the joint95-v5 physical-capacity closure.

Registration consumes the frozen v5 manifest and fixed pre-campaign model
artifacts, but never lists or reads the future v5 results directory.  Execution
accepts only the independent v5 waves 94--112 and post-release holdout wave 113;
v1-v4 joint visits have zero evidential role.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping

from algorithm.experiments import aligned_hardware_joint95_capacity_closure as base
from algorithm.experiments import critical_gpu_joint95_v5_dispatcher as dispatcher
from algorithm.experiments import critical_gpu_joint95_v5_registration as campaign


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
DEFAULT_COLLECTION_MANIFEST = campaign.DEFAULT_OUTPUT
DEFAULT_RESULTS_DIR = campaign.DEFAULT_RESULTS_DIR
DEFAULT_LCB_PROTOCOL = base.DEFAULT_LCB_PROTOCOL
DEFAULT_COUNT_PROTOCOL = base.DEFAULT_COUNT_PROTOCOL
DEFAULT_COUNT_RESULT = base.DEFAULT_COUNT_RESULT
DEFAULT_REGISTRATION = ARTIFACTS / (
    "aligned_hardware_joint95_v5_capacity_closure_20260919_protocol.json"
)
DEFAULT_RESULT = ARTIFACTS / (
    "aligned_hardware_joint95_v5_capacity_closure_20260919.json"
)

EXPERIMENT = "aligned_hardware_joint95_v5_capacity_closure"
GATE = "aligned_hardware_joint95_v5_capacity_closure_runner"
STATUS = "REGISTERED_AWAITING_JOINT95_V5_DATA"
DECISION_RULE = base.DECISION_RULE
ClosureRefusal = base.ClosureRefusal


def _collection_contract(
    manifest: Mapping[str, Any], *, results_dir: Path
) -> dict[str, Any]:
    coverage = manifest.get("coverage_contract") or {}
    validation = manifest.get("validation") or {}
    executor = manifest.get("executor_validation") or {}
    authorization = manifest.get("authorization_snapshot") or {}
    results = manifest.get("results_contract") or {}
    replacement = manifest.get("replacement_basis") or {}
    required = {
        "schema_version": campaign.SCHEMA_VERSION,
        "gate": campaign.GATE,
        "status": "REGISTERED_READY_FOR_SEPARATE_LAUNCH",
        "registered": True,
        "registration_only": True,
        "protocol": campaign.PROTOCOL,
        "collection_prefix": campaign.COLLECTION_PREFIX,
        "node": campaign.NODE,
        "launch_authorized": True,
        "measurements_executed": False,
        "completed_joint_units": 0,
        "evidence_status": "NO_V5_EMPIRICAL_EVIDENCE",
    }
    for key, value in required.items():
        if manifest.get(key) != value:
            raise ClosureRefusal(f"v5 collection manifest {key} differs")
    if manifest.get("observations") not in ([], None):
        raise ClosureRefusal("v5 registration contains empirical observations")
    if validation.get("pass") is not True or validation.get("error_count") != 0:
        raise ClosureRefusal("v5 collection manifest validation is not passing")
    if (
        executor.get("status") != "PASS"
        or executor.get("module") != campaign.V5_EXECUTOR_MODULE
        or executor.get("validated_scope")
        != list(campaign.V5_EXECUTOR_TEST_SCOPE)
        or executor.get("requires_nine_scheduler_interval_audits") is not True
        or executor.get("measurements_executed_before_authorization") is not False
        or executor.get("results_file_count_before_authorization") != 0
    ):
        raise ClosureRefusal("v5 executor validation contract differs")
    if (
        coverage.get("calibration_joint_waves")
        != list(campaign.CALIBRATION_WAVES)
        or coverage.get("holdout_joint_wave") != campaign.HOLDOUT_WAVE
        or coverage.get("calibration_unit_count") != 19
        or coverage.get("holdout_unit_count") != 1
        or coverage.get("finite_sample_rank") != 19
        or float(coverage.get("miscoverage_alpha") or 0.0) != 0.05
    ):
        raise ClosureRefusal("v5 collection coverage contract differs")

    functionals = list(manifest.get("scored_functionals") or [])
    if len(functionals) != 9 or any(
        row.get("direction") != "upper_time" for row in functionals
    ):
        raise ClosureRefusal("v5 collection must contain nine upper-time functionals")
    if manifest.get("score_contract") != base.SCORE_CONTRACT:
        raise ClosureRefusal("v5 collection score contract differs")

    history = manifest.get("historical_training_inputs") or {}
    if (
        history.get("role") != "frozen_point_predictors_only"
        or history.get("eligible_joint_calibration_units") != 0
        or history.get("post_hoc_pairing_forbidden") is not True
    ):
        raise ClosureRefusal("frozen predictor role differs")

    expected_results = Path(results_dir).resolve()
    registered_results = Path(str(results.get("results_directory") or "")).resolve()
    if registered_results != expected_results:
        raise ClosureRefusal("v5 results directory differs from its manifest")
    if authorization != {
        "results_directory": str(registered_results),
        "results_file_count": 0,
        "measurements_executed": False,
        "completed_joint_units": 0,
        "launch_performed": False,
    }:
        raise ClosureRefusal("v5 executor authorization snapshot differs")
    if (
        results.get("v5_result_prefix") != campaign.COLLECTION_PREFIX
        or results.get("v1_result_prefix_forbidden") != "critical_gpu_joint95_v1"
        or results.get("v2_result_prefix_forbidden") != "critical_gpu_joint95_v2"
        or results.get("v3_result_prefix_forbidden") != "critical_gpu_joint95_v3"
        or results.get("v4_result_prefix_forbidden") != "critical_gpu_joint95_v4"
        or results.get("premeasurement_controller_state_directory")
        != dispatcher.CONTROLLER_STATE_DIRNAME
        or results.get("campaign_fail_stop_sentinel")
        != dispatcher.FAIL_STOP_FILENAME
    ):
        raise ClosureRefusal("v5 result identity or fail-stop contract differs")
    if expected_results == campaign.DEFAULT_V1_FAIL_STOP.resolve().parent:
        raise ClosureRefusal("v5 capacity closure cannot use the v1 results directory")
    if expected_results == campaign.DEFAULT_V2_FAIL_STOP.resolve().parent:
        raise ClosureRefusal("v5 capacity closure cannot use the v2 results directory")
    if expected_results == campaign.DEFAULT_V3_FAIL_STOP.resolve().parent:
        raise ClosureRefusal("v5 capacity closure cannot use the v3 results directory")
    if expected_results == campaign.DEFAULT_V4_RESULTS_DIR.resolve():
        raise ClosureRefusal("v5 capacity closure cannot use the v4 results directory")

    zero_fields = (
        "v1_joint_units_eligible_for_v5_calibration",
        "v1_joint_units_eligible_for_v5_holdout",
        "v1_joint_scores_eligible_for_v5_margin",
        "v2_joint_units_eligible_for_v5_calibration",
        "v2_joint_units_eligible_for_v5_holdout",
        "v2_joint_scores_eligible_for_v5_margin",
        "v3_joint_units_eligible_for_v5_calibration",
        "v3_joint_units_eligible_for_v5_holdout",
        "v3_joint_scores_eligible_for_v5_margin",
        "v4_joint_units_eligible_for_v5_calibration",
        "v4_joint_units_eligible_for_v5_holdout",
        "v4_joint_scores_eligible_for_v5_margin",
    )
    if any(int(replacement.get(field, -1)) != 0 for field in zero_fields):
        raise ClosureRefusal("v1-v4 joint evidence is eligible under the v5 manifest")
    if any(replacement.get(f"v{version}_is_retry_parent") is not False for version in (1, 2, 3, 4)):
        raise ClosureRefusal("v5 is incorrectly marked as a retry")

    items = list(manifest.get("joint_work_items") or [])
    expected_waves = (*campaign.CALIBRATION_WAVES, campaign.HOLDOUT_WAVE)
    if (
        len(items) != 20
        or tuple(int(row.get("joint_wave") or -1) for row in items)
        != expected_waves
        or [row.get("split_role") for row in items]
        != ["calibration"] * 19 + ["holdout"]
        or any(
            not str(row.get("joint_unit_id") or "").startswith(
                campaign.COLLECTION_PREFIX
            )
            for row in items
        )
    ):
        raise ClosureRefusal("v5 joint work-item split differs")

    return {
        "schema_version": manifest["schema_version"],
        "gate": manifest["gate"],
        "status": manifest["status"],
        "registered": manifest["registered"],
        "protocol": manifest["protocol"],
        "collection_prefix": manifest["collection_prefix"],
        "node": manifest["node"],
        "node_bucket": manifest.get("node_bucket"),
        "hardware_class": manifest.get("hardware_class"),
        "coverage_contract": base._json(coverage),
        "score_contract": base._json(manifest["score_contract"]),
        "scored_functionals": base._json(functionals),
        "joint_work_items": [
            {
                "joint_wave": row["joint_wave"],
                "joint_unit_id": row["joint_unit_id"],
                "split_role": row["split_role"],
                "scored_functional_ids": base._json(
                    row.get("scored_functional_ids") or []
                ),
            }
            for row in items
        ],
        "results_contract": {
            "results_directory": str(registered_results),
            "v5_result_prefix": results["v5_result_prefix"],
            "v1_result_prefix_forbidden": results["v1_result_prefix_forbidden"],
            "v2_result_prefix_forbidden": results["v2_result_prefix_forbidden"],
            "v3_result_prefix_forbidden": results["v3_result_prefix_forbidden"],
            "v4_result_prefix_forbidden": results["v4_result_prefix_forbidden"],
            "premeasurement_controller_state_directory": results[
                "premeasurement_controller_state_directory"
            ],
            "campaign_fail_stop_sentinel": results[
                "campaign_fail_stop_sentinel"
            ],
        },
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
        "prior_joint_evidence_eligibility": {
            field: int(replacement[field]) for field in zero_fields
        },
    }


def _score_and_envelope_contract() -> dict[str, Any]:
    return {
        "cell_score": "actual_time / frozen_predicted_time - 1",
        "joint_wave_score": "max of the exactly nine registered cell scores",
        "finite_sample_rank": 19,
        "raw_rank_score": "sorted(v5_calibration_joint_scores)[18]",
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
    }


def build_registration(
    *,
    collection_manifest_path: Path = DEFAULT_COLLECTION_MANIFEST,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    lcb_protocol_path: Path = DEFAULT_LCB_PROTOCOL,
    count_protocol_path: Path = DEFAULT_COUNT_PROTOCOL,
    count_result_path: Path = DEFAULT_COUNT_RESULT,
) -> dict[str, Any]:
    """Build the protocol without inspecting any joint-results directory."""

    collection_manifest_path = Path(collection_manifest_path).resolve()
    frozen_results_dir = Path(results_dir).resolve()
    lcb_protocol_path = Path(lcb_protocol_path).resolve()
    count_protocol_path = Path(count_protocol_path).resolve()
    count_result_path = Path(count_result_path).resolve()

    manifest = base._read_object(collection_manifest_path)
    collection = _collection_contract(manifest, results_dir=frozen_results_dir)
    anchor = base._predictor_anchor(manifest)
    lcb_protocol = base._read_object(lcb_protocol_path)
    count_protocol = base._read_object(count_protocol_path)
    count_result = base._read_object(count_result_path)
    if lcb_protocol.get("experiment") != "aligned_hardware_lcb_frame_envelope_shadow_v1":
        raise ClosureRefusal("LCB protocol experiment differs")
    if tuple(float(value) for value in lcb_protocol.get("loads") or ()) != base.LOADS:
        raise ClosureRefusal("LCB load roster differs")
    point_model = base._model_from_snapshot(lcb_protocol.get("point_model") or {})
    count_model = base._model_from_snapshot(count_protocol.get("model") or {})
    if point_model.snapshot() != count_model.snapshot():
        raise ClosureRefusal("LCB and count-space point-model contracts differ")
    if point_model.node != base.NODE or point_model.lanes != base.BASE_LANES:
        raise ClosureRefusal("point reference must be the registered R2 jtl110gpu2 model")
    mappings = base._action_mapping(lcb_protocol, anchor)
    baseline_model, baseline_envelopes = base._planned_baseline_model(
        point_model, anchor, mappings, margin=0.0
    )
    candidate_source = base._candidate_source_snapshot(count_protocol, count_result)
    expected_pairs = sorted(
        (int(row["lanes"]), float(row["load"]))
        for row in count_protocol.get("primary_cases") or []
    )
    if expected_pairs != [
        (lanes, load)
        for lanes in base.CAPACITY_LANES
        for load in base.CANDIDATE_LOADS
    ]:
        raise ClosureRefusal("count-space candidate lane/load roster differs")

    release_path = frozen_results_dir / "holdout_release.json"
    return {
        "schema_version": 1,
        "gate": GATE,
        "experiment": EXPERIMENT,
        "status": STATUS,
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
            "v5_joint_results_directory_read": False,
            "v5_joint_visit_values_read": False,
            "v1_joint_results_directory_read": False,
            "v1_joint_visit_values_read": False,
            "v2_joint_results_directory_read": False,
            "v2_joint_visit_values_read": False,
            "v3_joint_results_directory_read": False,
            "v3_joint_visit_values_read": False,
            "v4_joint_results_directory_read": False,
            "v4_joint_visit_values_read": False,
            "margin_computed": False,
            "trajectory_built": False,
            "selector_invoked": False,
            "server_pull_performed": False,
            "rng_draws": 0,
        },
        "joint_data_eligibility": {
            "eligible_campaign": campaign.COLLECTION_PREFIX,
            "eligible_calibration_waves": list(campaign.CALIBRATION_WAVES),
            "eligible_holdout_wave": campaign.HOLDOUT_WAVE,
            "v1_joint_units_used": 0,
            "v1_joint_scores_used": 0,
            "v1_joint_results_read": 0,
            "v2_joint_units_used": 0,
            "v2_joint_scores_used": 0,
            "v2_joint_results_read": 0,
            "v3_joint_units_used": 0,
            "v3_joint_scores_used": 0,
            "v3_joint_results_read": 0,
            "v4_joint_units_used": 0,
            "v4_joint_scores_used": 0,
            "v4_joint_results_read": 0,
        },
        "collection_contract": collection,
        "predictor_anchor": anchor,
        "score_and_envelope_contract": _score_and_envelope_contract(),
        "action_mapping": mappings,
        "zero_margin_action_envelopes": baseline_envelopes,
        "point_reference_model": base._json(point_model.snapshot()),
        "fixed_capacity_contract": base._reference_capacity_contract(point_model),
        "candidate_evidence_source": candidate_source,
        "candidate_evidence_reuse_pre_audit": base._registration_candidate_audit(
            point_model, baseline_model
        ),
        "run_preconditions": [
            "v5 waves 94..112 each have exactly one dispatcher-valid first-attempt PASS terminal",
            "all nineteen v5 calibration terminals equal the registered predictor anchor",
            "the v5 holdout_release validates and freezes rank 19 (the maximum)",
            "v5 wave 113 has exactly one dispatcher-valid post-release PASS terminal",
            "no v1-v4 or unregistered result artifact is present in the v5 results directory",
            "all frozen source contracts still match this registration",
        ],
        "decision_rule": dict(DECISION_RULE),
        "prohibited": [
            "reading v5 visit values or the v5 results directory during registration",
            "using any v1-v4 joint visit, score, margin, release, or results directory",
            "post-result changes to functionals, score, rank, action mapping, lanes, loads, or arrival direction",
            "new trajectory construction or post-result load selection",
            "selector execution in this runner",
            "numeric reuse of epsilon when model or state-population contracts differ",
            "server pulls or paper edits",
        ],
        "claim_boundary": (
            "A one-future-visit 95% joint upper-time audit and conditional local "
            "complete-frame capacity calculation based only on the independent "
            "joint95-v5 19+1 cohort. It is not time-uniform physical coverage, a "
            "trajectory result, or candidate-search closure without separate "
            "same-model static support evidence."
        ),
    }


def register(output: Path = DEFAULT_REGISTRATION, **kwargs: Any) -> dict[str, Any]:
    output = Path(output)
    if output.exists():
        raise ClosureRefusal(f"registration artifact already exists: {output}")
    protocol = build_registration(**kwargs)
    base._write_new(output, protocol)
    return protocol


def _current_source_snapshot(protocol: Mapping[str, Any]) -> dict[str, Any]:
    paths = protocol.get("source_paths") or {}
    manifest = base._read_object(Path(paths["collection_manifest"]))
    collection = _collection_contract(
        manifest, results_dir=Path(paths["joint_results_directory"])
    )
    anchor = base._predictor_anchor(manifest)
    lcb_protocol = base._read_object(Path(paths["lcb_envelope_protocol"]))
    count_protocol = base._read_object(Path(paths["count_space_trajectory_protocol"]))
    count_result = base._read_object(Path(paths["count_space_trajectory_result"]))
    point_model = base._model_from_snapshot(lcb_protocol.get("point_model") or {})
    mappings = base._action_mapping(lcb_protocol, anchor)
    candidate_source = base._candidate_source_snapshot(count_protocol, count_result)
    return {
        "manifest": manifest,
        "collection_contract": collection,
        "predictor_anchor": anchor,
        "point_model": point_model,
        "point_reference_model": base._json(point_model.snapshot()),
        "action_mapping": mappings,
        "candidate_evidence_source": candidate_source,
    }


def _validate_registered_sources(
    protocol: Mapping[str, Any], current: Mapping[str, Any]
) -> None:
    checks = (
        ("v5 collection contract", current["collection_contract"], protocol.get("collection_contract")),
        ("predictor anchor", current["predictor_anchor"], protocol.get("predictor_anchor")),
        ("point reference model", current["point_reference_model"], protocol.get("point_reference_model")),
        ("action mapping", current["action_mapping"], protocol.get("action_mapping")),
        ("candidate evidence source", current["candidate_evidence_source"], protocol.get("candidate_evidence_source")),
    )
    for label, actual, expected in checks:
        if actual != expected:
            raise ClosureRefusal(f"{label} changed after registration")
    point_model = current["point_model"]
    baseline_model, baseline_envelopes = base._planned_baseline_model(
        point_model,
        current["predictor_anchor"],
        current["action_mapping"],
        margin=0.0,
    )
    if base._json(baseline_envelopes) != protocol.get("zero_margin_action_envelopes"):
        raise ClosureRefusal("zero-margin action-envelope contract changed")
    if base._reference_capacity_contract(point_model) != protocol.get(
        "fixed_capacity_contract"
    ):
        raise ClosureRefusal("fixed capacity direction or roster changed")
    if base._registration_candidate_audit(
        point_model, baseline_model
    ) != protocol.get("candidate_evidence_reuse_pre_audit"):
        raise ClosureRefusal("candidate-evidence reuse pre-audit changed")
    paths = protocol.get("source_paths") or {}
    results_dir = Path(paths["joint_results_directory"]).resolve()
    if Path(paths["holdout_release"]).resolve() != results_dir / "holdout_release.json":
        raise ClosureRefusal("holdout release path is outside the frozen v5 results directory")
    eligibility = protocol.get("joint_data_eligibility") or {}
    if eligibility != {
        "eligible_campaign": campaign.COLLECTION_PREFIX,
        "eligible_calibration_waves": list(campaign.CALIBRATION_WAVES),
        "eligible_holdout_wave": campaign.HOLDOUT_WAVE,
        "v1_joint_units_used": 0,
        "v1_joint_scores_used": 0,
        "v1_joint_results_read": 0,
        "v2_joint_units_used": 0,
        "v2_joint_scores_used": 0,
        "v2_joint_results_read": 0,
        "v3_joint_units_used": 0,
        "v3_joint_scores_used": 0,
        "v3_joint_results_read": 0,
        "v4_joint_units_used": 0,
        "v4_joint_scores_used": 0,
        "v4_joint_results_read": 0,
    }:
        raise ClosureRefusal("v5-only joint-data eligibility changed")


def _complete_v5_joint_evidence(
    protocol: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    results_dir = Path((protocol.get("source_paths") or {})["joint_results_directory"])
    fail_stop = dispatcher._campaign_fail_stop(results_dir)
    if fail_stop is not None:
        raise ClosureRefusal("v5 campaign fail-stop sentinel is active")
    foreign = dispatcher._foreign_result_paths(manifest, results_dir)
    if foreign:
        raise ClosureRefusal(
            "v1-v4 or unregistered result artifacts are present in the v5 results directory"
        )
    scans = dispatcher._scan_results(manifest, results_dir)
    for wave in (*campaign.CALIBRATION_WAVES, campaign.HOLDOUT_WAVE):
        scan = scans.get(wave) or {}
        attempts = list(scan.get("attempts") or [])
        successes = list(scan.get("successes") or [])
        if len(attempts) != 1 or len(successes) != 1:
            raise ClosureRefusal(
                f"v5 wave {wave} requires exactly one passing first attempt; "
                f"found {len(attempts)} attempts and {len(successes)} valid PASS terminals"
            )
    release_path = Path((protocol.get("source_paths") or {})["holdout_release"])
    try:
        release = dispatcher._read_object(release_path)
    except Exception as exc:
        raise ClosureRefusal(
            f"explicit v5 holdout release is unavailable: {exc}"
        ) from exc
    release_audit = dispatcher.validate_holdout_release(release, manifest, scans)
    if release_audit.get("pass") is not True:
        raise ClosureRefusal(
            "explicit v5 holdout release is invalid: "
            + "; ".join(str(value) for value in release_audit.get("errors") or [])
        )

    anchor = protocol.get("predictor_anchor") or []
    calibration = []
    for wave in campaign.CALIBRATION_WAVES:
        row = scans[wave]["successes"][0]
        calibration.append(
            {
                "path": str(row["path"]),
                **base._score_success(row["payload"], anchor),
            }
        )
    holdout_row = scans[campaign.HOLDOUT_WAVE]["successes"][0]
    holdout = {
        "path": str(holdout_row["path"]),
        **base._score_success(holdout_row["payload"], anchor),
    }
    try:
        released_at = float(release["released_at"])
        holdout_started_at = float(holdout_row["payload"]["started_at"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ClosureRefusal(
            "v5 release/holdout sequencing times are unavailable"
        ) from exc
    if (
        not math.isfinite(released_at)
        or not math.isfinite(holdout_started_at)
        or holdout_started_at < released_at
    ):
        raise ClosureRefusal(
            f"v5 wave {campaign.HOLDOUT_WAVE} is not an independent post-release holdout"
        )
    scores = [row["joint_wave_score"] for row in calibration]
    raw_rank_score, margin = base._rank19_margin(scores)
    frozen_margin = float(release_audit["frozen_margin"])
    if not math.isclose(
        margin,
        frozen_margin,
        rel_tol=base.TOLERANCE,
        abs_tol=base.TOLERANCE,
    ):
        raise ClosureRefusal(
            "v5 release did not freeze the recomputed rank-19 margin"
        )
    return {
        "calibration": calibration,
        "calibration_joint_scores": scores,
        "raw_rank19_score": raw_rank_score,
        "one_sided_margin": margin,
        "release": {
            "path": str(release_path),
            "released_at": released_at,
            "finite_sample_rank": release.get("finite_sample_rank"),
            "calibration_margin_frozen": release.get(
                "calibration_margin_frozen"
            ),
            "frozen_margin": frozen_margin,
            "v1_joint_units_used": int(release.get("v1_joint_units_used", -1)),
            "audit_passed": True,
        },
        "holdout": holdout,
        "holdout_started_at": holdout_started_at,
        "holdout_covered": holdout["joint_wave_score"] <= margin + base.TOLERANCE,
    }


def run_registered(
    registration: Path = DEFAULT_REGISTRATION,
    output: Path = DEFAULT_RESULT,
) -> dict[str, Any]:
    registration = Path(registration)
    output = Path(output)
    if output.exists():
        raise ClosureRefusal(f"result artifact already exists: {output}")
    protocol = base._read_object(registration)
    if (
        protocol.get("schema_version") != 1
        or protocol.get("gate") != GATE
        or protocol.get("experiment") != EXPERIMENT
        or protocol.get("registered") is not True
        or protocol.get("status") != STATUS
        or protocol.get("decision_rule") != DECISION_RULE
    ):
        raise ClosureRefusal("joint95-v5 closure registration is invalid")
    current = _current_source_snapshot(protocol)
    _validate_registered_sources(protocol, current)
    joint = _complete_v5_joint_evidence(protocol, current["manifest"])
    envelope_model, envelopes = base._planned_baseline_model(
        current["point_model"],
        protocol["predictor_anchor"],
        protocol["action_mapping"],
        margin=float(joint["one_sided_margin"]),
    )
    capacity = base._capacity_analysis(protocol, envelope_model)
    candidate = base._candidate_support_audit(protocol, envelope_model, capacity)
    decision = base._decision(
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
        "joint95_v5_audit": joint,
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
        "joint_data_accounting": {
            "v5_calibration_units": len(joint["calibration"]),
            "v5_holdout_units": 1,
            "v1_joint_units": 0,
            "v1_joint_scores": 0,
            "v2_joint_units": 0,
            "v2_joint_scores": 0,
            "v3_joint_units": 0,
            "v3_joint_scores": 0,
            "v4_joint_units": 0,
            "v4_joint_scores": 0,
        },
        "trajectory_runs": 0,
        "selector_calls": 0,
        "server_pulls": 0,
        "rng_draws": 0,
        "claim_boundary": protocol["claim_boundary"],
    }
    base._write_new(output, report)
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
    run_parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
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
