"""Freeze the corrected v3 trajectory comparison behind joint95 v4 only.

This adapter preserves the independent exact-renewal MaxWeight implementation,
the seeds 12041..12060, and the frozen 18-claim family from the corrected v3
runner.  Registration reads only v4 protocols.  Runtime requires the complete
v4 collection, v4 capacity closure, and v4 canonical static screen before any
arrival tape or selector is constructed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from algorithm.experiments import aligned_hardware_joint95_baseline_confirmation as base
from algorithm.experiments import aligned_hardware_joint95_v4_canonical_selector_screen as canonical_v4


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
STEM = "aligned_hardware_joint95_v4_baseline_confirmation_v3_20260913"
DEFAULT_RESULT = ARTIFACTS / f"{STEM}.json"
DEFAULT_PROTOCOL = ARTIFACTS / f"{STEM}_protocol.json"
DEFAULT_FIXED_SOURCES = ARTIFACTS / f"{STEM}_fixed_sources.json"
DEFAULT_CAPACITY_REGISTRATION = canonical_v4.DEFAULT_CAPACITY_REGISTRATION
DEFAULT_CAPACITY_RESULT = canonical_v4.DEFAULT_CAPACITY_RESULT
DEFAULT_CANONICAL_REGISTRATION = canonical_v4.DEFAULT_REGISTRATION
DEFAULT_CANONICAL_RESULT = canonical_v4.DEFAULT_RESULT

EXPERIMENT = "aligned_hardware_joint95_v4_baseline_confirmation_v3"
GATE = "aligned_hardware_joint95_v4_baseline_confirmation_v3_runner"
REGISTRATION_STATUS = "REGISTERED_AWAITING_JOINT95_V4_CANONICAL_SELECTOR"
PASS_DECISION = "BASELINE_CONFIRMATION_PASS"
THIS_SOURCE = (
    "algorithm/experiments/aligned_hardware_joint95_v4_baseline_confirmation_v3.py"
)
FIXED_SOURCE_PATHS = (
    "algorithm/canonical_count_space_bnb_frame_candidates.py",
    "algorithm/experiments/aligned_hardware_frame_experiment.py",
    "algorithm/experiments/aligned_hardware_frame_model.py",
    "algorithm/experiments/aligned_hardware_joint95_baseline_confirmation.py",
    "algorithm/experiments/aligned_hardware_joint95_v4_capacity_closure.py",
    "algorithm/experiments/aligned_hardware_joint95_v4_canonical_selector_screen.py",
    "algorithm/experiments/critical_gpu_joint95_v4_dispatcher.py",
    THIS_SOURCE,
)


class V4BaselineRefusal(RuntimeError):
    """The v4 upstream gate or corrected-v3 frozen contract is not satisfied."""


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V4BaselineRefusal(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise V4BaselineRefusal(f"JSON root must be an object: {path}")
    return value


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(dict(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")


def _current_fixed_sources() -> dict[str, str]:
    return {name: (ROOT / name).read_text(encoding="utf-8") for name in FIXED_SOURCE_PATHS}


def _capacity_contract(source: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return canonical_v4._capacity_contract(source)
    except canonical_v4.V4CanonicalScreenRefusal as exc:
        raise V4BaselineRefusal(str(exc)) from exc


def _canonical_contract(
    source: Mapping[str, Any], capacity_contract: Mapping[str, Any]
) -> dict[str, Any]:
    selector = source.get("selector_contract") or {}
    if (
        source.get("schema_version") != 1
        or source.get("gate") != canonical_v4.GATE
        or source.get("experiment") != canonical_v4.EXPERIMENT
        or source.get("status") != canonical_v4.REGISTRATION_STATUS
        or source.get("registered") is not True
        or source.get("capacity_registration_contract") != capacity_contract
        or selector.get("fixed_budget_by_lane") != {"8": 36, "16": 72}
        or selector.get("cold_canonical_reference_budget") != 4096
        or selector.get("registered_queue_count") != 512
        or source.get("v4_data_gate", {}).get("calibration_waves")
        != list(range(74, 93))
        or source.get("v4_data_gate", {}).get("holdout_wave") != 93
        or any(
            (source.get(f"v{version}_contribution") or {}).get(key) != 0
            for version in (1, 2, 3)
            for key in ("joint_units", "joint_scores", "capacity_rows", "holdout_rows")
        )
    ):
        raise V4BaselineRefusal("joint95 v4 canonical registration is invalid")
    return _json(source)


def _protocol(
    *,
    capacity_registration: Path,
    capacity_result: Path,
    canonical_registration: Path,
    canonical_result: Path,
    capacity_contract: Mapping[str, Any],
    canonical_contract: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "gate": GATE,
        "experiment": EXPERIMENT,
        "status": REGISTRATION_STATUS,
        "registered": True,
        "source_paths": {
            "capacity_registration": str(Path(capacity_registration).resolve()),
            "capacity_result": str(Path(capacity_result).resolve()),
            "canonical_registration": str(Path(canonical_registration).resolve()),
            "canonical_result": str(Path(canonical_result).resolve()),
            "collection_manifest": str(canonical_v4.DEFAULT_COLLECTION_MANIFEST.resolve()),
            "joint_results_directory": str(canonical_v4.DEFAULT_RESULTS_DIR.resolve()),
        },
        "capacity_registration_contract": _json(capacity_contract),
        "canonical_registration_contract": _json(canonical_contract),
        "trajectory_contract": {
            "implementation": "corrected independent exact-MW baseline v3",
            "lanes": list(base.LANES),
            "loads": list(base.LOADS),
            "arrival_seeds": list(base.ARRIVAL_SEEDS),
            "arrival_horizon_s": base.ARRIVAL_HORIZON_S,
            "fixed_budget_by_lane": {
                str(key): value for key, value in base.FIXED_BUDGET_BY_LANE.items()
            },
            "policies": list(base.POLICIES),
            "common_tape_rule": "one tape per R/load/seed replayed under all four policies",
            "case_seed_tapes": 120,
            "trajectory_runs": 480,
        },
        "statistics": {
            "primary_endpoint": base.PRIMARY_ENDPOINT,
            "familywise_alpha": base.FAMILYWISE_ALPHA,
            "primary_family_size": base.PRIMARY_FAMILY_SIZE,
            "primary_claims": base._primary_claims(),
            "exact_mw_noninferiority_margin_ratio": 1.01,
            "decision_rule": "all eighteen frozen primary claims must pass",
        },
        "v4_data_gate": {
            "calibration_waves": list(range(74, 93)),
            "holdout_wave": 93,
            "required_collection_status": "COMPLETE",
            "capacity_result_required": True,
            "canonical_result_required": True,
            "score_only_static_result_required": False,
        },
        "v1_contribution": {
            "joint_units": 0,
            "joint_scores": 0,
            "capacity_rows": 0,
            "holdout_rows": 0,
            "score_only_static_dependency": False,
        },
        "v2_contribution": {
            "joint_units": 0,
            "joint_scores": 0,
            "capacity_rows": 0,
            "holdout_rows": 0,
            "score_only_static_dependency": False,
        },
        "v3_contribution": {
            "joint_units": 0,
            "joint_scores": 0,
            "capacity_rows": 0,
            "holdout_rows": 0,
            "score_only_static_dependency": False,
        },
        "decision_rule": dict(base.DECISION_RULE),
        "claim_boundary": base.CLAIM_BOUNDARY,
        "registration_scope": {
            "v4_capacity_protocol_read": True,
            "v4_canonical_protocol_read": True,
            "v4_results_directory_read": False,
            "v4_capacity_result_read": False,
            "v4_canonical_result_read": False,
            "v1_score_only_static_protocol_read": False,
            "v1_score_only_static_result_read": False,
            "arrival_tape_constructed": False,
            "selector_invoked": False,
            "exact_mw_invoked": False,
            "rng_initializations": 0,
            "rng_draws": 0,
            "paper_edited": False,
        },
    }


def register(
    protocol_path: Path = DEFAULT_PROTOCOL,
    fixed_sources_path: Path = DEFAULT_FIXED_SOURCES,
    *,
    capacity_registration: Path = DEFAULT_CAPACITY_REGISTRATION,
    capacity_result: Path = DEFAULT_CAPACITY_RESULT,
    canonical_registration: Path = DEFAULT_CANONICAL_REGISTRATION,
    canonical_result: Path = DEFAULT_CANONICAL_RESULT,
) -> dict[str, Any]:
    if any(
        Path(path).exists()
        for path in (protocol_path, fixed_sources_path, DEFAULT_RESULT)
    ):
        raise V4BaselineRefusal("joint95 v4 baseline artifacts already exist")
    capacity_contract = _capacity_contract(_read_object(Path(capacity_registration)))
    canonical_contract = _canonical_contract(
        _read_object(Path(canonical_registration)), capacity_contract
    )
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        **_protocol(
            capacity_registration=capacity_registration,
            capacity_result=capacity_result,
            canonical_registration=canonical_registration,
            canonical_result=canonical_result,
            capacity_contract=capacity_contract,
            canonical_contract=canonical_contract,
        ),
    }
    _write_new(Path(protocol_path), protocol)
    _write_new(Path(fixed_sources_path), _current_fixed_sources())
    return protocol


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    trajectory = protocol.get("trajectory_contract") or {}
    statistics = protocol.get("statistics") or {}
    gate = protocol.get("v4_data_gate") or {}
    v1 = protocol.get("v1_contribution") or {}
    v2 = protocol.get("v2_contribution") or {}
    v3 = protocol.get("v3_contribution") or {}
    if (
        protocol.get("schema_version") != 1
        or protocol.get("gate") != GATE
        or protocol.get("experiment") != EXPERIMENT
        or protocol.get("status") != REGISTRATION_STATUS
        or protocol.get("registered") is not True
        or trajectory.get("arrival_seeds") != list(range(12041, 12061))
        or trajectory.get("case_seed_tapes") != 120
        or trajectory.get("trajectory_runs") != 480
        or statistics.get("primary_family_size") != 18
        or len(statistics.get("primary_claims") or ()) != 18
        or gate.get("score_only_static_result_required") is not False
        or any(v1.get(key) != 0 for key in ("joint_units", "joint_scores", "capacity_rows", "holdout_rows"))
        or v1.get("score_only_static_dependency") is not False
        or any(v2.get(key) != 0 for key in ("joint_units", "joint_scores", "capacity_rows", "holdout_rows"))
        or v2.get("score_only_static_dependency") is not False
        or any(v3.get(key) != 0 for key in ("joint_units", "joint_scores", "capacity_rows", "holdout_rows"))
        or v3.get("score_only_static_dependency") is not False
    ):
        raise V4BaselineRefusal("joint95 v4 baseline registration changed")


def _runtime_context(protocol: Mapping[str, Any]):
    try:
        collection = canonical_v4.require_v4_complete(protocol)
    except canonical_v4.V4CanonicalScreenRefusal as exc:
        raise V4BaselineRefusal(str(exc)) from exc
    paths = protocol["source_paths"]
    capacity_protocol = _read_object(Path(paths["capacity_registration"]))
    capacity_contract = _capacity_contract(capacity_protocol)
    if capacity_contract != protocol["capacity_registration_contract"]:
        raise V4BaselineRefusal("v4 capacity registration changed")
    canonical_protocol = _read_object(Path(paths["canonical_registration"]))
    if _canonical_contract(canonical_protocol, capacity_contract) != protocol[
        "canonical_registration_contract"
    ]:
        raise V4BaselineRefusal("v4 canonical registration changed")
    missing = [
        label
        for label in ("capacity_result", "canonical_result")
        if not Path(paths[label]).is_file()
    ]
    if missing:
        raise V4BaselineRefusal(
            "v4 prerequisite results unavailable before trajectory work: "
            + ", ".join(missing)
        )
    capacity = _read_object(Path(paths["capacity_result"]))
    canonical = _read_object(Path(paths["canonical_result"]))
    if (
        capacity.get("gate") != canonical_v4.CAPACITY_GATE
        or capacity.get("experiment") != canonical_v4.CAPACITY_EXPERIMENT
        or capacity.get("status") != "PASS"
        or capacity.get("protocol") != capacity_protocol
        or capacity.get("capacity_established") is not True
        or capacity.get("joint95_v4_audit", {}).get("holdout_covered") is not True
    ):
        raise V4BaselineRefusal("joint95 v4 capacity prerequisite has not passed")
    if (
        canonical.get("gate") != canonical_v4.GATE
        or canonical.get("experiment") != canonical_v4.EXPERIMENT
        or canonical.get("status") != "PASS"
        or canonical.get("protocol") != canonical_protocol
        or canonical.get("scientific_decision") != canonical_v4.PASS_DECISION
        or canonical.get("baseline_trajectory_advance") is not True
        or not all((canonical.get("checks") or {}).values())
    ):
        raise V4BaselineRefusal("joint95 v4 canonical prerequisite has not passed")
    capacity_model = (capacity.get("physical_complete_frame_envelope") or {}).get("model")
    canonical_model = (canonical.get("physical_complete_frame_envelope") or {}).get("model")
    if not capacity_model or capacity_model != canonical_model:
        raise V4BaselineRefusal("v4 capacity and canonical physical models differ")
    model = base._model_from_snapshot(capacity_model)
    lane_budgets = {
        int(row.get("lanes")): int(row.get("fixed_lp_budget", -1))
        for row in canonical.get("lane_summaries") or ()
    }
    if lane_budgets != base.FIXED_BUDGET_BY_LANE:
        raise V4BaselineRefusal("v4 canonical result does not certify K36/K72")
    rates: dict[tuple[int, float], tuple[float, ...]] = {}
    physical_rows = canonical.get("physical_capacity_normalized_loads") or {}
    for lanes in base.LANES:
        for row in physical_rows.get(str(lanes)) or ():
            values = tuple(float(value) for value in row.get("arrival_rates") or ())
            if len(values) == len(model.workloads) and all(value > 0.0 for value in values):
                rates[(lanes, float(row["load"]))] = values
    if set(rates) != {(lanes, load) for lanes in base.LANES for load in base.LOADS}:
        raise V4BaselineRefusal("v4 physical-capacity-normalized arrival rows are incomplete")
    return model, rates, {"collection": collection, "capacity": capacity, "canonical": canonical}


def run_registered(
    protocol_path: Path = DEFAULT_PROTOCOL,
    fixed_sources_path: Path = DEFAULT_FIXED_SOURCES,
    output: Path = DEFAULT_RESULT,
) -> dict[str, Any]:
    output = Path(output)
    if output.exists():
        raise V4BaselineRefusal(f"result already exists: {output}")
    protocol = _read_object(Path(protocol_path))
    _validate_protocol(protocol)
    if _read_object(Path(fixed_sources_path)) != _current_fixed_sources():
        raise V4BaselineRefusal("registered joint95 v4 baseline sources changed")
    model, rates, dependencies = _runtime_context(protocol)
    tasks = base._build_tasks(model, rates)
    started = time.perf_counter()
    with base.ProcessPoolExecutor(max_workers=base.EXECUTION_WORKERS) as pool:
        rows = list(pool.map(base._run_case_seed, tasks))
    claims = [
        base._paired_ratio_summary(rows, lanes, load, comparator)
        for lanes in base.LANES
        for load in base.LOADS
        for comparator in base.COMPARATORS
    ]
    checks = {
        "joint95_v4_complete": True,
        "v1_joint_contribution_is_zero": True,
        "v4_capacity_passed": True,
        "v4_canonical_selector_passed": True,
        "fresh_seed_roster_is_exact": len(tasks) == 120,
        "one_common_tape_per_case_seed": all(
            row["arrival_tape_generation_count"] == 1 for row in rows
        ),
        "all_four_policies_run_once_per_tape": all(
            set(row["policies"]) == set(base.POLICIES) for row in rows
        ),
        "all_jobs_complete": all(
            metrics["job_count"] == metrics["completed_jobs"] == row["arrival_count"]
            for row in rows
            for metrics in row["policies"].values()
        ),
        "exact_mw_exact_in_every_frame": all(
            row["policies"]["exact_renewal_mw_milp"]["exact_selection_frame_count"]
            == row["policies"]["exact_renewal_mw_milp"]["frame_count"]
            for row in rows
        ),
        "primary_family_is_complete": len(claims) == 18
        and all(row["sample_size"] == 20 for row in claims),
    }
    decision = base._decision(all(checks.values()), claims)
    report = {
        "schema_version": 1,
        "gate": GATE,
        "experiment": EXPERIMENT,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "protocol": protocol,
        "checks": checks,
        "v1_contribution": protocol["v1_contribution"],
        "v2_contribution": protocol["v2_contribution"],
        "v3_contribution": protocol["v3_contribution"],
        "v4_dependency_evidence": {
            "collection_status": dependencies["collection"]["status"],
            "capacity_decision": dependencies["capacity"].get("scientific_decision"),
            "canonical_decision": dependencies["canonical"].get("scientific_decision"),
        },
        "physical_complete_frame_envelope": {"model": _json(model.snapshot())},
        "runs": rows,
        "primary_simultaneous_claims": claims,
        "scientific_decision": decision,
        "baseline_confirmation_passed": decision == PASS_DECISION,
        "claim_boundary": base.CLAIM_BOUNDARY,
        "timing": {"wall_s": time.perf_counter() - started, "scientific_gate": False},
    }
    _write_new(output, report)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--register", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--fixed-sources", type=Path, default=DEFAULT_FIXED_SOURCES)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--capacity-registration", type=Path, default=DEFAULT_CAPACITY_REGISTRATION)
    parser.add_argument("--capacity-result", type=Path, default=DEFAULT_CAPACITY_RESULT)
    parser.add_argument("--canonical-registration", type=Path, default=DEFAULT_CANONICAL_REGISTRATION)
    parser.add_argument("--canonical-result", type=Path, default=DEFAULT_CANONICAL_RESULT)
    args = parser.parse_args(argv)
    try:
        payload = (
            register(
                args.protocol,
                args.fixed_sources,
                capacity_registration=args.capacity_registration,
                capacity_result=args.capacity_result,
                canonical_registration=args.canonical_registration,
                canonical_result=args.canonical_result,
            )
            if args.register
            else run_registered(args.protocol, args.fixed_sources, args.output)
        )
    except V4BaselineRefusal as exc:
        raise SystemExit(f"REFUSED: {exc}") from exc
    print(json.dumps({"status": payload["status"], "experiment": payload["experiment"]}, indent=2))


if __name__ == "__main__":
    main()
