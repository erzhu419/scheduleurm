"""Prospective exact-ledger confirmation scaffold for FJSP experiments.

This module is deliberately versioned and independent of the frozen v1 FJSP
artifacts.  It supports an outcome-blind instance-selection phase, a
fail-closed preregistration contract, and a later confirmation phase.  The
confirmation compares integer completion ledgers exactly.  Comparisons within
the generated candidate family are kept separate from an optional time-bounded
CP-SAT reference; neither comparison creates a global-optimality claim.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

from algorithm.experiments.fjsp_benchmark import POLICY_ORDER, build_schedule
from algorithm.experiments.fjsp_instances import (
    FJSPInstance,
    FJSPParseError,
    load_fjsp_instance,
)
from algorithm.experiments.fjsp_renewal_frame_upgrade import (
    build_fjsp_renewal_frame_schedule,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SELECTION_SCHEMA = "scheduleurm.fjsp_exact_ledger_selection.v2"
PREREGISTRATION_SCHEMA = "scheduleurm.fjsp_exact_ledger_preregistration.v2"
REPORT_SCHEMA = "scheduleurm.fjsp_exact_ledger_confirmation.v2"

DEFAULT_IMPLEMENTATION_PATHS = (
    "algorithm/experiments/fjsp_exact_ledger_confirmation_v2.py",
    "algorithm/experiments/fjsp_instances.py",
    "algorithm/experiments/fjsp_renewal_frame_upgrade.py",
    "algorithm/experiments/trajectory_renewal_frame_score.py",
    "algorithm/experiments/fjsp_trajectory_upgrade.py",
    "algorithm/experiments/fjsp_benchmark.py",
    "algorithm/experiments/or_exact_reference.py",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SELECTION_MEMBER_KEYS = {
    "relative_path",
    "family",
    "file_sha256",
    "canonical_source_sha256",
    "size_bytes",
    "job_count",
    "machine_count",
    "operation_count",
    "selection_scale",
}


class FJSPExactLedgerConfirmationError(ValueError):
    """Raised when selection, freezing, or exact confirmation fails closed."""


def select_unused_instances(
    source_root: str | Path,
    candidate_paths: Sequence[str | Path],
    *,
    used_relative_paths: Iterable[str] = (),
    used_source_sha256: Iterable[str] = (),
    per_family: int | None = None,
) -> dict[str, Any]:
    """Select a deterministic, outcome-blind subset of unused FJSP instances.

    A candidate is considered previously used when either its relative path,
    raw-file SHA-256, or parser-canonical SHA-256 is registered as used.  The
    function reads only source bytes and instance dimensions; it never invokes
    a scheduling policy or consumes a performance outcome.
    """

    root = Path(source_root).resolve()
    used_paths = {_normalized_relative_text(value) for value in used_relative_paths}
    used_hashes = {_validated_sha256(value, label="used source hash") for value in used_source_sha256}
    if per_family is not None and int(per_family) <= 0:
        raise FJSPExactLedgerConfirmationError("per_family must be positive")

    discovered: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    source_format_exclusions: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    for candidate in candidate_paths:
        path = _resolve_beneath(root, candidate)
        relative = path.relative_to(root).as_posix()
        if relative in seen_paths:
            raise FJSPExactLedgerConfirmationError(
                f"duplicate candidate path: {relative}"
            )
        seen_paths.add(relative)
        raw_hash = _file_sha256(path)
        try:
            instance = load_fjsp_instance(path)
        except FJSPParseError as exc:
            source_format_exclusions.append(
                {
                    "relative_path": relative,
                    "file_sha256": raw_hash,
                    "reason": str(exc),
                }
            )
            continue
        canonical_hash = instance.source_sha256
        if (
            relative in used_paths
            or raw_hash in used_hashes
            or canonical_hash in used_hashes
        ):
            excluded.append(
                {
                    "relative_path": relative,
                    "file_sha256": raw_hash,
                    "canonical_source_sha256": canonical_hash,
                    "reason": "previously_observed",
                }
            )
            continue
        parts = Path(relative).parts
        family = parts[0] if len(parts) > 1 else "root"
        discovered.append(
            {
                "relative_path": relative,
                "family": family,
                "file_sha256": raw_hash,
                "canonical_source_sha256": canonical_hash,
                "size_bytes": path.stat().st_size,
                "job_count": instance.job_count,
                "machine_count": instance.machine_count,
                "operation_count": instance.operation_count,
                "selection_scale": instance.operation_count * instance.machine_count,
            }
        )

    duplicate_hashes = _duplicates(
        row["canonical_source_sha256"] for row in discovered
    )
    if duplicate_hashes:
        raise FJSPExactLedgerConfirmationError(
            "unused candidate pool contains duplicate canonical instances: "
            + ", ".join(sorted(duplicate_hashes))
        )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in discovered:
        grouped.setdefault(str(row["family"]), []).append(row)
    selected: list[dict[str, Any]] = []
    for family in sorted(grouped):
        rows = sorted(grouped[family], key=_selection_sort_key)
        if per_family is None:
            selected.extend(rows)
            continue
        if len(rows) < int(per_family):
            raise FJSPExactLedgerConfirmationError(
                f"family {family!r} has {len(rows)} unused instances, "
                f"fewer than requested {int(per_family)}"
            )
        selected.extend(rows[index] for index in _stratified_indices(len(rows), int(per_family)))

    if not selected:
        raise FJSPExactLedgerConfirmationError("no unused FJSP instance was selected")
    selected.sort(key=lambda row: (str(row["family"]), _selection_sort_key(row)))
    return {
        "schema_version": SELECTION_SCHEMA,
        "selection_rule": {
            "description": (
                "Exclude registered paths and source hashes; group by first relative-path "
                "component; sort by (operation_count*machine_count, job_count, "
                "machine_count, operation_count, relative_path); then retain all unused "
                "instances or deterministic equally spaced ranks within each family."
            ),
            "per_family": int(per_family) if per_family is not None else None,
            "uses_algorithm_outcomes": False,
            "uses_cp_sat_outcomes": False,
            "source_format_exclusion_uses_outcomes": False,
        },
        "used_registry": {
            "relative_paths": sorted(used_paths),
            "source_sha256": sorted(used_hashes),
        },
        "candidate_count": len(candidate_paths),
        "unused_candidate_count": len(discovered),
        "selected_count": len(selected),
        "selected_instances": selected,
        "excluded_previously_observed": sorted(
            excluded, key=lambda row: row["relative_path"]
        ),
        "source_format_exclusions": sorted(
            source_format_exclusions, key=lambda row: row["relative_path"]
        ),
        "outcomes_observed_during_selection": False,
    }


def build_preregistration_draft(
    selection: Mapping[str, Any],
    *,
    source_identity: Mapping[str, Any],
    implementation_root: str | Path = REPO_ROOT,
    implementation_paths: Sequence[str | Path] = DEFAULT_IMPLEMENTATION_PATHS,
    registration_label: str,
    run_cp_sat_reference: bool = True,
    cp_sat_time_limit_s: float = 5.0,
    cp_sat_workers: int = 1,
) -> dict[str, Any]:
    """Build preregistration bytes without executing either policy or CP-SAT."""

    _validate_selection_plan(selection)
    if not str(registration_label).strip():
        raise FJSPExactLedgerConfirmationError("registration_label is required")
    if float(cp_sat_time_limit_s) <= 0.0 or int(cp_sat_workers) <= 0:
        raise FJSPExactLedgerConfirmationError("invalid CP-SAT reference budget")
    if not source_identity:
        raise FJSPExactLedgerConfirmationError("source_identity must not be empty")

    impl_root = Path(implementation_root).resolve()
    implementation_hashes: dict[str, str] = {}
    for value in implementation_paths:
        relative = _safe_relative_path(value)
        if relative.as_posix() in implementation_hashes:
            raise FJSPExactLedgerConfirmationError(
                f"duplicate implementation path: {relative.as_posix()}"
            )
        implementation_hashes[relative.as_posix()] = _file_sha256(impl_root / relative)
    if not implementation_hashes:
        raise FJSPExactLedgerConfirmationError("implementation freeze must not be empty")

    selection_snapshot = json.loads(json.dumps(selection, sort_keys=True))
    return {
        "schema_version": PREREGISTRATION_SCHEMA,
        "registration_label": str(registration_label),
        "freeze_before_outcomes": True,
        "source_identity": dict(source_identity),
        "selection_plan_sha256": _canonical_sha256(selection_snapshot),
        "selection": selection_snapshot,
        "implementation_freeze": {
            "hash_algorithm": "sha256",
            "files": implementation_hashes,
        },
        "protocol": {
            "policy": "scheduleurm_fjsp_renewal_frame",
            "exact_time_unit": "integer source-processing tick",
            "comparison_coordinates": [
                "makespan_ticks",
                "sum_completion_ticks",
            ],
            "same_action_disposition": "exact canonical operation-ledger fingerprint",
            "per_instance_tuning": False,
            "holdout_feedback_used_for_policy": False,
            "run_cp_sat_reference": bool(run_cp_sat_reference),
            "cp_sat_time_limit_s": float(cp_sat_time_limit_s),
            "cp_sat_workers": int(cp_sat_workers),
            "cp_sat_objective": "makespan_only",
            "cp_sat_gap_separate_from_candidate_family": True,
            "performance_is_not_protocol_validity": True,
        },
        "claim_boundary": {
            "scope": "only the preregistered source instances",
            "exact_argmax_only_over_generated_candidate_family": True,
            "cp_sat_feasible_is_not_optimality_proof": True,
            "cp_sat_makespan_optimality_does_not_prove_multiobjective_optimality": True,
            "global_fjsp_optimality_claimed": False,
            "arbitrary_family_dominance_claimed": False,
            "stochastic_stability_transfer_claimed": False,
        },
    }


def write_preregistration_draft(
    preregistration: Mapping[str, Any], output_path: str | Path
) -> dict[str, str]:
    """Write a new preregistration once; an existing path is never overwritten."""

    if preregistration.get("schema_version") != PREREGISTRATION_SCHEMA:
        raise FJSPExactLedgerConfirmationError("unexpected preregistration schema")
    target = Path(output_path)
    if target.exists():
        raise FJSPExactLedgerConfirmationError(
            f"refusing to overwrite preregistration: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_bytes(preregistration)
    target.write_bytes(payload)
    return {"path": str(target), "sha256": sha256(payload).hexdigest()}


def verify_frozen_contract(
    preregistration_path: str | Path,
    *,
    expected_preregistration_sha256: str,
    source_root: str | Path,
    implementation_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Verify preregistration, source, and implementation bytes fail closed."""

    prereg_path = Path(preregistration_path)
    expected_prereg_hash = _validated_sha256(
        expected_preregistration_sha256, label="expected preregistration hash"
    )
    actual_prereg_hash = _file_sha256(prereg_path)
    if actual_prereg_hash != expected_prereg_hash:
        raise FJSPExactLedgerConfirmationError("preregistration hash mismatch")
    preregistration = _load_json(prereg_path)
    if preregistration.get("schema_version") != PREREGISTRATION_SCHEMA:
        raise FJSPExactLedgerConfirmationError("unexpected preregistration schema")
    if preregistration.get("freeze_before_outcomes") is not True:
        raise FJSPExactLedgerConfirmationError("preregistration is not outcome-frozen")

    selection = preregistration.get("selection")
    if not isinstance(selection, Mapping):
        raise FJSPExactLedgerConfirmationError("missing selection plan")
    _validate_selection_plan(selection)
    if preregistration.get("selection_plan_sha256") != _canonical_sha256(selection):
        raise FJSPExactLedgerConfirmationError("selection-plan hash mismatch")

    root = Path(source_root).resolve()
    verified_instances: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for registered in selection["selected_instances"]:
        relative = _safe_relative_path(registered["relative_path"])
        source = root / relative
        actual_raw_hash = _file_sha256(source)
        if actual_raw_hash != registered["file_sha256"]:
            raise FJSPExactLedgerConfirmationError(
                f"source hash mismatch: {relative.as_posix()}"
            )
        instance = load_fjsp_instance(source)
        if instance.source_sha256 != registered["canonical_source_sha256"]:
            raise FJSPExactLedgerConfirmationError(
                f"canonical source hash mismatch: {relative.as_posix()}"
            )
        shape = {
            "job_count": instance.job_count,
            "machine_count": instance.machine_count,
            "operation_count": instance.operation_count,
            "selection_scale": instance.operation_count * instance.machine_count,
        }
        if shape != {key: int(registered[key]) for key in shape}:
            raise FJSPExactLedgerConfirmationError(
                f"registered instance shape mismatch: {relative.as_posix()}"
            )
        if actual_raw_hash in seen_hashes:
            raise FJSPExactLedgerConfirmationError("duplicate selected source bytes")
        seen_hashes.add(actual_raw_hash)
        verified_instances.append(
            {
                "relative_path": relative.as_posix(),
                "family": str(registered["family"]),
                "source": source,
                "instance": instance,
                "file_sha256": actual_raw_hash,
            }
        )

    freeze = preregistration.get("implementation_freeze") or {}
    if freeze.get("hash_algorithm") != "sha256":
        raise FJSPExactLedgerConfirmationError("unsupported implementation hash")
    implementation_files = freeze.get("files")
    if not isinstance(implementation_files, Mapping) or not implementation_files:
        raise FJSPExactLedgerConfirmationError("empty implementation freeze")
    impl_root = Path(implementation_root).resolve()
    verified_implementation = []
    for value, expected in sorted(implementation_files.items()):
        relative = _safe_relative_path(value)
        expected_hash = _validated_sha256(expected, label="implementation hash")
        actual_hash = _file_sha256(impl_root / relative)
        if actual_hash != expected_hash:
            raise FJSPExactLedgerConfirmationError(
                f"implementation hash mismatch: {relative.as_posix()}"
            )
        verified_implementation.append(
            {"path": relative.as_posix(), "sha256": actual_hash}
        )

    return {
        "ready": True,
        "preregistration": preregistration,
        "preregistration_sha256": actual_prereg_hash,
        "verified_instances": verified_instances,
        "verified_implementation": verified_implementation,
    }


def exact_completion_ledger(
    instance: FJSPInstance, schedule: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Return a fully integer schedule and completion ledger.

    This verifier intentionally rejects fractional timestamps instead of using
    a floating-point tolerance.  Public FJSPLIB processing times are integer
    ticks, so exact feasibility and exact Pareto comparisons are available.
    """

    rows: dict[tuple[int, int], dict[str, int]] = {}
    for source_row in schedule:
        job_id = _exact_int(source_row.get("job_id"), label="job_id")
        operation_id = _exact_int(
            source_row.get("operation_id"), label="operation_id"
        )
        machine_id = _exact_int(source_row.get("machine_id"), label="machine_id")
        start = _exact_int(source_row.get("start_time"), label="start_time")
        end = _exact_int(source_row.get("end_time"), label="end_time")
        key = (job_id, operation_id)
        if key in rows:
            raise FJSPExactLedgerConfirmationError(f"duplicate operation: {key}")
        try:
            operation = instance.operation(job_id, operation_id)
        except KeyError as exc:
            raise FJSPExactLedgerConfirmationError(
                f"unknown operation: {key}"
            ) from exc
        option = operation.option_for(machine_id)
        if option is None:
            raise FJSPExactLedgerConfirmationError(
                f"ineligible machine {machine_id} for operation {key}"
            )
        if start < 0 or end - start != option.processing_time:
            raise FJSPExactLedgerConfirmationError(
                f"nonexact processing interval for operation {key}"
            )
        rows[key] = {
            "job_id": job_id,
            "operation_id": operation_id,
            "machine_id": machine_id,
            "start_tick": start,
            "end_tick": end,
        }

    expected = {
        (operation.job_id, operation.operation_id)
        for job in instance.jobs
        for operation in job.operations
    }
    if set(rows) != expected:
        missing = sorted(expected - set(rows))
        extra = sorted(set(rows) - expected)
        raise FJSPExactLedgerConfirmationError(
            f"operation ledger mismatch; missing={missing!r}, extra={extra!r}"
        )

    for job in instance.jobs:
        for operation_id in range(1, len(job.operations)):
            previous = rows[(job.job_id, operation_id - 1)]
            current = rows[(job.job_id, operation_id)]
            if current["start_tick"] < previous["end_tick"]:
                raise FJSPExactLedgerConfirmationError(
                    f"precedence violation in job {job.job_id}"
                )
    for machine_id in range(instance.machine_count):
        machine_rows = sorted(
            (row for row in rows.values() if row["machine_id"] == machine_id),
            key=lambda row: (
                row["start_tick"],
                row["end_tick"],
                row["job_id"],
                row["operation_id"],
            ),
        )
        for previous, current in zip(machine_rows, machine_rows[1:]):
            if current["start_tick"] < previous["end_tick"]:
                raise FJSPExactLedgerConfirmationError(
                    f"machine overlap on machine {machine_id}"
                )

    operation_ledger = [rows[key] for key in sorted(rows)]
    completion = [
        rows[(job.job_id, len(job.operations) - 1)]["end_tick"]
        for job in instance.jobs
    ]
    makespan = max(completion)
    sum_completion = sum(completion)
    action_fingerprint = _canonical_sha256(operation_ledger)
    return {
        "time_unit": "integer source-processing tick",
        "operation_ledger": operation_ledger,
        "job_completion_ticks": completion,
        "makespan_ticks": makespan,
        "sum_completion_ticks": sum_completion,
        "action_fingerprint": action_fingerprint,
        "completion_ledger_fingerprint": _canonical_sha256(completion),
    }


def exact_ledger_relation(
    ours: Mapping[str, Any], comparator: Mapping[str, Any]
) -> str:
    """Classify an exact two-coordinate cost relation without tolerance."""

    if ours["action_fingerprint"] == comparator["action_fingerprint"]:
        return "same_action"
    ours_cost = (int(ours["makespan_ticks"]), int(ours["sum_completion_ticks"]))
    other_cost = (
        int(comparator["makespan_ticks"]),
        int(comparator["sum_completion_ticks"]),
    )
    if ours_cost == other_cost:
        return "metric_tie_distinct_action"
    if all(left <= right for left, right in zip(ours_cost, other_cost)):
        return "ours_dominates"
    if all(right <= left for left, right in zip(ours_cost, other_cost)):
        return "comparator_dominates"
    return "tradeoff"


def run_exact_ledger_confirmation(
    preregistration_path: str | Path,
    *,
    expected_preregistration_sha256: str,
    source_root: str | Path,
    implementation_root: str | Path = REPO_ROOT,
    cp_sat_runner: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the registered selector and produce separate exact comparison blocks."""

    contract = verify_frozen_contract(
        preregistration_path,
        expected_preregistration_sha256=expected_preregistration_sha256,
        source_root=source_root,
        implementation_root=implementation_root,
    )
    protocol = contract["preregistration"]["protocol"]
    rows = []
    for registered in contract["verified_instances"]:
        instance = registered["instance"]
        ours_run = build_fjsp_renewal_frame_schedule(instance)
        ours_ledger = exact_completion_ledger(instance, ours_run["schedule"])

        baseline_ledgers: dict[str, Any] = {}
        baseline_relations: dict[str, str] = {}
        for policy in POLICY_ORDER:
            baseline_run = build_schedule(instance, policy)
            ledger = exact_completion_ledger(instance, baseline_run["schedule"])
            baseline_ledgers[policy] = ledger
            baseline_relations[policy] = exact_ledger_relation(ours_ledger, ledger)

        candidate_family = {
            "scope": "registered generated family including fixed-policy seeds",
            "unique_candidate_count": int(
                ours_run["selection"]["unique_candidate_count"]
            ),
            "selector_exact_generated_family_argmax": bool(
                ours_run["selection"]["exact_generated_family_argmax"]
            ),
            "selector_generated_family_oracle_gap": ours_run["selection"][
                "generated_family_oracle_gap"
            ],
            "selected_exact_ledger": ours_ledger,
            "fixed_policy_exact_ledgers": baseline_ledgers,
            "fixed_policy_exact_relations": baseline_relations,
            "same_action_policies": sorted(
                policy
                for policy, relation in baseline_relations.items()
                if relation == "same_action"
            ),
            "fixed_policy_dominators": sorted(
                policy
                for policy, relation in baseline_relations.items()
                if relation == "comparator_dominates"
            ),
            "exact_pareto_nondominated_by_fixed_policy_seeds": not any(
                relation == "comparator_dominates"
                for relation in baseline_relations.values()
            ),
            "global_action_space_exact": False,
        }
        cp_sat_gap = _cp_sat_gap_block(
            instance,
            ours_run,
            ours_ledger,
            protocol=protocol,
            cp_sat_runner=cp_sat_runner,
        )
        rows.append(
            {
                "relative_path": registered["relative_path"],
                "family": registered["family"],
                "source_sha256": registered["file_sha256"],
                "candidate_family_confirmation": candidate_family,
                "cp_sat_reference_gap": cp_sat_gap,
            }
        )

    protocol_ready = bool(
        rows
        and all(
            row["candidate_family_confirmation"][
                "selector_exact_generated_family_argmax"
            ]
            for row in rows
        )
    )
    cp_sat_coverage = sum(
        row["cp_sat_reference_gap"]["reference_feasible"] for row in rows
    )
    return {
        "schema_version": REPORT_SCHEMA,
        "source_contract": {
            "preregistration_sha256": contract["preregistration_sha256"],
            "source_hashes_verified": True,
            "implementation_hashes_verified": True,
            "freeze_before_outcomes_verified": True,
        },
        "rows": rows,
        "candidate_family_aggregate": {
            "instance_count": len(rows),
            "exact_nondominated_count": sum(
                row["candidate_family_confirmation"][
                    "exact_pareto_nondominated_by_fixed_policy_seeds"
                ]
                for row in rows
            ),
            "same_action_disposition_count": sum(
                len(row["candidate_family_confirmation"]["same_action_policies"])
                for row in rows
            ),
            "comparison_uses_integer_ledgers": True,
        },
        "cp_sat_reference_aggregate": {
            "instance_count": len(rows),
            "feasible_reference_count": cp_sat_coverage,
            "optimality_proved_count": sum(
                row["cp_sat_reference_gap"]["reference_optimality_proved"]
                for row in rows
            ),
            "kept_separate_from_candidate_family_gate": True,
        },
        "gate": {
            "pass": protocol_ready,
            "status": (
                "FJSP_EXACT_LEDGER_CONFIRMATION_V2_PASS"
                if protocol_ready
                else "FJSP_EXACT_LEDGER_CONFIRMATION_V2_FAIL"
            ),
            "protocol_validity_independent_of_performance": True,
            "protocol_validity_independent_of_cp_sat_coverage": True,
            "global_fjsp_optimality_claim_ready": False,
        },
        "claim_boundary": {
            "scope": "only preregistered instances and the frozen generated family",
            "candidate_family_and_cp_sat_reference_are_distinct_evidence": True,
            "cp_sat_feasible_without_optimal_status_is_not_an_optimality_proof": True,
            "cp_sat_makespan_optimality_is_not_multiobjective_optimality": True,
            "global_fjsp_optimality_claimed": False,
        },
    }


def _cp_sat_gap_block(
    instance: FJSPInstance,
    ours_run: Mapping[str, Any],
    ours_ledger: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    cp_sat_runner: Callable[..., Mapping[str, Any]] | None,
) -> dict[str, Any]:
    if not bool(protocol["run_cp_sat_reference"]):
        return {
            "status": "NOT_RUN_BY_PREREGISTRATION",
            "reference_feasible": False,
            "reference_optimality_proved": False,
            "reference_outside_generated_candidate_family": True,
            "candidate_family_minus_reference": None,
            "global_optimality_claimed": False,
        }
    runner = cp_sat_runner
    if runner is None:
        from algorithm.experiments.or_exact_reference import solve_fjsp_cp_sat

        runner = solve_fjsp_cp_sat
    reference = runner(
        instance,
        time_limit_s=float(protocol["cp_sat_time_limit_s"]),
        workers=int(protocol["cp_sat_workers"]),
        incumbent_schedule=ours_run["schedule"],
    )
    feasible = bool(reference.get("feasible_solution"))
    if not feasible:
        return {
            "status": str(reference.get("status", "UNKNOWN")),
            "reference_feasible": False,
            "reference_optimality_proved": False,
            "reference_outside_generated_candidate_family": True,
            "candidate_family_minus_reference": None,
            "global_optimality_claimed": False,
        }
    verifier = reference.get("verifier")
    if isinstance(verifier, Mapping) and verifier.get("pass") is not True:
        raise FJSPExactLedgerConfirmationError("CP-SAT reference failed feasibility audit")
    reference_ledger = exact_completion_ledger(instance, reference["schedule"])
    return {
        "status": str(reference.get("status", "UNKNOWN")),
        "reference_feasible": True,
        "reference_optimality_proved": bool(reference.get("optimality_proved")),
        "reference_objective": "makespan_only",
        "reference_outside_generated_candidate_family": True,
        "reference_exact_ledger": reference_ledger,
        "exact_relation": exact_ledger_relation(ours_ledger, reference_ledger),
        "candidate_family_minus_reference": {
            "makespan_ticks": int(ours_ledger["makespan_ticks"])
            - int(reference_ledger["makespan_ticks"]),
            "sum_completion_ticks": int(ours_ledger["sum_completion_ticks"])
            - int(reference_ledger["sum_completion_ticks"]),
        },
        "global_optimality_claimed": False,
    }


def _validate_selection_plan(selection: Mapping[str, Any]) -> None:
    if selection.get("schema_version") != SELECTION_SCHEMA:
        raise FJSPExactLedgerConfirmationError("unexpected selection schema")
    rule = selection.get("selection_rule") or {}
    if rule.get("uses_algorithm_outcomes") is not False:
        raise FJSPExactLedgerConfirmationError("selection consumed algorithm outcomes")
    if rule.get("uses_cp_sat_outcomes") is not False:
        raise FJSPExactLedgerConfirmationError("selection consumed CP-SAT outcomes")
    if rule.get("source_format_exclusion_uses_outcomes") is not False:
        raise FJSPExactLedgerConfirmationError(
            "source-format exclusion consumed outcomes"
        )
    if selection.get("outcomes_observed_during_selection") is not False:
        raise FJSPExactLedgerConfirmationError("selection is not outcome blind")
    members = selection.get("selected_instances")
    if not isinstance(members, list) or not members:
        raise FJSPExactLedgerConfirmationError("selection must contain instances")
    if int(selection.get("selected_count", -1)) != len(members):
        raise FJSPExactLedgerConfirmationError("selected_count mismatch")
    paths = []
    hashes = []
    for row in members:
        if not isinstance(row, Mapping) or set(row) != _SELECTION_MEMBER_KEYS:
            raise FJSPExactLedgerConfirmationError(
                "selected member contains an outcome or unexpected field"
            )
        relative = _safe_relative_path(row["relative_path"]).as_posix()
        if not str(row["family"]):
            raise FJSPExactLedgerConfirmationError("empty family label")
        paths.append(relative)
        hashes.append(_validated_sha256(row["file_sha256"], label="source hash"))
        _validated_sha256(
            row["canonical_source_sha256"], label="canonical source hash"
        )
        for key in (
            "size_bytes",
            "job_count",
            "machine_count",
            "operation_count",
            "selection_scale",
        ):
            if int(row[key]) <= 0:
                raise FJSPExactLedgerConfirmationError(f"invalid selected {key}")
    if len(paths) != len(set(paths)) or len(hashes) != len(set(hashes)):
        raise FJSPExactLedgerConfirmationError("selected instances are not unique")
    exclusions = selection.get("source_format_exclusions")
    if not isinstance(exclusions, list):
        raise FJSPExactLedgerConfirmationError("missing source-format exclusion ledger")
    for row in exclusions:
        if not isinstance(row, Mapping) or set(row) != {
            "relative_path",
            "file_sha256",
            "reason",
        }:
            raise FJSPExactLedgerConfirmationError(
                "invalid source-format exclusion row"
            )
        _safe_relative_path(row["relative_path"])
        _validated_sha256(row["file_sha256"], label="excluded source hash")
        if not str(row["reason"]).strip():
            raise FJSPExactLedgerConfirmationError("empty source-format reason")


def _selection_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["selection_scale"]),
        int(row["job_count"]),
        int(row["machine_count"]),
        int(row["operation_count"]),
        str(row["relative_path"]),
    )


def _stratified_indices(count: int, selected_count: int) -> tuple[int, ...]:
    if selected_count > count:
        raise FJSPExactLedgerConfirmationError("stratified selection exceeds pool")
    if selected_count == 1:
        return (count // 2,)
    return tuple(
        (rank * (count - 1)) // (selected_count - 1)
        for rank in range(selected_count)
    )


def _duplicates(values: Iterable[str]) -> set[str]:
    seen: set[str] = set()
    repeated: set[str] = set()
    for value in values:
        if value in seen:
            repeated.add(value)
        seen.add(value)
    return repeated


def _exact_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or value is None:
        raise FJSPExactLedgerConfirmationError(f"{label} must be an integer tick")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise FJSPExactLedgerConfirmationError(
                f"{label} is fractional: {value!r}"
            )
        return int(value)
    text = str(value).strip()
    if not re.fullmatch(r"[+-]?\d+", text):
        raise FJSPExactLedgerConfirmationError(
            f"{label} is not an exact integer: {value!r}"
        )
    return int(text)


def _safe_relative_path(value: str | Path) -> Path:
    path = Path(str(value))
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise FJSPExactLedgerConfirmationError(f"unsafe relative path: {value}")
    return path


def _normalized_relative_text(value: str | Path) -> str:
    return _safe_relative_path(value).as_posix()


def _resolve_beneath(root: Path, value: str | Path) -> Path:
    candidate = Path(value)
    path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise FJSPExactLedgerConfirmationError(
            f"candidate lies outside source root: {value}"
        ) from exc
    if not path.is_file():
        raise FJSPExactLedgerConfirmationError(f"candidate is not a file: {value}")
    return path


def _validated_sha256(value: Any, *, label: str) -> str:
    text = str(value).lower()
    if not _SHA256_RE.fullmatch(text):
        raise FJSPExactLedgerConfirmationError(f"invalid {label}: {value!r}")
    return text


def _file_sha256(path: str | Path) -> str:
    source = Path(path)
    if not source.is_file():
        raise FJSPExactLedgerConfirmationError(f"missing frozen file: {source}")
    return sha256(source.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FJSPExactLedgerConfirmationError(
            f"cannot read preregistration: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise FJSPExactLedgerConfirmationError("preregistration must be a JSON object")
    return value


__all__ = [
    "DEFAULT_IMPLEMENTATION_PATHS",
    "FJSPExactLedgerConfirmationError",
    "build_preregistration_draft",
    "exact_completion_ledger",
    "exact_ledger_relation",
    "run_exact_ledger_confirmation",
    "select_unused_instances",
    "verify_frozen_contract",
    "write_preregistration_draft",
]
