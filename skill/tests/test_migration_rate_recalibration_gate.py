from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path

import pytest

from algorithm.experiments.migration_rate_recalibration_gate import (
    CNN_COST,
    DEFAULT_SPECS,
    ExactRateState,
    MigrationRateSpec,
    PROGRESS_POINTS,
    build_migration_rate_recalibration_gate,
    markdown_report,
    write_outputs,
)
from algorithm.theorem_dispatch.migration import MigrationCost
from simulation.service_cache import ProfileRecord, ServiceRateCache


def _profile(
    state: ExactRateState,
    *,
    per_task_rate: float,
    total_units: float,
    stable: bool = True,
    completion: bool = True,
) -> ProfileRecord:
    rates = tuple(per_task_rate for _ in range(state.colocation_count))
    return ProfileRecord(
        workload_key=state.workload_key,
        command_fingerprint="synthetic-test",
        resource_kind="cpu" if state.node.startswith("node") else "gpu",
        node_bucket=state.node_bucket,
        profile=state.profile,
        unit="step" if not state.workload_key.startswith("hybrid_rl") else "iter",
        total_units=total_units,
        aggregate_rate=sum(rates),
        per_task_rates=rates,
        source=f"synthetic://{state.node}/{state.resource_state}",
        capacity_boundary=False,
        gpu_count_observed=1,
        allocation_workers=state.allocation_workers,
        colocation_count=state.colocation_count,
        workload_env=state.workload_env,
        resource_state=state.resource_state,
        resident_mix=state.resident_mix,
        eta_source="task_native_tqdm_natural_completion_split_conformal_lcb",
        stable_rate_ready=stable,
        hardware_class=(
            "cpu_hpc_192c" if state.node.startswith("node") else "synthetic_gpu"
        ),
        completion_model_ready=completion,
        completion_model_sample_count=13 if completion else 0,
        startup_overhead_s=1.0,
        completion_unit_s=1.0 / per_task_rate,
        finalization_overhead_s=0.5,
        completion_total_wall_s=1.5 + total_units / per_task_rate,
        completion_group_total_units=total_units * state.colocation_count,
    )


def _complete_cache(
    path: Path,
    *,
    omit_migration_id: str = "",
    override: dict[tuple[str, str], dict[str, object]] | None = None,
) -> None:
    rates = {
        "cnn_jtl110gpu_to_jtl311linux": (2.0, 4.0, 100.0),
        "resac_ant_jtl110gpu_to_jtl311linux": (0.10, 0.20, 40.0),
        "cpu_node003_half_to_node005_empty": (1.0, 2.0, 160.0),
        "cpu_node003_full_to_node005_empty": (0.5, 2.0, 160.0),
    }
    records = []
    seen = set()
    for spec in DEFAULT_SPECS:
        if spec.migration_id == omit_migration_id:
            continue
        old_rate, new_rate, total = rates[spec.migration_id]
        for role, state, rate in (
            ("source", spec.source, old_rate),
            ("target", spec.target, new_rate),
        ):
            identity = (
                state.workload_key,
                state.workload_env,
                state.node_bucket,
                state.resource_state,
                state.profile,
                state.allocation_workers,
                state.colocation_count,
                state.resident_mix,
            )
            if identity in seen:
                continue
            seen.add(identity)
            changes = (override or {}).get((spec.migration_id, role), {})
            records.append(
                _profile(
                    state,
                    per_task_rate=float(changes.get("per_task_rate", rate)),
                    total_units=float(changes.get("total_units", total)),
                    stable=bool(changes.get("stable", True)),
                    completion=bool(changes.get("completion", True)),
                )
            )
    ServiceRateCache(records).save(path)


def _cost_row(signature, point: float, *, workload_key: str) -> dict[str, object]:
    cost = MigrationCost(
        checkpoint_flush_s=1.0,
        sync_s=2.0,
        environment_staging_s=0.5,
        resume_warmup_s=0.5,
        lost_work_s=0.25,
        risk_penalty_units=0.25,
    )
    sha = "a" * 64
    payload_bytes = signature.checkpoint_mib * 1024 * 1024
    return {
        **signature.snapshot(),
        "spec_id": f"cost_{signature.checkpoint_policy}",
        "family": "synthetic",
        "workload_key": workload_key,
        "progress_fraction": point,
        "launched": True,
        "measurement_valid": True,
        "status": "MEASURED",
        "checkpoint_meta": {
            "checkpoint_mib": signature.checkpoint_mib,
            "payload_bytes": payload_bytes,
            "sha256": sha,
            "checkpoint_flush_s": cost.checkpoint_flush_s,
        },
        "resume_meta": {
            "payload_bytes": payload_bytes,
            "sha256": sha,
            "expected_sha256": sha,
            "hash_ok": True,
            "resume_warmup_s": cost.resume_warmup_s,
        },
        "checkpoint_flush_s": cost.checkpoint_flush_s,
        "sync_s": cost.sync_s,
        "environment_staging_s": cost.environment_staging_s,
        "resume_warmup_s": cost.resume_warmup_s,
        "lost_work_s": cost.lost_work_s,
        "risk_penalty_units": cost.risk_penalty_units,
        "total_time_s": cost.total_time_s,
        "total_penalty_units": cost.total_penalty_units,
        # These are deliberately stale and must never drive the new decision.
        "remaining_work": 999999.0,
        "current_rate": 999999.0,
        "target_rate": 0.000001,
        "beneficial_by_threshold": False,
        "theorem_ready": False,
    }


def _cost_artifact(
    path: Path,
    *,
    signatures=None,
    corrupt_total: bool = False,
) -> None:
    signatures = tuple(signatures or {spec.cost_signature for spec in DEFAULT_SPECS})
    workload_for_policy = {
        "controlled_cnn_state_payload": "gpu_cnn_torch_resnet50",
        # The old physical payload was named HalfCheetah. Only K is reusable.
        "controlled_resac_cycle_state_payload": "hybrid_rl_resac_halfcheetah",
        "controlled_cpu_state_payload": "freqduet_cpu_surrogate",
    }
    rows = [
        _cost_row(
            signature,
            point,
            workload_key=workload_for_policy[signature.checkpoint_policy],
        )
        for signature in signatures
        for point in PROGRESS_POINTS
    ]
    if corrupt_total:
        rows[0]["total_penalty_units"] = 12345.0
    payload = {
        "gate": "live_checkpoint_migration_cost_gate",
        "status": "LIVE_CHECKPOINT_MIGRATION_COST_READY",
        "pass": True,
        "pending_count": 0,
        "blocked_count": 0,
        "measured_count": len(rows),
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_missing_final_cache_waits_without_manufacturing_rows(tmp_path: Path) -> None:
    report = build_migration_rate_recalibration_gate(
        final_cache_path=tmp_path / "missing.json",
        cost_artifact_paths=(),
    )
    assert report["status"] == "WAIT_FINAL_UNIFIED_CACHE"
    assert report["pass"] is False
    assert report["rows"] == []


def test_recalculates_all_points_from_exact_cache_and_ignores_old_flags(tmp_path: Path) -> None:
    cache = tmp_path / "final.json"
    costs = tmp_path / "costs.json"
    _complete_cache(cache)
    _cost_artifact(costs)

    report = build_migration_rate_recalibration_gate(
        final_cache_path=cache,
        cost_artifact_paths=(costs,),
    )

    assert report["status"] == "PASS"
    assert report["pass"] is True
    assert report["ready_row_count"] == 12
    assert report["old_cost_artifact_rates_used"] is False
    assert report["old_cost_artifact_beneficial_flags_inherited"] is False
    cnn_25 = next(
        row
        for row in report["rows"]
        if row["migration_id"] == "cnn_jtl110gpu_to_jtl311linux"
        and row["progress_fraction"] == 0.25
    )
    assert cnn_25["current_lower_service_units_per_s"] == 2.0
    assert cnn_25["target_lower_service_units_per_s"] == 4.0
    assert cnn_25["remaining_work_units"] == 75.0
    assert cnn_25["keep_remaining_completion_s"] == 37.5
    assert cnn_25["migrate_remaining_completion_s"] == pytest.approx(23.25)
    assert cnn_25["migration_net_saving_s"] == pytest.approx(14.25)
    assert cnn_25["beneficial_by_recalculated_threshold"] is True
    assert cnn_25["stale_cost_artifact_fields_audit"]["old_beneficial_by_threshold"] is False
    assert cnn_25["stale_cost_artifact_fields_audit"]["inherited"] is False


def test_halfcheetah_rate_row_cannot_substitute_for_registered_ant(tmp_path: Path) -> None:
    cache = tmp_path / "final.json"
    costs = tmp_path / "costs.json"
    _complete_cache(cache, omit_migration_id="resac_ant_jtl110gpu_to_jtl311linux")
    loaded = ServiceRateCache.load(cache)
    rl_spec = next(spec for spec in DEFAULT_SPECS if spec.family == "hybrid_rl")
    for state, rate in ((rl_spec.source, 0.1), (rl_spec.target, 0.2)):
        loaded.add(
            _profile(
                replace(
                    state,
                    workload_key="hybrid_rl_resac_halfcheetah",
                    workload_env="HalfCheetah-v2",
                ),
                per_task_rate=rate,
                total_units=40.0,
            )
        )
    loaded.save(cache)
    _cost_artifact(costs)

    report = build_migration_rate_recalibration_gate(
        final_cache_path=cache,
        cost_artifact_paths=(costs,),
        specs=(rl_spec,),
    )

    assert report["status"] == "BLOCKED"
    assert {row["block_code"] for row in report["rows"]} == {
        "EXACT_STATE_RATE_MISSING"
    }
    assert all(row["workload_env"] == "Ant-v2" for row in report["rows"])


def test_wrong_load_state_does_not_fall_back_to_legacy_or_neighbor(tmp_path: Path) -> None:
    cache = tmp_path / "final.json"
    costs = tmp_path / "costs.json"
    cpu_half = next(
        spec for spec in DEFAULT_SPECS if spec.migration_id.endswith("half_to_node005_empty")
    )
    wrong_state = replace(
        cpu_half.source,
        resource_state="unspecified",
        resident_mix="",
    )
    records = [
        _profile(wrong_state, per_task_rate=1.0, total_units=160.0),
        _profile(cpu_half.target, per_task_rate=2.0, total_units=160.0),
    ]
    ServiceRateCache(records).save(cache)
    _cost_artifact(costs, signatures=(cpu_half.cost_signature,))

    report = build_migration_rate_recalibration_gate(
        final_cache_path=cache,
        cost_artifact_paths=(costs,),
        specs=(cpu_half,),
    )

    assert report["pass"] is False
    assert {row["block_code"] for row in report["rows"]} == {
        "EXACT_STATE_RATE_MISSING"
    }


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"stable": False}, "STABLE_RATE_CERTIFICATE_MISSING"),
        ({"completion": False}, "COMPLETION_MODEL_CERTIFICATE_MISSING"),
    ],
)
def test_rate_row_requires_stable_and_completion_certificates(
    tmp_path: Path,
    changes: dict[str, object],
    expected: str,
) -> None:
    cache = tmp_path / "final.json"
    costs = tmp_path / "costs.json"
    cnn = DEFAULT_SPECS[0]
    _complete_cache(cache, override={(cnn.migration_id, "source"): changes})
    _cost_artifact(costs, signatures=(cnn.cost_signature,))

    report = build_migration_rate_recalibration_gate(
        final_cache_path=cache,
        cost_artifact_paths=(costs,),
        specs=(cnn,),
    )

    assert {row["block_code"] for row in report["rows"]} == {expected}


def test_cost_signature_mismatch_blocks_instead_of_borrowing_k(tmp_path: Path) -> None:
    cache = tmp_path / "final.json"
    costs = tmp_path / "costs.json"
    cnn = DEFAULT_SPECS[0]
    _complete_cache(cache)
    wrong_signature = replace(CNN_COST, checkpoint_mib=95)
    _cost_artifact(costs, signatures=(wrong_signature,))

    report = build_migration_rate_recalibration_gate(
        final_cache_path=cache,
        cost_artifact_paths=(costs,),
        specs=(cnn,),
    )

    assert report["status"] == "BLOCKED"
    assert {row["block_code"] for row in report["rows"]} == {
        "EXACT_COST_SIGNATURE_MISSING"
    }


def test_internally_inconsistent_cost_artifact_fails_closed(tmp_path: Path) -> None:
    cache = tmp_path / "final.json"
    costs = tmp_path / "costs.json"
    _complete_cache(cache)
    _cost_artifact(costs, corrupt_total=True)

    report = build_migration_rate_recalibration_gate(
        final_cache_path=cache,
        cost_artifact_paths=(costs,),
    )

    assert report["status"] == "FAIL_INVALID_MIGRATION_COST_ARTIFACT"
    assert report["rows"] == []
    assert report["validation_errors"][0]["code"] == "COST_ROW_INVALID"


def test_ordinary_user_task_is_always_excluded(tmp_path: Path) -> None:
    cache = tmp_path / "final.json"
    costs = tmp_path / "costs.json"
    _complete_cache(cache)
    _cost_artifact(costs)
    ordinary = replace(
        DEFAULT_SPECS[0],
        migration_id="ordinary_user_cnn",
        controlled_benchmark=False,
        ordinary_user_task=True,
    )

    report = build_migration_rate_recalibration_gate(
        final_cache_path=cache,
        cost_artifact_paths=(costs,),
        specs=(ordinary,),
    )

    assert report["ordinary_user_tasks_excluded"] is True
    assert {row["block_code"] for row in report["rows"]} == {
        "ORDINARY_USER_TASK_EXCLUDED"
    }
    assert all(row["migration_action_theorem_ready"] is False for row in report["rows"])


def test_json_and_markdown_outputs_are_deterministic_and_auditable(tmp_path: Path) -> None:
    cache = tmp_path / "final.json"
    costs = tmp_path / "costs.json"
    json_output = tmp_path / "gate.json"
    markdown_output = tmp_path / "gate.md"
    _complete_cache(cache)
    _cost_artifact(costs)
    report = build_migration_rate_recalibration_gate(
        final_cache_path=cache,
        cost_artifact_paths=(costs,),
    )

    write_outputs(report, json_output=json_output, markdown_output=markdown_output)

    assert json.loads(json_output.read_text(encoding="utf-8"))["status"] == "PASS"
    markdown = markdown_output.read_text(encoding="utf-8")
    assert markdown == markdown_report(report)
    assert "Old rates/flags inherited: `false`" in markdown
    assert "cpu_node003_full_to_node005_empty" in markdown
    assert not math.isnan(report["rows"][0]["migration_net_saving_s"])
