from __future__ import annotations

import hashlib
import json
from pathlib import Path

from algorithm.experiments.unified_hardware_or_replay import (
    ABLATION_POLICIES,
    EXPECTED_ARRIVAL_FAMILIES,
    EXPECTED_MIGRATION_MODES,
    EXPECTED_QUADRANTS,
    LEGACY_POLICY,
    OURS_POLICY,
    build_unified_hardware_or_replay,
)
from simulation.service_cache import ProfileRecord, ServiceRateCache
from simulation.sota_baselines import sota_baseline_specs


ETA_SOURCE = "task_native_tqdm_natural_completion_synthetic_fixture"


def test_valid_inputs_build_complete_fail_closed_replay_matrix(tmp_path):
    paths = _write_fixture(tmp_path)

    report = build_unified_hardware_or_replay(
        cache_path=paths["cache"],
        loaded_ledger_path=paths["ledger"],
        migration_certificate_path=paths["migration"],
        loads=(0.50,),
        seeds=(17,),
        jobs_per_workload=3,
    )

    assert report["status"] == "PASS"
    assert report["pass"] is True
    assert report["schema_version"] == 2
    assert all(report["coverage_checks"].values())
    assert {row["quadrant"] for row in report["scenarios"]} == set(EXPECTED_QUADRANTS)
    assert {row["arrival_family"] for row in report["runs"]} == set(EXPECTED_ARRIVAL_FAMILIES)
    assert {row["migration_mode"] for row in report["runs"]} == set(EXPECTED_MIGRATION_MODES)
    assert {row["policy"] for row in report["runs"] if row["policy_category"] == "sota_style"} == {
        spec.policy.name for spec in sota_baseline_specs()
    }
    assert {row["policy"] for row in report["runs"] if row["policy_category"] == "ablation"} == {
        name for name, _ in ABLATION_POLICIES
    }
    assert {row["policy"] for row in report["runs"] if row["policy_category"] == "legacy"} == {
        LEGACY_POLICY
    }
    assert any(row["policy"] == OURS_POLICY for row in report["runs"])
    assert all(row["completed_jobs"] == row["job_count"] for row in report["runs"])
    assert all(row["resource_simulation"] == "shared_lane_joint_event_v2" for row in report["runs"])
    assert all(row["mean_queue_backlog_jobs"] >= 0.0 for row in report["runs"])
    assert report["comparison_kind"] == "same-cache_policy-semantics"
    assert report["full_stack_external_binary_comparison"] is False
    assert "not direct full-stack" in report["claim_boundary"]
    assert report["performance_diagnostics"]["diagnostic_only_not_a_gate_condition"] is True
    assert report["legacy_comparison_rows"]
    assert report["performance_diagnostics"]["legacy"]["comparison_count"] == len(
        report["legacy_comparison_rows"]
    )
    assert all(
        not audit["loaded_action"] and not audit["migration_action"]
        for row in report["runs"]
        if row["policy"] == LEGACY_POLICY
        for audit in row["selection_audit"]
    )
    loaded_audits = [
        audit
        for row in report["runs"]
        for audit in row["selection_audit"]
        if audit["loaded_action"]
    ]
    assert loaded_audits
    assert all(
        audit["target_batch_width"] == 1
        and sum(audit["required_job_counts"].values()) == 2
        and sum(audit["resident_job_counts"].values()) == 1
        for audit in loaded_audits
    )
    joint_run = next(
        row
        for row in report["runs"]
        if row["scenario_id"] == "q01:gpu_test_dual_gpuA"
        and row["arrival_family"] == "static"
        and row["migration_mode"] == "without_migration"
        and row["policy"] == OURS_POLICY
    )
    workloads_by_lane = {}
    for audit in joint_run["selection_audit"]:
        workloads_by_lane.setdefault(audit["resource_lane"], set()).add(
            audit["target_workload_key"]
        )
    assert any(len(workloads) > 1 for workloads in workloads_by_lane.values())
    migration_ours = [
        row
        for row in report["runs"]
        if row["scenario_kind"] == "controlled_migration_counterfactual"
        and row["policy"] == OURS_POLICY
    ]
    assert migration_ours
    assert all(
        not any(action.startswith("migrate:") for action in row["selected_action_counts"])
        for row in migration_ours
        if row["migration_mode"] == "without_migration"
    )
    assert any(
        any(action.startswith("migrate:") for action in row["selected_action_counts"])
        for row in migration_ours
        if row["migration_mode"] == "with_migration"
    )


def test_missing_input_waits_without_partial_results(tmp_path):
    paths = _write_fixture(tmp_path)
    paths["ledger"].unlink()

    report = build_unified_hardware_or_replay(
        cache_path=paths["cache"],
        loaded_ledger_path=paths["ledger"],
        migration_certificate_path=paths["migration"],
        loads=(0.50,),
        seeds=(1,),
        jobs_per_workload=2,
    )

    assert report["status"] == "WAIT_INPUTS"
    assert report["pass"] is False
    assert report["runs"] == []
    assert report["pareto_rows"] == []
    assert report["legacy_comparison_rows"] == []
    assert report["blockers"] == [
        {
            "code": "MISSING_INPUT",
            "source": "loaded_action_ledger",
            "detail": str(paths["ledger"]),
        }
    ]


def test_stale_migration_rate_or_cache_binding_fails_closed(tmp_path):
    paths = _write_fixture(tmp_path)
    migration = json.loads(paths["migration"].read_text(encoding="utf-8"))
    migration["rows"][0]["current_lower_service"] *= 1.5
    _write_json(paths["migration"], migration)

    report = build_unified_hardware_or_replay(
        cache_path=paths["cache"],
        loaded_ledger_path=paths["ledger"],
        migration_certificate_path=paths["migration"],
        loads=(0.50,),
        seeds=(1,),
        jobs_per_workload=2,
    )

    assert report["status"] == "FAIL_INPUT_CONTRACT"
    assert report["runs"] == []
    assert "source rate mismatch" in report["blockers"][0]["detail"]


def test_migration_certificate_must_bind_exact_cache_hash(tmp_path):
    paths = _write_fixture(tmp_path)
    migration = json.loads(paths["migration"].read_text(encoding="utf-8"))
    migration["service_cache_sha256"] = "0" * 64
    _write_json(paths["migration"], migration)

    report = build_unified_hardware_or_replay(
        cache_path=paths["cache"],
        loaded_ledger_path=paths["ledger"],
        migration_certificate_path=paths["migration"],
        loads=(0.50,),
        seeds=(1,),
        jobs_per_workload=2,
    )

    assert report["status"] == "FAIL_INPUT_CONTRACT"
    assert report["runs"] == []
    assert "not bound to the supplied cache hash" in report["blockers"][0]["detail"]


def test_loaded_action_must_match_both_exact_cache_coordinates(tmp_path):
    paths = _write_fixture(tmp_path)
    ledger = json.loads(paths["ledger"].read_text(encoding="utf-8"))
    ledger["actions"][0]["lower_service_vector"]["gpu_llm_distilgpt2"] += 0.25
    _write_json(paths["ledger"], ledger)

    report = build_unified_hardware_or_replay(
        cache_path=paths["cache"],
        loaded_ledger_path=paths["ledger"],
        migration_certificate_path=paths["migration"],
        loads=(0.50,),
        seeds=(1,),
        jobs_per_workload=2,
    )

    assert report["status"] == "FAIL_INPUT_CONTRACT"
    assert report["runs"] == []
    assert "resident lower service mismatch" in report["blockers"][0]["detail"]


def test_history_eta_is_never_admitted_to_natural_completion_view(tmp_path):
    paths = _write_fixture(tmp_path)
    cache = json.loads(paths["cache"].read_text(encoding="utf-8"))
    row = next(record for record in cache["records"] if record["workload_key"] == "light_control_local")
    row["eta_source"] = "scheduler_history_eta"
    _write_json(paths["cache"], cache)
    migration = json.loads(paths["migration"].read_text(encoding="utf-8"))
    migration["service_cache_sha256"] = _sha256(paths["cache"])
    _write_json(paths["migration"], migration)

    report = build_unified_hardware_or_replay(
        cache_path=paths["cache"],
        loaded_ledger_path=paths["ledger"],
        migration_certificate_path=paths["migration"],
        loads=(0.50,),
        seeds=(1,),
        jobs_per_workload=2,
    )

    assert report["status"] == "FAIL_INPUT_CONTRACT"
    assert report["runs"] == []
    assert "history ETA is forbidden" in report["blockers"][0]["detail"]


def test_completion_points_change_metrics_without_entering_the_selector(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    paths_a = _write_fixture(first)
    paths_b = _write_fixture(second, completion_scale=2.0)

    report_a = build_unified_hardware_or_replay(
        cache_path=paths_a["cache"],
        loaded_ledger_path=paths_a["ledger"],
        migration_certificate_path=paths_a["migration"],
        loads=(0.50,),
        seeds=(5,),
        jobs_per_workload=3,
    )
    report_b = build_unified_hardware_or_replay(
        cache_path=paths_b["cache"],
        loaded_ledger_path=paths_b["ledger"],
        migration_certificate_path=paths_b["migration"],
        loads=(0.50,),
        seeds=(5,),
        jobs_per_workload=3,
    )

    assert report_a["pass"] and report_b["pass"]
    first_a = {
        (row["scenario_id"], row["trace_id"], row["migration_mode"], row["policy"]): row["selection_audit"][0]["action_id"]
        for row in report_a["runs"]
        if row["arrival_family"] == "static"
    }
    first_b = {
        (row["scenario_id"], row["trace_id"], row["migration_mode"], row["policy"]): row["selection_audit"][0]["action_id"]
        for row in report_b["runs"]
        if row["arrival_family"] == "static"
    }
    assert first_a == first_b
    assert all(
        audit["selection_service_view"] == "lower_service"
        for report in (report_a, report_b)
        for row in report["runs"]
        for audit in row["selection_audit"]
    )
    metrics_a = [(row["makespan_s"], row["mean_flow_s"]) for row in report_a["runs"]]
    metrics_b = [(row["makespan_s"], row["mean_flow_s"]) for row in report_b["runs"]]
    assert metrics_a != metrics_b


def test_migration_uses_per_task_lower_service_for_profile_two(tmp_path):
    paths = _write_fixture(tmp_path)
    cache = json.loads(paths["cache"].read_text(encoding="utf-8"))
    source = next(
        row
        for row in cache["records"]
        if row["workload_key"] == "hybrid_rl_resac_ant"
        and row["node_bucket"] == "gpu_test_dual:gpuA"
        and row["profile"] == 2
        and row["resource_state"] == "empty"
    )
    destination = dict(source)
    destination["node_bucket"] = "gpu_test_dual:gpuB"
    destination["command_fingerprint"] = "synthetic:hybrid-profile-two-destination"
    destination["aggregate_rate"] = 2.4
    destination["per_task_rates"] = [1.2, 1.2]
    cache["records"].append(destination)
    _write_json(paths["cache"], cache)

    migration = json.loads(paths["migration"].read_text(encoding="utf-8"))
    rows = [row for row in migration["rows"] if row["family"] == "hybrid_rl"]
    for row in rows:
        row["source_service_key"] = _service_key(ProfileRecord.from_snapshot(source))
        row["destination_service_key"] = _service_key(ProfileRecord.from_snapshot(destination))
        row["current_lower_service"] = 0.75
        row["target_lower_service"] = 1.2
        penalty = row["migration_cost"]["total_penalty_units"]
        beneficial = (
            row["remaining_work"] / row["current_lower_service"]
            > penalty + row["remaining_work"] / row["target_lower_service"]
        )
        row["beneficial_by_threshold"] = beneficial
        row["theorem_ready"] = beneficial
    migration["service_cache_sha256"] = _sha256(paths["cache"])
    _write_json(paths["migration"], migration)

    report = build_unified_hardware_or_replay(
        cache_path=paths["cache"],
        loaded_ledger_path=paths["ledger"],
        migration_certificate_path=paths["migration"],
        loads=(0.50,),
        seeds=(1,),
        jobs_per_workload=2,
    )

    assert report["status"] == "PASS"


def test_nonbeneficial_migration_is_not_an_admitted_action(tmp_path):
    paths = _write_fixture(tmp_path)
    migration = json.loads(paths["migration"].read_text(encoding="utf-8"))
    row = next(item for item in migration["rows"] if item["family"] == "pure_gpu")
    cost = row["migration_cost"]
    cost["risk_penalty_units"] = 10_000.0
    cost["total_penalty_units"] = cost["total_time_s"] + cost["risk_penalty_units"]
    row["beneficial_by_threshold"] = False
    row["theorem_ready"] = False
    _write_json(paths["migration"], migration)

    report = build_unified_hardware_or_replay(
        cache_path=paths["cache"],
        loaded_ledger_path=paths["ledger"],
        migration_certificate_path=paths["migration"],
        loads=(0.50,),
        seeds=(1,),
        jobs_per_workload=2,
    )

    assert report["status"] == "PASS"
    scenario = next(
        item
        for item in report["scenarios"]
        if item["scenario_id"] == f"migration:{row['action_id']}"
    )
    assert scenario["action_ids"] == [f"keep:{row['action_id']}"]


def test_diagonal_work_unit_rescaling_preserves_policy_trajectory_and_jct(tmp_path):
    original_paths = _write_fixture(tmp_path / "original")
    scaled_paths = _write_fixture(tmp_path / "scaled")
    cache = json.loads(scaled_paths["cache"].read_text(encoding="utf-8"))
    factor = 1000.0
    for row in cache["records"]:
        if row["workload_key"] != "gpu_llm_distilgpt2":
            continue
        row["total_units"] *= factor
        row["aggregate_rate"] *= factor
        row["per_task_rates"] = [value * factor for value in row["per_task_rates"]]
        if row.get("completion_group_total_units", 0.0) > 0.0:
            row["completion_group_total_units"] *= factor
        if row.get("completion_unit_s", 0.0) > 0.0:
            row["completion_unit_s"] /= factor
    _write_json(scaled_paths["cache"], cache)

    ledger = json.loads(scaled_paths["ledger"].read_text(encoding="utf-8"))
    for action in ledger["actions"]:
        vector = action["lower_service_vector"]
        if "gpu_llm_distilgpt2" in vector:
            vector["gpu_llm_distilgpt2"] *= factor
    _write_json(scaled_paths["ledger"], ledger)
    migration = json.loads(scaled_paths["migration"].read_text(encoding="utf-8"))
    migration["service_cache_sha256"] = _sha256(scaled_paths["cache"])
    _write_json(scaled_paths["migration"], migration)

    kwargs = {"loads": (0.50,), "seeds": (7,), "jobs_per_workload": 3}
    original = build_unified_hardware_or_replay(
        cache_path=original_paths["cache"],
        loaded_ledger_path=original_paths["ledger"],
        migration_certificate_path=original_paths["migration"],
        **kwargs,
    )
    scaled = build_unified_hardware_or_replay(
        cache_path=scaled_paths["cache"],
        loaded_ledger_path=scaled_paths["ledger"],
        migration_certificate_path=scaled_paths["migration"],
        **kwargs,
    )

    assert original["pass"] and scaled["pass"]
    identity_fields = ("scenario_id", "trace_id", "migration_mode", "policy")
    metric_fields = (
        "makespan_s",
        "mean_flow_s",
        "p90_flow_s",
        "mean_queue_backlog_jobs",
    )
    for left, right in zip(original["runs"], scaled["runs"], strict=True):
        assert tuple(left[field] for field in identity_fields) == tuple(
            right[field] for field in identity_fields
        )
        assert left["selected_action_counts"] == right["selected_action_counts"]
        assert [row["action_id"] for row in left["selection_audit"]] == [
            row["action_id"] for row in right["selection_audit"]
        ]
        assert left["max_queue_backlog_jobs"] == right["max_queue_backlog_jobs"]
        for field in metric_fields:
            assert abs(left[field] - right[field]) <= 1e-12 * max(
                1.0, abs(left[field]), abs(right[field])
            )


def _write_fixture(root: Path, *, completion_scale: float = 1.0) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    cache_path = root / "unified_cache.json"
    ledger_path = root / "loaded_ledger.json"
    migration_path = root / "migration_certificate.json"

    records = [
        _completion_record("light_control_local", "light_control", "node003:cpu_hpc_192c", 1, 2.0, 2.4, completion_scale),
        _completion_record("light_control_local", "light_control", "node003:cpu_hpc_192c", 2, 3.0, 3.4, completion_scale, state="controlled_colocation", mix="same_workload"),
        _completion_record("cpu_heavy_local_bench", "cpu", "node001:cpu_hpc_192c", 1, 2.0, 2.3, completion_scale),
        _completion_record("cpu_heavy_local_bench", "cpu", "node002:cpu_hpc_192c", 1, 3.0, 3.2, completion_scale),
        _completion_record("gpu_heavy_jax_matmul", "jax", "gpu_test_dual:gpuA", 1, 5.0, 5.5, completion_scale),
        _completion_record("gpu_heavy_jax_matmul", "jax", "gpu_test_dual:gpuA", 2, 7.0, 7.7, completion_scale),
        _completion_record("gpu_cnn_torch_resnet50", "resnet50", "gpu_test_dual:gpuA", 1, 4.0, 4.5, completion_scale),
        _completion_record("gpu_cnn_torch_resnet50", "resnet50", "gpu_test_dual:gpuA", 2, 6.0, 6.7, completion_scale),
        _completion_record("gpu_cnn_torch_resnet50", "resnet50", "gpu_test_dual:gpuB", 1, 6.0, 6.8, completion_scale),
        _completion_record("gpu_llm_distilgpt2", "distilgpt2", "gpu_test_dual:gpuA", 1, 3.0, 3.5, completion_scale),
        _completion_record("gpu_llm_distilgpt2", "distilgpt2", "gpu_test_dual:gpuA", 2, 5.0, 5.6, completion_scale),
        _completion_record("hybrid_rl_resac_ant", "resac_ant", "gpu_test_dual:gpuA", 1, 1.0, 1.2, completion_scale),
        _completion_record("hybrid_rl_resac_ant", "resac_ant", "gpu_test_dual:gpuA", 2, 1.5, 1.8, completion_scale),
        _completion_record("hybrid_rl_resac_ant", "resac_ant", "gpu_test_dual:gpuB", 1, 2.0, 2.2, completion_scale),
        _completion_record(
            "gpu_cnn_torch_resnet50",
            "resnet50",
            "gpu_test_dual:gpuA",
            2,
            2.5,
            3.0,
            completion_scale,
            state="mixed_colocation",
            mix="resident_llm_target_cnn",
            total_units=20.0,
        ),
        _lower_only_record(
            "gpu_llm_distilgpt2",
            "distilgpt2",
            "gpu_test_dual:gpuA",
            2,
            1.7,
            state="mixed_colocation",
            mix="resident_llm_target_cnn",
        ),
        _completion_record(
            "hybrid_rl_resac_ant",
            "resac_ant",
            "gpu_test_dual:gpuA",
            2,
            0.8,
            1.0,
            completion_scale,
            state="mixed_colocation",
            mix="resident_cnn_target_hybrid_rl",
            total_units=20.0,
        ),
        _lower_only_record(
            "gpu_cnn_torch_resnet50",
            "resnet50",
            "gpu_test_dual:gpuA",
            2,
            2.1,
            state="mixed_colocation",
            mix="resident_cnn_target_hybrid_rl",
        ),
    ]
    cache = ServiceRateCache(records)
    _write_json(cache_path, cache.snapshot())

    ledger = {
        "schema_version": 1,
        "ledger": "critical_gpu_loaded_action_ledger",
        "measurement_protocol": "synthetic_joint_holdout_fixture",
        "action_count": 2,
        "actions": [
            _loaded_action(
                action_id="gpu_loaded:gpuA:cnn_after_llm",
                node_bucket="gpu_test_dual:gpuA",
                resident_key="gpu_llm_distilgpt2",
                resident_env="distilgpt2",
                target_key="gpu_cnn_torch_resnet50",
                target_env="resnet50",
                resident_mix="resident_llm_target_cnn",
                resident_rate=1.7,
                target_rate=2.5,
                upper_s=1000.0,
            ),
            _loaded_action(
                action_id="gpu_loaded:gpuA:rl_after_cnn",
                node_bucket="gpu_test_dual:gpuA",
                resident_key="gpu_cnn_torch_resnet50",
                resident_env="resnet50",
                target_key="hybrid_rl_resac_ant",
                target_env="resac_ant",
                resident_mix="resident_cnn_target_hybrid_rl",
                resident_rate=2.1,
                target_rate=0.8,
                upper_s=1000.0,
            ),
        ],
    }
    _write_json(ledger_path, ledger)

    cache_hash = _sha256(cache_path)
    migration_rows = []
    migration_contracts = (
        (
            "pure_gpu",
            "gpu_cnn_torch_resnet50",
            _service_key(records[6]),
            _service_key(records[8]),
            4.0,
            6.0,
        ),
        (
            "hybrid_rl",
            "hybrid_rl_resac_ant",
            _service_key(records[11]),
            _service_key(records[13]),
            1.0,
            2.0,
        ),
        (
            "pure_cpu",
            "cpu_heavy_local_bench",
            _service_key(records[2]),
            _service_key(records[3]),
            2.0,
            3.0,
        ),
    )
    for family, workload, source, destination, current_rate, target_rate in migration_contracts:
        for point in (0.25, 0.50, 0.75):
            remaining = 100.0 * (1.0 - point)
            penalty = 0.1
            migration_rows.append(
                {
                    "action_id": f"migrate:{family}:p{int(point * 100)}",
                    "family": family,
                    "workload_key": workload,
                    "progress_fraction": point,
                    "measurement_valid": True,
                    "controlled_benchmark": True,
                    "checkpoint_verified": True,
                    "resume_verified": True,
                    "rates_recomputed_from_cache": True,
                    "ordinary_running_tasks_touched": False,
                    "source_service_key": source,
                    "destination_service_key": destination,
                    "current_lower_service": current_rate,
                    "target_lower_service": target_rate,
                    "remaining_work": remaining,
                    "remaining_work_unit": "iteration",
                    "migration_cost": {
                        "checkpoint_flush_s": 0.01,
                        "sync_s": 0.02,
                        "environment_staging_s": 0.01,
                        "resume_warmup_s": 0.01,
                        "lost_work_s": 0.0,
                        "risk_penalty_units": 0.05,
                        "total_time_s": 0.05,
                        "total_penalty_units": penalty,
                    },
                    "beneficial_by_threshold": remaining * (target_rate - current_rate) > penalty,
                    "theorem_ready": True,
                }
            )
    migration = {
        "gate": "unified_migration_recalibration_certificate",
        "schema_version": 1,
        "status": "MIGRATION_RECALIBRATION_PASS",
        "pass": True,
        "service_cache_sha256": cache_hash,
        "rates_recomputed_from_final_cache": True,
        "no_touch_safety_ready": True,
        "rows": migration_rows,
    }
    _write_json(migration_path, migration)
    return {"cache": cache_path, "ledger": ledger_path, "migration": migration_path}


def _completion_record(
    workload_key: str,
    workload_env: str,
    node_bucket: str,
    profile: int,
    lower_rate: float,
    point_rate: float,
    completion_scale: float,
    *,
    state: str = "empty",
    mix: str = "",
    total_units: float | None = None,
) -> ProfileRecord:
    group_units = float(total_units if total_units is not None else profile * 20)
    wall = group_units / point_rate * completion_scale
    return ProfileRecord(
        workload_key=workload_key,
        command_fingerprint=f"synthetic:{workload_key}:{node_bucket}:{state}:{mix}:p{profile}",
        resource_kind="synthetic",
        node_bucket=node_bucket,
        profile=profile,
        unit="iteration",
        total_units=group_units,
        aggregate_rate=lower_rate,
        per_task_rates=tuple(lower_rate / profile for _ in range(profile)),
        source="synthetic-fixture",
        workload_env=workload_env,
        resource_state=state,
        resident_mix=mix,
        eta_source=ETA_SOURCE,
        stable_rate_ready=True,
        completion_model_ready=True,
        completion_model_sample_count=12,
        completion_total_wall_s=wall,
        completion_group_total_units=group_units,
        completion_unit_s=wall / group_units,
        allocation_workers=1,
        colocation_count=profile,
    )


def _lower_only_record(
    workload_key: str,
    workload_env: str,
    node_bucket: str,
    profile: int,
    lower_rate: float,
    *,
    state: str,
    mix: str,
) -> ProfileRecord:
    return ProfileRecord(
        workload_key=workload_key,
        command_fingerprint=f"synthetic-lower:{workload_key}:{node_bucket}:{state}:{mix}:p{profile}",
        resource_kind="synthetic",
        node_bucket=node_bucket,
        profile=profile,
        unit="iteration",
        total_units=float(profile * 20),
        aggregate_rate=lower_rate,
        per_task_rates=tuple(lower_rate / profile for _ in range(profile)),
        source="synthetic-fixture",
        workload_env=workload_env,
        resource_state=state,
        resident_mix=mix,
        eta_source="task_native_tqdm_loaded_overlap_fixture",
        stable_rate_ready=True,
        completion_model_ready=False,
        allocation_workers=1,
        colocation_count=profile,
    )


def _loaded_action(
    *,
    action_id: str,
    node_bucket: str,
    resident_key: str,
    resident_env: str,
    target_key: str,
    target_env: str,
    resident_mix: str,
    resident_rate: float,
    target_rate: float,
    upper_s: float,
) -> dict[str, object]:
    return {
        "action_id": action_id,
        "action_kind": "directional_gpu_colocation_trajectory",
        "physical_node": "gpuA",
        "node_bucket": node_bucket,
        "hardware_class": "synthetic_dual_gpu",
        "resource_state": "mixed_colocation",
        "resident_mix": resident_mix,
        "profile": 2,
        "profile_axis": "total_tasks_per_gpu",
        "resident_workload_key": resident_key,
        "resident_workload_env": resident_env,
        "target_workload_key": target_key,
        "target_workload_env": target_env,
        "lower_service_vector": {
            resident_key: resident_rate,
            target_key: target_rate,
        },
        "bounded_action_penalty_units": 0.0,
        "simultaneous_upper_target_completion_s": upper_s,
        "target_completion_holdout_covered": True,
        "resident_service_holdout_covered": True,
        "source": "synthetic-fixture",
    }


def _service_key(record: ProfileRecord) -> dict[str, object]:
    return {
        "workload_key": record.workload_key,
        "workload_env": record.workload_env,
        "node_bucket": record.node_bucket,
        "resource_state": record.resource_state,
        "resident_mix": record.resident_mix,
        "profile": record.profile,
        "allocation_workers": record.allocation_workers,
        "colocation_count": record.colocation_count,
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
