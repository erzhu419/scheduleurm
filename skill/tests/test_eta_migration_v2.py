from algorithm.experiments.eta_migration_corner_case_matrix import build_corner_case_matrix
from algorithm.experiments.eta_server_coverage_gate import EXPECTED_ROWS, build_eta_server_coverage_gate
from algorithm.experiments.full_factorial_eta_design import (
    HardwareType,
    ResidentScenario,
    WorkloadType,
    _boundary_map,
    _measurement_status,
)
from algorithm.experiments.full_factorial_cpu_eta_probe_runner import _progress_rates, _stable_tail
from algorithm.experiments.full_factorial_eta_probe_runner import _add_result_to_cache
from algorithm.experiments.full_factorial_parallel_probe_plan import build_parallel_probe_plan
from algorithm.experiments.live_marginal_under_load_matrix import (
    _benchmark_command,
    _resac_env_for_kind,
    _wrapped_command,
    build_live_marginal_from_design,
)
from algorithm.experiments.migration_cost_gate import build_migration_cost_gate
from algorithm.experiments.or_generalization_benchmarks import build_or_generalization_gate
from algorithm.features import gpu_candidate_features
from algorithm.theorem_dispatch.global_dispatch import select_global_action
from algorithm.theorem_dispatch.migration import MigrationCost, build_migration_action_row
from algorithm.theorem_dispatch.service_registry import bind_service, infer_workload_env, infer_workload_key
from simulation.service_cache import (
    ProfileRecord,
    ServiceRateCache,
    StatewiseLookupResult,
)


def _record(node_bucket: str, env: str, rate: float, workload_key: str = "hybrid_rl_resac_halfcheetah") -> ProfileRecord:
    return ProfileRecord(
        workload_key=workload_key,
        command_fingerprint="live_tqdm_service_cache_v2_noearly_20260629",
        resource_kind="hybrid_rl",
        node_bucket=node_bucket,
        profile=1,
        unit="iter",
        total_units=100.0,
        aggregate_rate=rate,
        per_task_rates=(rate,),
        source="unit:tqdm",
        workload_env=env,
        resource_state="empty",
        eta_source="tqdm/progress",
        stable_rate_ready=True,
    )


def test_service_cache_v2_prefers_env_node_state_exact_row():
    cache = ServiceRateCache(
        (
            _record("node007:gpu_node007_4x12gb", "halfcheetah", 1.0),
            _record("jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast", "halfcheetah", 3.0),
        )
    )

    record = cache.get_statewise(
        "hybrid_rl_resac_halfcheetah",
        1,
        node_bucket="jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast",
        workload_env="HalfCheetah-v2",
        resource_state="empty",
        strict=True,
    )

    assert record is not None
    assert record.node_bucket.startswith("jtl311linux")
    assert record.aggregate_rate == 3.0


def test_exact_statewise_lookup_never_broadens_state_node_or_hardware():
    row = ProfileRecord(
        workload_key="gpu_cnn_torch_resnet50",
        command_fingerprint="live_tqdm_service_cache_v2_unit",
        resource_kind="gpu_cnn",
        node_bucket="jtl110gpu2:gpu_3080ti_12gb_dual",
        profile=2,
        unit="step",
        total_units=100.0,
        aggregate_rate=20.0,
        per_task_rates=(10.0, 10.0),
        source="service_cache_v2_unit",
        workload_env="cnn",
        resource_state="unspecified",
        eta_source="tqdm/progress",
        stable_rate_ready=True,
        hardware_class="gpu_3080ti_12gb_dual",
    )
    shortened_node_row = ProfileRecord(
        workload_key="gpu_cnn_torch_resnet50",
        command_fingerprint="live_tqdm_service_cache_v2_unit",
        resource_kind="gpu_cnn",
        node_bucket="node007",
        profile=3,
        unit="step",
        total_units=100.0,
        aggregate_rate=24.0,
        per_task_rates=(8.0, 8.0, 8.0),
        source="service_cache_v2_short_node_unit",
        workload_env="cnn",
        resource_state="empty",
        eta_source="tqdm/progress",
        stable_rate_ready=True,
        hardware_class="gpu_node007_4x12gb",
    )
    cache = ServiceRateCache((row, shortened_node_row))

    permissive_state = cache.get_statewise(
        row.workload_key,
        2,
        node_bucket=row.node_bucket,
        workload_env="cnn",
        resource_state="empty",
        strict=True,
    )
    permissive_node = cache.get_statewise(
        row.workload_key,
        2,
        node_bucket="jtl110gpu:gpu_3080ti_12gb_dual",
        workload_env="cnn",
        resource_state="unspecified",
        strict=True,
    )
    permissive_short_node = cache.get_statewise(
        shortened_node_row.workload_key,
        3,
        node_bucket="node007:gpu_node007_4x12gb",
        workload_env="cnn",
        resource_state="empty",
        strict=True,
    )
    wrong_state = cache.lookup_statewise_exact(
        row.workload_key,
        2,
        node_bucket=row.node_bucket,
        workload_env="cnn",
        resource_state="empty",
    )
    wrong_node = cache.lookup_statewise_exact(
        row.workload_key,
        2,
        node_bucket="jtl110gpu:gpu_3080ti_12gb_dual",
        workload_env="cnn",
        resource_state="unspecified",
    )
    shortened_node = cache.lookup_statewise_exact(
        shortened_node_row.workload_key,
        3,
        node_bucket="node007:gpu_node007_4x12gb",
        workload_env="cnn",
        resource_state="empty",
    )
    exact = cache.lookup_statewise_exact(
        row.workload_key,
        2,
        node_bucket=row.node_bucket,
        workload_env="cnn",
        resource_state="unspecified",
    )

    assert permissive_state is row
    assert permissive_node is row
    assert permissive_short_node is shortened_node_row
    assert isinstance(wrong_state, StatewiseLookupResult)
    assert wrong_state.status == "missing"
    assert wrong_state.is_missing
    assert wrong_state.record is None
    assert wrong_node.status == "missing"
    assert shortened_node.status == "missing"
    assert exact.status == "exact"
    assert exact.is_exact
    assert exact.record is row


def test_exact_statewise_lookup_enforces_v2_boundary_before_legacy_positive():
    node_bucket = "jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast"
    legacy_positive = ProfileRecord(
        workload_key="gpu_cnn_torch_resnet50",
        command_fingerprint="legacy_cnn_curve",
        resource_kind="gpu_cnn",
        node_bucket=node_bucket,
        profile=4,
        unit="step",
        total_units=100.0,
        aggregate_rate=40.0,
        per_task_rates=(10.0, 10.0, 10.0, 10.0),
        source="legacy-unit",
        workload_env="cnn",
        resource_state="unspecified",
    )
    boundary = ProfileRecord(
        workload_key="gpu_cnn_torch_resnet50",
        command_fingerprint="live_tqdm_service_cache_v2_unit",
        resource_kind="gpu_cnn",
        node_bucket=node_bucket,
        profile=3,
        unit="step",
        total_units=100.0,
        aggregate_rate=0.0,
        per_task_rates=(),
        source="service_cache_v2_boundary_unit",
        capacity_boundary=True,
        workload_env="cnn",
        resource_state="empty",
        hardware_class="gpu_rtx2080_8gb_dual_cpu_fast",
    )
    cache = ServiceRateCache((legacy_positive, boundary))

    permissive = cache.get_statewise(
        boundary.workload_key,
        4,
        node_bucket=node_bucket,
        workload_env="cnn",
        resource_state="empty",
        strict=True,
    )
    exact = cache.lookup_statewise_exact(
        boundary.workload_key,
        4,
        node_bucket=node_bucket,
        workload_env="cnn",
        resource_state="empty",
    )
    other_node = cache.lookup_statewise_exact(
        boundary.workload_key,
        4,
        node_bucket="node007:gpu_node007_4x12gb",
        workload_env="cnn",
        resource_state="empty",
    )

    assert permissive is legacy_positive
    assert permissive.aggregate_rate > 0
    assert exact.status == "scoped_boundary"
    assert exact.is_scoped_boundary
    assert exact.record is None
    assert exact.boundary is boundary
    assert other_node.status == "missing"
    assert other_node.boundary is None


def test_scoped_capacity_boundary_does_not_block_other_node_valid_profile():
    boundary = ProfileRecord(
        workload_key="gpu_cnn_torch_resnet50",
        command_fingerprint="live_tqdm_service_cache_v2_noearly_20260629",
        resource_kind="gpu_cnn",
        node_bucket="jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast",
        profile=4,
        unit="step",
        total_units=100.0,
        aggregate_rate=1.0,
        per_task_rates=(1.0,),
        source="unit:boundary",
        capacity_boundary=True,
        workload_env="cnn",
        resource_state="empty",
        eta_source="tqdm/progress",
        stable_rate_ready=True,
        hardware_class="gpu_rtx2080_8gb_dual_cpu_fast",
    )
    valid = ProfileRecord(
        workload_key="gpu_cnn_torch_resnet50",
        command_fingerprint="unit",
        resource_kind="gpu_cnn",
        node_bucket="node007:gpu_node007_4x12gb",
        profile=4,
        unit="step",
        total_units=100.0,
        aggregate_rate=8.0,
        per_task_rates=(2.0, 2.0, 2.0, 2.0),
        source="unit:valid",
        workload_env="cnn",
        resource_state="empty",
        eta_source="tqdm/progress",
        stable_rate_ready=True,
        hardware_class="gpu_node007_4x12gb",
    )
    cache = ServiceRateCache((boundary, valid))

    record = cache.get_statewise(
        "gpu_cnn_torch_resnet50",
        4,
        node_bucket="node007:gpu_node007_4x12gb",
        workload_env="cnn",
        resource_state="empty",
        strict=True,
    )

    assert record is not None
    assert not record.capacity_boundary
    assert record.node_bucket.startswith("node007")
    assert record.aggregate_rate == 8.0
    assert cache.profiles("gpu_cnn_torch_resnet50") == []
    assert any(
        row["profile"] == 4
        and row["node_bucket"].startswith("node007")
        and not row["capacity_boundary"]
        for row in cache.snapshot()["records"]
    )


def test_design_uses_hardware_class_capacity_boundary_for_equivalent_nodes_only():
    cache = ServiceRateCache(
        (
            ProfileRecord(
                workload_key="gpu_cnn_torch_resnet50",
                command_fingerprint="unit-boundary",
                resource_kind="gpu_cnn",
                node_bucket="jtl110gpu2:gpu_3080ti_12gb_dual",
                profile=5,
                unit="step",
                total_units=100.0,
                aggregate_rate=40.0,
                per_task_rates=(8.0, 8.0, 8.0, 8.0, 8.0),
                source="unit:boundary",
                capacity_boundary=True,
                workload_env="cnn",
                resource_state="full_loaded",
                eta_source="tqdm/progress",
                stable_rate_ready=True,
                hardware_class="gpu_3080ti_12gb_dual",
            ),
        )
    )
    workload = WorkloadType(
        "gpu_cnn_torch_resnet50",
        "cnn",
        "cnn",
        "q01_low_cpu_high_gpu",
        "gpu",
        (1, 2, 3, 4, 5, 6, 8),
        True,
        "torch_cnn_resnet50.cmd.tpl",
        "unit",
    )
    scenario = ResidentScenario(
        "full_loaded",
        "same_workload_to_capacity_boundary",
        ("gpu",),
        ("gpu",),
        "to_boundary",
        "required",
        "unit",
    )
    same_class = HardwareType(
        "gpu_3080ti_12gb_dual",
        "jtl110gpu",
        "jtl110gpu:gpu_3080ti_12gb_dual",
        ("jtl110gpu2",),
        "gpu",
        2,
        0,
        "unit",
    )
    different_class = HardwareType(
        "gpu_node007_4x11gb",
        "node007",
        "node007:gpu_node007_4x12gb",
        (),
        "gpu",
        4,
        64,
        "unit",
    )
    boundaries = _boundary_map(cache)

    same_status = _measurement_status(
        cache,
        workload,
        same_class,
        scenario,
        (5, 6, 8),
        boundary_map=boundaries,
    )
    different_status = _measurement_status(
        cache,
        workload,
        different_class,
        scenario,
        (5, 6, 8),
        boundary_map=boundaries,
    )

    assert same_status == ("capacity_boundary_closed", [], [], [5, 6, 8])
    assert different_status == ("pending_probe", [], [5, 6, 8], [])


def test_design_status_uses_same_workload_resident_mix_for_t0_rows():
    cache = ServiceRateCache(
        (
            ProfileRecord(
                workload_key="hybrid_rl_bapr_ant",
                command_fingerprint="unit-half-loaded",
                resource_kind="hybrid_rl",
                node_bucket="node007:gpu_node007_4x12gb",
                profile=2,
                unit="iter",
                total_units=200.0,
                aggregate_rate=0.1,
                per_task_rates=(0.1,),
                source="unit",
                workload_env="bapr_ant",
                resource_state="half_loaded",
                resident_mix="same_workload_half_capacity",
                eta_source="tqdm/progress",
                stable_rate_ready=True,
            ),
        )
    )
    workload = WorkloadType(
        "hybrid_rl_bapr_ant",
        "bapr_ant",
        "hybrid_rl",
        "q11_high_cpu_high_gpu",
        "gpu",
        (2, 3, 4),
        True,
        "bapr_ant_real.cmd.tpl",
        "unit",
    )
    scenario = ResidentScenario(
        "half_loaded",
        "same_workload_half_capacity",
        ("gpu",),
        ("gpu",),
        "middle_profiles",
        "required",
        "unit",
    )
    hw = HardwareType(
        "gpu_node007_4x11gb",
        "node007",
        "node007:gpu_node007_4x12gb",
        (),
        "gpu",
        4,
        64,
        "unit",
    )

    status = _measurement_status(cache, workload, hw, scenario, (2, 3), boundary_map={})

    assert status == ("partial_pending_probe", [2], [3], [])


def test_legacy_full_loaded_snapshot_infers_same_workload_resident_mix():
    record = ProfileRecord.from_snapshot(
        {
            "workload_key": "gpu_cnn_torch_resnet50",
            "command_fingerprint": "live_tqdm_service_cache_v2_full_factorial_20260630",
            "resource_kind": "gpu_cnn",
            "node_bucket": "node001:cpu_hpc_192c",
            "profile": 8,
            "unit": "step",
            "total_units": 160.0,
            "aggregate_rate": 1.0,
            "per_task_rates": [1.0],
            "source": "legacy-snapshot-without-resident-mix",
            "workload_env": "cnn",
            "resource_state": "full_loaded",
            "eta_source": "tqdm/progress",
            "stable_rate_ready": True,
        }
    )

    assert record.resident_mix == "same_workload_to_capacity_boundary"


def test_node_specific_empty_row_stays_out_of_legacy_profile_index():
    empty = ProfileRecord(
        workload_key="gpu_cnn_torch_resnet50",
        command_fingerprint="unit-empty",
        resource_kind="gpu_cnn",
        node_bucket="jtl110gpu:gpu_3080ti_12gb_dual",
        profile=3,
        unit="step",
        total_units=100.0,
        aggregate_rate=30.0,
        per_task_rates=(10.0, 10.0, 10.0),
        source="unit:empty",
        workload_env="cnn",
        resource_state="empty",
        eta_source="tqdm/progress",
        stable_rate_ready=True,
    )
    mixed = ProfileRecord(
        workload_key="gpu_cnn_torch_resnet50",
        command_fingerprint="unit-mixed",
        resource_kind="gpu_cnn",
        node_bucket="jtl110gpu:gpu_3080ti_12gb_dual",
        profile=3,
        unit="step",
        total_units=100.0,
        aggregate_rate=6.0,
        per_task_rates=(2.0, 2.0, 2.0),
        source="unit:mixed",
        workload_env="cnn",
        resource_state="mixed_colocation",
        eta_source="tqdm/progress",
        stable_rate_ready=True,
    )
    cache = ServiceRateCache((empty, mixed))

    legacy = cache.get("gpu_cnn_torch_resnet50", 3)
    statewise = cache.get_statewise(
        "gpu_cnn_torch_resnet50",
        3,
        node_bucket="jtl110gpu:gpu_3080ti_12gb_dual",
        workload_env="cnn",
        resource_state="mixed_colocation",
        strict=True,
    )

    assert legacy is None
    assert statewise is not None
    assert statewise.resource_state == "mixed_colocation"
    assert statewise.aggregate_rate == 6.0


def test_resident_mix_specific_rows_do_not_overwrite_each_other():
    rl_mix = ProfileRecord(
        workload_key="gpu_cnn_torch_resnet50",
        command_fingerprint="unit-rl",
        resource_kind="gpu_cnn",
        node_bucket="jtl110gpu2:gpu_3080ti_12gb_dual",
        profile=1,
        unit="step",
        total_units=100.0,
        aggregate_rate=47.0,
        per_task_rates=(47.0,),
        source="unit:rl_mix",
        workload_env="cnn",
        resource_state="mixed_colocation",
        resident_mix="cnn_plus_hybrid_rl",
        eta_source="tqdm/progress",
        stable_rate_ready=True,
        hardware_class="gpu_3080ti_12gb_dual",
    )
    triple_mix = ProfileRecord(
        workload_key="gpu_cnn_torch_resnet50",
        command_fingerprint="unit-triple",
        resource_kind="gpu_cnn",
        node_bucket="jtl110gpu2:gpu_3080ti_12gb_dual",
        profile=1,
        unit="step",
        total_units=100.0,
        aggregate_rate=41.0,
        per_task_rates=(41.0,),
        source="unit:triple_mix",
        workload_env="cnn",
        resource_state="mixed_colocation",
        resident_mix="cnn_plus_llm_plus_hybrid_rl",
        eta_source="tqdm/progress",
        stable_rate_ready=True,
        hardware_class="gpu_3080ti_12gb_dual",
    )
    cache = ServiceRateCache((rl_mix, triple_mix))

    rl = cache.get_statewise(
        "gpu_cnn_torch_resnet50",
        1,
        node_bucket="jtl110gpu:gpu_3080ti_12gb_dual",
        workload_env="cnn",
        resource_state="mixed_colocation",
        resident_mix="cnn_plus_hybrid_rl",
        strict=True,
    )
    triple = cache.get_statewise(
        "gpu_cnn_torch_resnet50",
        1,
        node_bucket="jtl110gpu:gpu_3080ti_12gb_dual",
        workload_env="cnn",
        resource_state="mixed_colocation",
        resident_mix="cnn_plus_llm_plus_hybrid_rl",
        strict=True,
    )
    ambiguous = cache.get_statewise(
        "gpu_cnn_torch_resnet50",
        1,
        node_bucket="jtl110gpu:gpu_3080ti_12gb_dual",
        workload_env="cnn",
        resource_state="mixed_colocation",
        strict=True,
    )

    assert rl is not None and rl.aggregate_rate == 47.0
    assert triple is not None and triple.aggregate_rate == 41.0
    assert ambiguous is None
    assert len(cache.snapshot()["records"]) == 2


def test_live_v2_empty_node_row_stays_out_of_legacy_profile_index():
    row = ProfileRecord(
        workload_key="gpu_cnn_torch_resnet50",
        command_fingerprint="live_tqdm_service_cache_v2_noearly_20260629",
        resource_kind="gpu_cnn",
        node_bucket="jtl110gpu:gpu_3080ti_12gb_dual",
        profile=3,
        unit="step",
        total_units=100.0,
        aggregate_rate=30.0,
        per_task_rates=(10.0, 10.0, 10.0),
        source="md/experiment_artifacts/service_cache_v2_full_factorial_eta_unit.json",
        workload_env="cnn",
        resource_state="empty",
        eta_source="tqdm/progress",
        stable_rate_ready=True,
        hardware_class="gpu_3080ti_12gb_dual",
    )
    cache = ServiceRateCache((row,))

    legacy = cache.get("gpu_cnn_torch_resnet50", 3)
    statewise = cache.get_statewise(
        "gpu_cnn_torch_resnet50",
        3,
        node_bucket="jtl110gpu:gpu_3080ti_12gb_dual",
        workload_env="cnn",
        resource_state="empty",
        strict=True,
    )

    assert legacy is None
    assert statewise is not None
    assert statewise.aggregate_rate == 30.0


def test_eta_server_coverage_gate_uses_resident_mix(tmp_path):
    cache = ServiceRateCache()
    for expected in EXPECTED_ROWS:
        for profile in expected.profiles:
            cache.add(
                ProfileRecord(
                    workload_key=expected.workload_key,
                    command_fingerprint="unit-v2",
                    resource_kind="gpu" if "gpu" in expected.node_bucket else "cpu",
                    node_bucket=expected.node_bucket,
                    profile=profile,
                    unit="step",
                    total_units=100.0,
                    aggregate_rate=10.0 + profile,
                    per_task_rates=(10.0 + profile,),
                    source="service_cache_v2_unit",
                    workload_env=expected.workload_env,
                    resource_state=expected.resource_state,
                    resident_mix=expected.resident_mix,
                    eta_source="tqdm/progress",
                    stable_rate_ready=True,
                )
            )
    cache_path = tmp_path / "cache.json"
    cache.save(cache_path)

    report = build_eta_server_coverage_gate(cache_path=cache_path)

    assert report["pass"] is True
    assert report["required_missing_count"] == 0


def test_parallel_probe_plan_uses_node_level_locks(tmp_path):
    design = tmp_path / "design.csv"
    design.write_text(
        "\n".join(
            [
                "row_id,tier,quadrant,node,node_bucket,hardware_group,resource_kind,resource_state,resident_mix,workload_key,workload_env,missing_profiles,measured_profiles,boundary_closed_profiles,status,runner",
                "gpu-row,T0_core_curve,q01,jtl311linux,jtl311linux:gpu,gpu_rtx2080_8gb_dual_cpu_fast,gpu,full_loaded,same_workload_to_capacity_boundary,gpu_cnn_torch_resnet50,cnn,1,2,,pending_probe,torch",
                "cpu-row,T0_core_curve,q10,jtl311linux,jtl311linux:cpu,gpu_rtx2080_8gb_dual_cpu_fast,cpu,full_loaded,same_workload_to_capacity_boundary,cpu_heavy_local_bench,cpu,1,2,,pending_probe,cpu",
                "node007-row,T0_core_curve,q01,node007,node007:gpu,gpu_node007_4x11gb,gpu,full_loaded,same_workload_to_capacity_boundary,gpu_llm_distilgpt2,llm,1,2,,pending_probe,torch",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_parallel_probe_plan(design_csv=design, max_rows_per_lane=1, include_tiers={"T0_core_curve"})
    selected = report["selected_rows"]
    commands = [cmd for lane in report["lanes"] for cmd in lane["commands"]]

    assert {row["row_id"] for row in selected} == {"gpu-row", "node007-row"}
    assert any(row["eligible_nodes"] == "jtl311linux" for row in selected if row["row_id"] == "gpu-row")
    assert all("scheduler.py" not in cmd for cmd in commands)
    assert any("full_factorial_eta_probe_runner" in cmd for cmd in commands)


def test_parallel_probe_plan_uses_availability_audit_for_hybrid_rows(tmp_path):
    design = tmp_path / "design.csv"
    design.write_text(
        "\n".join(
            [
                "row_id,tier,quadrant,node,node_bucket,hardware_group,resource_kind,resource_state,resident_mix,workload_key,workload_env,missing_profiles,measured_profiles,boundary_closed_profiles,status,runner",
                "hybrid-row,T0_core_curve,q11,jtl311linux,jtl311linux:gpu,gpu_rtx2080_8gb_dual_cpu_fast,gpu,full_loaded,same_workload_to_capacity_boundary,hybrid_rl_resac_ant,ant,1,2,,pending_probe,torch",
                "gpu-row,T0_core_curve,q01,jtl311linux,jtl311linux:gpu,gpu_rtx2080_8gb_dual_cpu_fast,gpu,full_loaded,same_workload_to_capacity_boundary,gpu_llm_distilgpt2,llm,1,2,,pending_probe,torch",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    audit = tmp_path / "audit.json"
    audit.write_text(
        """
        {
          "nodes": [
            {
              "node": "jtl311linux",
              "parsed": {
                "availability_class": "cpu_busy_gpu_idle",
                "safe_for_gpu_only_probe": true,
                "safe_for_hybrid_probe": false,
                "safe_for_cpu_probe": false
              }
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    report = build_parallel_probe_plan(
        design_csv=design,
        max_rows_per_lane=1,
        include_tiers={"T0_core_curve"},
        availability_json=audit,
    )

    assert {row["row_id"] for row in report["selected_rows"]} == {"gpu-row"}
    assert {row["row_id"] for row in report["availability_blocked_rows"]} == {"hybrid-row"}


def test_parallel_probe_plan_does_not_launch_cpu_resident_with_core_runner(tmp_path):
    design = tmp_path / "design.csv"
    design.write_text(
        "\n".join(
            [
                "row_id,tier,quadrant,node,node_bucket,hardware_group,resource_kind,resource_state,resident_mix,workload_key,workload_env,missing_profiles,measured_profiles,boundary_closed_profiles,status,runner",
                "cpu-resident-row,T2_load_state,q10,jtl110cpu,jtl110cpu:cpu,cpu_jtl110_128c,cpu,cpu_resident,cpu_worker_resident,light_control_local,light,1,,,pending_probe,cpu",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_parallel_probe_plan(
        design_csv=design,
        max_rows_per_lane=1,
        include_tiers={"T2_load_state"},
    )

    assert report["selected_rows"] == []
    assert {row["row_id"] for row in report["runner_not_ready_rows"]} == {"cpu-resident-row"}


def test_nonmonotone_unstable_profile_is_not_capacity_boundary(tmp_path):
    p3 = tmp_path / "profile_3_per_gpu_summary.json"
    p4 = tmp_path / "profile_4_per_gpu_summary.json"
    p3.write_text(
        """
        {
          "measurement_valid": false,
          "running_count": 6,
          "stable_rate_ready_count": 5,
          "aggregate_stable_rate_unit_s": 600.0,
          "stable_rates_unit_s": [100, 100, 100, 100, 100],
          "status_counts": {"sampled": 6}
        }
        """,
        encoding="utf-8",
    )
    p4.write_text(
        """
        {
          "measurement_valid": true,
          "running_count": 8,
          "stable_rate_ready_count": 8,
          "aggregate_stable_rate_unit_s": 800.0,
          "stable_rates_unit_s": [100, 100, 100, 100, 100, 100, 100, 100],
          "status_counts": {"sampled": 8}
        }
        """,
        encoding="utf-8",
    )
    result = {
        "workload_key": "gpu_llm_distilgpt2",
        "family": "q01_low_cpu_high_gpu",
        "node_bucket": "jtl110gpu:gpu_3080ti_12gb_dual",
        "workload_env": "llm",
        "resource_state": "empty",
        "hardware_group": "gpu_3080ti_12gb_dual",
        "profile_rows": [
            {"profile": 3, "measurement_valid": False, "summary_path": str(p3)},
            {"profile": 4, "measurement_valid": True, "summary_path": str(p4)},
        ],
    }
    cache = ServiceRateCache()

    _add_result_to_cache(cache, result)

    snapshot = cache.snapshot()
    assert [row["profile"] for row in snapshot["records"]] == [4]
    assert snapshot["capacity_boundaries"] == []


def test_zero_rate_environment_failure_is_not_capacity_boundary(tmp_path):
    p1 = tmp_path / "profile_1_per_gpu_summary.json"
    p1.write_text(
        """
        {
          "measurement_valid": false,
          "running_count": 2,
          "stable_rate_ready_count": 0,
          "aggregate_stable_rate_unit_s": 0.0,
          "stable_rates_unit_s": [],
          "status_counts": {"failed": 2}
        }
        """,
        encoding="utf-8",
    )
    result = {
        "workload_key": "gpu_heavy_jax_matmul",
        "family": "q01_low_cpu_high_gpu",
        "node_bucket": "jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast",
        "workload_env": "gpu_matmul",
        "resource_state": "empty",
        "hardware_group": "gpu_rtx2080_8gb_dual_cpu_fast",
        "profile_rows": [
            {"profile": 1, "measurement_valid": False, "summary_path": str(p1)},
        ],
    }
    cache = ServiceRateCache()

    _add_result_to_cache(cache, result)

    assert cache.snapshot() == {"records": [], "capacity_boundaries": []}


def test_cpu_progress_parser_excludes_done_average_rate():
    text = "\n".join(
        [
            "CPU_PARALLEL_PROGRESS Step 72/80 rate=29.2 step/s total_rate=25.4 step/s ETA 0.3s",
            "CPU_PARALLEL_PROGRESS Step 74/80 rate=29.3 step/s total_rate=25.5 step/s ETA 0.2s",
            "CPU_PARALLEL_PROGRESS Step 76/80 rate=29.2 step/s total_rate=25.6 step/s ETA 0.2s",
            "CPU_PARALLEL_PROGRESS Step 78/80 rate=29.2 step/s total_rate=25.7 step/s ETA 0.1s",
            "CPU_PARALLEL_PROGRESS Step 80/80 rate=29.1 step/s total_rate=25.8 step/s ETA 0.0s",
            "CPU_PARALLEL_DONE label=x steps=80 elapsed=3.2s rate=24.7 step/s checksum=1",
        ]
    )

    rates = _progress_rates(text)
    stable = _stable_tail(rates, min_samples=5)

    assert rates == [29.2, 29.3, 29.2, 29.2, 29.1]
    assert stable["ready"]
    assert stable["mean_rate"] > 29.0


def test_cpu_parallel_parser_can_use_cumulative_total_rate_for_eta():
    text = "\n".join(
        [
            "CPU_PARALLEL_PROGRESS Step 152/160 rate=24.0 step/s total_rate=17.947 step/s ETA 0.4s",
            "CPU_PARALLEL_PROGRESS Step 154/160 rate=17.6 step/s total_rate=17.943 step/s ETA 0.3s",
            "CPU_PARALLEL_PROGRESS Step 156/160 rate=17.5 step/s total_rate=17.937 step/s ETA 0.2s",
            "CPU_PARALLEL_PROGRESS Step 158/160 rate=17.1 step/s total_rate=17.926 step/s ETA 0.1s",
            "CPU_PARALLEL_PROGRESS Step 160/160 rate=17.1 step/s total_rate=17.914 step/s ETA 0.0s",
            "CPU_PARALLEL_DONE label=x steps=160 elapsed=9.0s rate=17.75 step/s checksum=1",
        ]
    )

    window_rates = _progress_rates(text)
    total_rates = _progress_rates(text, prefer_total=True)
    stable = _stable_tail(total_rates, min_samples=5)

    assert window_rates[0] == 24.0
    assert total_rates == [17.947, 17.943, 17.937, 17.926, 17.914]
    assert stable["ready"]


def test_marginal_runner_supports_non_cnn_add_one_commands():
    assert "torch_llm_progress_benchmark.py" in _benchmark_command("llm_add_light", label="x", steps=8)
    assert "gpu_progress_benchmark.py" in _benchmark_command("matmul_add_light", label="x", steps=8)
    assert _resac_env_for_kind("rl_resac_halfcheetah_add_light") == "HalfCheetah-v2"
    assert _resac_env_for_kind("rl_resac_walker2d_resident_light") == "Walker2d-v2"
    assert "algo bapr" in _wrapped_command(
        "bapr_ant_add_light",
        label="x",
        total=8,
        terminate_on_stable=True,
        gpu_prefix="CUDA_VISIBLE_DEVICES=0 ",
    )


def test_marginal_from_design_preserves_resident_mix(tmp_path):
    design = tmp_path / "design.csv"
    design.write_text(
        "\n".join(
            [
                "row_id,tier,quadrant,node,node_bucket,hardware_group,resource_kind,resource_state,resident_mix,workload_key,workload_env,missing_profiles,measured_profiles,boundary_closed_profiles,status,runner,family",
                "row1,T2_load_state,q01,jtl110gpu,jtl110gpu:gpu_3080ti_12gb_dual,gpu_3080ti_12gb_dual,gpu,mixed_colocation,cnn_plus_llm_plus_hybrid_rl,gpu_cnn_torch_resnet50,cnn,1,,,pending_probe,torch,cnn",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_live_marginal_from_design(
        design_csv=design,
        run_id="unit",
        allow_launch=False,
        assigned_node="jtl110gpu2",
        hardware_groups=["gpu_3080ti_12gb_dual"],
        max_rows=1,
    )

    assert report["selected_scenario_count"] == 1
    row = report["rows"][0]
    assert row["node"] == "jtl110gpu2"
    assert row["resident_mix"] == "cnn_plus_llm_plus_hybrid_rl"
    assert row["background_kind"] == "llm_resident_light"
    assert row["add_kind"] == "cnn_add_light"


def test_bind_service_uses_description_env_and_node_bucket():
    cache = ServiceRateCache(
        (
            _record("node007:gpu_node007_4x12gb", "halfcheetah", 1.0),
            _record("jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast", "halfcheetah", 3.0),
        )
    )
    task = {
        "id": "resac_halfcheetah",
        "description": "RE-SAC HalfCheetah-v2 controlled benchmark",
        "est_vram_mb": 2048,
    }
    node = {"name": "jtl311linux", "node_bucket": "jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast"}
    gpu = {"idx": 0, "total_mb": 8192, "used_mb": 0, "free_mb": 8192, "running_task_count": 0}
    features = gpu_candidate_features(task, node, gpu)
    binding = bind_service(task, features, cache=cache)

    assert infer_workload_env(task) == "halfcheetah"
    assert infer_workload_key(task, cache) == "hybrid_rl_resac_halfcheetah"
    assert binding.certified
    assert binding.node_bucket.startswith("jtl311linux")
    assert binding.lower_service == 3.0


def test_bind_service_fails_closed_when_only_legacy_profile_matches():
    base_record = ProfileRecord(
        workload_key="hybrid_rl_resac_halfcheetah",
        command_fingerprint="legacy_halfcheetah_curve",
        resource_kind="hybrid_rl",
        node_bucket="node007:gpu_node007_4x12gb",
        profile=1,
        unit="iter",
        total_units=100.0,
        aggregate_rate=1.5,
        per_task_rates=(1.5,),
        source="unit:legacy",
    )
    cache = ServiceRateCache(
        (
            base_record,
        )
    )
    task = {
        "id": "resac_halfcheetah_unit",
        "description": "RE-SAC HalfCheetah-v2 controlled benchmark",
        "est_vram_mb": 512,
    }
    node = {"name": "unit-node", "node_bucket": "unit-node"}
    gpu = {"idx": 0, "total_mb": 12000, "used_mb": 100, "free_mb": 11900, "running_task_count": 0}
    features = gpu_candidate_features(task, node, gpu)

    binding = bind_service(task, features, cache=cache)

    assert binding.certified is False
    assert binding.reason == "exact_statewise_profile_missing"
    assert binding.lower_service == 0.0
    assert binding.lookup_status == "missing"
    assert binding.node_bucket == "unit-node"


def test_global_dispatch_selects_beneficial_migration_and_blocks_unsafe_rows():
    task = {
        "id": "controlled_cnn",
        "controlled_benchmark": True,
        "checkpoint_verified": True,
        "resume_verified": True,
    }
    beneficial = build_migration_action_row(
        task=task,
        workload_key="gpu_cnn_torch_resnet50",
        from_node="node007",
        to_node="jtl311linux",
        current_rate=1.0,
        target_lower_service=2.0,
        remaining_work=100.0,
        progress_fraction=0.5,
        cost=MigrationCost(checkpoint_flush_s=1, sync_s=1, resume_warmup_s=1),
    )
    expensive = build_migration_action_row(
        task={**task, "id": "controlled_cnn_expensive"},
        workload_key="gpu_cnn_torch_resnet50",
        from_node="node007",
        to_node="jtl311linux",
        current_rate=1.0,
        target_lower_service=2.0,
        remaining_work=10.0,
        progress_fraction=0.75,
        cost=MigrationCost(checkpoint_flush_s=20, sync_s=20, resume_warmup_s=20),
    )
    unsafe = build_migration_action_row(
        task={"id": "user_job", "production_running": True},
        workload_key="gpu_cnn_torch_resnet50",
        from_node="node007",
        to_node="jtl311linux",
        current_rate=1.0,
        target_lower_service=2.0,
        remaining_work=100.0,
        progress_fraction=0.25,
        cost=MigrationCost(checkpoint_flush_s=1),
    )

    result = select_global_action(
        [expensive, unsafe, beneficial],
        {"gpu_cnn_torch_resnet50": 20.0},
        max_batch_size=1,
    )

    assert beneficial["theorem_ready"] is True
    assert expensive["theorem_ready"] is False
    assert unsafe["controlled_migration_ready"] is False
    assert result.selected_action_ids == (beneficial["action_id"],)


def test_migration_and_or_generalization_gates_pass():
    migration = build_migration_cost_gate()
    generalization = build_or_generalization_gate()
    corners = build_corner_case_matrix()

    assert migration["pass"] is True
    assert migration["no_touch_safety_ready"] is True
    assert generalization["pass"] is True
    assert corners["pass"] is True


def test_live_checkpoint_gate_accepts_extra_cpu_pairs_dry_run():
    from algorithm.experiments.live_checkpoint_migration_cost_gate import build_live_checkpoint_migration_cost_gate

    report = build_live_checkpoint_migration_cost_gate(
        allow_launch=False,
        specs=("cpu_freqduet_surrogate_checkpoint_node005_to_node006",),
        extra_cpu_pairs=("node005:node006",),
        run_id="unit_extra_cpu_pair",
    )

    assert report["row_count"] == 3
    assert {row["source_node"] for row in report["rows"]} == {"node005"}
    assert {row["dest_node"] for row in report["rows"]} == {"node006"}
