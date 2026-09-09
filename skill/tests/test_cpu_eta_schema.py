import json
from pathlib import Path

import pytest

from algorithm.experiments import full_factorial_eta_design as eta_design
from algorithm.experiments import full_factorial_cpu_eta_probe_runner as cpu_runner
from simulation.service_cache import (
    ProfileRecord,
    ServiceRateCache,
    records_from_summary_file,
)


def _parallel_output(*, done: bool) -> str:
    lines = [
        (
            f"CPU_PARALLEL_PROGRESS Step {step}/160 rate=10 step/s "
            "total_rate=10 step/s ETA 1.0s"
        )
        for step in range(2, 18, 2)
    ]
    if done:
        lines.append(
            "CPU_PARALLEL_DONE label=unit steps=160 elapsed=16s "
            "rate=10 step/s checksum=1"
        )
    return "\n".join(lines)


def _light_output(colocation_count: int) -> str:
    sections = []
    for child in range(colocation_count):
        lines = [
            f"__SCHEDULEURM_CPU_LOG_BEGIN__ {child}.log",
            *[
                (
                    f"Step {step}/160 dt=0.1s elapsed={step / 10:.1f}s "
                    "rate=10 step/s ETA 1.0s"
                )
                for step in range(1, 9)
            ],
            (
                f"CPU_BENCH_DONE label=unit_{child} mode=light steps=160 "
                "elapsed=16s rate=10 step/s checksum=1"
            ),
            f"__SCHEDULEURM_CPU_LOG_END__ {child}.log",
        ]
        sections.extend(lines)
    return "\n".join(sections)


def _measure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    profile: int,
    use_parallel: bool,
    returncode: int,
    output: str,
) -> tuple[dict, dict, str]:
    raw_dir = tmp_path / "raw"
    report_dir = tmp_path / "reports"
    raw_dir.mkdir()
    report_dir.mkdir()
    launched_command = ""

    def fake_capture(
        node: str,
        command: str,
        prefix: Path,
        *,
        timeout_s: int,
    ) -> tuple[int, str, str]:
        nonlocal launched_command
        launched_command = command
        return returncode, output, ""

    monkeypatch.setattr(cpu_runner, "_windows_node", lambda node: False)
    monkeypatch.setattr(cpu_runner, "_run_remote_capture", fake_capture)
    row = cpu_runner._measure_profile(
        node="unit-node",
        run_id="unit-run",
        raw_dir=raw_dir,
        report_dir=report_dir,
        profile=profile,
        workload_key="cpu_heavy_local_bench",
        config={
            "steps": 160,
            "work_items": 100,
            "timeout_s": 30,
            "mode": "cpu" if use_parallel else "light",
            "use_parallel": use_parallel,
            "log_interval": 2,
        },
    )
    summary = json.loads(Path(row["summary_path"]).read_text(encoding="utf-8"))
    return row, summary, launched_command


def _cache_result(profile_rows: list[dict], *, use_parallel: bool) -> dict:
    return {
        "workload_key": "cpu_heavy_local_bench",
        "family": "cpu",
        "node_bucket": "node001:cpu_hpc_192c",
        "workload_env": "cpu",
        "resource_state": "empty",
        "resident_mix": "",
        "hardware_group": "cpu_hpc_192c",
        "config": {"steps": 160, "use_parallel": use_parallel},
        "profile_rows": profile_rows,
    }


def test_parallel_workers_are_one_colocated_task_with_real_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row, summary, command = _measure(
        tmp_path,
        monkeypatch,
        profile=16,
        use_parallel=True,
        returncode=0,
        output=_parallel_output(done=True),
    )

    assert "--workers 16" in command
    assert summary["profile"] == 16
    assert summary["allocation_workers"] == 16
    assert summary["colocation_count"] == 1
    assert summary["running_count"] == 1
    assert summary["steps"] == 160
    assert summary["total_units"] == 160
    assert summary["steady_state_rate_ready"] is True
    assert summary["natural_completion_observed"] is True
    assert summary["completion_model_ready"] is False

    cache = ServiceRateCache()
    cpu_runner._add_result_to_cache(
        cache, _cache_result([row], use_parallel=True)
    )
    records = cache.snapshot()["records"]
    assert len(records) == 1
    assert records[0]["profile"] == 1
    assert records[0]["allocation_workers"] == 16
    assert records[0]["colocation_count"] == 1
    assert records[0]["total_units"] == 160
    assert records[0]["stable_rate_ready"] is True
    assert records[0]["completion_model_ready"] is False
    assert cache.get("cpu_heavy_local_bench", 1) is None
    assert cache.get("cpu_heavy_local_bench", 16) is None

    exact = cache.get_statewise(
        "cpu_heavy_local_bench",
        allocation_workers=16,
        colocation_count=1,
        node_bucket="node001:cpu_hpc_192c",
        workload_env="cpu",
        resource_state="empty",
        strict=True,
    )
    assert exact is not None
    assert exact.total_units == 160


def test_stable_timeout_is_valid_service_but_not_natural_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row, summary, _ = _measure(
        tmp_path,
        monkeypatch,
        profile=8,
        use_parallel=True,
        returncode=124,
        output=_parallel_output(done=False),
    )

    assert row["measurement_valid"] is True
    assert summary["returncode_valid_count"] == 0
    assert summary["returncode_accepted_count"] == 1
    assert summary["steady_state_rate_ready"] is True
    assert summary["natural_completion_observed"] is False
    assert summary["natural_completion_count"] == 0
    assert summary["completion_model_ready"] is False
    assert summary["require_stable_rate"] is True
    assert summary["terminate_on_stable"] is False


def test_light_profile_remains_colocation_not_worker_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, summary, command = _measure(
        tmp_path,
        monkeypatch,
        profile=3,
        use_parallel=False,
        returncode=1,
        output=_light_output(3),
    )

    assert "seq 0 2" in command
    assert summary["allocation_workers"] == 1
    assert summary["colocation_count"] == 3
    assert summary["running_count"] == 3
    assert summary["stable_rate_ready_count"] == 3
    assert summary["natural_completion_count"] == 3
    assert summary["natural_completion_observed"] is True
    assert summary["steps"] == 160


def test_windows_parallel_summary_uses_the_same_cpu_dimensions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = tmp_path / "raw"
    report_dir = tmp_path / "reports"
    raw_dir.mkdir()
    report_dir.mkdir()
    launched_ps = ""

    def fake_windows_capture(
        scheduler: object,
        node: str,
        ps: str,
        prefix: Path,
        *,
        timeout_s: int,
    ) -> tuple[int, str, str]:
        nonlocal launched_ps
        launched_ps = ps
        return 0, _parallel_output(done=True), ""

    monkeypatch.setattr(cpu_runner, "_windows_node", lambda node: True)
    monkeypatch.setattr(cpu_runner, "_scheduler_module", lambda: object())
    monkeypatch.setattr(cpu_runner, "_windows_python", lambda node: "python")
    monkeypatch.setattr(
        cpu_runner, "_run_windows_ps_capture", fake_windows_capture
    )

    row = cpu_runner._measure_profile(
        node="windows-unit",
        run_id="windows-unit-run",
        raw_dir=raw_dir,
        report_dir=report_dir,
        profile=12,
        workload_key="cpu_heavy_local_bench",
        config={
            "steps": 160,
            "work_items": 100,
            "timeout_s": 30,
            "mode": "cpu",
            "use_parallel": True,
            "log_interval": 2,
        },
    )
    summary = json.loads(Path(row["summary_path"]).read_text(encoding="utf-8"))

    assert "--workers 12" in launched_ps
    assert summary["allocation_workers"] == 12
    assert summary["colocation_count"] == 1
    assert summary["running_count"] == 1
    assert summary["steps"] == 160
    assert summary["natural_completion_observed"] is True


def test_old_cpu_summary_uses_runner_config_to_recover_shape_and_steps(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "profile_32_per_resource_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "profile": 32,
                "measurement_valid": True,
                "aggregate_stable_rate_unit_s": 12.5,
                "stable_rates_unit_s": [12.5],
                "all_stable_rate_ready": True,
                "stable_rate_ready_count": 1,
            }
        ),
        encoding="utf-8",
    )
    row = {
        "profile": 32,
        "measurement_valid": True,
        "summary_path": str(summary_path),
    }
    cache = ServiceRateCache()

    cpu_runner._add_result_to_cache(
        cache, _cache_result([row], use_parallel=True)
    )

    record = cache.get_statewise(
        "cpu_heavy_local_bench",
        allocation_workers=32,
        colocation_count=1,
        node_bucket="node001:cpu_hpc_192c",
        workload_env="cpu",
        resource_state="empty",
        strict=True,
    )
    assert record is not None
    assert record.profile == 1
    assert record.total_units == 160
    assert cache.get("cpu_heavy_local_bench", 32) is None


def test_missing_cpu_steps_never_falls_back_to_one_unit(tmp_path: Path) -> None:
    summary_path = tmp_path / "profile_4_per_resource_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "profile": 4,
                "allocation_workers": 4,
                "colocation_count": 1,
                "measurement_valid": True,
                "aggregate_stable_rate_unit_s": 10,
                "stable_rates_unit_s": [10],
            }
        ),
        encoding="utf-8",
    )
    result = _cache_result(
        [
            {
                "profile": 4,
                "measurement_valid": True,
                "summary_path": str(summary_path),
            }
        ],
        use_parallel=True,
    )
    result["config"] = {"use_parallel": True}
    cache = ServiceRateCache()

    cpu_runner._add_result_to_cache(cache, result)

    assert cache.snapshot()["records"] == []


def test_old_snapshot_defaults_to_one_worker_per_legacy_profile() -> None:
    record = ProfileRecord.from_snapshot(
        {
            "workload_key": "legacy",
            "command_fingerprint": "legacy-v1",
            "resource_kind": "cpu",
            "node_bucket": "",
            "profile": 4,
            "unit": "step",
            "total_units": 20,
            "aggregate_rate": 8,
            "per_task_rates": [2, 2, 2, 2],
            "source": "legacy.json",
        }
    )

    assert record.profile == 4
    assert record.allocation_workers == 1
    assert record.colocation_count == 4
    restored = ProfileRecord.from_snapshot(record.snapshot())
    assert restored == record


def test_generic_summary_loader_prefers_explicit_cpu_dimensions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "profile_16_per_resource_summary.json"
    path.write_text(
        json.dumps(
            {
                "phase": "profile_16_per_resource",
                "profile": 16,
                "allocation_workers": 16,
                "colocation_count": 1,
                "measurement_valid": True,
                "placement_valid": True,
                "running_count": 1,
                "returncode_valid_count": 1,
                "rate_units": ["step"],
                "aggregate_stable_rate_unit_s": 10,
                "stable_rates_unit_s": [10],
                "all_stable_rate_ready": True,
                "stable_rate_ready_count": 1,
            }
        ),
        encoding="utf-8",
    )

    record = records_from_summary_file(
        path,
        workload_key="cpu",
        command_fingerprint="unit",
        resource_kind="cpu",
        total_units=160,
    )[0]

    assert record.profile == 1
    assert record.allocation_workers == 16
    assert record.colocation_count == 1
    assert record.mean_rate == 10


def test_worker_allocations_do_not_overwrite_each_other_or_legacy_index() -> None:
    def record(workers: int, rate: float) -> ProfileRecord:
        return ProfileRecord(
            workload_key="cpu",
            command_fingerprint="cpu_allocation_schema_v2",
            resource_kind="cpu",
            node_bucket="node001:cpu_hpc_192c",
            profile=1,
            unit="step",
            total_units=160,
            aggregate_rate=rate,
            per_task_rates=(rate,),
            source=f"workers-{workers}.json",
            allocation_workers=workers,
            colocation_count=1,
            workload_env="cpu",
            resource_state="empty",
            eta_source="tqdm/progress",
            stable_rate_ready=True,
        )

    cache = ServiceRateCache((record(8, 20), record(16, 15)))

    assert len(cache.snapshot()["records"]) == 2
    assert cache.available_workloads() == ["cpu"]
    assert cache.get("cpu", 1) is None
    assert cache.get_statewise(
        "cpu",
        allocation_workers=8,
        colocation_count=1,
        node_bucket="node001:cpu_hpc_192c",
        workload_env="cpu",
        resource_state="empty",
        strict=True,
    ).aggregate_rate == 20
    assert cache.get_statewise(
        "cpu",
        allocation_workers=16,
        colocation_count=1,
        node_bucket="node001:cpu_hpc_192c",
        workload_env="cpu",
        resource_state="empty",
        strict=True,
    ).aggregate_rate == 15

    exact = cache.lookup_statewise_exact(
        "cpu",
        allocation_workers=8,
        colocation_count=1,
        node_bucket="node001:cpu_hpc_192c",
        workload_env="cpu",
        resource_state="empty",
    )
    wrong_workers = cache.lookup_statewise_exact(
        "cpu",
        allocation_workers=12,
        colocation_count=1,
        node_bucket="node001:cpu_hpc_192c",
        workload_env="cpu",
        resource_state="empty",
    )
    wrong_colocation = cache.lookup_statewise_exact(
        "cpu",
        allocation_workers=8,
        colocation_count=2,
        node_bucket="node001:cpu_hpc_192c",
        workload_env="cpu",
        resource_state="empty",
    )

    assert exact.status == "exact"
    assert exact.record is not None
    assert exact.record.aggregate_rate == 20
    assert exact.request_tuple[-3:-1] == (8, 1)
    assert wrong_workers.status == "missing"
    assert wrong_colocation.status == "missing"


def test_exact_scoped_boundary_stays_on_its_allocation_axis() -> None:
    def record(
        workers: int,
        *,
        rate: float,
        capacity_boundary: bool = False,
    ) -> ProfileRecord:
        return ProfileRecord(
            workload_key="cpu",
            command_fingerprint="cpu_allocation_schema_v2",
            resource_kind="cpu",
            node_bucket="node001:cpu_hpc_192c",
            profile=1,
            unit="step",
            total_units=160,
            aggregate_rate=rate,
            per_task_rates=() if capacity_boundary else (rate,),
            source=f"workers-{workers}.json",
            capacity_boundary=capacity_boundary,
            allocation_workers=workers,
            colocation_count=1,
            workload_env="cpu",
            resource_state="empty",
            eta_source="tqdm/progress",
            stable_rate_ready=not capacity_boundary,
        )

    boundary = record(16, rate=0, capacity_boundary=True)
    positive_above_boundary = record(32, rate=30)
    cache = ServiceRateCache((boundary, positive_above_boundary))

    blocked = cache.lookup_statewise_exact(
        "cpu",
        allocation_workers=32,
        colocation_count=1,
        node_bucket="node001:cpu_hpc_192c",
        workload_env="cpu",
        resource_state="empty",
    )
    different_colocation_axis = cache.lookup_statewise_exact(
        "cpu",
        allocation_workers=32,
        colocation_count=2,
        node_bucket="node001:cpu_hpc_192c",
        workload_env="cpu",
        resource_state="empty",
    )

    assert blocked.status == "scoped_boundary"
    assert blocked.boundary is boundary
    assert blocked.record is None
    assert different_colocation_axis.status == "missing"
    assert different_colocation_axis.boundary is None


def test_full_factorial_design_caps_jtl311_cpu_ladders_to_real_cores() -> None:
    hardware = next(
        row for row in eta_design.HARDWARE_TYPES
        if row.primary_node == "jtl311linux"
    )
    assert hardware.cpu_cores == 8

    rows = eta_design._eta_rows(ServiceRateCache())
    jtl_cpu_rows = [
        row for row in rows
        if row["node"] == "jtl311linux" and row["resource_kind"] == "cpu"
    ]
    assert jtl_cpu_rows
    assert all(
        max(int(profile) for profile in row["profile_ladder"].split(",")) <= 8
        for row in jtl_cpu_rows
    )

    synthetic = next(
        row for row in jtl_cpu_rows
        if row["workload_key"] == "cpu_heavy_local_bench"
        and row["resource_state"] == "empty"
    )
    assert synthetic["profile_axis"] == "allocation_workers"
    assert synthetic["profile_ladder"] == "1,2,4,8"
    assert synthetic["allocation_workers_ladder"] == "1,2,4,8"
    assert synthetic["colocation_count_ladder"] == "1,1,1,1"


def test_full_factorial_native_cpu_profiles_are_colocation_counts() -> None:
    rows = eta_design._eta_rows(ServiceRateCache())

    for workload_key in (
        "light_control_local",
        "freqduet_cpu_ablation_c17_32",
        "freqduet_cpu_native",
        "sumo_eval_cpu",
        "sumo_eval_cpu_native",
    ):
        workload = next(
            row for row in eta_design.WORKLOAD_TYPES
            if row.workload_key == workload_key
        )
        assert workload.profile_axis == "colocation_count"
        assert eta_design._profile_dimensions(workload, 8) == (1, 8)

        design_row = next(
            row for row in rows
            if row["node"] == "node001"
            and row["workload_key"] == workload_key
            and row["resource_state"] == "empty"
        )
        profiles = design_row["profile_ladder"].split(",")
        assert design_row["profile_axis"] == "colocation_count"
        assert design_row["allocation_workers_ladder"].split(",") == [
            "1"
        ] * len(profiles)
        assert design_row["colocation_count_ladder"].split(",") == profiles


def test_native_completion_factorial_keeps_p1_and_p2_p4_states_separate() -> None:
    records = []
    for workload_key, workload_env in (
        ("freqduet_cpu_native", "freqduet"),
        ("sumo_eval_cpu_native", "sumo"),
    ):
        for profile, resource_state in (
            (1, "empty"),
            (2, "controlled_colocation"),
            (4, "controlled_colocation"),
        ):
            total_units = 6.0 * profile
            records.append(
                ProfileRecord(
                    workload_key=workload_key,
                    command_fingerprint="native-completion-unit",
                    resource_kind="cpu",
                    node_bucket="node001:cpu_hpc_192c",
                    profile=profile,
                    allocation_workers=1,
                    colocation_count=profile,
                    unit="episode",
                    total_units=total_units,
                    aggregate_rate=0.1 * profile,
                    per_task_rates=tuple(0.1 for _ in range(profile)),
                    source="unit",
                    workload_env=workload_env,
                    resource_state=resource_state,
                    eta_source="task_native_progress",
                    stable_rate_ready=True,
                    hardware_class="cpu_hpc_192c",
                    completion_model_ready=True,
                    completion_model_sample_count=13,
                    startup_overhead_s=1.0,
                    completion_unit_s=10.0,
                    finalization_overhead_s=1.0,
                    completion_total_wall_s=2.0 + total_units * 10.0,
                )
            )

    rows = [
        row
        for row in eta_design._eta_rows(ServiceRateCache(records))
        if row["node"] == "node001"
        and row["workload_key"]
        in {"freqduet_cpu_native", "sumo_eval_cpu_native"}
        and row["resource_state"] != "unknown_env"
    ]

    assert len(rows) == 4
    assert all(row["eta_status"] == "eta_measured" for row in rows)
    assert {
        (row["workload_key"], row["resource_state"], row["profile_ladder"])
        for row in rows
    } == {
        ("freqduet_cpu_native", "empty", "1"),
        ("freqduet_cpu_native", "controlled_colocation", "2,4"),
        ("sumo_eval_cpu_native", "empty", "1"),
        ("sumo_eval_cpu_native", "controlled_colocation", "2,4"),
    }

    old_key = next(
        row
        for row in eta_design._eta_rows(ServiceRateCache(records))
        if row["node"] == "node001"
        and row["workload_key"] == "freqduet_cpu_ablation_c17_32"
        and row["resource_state"] == "empty"
    )
    assert old_key["eta_status"] == "eta_pending_probe"


def test_full_factorial_synthetic_cpu_profiles_are_worker_allocations() -> None:
    rows = eta_design._eta_rows(ServiceRateCache())
    design_row = next(
        row for row in rows
        if row["node"] == "node001"
        and row["workload_key"] == "cpu_heavy_local_bench"
        and row["resource_state"] == "empty"
    )
    profiles = design_row["profile_ladder"].split(",")

    assert design_row["profile_axis"] == "allocation_workers"
    assert design_row["allocation_workers_ladder"].split(",") == profiles
    assert design_row["colocation_count_ladder"].split(",") == [
        "1"
    ] * len(profiles)


def test_cpu_design_declares_exact_controlled_half_and_full_residents() -> None:
    rows = eta_design._eta_rows(ServiceRateCache())
    selected = {
        (row["resource_state"], row["resident_mix"]): row
        for row in rows
        if row["node"] == "node001"
        and row["workload_key"] == "cpu_heavy_local_bench"
        and row["resident_mix"].startswith("controlled_resident_workers=")
    }

    assert set(selected) == {
        ("half_loaded", "controlled_resident_workers=96"),
        ("full_loaded", "controlled_resident_workers=180"),
    }
    assert all(
        row["profile_axis"] == "allocation_workers"
        for row in selected.values()
    )
    assert all(
        "natural_completion" in row["stable_gate"]
        for row in selected.values()
    )
    assert not any(
        row["resident_mix"].startswith("controlled_resident_workers=")
        and row["node"] == "jtl110cpu"
        for row in rows
    )


def test_factorial_design_separates_service_from_completion_eta() -> None:
    service_only = ProfileRecord(
        workload_key="cpu_heavy_local_bench",
        command_fingerprint="steady-only",
        resource_kind="cpu",
        node_bucket="node001:cpu_hpc_192c",
        profile=1,
        allocation_workers=1,
        colocation_count=1,
        unit="step",
        total_units=160.0,
        aggregate_rate=10.0,
        per_task_rates=(10.0,),
        source="unit",
        workload_env="cpu",
        resource_state="empty",
        eta_source="tqdm/progress",
        stable_rate_ready=True,
        completion_model_ready=False,
    )

    row = next(
        row
        for row in eta_design._eta_rows(ServiceRateCache((service_only,)))
        if row["node"] == "node001"
        and row["workload_key"] == "cpu_heavy_local_bench"
        and row["resource_state"] == "empty"
    )

    assert "1" in row["measured_profiles"].split(",")
    assert "1" in row["service_only_profiles"].split(",")
    assert "1" not in row["eta_measured_profiles"].split(",")


def test_cpu_probe_runner_fails_closed_on_design_profile_axis_mismatch() -> None:
    row = {
        "row_id": "freqduet-native-colocation",
        "node": "node001",
        "node_bucket": "node001:cpu_hpc_192c",
        "hardware_group": "cpu_hpc_192c",
        "workload_key": "freqduet_cpu_ablation_c17_32",
        "workload_env": "freqduet",
        "family": "freqduet_cpu",
        "resource_state": "empty",
        "resident_mix": "none",
        "missing_profiles": "2,4",
        "profile_axis": "colocation_count",
    }

    manifest = cpu_runner._manifest(
        row,
        run_prefix="unit",
        max_profiles=0,
    )

    assert manifest["profile_axis"] == "allocation_workers"
    assert manifest["design_profile_axis"] == "colocation_count"
    assert manifest["profile_axis_match"] is False
    assert manifest["launch_ready"] is False
    assert (
        manifest["pending_reason"]
        == "profile_axis_mismatch_requires_native_colocation_runner"
    )


def test_design_measurement_status_does_not_cross_cpu_axes_or_states() -> None:
    hardware = next(
        row for row in eta_design.HARDWARE_TYPES
        if row.primary_node == "node001"
    )
    empty = next(
        row for row in eta_design.RESIDENT_SCENARIOS
        if row.resource_state == "empty"
    )
    synthetic = next(
        row for row in eta_design.WORKLOAD_TYPES
        if row.workload_key == "cpu_heavy_local_bench"
    )
    legacy_colocation = ProfileRecord.from_snapshot(
        {
            "workload_key": synthetic.workload_key,
            "command_fingerprint": "legacy-profile-index",
            "resource_kind": "cpu",
            "node_bucket": hardware.node_bucket,
            "profile": 8,
            "unit": "step",
            "total_units": 160,
            "aggregate_rate": 8,
            "per_task_rates": [1] * 8,
            "source": "legacy-profile-8.json",
            "workload_env": synthetic.workload_env,
            "resource_state": "empty",
            "stable_rate_ready": True,
        }
    )
    wrong_state_workers = ProfileRecord(
        workload_key=synthetic.workload_key,
        command_fingerprint="explicit-cpu-dimensions",
        resource_kind="cpu",
        node_bucket=hardware.node_bucket,
        profile=1,
        unit="step",
        total_units=160,
        aggregate_rate=80,
        per_task_rates=(80,),
        source="workers-8-full-loaded.json",
        allocation_workers=8,
        colocation_count=1,
        workload_env=synthetic.workload_env,
        resource_state="full_loaded",
        stable_rate_ready=True,
    )

    status = eta_design._measurement_status(
        ServiceRateCache((legacy_colocation, wrong_state_workers)),
        synthetic,
        hardware,
        empty,
        (8,),
        boundary_map={},
    )
    assert status == ("pending_probe", [], [8], [])

    exact_workers = ProfileRecord(
        workload_key=synthetic.workload_key,
        command_fingerprint="explicit-cpu-dimensions",
        resource_kind="cpu",
        node_bucket=hardware.node_bucket,
        profile=1,
        unit="step",
        total_units=160,
        aggregate_rate=80,
        per_task_rates=(80,),
        source="workers-8-empty.json",
        allocation_workers=8,
        colocation_count=1,
        workload_env=synthetic.workload_env,
        resource_state="empty",
        stable_rate_ready=True,
    )
    status = eta_design._measurement_status(
        ServiceRateCache((legacy_colocation, exact_workers)),
        synthetic,
        hardware,
        empty,
        (8,),
        boundary_map={},
    )
    assert status == ("measured", [8], [], [])

    freqduet = next(
        row for row in eta_design.WORKLOAD_TYPES
        if row.workload_key == "freqduet_cpu_ablation_c17_32"
    )
    legacy_freqduet = ProfileRecord.from_snapshot(
        {
            "workload_key": freqduet.workload_key,
            "command_fingerprint": "legacy-native-colocation",
            "resource_kind": "cpu",
            "node_bucket": hardware.node_bucket,
            "profile": 8,
            "unit": "episode",
            "total_units": 80,
            "aggregate_rate": 8,
            "per_task_rates": [1] * 8,
            "source": "legacy-freqduet-profile-8.json",
            "workload_env": freqduet.workload_env,
            "resource_state": "empty",
            "stable_rate_ready": True,
        }
    )
    status = eta_design._measurement_status(
        ServiceRateCache((legacy_freqduet,)),
        freqduet,
        hardware,
        empty,
        (8,),
        boundary_map={},
    )
    assert status == ("measured", [8], [], [])
