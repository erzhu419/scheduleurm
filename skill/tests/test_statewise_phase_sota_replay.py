from __future__ import annotations

import json
from pathlib import Path

import pytest

from algorithm.experiments.statewise_phase_sota_replay_gate import (
    build_statewise_phase_sota_replay_gate,
)
from simulation.service_cache import ProfileRecord, ServiceRateCache
from simulation.statewise_replay import (
    ExactReplayProfile,
    build_exact_replay_views,
)


def _record(
    workload_key: str,
    workload_env: str,
    profile: int,
    *,
    eta_source: str = "task_native_durable_progress",
) -> ProfileRecord:
    total_units = float(6 * profile)
    wall_s = 20.0 + 70.0 * profile
    return ProfileRecord(
        workload_key=workload_key,
        command_fingerprint="statewise-test",
        resource_kind="cpu",
        node_bucket="node001:cpu_hpc_192c",
        profile=profile,
        unit="episode",
        total_units=total_units,
        aggregate_rate=0.05 * profile,
        per_task_rates=tuple(0.05 for _ in range(profile)),
        source="unit-test",
        workload_env=workload_env,
        resource_state="empty" if profile == 1 else "controlled_colocation",
        eta_source=eta_source,
        stable_rate_ready=True,
        hardware_class="cpu_hpc_192c",
        completion_model_ready=True,
        completion_model_sample_count=13,
        startup_overhead_s=5.0,
        completion_unit_s=(wall_s - 7.0) / total_units,
        finalization_overhead_s=2.0,
        completion_total_wall_s=wall_s,
        allocation_workers=1,
        colocation_count=profile,
    )


def _request(
    workload_key: str,
    workload_env: str,
    profile: int,
    *,
    allocation_workers: int = 1,
) -> ExactReplayProfile:
    return ExactReplayProfile(
        workload_key=workload_key,
        workload_env=workload_env,
        node_bucket="node001:cpu_hpc_192c",
        resource_state="empty" if profile == 1 else "controlled_colocation",
        allocation_workers=allocation_workers,
        colocation_count=profile,
    )


def test_exact_projection_separates_lower_service_and_completion_point():
    source_record = _record("freqduet_cpu_native", "freqduet", 2)
    source = ServiceRateCache((source_record,))

    views = build_exact_replay_views(
        source,
        (_request("freqduet_cpu_native", "freqduet", 2),),
    )

    assert source.get("freqduet_cpu_native", 2) is None
    lower = views.lower_service.get("freqduet_cpu_native", 2)
    point = views.completion_point.get("freqduet_cpu_native", 2)
    assert lower is not None and point is not None
    assert lower.aggregate_rate == pytest.approx(source_record.aggregate_rate)
    assert point.aggregate_rate == pytest.approx(
        source_record.total_units / source_record.completion_total_wall_s
    )
    assert point.aggregate_rate != pytest.approx(lower.aggregate_rate)
    assert views.rows[0]["completion_point_rate_formula"] == (
        "source_group_total_units/completion_total_wall_s"
    )


def test_exact_projection_rejects_worker_colocation_axis_mismatch():
    source = ServiceRateCache(
        (_record("freqduet_cpu_native", "freqduet", 2),)
    )

    with pytest.raises(KeyError, match="missing exact replay service tuple"):
        build_exact_replay_views(
            source,
            (
                _request(
                    "freqduet_cpu_native",
                    "freqduet",
                    2,
                    allocation_workers=2,
                ),
            ),
        )


def test_exact_projection_rejects_history_eta():
    source = ServiceRateCache(
        (
            _record(
                "freqduet_cpu_native",
                "freqduet",
                1,
                eta_source="history_fallback",
            ),
        )
    )

    with pytest.raises(ValueError, match="history_eta_forbidden"):
        build_exact_replay_views(
            source,
            (_request("freqduet_cpu_native", "freqduet", 1),),
        )


def test_cpu_phase_sota_gate_uses_exact_completion_rows(tmp_path: Path):
    records = []
    for workload_key, workload_env in (
        ("freqduet_cpu_native", "freqduet"),
        ("sumo_eval_cpu_native", "sumo"),
    ):
        records.extend(
            _record(workload_key, workload_env, profile)
            for profile in (1, 2, 4)
        )
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(
        json.dumps(ServiceRateCache(records).snapshot()),
        encoding="utf-8",
    )

    result = build_statewise_phase_sota_replay_gate(
        cache_path=cache_path,
        node="node001",
        profiles=(1, 2, 4),
        arrivals=("static",),
        task_count=8,
        total_units_per_task=6.0,
    )

    assert result["status"] == "PASS"
    assert result["exact_replay_projection"]["row_count"] == 6
    assert result["checks"]["candidate_selection_bound_to_lower_service"]
    scenario = result["scenarios"][0]
    assert scenario["candidate"]["completed_jobs"] == 16
    assert all(row["completed_jobs"] == 16 for row in scenario["sota"])
