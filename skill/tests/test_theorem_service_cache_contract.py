from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from algorithm.theorem_dispatch.policy import TheoremMaxWeightPlacementPolicy
from algorithm.theorem_dispatch.service_registry import theorem_service_cache
from simulation.service_cache import ProfileRecord, ServiceRateCache


def _statewise_record() -> ProfileRecord:
    return ProfileRecord(
        workload_key="cpu_native_completion",
        command_fingerprint="test-native-phase-aware",
        resource_kind="cpu",
        node_bucket="node001:cpu_epyc_192c",
        profile=1,
        unit="outer_iter",
        total_units=12.0,
        aggregate_rate=0.25,
        per_task_rates=(0.25,),
        source="unit-test",
        workload_env="native_cpu",
        resource_state="empty",
        resident_mix="none",
        eta_source="tqdm/progress",
        stable_rate_ready=True,
        hardware_class="cpu_epyc_192c",
        completion_model_ready=True,
        completion_model_sample_count=12,
        startup_overhead_s=1.0,
        completion_unit_s=4.0,
        finalization_overhead_s=2.0,
        completion_total_wall_s=51.0,
        allocation_workers=1,
        colocation_count=1,
    )


def _write_cache(path: Path, *, statewise: bool = True) -> bytes:
    record = _statewise_record()
    if not statewise:
        record = ProfileRecord(
            workload_key=record.workload_key,
            command_fingerprint=record.command_fingerprint,
            resource_kind=record.resource_kind,
            node_bucket="",
            profile=record.profile,
            unit=record.unit,
            total_units=record.total_units,
            aggregate_rate=record.aggregate_rate,
            per_task_rates=record.per_task_rates,
            source=record.source,
            workload_env="",
            resource_state="unspecified",
        )
    payload = ServiceRateCache((record,)).snapshot()
    payload["schema_version"] = (
        "service_cache_v2_statewise" if statewise else "legacy_service_cache"
    )
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    path.write_bytes(raw)
    return raw


def test_theorem_cache_identity_is_content_addressed(tmp_path: Path) -> None:
    path = tmp_path / "service-cache-v2.json"
    raw = _write_cache(path)

    cache, identity = theorem_service_cache(path)

    assert cache.available_workloads() == ["cpu_native_completion"]
    assert identity.path == str(path.resolve())
    assert identity.sha256 == hashlib.sha256(raw).hexdigest()
    assert identity.schema_version == "service_cache_v2_statewise"
    assert identity.record_count == 1
    assert identity.statewise_record_count == 1
    assert identity.workload_count == 1


def test_theorem_cache_missing_path_fails_closed(tmp_path: Path) -> None:
    missing = tmp_path / "missing-service-cache.json"

    with pytest.raises(FileNotFoundError, match="theorem service cache is required"):
        theorem_service_cache(missing)


def test_theorem_cache_rejects_legacy_only_artifact(tmp_path: Path) -> None:
    path = tmp_path / "legacy-service-cache.json"
    _write_cache(path, statewise=False)

    with pytest.raises(ValueError, match="not a statewise v2 artifact"):
        theorem_service_cache(path)


def test_policy_snapshot_carries_exact_cache_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "service-cache-v2.json"
    raw = _write_cache(path)
    monkeypatch.setenv("SCHEDULEURM_THEOREM_SERVICE_CACHE_PATH", str(path))

    policy = TheoremMaxWeightPlacementPolicy()
    identity = policy.snapshot()["service_cache_identity"]

    assert identity["path"] == str(path.resolve())
    assert identity["sha256"] == hashlib.sha256(raw).hexdigest()
    assert identity["schema_version"] == "service_cache_v2_statewise"
    assert identity["record_count"] == 1
    assert identity["statewise_record_count"] == 1
    assert identity["workload_count"] == 1
