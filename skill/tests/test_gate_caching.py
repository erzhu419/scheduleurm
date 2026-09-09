from __future__ import annotations

import pytest

from algorithm.experiments import production_organic_readiness_bridge_gate as readiness
from algorithm.experiments.production_load_certificate import load_scheduler_records
from algorithm.experiments import production_wide_organic_trace_gate as wide
from algorithm.experiments import universal_claim_closure_gate as universal


@pytest.fixture(autouse=True)
def clear_gate_caches():
    wide._cached_production_wide_organic_trace_gate.cache_clear()
    readiness._cached_production_organic_readiness_bridge_gate.cache_clear()
    universal._cached_universal_claim_closure_gate.cache_clear()
    yield
    wide._cached_production_wide_organic_trace_gate.cache_clear()
    readiness._cached_production_organic_readiness_bridge_gate.cache_clear()
    universal._cached_universal_claim_closure_gate.cache_clear()


def test_production_wide_gate_cache_is_deepcopy_and_trace_fingerprint_invalidates(monkeypatch, tmp_path):
    calls = {"records": 0}

    def records(**_kwargs):
        calls["records"] += 1
        return []

    monkeypatch.setattr(wide, "load_scheduler_records", records)
    monkeypatch.setattr(wide, "build_production_live_theorem_trace_gate", lambda **_: {
        "pass": True,
        "status": "WAIT_NO_QUEUED_PRODUCTION",
        "production_wide_live_trace_closed": False,
        "queue_summary": {},
        "trace_report": {},
    })
    monkeypatch.setattr(wide, "build_organic_production_canary_recorder_gate", lambda **_: {
        "organic_production_canary_recorder_ready": True,
        "history_large_scale_organic_completion_ready": True,
        "unadmitted_launched_count": 0,
    })
    monkeypatch.setattr(wide, "build_production_launch_completion_gate", lambda **_: {
        "large_scale_active_progress_ready": False,
        "launch_status": "LAUNCH_DISABLED",
    })

    trace = tmp_path / "trace.jsonl"
    first = wide.build_production_wide_organic_trace_gate(trace_path=trace)
    first["status"] = "MUTATED"
    second = wide.build_production_wide_organic_trace_gate(trace_path=trace)

    assert calls["records"] == 1
    assert second["status"] != "MUTATED"

    trace.write_text('{"event":"launch"}\n', encoding="utf-8")
    wide.build_production_wide_organic_trace_gate(trace_path=trace)
    assert calls["records"] == 2


def test_production_readiness_bridge_cache_is_deepcopy(monkeypatch, tmp_path):
    calls = {"records": 0}

    def records(**_kwargs):
        calls["records"] += 1
        return []

    monkeypatch.setattr(readiness, "load_scheduler_records", records)
    monkeypatch.setattr(readiness, "build_production_live_theorem_trace_gate", lambda **_: {
        "pass": True,
        "status": "WAIT_NO_QUEUED_PRODUCTION",
        "production_wide_live_trace_closed": False,
        "queue_summary": {},
    })
    monkeypatch.setattr(readiness, "build_organic_production_canary_recorder_gate", lambda: {
        "organic_production_canary_recorder_ready": True,
        "history_large_scale_organic_completion_ready": True,
        "history_completion_snapshot": {},
    })
    monkeypatch.setattr(readiness, "build_production_shadow_theorem_trace", lambda **_: {
        "production_shadow_trace_closed": True,
        "active_production_shadow_task_count": 8,
        "theorem_slot_count": 8,
        "candidate_count_total": 8,
    })
    monkeypatch.setattr(readiness, "generate_production_queued_theorem_trace", lambda **_: {
        "pass": True,
        "queued_admissible_count": 1,
        "theorem_slot_count": 1,
        "candidate_count_total": 1,
        "oracle_bridge": {},
    })

    first = readiness.build_production_organic_readiness_bridge_gate(
        trace_path=tmp_path / "shadow.jsonl",
        queued_trace_path=tmp_path / "queued.jsonl",
    )
    first["status"] = "MUTATED"
    second = readiness.build_production_organic_readiness_bridge_gate(
        trace_path=tmp_path / "shadow.jsonl",
        queued_trace_path=tmp_path / "queued.jsonl",
    )

    assert calls["records"] == 1
    assert second["status"] != "MUTATED"


def test_universal_claim_gate_cache_is_deepcopy(monkeypatch):
    calls = {"sota": 0}

    def report(name):
        return {
            "gate": name,
            "status": f"{name}_READY",
            "scoped_claim_ready": True,
            "strong_claim_ready": False,
            "pass": True,
        }

    def sota():
        calls["sota"] += 1
        return report("sota")

    monkeypatch.setattr(universal, "build_registered_sota_adapter_closure_gate", sota)
    monkeypatch.setattr(universal, "build_future_workload_protocol_gate", lambda: report("future"))
    monkeypatch.setattr(universal, "build_multinode_original_deployment_gate", lambda **_: report("multinode"))
    monkeypatch.setattr(universal, "build_decima_same_domain_benchmark_gate", lambda: report("decima"))
    monkeypatch.setattr(universal, "build_production_wide_organic_trace_gate", lambda: report("production"))

    first = universal.build_universal_claim_closure_gate()
    first["rows"][0]["status"] = "MUTATED"
    second = universal.build_universal_claim_closure_gate()

    assert calls["sota"] == 1
    assert second["rows"][0]["status"] != "MUTATED"


def test_load_scheduler_records_keeps_default_deepcopy_contract(tmp_path):
    queue = tmp_path / "queue.json"
    archive = tmp_path / "queue_archive.jsonl"
    queue.write_text('{"tasks":[{"id":"t1","status":"done"}]}', encoding="utf-8")
    archive.write_text("", encoding="utf-8")

    first = load_scheduler_records(queue_path=queue, archive_path=archive)
    first[0]["status"] = "mutated"
    second = load_scheduler_records(queue_path=queue, archive_path=archive)
    shared_a = load_scheduler_records(queue_path=queue, archive_path=archive, copy_records=False)
    shared_b = load_scheduler_records(queue_path=queue, archive_path=archive, copy_records=False)

    assert second[0]["status"] == "done"
    assert shared_a is shared_b
