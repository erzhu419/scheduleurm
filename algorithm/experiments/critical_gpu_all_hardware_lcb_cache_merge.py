"""Append the jtl311linux hardware-local LCB rows to the available GPU cache."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from simulation.service_cache import ServiceRateCache

from .critical_gpu_available_lcb_cache_merge import (
    DEFAULT_CACHE_OUTPUT as AVAILABLE_CACHE,
    DEFAULT_REPORT as AVAILABLE_REPORT,
)
from .critical_gpu_lcb_cache_merge import (
    _legacy_index_snapshot,
    _profile_record,
    _validated_gate_rows,
    write_merge_outputs,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
NODE = "jtl311linux"
EFFECTIVE_NODE_BUCKET = "gpu_2080_8gb_dual:jtl311linux"
DEFAULT_JTL311_GATE = (
    ARTIFACT_ROOT / "critical_gpu_stochastic_lcb_gate_v5_jtl311linux_20260803.json"
)
DEFAULT_REPORT = ARTIFACT_ROOT / "critical_gpu_all_hardware_lcb_cache_merge_20260809.json"
DEFAULT_MARKDOWN = DEFAULT_REPORT.with_suffix(".md")
DEFAULT_CACHE_OUTPUT = ARTIFACT_ROOT / "service_cache_v2_all_hardware_gpu_phase_20260809.json"


def build_critical_gpu_all_hardware_lcb_cache_merge(
    *,
    available_cache_path: Path = AVAILABLE_CACHE,
    available_report_path: Path = AVAILABLE_REPORT,
    jtl311_gate_path: Path = DEFAULT_JTL311_GATE,
) -> dict[str, Any]:
    waits = []
    errors = []
    for label, path in (
        ("available_cache", available_cache_path),
        ("available_report", available_report_path),
        ("jtl311_gate", jtl311_gate_path),
    ):
        if not Path(path).is_file():
            waits.append({"code": "MISSING_SOURCE", "source": label, "path": str(path)})
    common = {
        "gate": "critical_gpu_all_hardware_lcb_cache_merge",
        "schema_version": 1,
        "available_cache_path": str(available_cache_path),
        "available_report_path": str(available_report_path),
        "jtl311_gate_path": str(jtl311_gate_path),
        "new_physical_node": NODE,
        "new_operational_execution_class": EFFECTIVE_NODE_BUCKET,
        "expected_registered_count": 9,
        "expected_inserted_count": None,
        "inserted_count": 0,
        "inserted_rows": [],
        "training_capacity_excluded_count": 0,
        "training_capacity_excluded_rows": [],
        "legacy_index_unchanged": False,
        "all_four_gpu_nodes_ready": False,
        "wait_reasons": waits,
        "validation_errors": errors,
        "claim_boundary": (
            "This gate preserves the previously validated three-node cache and "
            "adds exactly the training-admitted jtl311linux empty-state actions "
            "under a separate operational execution class. All nine registered "
            "profiles remain audited; training-discovered capacity exclusions "
            "carry zero lower service and are not silently re-admitted. It neither "
            "pools hardware nor admits "
            "loaded states, unregistered profiles, or legacy fallback rows."
        ),
    }
    if waits:
        return {**common, "status": "WAIT_SOURCES", "pass": False}
    try:
        available_bytes = Path(available_cache_path).read_bytes()
        cache = ServiceRateCache.from_snapshot(json.loads(available_bytes.decode("utf-8")))
        available_report_bytes = Path(available_report_path).read_bytes()
        available_report = json.loads(available_report_bytes.decode("utf-8"))
        gate_bytes = Path(jtl311_gate_path).read_bytes()
        gate = json.loads(gate_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        errors.append({"code": "INVALID_SOURCE", "detail": f"{type(exc).__name__}: {exc}"})
        return {**common, "status": "FAIL_SOURCE", "pass": False}
    if not (
        available_report.get("gate") == "critical_gpu_available_lcb_cache_merge"
        and available_report.get("status") == "PASS"
        and available_report.get("pass") is True
        and available_report.get("available_scope_cache_ready") is True
        and int(available_report.get("inserted_count") or 0) == 27
    ):
        errors.append({"code": "AVAILABLE_CACHE_GATE_NOT_PASSING"})
        return {**common, "status": "FAIL_SOURCE", "pass": False}
    expected_available_hash = str(available_report.get("cache_output_sha256") or "")
    actual_available_hash = hashlib.sha256(available_bytes).hexdigest()
    if expected_available_hash and expected_available_hash != actual_available_hash:
        errors.append({"code": "AVAILABLE_CACHE_HASH_MISMATCH"})
        return {**common, "status": "FAIL_SOURCE", "pass": False}

    try:
        rows = _validated_gate_rows(gate, node=NODE, path=Path(jtl311_gate_path))
    except (KeyError, TypeError, ValueError) as exc:
        errors.append({"code": "JTL311_GATE_INVALID", "detail": f"{type(exc).__name__}: {exc}"})
        return {**common, "status": "FAIL_SOURCE", "pass": False}
    excluded_rows = gate.get("training_capacity_excluded_cells") or []
    expected_inserted_count = len(rows)
    if expected_inserted_count + len(excluded_rows) != 9:
        errors.append(
            {
                "code": "REGISTERED_SUPPORT_COUNT_MISMATCH",
                "detail": (
                    f"admitted={expected_inserted_count}, "
                    f"excluded={len(excluded_rows)}, registered=9"
                ),
            }
        )
        return {**common, "status": "FAIL_SOURCE", "pass": False}
    common["expected_inserted_count"] = expected_inserted_count
    common["training_capacity_excluded_count"] = len(excluded_rows)
    common["training_capacity_excluded_rows"] = excluded_rows

    legacy_before = _legacy_index_snapshot(cache)
    inserted = []
    exact_keys = set()
    try:
        for row in rows:
            source_record = _profile_record(row, source=Path(jtl311_gate_path))
            record = replace(
                source_record,
                node_bucket=EFFECTIVE_NODE_BUCKET,
                source=(
                    f"{jtl311_gate_path};physical_node={NODE};"
                    f"source_node_bucket={source_record.node_bucket}"
                ),
            )
            key = (
                record.workload_key,
                record.workload_env,
                record.node_bucket,
                record.resource_state,
                record.profile,
                record.resident_mix,
            )
            if key in exact_keys:
                raise ValueError(f"duplicate exact key {key!r}")
            exact_keys.add(key)
            cache.add(record, force_replace=True)
            lookup = cache.lookup_statewise_exact(
                record.workload_key,
                workload_env=record.workload_env,
                node_bucket=record.node_bucket,
                resource_state=record.resource_state,
                allocation_workers=record.allocation_workers,
                colocation_count=record.colocation_count,
                resident_mix=record.resident_mix,
            )
            if not lookup.is_exact or lookup.record != record:
                raise ValueError(f"exact lookup failed for {key!r}")
            inserted.append({**record.snapshot(), "physical_node": NODE})
    except (KeyError, TypeError, ValueError) as exc:
        errors.append({"code": "CACHE_INSERT_FAILED", "detail": f"{type(exc).__name__}: {exc}"})
        return {**common, "status": "FAIL_INSERTION", "pass": False}
    if len(inserted) != expected_inserted_count:
        errors.append(
            {
                "code": "INSERTED_COUNT_MISMATCH",
                "detail": (
                    f"expected {expected_inserted_count}, observed {len(inserted)}"
                ),
            }
        )
        return {**common, "status": "FAIL_INSERTION", "pass": False}
    legacy_after = _legacy_index_snapshot(cache)
    if legacy_after != legacy_before:
        errors.append({"code": "LEGACY_INDEX_CHANGED"})
        return {**common, "status": "FAIL_LEGACY_INDEX", "pass": False}
    return {
        **common,
        "status": "PASS",
        "pass": True,
        "available_cache_sha256": actual_available_hash,
        "available_report_sha256": hashlib.sha256(available_report_bytes).hexdigest(),
        "jtl311_gate_sha256": hashlib.sha256(gate_bytes).hexdigest(),
        "inserted_count": len(inserted),
        "inserted_rows": inserted,
        "legacy_index_unchanged": True,
        "all_four_gpu_nodes_ready": True,
        "service_cache_snapshot": cache.snapshot(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--available-cache", type=Path, default=AVAILABLE_CACHE)
    parser.add_argument("--available-report", type=Path, default=AVAILABLE_REPORT)
    parser.add_argument("--jtl311-gate", type=Path, default=DEFAULT_JTL311_GATE)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--cache-output", type=Path, default=DEFAULT_CACHE_OUTPUT)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = build_critical_gpu_all_hardware_lcb_cache_merge(
        available_cache_path=args.available_cache,
        available_report_path=args.available_report,
        jtl311_gate_path=args.jtl311_gate,
    )
    disk = write_merge_outputs(
        report,
        report_path=args.report_output,
        markdown_path=args.markdown_output,
        cache_output_path=args.cache_output,
    )
    print(json.dumps({"status": disk["status"], "pass": disk["pass"], "cache_output_written": disk["cache_output_written"]}, indent=2, sort_keys=True))
    return 0 if disk["pass"] else (3 if disk["status"].startswith("WAIT") else 2)


if __name__ == "__main__":
    raise SystemExit(main())
